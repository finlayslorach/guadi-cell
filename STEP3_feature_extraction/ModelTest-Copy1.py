from fastai import * 
from fastai.data.all import *
from fastai.vision.data import * 
from fastai.vision.core import *
from fastai.vision.all import *
from torchvision import transforms
from torch_lr_finder import LRFinder
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
from torch import nn, optim
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
import re
import os
import glob
from skimage import io, color, img_as_float32, img_as_uint
import random
import numpy as np
import PIL
import glob
import gc
from torch.optim.lr_scheduler import _LRScheduler
from sklearn.metrics import fbeta_score
import sys
import builtins
import os
sys.stdout = open("pre_trained_test2.txt", "w", buffering=1)
def print(text):
    builtins.print(text)
    os.fsync(sys.stdout)
    return text
    
print('----------------------running script---------------')

fnames = get_image_files('/hpc/scratch/hdd2/fs541623/CellProfilerFeatureExtraction/CP_Cropped_Cells')
path_glob = glob.glob('/hpc/scratch/hdd2/fs541623/CellProfilerFeatureExtraction/CP_Cropped_Cells/*/*/*/*')
path_img = '/hpc/scratch/hdd2/fs541623/CellProfilerFeatureExtraction/CP_Cropped_Cells'

def label_func(fname):
    time, well= (re.match(r'.*(Time_\d+hrs).*(Well_\d+).*', fname.name).groups())
    return time, well

labels = np.unique(list(zip(*(set(fnames.map(label_func))))))
fnames.map(label_func)
label_n = len(np.unique(labels))
labels_encoder = {metadata:l for l, metadata in enumerate(labels)}

def label_encoder(fname):
    time, well = re.match(r'.*(Time_\d+hrs).*(Well_\d+).*', fname.name).groups()
    return labels_encoder[time], labels_encoder[well]
indxs = np.random.permutation(len(fnames))
dset_cut = int(len(indxs)*0.9)
train_files = fnames[indxs[:dset_cut]]
valid_files = fnames[indxs[dset_cut:]]
train_y = train_files.map(label_encoder)
valid_y = valid_files.map(label_encoder)
class Dataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x   
        self.y = y
        self.transform = transforms.Resize(224)
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img = img_as_float32(Image.open(self.x[idx]))
        
        out = np.zeros((1,14), int) # TO DO : refactor # get encoded label 
        out[[0,0], np.array(self.y[idx])] = 1
        return (self.transform(torch.tensor(img[None], dtype=torch.float32)), 
                torch.tensor(out, dtype=torch.float32).squeeze())
train_ds = Dataset(train_files, train_y) 
valid_ds = Dataset(valid_files, valid_y)
def label_decoder(labels):
    label_array=np.array(list(labels_encoder))
    idx = np.array(labels).astype(int) > 0 
    return label_array[idx]
train_iterator = data.DataLoader(train_ds, batch_size=128,shuffle=False, pin_memory=True, num_workers=32)
valid_iterator = data.DataLoader(valid_ds, batch_size=128,shuffle=False, pin_memory=True, num_workers=32)
pretrained_resnet = models.resnet50(pretrained=True)
# modify 1st conv layer for grayscale
state_dict = pretrained_resnet.state_dict()
conv1_weight = state_dict['conv1.weight'].sum(1, keepdim=True)
state_dict['conv1.weight'] = conv1_weight
pretrained_resnet.conv1.weight = torch.nn.Parameter(conv1_weight)
pretrained_resnet.conv1.in_channels = 1
classes = len(labels)
IN_FEATURES = pretrained_resnet.fc.in_features 
fc = nn.Linear(IN_FEATURES, classes)
pretrained_resnet.fc = fc
model = pretrained_resnet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_func = nn.BCEWithLogitsLoss()
model = model.to(device)
loss_func = loss_func.to(device)
lr, epochs, bs = 3e-5, 50, 64
params = [
        {'params':model.conv1.parameters(), 'lr': lr/10},
        {'params':model.layer1.parameters(), 'lr': lr/8},
        {'params':model.layer2.parameters(), 'lr': lr/6},
        {'params':model.layer3.parameters(), 'lr': lr/4},
        {'params':model.layer4.parameters(), 'lr': lr/2},
        {'params':model.fc.parameters()}]
optimizer = optim.Adam(params,lr=lr)
total_steps = epochs * len(train_iterator) # epochs * number of batches
max_lr = [p['lr'] for p in optimizer.param_groups]
scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr, total_steps)

def train(model, loader, scheduler, optimizer, loss_func, device='cuda', train_head=False):
    epoch_loss = 0
    model.train()
        
    if train_head:
        model.requires_grad_(False)
        model.layer4.requires_grad_(True)
        model.fc.requires_grad_(True)
    else: model.requires_grad_(True)
        
    for batch_idx, (data, targets) in enumerate(iter(loader)):   
        data = data.to(device)
        targets = targets.to(device)
        y_pred = model(data)
        loss = loss_func(y_pred, targets)
        epoch_loss += loss.item()
        
        loss.backward()
        # updates gradient base on gradient in params
        optimizer.step()
        scheduler.step()
    # batch loss
    return epoch_loss/len(loader)

def evaluate(model, loader, loss_func, device='cuda', train_head=False):
    model.eval()
    epoch_loss = 0
    fbeta = 0

    with torch.no_grad():
        for (data, targets) in iter(loader):
            data = data.to(device)
            targets = targets.to(device)
            y_pred = model(data)
            loss = loss_func(y_pred, targets)
            epoch_loss += loss.item()
            
            # fbeta batch score
            y_pred = torch.sigmoid(y_pred) > 0.5
            fbeta += fbeta_score(targets.cpu(), y_pred.detach().cpu(), beta=2, average='samples') 

    # average fbeta score 
    fbeta /= len(loader)
    epoch_loss /= len(loader)
    print(epoch_loss)
    return fbeta, epoch_loss

def epoch_time(start, end):
    diff_min = int(end - start)/60 
    secs = int(diff_min - (diff_min * 60))
    return diff_min, secs

print('loading weights')
model.load_state_dict(torch.load('resnet50-scratch.pt'))
best_valid_loss = 0.352
epochs=50
for epoch in range(epochs):
    start_time = time.monotonic()
    print(start_time)
    train_loss = train(model, train_iterator, scheduler, optimizer, loss_func, train_head=True)
    valid_acc, valid_loss = evaluate(model, valid_iterator, loss_func, train_head=True)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'resnet50-head-test2-stage2.pt')
    end_time = time.monotonic()
    
    mins, secs = epoch_time(start_time, end_time)
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f'Epoch {epoch:02} | Epoch Time: {mins}m {secs}s')
    print(f'\tTrain Loss: {train_loss:.3f}')
    print(f'\tValid Loss: {valid_loss:.3f} | Valid Accuracy: {valid_acc}')
    
