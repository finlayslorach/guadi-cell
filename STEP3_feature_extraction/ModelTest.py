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
sys.stdout = open("performanceTest5.txt", "w", buffering=1)
def print(text):
    builtins.print(text)
    os.fsync(sys.stdout)
    return text
    
print('----------------------running script---------------')

fnames = get_image_files('/hpc/scratch/hdd2/fs541623/CellProfilerFeatureExtraction/CP_Cropped_Cells')
path_glob = glob.glob('/hpc/scratch/hdd2/fs541623/CellProfilerFeatureExtraction/CP_Cropped_Cells/*/*/*/*')
path_img = '/hpc/scratch/hdd2/fs541623/CellProfilerFeatureExtraction/CP_Cropped_Cells'

def label_func(fname):
    return (re.match(r'.*(Time_\d+hrs).*(Well_\d+).*', fname.name).groups())
labels = np.unique(list(zip(*(set(fnames.map(label_func))))))
label_n = len(np.unique(labels))
classes=len(labels)


labels_encoder = {metadata:l for l, metadata in enumerate(labels)}
def label_encoder(fname):
    time, well = re.match(r'.*(Time_\d+hrs).*(Well_\d+).*', fname.name).groups()
    return labels_encoder[time], labels_encoder[well]
indxs = np.random.permutation(range(int(len(fnames))))
dset_cut = int(len(fnames)*0.8)


## Split into train & val
train_files = fnames[indxs[:dset_cut]]
valid_files = fnames[indxs[dset_cut:]]

## Get labels for shuffled files
train_y = train_files.map(label_encoder)
valid_y = valid_files.map(label_encoder)

## Dataset 
class Dataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.transform = transforms.Resize(224)
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img = img_as_float32(Image.open(self.x[idx]))
        out = np.zeros((1,14), int) # TO DO : refactor 
        out[[0,0], np.array(self.y[idx])] = 1
        return (self.transform(torch.tensor((img[None]))), torch.tensor(out, dtype=float).squeeze())
    
    
## Create Datasets 
train_ds = Dataset(train_files, train_y) 
valid_ds = Dataset(valid_files, valid_y)


def label_decoder(labels):
    label_array=np.array(list(labels_encoder))
    idx = np.array(labels).astype(int) > 0 
    return label_array[idx]

#dataloaders
train_iterator = data.DataLoader(train_ds, batch_size=32,shuffle=False, pin_memory=True, num_workers=32)
valid_iterator = data.DataLoader(valid_ds, batch_size=32,shuffle=False, pin_memory=True, num_workers=32)

print('---------------Getting model-----------')
# model 
def conv_layer(inputs, outputs, ks, stride, padding, use_activation=None):
    layers=[nn.Conv2d(inputs, outputs, ks, stride, padding, bias=False), nn.BatchNorm2d(outputs)]
    if use_activation: layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)
# Residual Block 
class residual(nn.Module):
    def __init__(self, input_channels, out_channels, stride=1, activation: torch.nn.Module = nn.ReLU(inplace=True)):
        super().__init__()
    
        # 64 -> 256 ; always first block in resnetlayer
        self.convs = nn.Sequential(*[conv_layer(input_channels, out_channels, 1, 1, 0, use_activation=True),
                                     conv_layer(out_channels, out_channels, 3, stride, 1, use_activation=True),
                                     conv_layer(out_channels, out_channels*4, 1, 1, 0, use_activation=True)])
        
        # if 256 == 4*64 (256) e.g. for other blocks of resnet layer 
        if input_channels == out_channels*4: 
            self.conv4 = nn.Identity()
            print(f'identity layer:{input_channels, out_channels, out_channels*4}')
        else: 
            # if 64 != 256 ( 4*64) -> do convolutional layer
            print(f'residual conv layer:{input_channels, out_channels, out_channels*4}')
            self.conv4 = conv_layer(input_channels, out_channels*4, 1, stride, 0)
        
        self.activation = activation
        
    def forward(self, X):
        return self.activation((self.convs(X) + self.conv4(X)))

## Need to refactor 
class resnetmodel(nn.Module):
    def __init__(self, channels, n_blocks, classes=classes):
        super().__init__()
        self.in_channels = channels[0] # 64
        
        ## to work with 1 channel images
        self.model_stem = nn.Sequential(*[conv_layer(1, self.in_channels, ks=7, stride=2, padding=3, use_activation=True), 
                                     nn.MaxPool2d(3, stride=2, padding=1)])
        self.res_layer1 = self._make_res(residual, channels[0], n_blocks[0])
        self.res_layer2 = self._make_res(residual, channels[1], n_blocks[1], stride=2)
        self.res_layer3 = self._make_res(residual,channels[2], n_blocks[2], stride=2)
        self.res_layer4 = self._make_res(residual, channels[3], n_blocks[3], stride=2)
        
        # inchannels = 2048??
        self.adpool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(self.in_channels, classes)
        
    
    def _make_res(self, residual, channels, n_blocks, stride=1):
        # 1st reslayer doesnt have stride == 2 
        layers = []
        
        # 1st block of each res layer always has stride == 1
        # e.g. inchannels = 64, channels = 64 --> ends up outputting channels 4*64 = 256
        
        print(f'input channels to next layer: {channels}')
        
        # convolution block
        layers.append(residual(self.in_channels, channels)) # 256 -> 128 (128 * 4 = 512)
        
        # identity blocks
        for i in range(1, n_blocks):
            # input channels = 256 -> 64 
            layers.append(residual(channels*4, channels)) # 128*4 = 512 -> 512 (128 * 4)
        self.in_channels = 4*channels # set in_channels for next convolution block
        print(f'outchannels: {self.in_channels}')
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.model_stem(x)
        x = self.res_layer1(x) 
        x = self.res_layer2(x)
        x = self.res_layer3(x)
        x = self.res_layer4(x)
        x = self.adpool(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x
model = resnetmodel(channels=[64,128,256,512], n_blocks=[3,4,6,3])

lr, epochs, bs = 3e-5, 10, 8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Scheduler
params = [
        {'params':model.model_stem.parameters(), 'lr': lr/10},
        {'params':model.res_layer1.parameters(), 'lr': lr/8},
        {'params':model.res_layer2.parameters(), 'lr': lr/6},
        {'params':model.res_layer3.parameters(), 'lr': lr/4},
        {'params':model.res_layer4.parameters(), 'lr': lr/2},
        {'params':model.linear.parameters()}]
optimizer = optim.Adam(params,lr=lr)
total_steps = epochs * len(train_iterator) # epochs * number of batches
max_lr = [p['lr'] for p in optimizer.param_groups]
scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr, total_steps)
loss_func = nn.BCEWithLogitsLoss()
model = model.to(device)
loss_func = loss_func.to(device)


def train(model, loader, scheduler, optimizer, loss_func, device='cuda'):
    model.train()
    epoch_loss = 0
    
    for idx, (data, targets) in enumerate(iter(loader)):        
        data = data.to(device)
        targets = targets.to(device)
        y_pred = model(data)
        loss = loss_func(y_pred, targets)
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        # updates gradient base on gradient in params
        optimizer.step()
        scheduler.step()        
    # calculate average batch loss
    return epoch_loss/len(loader)

def evaluate(model, loader, loss_func, device='cuda'):
    model.eval()
    epoch_loss = 0
    fbeta1 = 0
    fbeta2 = 0

    with torch.no_grad():
        for (data, targets) in iter(loader):
            data = data.to(device)
            targets = targets.to(device)
            y_pred = model(data)
            loss = loss_func(y_pred, targets)
            epoch_loss += loss.item()
            
            # fbeta batch score
            y_pred = torch.sigmoid(y_pred) > 0.5
            fbeta1 += fbeta_score(targets.cpu(), y_pred.detach().cpu(), beta=1, average='samples') 
            fbeta2 += fbeta_score(targets.cpu(), y_pred.detach().cpu(), beta=2, average='samples') 

    # average fbeta score 
    fbeta1 /= len(loader)
    fbeta2 /= len(loader)
    epoch_loss /= len(loader)
    print(epoch_loss)
    return fbeta1, fbeta2, epoch_loss

# gives time in seconds 
def epoch_time(start, end):
    diff_min = int(end - start)/60 
    secs = int(diff_min - (diff_min * 60))
    return diff_min, secs

print('-------------------training model--------------------')
# train

print('loading weights')
model.load_state_dict(torch.load('resnet50-scratch-p3.pt'))
best_valid_loss = float(0.183)
for epoch in range(epochs):
    start_time = time.monotonic()
    print('training')
    train_loss = train(model, train_iterator, scheduler, optimizer, loss_func)
    
    print('evaluating')
    valid_acc1, valid_acc2, valid_loss = evaluate(model, valid_iterator, loss_func)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'resnet50-scratch-p5.pt')
    else:
        torch.save(model.state_dict(), 'resnet50-scratch-overfit-weights-p5.pt')
    end_time = time.monotonic()
    
    mins, secs = epoch_time(start_time, end_time)
    
    print(f'Epoch {epoch:02} | Epoch Time: {mins}m {secs}s')
    print(f'\tTrain Loss: {train_loss:.3f}')
    print(f'\tValid Loss: {valid_loss:.3f} | fbeta1: {valid_acc1} | fbeta2: {valid_acc2}')