#!/usr/bin/env python
# coding: utf-8

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
from torch.optim.lr_scheduler import _LRScheduler

path_img = '/hpc/scratch/hdd2/fs541623/CellProfilerFeatureExtraction/CP_Cropped_Cells'
fnames = get_image_files('/hpc/scratch/hdd2/fs541623/CellProfilerFeatureExtraction/CP_Cropped_Cells')
path_glob = glob.glob('/hpc/scratch/hdd2/fs541623/CellProfilerFeatureExtraction/CP_Cropped_Cells/*/*/*/*')


def label_func(fname):
    return (re.match(r'.*(Time_\d+hrs).*(Well_\d+).*', fname.name).groups())

labels = np.unique(list(zip(*(set(fnames.map(label_func))))))
label_n = len(np.unique(labels))
labels_encoder = {metadata:l for l, metadata in enumerate(labels)}

def label_encoder(fname):
    time, well = re.match(r'.*(Time_\d+hrs).*(Well_\d+).*', fname.name).groups()
    return labels_encoder[time], labels_encoder[well]

indxs = np.random.permutation(range(len(fnames)))
dset_cut = int(0.9*len(fnames))
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
        out = np.zeros((1,14), int) # TO DO : refactor 
        out[[0,0], np.array(self.y[idx])] = 1
        return (self.transform(torch.tensor((img[None]))), torch.tensor(out, dtype=float).squeeze())
    
train_ds = Dataset(train_files, train_y) 
valid_ds = Dataset(valid_files, valid_y)

def label_decoder(labels):
    label_array=np.array(list(labels_encoder))
    idx = np.array(labels).astype(int) > 0 
    return label_array[idx]

train_iterator = data.DataLoader(train_ds, batch_size=8,shuffle=False)
valid_iterator = data.DataLoader(valid_ds, batch_size=8,shuffle=False)
classes = len(labels)

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

## put in utils file 
class LRFinder:
    def __init__(self, model, optimizer, criterion, device):
        
        self.optimizer = optimizer
        self.model = model
        self.criterion = criterion
        self.device = device
        
        # save initial state of model
        torch.save(model.state_dict(), 'init_params.pt')

    def range_test(self, iterator, end_lr = 10, num_iter = 100, 
                   smooth_f = 0.05, diverge_th = 5):
        
        lrs = []
        losses = []
        best_loss = float('inf')

        lr_scheduler = ExponentialLR(self.optimizer, end_lr, num_iter)
        
        iterator = IteratorWrapper(iterator)
        
        for iteration in range(num_iter):

            loss = self._train_batch(iterator)

            #update lr
            lr_scheduler.step()
            
            lrs.append(lr_scheduler.get_lr()[0])

            if iteration > 0:
                loss = smooth_f * loss + (1 - smooth_f) * losses[-1]
                
            if loss < best_loss:
                best_loss = loss

            losses.append(loss)
            
            if loss > diverge_th * best_loss:
                print("Stopping early, the loss has diverged")
                break
                       
        #reset model to initial parameters
        model.load_state_dict(torch.load('init_params.pt'))
                    
        return lrs, losses

    def _train_batch(self, iterator):
        
        self.model.train()
        
        self.optimizer.zero_grad()
        
        x, y = iterator.get_batch()
        
        x = x.to(self.device)
        y = y.to(self.device)
        
        y_pred = self.model(x)
                
        loss = self.criterion(y_pred, y)
        
        loss.backward()
        
        self.optimizer.step()
        
        return loss.item()

class ExponentialLR(_LRScheduler):
    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        curr_iter = self.last_epoch + 1
        r = curr_iter / self.num_iter
        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]

class IteratorWrapper:
    def __init__(self, iterator):
        self.iterator = iterator
        self._iterator = iter(iterator)

    def __next__(self):
        try:
            inputs, labels = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.iterator)
            inputs, labels, *_ = next(self._iterator)

        return inputs, labels

    def get_batch(self):
        return next(self)

start_lr = 1e-7
optimizer = optim.Adam(model.parameters(), lr=start_lr)
device = 'cuda'
loss_func = nn.BCEWithLogitsLoss()
model = model.to(device)
loss_func = loss_func.to(device)
lr_finder = LRFinder(model, optimizer, loss_func, device='cuda')
lrs, losses = lr_finder.range_test(train_iterator)

def plot_lr_finder(lrs, losses, skip_start = 5, skip_end = 5):
    
    if skip_end == 0:
        lrs = lrs[skip_start:]
        losses = losses[skip_start:]
    else:
        lrs = lrs[skip_start:-skip_end]
        losses = losses[skip_start:-skip_end]
    
    fig = plt.figure(figsize = (16,8))
    ax = fig.add_subplot(1,1,1)
    ax.plot(lrs, losses)
    ax.set_xscale('log')
    ax.set_xlabel('Learning rate')
    ax.set_ylabel('Loss')
    ax.grid(True, 'both', 'x')
    plt.show()
plot_lr_finder(lrs, losses, skip_start = 30, skip_end = 30)

lr, epochs, bs = 3e-3, 10, 64

# deeper => higher learning rate 
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

def train(model, loader, scheduler, optimizer, loss_func, device='cuda'):
    model.train()
    epoch_loss = 0
    
    for (data, targets) in iter(loader):        
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
    return epoch_loss/len(iterator)

def evaluate(model, loader, loss_func, device='cuda'):
    model.eval()
    epoch_loss = 0
    accuracy = 0

    with torch.no_grad():
        for (data, targets) in iter(loader):
            data = data.to(device)
            targets = targets.to(device)
            y_pred = model(data)
            loss = loss_func(y_pred, targets)
            epoch_loss += loss.item
            accuracy += (y_pred == targets).float.sum()
    return (accuracy/len(iterator), epoch_loss/len(iterator))

# gives time in seconds 
def epoch_time(start, end):
    diff_min = int(end - start)/60 
    secs = int(diff_min - (diff_min * 60))
    return diff_min, secs

## Training 
best_valid_loss = float('inf')
for epoch in range(epochs):
    start_time = time.monotonic()
    train_loss = train(model, train_iterator, scheduler, optimizer, loss_func)
    valid_acc, valid_loss = evaluate(model, valid_iterator, loss_func)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'resnet50-scratch.pt')
    end_time = time.monotonic()
    
    mins, secs = epoch_time(end_time, start_time)
    
    print(f'Epoch {epoch:02} | Epoch Time: {mins}m {secs}s')
    print(f'\tTrain Loss: {train_loss:.3f}')
    print(f'\tValid Loss: {valid_loss:.3f} | Valid Accuracy: {valid_acc*100}')