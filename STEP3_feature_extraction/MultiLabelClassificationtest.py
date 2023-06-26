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
sys.stdout = open("multilabel.txt", "w", buffering=1)
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
    return time+','+well
labels_fast = fnames.map(label_func)
dls = ImageDataLoaders.from_lists(path_img, fnames, labels_fast, y_block=CategoryBlock, bs=32)
learn50 = cnn_learner(dls, models.resnet50, metrics=[partial(accuracy_multi, thresh=0.5), fbeta_score])
learn50.fine_tune(2)
learn50.unfreeze()
learn50.fit_one_cycle(5, lr_max=3e-3)
learn50.save('multilabel_weights')
