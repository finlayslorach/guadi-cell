#!/usr/bin/env python
# coding: utf-8


from fastai import * 
from fastai.data.all import *
from fastai.vision.data import * 
from fastai.vision.core import *
from fastai.vision.all import *
import re
import os
import glob
from skimage import io
import random
import numpy as np
import PIL


## 892104 images
path_img = '/hpc/scratch/hdd2/fs541623/CellProfilerFeatureExtraction/CP_Cropped_Cells'
fnames = get_image_files('/hpc/scratch/hdd2/fs541623/CellProfilerFeatureExtraction/CP_Cropped_Cells')

# Need to specifiy it is grayscale image
data = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                get_items = get_image_files,
                get_y=Pipeline([attrgetter('name'), RegexLabeller(pat=r'.*(Time_\d+hrs).*(Well_\d+).*')]),
                splitter=RandomSubsetSplitter(train_sz=0.9, valid_sz=0.1, seed=2),
                item_tfms=Resize(224), 
                batch_tfms=[Normalize.from_stats(*imagenet_stats)])
dls = data.dataloaders(path_img, bs=32)
dls = dls.cuda()
learn50 = cnn_learner(dls, models.resnet50, metrics=partial(accuracy_multi, thresh=0.5))

learn50.fit_one_cycle(5, lr_max=slice(0.01))
learn50.save('stage-5-head-multi_class')

learn50.unfreeze()
learn50.fit_one_cycle(10, lr_max=slice(1e-6, 1.7e-4))
learn50.save('stage-2-full-multi_class')




