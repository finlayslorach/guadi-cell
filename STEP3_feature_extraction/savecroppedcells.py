#!/usr/bin/env python
# coding: utf-8

# In[158]:


from skimage import io, exposure, measure, util, filters
import pandas as pd
import numpy as np 
import os 
import glob 
import matplotlib.pyplot as plt
import re
import cv2
from pathlib import Path
from scipy.spatial.distance import cdist



def main(input_masks, input_BF_dir, output_dir):
    for file in input_masks:
    
        ## Get Pixel Identity Map
        mask_img = io.imread(file)
        px_list = list(np.unique(mask_img)); pxs = px_list.remove(0)

        ## Get BF path using metadata from filename of mask
        time, well, field = get_metadata(file, re_time, re_well, re_field)
        BF_path = input_BF_dir/time/well/'Brightfield'/f'{time}_{well}_{field}_C1.tif'

        ## Crop objects 
        get_cropped_obj(mask_img, px_list, BF_path, output_dir, time, well, field)
     
    return 0


def get_blurred_mask(BF_file, mask_img, centre_x, centre_y): 
    
    ## Convert mask & BF to same scale?
    BF_img = io.imread(BF_file).astype('float')/255
    mask_img = mask_img.astype('float')/255
    
    ## Blurr to enlarge mask
    mask_blurred  = filters.gaussian(mask_img,1)
    mask_large=np.where(mask_blurred>0, BF_img, 1)
    
    ## Mask cell from BF 
    mask_large_blurred = filters.gaussian(mask_large, 3)
    img=mask_large_blurred.astype('float')/255
    
    ## Convert back to correct image format 
    img=((img*BF_img)*255).astype('uint16')
    return (crop_img_test(img, centre_x, centre_y ))



## Extract metadata needed to save filenames
def get_metadata(mask_file, re_time, re_well, re_field):
    time = re_time.findall(str(mask_file))[0]; well = re_well.findall(str(mask_file))[0]
    field=re_field.findall(str(Path(mask_file).name))[0]
    return (time, well, field)


## Get Cropped mask objects  
def get_cropped_obj(mask_img, px_list, BF_path, output_dir, time, well, field): 
    
    ## Set obj number to match csv from cellprofiler 
    Obj_number = 0
    for pixel in px_list:
        mask_bin = np.where(mask_img == pixel, 1, 0)

        ## get centre coordinates of obj
        y,x = [(region.centroid[0], region.centroid[1]) for region in measure.regionprops(mask_bin)][0]
        
        ## Remove any objects 100px around border 
        if (y < 100) or (y > mask_bin.shape[0]-100) or (x < 100) or (x > mask_bin.shape[1]-100):
            continue

        ## Crop BF_img using centre coordinates
        img = get_blurred_mask(BF_path, mask_bin, x, y)
        Obj_number += 1

        ## Save cropped cell 
        output = output_dir/time/well/field
        file = f'{time}_{well}_{field}_CytObj{Obj_number}.tif'
        save_crop(img, output, file)
    return 0


## save image
def save_crop(img, output_dir, file):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    io.imsave(output_dir/file, img)
    print('saving image')
    
    return 0


## 200 px crop
def crop_img_test(img,centre_X, centre_Y):

    # Crop using centre
    y2, y1, x2, x1 = int(centre_Y+100), int(centre_Y-100),  int(centre_X+100), int(centre_X-100)
    cropped = img[y1:y2, x1:x2]
    
    return cropped


if __name__ == '__main__':
    
    ## Input/Output Files 
    output_dir = Path('/hpc/scratch/hdd2/fs541623/CellProfilerFeatureExtraction/CP_Cropped_Cells')
    input_masks = sorted(glob.glob('/hpc/scratch/hdd2/fs541623/Pre_processed_Images/*/*/Cyt/*cp_masks*'), key=os.path.getctime)
    input_BF_dir = Path('/hpc/scratch/hdd2/fs541623/Pre_processed_Images')

    ## metadata to extract filenames; for output file names  
    re_time = re.compile(r'(Time_\d+hrs_R\d)')
    re_well = re.compile(r'(Well_\d+)')
    re_field=re.compile(r'(Field_\d+)')

    
    # Save centre cropped cells for each img in each timepoint/well
    main(input_masks, input_BF_dir, output_dir)

