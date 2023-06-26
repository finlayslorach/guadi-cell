#!/usr/bin/env python
# coding: utf-8

# In[2]:


from skimage import io, exposure
import pandas as pd
import numpy as np 
import os 
import glob 
import matplotlib.pyplot as plt
import re
import cv2


# In[4]:


# Get CSV for each timepoint/well
def main(well_csv_list, re_time, re_well, BF_path, output):
    for csv in well_csv_list:
        print(f'--current csv: {csv}--')

        # BF_file and Obj csv file located in different dir; use csv to match each obj to correct BF_image 
        df = pd.read_csv(csv); BF_files = df.iloc[:, df.columns.get_loc('FileName_BF')]

        # For each csv (timepoint/well) split csv into different FOV (1-9)
        # Reset field number for each csv file i.e. well
        for f, field in enumerate(pd.unique(BF_files)): 
            
            # Locate correct BF_image using metadata extracted from csv file
            time = re_time.findall(str(field))[0]; well = re_well.findall(str(field))[0]
            BF_file = f'{BF_path}/{time}/{well}/Brightfield/{field}'

            # Split df on FOV; get centre coordinates for each obj
            df_field = df[df['FileName_BF'].isin([field])]
            
            # Get Image masks for every FOV in Well
            mask_path = f'/hpc/scratch/hdd2/fs541623/CellProfilerFeatureExtraction/CP_Cropped_Cells/{time}/{well}/{field}/FilteredCyt*'
            img_list=sorted(glob.glob(mask_path), key=os.path.getctime)
            img_masks = io.imread_collection(img_list)
            
            # Centre crop all objects & Save to new CentreCrops/Timepoint/Well/*
            output_dir = os.path.join(output, time, well)

            for obj in range(0,len(df_field)):
                obj_filename = f'{time}_{well}_Field_{f+1}_obj{obj}.tif'
                cropped_img = get_centre_crop(df_field, BF_file, obj, img_masks[obj])
                save_crop(obj_filename, cropped_img, output_dir)

            # Update field number 
            print(f'outer -----------Object: {obj} for field: {field} {f+1}successfully cropped-----------')
            
    return 0


# In[5]:


# Centre Crop Cells
def crop_img(df, BF_file, obj_number, mask):
    
    # Load Image
    BF_img = skimage.io.imread(BF_file).astype('float')/255
    
    # Get Coordinates
    centre_X = int(df.iloc[obj_number, df.columns.get_loc('Location_Center_X')])
    centre_Y = int(df.iloc[obj_number, df.columns.get_loc('Location_Center_Y')])
    
    # Crop using centre
    y2, y1, x2, x1 = int(centre_Y+100), int(centre_Y-100),  int(centre_X+100), int(centre_X-100)
    cropped = BF_img[y1:y2, x1:x2]

    return (cropped, centre_X, centre_Y)


# In[6]:


# Save Image to timepoint/well dir
def save_crop(file, img, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    grayscale_stack = np.dstack([img, img, img])
    skimage.io.imsave(os.path.join(output_dir, file), grayscale_stack)
    
    return 0


# In[315]:


def get_centre_crop(df, BF_img, obj_number, mask):    
    mask_blurred  = cv2.GaussianBlur(mask,(3,3),0)
    mask_large=np.where(mask_blurred>0, BF_img, 1)
    mask_large_blurred = cv2.GaussianBlur(mask_large, (11,11),0)
    img=mask_large_blurred.astype('float')/255
    img=((img*BF_img)*255).astype('uint16')
    return (crop_img_test(df, img, obj_number))


# In[4]:


if __name__ == '__main__':
    
    # Input/output dir
    input_obj_csv = sorted(glob.glob('/hpc/scratch/hdd2/fs541623/CellProfilerFeatureExtraction/CP_Cropped_Cells/*/*/FilteredCytObj.csv'), key=os.path.getctime)
    output_path = '/hpc/scratch/hdd2/fs541623/IndividuallyCroppedCells'
    BF_path = '/hpc/scratch/hdd2/fs541623/Pre_processed_Images'
        
    # Compile Regex; pass into main to extract img metadata to create filenames
    re_time = re.compile(r'(Time_\d+hrs_R\d)')
    re_well = re.compile(r'(Well_\d+)')
    
    # Save centre cropped cells for each img in each timepoint/well
    main(input_obj_csv, re_time, re_well, BF_path, output_path)

