import skimage 
import pandas as pd
import numpy as np 
import os 
import glob 
import matplotlib.pyplot as plt
import re


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

            # Centre crop all objects & Save to new CentreCrops/Timepoint/Well/*
            output_dir = os.path.join(output, time, well)

            for obj in range(0,len(df_field)):
                obj_filename = f'{time}_{well}_Field_{f+1}_obj{obj}.tif'
                cropped_img = crop_img(df_field, BF_file, obj)
                save_crop(obj_filename, cropped_img, output_dir)


            # Update field number 
            print(f'outer -----------Object: {obj} for field: {field} {f+1}successfully cropped-----------')
            
    return 0

# Centre Crop Cells
def crop_img(df, BF_file, obj_number):
    
    # Load Image
    BF_img = skimage.io.imread(BF_file)
    
    # Get Coordinates
    centre_X = int(df.iloc[obj_number, df.columns.get_loc('Location_Center_X')])
    centre_Y = int(df.iloc[obj_number, df.columns.get_loc('Location_Center_Y')])
    
    # Crop using centre
    y2, y1, x2, x1 = int(centre_Y+100), int(centre_Y-100),  int(centre_X+100), int(centre_X-100)
    cropped = BF_img[y1:y2, x1:x2]

    return cropped

# Save Image to timepoint/well dir
def save_crop(file, img, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    grayscale_stack = np.dstack([img, img, img])
    skimage.io.imsave(os.path.join(output_dir, file), grayscale_stack)
    
    return 0




if __name__ == '__main__':
    
    # Input/output dir
    input_obj_csv = sorted(glob.glob('/hpc/scratch/hdd2/fs541623/CellProfilerFeatureExtraction/CP_Data/*/*/FilteredCytObj.csv'), key=os.path.getctime)
    output_path = '/hpc/scratch/hdd2/fs541623/CentreCrops'
    BF_path = '/hpc/scratch/hdd2/fs541623/Pre_processed_Images'
        
    # Compile Regex; pass into main to extract img metadata for filenames
    re_time = re.compile(r'(Time_\d+hrs_R\d)')
    re_well = re.compile(r'(Well_\d+)')
    
    # Save centre cropped cells for each img in each timepoint/well
    main(input_obj_csv, re_time, re_well, BF_path, output_path)