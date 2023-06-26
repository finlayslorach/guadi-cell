#!/bin/bash 

#SBATCH --job-name=MaxZ_pre_processing
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=MaxZ_pre_processing%A.out
#SBATCH --error=MaxZ_pre_processing%A_%a.err
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=20
#SBATCH --partition=stv-cpu

# Set Environment
module purge
module load anaconda3
source activate cell_segmentation_12032021

# Run pre_processing script
python /hpc/scratch/hdd2/fs541623/Cell_Tox_Assay_080421/STEP1_Pre_processing_CellTox260421/pre_processing/STEP1_Pre_Processing.py '/hpc/scratch/hdd2/fs541623/QST' '/hpc/scratch/hdd2/fs541623/Pre_processed_Images'