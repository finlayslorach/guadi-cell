#!/bin/bash 

#SBATCH --time=4-00:00:00
#SBATCH --nodes=1
#SBATCH --job-name=SaveCroppedCells1
#SBATCH --ntasks=1
#SBATCH --output=SaveCroppedCells_1_%A.out
#SBATCH --error=SaveCroppedCells_1_%A_%a.err
#SBATCH --cpus-per-task=10
#SBATCH --partition=stv-gpu

module purge 
module load anaconda3

# Load enviornment containing cellpose
source activate cell_segmentation_12032021

# Run CP_FeatureProcessing 
python /hpc/scratch/hdd2/fs541623/Cell_Tox_Assay_080421/FEATURE_EXTRACTION/savecroppedcells.py