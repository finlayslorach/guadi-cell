#!/bin/bash 

#SBATCH --job-name=scratch_model3
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=scratch_model2%A.out
#SBATCH --error=scratch_model2%A_%a.err
#SBATCH --partition=stv-gpu
#SBATCH --gres=gpu:8
#SBATCH --gres=gpu:a6000:1

module purge 
module load anaconda3

# Load enviornment containing cellpose
source activate fastai


# Run Segmentation only on RGB Images 
python /hpc/scratch/hdd2/fs541623/Cell_Tox_Assay_080421/FEATURE_EXTRACTION/ModelTest.py