#!/bin/bash 

#SBATCH --job-name=CellposeSegmentation270421
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=Cellpose_Segmentation_%A.out
#SBATCH --error=cellpose_Segmentation_%A_%a.err
#SBATCH --cpus-per-task=6
#SBATCH --partition=stv-gpu
#SBATCH --gpus=1


module purge 
module load anaconda3

# Load enviornment containing cellpose
source activate cell_segmentation_12032021


# Run Segmentation only on RGB Images 
for image in /hpc/scratch/hdd2/fs541623/Pre_processed_Images/*/*/Cyt; do 
	python -m cellpose --dir $image --use_gpu --pretrained_model cyto --chan 2 --diameter 0 --save_png
done


