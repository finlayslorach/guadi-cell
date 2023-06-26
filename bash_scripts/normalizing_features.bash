#!/bin/bash 

#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --job-name=Normalizingfeatures
#SBATCH --ntasks=1
#SBATCH --output=Normalizingfeatures%A.out
#SBATCH --error=Normalizingfeatures%A_%a.err
#SBATCH --cpus-per-task=12
#SBATCH --partition=stv-cpu


module purge 
module load R

Rscript /hpc/scratch/hdd2/fs541623/Cell_Tox_Assay_080421/FEATURE_EXTRACTION/cytominer.R