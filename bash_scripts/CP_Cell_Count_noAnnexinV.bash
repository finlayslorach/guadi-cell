#!/bin/bash 

#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --job-name=CellprofilerFeatureNoAnnexin040521
#SBATCH --ntasks=1
#SBATCH --output=CP_FeatureExtractionNoAnnexin040521_%A.out
#SBATCH --error=CP_FeatureExtractionNoAnnexin040521_%A_%a.err
#SBATCH --cpus-per-task=10
#SBATCH --partition=stv-gpu

module purge 
module load anaconda3

# Load enviornment containing cellpose
source activate cell_profiler_v4.0

for imageset in /hpc/scratch/hdd2/fs541623/Pre_Processed_Images_No_AnnexinV/*/*; do 
	timepoint=$(echo $imageset | grep -Eo 'Time_[0-9]+hrs_R[0-9]')
	well=$(echo $imageset | grep -Eo 'Well_[0-9]+')
	
	mkdir -p CP_Data/${timepoint}/${well}

	cellprofiler -c -r -p /hpc/scratch/hdd2/fs541623/CellProfilerFeatureExtraction/CP_pipeline/Dead_vs_Live_Count_190421_noAnnexin.cpproj -i $imageset -o /hpc/scratch/hdd2/fs541623/CellProfilerFeatureExtraction/CP_Data/${timepoint}/${well} 
done