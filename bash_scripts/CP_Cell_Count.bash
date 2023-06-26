#!/bin/bash 

#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --job-name=CellprofilerFeatureAnnexin040521
#SBATCH --ntasks=1
#SBATCH --output=CP_FeatureExtractionAnnexin040521_%A.out
#SBATCH --error=CP_FeatureExtractionAnnexin040521_%A_%a.err
#SBATCH --cpus-per-task=10
#SBATCH --partition=stv-gpu

module purge 
module load anaconda3

# Load enviornment containing cellprofiler
source activate cell_profiler_v4.0

for imageset in /hpc/scratch/hdd2/fs541623/Pre_processed_Images/*/*; do 
	timepoint=$(echo $imageset | grep -Eo 'Time_[0-9]+hrs_R[0-9]')
	well=$(echo $imageset | grep -Eo 'Well_[0-9]+')
	
	mkdir -p CP_Data/${timepoint}/${well}

	cellprofiler -c -r -p /hpc/scratch/hdd2/fs541623/CellProfilerFeatureExtraction/CP_pipeline/Dead_vs_Live_Count_190421.cpproj -i $imageset -o /hpc/scratch/hdd2/fs541623/CellProfilerFeatureExtraction/CP_Data/${timepoint}/${well} 
done