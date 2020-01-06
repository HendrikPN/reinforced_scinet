#!/bin/bash

# run this file with slurm as `sbatch _run_remote.sh`

# set name of job
#SBATCH --job-name=OMR

# set name of output
#SBATCH --output=OMR.out

# get partition
#SBATCH -p tv

# set number of nodes
#SBATCH -N 1

# create required directories
[ ! -d "saved_models" ] && { echo "Create network directory"; mkdir saved_models; }
[ ! -d "results_log" ] && { echo "Create result directory"; mkdir results_log; }

# check whether results exist and delete
[ -f "results_log/results.txt" ] && { echo "Remove previous results"; rm results_log/results.txt; }
[ -f "results_log/results_loss.txt" ] && { echo "Remove previous AE results"; rm results_log/results_loss.txt; }
[ -f "results_log/selection.txt" ] && { echo "Remove previous selection results"; rm results_log/selection.txt; }

# run main
module purge
module load anaconda3
source activate pytorch13

export OMP_NUM_THREADS=1
python main.py
