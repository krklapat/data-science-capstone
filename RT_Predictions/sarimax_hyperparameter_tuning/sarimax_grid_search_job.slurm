#!/bin/bash
#SBATCH --job-name=sarimax_grid_search
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err
#SBATCH --array=0-35                # Matches IDs 0 through 17 in scaled_params.json
#SBATCH --cpus-per-task=4           # Allocate 4 CPU cores per SARIMAX job
#SBATCH --mem=8G                    # RAM per parallel task
#SBATCH --time=04:00:00             # Time wall limit

module load anaconda
conda activate capstone_gpu

mkdir -p logs run_3_results

python sarimax_tuning.py $SLURM_ARRAY_TASK_ID
