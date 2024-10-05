#!/bin/bash
#SBATCH --job-name=dataset
#SBATCH --gres=gpu:0
#SBATCH --nodes=1
#SBATCH --array=1-5
#SBATCH --cpus-per-task 100
#SBATCH --mem-per-cpu=1GB
#SBATCH --partition=main
#SBATCH --spread-job

        ./local-venv.sh python3 -m ray_tools.hist_optimizer.create_hist_data $SLURM_ARRAY_TASK_ID
