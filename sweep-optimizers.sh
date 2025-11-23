#!/bin/bash
#SBATCH --job-name=sweep-bl-opti
#SBATCH --gres=gpu:0
#SBATCH --nodes=1
#SBATCH --cpus-per-task 10
#SBATCH --mem=10GB
#SBATCH --array=0,1,2,4
#SBATCH --partition=main
#SBATCH --spread-job

        srun ./local-venv.sh python3 -m ray_tools.hist_optimizer.evaluate_optimizers $SLURM_ARRAY_TASK_ID
