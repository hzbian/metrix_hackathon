#!/bin/bash
#SBATCH --job-name=eval-bl-opti
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task 10
#SBATCH --mem=70GB
#SBATCH --partition=main
#SBATCH --spread-job
#SBATCH --exclude=gpu-a100-1

        srun ./local-venv.sh python3 -m ray_tools.hist_optimizer.evaluation_hist_optimizer
