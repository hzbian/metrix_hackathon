#!/bin/bash
#SBATCH --job-name=nn
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task 15
#SBATCH --mem=200GB
#SBATCH --partition=main
#SBATCH --spread-job

        srun ./local-venv.sh python3 -m ray_nn.nn.xy_hist_data_models
