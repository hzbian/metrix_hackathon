#!/bin/bash
#SBATCH --job-name=nn
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task 64
#SBATCH --mem-per-cpu=1GB
#SBATCH --partition=main
#SBATCH --spread-job

        ./local-venv.sh python3 -m ray_nn.nn.xy_hist_data_models
