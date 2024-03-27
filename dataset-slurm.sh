#!/bin/bash
#SBATCH --job-name=dataset
#SBATCH --gres=gpu:0
#SBATCH --nodes=1
#SBATCH --cpus-per-task 125
#SBATCH --mem-per-cpu=1GB
#SBATCH --partition=main
#SBATCH --spread-job

        ./local-venv.sh python3 -m datasets.metrix_simulation.ray_emergency_surrogate.rebin #datasets.dataset_tools.script_generate_data
