#!/bin/bash
#SBATCH --job-name=dataset
#SBATCH --gres=gpu:0
#SBATCH --nodes=1
#SBATCH --array=1-11
#SBATCH --cpus-per-task 100
#SBATCH --mem-per-cpu=1GB
#SBATCH --partition=main
#SBATCH --spread-job

        ./local-venv.sh python3 -m datasets.dataset_tools.script_generate_data
