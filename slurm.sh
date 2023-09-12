#!/bin/bash
#SBATCH --job-name=bl_tpe
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task 2
#SBATCH --mem-per-cpu=2GB
#SBATCH --partition=main
#SBATCH --spread-job

python3 -m  "sub_projects.ray_optimization.ray_optimization"
