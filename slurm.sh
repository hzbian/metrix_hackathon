#!/bin/bash
#SBATCH --job-name=bl_tpe
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task 25
#SBATCH --mem-per-cpu=2GB
#SBATCH --partition=main
#SBATCH --spread-job

./local-venv.sh python3 -m  "sub_projects.ray_optimization.ray_optimization" target_configuration=metrixs_real target_configuration.max_target_deviation=0.3 # ray_optimizer.criterion._target_="sub_projects.ray_optimization.losses.torch.MSELoss" #+study_name.appendix="big_e2_tY+image_plane_tX" optimization_target_configuration.param_func.parameters={E2.translationYerror:"[-3.,3.]"} target_configuration.param_func.parameters={ImagePlane.translationXerror:"[-10.0,10.0]"}
