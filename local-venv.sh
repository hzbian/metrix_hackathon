#!/bin/bash
ulimit -n 4096
export MAMBA_ROOT_PREFIX="/scratch/$USER/micromamba"
micromamba="/mnt/work/xfel/env/micromamba-1.4.4-0/bin/micromamba"
eval "$($micromamba shell hook --shell bash)"
micromamba activate ray-ssd
if [ ! $? -eq 0 ]; then
	micromamba create -y -n ray-ssd -c conda-forge python=3.11
	micromamba activate ray-ssd
fi
pip install -U torch==2.8.0 torchvision pytorch-lightning lightning docker hydra-core h5py ax-platform botorch evotorch optuna plotly matplotlib scipy numpy pandas raypyng wandb blop==0.8.0 seaborn scikit-learn xgboost jsonargparse[signatures]
pip install -U pykeops geomloss
micromamba install -y -c nvidia cuda-nvrtc-dev
"$@"
