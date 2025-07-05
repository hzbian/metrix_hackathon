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
pip install -U torch torchvision pytorch-lightning lightning docker hydra-core h5py ax-platform evotorch optuna plotly matplotlib scipy numpy pandas raypyng wandb blop
pip install -U pykeops geomloss
micromamba install -y -c nvidia cuda-nvrtc-dev
"$@"
