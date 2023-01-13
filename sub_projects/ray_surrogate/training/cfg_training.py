import os, sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

from collections import OrderedDict

import torch
from torch.utils.data import WeightedRandomSampler, DataLoader
from pytorch_lightning.trainer.supporters import CombinedLoader

from ray_tools.simulation.torch_datasets import RayDataset

from ray_nn.data.transform import SurrogatePreparation
from ray_nn.data.lightning_data_module import DefaultDataModule
from ray_nn.utils.ray_processing import HistSubsampler

from sub_projects.ray_surrogate.callbacks import LogPredictionsCallback
from sub_projects.ray_surrogate.nn_models import MLP, SurrogateModel
from sub_projects.ray_surrogate.losses import SurrogateLoss

from cfg_params_all import *

# --- Global ---

# Important fix to make custom collate_fn work
# https://forums.fast.ai/t/runtimeerror-received-0-items-of-ancdata/48935
torch.multiprocessing.set_sharing_strategy('file_system')

# --- Name & Paths ---
RUN_ID = 'ip_v1_nothing_given_full_data_rerun'
RESULTS_PATH = 'results'
RUN_PATH = os.path.join(RESULTS_PATH, RUN_ID)
WANDB_ONLINE = True
RESUME_RUN = False

# --- Devices & Global Seed ---
DEVICE = 'cuda'
GPU_ID = 0
TRAINING_SEED = 42

# --- Dataset ---
H5_PATH = os.path.join('/scratch/metrix-hackathon/datasets/metrix_simulation/ray_enhance_final')
PARAMS_KEY = '1e6/params'
HIST_KEY = '1e6/ray_output/ImagePlane/ml/0'

N_RAYS = torch.load('/scratch/metrix-hackathon/datasets/metrix_simulation/n_rays_ray_enhance_final.pt')
DATASET = RayDataset(h5_files=[os.path.join(H5_PATH, file) for file in os.listdir(H5_PATH) if file.endswith('.h5')],
                     nested_groups=False,
                     sub_groups=[PARAMS_KEY, HIST_KEY],
                     transform=SurrogatePreparation(params_key=PARAMS_KEY,
                                                    params_info=PARAMS_INFO,
                                                    hist_key=HIST_KEY,
                                                    hist_subsampler=HistSubsampler(factor=8)))

# --- Dataloaders ---
MAX_EPOCHS = 150
FRAC_TRAIN_SAMPLES = 0.15
FRAC_VAL_SAMPLES = 0.025
BATCH_SIZE_TRAIN = 256
BATCH_SIZE_VAL = 256
DATA_SPLIT = [0.95, 0.05, 0.00]
DL_NUM_WORKERS = 25
NUM_SANITY_VAL_STEPS = 0
SEED_TRAIN_DATA = 43
SEED_DATA_SPLIT = 44

data_module = DefaultDataModule(dataset=DATASET,
                                batch_size_train=BATCH_SIZE_TRAIN,
                                batch_size_val=BATCH_SIZE_VAL,
                                split=DATA_SPLIT,
                                num_workers=DL_NUM_WORKERS,
                                on_gpu=f'{DEVICE}:{GPU_ID}',
                                seed_split=SEED_DATA_SPLIT,
                                seed_train=SEED_TRAIN_DATA)
data_module.setup()

TRAIN_DATALOADER = data_module.train_dataloader()
VAL_DATALOADER = CombinedLoader(
    OrderedDict([('reference', data_module.val_dataloader()),
                 ('many rays', DataLoader(DATASET,
                                          sampler=WeightedRandomSampler(
                                              weights=N_RAYS,
                                              num_samples=5000,
                                              replacement=False),
                                          batch_size=BATCH_SIZE_VAL,
                                          num_workers=DL_NUM_WORKERS,
                                          pin_memory=False if DEVICE == 'cpu' else True,
                                          pin_memory_device=f'{DEVICE}:{GPU_ID}'))
                 ]),
    mode="max_size_cycle")

# --- Loss & Validation Metrics ---
LOSS_FUNC = (SurrogateLoss, dict(sinkhorn_p=1,
                                 sinkhorn_blur=0.05,
                                 sinkhorn_normalize=True,
                                 sinkhorn_n_rays_weighting=False,
                                 sinkhorn_standardize_lims=True,
                                 sinkhorn_weight=1.0,
                                 mae_lims_weight=0.0,
                                 mae_hist_weight=0.0,
                                 mae_n_rays_weight=1e-3))
VAL_METRICS = []
MONITOR_VAL_LOSS = 'val/loss/reference'

# --- Optimization ---
OPTIMIZER = (torch.optim.Adam, {"lr": 2e-4, "eps": 1e-5, "weight_decay": 1e-4})
SCHEDULER = (torch.optim.lr_scheduler.StepLR, {"step_size": 2000, "gamma": 0.98})

# --- Callbacks ---
CALLBACKS = [
    LogPredictionsCallback(num_plots=50, overwrite_epoch=False)
]

# --- Surrogate Model ---
DIM_HIST = 1024
DIM_BOTTLENECK = 2 * 1024

if RESUME_RUN:
    SURROGATE = SurrogateModel.load_from_checkpoint(os.path.join(RUN_PATH, 'last.ckpt'))
else:
    SURROGATE = SurrogateModel(dim_bottleneck=DIM_BOTTLENECK,
                               net_bottleneck=MLP,
                               net_bottleneck_params=dict(dim_in=DIM_BOTTLENECK,
                                                          dim_out=DIM_BOTTLENECK,
                                                          dim_hidden=6 * [DIM_BOTTLENECK]),
                               net_lims=(MLP, MLP),
                               net_lims_params=(dict(dim_in=2,
                                                     dim_out=DIM_BOTTLENECK,
                                                     dim_hidden=[DIM_BOTTLENECK // 8,
                                                                 DIM_BOTTLENECK // 4,
                                                                 DIM_BOTTLENECK // 2]),
                                                dict(dim_in=DIM_BOTTLENECK,
                                                     dim_out=2,
                                                     dim_hidden=[DIM_BOTTLENECK // 2,
                                                                 DIM_BOTTLENECK // 4,
                                                                 DIM_BOTTLENECK // 8])),
                               net_hist=(MLP, MLP),
                               net_hist_params=(dict(dim_in=DIM_HIST,
                                                     dim_out=DIM_BOTTLENECK,
                                                     dim_hidden=6 * [DIM_HIST]),
                                                dict(dim_in=DIM_BOTTLENECK,
                                                     dim_out=DIM_HIST,
                                                     dim_hidden=6 * [DIM_HIST])),
                               enc_params=MLP,
                               enc_params_params=dict(dim_in=len(PARAMS_INFO),
                                                      dim_out=DIM_BOTTLENECK,
                                                      dim_hidden=[DIM_BOTTLENECK // 2,
                                                                  DIM_BOTTLENECK // 4,
                                                                  DIM_BOTTLENECK // 8]),
                               loss_func=LOSS_FUNC[0],
                               loss_func_params=LOSS_FUNC[1],
                               optimizer=OPTIMIZER[0],
                               optimizer_params=OPTIMIZER[1],
                               scheduler=SCHEDULER[0],
                               scheduler_params=SCHEDULER[1],
                               val_metrics=VAL_METRICS)
