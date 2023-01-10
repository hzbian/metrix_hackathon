import os, sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

from collections import OrderedDict

import torch
from torch.utils.data import WeightedRandomSampler, DataLoader
from pytorch_lightning.trainer.supporters import CombinedLoader

from ray_tools.simulation.torch_datasets import RayDataset

from ray_nn.data.transform import SurrogatePreparation
from ray_nn.data.lightning_data_module import DefaultDataModule
from ray_nn.metrics.geometric import SinkhornLoss
from ray_nn.utils.ray_processing import HistSubsampler, HistToPointCloud

from sub_projects.ray_surrogate.callbacks import LogPredictionsCallback
from sub_projects.ray_surrogate.nn_models import MLP

# --- Name & Paths ---
RUN_ID = 'test_v1_436347'
RESULTS_PATH = 'results'
RUN_PATH = os.path.join(RESULTS_PATH, RUN_ID)
WANDB_ONLINE = False
RESUME_RUN = False

# --- Devices & Global Seed ---
DEVICE = 'cuda'
GPU_ID = 0
TRAINING_SEED = 42

# --- Surrogate Model ---
LIST_PARAMS = ['ASBL.totalHeight', 'ASBL.totalWidth', 'ASBL.translationXerror', 'ASBL.translationYerror',
               'E1.longHalfAxisA', 'E1.rotationXerror', 'E1.rotationYerror', 'E1.rotationZerror', 'E1.shortHalfAxisB',
               'E1.translationYerror', 'E1.translationZerror', 'E2.longHalfAxisA', 'E2.rotationXerror',
               'E2.rotationYerror', 'E2.rotationZerror', 'E2.shortHalfAxisB', 'E2.translationYerror',
               'E2.translationZerror', 'ExitSlit.rotationZerror', 'ExitSlit.totalHeight', 'ExitSlit.translationZerror',
               'ImagePlane.distanceImagePlane', 'M1_Cylinder.radius', 'M1_Cylinder.rotationXerror',
               'M1_Cylinder.rotationYerror', 'M1_Cylinder.rotationZerror', 'M1_Cylinder.translationXerror',
               'M1_Cylinder.translationYerror', 'SphericalGrating.radius', 'SphericalGrating.rotationYerror',
               'SphericalGrating.rotationZerror', 'U41_318eV.numberRays', 'U41_318eV.rotationXerror',
               'U41_318eV.rotationYerror', 'U41_318eV.translationXerror', 'U41_318eV.translationYerror']

PC_SUPP_DIM = 1024
PARAM_DIM = len(LIST_PARAMS)
PARAM_EMBEDDING_DIM = 128
BASE_NET = (MLP, dict(dim_in=3 * PC_SUPP_DIM + PARAM_EMBEDDING_DIM,
                      dim_out=3 * PC_SUPP_DIM,
                      dim_hidden=3 * [5 * PC_SUPP_DIM]))
PARAM_PREPROCESSOR = (MLP, dict(dim_in=len(LIST_PARAMS),
                                dim_out=PARAM_EMBEDDING_DIM,
                                dim_hidden=3 * [PARAM_EMBEDDING_DIM]))
HIST_SUBSAMPLER = (HistSubsampler, dict(factor=8))
HIST_TO_PC = (HistToPointCloud, dict())

# --- Data ---
H5_PATH = os.path.join('/scratch/metrix-hackathon/datasets/metrix_simulation/ray_enhance_final')

KEY_PARAMS = '1e6/params'
KEY_HIST = '1e6/ray_output/ImagePlane/ml/0'

FRAC_TRAIN_SAMPLES = 0.05
FRAC_VAL_SAMPLES = 0.05
BATCH_SIZE_TRAIN = 256
BATCH_SIZE_VAL = 256
DATA_SPLIT = [0.95, 0.05, 0.00]
DL_NUM_WORKERS = 25
NUM_SANITY_VAL_STEPS = 0

SEED_TRAIN_DATA = 43
SEED_DATA_SPLIT = 44

# Important fix to make custom collate_fn work
# https://forums.fast.ai/t/runtimeerror-received-0-items-of-ancdata/48935
torch.multiprocessing.set_sharing_strategy('file_system')

dataset = RayDataset(h5_files=[os.path.join(H5_PATH, file) for file in os.listdir(H5_PATH) if file.endswith('.h5')],
                     nested_groups=False,
                     sub_groups=[KEY_PARAMS, KEY_HIST],
                     transform=SurrogatePreparation(key_params=KEY_PARAMS,
                                                    list_params=LIST_PARAMS,
                                                    key_hist=KEY_HIST))

data_module = DefaultDataModule(dataset=dataset,
                                batch_size_train=BATCH_SIZE_TRAIN,
                                batch_size_val=BATCH_SIZE_VAL,
                                split=DATA_SPLIT,
                                num_workers=DL_NUM_WORKERS,
                                on_gpu=f'{DEVICE}:{GPU_ID}',
                                seed_split=SEED_DATA_SPLIT,
                                seed_train=SEED_TRAIN_DATA)
data_module.setup()

N_RAYS = torch.load('/scratch/metrix-hackathon/datasets/metrix_simulation/n_rays_ray_enhance_final.pt')

TRAIN_DATALOADER = data_module.train_dataloader()
VAL_DATALOADER = CombinedLoader(
    OrderedDict([('reference', data_module.val_dataloader()),
                 ('many rays', DataLoader(dataset,
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

# --- Callbacks ---
CALLBACKS = [LogPredictionsCallback(num_plots=50, overwrite_epoch=True)]

# --- Loss & Validation Metrics ---
LOSS_FUNC = (SinkhornLoss, dict(p=2, normalize_weights=False, backend='online', reduction='mean'))
VAL_METRICS = []
MONITOR_VAL_LOSS = 'val/loss/reference'

# --- Training ---
MAX_EPOCHS = 25
OPTIMIZER = (torch.optim.Adam, {"lr": 2e-4, "eps": 1e-5, "weight_decay": 1e-4})
SCHEDULER = (torch.optim.lr_scheduler.StepLR, {"step_size": 1, "gamma": 1.0})
