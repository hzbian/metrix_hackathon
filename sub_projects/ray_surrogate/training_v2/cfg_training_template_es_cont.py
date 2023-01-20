import os, sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import itertools

import torch
from pytorch_lightning.trainer.supporters import CombinedLoader

from ray_tools.simulation.torch_datasets import RayDataset

from ray_nn.data.transform import SurrogateModelPreparation
from ray_nn.data.lightning_data_module import DefaultDataModule
from ray_nn.utils.ray_processing import HistSubsampler
from ray_nn.nn.callbacks import ImagePlaneCallback, MemoryMonitor
from ray_nn.nn.models import SurrogateModel
from ray_nn.nn.backbones import TransformerBackbone, MLP
from ray_nn.metrics.geometric import SurrogateLoss, HistZeroAccuracy, NRaysAccuracy

from cfg_params_im2im import *

# --- Global ---

# Important fix to make custom collate_fn work
# https://forums.fast.ai/t/runtimeerror-received-0-items-of-ancdata/48935
torch.multiprocessing.set_sharing_strategy('file_system')

# --- Name & Paths ---
RUN_ID = 'template_es_n_rays_known__reference__cont_v3'
RESULTS_PATH = 'results'
RUN_PATH = os.path.join(RESULTS_PATH, RUN_ID)
WANDB_ONLINE = True
RESUME_RUN = False

# --- Devices & Global Seed ---
DEVICE = 'cuda'
GPU_ID = 0
TRAINING_SEED = 42

# --- Dataset ---
H5_PATH = os.path.join('/scratch/metrix-hackathon/datasets/metrix_simulation/ray_surrogate')

PLANES_SUB = ["Exit Slit"]
N_PLANES = len(PLANES_SUB)
PLANES_INFO_SUB = {plane: PLANES_INFO[plane] for plane in PLANES_SUB}
HIST_KEYS = list(itertools.chain(*[PLANES_INFO[plane][0] for plane in PLANES_SUB]))

DATASET = RayDataset(h5_files=[os.path.join(H5_PATH, file) for file in os.listdir(H5_PATH) if file.endswith('.h5')],
                     nested_groups=False,
                     sub_groups=[PARAMS_KEY] + HIST_KEYS,
                     transform=SurrogateModelPreparation(planes_info=PLANES_INFO_SUB,
                                                         params_key=PARAMS_KEY,
                                                         params_info=PARAMS_INFO,
                                                         hist_subsampler=HistSubsampler(factor=8)))

# --- Dataloaders ---
MAX_EPOCHS = 100
FRAC_TRAIN_SAMPLES = 1.0
FRAC_VAL_SAMPLES = 1.0
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
    OrderedDict([('reference', data_module.val_dataloader())]),
    mode="max_size_cycle")

# --- Loss & Validation Metrics ---
LOSS_FUNC = ({plane: SurrogateLoss for plane in PLANES},
             {plane: dict(sinkhorn_p=1,
                          sinkhorn_blur=0.05,
                          sinkhorn_normalize='weights2',
                          sinkhorn_standardize_lims=True,
                          total_weight=1.0,
                          lims_loss_weight=0.0,
                          n_rays_loss_weight=1.0,
                          hist_zero_loss_weight=1.0) for plane in PLANES})
VAL_METRICS = [('hist_zero_acc', HistZeroAccuracy()), ('n_rays_acc', NRaysAccuracy())]
MONITOR_VAL_LOSS = 'val/loss/reference'

# --- Optimization ---
OPTIMIZER = (torch.optim.Adam, {"lr": 2e-4, "eps": 1e-5, "weight_decay": 1e-4})
SCHEDULER = (torch.optim.lr_scheduler.StepLR, {"step_size": 2000, "gamma": 0.97})

# --- Callbacks ---
CALLBACKS = []
CALLBACKS += [ImagePlaneCallback(plane=plane, num_plots=20, overwrite_epoch=True) for plane in PLANES_SUB]
CALLBACKS += [MemoryMonitor()]

# --- Surrogate Model ---
if RESUME_RUN:
    SURROGATE = SurrogateModel.load_from_checkpoint(os.path.join(RUN_PATH, 'last.ckpt'))
else:
    n_hist_layers = [len(PLANES_INFO[plane][0]) for plane in PLANES]
    BACKBONE = ({plane: TransformerBackbone for plane in PLANES},
                {plane: dict(hist_dim=(32, 32),
                             n_hist_layers_inp=8,
                             n_hist_layers_out=n_hist_layers[idx],
                             param_dim=len(PARAMS_INFO),
                             transformer_dim=1024,
                             transformer_mlp_dim=2048,
                             transformer_heads=4,
                             transformer_layers=3,
                             use_inp_template=True) for idx, plane in enumerate(PLANES)})

    N_RAYS_PREDICTOR = ({plane: MLP for plane in PLANES},
                        {plane: dict(dim_in=len(PARAMS_INFO),
                                     dim_hidden=5 * [256],
                                     dim_out=n_hist_layers[idx]) for idx, plane in enumerate(PLANES)})

    HIST_ZERO_CLASSIFIER = ({plane: MLP for plane in PLANES},
                            {plane: dict(dim_in=len(PARAMS_INFO),
                                         dim_hidden=5 * [256],
                                         dim_out=1) for idx, plane in enumerate(PLANES)})

    SURROGATE = SurrogateModel(
        planes=PLANES,
        backbone=BACKBONE[0],
        backbone_params=BACKBONE[1],
        use_prev_plane_pred=True,  # TODO: add config param
        n_rays_known=False,
        loss_func=LOSS_FUNC[0],
        loss_func_params=LOSS_FUNC[1],
        hist_zero_classifier=HIST_ZERO_CLASSIFIER[0],
        hist_zero_classifier_params=HIST_ZERO_CLASSIFIER[1],
        n_rays_predictor=N_RAYS_PREDICTOR[0],
        n_rays_predictor_params=N_RAYS_PREDICTOR[1],
        optimizer=OPTIMIZER[0],
        optimizer_params=OPTIMIZER[1],
        scheduler=SCHEDULER[0],
        scheduler_params=SCHEDULER[1],
        val_metrics=VAL_METRICS)

    # load model to be continued
    ckpt = torch.load(os.path.join(RESULTS_PATH, 'template_es_n_rays_known__reference', 'last.ckpt'))
    SURROGATE.load_state_dict(ckpt['state_dict'], strict=False)

    SURROGATE.freeze()

    SURROGATE.planes = PLANES_SUB
    for plane in PLANES_SUB:
        for p in SURROGATE.n_rays_predictor[plane].parameters():
            p.requires_grad = True
