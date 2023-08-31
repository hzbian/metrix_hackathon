import optuna
import torchvision.ops
from optuna.samplers import TPESampler
from scipy.optimize import basinhopping

from ray_optim.ray_optimizer import OptimizerBackendOptuna, OptimizerBackendBasinhopping, OptimizerBackendEvoTorch
from ray_tools.base.engine import GaussEngine
from ray_tools.base.parameter import RayParameterContainer, NumericalParameter, RandomParameter
from ray_tools.base.transform import MultiLayer
from ray_tools.base.utils import RandomGenerator
from sub_projects.ray_optimization.losses import BoxIoULoss
from sub_projects.ray_optimization.ray_optimization import RayOptimization

# objective
REAL_DATA_DIR = None
EXPORTED_PLANE = "ImagePlane"
MAX_TARGET_DEVIATION = 0.3
MAX_OFFSET_SEARCH_DEVIATION = 0.3
N_RAYS = ['1e4']
Z_LAYERS = [0., 5.]#(-15, -10, -5, 0, 5, 10, 15, 20, 25, 30)
TRANSFORMS = MultiLayer(Z_LAYERS, copy_directions=False)
NUM_BEAMLINE_PARAM_SAMPLES = 4
RG = RandomGenerator(seed=42)

PARAM_FUNC = lambda: RayParameterContainer([
    ("number_rays", NumericalParameter(value=1e2)),
    ("x_dir", NumericalParameter(value=0.)),
    ("y_dir", NumericalParameter(value=0.)),
    ("z_dir", NumericalParameter(value=1.)),
    ("direction_spread", NumericalParameter(value=0.)),
    ("correlation_factor", RandomParameter(value_lims=(-0.8, 0.8), rg=RG)),
    ("x_mean", RandomParameter(value_lims=(-2, 2.), rg=RG)),
    ("y_mean", RandomParameter(value_lims=(-2., 2.), rg=RG)),
    ("x_var", RandomParameter(value_lims=(1e-10, 0.01), rg=RG, enforce_lims=True)),
    ("y_var", RandomParameter(value_lims=(1e-10, 0.01), rg=RG, enforce_lims=True)),
])

FIXED_PARAMS = []#[k for k, v in PARAM_FUNC().items() if k not in 'y_var' and isinstance(v, RandomParameter)]
OVERWRITE_OFFSET = lambda: RayParameterContainer([
#    ("y_var", RandomParameter(value_lims=(1e-10, 0.00001), rg=RG)),
])

# multi objective
MULTI_OBJECTIVE = False
MULTI_OBJECTIVE_DIRECTIONS = ['minimize', 'minimize']

# optimization
CRITERION = BoxIoULoss(torchvision.ops.complete_box_iou_loss, reduction='mean')
ITERATIONS = 1000
OPTIMIZER = ['optuna', 'evotorch', 'basinhopping'][0]
SAMPLER = TPESampler()  # n_startup_trials=100, n_ei_candidates=100) #optuna.samplers.CmaEsSampler()

# logging
STUDY_NAME = '-'.join(
    [str(sum(isinstance(x, RandomParameter) for x in PARAM_FUNC().values()) - len(FIXED_PARAMS)), 'gauss', str(MAX_TARGET_DEVIATION),
     OPTIMIZER, '-v23'])
WANDB_ENTITY = 'hzb-aos'
WANDB_PROJECT = 'metrix_hackathon_gauss'
OPTUNA_STORAGE_PATH = "sqlite:////dev/shm/db.sqlite2"
PLOT_INTERVAL = 100
LOGGING = False
VERBOSE = False

directions = MULTI_OBJECTIVE_DIRECTIONS if MULTI_OBJECTIVE else None
ENGINE = GaussEngine()
if OPTIMIZER == 'optuna':
    optuna_storage_path = OPTUNA_STORAGE_PATH if LOGGING else None
    optuna_study = optuna.create_study(directions=directions, sampler=SAMPLER,
                                       pruner=optuna.pruners.HyperbandPruner(),
                                       storage=optuna_storage_path, study_name=STUDY_NAME, load_if_exists=True)
    optimizer_backend = OptimizerBackendOptuna(optuna_study)
elif OPTIMIZER == 'evotorch':
    optimizer_backend = OptimizerBackendEvoTorch()
else:
    optimizer_backend = OptimizerBackendBasinhopping(basinhopping)

OPTIMIZER_BACKEND = optimizer_backend
RayOptimization(engine=ENGINE, optimizer_backend=OPTIMIZER_BACKEND, z_layers=Z_LAYERS, criterion=CRITERION, param_func=PARAM_FUNC, fixed_params=FIXED_PARAMS, overwrite_offset_func=OVERWRITE_OFFSET, max_offset_search_deviation=MAX_OFFSET_SEARCH_DEVIATION, max_target_deviation=MAX_TARGET_DEVIATION, iterations=ITERATIONS, study_name=STUDY_NAME, transforms=TRANSFORMS, wandb_entity=WANDB_ENTITY, wandb_project=WANDB_PROJECT, exported_plane=EXPORTED_PLANE, plot_interval=PLOT_INTERVAL, num_beamline_samples=NUM_BEAMLINE_PARAM_SAMPLES, real_data_configuration=None, logging=LOGGING, rg=RG)
