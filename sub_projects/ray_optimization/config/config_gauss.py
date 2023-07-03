from optuna.samplers import TPESampler

from ray_tools.base.engine import GaussEngine
from ray_tools.base.parameter import RayParameterContainer, NumericalParameter, RandomParameter
from ray_tools.base.transform import MultiLayer
from ray_tools.base.utils import RandomGenerator

# objective
REAL_DATA_DIR = None
EXPORTED_PLANE = "ImagePlane"
MAX_DEVIATION = 0.3
N_RAYS = ['1e4']
Z_LAYERS = [-15, -10, -5, 0, 5, 10, 15, 20, 25, 30]
TRANSFORMS = MultiLayer(Z_LAYERS, copy_directions=False)
NUM_BEAMLINE_PARAM_SAMPLES = 22
RG = RandomGenerator(seed=42)

PARAM_FUNC = lambda: RayParameterContainer([
    ("number_rays", NumericalParameter(value=1e4)),
    ("x_dir", NumericalParameter(value=1.)),
    ("y_dir", NumericalParameter(value=1.)),
    ("z_dir", NumericalParameter(value=1.)),
    ("direction_spread", NumericalParameter(value=0.)),
    ("x_mean", RandomParameter(value_lims=(-1., 1.), rg=RG)),
    ("y_mean", RandomParameter(value_lims=(-1., 1.), rg=RG)),
    ("x_var", RandomParameter(value_lims=(0.003, 0.006), rg=RG)),
    ("y_var", RandomParameter(value_lims=(0.003, 0.006), rg=RG)),
])

FIXED_PARAMS = []  # [k for k in PARAM_FUNC().keys()][1:-3]

# multi objective
MULTI_OBJECTIVE = False
MULTI_OBJECTIVE_DIRECTIONS = ['minimize', 'minimize']

# optimization
ITERATIONS = 1000
OPTIMIZER = ['optuna', 'evotorch'][0]
SAMPLER = TPESampler()  # n_startup_trials=100, n_ei_candidates=100) #optuna.samplers.CmaEsSampler()

# logging
STUDY_NAME = '-'.join(
    [str(len(PARAM_FUNC()) - len(FIXED_PARAMS)), 'gauss', str(MAX_DEVIATION),
     OPTIMIZER, 'v5'])
WANDB_ENTITY = 'hzb-aos'
WANDB_PROJECT = 'metrix_hackathon_gauss'
OPTUNA_STORAGE_PATH = "sqlite:////dev/shm/db.sqlite2"
LOGGING = True
VERBOSE = True

ENGINE = GaussEngine()
