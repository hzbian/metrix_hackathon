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
Z_LAYERS = [0]#[-15, -10, -5, 0, 5, 10, 15, 20, 25, 30]
TRANSFORMS = MultiLayer(Z_LAYERS, copy_directions=False)
NUM_BEAMLINE_PARAM_SAMPLES = 1
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
    ("x_var", RandomParameter(value_lims=(0.5, 1.0), rg=RG)),
    ("y_var", RandomParameter(value_lims=(0.5, 1.0), rg=RG)),
])

FIXED_PARAMS = []#[k for k, v in PARAM_FUNC().items() if k not in 'y_var' and isinstance(v, RandomParameter)]

# multi objective
MULTI_OBJECTIVE = False
MULTI_OBJECTIVE_DIRECTIONS = ['minimize', 'minimize']

# optimization
ITERATIONS = 1000
OPTIMIZER = ['optuna', 'evotorch', 'basinhopping'][0]
SAMPLER = TPESampler()  # n_startup_trials=100, n_ei_candidates=100) #optuna.samplers.CmaEsSampler()

# logging
STUDY_NAME = '-'.join(
    [str(sum(isinstance(x, RandomParameter) for x in PARAM_FUNC().values()) - len(FIXED_PARAMS)), 'gauss', str(MAX_DEVIATION),
     OPTIMIZER, '-v23'])
WANDB_ENTITY = 'hzb-aos'
WANDB_PROJECT = 'metrix_hackathon_gauss'
OPTUNA_STORAGE_PATH = "sqlite:////dev/shm/db.sqlite2"
PLOT_INTERVAL = 100
LOGGING = True
VERBOSE = False

ENGINE = GaussEngine()
