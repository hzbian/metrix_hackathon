import os

from optuna.samplers import TPESampler

from ray_tools.base.parameter import RayParameterContainer, NumericalParameter, RandomParameter
from ray_tools.base.transform import MultiLayer
from ray_tools.base.utils import RandomGenerator

# logging
STUDY_NAME = 'real-data-test-0.3-new'
WANDB_ENTITY = 'hzb-aos'
WANDB_PROJECT = 'metrix_hackathon_offsets'
OPTUNA_STORAGE_PATH = "sqlite:////dev/shm/db.sqlite2"
LOGGING = True
VERBOSE = False

# paths
ROOT_DIR = '../../'
RML_BASEFILE = os.path.join(ROOT_DIR, 'rml_src', 'METRIX_U41_G1_H1_318eV_PS_MLearn.rml')
RAY_WORKDIR = os.path.join(ROOT_DIR, 'ray_workdir', 'optimization')

# objective
REAL_DATA_DIR = '../../datasets/metrix_real_data/2021_march_selected'
EXPORTED_PLANE = "ImagePlane"
MAX_DEVIATION = 0.3
N_RAYS = ['1e4']
Z_LAYERS = [-15, -10, -5, 0, 5, 10, 15, 20, 25, 30]
TRANSFORMS = MultiLayer(Z_LAYERS, copy_directions=False)
NUM_BEAMLINE_PARAM_SAMPLES = 22
FIXED_PARAMS = []
RG = RandomGenerator(seed=42)
PARAM_FUNC = lambda: RayParameterContainer([
    ("U41_318eV.numberRays", NumericalParameter(value=1e4)),
    ("U41_318eV.translationXerror", RandomParameter(value_lims=(-0.25, 0.25), rg=RG)),
    ("U41_318eV.translationYerror", RandomParameter(value_lims=(-0.25, 0.25), rg=RG)),
    ("U41_318eV.rotationXerror", RandomParameter(value_lims=(-0.05, 0.05), rg=RG)),
    ("U41_318eV.rotationYerror", RandomParameter(value_lims=(-0.05, 0.05), rg=RG)),
    ("ASBL.totalWidth", RandomParameter(value_lims=(1.9, 2.1), rg=RG)),
    ("ASBL.totalHeight", RandomParameter(value_lims=(0.9, 1.1), rg=RG)),
    ("ASBL.translationXerror", RandomParameter(value_lims=(-0.2, 0.2), rg=RG)),
    ("ASBL.translationYerror", RandomParameter(value_lims=(-0.2, 0.2), rg=RG)),
    ("M1_Cylinder.radius", RandomParameter(value_lims=(174.06, 174.36), rg=RG)),
    ("M1_Cylinder.rotationXerror", RandomParameter(value_lims=(-0.25, 0.25), rg=RG)),
    ("M1_Cylinder.rotationYerror", RandomParameter(value_lims=(-1., 1.), rg=RG)),
    ("M1_Cylinder.rotationZerror", RandomParameter(value_lims=(-1., 1.), rg=RG)),
    ("M1_Cylinder.translationXerror", RandomParameter(value_lims=(-1., 1.), rg=RG)),
    ("M1_Cylinder.translationYerror", RandomParameter(value_lims=(-1., 1.), rg=RG)),
    ("SphericalGrating.radius", RandomParameter(value_lims=(109741., 109841.), rg=RG)),
    ("SphericalGrating.rotationYerror", RandomParameter(value_lims=(-1., 1.), rg=RG)),
    ("SphericalGrating.rotationZerror", RandomParameter(value_lims=(-2.5, 2.5), rg=RG)),
    ("ExitSlit.totalHeight", RandomParameter(value_lims=(0.009, 0.011), rg=RG)),
    ("ExitSlit.translationZerror", RandomParameter(value_lims=(-31., 31.), rg=RG)),
    ("ExitSlit.rotationZerror", RandomParameter(value_lims=(-0.3, 0.3), rg=RG)),
    ("E1.longHalfAxisA", RandomParameter(value_lims=(20600., 20900.), rg=RG)),
    ("E1.shortHalfAxisB", RandomParameter(value_lims=(300.721702601, 304.721702601), rg=RG)),
    ("E1.rotationXerror", RandomParameter(value_lims=(-0.5, 0.5), rg=RG)),
    ("E1.rotationYerror", RandomParameter(value_lims=(-7.5, 7.5), rg=RG)),
    ("E1.rotationZerror", RandomParameter(value_lims=(-4, 4), rg=RG)),
    ("E1.translationYerror", RandomParameter(value_lims=(-1, 1), rg=RG)),
    ("E1.translationZerror", RandomParameter(value_lims=(-1, 1), rg=RG)),
    ("E2.longHalfAxisA", RandomParameter(value_lims=(4325., 4425.), rg=RG)),
    ("E2.shortHalfAxisB", RandomParameter(value_lims=(96.1560870104, 98.1560870104), rg=RG)),
    ("E2.rotationXerror", RandomParameter(value_lims=(-0.5, 0.5), rg=RG)),
    ("E2.rotationYerror", RandomParameter(value_lims=(-7.5, 7.5), rg=RG)),
    ("E2.rotationZerror", RandomParameter(value_lims=(-4, 4), rg=RG)),
    ("E2.translationYerror", RandomParameter(value_lims=(-1, 1), rg=RG)),
    ("E2.translationZerror", RandomParameter(value_lims=(-1, 1), rg=RG)),
])

# multi objective
MULTI_OBJECTIVE = False
MULTI_OBJECTIVE_DIRECTIONS = ['minimize', 'minimize']

# optimization
ITERATIONS = 1000
OPTIMIZER = 'optuna'
SAMPLER = TPESampler()  # n_startup_trials=100, n_ei_candidates=100) #optuna.samplers.CmaEsSampler()