import os
import sys

from optuna.samplers import TPESampler

sys.path.insert(0, '../../../')
from ray_tools.base.parameter import RayParameterContainer, NumericalParameter, RandomParameter, RandomOutputParameter
from ray_tools.base.transform import MultiLayer
from ray_tools.base.utils import RandomGenerator

# logging
STUDY_NAME = '34-3-12-v2-Layer-0.3-TPE_consider_prior_false'
WANDB_ENTITY = 'hzb-aos'
WANDB_PROJECT = 'metrix_hackathon_offsets'
OPTUNA_STORAGE_PATH = "sqlite:////dev/shm/db.sqlite2"
LOGGING = False
VERBOSE = False

# paths
ROOT_DIR = '../../'
RML_BASEFILE = os.path.join(ROOT_DIR, 'rml_src', 'METRIX_U41_G1_H1_318eV_PS_MLearn.rml')
RAY_WORKDIR = os.path.join(ROOT_DIR, 'ray_workdir', 'optimization')

# objective
REAL_DATA_DIR = None
EXPORTED_PLANE = "ImagePlane"
MAX_DEVIATION = 0.3
N_RAYS = ['1e4']
Z_LAYERS = [-26, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30]
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
    ("M1_Cylinder.translationXerror", RandomParameter(value_lims=(-0.15, 0.15), rg=RG)),
    ("M1_Cylinder.translationYerror", RandomParameter(value_lims=(-1., 1.), rg=RG)),
    ("SphericalGrating.radius", RandomParameter(value_lims=(109741., 109841.), rg=RG)),
    ("SphericalGrating.rotationYerror", RandomParameter(value_lims=(-1., 1.), rg=RG)),
    ("SphericalGrating.rotationZerror", RandomParameter(value_lims=(-2.5, 2.5), rg=RG)),
    ("ExitSlit.totalHeight", RandomParameter(value_lims=(0.009, 0.011), rg=RG)),
    ("ExitSlit.translationZerror", RandomParameter(value_lims=(-30., 30.), rg=RG)),
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
    ("ImagePlane.translationXerror", RandomOutputParameter(value_lims=(-1, 1), rg=RG)),
    ("ImagePlane.translationYerror", RandomOutputParameter(value_lims=(-1, 1), rg=RG)),
    ("ImagePlane.translationZerror", RandomOutputParameter(value_lims=(-1, 1), rg=RG))
])

# multi objective
MULTI_OBJECTIVE = False
MULTI_OBJECTIVE_DIRECTIONS = ['minimize', 'minimize']

# optimization
ITERATIONS = 1000
OPTIMIZER = 'optuna'
SAMPLER = TPESampler(consider_prior=False)  # optuna.samplers.CmaEsSampler()
