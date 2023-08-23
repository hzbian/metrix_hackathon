import os

import torchvision.ops
from optuna.samplers import TPESampler

from ray_tools.base.backend import RayBackendDockerRAYUI
from ray_tools.base.engine import RayEngine
from ray_tools.base.parameter import RayParameterContainer, NumericalParameter, RandomParameter, RandomOutputParameter
from ray_tools.base.transform import MultiLayer
from ray_tools.base.utils import RandomGenerator
from sub_projects.ray_optimization.losses import SinkhornLoss
from sub_projects.testing.loss_exploration import BoxIoULoss

# paths
ROOT_DIR = '../../'
RML_BASEFILE = os.path.join(ROOT_DIR, 'rml_src', 'METRIX_U41_G1_H1_318eV_PS_MLearn.rml')
RAY_WORKDIR = '/dev/shm'

# objective

## real data
REAL_DATA_DIR = '../../datasets/metrix_real_data/2021_march_complete'
REAL_DATA_TRAIN_SET = ['M03', 'M10', 'M18', 'M22', 'M23', 'M24', 'M25', 'M27', 'M28', 'M29', 'M30', 'M32', 'M33', 'M36',
                       'M37', 'M40', 'M41', 'M42', 'M43', 'M44']
REAL_DATA_VALIDATION_SET = ['M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M11', 'M12', 'M13', 'M14', 'M15', 'M16', 'M17',
                            'M18', 'M19', 'M20', 'M21', 'M26', 'M31', 'M34', 'M35', 'M38', 'M39']
EXPORTED_PLANE = "ImagePlane"
MAX_TARGET_DEVIATION = 0.3
MAX_OFFSET_SEARCH_DEVIATION = 0.3
N_RAYS = ['1e4']
Z_LAYERS = [-15, -10, -5, 0, 5, 10, 15, 20, 25, 30]
TRANSFORMS = MultiLayer(Z_LAYERS, copy_directions=False)
NUM_BEAMLINE_PARAM_SAMPLES = 22
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
    ("ExitSlit.translationZerror", RandomParameter(value_lims=(-150., 150.), rg=RG)),
    ("ExitSlit.rotationZerror", RandomParameter(value_lims=(-0.3, 0.3), rg=RG)),
    ("E1.longHalfAxisA", RandomParameter(value_lims=(20600., 20900.), rg=RG)),
    ("E1.shortHalfAxisB", RandomParameter(value_lims=(300.721702601, 304.721702601), rg=RG)),
    ("E1.rotationXerror", RandomParameter(value_lims=(-1.5, 1.5), rg=RG)),
    ("E1.rotationYerror", RandomParameter(value_lims=(-7.5, 7.5), rg=RG)),
    ("E1.rotationZerror", RandomParameter(value_lims=(7, 14), rg=RG)),
    ("E1.translationYerror", RandomParameter(value_lims=(-1, 1), rg=RG)),
    ("E1.translationZerror", RandomParameter(value_lims=(-1, 1), rg=RG)),
    ("E2.longHalfAxisA", RandomParameter(value_lims=(4325., 4425.), rg=RG)),
    ("E2.shortHalfAxisB", RandomParameter(value_lims=(96.1560870104, 98.1560870104), rg=RG)),
    ("E2.rotationXerror", RandomParameter(value_lims=(-0.5, 0.5), rg=RG)),
    ("E2.rotationYerror", RandomParameter(value_lims=(-7.5, 7.5), rg=RG)),
    ("E2.rotationZerror", RandomParameter(value_lims=(22., 32.), rg=RG)),
    ("E2.translationYerror", RandomParameter(value_lims=(-1, 1), rg=RG)),
    ("E2.translationZerror", RandomParameter(value_lims=(-1, 1), rg=RG)),
    ("ImagePlane.translationXerror", RandomOutputParameter(value_lims=(-3.33, 3.33), rg=RG)),
    ("ImagePlane.translationYerror", RandomOutputParameter(value_lims=(-3.33, 3.33), rg=RG)),
    ("ImagePlane.translationZerror", RandomOutputParameter(value_lims=(-3.33, 3.33), rg=RG)),
])
FIXED_PARAMS = []  # [k for k in PARAM_FUNC().keys()][1:-3]
OVERWRITE_OFFSET = lambda: RayParameterContainer([])

# multi objective
MULTI_OBJECTIVE = False
MULTI_OBJECTIVE_DIRECTIONS = ['minimize', 'minimize']

# optimization
CRITERION = SinkhornLoss() #BoxIoULoss(torchvision.ops.complete_box_iou_loss)
ITERATIONS = 1000
OPTIMIZER = ['optuna', 'evotorch'][0]
SAMPLER = TPESampler()  # n_startup_trials=100, n_ei_candidates=100) #optuna.samplers.CmaEsSampler()

# logging
STUDY_NAME = '-'.join(
    [str(len(PARAM_FUNC()) - len(FIXED_PARAMS)), 'real' if REAL_DATA_DIR is not None else 'sim', str(MAX_TARGET_DEVIATION),
     OPTIMIZER, 'v7'])
WANDB_ENTITY = 'hzb-aos'
WANDB_PROJECT = 'metrix_hackathon_offsets'
OPTUNA_STORAGE_PATH = "sqlite:////dev/shm/db.sqlite2"
PLOT_INTERVAL = 10
LOGGING = False
VERBOSE = False
import random
ENGINE = RayEngine(rml_basefile=RML_BASEFILE,
          exported_planes=[EXPORTED_PLANE],
          ray_backend=RayBackendDockerRAYUI(docker_image='ray-ui-service',
                                            dockerfile_path='../../ray_docker/rayui',
                                            docker_container_name=STUDY_NAME,
                                            ray_workdir=os.path.join(RAY_WORKDIR, STUDY_NAME+str(random.random())),
                                            verbose=VERBOSE),
          num_workers=-2,
          as_generator=False)
