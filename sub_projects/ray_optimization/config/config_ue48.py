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
RML_BASEFILE = os.path.join(ROOT_DIR, 'rml_src', 'UE48-PGM-CAT.rml')
ADDITIONAL_MOUNT_FILES = [os.path.join(ROOT_DIR, 'rml_src', 'wave_linhor_3_Harm_600eV.ray')]
RAY_WORKDIR = '/dev/shm'

# objective

## real data
REAL_DATA_DIR = None #'../../datasets/metrix_real_data/2021_march_complete'
#REAL_DATA_TRAIN_SET = ['M03', 'M10', 'M18', 'M22', 'M23', 'M24', 'M25', 'M27', 'M28', 'M29', 'M30', 'M32', 'M33', 'M36',
#                       'M37', 'M40', 'M41', 'M42', 'M43', 'M44']
#REAL_DATA_VALIDATION_SET = ['M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M11', 'M12', 'M13', 'M14', 'M15', 'M16', 'M17',
#                            'M18', 'M19', 'M20', 'M21', 'M26', 'M31', 'M34', 'M35', 'M38', 'M39']
EXPORTED_PLANE = "ImagePlane"
MAX_DEVIATION = 0.3
N_RAYS = ['1e4']
Z_LAYERS = [-15, -10, -5, 0, 5, 10, 15, 20, 25, 30]
TRANSFORMS = MultiLayer(Z_LAYERS, copy_directions=False)
NUM_BEAMLINE_PARAM_SAMPLES = 22
RG = RandomGenerator(seed=42)
PARAM_FUNC = lambda: RayParameterContainer([
    ("Undulator.numberRays", NumericalParameter(value=1e4)),
    ("Toroid.rotationXerror", RandomParameter(value_lims=(-0.015, 0.015), rg=RG)),
    ("Cylinder.rotationXerror", RandomParameter(value_lims=(-0.001, 0.001), rg=RG)),
    ("Cylinder.translationYerror", RandomParameter(value_lims=(-0.001, 0.001), rg=RG)),
    ("ImagePlane.translationXerror", RandomOutputParameter(value_lims=(-3.33, 3.33), rg=RG)),
    ("ImagePlane.translationYerror", RandomOutputParameter(value_lims=(-3.33, 3.33), rg=RG)),
    ("ImagePlane.translationZerror", RandomOutputParameter(value_lims=(-3.33, 3.33), rg=RG)),
])
FIXED_PARAMS = []  # [k for k in PARAM_FUNC().keys()][1:-3]
OVERWRITE_OFFSET = lambda: RayParameterContainer([
    ("Toroid.rotationXerror", RandomParameter(value_lims=(-0.15, 0.15), rg=RG)),
    ("Cylinder.rotationXerror", RandomParameter(value_lims=(-40.0, 40.0), rg=RG)),
    ("Cylinder.translationYerror", RandomParameter(value_lims=(-0.4, 0.4), rg=RG)),
])

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
    [str(len(PARAM_FUNC()) - len(FIXED_PARAMS)), 'real' if REAL_DATA_DIR is not None else 'sim', str(MAX_DEVIATION),
     OPTIMIZER, 'v333'])
WANDB_ENTITY = 'hzb-aos'
WANDB_PROJECT = 'emil_offsets'
OPTUNA_STORAGE_PATH = "sqlite:////dev/shm/db.sqlite2"
PLOT_INTERVAL = 10
LOGGING = False
VERBOSE = False

ENGINE = RayEngine(rml_basefile=RML_BASEFILE,
          exported_planes=[EXPORTED_PLANE],
          ray_backend=RayBackendDockerRAYUI(docker_image='ray-ui-service',
                                            dockerfile_path='../../ray_docker/rayui',
                                            docker_container_name=STUDY_NAME,
                                            ray_workdir=os.path.join(RAY_WORKDIR, STUDY_NAME),
                                            verbose=VERBOSE, additional_mount_files=ADDITIONAL_MOUNT_FILES),
          num_workers=-2,
          as_generator=False)
