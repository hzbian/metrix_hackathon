import os

from definitions import ROOT_DIR
from ray_tools.base.parameter import NumericalParameter, RandomParameter, RayParameterContainer
from ray_tools.base.transform import Histogram, RayTransformConcat, XYHistogram
from ray_tools.base.utils import RandomGenerator

DATASET_NAME = 'ray_emergency_surrogate'
H5_MAX_SIZE = 10000
BATCH_SIZE = 1000
H5_IDX_RANGE = range(10)
H5_DATADIR = os.path.join(ROOT_DIR, 'datasets', 'metrix_simulation', DATASET_NAME)

RML_BASEFILE = os.path.join(ROOT_DIR, 'rml_src', 'METRIX_U41_G1_H1_318eV_PS_MLearn_1.15.rml')
RAY_WORKDIR = '/dev/shm/ray_workdir'
task_id = os.environ['$SLURM_ARRAY_TASK_ID']
if task_id != '':
    task_id = int(task_id)
else:
    task_id = 0
SEED = 42 + task_id
RG = RandomGenerator(SEED)

N_RAYS = ['1e5']

EXPORTED_PLANES = ["ImagePlane"]

TRANSFORMS = [
        XYHistogram(50, (-10., 10.), (-3., 3.))
]

PARAM_CONTAINER_FUNC = lambda: RayParameterContainer([
    ('U41_318eV.numberRays', NumericalParameter(value=float(N_RAYS[0]))),
    ('U41_318eV.translationXerror', RandomParameter(value_lims=(-0.25, 0.25), rg=RG)),
    ('U41_318eV.translationYerror', RandomParameter(value_lims=(-0.25, 0.25))),
    ('U41_318eV.rotationXerror', RandomParameter(value_lims=(-0.05, 0.05), rg=RG)),
    ('U41_318eV.rotationYerror', RandomParameter(value_lims=(-0.05, 0.05), rg=RG)),
    ('ASBL.totalWidth', RandomParameter(value_lims=(1.9, 2.1), rg=RG)),
    ('ASBL.totalHeight', RandomParameter(value_lims=(0.9, 1.1), rg=RG)),
    ('ASBL.translationXerror', RandomParameter(value_lims=(-0.2, 0.2), rg=RG)),
    ('ASBL.translationYerror', RandomParameter(value_lims=(-0.2, 0.2), rg=RG)),
    ('M1_Cylinder.radius', RandomParameter(value_lims=(174.06, 174.36), rg=RG)),
    ('M1_Cylinder.rotationXerror', RandomParameter(value_lims=(-0.25, 0.25), rg=RG)),
    ('M1_Cylinder.rotationYerror', RandomParameter(value_lims=(-1., 1.), rg=RG)),
    ('M1_Cylinder.rotationZerror', RandomParameter(value_lims=(-1., 1.), rg=RG)),
    ('M1_Cylinder.translationXerror', RandomParameter(value_lims=(-1., 1.), rg=RG)),
    ('M1_Cylinder.translationYerror', RandomParameter(value_lims=(-1., 1.), rg=RG)),
    ('SphericalGrating.radius', RandomParameter(value_lims=(109741., 109841.), rg=RG)),
    ('SphericalGrating.rotationYerror', RandomParameter(value_lims=(-1., 1.), rg=RG)),
    ('SphericalGrating.rotationZerror', RandomParameter(value_lims=(-2.5, 2.5), rg=RG)),
    ('ExitSlit.totalHeight', RandomParameter(value_lims=(0.009, 0.011), rg=RG)),
    ('ExitSlit.translationZerror', RandomParameter(value_lims=(-150., 150.), rg=RG)),
    ('ExitSlit.rotationZerror', RandomParameter(value_lims=(-0.3, 0.3), rg=RG)),
    ('E1.longHalfAxisA', RandomParameter(value_lims=(20600., 20900.), rg=RG)),
    ('E1.shortHalfAxisB', RandomParameter(value_lims=(300.721702601, 304.721702601), rg=RG)),
    ('E1.rotationXerror', RandomParameter(value_lims=(-1.5, 1.5), rg=RG)),
    ('E1.rotationYerror', RandomParameter(value_lims=(-7.5, 7.5), rg=RG)),
    ('E1.rotationZerror', RandomParameter(value_lims=(7, 14), rg=RG)),
    ('E1.translationYerror', RandomParameter(value_lims=(-1, 1), rg=RG)),
    ('E1.translationZerror', RandomParameter(value_lims=(-1, 1), rg=RG)),
    ('E2.longHalfAxisA', RandomParameter(value_lims=(4325., 4425.), rg=RG)),
    ('E2.shortHalfAxisB', RandomParameter(value_lims=(96.1560870104, 98.1560870104), rg=RG)),
    ('E2.rotationXerror', RandomParameter(value_lims=(-0.5, 0.5), rg=RG)),
    ('E2.rotationYerror', RandomParameter(value_lims=(-7.5, 7.5), rg=RG)),
    ('E2.rotationZerror', RandomParameter(value_lims=(22.0, 32.0), rg=RG)),
    ('E2.translationYerror', RandomParameter(value_lims=(-1, 1), rg=RG)),
    ('E2.translationZerror', RandomParameter(value_lims=(-1, 1), rg=RG)),
])
