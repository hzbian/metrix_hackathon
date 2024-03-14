import os

from definitions import ROOT_DIR
from ray_tools.base.parameter import RandomParameter, GridParameter, RayParameterContainer
from ray_tools.base.transform import Histogram, RayTransformConcat

DATASET_NAME = 'ray_emergency_surrogate'
H5_MAX_SIZE = 10000
BATCH_SIZE = 1000
H5_IDX_RANGE = range(100)
H5_DATADIR = os.path.join(ROOT_DIR, 'datasets', 'metrix_simulation', DATASET_NAME)

RML_BASEFILE = os.path.join(ROOT_DIR, 'rml_src', 'METRIX_U41_G1_H1_318eV_PS_MLearn.rml')
RAY_WORKDIR = '/dev/shm/ray_workdir' 

N_RAYS = ['1e5']

EXPORTED_PLANES = ["ImagePlane"]

TRANSFORMS = [
    RayTransformConcat({
        'hist': Histogram(n_bins=1024),
        'hist_small': Histogram(n_bins=256),
    })
]

PARAM_CONTAINER_FUNC = lambda: RayParameterContainer([
    ('U41_318eV.numberRays', GridParameter(value=[[float(n) for n in N_RAYS]])),
    ('U41_318eV.translationXerror', RandomParameter(value_lims=(-0.25, 0.25))),
    ('U41_318eV.translationYerror', RandomParameter(value_lims=(-0.25, 0.25))),
    ('U41_318eV.rotationXerror', RandomParameter(value_lims=(-0.05, 0.05))),
    ('U41_318eV.rotationYerror', RandomParameter(value_lims=(-0.05, 0.05))),
    ('ASBL.totalWidth', RandomParameter(value_lims=(1.9, 2.1))),
    ('ASBL.totalHeight', RandomParameter(value_lims=(0.9, 1.1))),
    ('ASBL.translationXerror', RandomParameter(value_lims=(-0.2, 0.2))),
    ('ASBL.translationYerror', RandomParameter(value_lims=(-0.2, 0.2))),
    ('M1_Cylinder.radius', RandomParameter(value_lims=(174.06, 174.36))),
    ('M1_Cylinder.rotationXerror', RandomParameter(value_lims=(-0.25, 0.25))),
    ('M1_Cylinder.rotationYerror', RandomParameter(value_lims=(-1., 1.))),
    ('M1_Cylinder.rotationZerror', RandomParameter(value_lims=(-1., 1.))),
    ('M1_Cylinder.translationXerror', RandomParameter(value_lims=(-1., 1.))),
    ('M1_Cylinder.translationYerror', RandomParameter(value_lims=(-1., 1.))),
    ('SphericalGrating.radius', RandomParameter(value_lims=(109741., 109841.))),
    ('SphericalGrating.rotationYerror', RandomParameter(value_lims=(-1., 1.))),
    ('SphericalGrating.rotationZerror', RandomParameter(value_lims=(-2.5, 2.5))),
    ('ExitSlit.totalHeight', RandomParameter(value_lims=(0.009, 0.011))),
    ('ExitSlit.translationZerror', RandomParameter(value_lims=(-150., 150.))),
    ('ExitSlit.rotationZerror', RandomParameter(value_lims=(-0.3, 0.3))),
    ('E1.longHalfAxisA', RandomParameter(value_lims=(20600., 20900.))),
    ('E1.shortHalfAxisB', RandomParameter(value_lims=(300.721702601, 304.721702601))),
    ('E1.rotationXerror', RandomParameter(value_lims=(-1.5, 1.5))),
    ('E1.rotationYerror', RandomParameter(value_lims=(-7.5, 7.5))),
    ('E1.rotationZerror', RandomParameter(value_lims=(7, 14))),
    ('E1.translationYerror', RandomParameter(value_lims=(-1, 1))),
    ('E1.translationZerror', RandomParameter(value_lims=(-1, 1))),
    ('E2.longHalfAxisA', RandomParameter(value_lims=(4325., 4425.))),
    ('E2.shortHalfAxisB', RandomParameter(value_lims=(96.1560870104, 98.1560870104))),
    ('E2.rotationXerror', RandomParameter(value_lims=(-0.5, 0.5))),
    ('E2.rotationYerror', RandomParameter(value_lims=(-7.5, 7.5))),
    ('E2.rotationZerror', RandomParameter(value_lims=(22.0, 32.0))),
    ('E2.translationYerror', RandomParameter(value_lims=(-1, 1))),
    ('E2.translationZerror', RandomParameter(value_lims=(-1, 1))),
])
