import os
import sys

sys.path.insert(0, '../../')

from ray_tools.base.engine import RayEngine
from ray_tools.simulation.torch_data_tools import RandomRayDatasetGenerator
from ray_tools.base.parameter_builder import build_parameter_grid
from ray_tools.base.backend import RayBackendDockerRAYUI
from ray_tools.base.parameter import RandomParameter, GridParameter, RayParameterContainer
from ray_tools.base.transform import Histogram, RayTransformConcat, ToDict

# -----------------

# TODO: Better paths
# in __init__.py: PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# TODO: separate config file

DATASET_NAME = 'ray_enhance'
H5_MAX_SIZE = 10000
H5_IDX_RANGE = range(100)
H5_DATADIR = f'./{DATASET_NAME}'

RML_BASEFILE = '../../rml_src/METRIX_U41_G1_H1_318eV_PS_MLearn.rml'
RAY_WORKDIR = f'../../ray_workdir/{DATASET_NAME}'

N_RAYS = ['1e4', '1e6']

EXPORTED_PLANES = ["ImagePlane"]

TRANSFORMS = [
    RayTransformConcat({
        'hist': Histogram(n_bins=1024),
        'raw': ToDict(),
    }),
    Histogram(n_bins=1024)
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
    ('M1_Cylinder.translationXerror', RandomParameter(value_lims=(-0.15, 0.15))),
    ('M1_Cylinder.translationYerror', RandomParameter(value_lims=(-1., 1.))),
    ('SphericalGrating.radius', RandomParameter(value_lims=(109741., 109841.))),
    ('SphericalGrating.rotationYerror', RandomParameter(value_lims=(-1., 1.))),
    ('SphericalGrating.rotationZerror', RandomParameter(value_lims=(-2.5, 2.5))),
    ('ExitSlit.totalHeight', RandomParameter(value_lims=(0.009, 0.011))),
    ('ExitSlit.translationZerror', RandomParameter(value_lims=(-29., 31.))),
    ('ExitSlit.rotationZerror', RandomParameter(value_lims=(-0.3, 0.3))),
    ('E1.longHalfAxisA', RandomParameter(value_lims=(20600., 20900.))),
    ('E1.shortHalfAxisB', RandomParameter(value_lims=(300.721702601, 304.721702601))),
    ('E1.rotationXerror', RandomParameter(value_lims=(-0.5, 0.5))),
    ('E1.rotationYerror', RandomParameter(value_lims=(-7.5, 7.5))),
    ('E1.rotationZerror', RandomParameter(value_lims=(-4, 4))),
    ('E1.translationYerror', RandomParameter(value_lims=(-1, 1))),
    ('E1.translationZerror', RandomParameter(value_lims=(-1, 1))),
    ('E2.longHalfAxisA', RandomParameter(value_lims=(4325., 4425.))),
    ('E2.shortHalfAxisB', RandomParameter(value_lims=(96.1560870104, 98.1560870104))),
    ('E2.rotationXerror', RandomParameter(value_lims=(-0.5, 0.5))),
    ('E2.rotationYerror', RandomParameter(value_lims=(-7.5, 7.5))),
    ('E2.rotationZerror', RandomParameter(value_lims=(-4, 4))),
    ('E2.translationYerror', RandomParameter(value_lims=(-1, 1))),
    ('E2.translationZerror', RandomParameter(value_lims=(-1, 1))),
])

# -----------------

os.makedirs(H5_DATADIR, exist_ok=True)

param_container_sampler = RandomRayDatasetGenerator.build_param_container_sampler(
    param_container_func=lambda: build_parameter_grid(PARAM_CONTAINER_FUNC()),
    idx_sub=N_RAYS,
    transform=[{exported_plane: transform for exported_plane in EXPORTED_PLANES} for transform in TRANSFORMS]
)

generator = RandomRayDatasetGenerator(
    ray_engine=RayEngine(rml_basefile=RML_BASEFILE,
                         exported_planes=EXPORTED_PLANES,
                         ray_backend=RayBackendDockerRAYUI(docker_image='ray-ui-service',
                                                           ray_workdir=RAY_WORKDIR,
                                                           verbose=True),
                         num_workers=-1,
                         as_generator=False),
    param_container_sampler=param_container_sampler,
    h5_datadir=H5_DATADIR,
    h5_basename='data_raw',
    h5_max_size=H5_MAX_SIZE)

for h5_idx in H5_IDX_RANGE:
    generator.generate(h5_idx=h5_idx, batch_size=-1)

generator.ray_engine.ray_backend.kill()
