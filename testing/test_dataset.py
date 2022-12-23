import sys

sys.path.insert(0, '../')

from ray_tools.base.engine import RayEngine
from ray_tools.simulation.torch_data_tools import RandomRayDatasetGenerator
from ray_tools.base.parameter_builder import build_parameter_grid
from ray_tools.base.backend import RayBackendDockerRAYUI
from ray_tools.base.parameter import RandomParameter, GridParameter, RayParameterContainer
from ray_tools.base.transform import Histogram, RayTransformDummy, RayTransformConcat

param_container_func = lambda: RayParameterContainer([
    ('U41_318eV.numberRays', GridParameter(value=[1e5])),
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

default_transform = RayTransformConcat({
    # 'hist': Histogram(n_bins=256, x_lims=(-.25, .25), y_lims=(-.25, .25), auto_center=True),
    'hist': Histogram(n_bins=256),
    # 'raw': RayTransformDummy(),
})

exported_planes = ["U41_318eV",
                   "ASBL",
                   "M1-Cylinder",
                   "Spherical Grating",
                   "Exit Slit",
                   "E1",
                   "E2",
                   "ImagePlane"]
# "U41_318eV" "ASBL" "M1-Cylinder" "Spherical Grating" "Exit Slit" "E1" "E2" "ImagePlane"

param_container_sampler = RandomRayDatasetGenerator.build_param_container_sampler(
    param_container_func=lambda: build_parameter_grid(param_container_func()),
    idx_sub=['1e5'],
    transform=1 * [{exported_plane: default_transform for exported_plane in exported_planes}]
)

generator = RandomRayDatasetGenerator(
    ray_engine=RayEngine(rml_basefile='../rml_src/METRIX_U41_G1_H1_318eV_PS_MLearn.rml',
                         exported_planes=exported_planes,
                         ray_backend=RayBackendDockerRAYUI(docker_image='ray-ui-service',
                                                           ray_workdir='../ray_workdir',
                                                           verbose=True),
                         num_workers=-1,
                         as_generator=False),
    param_container_sampler=param_container_sampler,
    h5_datadir='../datasets/metrix_simulation',
    h5_basename='data_raw',
    h5_max_size=1000)

generator.generate(h5_idx=0, batch_size=-1)

generator.ray_engine.ray_backend.kill()
