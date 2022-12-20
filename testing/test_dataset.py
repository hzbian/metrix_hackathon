import sys

sys.path.insert(0, '../')

from ray_tools.simulation.torch_data_tools import RandomDatasetGenerator, build_param_container_sampler
from ray_tools.base.parameter_builder import build_parameter_grid
from ray_tools.base.backend import RayBackendDockerRAYX
from ray_tools.base.parameter import RandomParameter, GridParameter, RayParameterContainer
from ray_tools.base.transform import Crop, RayTransformCompose, ToDict, Histogram

param_container_func = lambda: RayParameterContainer([
    ('U41_318eV.numberRays', GridParameter(value=[1e2, 1e3, 1e4])),
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

param_container_sampler = build_param_container_sampler(
    param_container_func=lambda: build_parameter_grid(param_container_func()),
    idx_sub=['1e2', '1e3', '1e4'],
    transform=3 * [RayTransformCompose(Histogram(n_bins=256, x_lims=(-1.0, 1.0), y_lims=(-1.0, 1.0)),
                                       # ToDict(),
                                       Crop(x_lims=(-1.0, 1.0), y_lims=(-1.0, 1.0)))]
)

generator = RandomDatasetGenerator(rml_basefile='../rml_src/METRIX_U41_G1_H1_318eV_PS_MLearn.rml',
                                   ray_workdir='../ray_workdir',
                                   ray_backend=RayBackendDockerRAYX(docker_image='ray-service',
                                                                    ray_workdir='../ray_workdir',
                                                                    verbose=True),
                                   num_workers=50,
                                   param_container_sampler=param_container_sampler,
                                   h5_datadir='../datasets/metrix_simulation',
                                   h5_basename='data_raw',
                                   h5_max_size=1000)

generator.generate(h5_idx=0)
