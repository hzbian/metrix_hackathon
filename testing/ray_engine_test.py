import sys

sys.path.insert(0, '../')

from ray_tools.base.parameter_builder import build_parameter_grid
from ray_tools.base.engine import RayEngine
from ray_tools.base.backend import RayBackendDockerRAYX
from ray_tools.base.parameter import NumericalParameter, RandomParameter, GridParameter, RayParameterContainer
from ray_tools.base.transform import Crop, Histogram, RayTransformCompose

engine = RayEngine(rml_basefile='../rml_src/METRIX_U41_G1_H1_318eV_PS_MLearn.rml',
                   ray_backend=RayBackendDockerRAYX(docker_image='ray-service',
                                                    ray_workdir='../ray_workdir',
                                                    verbose=True),
                   num_workers=-1,
                   as_generator=False)

param_func = lambda: RayParameterContainer([
    (engine.template.U41_318eV.numberRays, NumericalParameter(value=1e4)),
    (engine.template.U41_318eV.translationXerror, RandomParameter(value_lims=(-0.25, 0.25))),
    (engine.template.U41_318eV.translationYerror, RandomParameter(value_lims=(-0.25, 0.25))),
    (engine.template.U41_318eV.rotationXerror, RandomParameter(value_lims=(-0.05, 0.05))),
    (engine.template.U41_318eV.rotationYerror, RandomParameter(value_lims=(-0.05, 0.05))),
    (engine.template.ASBL.totalWidth, RandomParameter(value_lims=(1.9, 2.1))),
    (engine.template.ASBL.totalHeight, RandomParameter(value_lims=(0.9, 1.1))),
    (engine.template.ASBL.translationXerror, RandomParameter(value_lims=(-0.2, 0.2))),
    (engine.template.ASBL.translationYerror, RandomParameter(value_lims=(-0.2, 0.2))),
    (engine.template.M1_Cylinder.radius, RandomParameter(value_lims=(174.06, 174.36))),
    (engine.template.M1_Cylinder.rotationXerror, RandomParameter(value_lims=(-0.25, 0.25))),
    (engine.template.M1_Cylinder.rotationYerror, RandomParameter(value_lims=(-1., 1.))),
    (engine.template.M1_Cylinder.rotationZerror, RandomParameter(value_lims=(-1., 1.))),
    (engine.template.M1_Cylinder.translationXerror, RandomParameter(value_lims=(-0.15, 0.15))),
    (engine.template.M1_Cylinder.translationYerror, RandomParameter(value_lims=(-1., 1.))),
    (engine.template.SphericalGrating.radius, RandomParameter(value_lims=(109741., 109841.))),
    (engine.template.SphericalGrating.rotationYerror, RandomParameter(value_lims=(-1., 1.))),
    (engine.template.SphericalGrating.rotationZerror, RandomParameter(value_lims=(-2.5, 2.5))),
    (engine.template.ExitSlit.totalHeight, RandomParameter(value_lims=(0.009, 0.011))),
    (engine.template.ExitSlit.translationZerror, RandomParameter(value_lims=(-29., 31.))),
    (engine.template.ExitSlit.rotationZerror, RandomParameter(value_lims=(-0.3, 0.3))),
    (engine.template.E1.longHalfAxisA, RandomParameter(value_lims=(20600., 20900.))),
    (engine.template.E1.shortHalfAxisB, RandomParameter(value_lims=(300.721702601, 304.721702601))),
    (engine.template.E1.rotationXerror, RandomParameter(value_lims=(-0.5, 0.5))),
    (engine.template.E1.rotationYerror, RandomParameter(value_lims=(-7.5, 7.5))),
    (engine.template.E1.rotationZerror, RandomParameter(value_lims=(-4, 4))),
    (engine.template.E1.translationYerror, RandomParameter(value_lims=(-1, 1))),
    (engine.template.E1.translationZerror, RandomParameter(value_lims=(-1, 1))),
    (engine.template.E2.longHalfAxisA, RandomParameter(value_lims=(4325., 4425.))),
    (engine.template.E2.shortHalfAxisB, RandomParameter(value_lims=(96.1560870104, 98.1560870104))),
    (engine.template.E2.rotationXerror, RandomParameter(value_lims=(-0.5, 0.5))),
    (engine.template.E2.rotationYerror, RandomParameter(value_lims=(-7.5, 7.5))),
    (engine.template.E2.rotationZerror, RandomParameter(value_lims=(-4, 4))),
    (engine.template.E2.translationYerror, RandomParameter(value_lims=(-1, 1))),
    (engine.template.E2.translationZerror, RandomParameter(value_lims=(-1, 1))),
])

params = [param_func() for _ in range(10)]
result = engine.run(params, transforms=RayTransformCompose(  # Histogram(n_bins=256, lim=1.0),
    Crop(x_lims=(-1.0, 1.0), y_lims=(-1.0, 1.0))
))

import matplotlib.pyplot as plt

for idx in range(10):
    plt.figure()
    plt.scatter(result[idx]['ray_output'][0].y_loc, result[idx]['ray_output'][0].x_loc, s=0.001)
    plt.xlim((-1.0, 1.0))
    plt.ylim((-1.0, 1.0))
    plt.show()

    plt.figure()
    plt.imshow(
        Histogram(n_bins=256, x_lims=(-1.0, 1.0), y_lims=(-1.0, 1.0))(result[idx]['ray_output'][0])['histogram'])
    plt.show()

param_container = RayParameterContainer([
    (engine.template.U41_318eV.numberRays, NumericalParameter(value=1e4)),
    (engine.template.U41_318eV.translationXerror, GridParameter(value=[0.0, 1.0, 2.0])),
    (engine.template.U41_318eV.translationYerror, GridParameter(value=[3.0, 4.0, 5.0])),
    (engine.template.U41_318eV.rotationXerror, RandomParameter(value_lims=(-0.05, 0.05))),
    (engine.template.U41_318eV.rotationYerror, RandomParameter(value_lims=(-0.05, 0.05))),
])

params = build_parameter_grid(param_container)
result = engine.run(params)

engine.ray_backend.kill()
