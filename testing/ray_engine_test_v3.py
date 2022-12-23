import sys

sys.path.insert(0, '../')

import numpy as np

import matplotlib.pyplot as plt

from ray_tools.base.engine import RayEngine
from ray_tools.base.backend import RayBackendDockerRAYUI
from ray_tools.base.parameter import NumericalParameter, RandomParameter, RayParameterContainer
from ray_tools.base.transform import Histogram, RayTransformConcat, RayTransformDummy
from ray_tools.base.utils import RandomGenerator

n_rays = 1e5

exported_planes = ["U41_318eV",
                   "ASBL",
                   "M1-Cylinder",
                   "Spherical Grating",
                   "Exit Slit",
                   "E1",
                   "E2",
                   "ImagePlane"]

engine = RayEngine(rml_basefile='../rml_src/METRIX_U41_G1_H1_318eV_PS_MLearn.rml',
                   exported_planes=exported_planes,
                   ray_backend=RayBackendDockerRAYUI(docker_image='ray-ui-service',
                                                     docker_container_name='ray-ui-service-test',
                                                     ray_workdir='../ray_workdir',
                                                     verbose=True),
                   num_workers=-1,
                   as_generator=False)

rg = RandomGenerator(seed=42)

param_func = lambda: RayParameterContainer([
    (engine.template.U41_318eV.numberRays, NumericalParameter(value=n_rays)),
    (engine.template.U41_318eV.translationXerror, RandomParameter(value_lims=(-0.25, 0.25), rg=rg)),
    (engine.template.U41_318eV.translationYerror, RandomParameter(value_lims=(-0.25, 0.25), rg=rg)),
    (engine.template.U41_318eV.rotationXerror, RandomParameter(value_lims=(-0.05, 0.05), rg=rg)),
    (engine.template.U41_318eV.rotationYerror, RandomParameter(value_lims=(-0.05, 0.05), rg=rg)),
    (engine.template.ASBL.totalWidth, RandomParameter(value_lims=(1.9, 2.1), rg=rg)),
    (engine.template.ASBL.totalHeight, RandomParameter(value_lims=(0.9, 1.1), rg=rg)),
    (engine.template.ASBL.translationXerror, RandomParameter(value_lims=(-0.2, 0.2), rg=rg)),
    (engine.template.ASBL.translationYerror, RandomParameter(value_lims=(-0.2, 0.2), rg=rg)),
    (engine.template.M1_Cylinder.radius, RandomParameter(value_lims=(174.06, 174.36), rg=rg)),
    (engine.template.M1_Cylinder.rotationXerror, RandomParameter(value_lims=(-0.25, 0.25), rg=rg)),
    (engine.template.M1_Cylinder.rotationYerror, RandomParameter(value_lims=(-1., 1.), rg=rg)),
    (engine.template.M1_Cylinder.rotationZerror, RandomParameter(value_lims=(-1., 1.), rg=rg)),
    (engine.template.M1_Cylinder.translationXerror, RandomParameter(value_lims=(-0.15, 0.15), rg=rg)),
    (engine.template.M1_Cylinder.translationYerror, RandomParameter(value_lims=(-1., 1.), rg=rg)),
    (engine.template.SphericalGrating.radius, RandomParameter(value_lims=(109741., 109841.), rg=rg)),
    (engine.template.SphericalGrating.rotationYerror, RandomParameter(value_lims=(-1., 1.), rg=rg)),
    (engine.template.SphericalGrating.rotationZerror, RandomParameter(value_lims=(-2.5, 2.5), rg=rg)),
    (engine.template.ExitSlit.totalHeight, RandomParameter(value_lims=(0.009, 0.011), rg=rg)),
    (engine.template.ExitSlit.translationZerror, RandomParameter(value_lims=(-29., 31.), rg=rg)),
    (engine.template.ExitSlit.rotationZerror, RandomParameter(value_lims=(-0.3, 0.3), rg=rg)),
    (engine.template.E1.longHalfAxisA, RandomParameter(value_lims=(20600., 20900.), rg=rg)),
    (engine.template.E1.shortHalfAxisB, RandomParameter(value_lims=(300.721702601, 304.721702601), rg=rg)),
    (engine.template.E1.rotationXerror, RandomParameter(value_lims=(-0.5, 0.5), rg=rg)),
    (engine.template.E1.rotationYerror, RandomParameter(value_lims=(-7.5, 7.5), rg=rg)),
    (engine.template.E1.rotationZerror, RandomParameter(value_lims=(-4, 4), rg=rg)),
    (engine.template.E1.translationYerror, RandomParameter(value_lims=(-1, 1), rg=rg)),
    (engine.template.E1.translationZerror, RandomParameter(value_lims=(-1, 1), rg=rg)),
    (engine.template.E2.longHalfAxisA, RandomParameter(value_lims=(4325., 4425.), rg=rg)),
    (engine.template.E2.shortHalfAxisB, RandomParameter(value_lims=(96.1560870104, 98.1560870104), rg=rg)),
    (engine.template.E2.rotationXerror, RandomParameter(value_lims=(-0.5, 0.5), rg=rg)),
    (engine.template.E2.rotationYerror, RandomParameter(value_lims=(-7.5, 7.5), rg=rg)),
    (engine.template.E2.rotationZerror, RandomParameter(value_lims=(-4, 4), rg=rg)),
    (engine.template.E2.translationYerror, RandomParameter(value_lims=(-1, 1), rg=rg)),
    (engine.template.E2.translationZerror, RandomParameter(value_lims=(-1, 1), rg=rg)),
])

n_examples = 20
transform = RayTransformConcat({
    'hist': Histogram(n_bins=256, x_lims=(-.25, .25), y_lims=(-.25, .25), auto_center=True),
    'hist2': Histogram(n_bins=1024),
    'raw': RayTransformDummy(),
})

result = engine.run(param_containers=[param_func() for _ in range(n_examples)],
                    transforms={exported_plane: transform for exported_plane in exported_planes})

show_examples = [7]  # range(n_examples)

for idx in show_examples:
    for exported_plane in exported_planes:
        plt.figure()
        plt.title(exported_plane + ' ' + str(idx))
        plt.scatter(result[idx]['ray_output'][exported_plane]['raw'].x_loc,
                    result[idx]['ray_output'][exported_plane]['raw'].y_loc,
                    s=0.01)
        plt.show()

        plt.figure(figsize=(10, 10))
        plt.title(exported_plane + ' ' + str(idx))
        plt.imshow(np.fliplr(result[idx]['ray_output'][exported_plane]['hist']['histogram'].T),
                   cmap='Greys')
        print(result[idx]['ray_output'][exported_plane]['hist']['n_rays'],
              result[idx]['ray_output'][exported_plane]['hist']['x_lims'],
              result[idx]['ray_output'][exported_plane]['hist']['y_lims'])
        plt.show()

engine.ray_backend.kill()
