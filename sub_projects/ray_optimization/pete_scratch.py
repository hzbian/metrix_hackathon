import random
from BBOptim.build.lib.bboptimizer.optimizer import Optimizer
from ray_tools.base.parameter import GridParameter
from ray_tools.base.engine import RayEngine as engine
import sys
import numpy as np
import matplotlib.pyplot as plt
from ray_tools.base.engine import RayEngine
from ray_tools.base.backend import RayBackendDockerRAYUI
from ray_tools.base.parameter import NumericalParameter, RandomParameter, RayParameterContainer
from ray_tools.base.transform import Histogram, RayTransformConcat, MultiLayer
from ray_tools.base.utils import RandomGenerator

sys.path.insert(0, '../../')
n_rays = 1e5
exported_planes = ["ImagePlane"]

engine = RayEngine(rml_basefile='../../rml_src/METRIX_U41_G1_H1_318eV_PS_MLearn.rml',
                   exported_planes=exported_planes,
                   ray_backend=RayBackendDockerRAYUI(docker_image='ray-ui-service',
                                                     docker_container_name='ray-ui-service-test',
                                                     ray_workdir='../../ray_workdir',
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

n_examples = 1
dist_layers = [-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30]
transform = RayTransformConcat({
    'ml': MultiLayer(dist_layers=dist_layers,
                     copy_directions=False,
                     transform=Histogram(n_bins=256)),
    'hist': Histogram(n_bins=1024),

})

result = engine.run(param_containers=[param_func() for _ in range(n_examples)],
                    transforms={exported_plane: transform for exported_plane in exported_planes})

show_examples = [7]  # range(n_examples)

for idx in show_examples:
    for dist in dist_layers:
        plt.figure(figsize=(10, 10))
        plt.title(str(dist) + ' ' + str(idx))
        plt.imshow(np.flipud(result[idx]['ray_output']['ImagePlane']['ml'][str(dist)]['histogram'].T),
                   cmap='Greys')
        plt.xlabel(str(result[idx]['ray_output']['ImagePlane']['ml'][str(dist)]['n_rays']) + ' ' +
                   str(result[idx]['ray_output']['ImagePlane']['ml'][str(dist)]['x_lims']) + ' ' +
                   str(result[idx]['ray_output']['ImagePlane']['ml'][str(dist)]['y_lims']))
        plt.show()

engine.ray_backend.kill()



def loss_func(x):
    #target =
    #sample =
    loss = 0
    return loss

def trace(params):
    ray_out = engine.run(params)
    return ray_out


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


N_RAYS = ['1e4', '1e6']
params_conf = [
    {"name": 'U41_318eV.numberRays', "domain": GridParameter(value=[[float(n) for n in N_RAYS]]), "type": "integer"},
    {"name": 'U41_318eV.translationXerror', "domain": RandomParameter(value_lims=(-0.25, 0.25)), "type": "continuous"},
    {"name": 'U41_318eV.translationYerror', "domain": RandomParameter(value_lims=(-0.25, 0.25)), "type": "continuous"},
    {"name": 'U41_318eV.rotationXerror', "domain": RandomParameter(value_lims=(-0.05, 0.05)), "type": "continuous"},
    {"name": 'U41_318eV.rotationYerror', "domain": RandomParameter(value_lims=(-0.05, 0.05)), "type": "continuous"},
    {"name": 'ASBL.totalWidth', "domain": RandomParameter(value_lims=(1.9, 2.1)), "type": "continuous"},
    {"name": 'ASBL.totalHeight', "domain": RandomParameter(value_lims=(0.9, 1.1)), "type": "continuous"},
    {"name": 'ASBL.translationXerror', "domain": RandomParameter(value_lims=(-0.2, 0.2)), "type": "continuous"},
    {"name": 'ASBL.translationYerror', "domain": RandomParameter(value_lims=(-0.2, 0.2)), "type": "continuous"},
    {"name": 'M1_Cylinder.radius', "domain": RandomParameter(value_lims=(174.06, 174.36)), "type": "continuous"},
    {"name": 'M1_Cylinder.rotationXerror', "domain": RandomParameter(value_lims=(-0.25, 0.25)), "type": "continuous"},
    {"name": 'M1_Cylinder.rotationYerror', "domain": RandomParameter(value_lims=(-1., 1.)), "type": "continuous"},
    {"name": 'M1_Cylinder.rotationZerror', "domain": RandomParameter(value_lims=(-1., 1.)), "type": "continuous"},
    {"name": 'M1_Cylinder.translationXerror', "domain": RandomParameter(value_lims=(-0.15, 0.15)), "type": "continuous"},
    {"name": 'M1_Cylinder.translationYerror', "domain": RandomParameter(value_lims=(-1., 1.)), "type": "continuous"},
    {"name": 'SphericalGrating.radius', "domain": RandomParameter(value_lims=(109741., 109841.)), "type": "continuous"},
    {"name": 'SphericalGrating.rotationYerror', "domain": RandomParameter(value_lims=(-1., 1.)), "type": "continuous"},
    {"name": 'SphericalGrating.rotationZerror', "domain": RandomParameter(value_lims=(-2.5, 2.5)), "type": "continuous"},
    {"name": 'ExitSlit.totalHeight', "domain": RandomParameter(value_lims=(0.009, 0.011)), "type": "continuous"},
    {"name": 'ExitSlit.translationZerror', "domain": RandomParameter(value_lims=(-29., 31.)), "type": "continuous"},
    {"name": 'ExitSlit.rotationZerror', "domain": RandomParameter(value_lims=(-0.3, 0.3)), "type": "continuous"},
    {"name": 'E1.longHalfAxisA', "domain": RandomParameter(value_lims=(20600., 20900.)), "type": "continuous"},
    {"name": 'E1.shortHalfAxisB', "domain": RandomParameter(value_lims=(300.721702601, 304.721702601)), "type": "continuous"},
    {"name": 'E1.rotationXerror', "domain": RandomParameter(value_lims=(-0.5, 0.5)), "type": "continuous"},
    {"name": 'E1.rotationYerror', "domain": RandomParameter(value_lims=(-7.5, 7.5)), "type": "continuous"},
    {"name": 'E1.rotationZerror', "domain": RandomParameter(value_lims=(-4, 4)), "type": "integer"},
    {"name": 'E1.translationYerror', "domain": RandomParameter(value_lims=(-1, 1)), "type": "integer"},
    {"name": 'E1.translationZerror', "domain": RandomParameter(value_lims=(-1, 1)), "type": "integer"},
    {"name": 'E2.longHalfAxisA', "domain": RandomParameter(value_lims=(4325., 4425.)), "type": "continuous"},
    {"name": 'E2.shortHalfAxisB', "domain": RandomParameter(value_lims=(96.1560870104, 98.1560870104)), "type": "continuous"},
    {"name": 'E2.rotationXerror', "domain": RandomParameter(value_lims=(-0.5, 0.5)), "type": "continuous"},
    {"name": 'E2.rotationYerror', "domain": RandomParameter(value_lims=(-7.5, 7.5)), "type": "continuous"},
    {"name": 'E2.rotationZerror', "domain": RandomParameter(value_lims=(-4, 4)), "type": "integer"},
    {"name": 'E2.translationYerror', "domain": RandomParameter(value_lims=(-1, 1)), "type": "integer"},
    {"name": 'E2.translationZerror', "domain": RandomParameter(value_lims=(-1, 1)), "type": "integer"},
    {"name": 'ImagePlane.distanceImagePlane', "domain": RandomParameter(value_lims=(990, 1010)), "type": "integer"},
]

# params_conf = [
#     {"name": "x1", "domain": (.1, 5), "type": "continuous",
#      "num_grid": 5, "scale": "log"},
#     {"name": "x2", "domain": (-5, 5), "type": "continuous",
#      "num_grid": 5, "dimensionality": 2},
#     {"name": "x4", "domain": ("linear", "sin", "square"),
#      "type": "categorical"},
# ]


if __name__ == "__main__":
    np.random.seed(0)
    random.seed(0)
    bayes_opt = Optimizer(loss_func, params_conf, sampler="bayes", r_min=10, maximize=False)
    print("****************************")
    print("bayes")
    print(bayes_opt.search(num_iter=50))
