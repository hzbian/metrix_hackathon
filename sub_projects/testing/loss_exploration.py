import sys

from matplotlib import pyplot as plt

sys.path.insert(0, '../../')
from ray_tools.base.engine import GaussEngine

from ray_tools.base.parameter import RayParameterContainer, NumericalParameter, RandomParameter
from ray_tools.base.utils import RandomGenerator

from ray_optim.ray_optimizer import RayOptimizer

from ray_tools.base.transform import MultiLayer

RG = RandomGenerator(seed=42)
PARAM_FUNC = lambda: RayParameterContainer([
    ("number_rays", NumericalParameter(value=1e4)),
    ("x_dir", NumericalParameter(value=1.)),
    ("y_dir", NumericalParameter(value=1.)),
    ("z_dir", NumericalParameter(value=1.)),
    ("direction_spread", RandomParameter(value_lims=(0., 0.), rg=RG)),
    ("x_mean", RandomParameter(value_lims=(-1.5, 1.5), rg=RG)),
    ("y_mean", RandomParameter(value_lims=(-1.5, 1.5), rg=RG)),
    ("x_var", RandomParameter(value_lims=(0.001, 0.01), rg=RG)),
    ("y_var", RandomParameter(value_lims=(0.001, 0.01), rg=RG)),
])

engine = GaussEngine()
ray_outputs = engine.run(PARAM_FUNC(), transforms=MultiLayer([0, 10, 20]))
ray_outputs = RayOptimizer.ray_output_to_tensor(ray_output=ray_outputs, exported_plane='ImagePlane')
fig = RayOptimizer.fixed_position_plot(ray_outputs, ray_outputs, ray_outputs, xlim=[-2, 2], ylim=[-2, 2])
plt.plot()
plt.savefig('out.png')
