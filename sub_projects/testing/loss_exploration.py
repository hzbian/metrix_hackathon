import sys

from matplotlib import pyplot as plt

import numpy as np
from sub_projects.ray_optimization.losses import sinkhorn_loss

sys.path.insert(0, '../../')
from ray_tools.base.engine import GaussEngine

from ray_tools.base.parameter import RayParameterContainer, NumericalParameter, RandomParameter, RayParameter
from ray_tools.base.utils import RandomGenerator

from ray_optim.ray_optimizer import RayOptimizer

from ray_tools.base.transform import MultiLayer

RG = RandomGenerator(seed=42)
PARAM_FUNC = lambda: RayParameterContainer([
    ("number_rays", NumericalParameter(value=1e2)),
    ("x_dir", NumericalParameter(value=0.)),
    ("y_dir", NumericalParameter(value=0.)),
    ("z_dir", NumericalParameter(value=1.)),
    ("direction_spread", NumericalParameter(value=0.)),
    ("x_mean", NumericalParameter(value=0)),
    ("y_mean", NumericalParameter(value=0)),
    ("x_var", NumericalParameter(value=0.005)),
    ("y_var", NumericalParameter(value=0.005)),
])
num_samples = 300
engine = GaussEngine()
params = [PARAM_FUNC() for i in range(3)]
offset = lambda: RayParameterContainer([("y_mean", RandomParameter(value_lims=(0.0, 100.5), rg=RG))])
offset_list = []
params_list = []
for i in range(num_samples):
    perturbed_parameters: list[RayParameterContainer[str, RayParameter]] = [v.clone() for v in params]
    offset_instance = offset()
    if i != 0:
        for configuration in perturbed_parameters:
            configuration.perturb(offset_instance)
    params_list.append(perturbed_parameters)
    offset_list.append(offset_instance['y_mean'].get_value())

outputs_list = [engine.run(params_entry, transforms=MultiLayer([0])) for params_entry in params_list]
outputs_list = [RayOptimizer.ray_output_to_tensor(ray_output=outputs_entry, exported_plane='ImagePlane') for
                outputs_entry in outputs_list]

distances_list = []
for i in range(num_samples):
    distance = sinkhorn_loss(outputs_list[0][0], outputs_list[i][0])
    distances_list.append(distance.item())
fig = RayOptimizer.fixed_position_plot(outputs_list[0], outputs_list[1], outputs_list[2], xlim=[-2, 2], ylim=[-2, 2])
plt.plot()
plt.savefig('out.png')

plt.clf()
fig2 = plt.scatter(np.array(offset_list[1:]), np.array(distances_list[1:]),s=0.2)
plt.plot()
plt.savefig('scatter.png')

plt.clf()
plt.plot(np.array(offset_list[1:]) / np.array(distances_list[1:]))
plt.plot()
plt.savefig('plot.png')
