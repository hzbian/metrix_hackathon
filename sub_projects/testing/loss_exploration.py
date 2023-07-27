import sys

import torch.nn
from matplotlib import pyplot as plt

import numpy as np
from tqdm import trange

from ray_tools.base import RayOutput

sys.path.insert(0, '../../')
from sub_projects.ray_optimization.losses import sinkhorn_loss

from ray_tools.base.engine import GaussEngine

from ray_tools.base.parameter import RayParameterContainer, NumericalParameter, RandomParameter, RayParameter
from ray_tools.base.utils import RandomGenerator

from ray_optim.ray_optimizer import RayOptimizer

from ray_tools.base.transform import MultiLayer, Histogram


def histogram_mse(a, b):
    a = a[0]['ray_output']['ImagePlane']['0']
    b = b[0]['ray_output']['ImagePlane']['0']
    x_min = min(a.x_loc.min(), b.x_loc.min())
    x_max = max(a.x_loc.max(), b.x_loc.max())
    y_min = min(a.y_loc.min(), b.y_loc.min())
    y_max = max(a.y_loc.max(), b.y_loc.max())
    hist_a = Histogram(100, (x_min, x_max), (y_min, y_max))(a)['histogram']
    hist_b = Histogram(100, (x_min, x_max), (y_min, y_max))(b)['histogram']
    # plt.imshow(hist_a)
    # plt.savefig('a.png')
    # plt.imshow(hist_b)
    # plt.savefig('b.png')
    # plt.imshow((hist_a - hist_b)**2)
    # plt.savefig('a-b.png')
    return ((hist_a - hist_b) ** 2).mean()


def sinkhorn_output_loss(a, b):
    a = RayOptimizer.ray_output_to_tensor(ray_output=a, exported_plane='ImagePlane')
    b = RayOptimizer.ray_output_to_tensor(ray_output=b, exported_plane='ImagePlane')
    return sinkhorn_loss(a[0], b[0])


kld = torch.nn.KLDivLoss(log_target=True)

def kld_loss(a, b):
    a = RayOptimizer.ray_output_to_tensor(ray_output=a, exported_plane='ImagePlane')
    b = RayOptimizer.ray_output_to_tensor(ray_output=b, exported_plane='ImagePlane')
    #print(a.shape, b.shape)
    return kld(a[0], b[0])

def js_loss(a, b):
    a = RayOptimizer.ray_output_to_tensor(ray_output=a, exported_plane='ImagePlane')[0]
    b = RayOptimizer.ray_output_to_tensor(ray_output=b, exported_plane='ImagePlane')[0]
    M = 0.5 * (a + b)
    return 0.5 * (kld(a, M) + kld(b, M))

#                outputs_entry in outputs_list
def investigate_var(var_name: str, value_lims, loss_function, loss_string):
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
    params = [PARAM_FUNC() for _ in range(3)]
    offset = lambda: RayParameterContainer([(var_name, RandomParameter(value_lims=value_lims, rg=RG))])
    offset_list = []
    params_list = []
    for i in range(num_samples):
        perturbed_parameters: list[RayParameterContainer[str, RayParameter]] = [v.clone() for v in params]
        offset_instance = offset()
        if i != 0:
            for configuration in perturbed_parameters:
                configuration.perturb(offset_instance)
        params_list.append(perturbed_parameters)
        offset_list.append(offset_instance[var_name].get_value())

    outputs_list = [engine.run(params_entry, transforms=MultiLayer([0])) for params_entry in params_list]
    # outputs_list = [RayOptimizer.ray_output_to_tensor(ray_output=outputs_entry, exported_plane='ImagePlane') for
    #                outputs_entry in outputs_list]

    distances_list = []
    for i in trange(1,num_samples):
        # distance = sinkhorn_loss(outputs_list[0][0], outputs_list[i][0]).mean()
        distance = loss_function(outputs_list[0], outputs_list[i])
        print(i, distance)
        # distance = histogram_mse(outputs_list[0], outputs_list[i])
        distances_list.append(distance.item())
    # RayOptimizer.fixed_position_plot(outputs_list[0], outputs_list[1], outputs_list[2], xlim=[-2, 2],
    #                                 ylim=[-2, 2])
    # plt.plot()
    # plt.savefig('out_' + var_name + '.png')

    plt.clf()
    plt.scatter(np.array(offset_list[1:]), np.array(distances_list[:]), s=0.2)
    #plt.xlim = ([0, 2])
    #plt.ylim([0, 2])
    plt.xlabel('Absolute error')
    plt.ylabel(loss_string+' distance')
    plt.tight_layout()
    plt.savefig(loss_string+'_' + var_name + '.png')
    plt.clf()


investigate_var('y_mean', value_lims=(0.0, 2), loss_function=kld_loss, loss_string="kld")
investigate_var('y_var', value_lims=(0.0, 2), loss_function=kld_loss, loss_string="kld")
