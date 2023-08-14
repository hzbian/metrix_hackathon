import sys
import os

import numpy as np
import torch.nn
from matplotlib import pyplot as plt
from tqdm import tqdm

from sub_projects.ray_optimization.losses import RayLoss, SinkhornLoss, RayCountMSE, ScheduledLoss, HistogramMSE, \
    CovMSE, KLDLoss, MSELoss, JSLoss, SSIMHistogramLoss

sys.path.insert(0, '../../')

from ray_tools.base.engine import GaussEngine

from ray_tools.base.parameter import RayParameterContainer, NumericalParameter, RandomParameter, RayParameter
from ray_tools.base.utils import RandomGenerator

from ray_optim.ray_optimizer import RayOptimizer

from ray_tools.base.transform import MultiLayer
from sub_projects.ray_optimization.utils import ray_output_to_tensor


def to_tensor(a):
    return ray_output_to_tensor(ray_output=a, exported_plane='ImagePlane')


def evaluate_single_var(var_name: str, value: float, loss_function):
    params_list, offset_list = create_params_offset_list(var_name=var_name, num_samples=2, value_lims=(value, value))
    engine = GaussEngine()
    outputs_list = [engine.run(params_entry, transforms=MultiLayer([0, 10])) for params_entry in params_list]
    distance = loss_function(outputs_list[0], outputs_list[1])
    RayOptimizer.fixed_position_plot_base([to_tensor(list_entry) for list_entry in outputs_list],
                                          xlim=[-2, 2], ylim=[-2, 2],
                                          ylabel=['reference'] + ["{:10.2f}".format(distance.item())])
    plt.plot()
    return distance.item()


def create_params_offset_list(var_name: str, value_lims, num_samples: int = 1):
    RG = RandomGenerator(seed=42)
    PARAM_FUNC = lambda: RayParameterContainer([
        ("number_rays", NumericalParameter(value=1e2)),
        ("x_dir", NumericalParameter(value=0.)),
        ("y_dir", NumericalParameter(value=0.)),
        ("z_dir", NumericalParameter(value=1.)),
        ("correlation_factor", NumericalParameter(value=0.)),
        ("direction_spread", NumericalParameter(value=0.)),
        ("x_mean", NumericalParameter(value=0)),
        ("y_mean", NumericalParameter(value=0)),
        ("x_var", NumericalParameter(value=0.005)),
        ("y_var", NumericalParameter(value=0.005)),
    ])
    params = [PARAM_FUNC() for _ in range(3)]
    offset = lambda: RayParameterContainer([(var_name, RandomParameter(value_lims=value_lims, rg=RG))])
    offset_list = []
    params_list = []
    for i in range(num_samples+1):
        perturbed_parameters: list[RayParameterContainer[str, RayParameter]] = [v.clone() for v in params]
        offset_instance = offset()
        if i != 0:
            for configuration in perturbed_parameters:
                configuration.perturb(offset_instance)
        params_list.append(perturbed_parameters)
        offset_list.append(offset_instance[var_name].get_value())
    return params_list, offset_list


def investigate_var(var_name: str, value_lims, loss: RayLoss, loss_string):
    engine = GaussEngine()
    num_samples = 300
    params_list, offset_list = create_params_offset_list(var_name, value_lims, num_samples=num_samples)
    outputs_list = [engine.run(params_entry, transforms=MultiLayer([0, 10])) for params_entry in params_list]

    distances_list = []
    for i in range(num_samples):
        distance = torch.stack(
            [loss.loss_fn(outputs_list[0][j], outputs_list[i+1][j], exported_plane="ImagePlane") for j in
             range(len(outputs_list[0]))]).mean()
        if isinstance(loss, ScheduledLoss):
            loss.end_epoch()
        distances_list.append(distance.item())
    plt.scatter(np.array(offset_list[1:]), np.array(distances_list[:]), s=0.2)
    plt.xlabel('Absolute error')
    plt.ylabel(loss_string + ' distance')
    plt.tight_layout()
    output_directory = 'plots/'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    plt.savefig(os.path.join(output_directory, loss_string + '_' + var_name + '.png'))
    plt.clf()


if __name__ == '__main__':
    # investigate_loss = {"generalized_box_iou_loss": BoxIoULoss(torchvision.ops.generalized_box_iou_loss,
    # reduction='mean')} investigate_loss = {"sinkhorn_loss": SinkhornLoss()} investigate_loss = {"raycount_mse":
    # RayCountMSE()}
    # investigate_loss = {"scheduled_loss": ScheduledLoss(RayCountMSE(), SinkhornLoss(), 150)}
    # investigate_loss = {"histogram_mse_10": HistogramMSE(n_bins=10)}

    # investigate_loss = {"cov_mse": CovMSE()}

    investigate_loss = {"ssim_histogram_loss": SSIMHistogramLoss(100)}
    #investigate_loss = {"js_loss": JSLoss()}
    for loss_string, loss in tqdm(investigate_loss.items()):
        investigate_var('y_mean', value_lims=(0.0, 1.0), loss=loss, loss_string=loss_string)
        if isinstance(loss, ScheduledLoss):
            loss.reset_passed_epochs()
        investigate_var('y_var', value_lims=(0.0, 1.0), loss=loss, loss_string=loss_string)

