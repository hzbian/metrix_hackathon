import sys
import os

import numpy as np
import torch.nn
import torchvision
from matplotlib import pyplot as plt
from tqdm import tqdm

sys.path.insert(0, '../../')
from ray_optim.plot import Plot
from sub_projects.ray_optimization.losses.losses import RayLoss, RayCountMSE, ScheduledLoss
from sub_projects.ray_optimization.losses.geometric import SinkhornLoss
from sub_projects.ray_optimization.losses.torch import HistogramMSE, CovMSE, KLDLoss, MSELoss, JSLoss, BoxIoULoss, MeanMSELoss, VarMSELoss

from ray_tools.base.engine import GaussEngine

from ray_tools.base.parameter import RayParameterContainer, NumericalParameter, RandomParameter, RayParameter
from ray_tools.base.utils import RandomGenerator

from ray_tools.base.transform import MultiLayer
from sub_projects.ray_optimization.utils import ray_output_to_tensor


def to_tensor(a):
    return ray_output_to_tensor(ray_output=a, exported_plane='ImagePlane')


def evaluate_single_var(var_name: str, value: float, ray_loss: RayLoss):
    params_list, offset_list = create_params_offset_list(var_name=var_name, num_samples=1, value_lims=(value, value))
    engine = GaussEngine()
    outputs_list = [engine.run(params_entry, transforms=MultiLayer([0, 10])) for params_entry in params_list]
    distance = ray_loss.loss_fn(outputs_list[0][0], outputs_list[1][0], exported_plane='ImagePlane')
    Plot.fixed_position_plot_base([to_tensor(list_entry) for list_entry in outputs_list],
                                          xlim=(-2., 2.), ylim=(-2., 2.),
                                          ylabel=['reference'] + ["{:10.4f}".format(distance.item())],  covariance_ellipse=False)
    Plot.fixed_position_plot_base
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
        ("x_var", NumericalParameter(value=0.0001)),
        ("y_var", NumericalParameter(value=0.0001)),
    ])
    params = [PARAM_FUNC() for _ in range(3)]

    def offset():
        return RayParameterContainer([(var_name, RandomParameter(value_lims=value_lims, rg=RG))])

    offset_list = []
    params_list = []
    for i in range(num_samples + 1):
        perturbed_parameters: list[RayParameterContainer] = [v.clone() for v in params]
        offset_instance = offset()
        if i != 0:
            for configuration in perturbed_parameters:
                configuration.perturb(offset_instance)
        params_list.append(perturbed_parameters)
        offset_list.append(offset_instance[var_name].get_value())
    return params_list, offset_list


def investigate_var(var_name: str, value_lims, loss: RayLoss, loss_string):
    engine = GaussEngine()
    num_samples = 1000
    params_list, offset_list = create_params_offset_list(var_name, value_lims, num_samples=num_samples)
    outputs_list = [engine.run(params_entry, transforms=MultiLayer([0, 10])) for params_entry in params_list]

    distances_list = []
    for i in range(num_samples):
        distance = torch.stack(
            [loss.loss_fn(outputs_list[0][j], outputs_list[i + 1][j], exported_plane="ImagePlane") for j in
             range(len(outputs_list[0]))]).mean()
        if isinstance(loss, ScheduledLoss):
            loss.end_epoch()
        distances_list.append(distance.item())
    return offset_list, distances_list


def plot_investigated_var(offsets, distances, loss_string):
    plt.scatter(np.array(offsets), np.array(distances), s=0.5, alpha=0.7, label=loss_string)

def save_plot(var_name: str, output_directory: str='plots/'):
    ax = plt.gca()
    #ax.set_yscale('log')
    #ax.set_xscale('log')
    plt.xlabel('Absolute error [a.u.]')
    plt.ylabel('Loss distance [a.u.]')
    plt.tight_layout()
    plt.legend()

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    plt.savefig(os.path.join(output_directory, var_name + '.png'), dpi=300)
    plt.clf()

def investigate_and_plot_var(var_name, value_lims, loss, loss_string): 
    offsets, distances = investigate_var(var_name, value_lims=value_lims, loss=loss, loss_string=loss_string)
    plot_investigated_var(offsets=offsets[1:], distances=distances, loss_string=loss_string)

if __name__ == '__main__':
    iou_loss = {
        "complete_box_iou_loss": BoxIoULoss(torchvision.ops.complete_box_iou_loss, reduction='mean'),
        "distance_box_iou_loss": BoxIoULoss(torchvision.ops.distance_box_iou_loss, reduction='mean'),
        "generalized_box_iou_loss": BoxIoULoss(torchvision.ops.generalized_box_iou_loss, reduction='mean'),
    }
    sinkhorn_loss = {
        "Sinkhorn": SinkhornLoss(),
    }
    ray_count_loss = {
    #    "ray_count_mse": RayCountMSE(),
    }
    scheduled_loss = {
        "scheduled_loss": ScheduledLoss(RayCountMSE(), SinkhornLoss(), 150),
    }
    histogram_mse = {
        "histogram_mse_10": HistogramMSE(n_bins=10),
        "histogram_mse_100": HistogramMSE(n_bins=100),
    }
    cov_mse = {
        "cov_mse": CovMSE(),
    }
    torch_loss = {
    #    "js_loss": JSLoss(),
        "KLD": KLDLoss(),
        "MSE": MSELoss(),
        "Var-MSE": VarMSELoss(),
        "Mean-MSE": MeanMSELoss(),

    }
    investigate_loss = torch_loss | sinkhorn_loss #| iou_loss # | cov_mse | histogram_mse | ray_count_loss | sinkhorn_loss | iou_loss
    for lim in [2.0, 0.001]:
        for var_name in ['y_mean', 'y_var']:
            for loss_string, loss in tqdm(investigate_loss.items()):
                investigate_and_plot_var(var_name=var_name, value_lims=(0.0, lim), loss=loss, loss_string=loss_string)
                if isinstance(loss, ScheduledLoss):
                    loss.reset_passed_epochs()
            save_plot(var_name+str(lim))