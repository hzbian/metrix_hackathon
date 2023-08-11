import sys
from abc import ABC, abstractmethod
from typing import Union, Dict, List, Iterable, Callable

import numpy as np
import torch.nn
from matplotlib import pyplot as plt
from tqdm import trange

sys.path.insert(0, '../../')
from sub_projects.ray_optimization.losses import sinkhorn_loss

from ray_tools.base.engine import GaussEngine

from ray_tools.base.parameter import RayParameterContainer, NumericalParameter, RandomParameter, RayParameter
from ray_tools.base.utils import RandomGenerator

from ray_optim.ray_optimizer import RayOptimizer

from ray_tools.base.transform import MultiLayer, Histogram
import ignite


# ssim_fun = ignite.metrics.SSIM(1.0)

class RayLoss(ABC):
    """
    Base class for defining losses.
    """

    @abstractmethod
    def loss_fn(self, a: Union[Dict, List[Dict], Iterable[Dict]], b: Union[Dict, List[Dict], Iterable[Dict]],
                exported_plane: str) -> torch.Tensor:
        """
        Function that calculates the loss from input parameters a and b.
        :param a: Ray_output that should be compared.
        :param b: Ray_output that should be compared.
        :param exported_plane: The plane of ray_output that should be compared.
        :return: Calculated loss.
        """
        pass


class ScheduledLoss(RayLoss):
    """
    Schedules two losses. The first ``loss_a_epochs`` it uses ``loss_a``, after that ``loss_b`` is taken.
    :param loss_a: Loss being used the first ``loss_a_epochs``.
    :param loss_b: Loss being used after ``loss_a_epochs``.
    :param loss_a_epochs: The number of epochs, that ``loss_a`` is taken, after that, use ``loss_b``.
    """

    def __init__(self, loss_a: RayLoss, loss_b: RayLoss, loss_a_epochs: int):
        self.loss_a: RayLoss = loss_a
        self.loss_b: RayLoss = loss_b
        self.loss_a_epochs: int = loss_a_epochs
        self.passed_epochs: int = 0

    def loss_fn(self, a: Union[Dict, List[Dict], Iterable[Dict]], b: Union[Dict, List[Dict], Iterable[Dict]],
                exported_plane: str) -> torch.Tensor:
        if self.passed_epochs > self.loss_a_epochs:
            self.passed_epochs += 1
            return self.loss_a.loss_fn(a, b, exported_plane)
        else:
            return self.loss_b.loss_fn(a, b, exported_plane)


class BoxIoULoss(RayLoss):
    """
    Implementation of box intersection-over-union losses. This class is meant to be used with a Torchvision function
    as input as described in the `PyTorch documentation <https://pytorch.org/vision/master/ops.html#losses>`_
    :param base_fn can be one of `torchvision.ops.complete_box_iou`, ``torchvision.ops.distance_box_iou_loss`` or
    ``torchvision.ops.generalized_box_iou_loss``.
    """

    def __init__(self, base_fn: Callable[..., torch.Tensor]):
        self.base_fn: Callable[..., torch.Tensor] = base_fn

    def loss_fn(self, a: Union[Dict, List[Dict], Iterable[Dict]], b: Union[Dict, List[Dict], Iterable[Dict]],
                exported_plane: str) -> torch.Tensor:
        a_dict = a['ray_output'][exported_plane]
        b_dict = b['ray_output'][exported_plane]
        box_a_list = []
        box_b_list = []

        for key, a in a_dict.items():
            b = b_dict[key]
            global_x_min = min(a.x_loc.min(), b.x_loc.min())
            global_y_min = min(a.y_loc.min(), b.y_loc.min())
            shift_x = - global_x_min if global_x_min < -1 else 0
            shift_y = - global_y_min if global_y_min < -1 else 0
            box_a = torch.tensor(
                (shift_x + a.x_loc.min(), shift_y + a.y_loc.min(), shift_x + a.x_loc.max(),
                 shift_y + a.y_loc.max())).unsqueeze(
                -1)
            box_a_list.append(box_a)
            box_b = torch.tensor(
                (shift_x + b.x_loc.min(), shift_y + b.y_loc.min(), shift_x + b.x_loc.max(),
                 shift_y + b.y_loc.max())).unsqueeze(
                -1)
            box_b_list.append(box_b)

        return self.base_fn(torch.stack(box_a_list), torch.stack(box_b_list))


class HistogramMSE(RayLoss):
    """
    Calculates the histograms of two ray_outputs and the MSE between those two histograms.
    :param n_bins The amount of bins the histograms are generated with.
    """
    def __init__(self, n_bins: int):
        self.n_bins: int = n_bins
    def loss_fn(self, a: Union[Dict, List[Dict], Iterable[Dict]], b: Union[Dict, List[Dict], Iterable[Dict]],
                exported_plane: str) -> torch.Tensor:
        a_dict = a['ray_output'][exported_plane]
        b_dict = b['ray_output'][exported_plane]
        hist_a_list = []
        hist_b_list = []

        for key, a in a_dict.items():
            b = b_dict[key]
            x_min = min(a.x_loc.min(), b.x_loc.min())
            x_max = max(a.x_loc.max(), b.x_loc.max())
            y_min = min(a.y_loc.min(), b.y_loc.min())
            y_max = max(a.y_loc.max(), b.y_loc.max())
            hist_a_list.append(Histogram(self.n_bins, (x_min, x_max), (y_min, y_max))(a)['histogram'])
            hist_b_list.append(Histogram(self.n_bins, (x_min, x_max), (y_min, y_max))(b)['histogram'])
        return ((torch.stack(hist_a_list) - torch.stack(hist_b_list)) ** 2).mean()


def ssim(a, b):
    a = RayOptimizer.ray_output_to_tensor(ray_output=a, exported_plane='ImagePlane')[0].unsqueeze(0)
    b = RayOptimizer.ray_output_to_tensor(ray_output=b, exported_plane='ImagePlane')[0].unsqueeze(0)
    # ssim_fun.update((a, b))
    # return ssim_fun.compute()


def cov_mse(a, b, exported_plane: str):
    a = RayOptimizer.ray_output_to_tensor(ray_output=a, exported_plane=exported_plane)
    b = RayOptimizer.ray_output_to_tensor(ray_output=b, exported_plane=exported_plane)
    cov_a = torch.stack([torch.cov(element) for element in a])
    cov_b = torch.stack([torch.cov(element) for element in b])
    return ((cov_a - cov_b) ** 2).mean()


def histogram_ssim(a, b):
    a = a[0]['ray_output']['ImagePlane']['0']
    b = b[0]['ray_output']['ImagePlane']['0']
    x_min = min(a.x_loc.min(), b.x_loc.min())
    x_max = max(a.x_loc.max(), b.x_loc.max())
    y_min = min(a.y_loc.min(), b.y_loc.min())
    y_max = max(a.y_loc.max(), b.y_loc.max())
    hist_a = Histogram(100, (x_min, x_max), (y_min, y_max))(a)['histogram']
    hist_b = Histogram(100, (x_min, x_max), (y_min, y_max))(b)['histogram']
    hist_a = torch.from_numpy(hist_a).unsqueeze(0).unsqueeze(0).float()
    hist_b = torch.from_numpy(hist_b).unsqueeze(0).unsqueeze(0).float()
    # plt.imshow(hist_a)
    # plt.savefig('a.png')
    # plt.imshow(hist_b)
    # plt.savefig('b.png')
    # plt.imshow((hist_a - hist_b)**2)
    # plt.savefig('a-b.png')
    ## ssim_fun.update((hist_a, hist_b))
    ## return torch.tensor(ssim_fun.compute())


def to_tensor(a):
    return RayOptimizer.ray_output_to_tensor(ray_output=a, exported_plane='ImagePlane')


def sinkhorn_output_loss(a, b):
    return sinkhorn_loss(a, b, exported_plane='ImagePlane')


kld = torch.nn.KLDivLoss(log_target=True)
kld = torch.nn.MSELoss(reduction='mean')


def kld_loss(a, b):
    a = RayOptimizer.ray_output_to_tensor(ray_output=a, exported_plane='ImagePlane')
    b = RayOptimizer.ray_output_to_tensor(ray_output=b, exported_plane='ImagePlane')
    # print(a.shape, b.shape)
    return kld(a[0], b[0])


def js_loss(a, b):
    a = RayOptimizer.ray_output_to_tensor(ray_output=a, exported_plane='ImagePlane')[0]
    b = RayOptimizer.ray_output_to_tensor(ray_output=b, exported_plane='ImagePlane')[0]
    M = 0.5 * (a + b)
    return 0.5 * (kld(a, M) + kld(b, M))


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
        ("number_rays", NumericalParameter(value=1e3)),
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
    for i in range(num_samples):
        perturbed_parameters: list[RayParameterContainer[str, RayParameter]] = [v.clone() for v in params]
        offset_instance = offset()
        if i != 0:
            for configuration in perturbed_parameters:
                configuration.perturb(offset_instance)
        params_list.append(perturbed_parameters)
        offset_list.append(offset_instance[var_name].get_value())
    return params_list, offset_list


def investigate_var(var_name: str, value_lims, loss_function, loss_string):
    engine = GaussEngine()
    num_samples = 300
    params_list, offset_list = create_params_offset_list(var_name, value_lims, num_samples=num_samples)
    outputs_list = [engine.run(params_entry, transforms=MultiLayer([0, 10])) for params_entry in params_list]

    distances_list = []
    for i in trange(1, num_samples):
        distance = loss_function(outputs_list[0], outputs_list[i])
        distances_list.append(distance.item())
    plot_len = 5
    RayOptimizer.fixed_position_plot_base([to_tensor(list_entry) for list_entry in outputs_list[:plot_len]],
                                          xlim=[-2, 2], ylim=[-2, 2],
                                          ylabel=['reference'] + ["{:10.2f}".format(distances_list[list_idx]) for
                                                                  list_idx in range(plot_len - 1)])
    plt.plot()
    plt.savefig('out_' + var_name + '.png')

    plt.clf()
    plt.scatter(np.array(offset_list[1:]), np.array(distances_list[:]), s=0.2)
    # plt.xlim = ([0, 2])
    # plt.ylim([0, 2])
    plt.xlabel('Absolute error')
    plt.ylabel(loss_string + ' distance')
    plt.tight_layout()
    plt.savefig(loss_string + '_' + var_name + '.png')
    plt.clf()


if __name__ == '__main__':
    # print(evaluate_single_var("y_mean", 0.15, loss_function=kld_loss))
    investigate_var('y_mean', value_lims=(0.0, 1.0), loss_function=box_iou, loss_string="generalized_box_iou_loss")
    investigate_var('y_var', value_lims=(0.0, 1.0), loss_function=box_iou, loss_string="generalized_box_iou_loss")
