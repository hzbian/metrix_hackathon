from typing import Callable, Dict, Iterable, List, Union
import torch
import torchvision
from ray_tools.base.transform import Histogram
from sub_projects.ray_optimization.losses.losses import RayLoss
from sub_projects.ray_optimization.utils import ray_output_to_tensor


class TorchLoss(RayLoss):
    """
    Implementation of PyTorch losses. This class is meant to be used with a Torch loss function module.
    """

    def __init__(self, base_fn: torch.nn.Module):
        self.base_fn: torch.nn.Module = base_fn

    def loss_fn(self, a: Union[Dict, List[Dict], Iterable[Dict]], b: Union[Dict, List[Dict], Iterable[Dict]],
                exported_plane: str) -> torch.Tensor:
        a = ray_output_to_tensor(ray_output=a, exported_plane=exported_plane)
        b = ray_output_to_tensor(ray_output=b, exported_plane=exported_plane)

        losses = torch.stack([self.base_fn(element, b[i]) for i, element in enumerate(a)])
        return losses.mean()

class KLDLoss(TorchLoss):
    def __init__(self, reduction='none'):
        super().__init__(torch.nn.KLDivLoss(reduction=reduction, log_target=True))


class MSELoss(TorchLoss):
    def __init__(self, reduction='none'):
        super().__init__(torch.nn.MSELoss(reduction=reduction))


class JSLoss(TorchLoss):
    def __init__(self, reduction='none'):
        class OwnModule(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.kld = torch.nn.KLDivLoss(reduction=reduction, log_target=True)

            def __call__(self, a, b):
                m = 0.5 * (a + b)
                return 0.5 * (self.kld(a, m) + self.kld(b, m))

        super().__init__(OwnModule())

class CovMSE(RayLoss):
    """
    Estimates covariance matrices for both inputs and calculates their mean squared error.
    """

    def loss_fn(self, a: Union[Dict, List[Dict], Iterable[Dict]], b: Union[Dict, List[Dict], Iterable[Dict]],
                exported_plane: str) -> torch.Tensor:
        a = ray_output_to_tensor(ray_output=a, exported_plane=exported_plane)
        b = ray_output_to_tensor(ray_output=b, exported_plane=exported_plane)
        cov_a = torch.stack([torch.cov(element) for element in a])
        cov_b = torch.stack([torch.cov(element) for element in b])
        return ((cov_a - cov_b) ** 2).mean()
    


class BoxIoULoss(RayLoss):
    """
    Implementation of box intersection-over-union losses. This class is meant to be used with a Torchvision function
    as input as described in the `PyTorch documentation <https://pytorch.org/vision/master/ops.html#losses>`_
    :param base_fn can be one of ``torchvision.ops.complete_box_iou_loss``, ``torchvision.ops.distance_box_iou_loss`` or
    ``torchvision.ops.generalized_box_iou_loss``.
    """

    def __init__(self, base_fn: Union[Callable[..., torch.Tensor], str], reduction='none'):
        if isinstance(base_fn, str):
            if base_fn == 'torchvision.ops.complete_box_iou_loss':
                base_fn = torchvision.ops.complete_box_iou_loss
            if base_fn == 'torchvision.ops.distance_box_iou_loss':
                base_fn = torchvision.ops.distance_box_iou_loss
            if base_fn == 'torchvision.ops.generalized_box_iou_loss':
                base_fn = torchvision.ops.generalized_box_iou_loss
        self.base_fn: Callable[..., torch.Tensor] = base_fn
        self.reduction = reduction

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
                 shift_y + a.y_loc.max()))
            box_a_list.append(box_a)
            box_b = torch.tensor(
                (shift_x + b.x_loc.min(), shift_y + b.y_loc.min(), shift_x + b.x_loc.max(),
                 shift_y + b.y_loc.max()))
            box_b_list.append(box_b)

        return self.base_fn(torch.stack(box_a_list), torch.stack(box_b_list), reduction=self.reduction)


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

