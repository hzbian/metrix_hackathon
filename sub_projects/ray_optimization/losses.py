import sys
from abc import ABC, abstractmethod
from typing import Union, Dict, List, Iterable, Callable

import ignite
import torch
import torchvision

sys.path.insert(0, '../../')
from ray_tools.base.transform import Histogram
from sub_projects.ray_optimization.utils import ray_output_to_tensor
from ray_nn.metrics.geometric import SinkhornLoss


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


sinkhorn_function = SinkhornLoss(normalize_weights='weights1', p=1, backend='online', reduction=None)


class SinkhornLoss(RayLoss):
    def loss_fn(self, a: Union[Dict, List[Dict], Iterable[Dict]], b: Union[Dict, List[Dict], Iterable[Dict]],
                exported_plane: str) -> torch.Tensor:
        a = ray_output_to_tensor(a, exported_plane=exported_plane)
        b = ray_output_to_tensor(b, exported_plane=exported_plane)
        if torch.cuda.is_available():
            a = a.cuda()

        if a.shape[1] == 0 or a.shape[1] == 1:
            a = torch.ones((a.shape[0], 2, 2), device=a.device, dtype=a.dtype) * -2

        if torch.cuda.is_available():
            b = b.cuda()
        if b.shape[1] == 0 or b.shape[1] == 1:
            b = torch.ones((b.shape[0], 2, 2), device=b.device, dtype=b.dtype) * -1
        loss = sinkhorn_function(a.contiguous(), b.contiguous(), torch.ones_like(a[..., 1]),
                                 torch.ones_like(b[..., 1]))
        return loss.mean()


class RayCountMSE(RayLoss):
    def loss_fn(self, a: Union[Dict, List[Dict], Iterable[Dict]], b: Union[Dict, List[Dict], Iterable[Dict]],
                exported_plane: str) -> torch.Tensor:
        a_tensor = ray_output_to_tensor(a, exported_plane=exported_plane)
        b_tensor = ray_output_to_tensor(b, exported_plane=exported_plane)
        return torch.tensor((a_tensor.shape[1] - b_tensor.shape[1]) ** 2 / 2, dtype=a_tensor.dtype,
                            device=a_tensor.device)


class MultiObjectiveLoss(RayLoss):
    def __init__(self, loss_fn_a: RayLoss, loss_fn_b: RayLoss):
        self.loss_fn_a: RayLoss = loss_fn_a
        self.loss_fn_b: RayLoss = loss_fn_b

    def loss_fn(self, a: Union[Dict, List[Dict], Iterable[Dict]], b: Union[Dict, List[Dict], Iterable[Dict]],
                exported_plane: str) -> torch.Tensor:
        return torch.stack((self.loss_fn_a.loss_fn(a, b, exported_plane), self.loss_fn_b.loss_fn(a, b, exported_plane)))


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

    def reset_passed_epochs(self):
        self.passed_epochs = 0

    def end_epoch(self):
        self.passed_epochs += 1

    def loss_fn(self, a: Union[Dict, List[Dict], Iterable[Dict]], b: Union[Dict, List[Dict], Iterable[Dict]],
                exported_plane: str) -> torch.Tensor:
        if self.passed_epochs < self.loss_a_epochs:
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

    def __init__(self, base_fn: Union[Callable[..., torch.Tensor], str], reduction='none'):
        if isinstance(base_fn, str):
            if base_fn == 'torchvision.ops.complete_box_iou':
                base_fn == torchvision.ops.complete_box_iou
            if base_fn == 'torchvision.ops.distance_box_iou_loss':
                base_fn == torchvision.ops.distance_box_iou_loss
            if base_fn == 'torchvision.ops.generalized_box_iou_loss':
                base_fn == torchvision.ops.generalized_box_iou_loss
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
            hist_a_list.append(torch.from_numpy(Histogram(self.n_bins, (x_min, x_max), (y_min, y_max))(a)['histogram']))
            hist_b_list.append(torch.from_numpy(Histogram(self.n_bins, (x_min, x_max), (y_min, y_max))(b)['histogram']))
        return ((torch.stack(hist_a_list) - torch.stack(hist_b_list)) ** 2).mean()


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


class TorchLoss(RayLoss):
    """
    Implementation of PyTorch losses. This class is meant to be used with a Torchvision function
    as input as described in the `PyTorch documentation <https://pytorch.org/vision/master/ops.html#losses>`_
    :param base_fn can be one of `torchvision.ops.complete_box_iou`, ``torchvision.ops.distance_box_iou_loss`` or
    ``torchvision.ops.generalized_box_iou_loss``.
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


class SSIMHistogramLoss(RayLoss):
    """
    Calculates the histograms of two ray_outputs and the SSIM between those two histograms.
    :param n_bins The amount of bins the histograms are generated with.
    """

    def __init__(self, n_bins: int):
        self.n_bins: int = n_bins
        self.ssim_fun = ignite.metrics.SSIM(1.0)

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
            hist_a_list.append(torch.from_numpy(Histogram(self.n_bins, (x_min, x_max), (y_min, y_max))(a)['histogram']))
            hist_b_list.append(torch.from_numpy(Histogram(self.n_bins, (x_min, x_max), (y_min, y_max))(b)['histogram']))
        stack_a = torch.stack(hist_a_list).unsqueeze(1).float()
        stack_b = torch.stack(hist_b_list).unsqueeze(1).float()
        self.ssim_fun.update((stack_a, stack_b))
        return torch.Tensor([self.ssim_fun.compute()])
