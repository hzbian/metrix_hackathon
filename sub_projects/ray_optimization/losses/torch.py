from collections.abc import Callable
import torch
import torchvision
from ray_tools.base.backend import RayOutput
from ray_tools.base.transform import Histogram
from sub_projects.ray_optimization.losses.losses import RayLoss
from sub_projects.ray_optimization.utils import ray_dict_to_tensor


class TorchLoss(RayLoss):
    """
    Implementation of PyTorch losses. This class is meant to be used with a Torch loss function module. If shapes of `a` and `b` are different, the smaller sizes are taken and the excessing rays get discarded.
    """

    def __init__(self, base_fn: Callable, ray_count_distance_fn: Callable | None = None):
        self.base_fn: Callable = base_fn
        if ray_count_distance_fn is None:
            self.ray_count_distance_fn: Callable = self.base_fn
        else:
            self.ray_count_distance_fn: Callable = ray_count_distance_fn
        self.ray_count_empty_threshold: int = 1

    def loss_fn(
        self,
        a: dict,
        b: dict,
        exported_plane: str,
    ) -> torch.Tensor:
        a_tensor: torch.Tensor = ray_dict_to_tensor(a, exported_plane=exported_plane)
        b_tensor: torch.Tensor = ray_dict_to_tensor(b, exported_plane=exported_plane)

        # if both are empty, it is a perfect fit
        if a_tensor.nelement() <= self.ray_count_empty_threshold and b_tensor.nelement() <= self.ray_count_empty_threshold:
            return torch.tensor([0.0], device=a_tensor.device)

        # if not one of them is empty, cut the larger one
        if a_tensor.nelement() > self.ray_count_empty_threshold and b_tensor.nelement() > self.ray_count_empty_threshold:
            new_size = torch.min(torch.tensor(a_tensor.shape), torch.tensor(b_tensor.shape))
            a_tensor = a_tensor[[slice(0, new_size[i]) for i in range(len(new_size))]]
            b_tensor = b_tensor[[slice(0, new_size[i]) for i in range(len(new_size))]]

        # if one is empty, the other one is not, let us count the rays and take the difference with base function
        if a_tensor.nelement() <= self.ray_count_empty_threshold or b_tensor.nelement() <= self.ray_count_empty_threshold:
            return self.ray_count_distance_fn(
                torch.tensor(a_tensor.nelement(), device=a_tensor.device).float(),
                torch.tensor(b_tensor.nelement(), device=a_tensor.device).float(),
            )
        
        assert a_tensor.nelement() > self.ray_count_empty_threshold and b_tensor.nelement() > self.ray_count_empty_threshold

        losses = torch.stack(
            [self.base_fn(element, b_tensor[i]) for i, element in enumerate(a_tensor)]
        )
        return losses.mean()


class KLDLoss(TorchLoss):
    def __init__(self, reduction="none"):
        super().__init__(torch.nn.KLDivLoss(reduction=reduction, log_target=True))


class MSELoss(TorchLoss):
    def __init__(self, reduction="none"):
        super().__init__(torch.nn.MSELoss(reduction=reduction))

class MeanMSELoss(TorchLoss):
    def __init__(self, reduction="none"):
        mse  = torch.nn.MSELoss(reduction=reduction)
        def base_fn(a: torch.Tensor, b: torch.Tensor):
            assert a.numel() > self.ray_count_empty_threshold
            assert b.numel() > self.ray_count_empty_threshold
            return mse(a.mean(dim=0).sum(), b.mean(dim=0).sum())
        super().__init__(base_fn=base_fn, ray_count_distance_fn=mse)

class VarMSELoss(TorchLoss):
    def __init__(self, reduction="none"):
        mse  = torch.nn.MSELoss(reduction=reduction)
        def base_fn(a: torch.Tensor, b: torch.Tensor):
            assert a.numel() > self.ray_count_empty_threshold
            assert b.numel() > self.ray_count_empty_threshold
            return mse(a.var(dim=0).sum(), b.var(dim=0).sum())
        super().__init__(base_fn=base_fn, ray_count_distance_fn=mse)


class JSLoss(TorchLoss):
    def __init__(self, reduction="none"):
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

    def loss_fn(
        self,
        a: dict,
        b: dict,
        exported_plane: str,
    ) -> torch.Tensor:
        a_tensor: torch.Tensor = ray_dict_to_tensor(a, exported_plane=exported_plane)
        b_tensor: torch.Tensor = ray_dict_to_tensor(b, exported_plane=exported_plane)
        cov_a = torch.stack([torch.cov(element) for element in a_tensor])
        cov_b = torch.stack([torch.cov(element) for element in b_tensor])
        return ((cov_a - cov_b) ** 2).mean()


class BoxIoULoss(RayLoss):
    """
    Implementation of box intersection-over-union losses. This class is meant to be used with a Torchvision function
    as input as described in the `PyTorch documentation <https://pytorch.org/vision/master/ops.html#losses>`_
    :param base_fn can be one of ``torchvision.ops.complete_box_iou_loss``, ``torchvision.ops.distance_box_iou_loss`` or
    ``torchvision.ops.generalized_box_iou_loss``.
    """

    def __init__(
        self, base_fn: Callable[..., torch.Tensor] | str, reduction="none"
    ):
        if isinstance(base_fn, str):
            if base_fn == "complete_box_iou_loss":
                base_fn = torchvision.ops.complete_box_iou_loss
            if base_fn == "distance_box_iou_loss":
                base_fn = torchvision.ops.distance_box_iou_loss
            if base_fn == "generalized_box_iou_loss":
                base_fn = torchvision.ops.generalized_box_iou_loss
        assert isinstance(base_fn, Callable)
        self.base_fn: Callable[..., torch.Tensor] = base_fn
        self.reduction = reduction

    def loss_fn(
        self,
        a: dict,
        b: dict,
        exported_plane: str,
    ) -> torch.Tensor:
        a_dict = a["ray_output"][exported_plane]
        b_dict = b["ray_output"][exported_plane]
        box_a_list = []
        box_b_list = []

        for key, a in a_dict.items():
            b = b_dict[key]
            assert isinstance(a, RayOutput)
            assert isinstance(b, RayOutput)
            global_x_min = torch.min(a.x_loc.min(), b.x_loc.min())
            global_y_min = torch.min(a.y_loc.min(), b.y_loc.min())
            shift_x = -global_x_min if global_x_min < -1 else 0
            shift_y = -global_y_min if global_y_min < -1 else 0
            box_a = torch.tensor(
                (
                    shift_x + a.x_loc.min(),
                    shift_y + a.y_loc.min(),
                    shift_x + a.x_loc.max(),
                    shift_y + a.y_loc.max(),
                )
            )
            box_a_list.append(box_a)
            box_b = torch.tensor(
                (
                    shift_x + b.x_loc.min(),
                    shift_y + b.y_loc.min(),
                    shift_x + b.x_loc.max(),
                    shift_y + b.y_loc.max(),
                )
            )
            box_b_list.append(box_b)

        return self.base_fn(
            torch.stack(box_a_list), torch.stack(box_b_list), reduction=self.reduction
        )


class HistogramMSE(RayLoss):
    """
    Calculates the histograms of two ray_outputs and the MSE between those two histograms.
    :param n_bins The amount of bins the histograms are generated with.
    """

    def __init__(self, n_bins: int):
        self.n_bins: int = n_bins

    def loss_fn(
        self,
        a: dict,
        b: dict,
        exported_plane: str,
    ) -> torch.Tensor:
        a_dict = a["ray_output"][exported_plane]
        b_dict = b["ray_output"][exported_plane]
        hist_a_list = []
        hist_b_list = []

        for key, a in a_dict.items():
            b = b_dict[key]
            assert isinstance(a, RayOutput)
            assert isinstance(b, RayOutput)
            x_min = torch.min(a.x_loc.min(), b.x_loc.min()).item()
            x_max = torch.max(a.x_loc.max(), b.x_loc.max()).item()
            y_min = torch.min(a.y_loc.min(), b.y_loc.min()).item()
            y_max = torch.max(a.y_loc.max(), b.y_loc.max()).item()
            hist_a_list.append(
                Histogram(self.n_bins, (x_min, x_max), (y_min, y_max))(a)["histogram"]
            )
            hist_b_list.append(
                Histogram(self.n_bins, (x_min, x_max), (y_min, y_max))(b)["histogram"]
            )
        return ((torch.stack(hist_a_list) - torch.stack(hist_b_list)) ** 2).mean()
