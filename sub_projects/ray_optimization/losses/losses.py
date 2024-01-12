from abc import ABC, abstractmethod

import torch

from sub_projects.ray_optimization.utils import ray_dict_to_tensor


class RayLoss(ABC):
    """
    Base class for defining losses.
    """

    @abstractmethod
    def loss_fn(
        self,
        a: dict,
        b: dict,
        exported_plane: str,
    ) -> torch.Tensor:
        """
        Function that calculates the loss from input parameters a and b.
        :param a: Ray_output that should be compared.
        :param b: Ray_output that should be compared.
        :param exported_plane: The plane of ray_output that should be compared.
        :return: Calculated loss.
        """
        pass

class LogLoss(RayLoss):
    def __init__(self, base_loss: RayLoss):
        self.base_loss: RayLoss = base_loss
    def loss_fn(
        self,
        a: dict,
        b: dict,
        exported_plane: str,
    ) -> torch.Tensor:
        return self.base_loss.loss_fn(a, b, exported_plane=exported_plane).log()
       
class RayCountMSE(RayLoss):
    def loss_fn(
        self,
        a: dict,
        b: dict,
        exported_plane: str,
    ) -> torch.Tensor:
        a_tensor = ray_dict_to_tensor(a, exported_plane=exported_plane)
        b_tensor = ray_dict_to_tensor(b, exported_plane=exported_plane)
        return torch.tensor(
            (a_tensor.shape[1] - b_tensor.shape[1]) ** 2 / 2,
            dtype=a_tensor.dtype,
            device=a_tensor.device,
        )


class MultiObjectiveLoss(RayLoss):
    def __init__(self, loss_fn_a: RayLoss, loss_fn_b: RayLoss):
        self.loss_fn_a: RayLoss = loss_fn_a
        self.loss_fn_b: RayLoss = loss_fn_b

    def loss_fn(
        self,
        a: dict,
        b: dict,
        exported_plane: str,
    ) -> torch.Tensor:
        return torch.stack(
            (
                self.loss_fn_a.loss_fn(a, b, exported_plane),
                self.loss_fn_b.loss_fn(a, b, exported_plane),
            )
        )


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

    def loss_fn(
        self,
        a: dict,
        b: dict,
        exported_plane: str,
    ) -> torch.Tensor:
        if self.passed_epochs < self.loss_a_epochs:
            return self.loss_a.loss_fn(a, b, exported_plane)
        else:
            return self.loss_b.loss_fn(a, b, exported_plane)
