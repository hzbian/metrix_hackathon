import torch
from ray_tools.base.parameter import RayParameterContainer


class Sample:
    def __init__(
        self,
        params=RayParameterContainer(),
        rays: list[torch.Tensor] | None = None,
        loss: float = float("inf"),
        epoch: int = 0,
    ):
        self._params: RayParameterContainer = params
        self._rays: list[torch.Tensor] | None = rays
        self._loss: float = loss
        self._epoch: int = epoch

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value: RayParameterContainer):
        self._params = value

    @property
    def rays(self):
        return self._rays

    @rays.setter
    def rays(self, rays: list[torch.Tensor] | None):
        self._rays = rays

    @property
    def loss(self):
        return self._loss

    @loss.setter
    def loss(self, loss: float):
        self._loss = loss

    @property
    def epoch(self):
        return self._epoch

    @epoch.setter
    def epoch(self, epoch: int):
        self._epoch = epoch
    
    def to_cpu(self):
        if self.rays is not None:
            rays =  [rays.cpu() for rays in self.rays]
        else:
            rays = None
        return Sample(self.params, rays, self.loss, self.epoch)
