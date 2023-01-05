import torch
from torch import nn
from geomloss import SamplesLoss


class SinkhornLoss(nn.Module):

    def __init__(self, p: int = 2, blur: float = 0.05, backend: str = 'tensorized', reduction='mean') -> None:
        super().__init__()
        self.p = p
        self.blur = blur
        self.backend = backend
        self.reduction = reduction
        self._loss_func = SamplesLoss("sinkhorn", p=self.p, blur=self.blur, backend=self.backend)

    def forward(self,
                inp1: torch.Tensor,
                inp2: torch.Tensor,
                weights1: torch.tensor = None,
                weights2: torch.tensor = None) -> torch.Tensor:
        if weights1 is not None and weights2 is not None:
            loss = self._loss_func(weights1, inp1, weights2, inp2)
        else:
            loss = self._loss_func(inp1, inp2)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction is None:
            return loss
        else:
            raise NotImplementedError
