from typing import Dict, Union

import torch
from torch import nn
from geomloss import SamplesLoss

from ..utils.ray_processing import HistToPointCloud


class SinkhornLoss(nn.Module):

    def __init__(self,
                 p: int = 2,
                 blur: float = 0.05,
                 normalize_weights: Union[bool, str] = False,
                 backend: str = 'tensorized',
                 reduction='mean') -> None:
        super().__init__()
        self.p = p
        self.blur = blur
        self.backend = backend
        self.normalize_weights = normalize_weights
        self.reduction = reduction
        self._loss_func = SamplesLoss("sinkhorn", p=self.p, blur=self.blur, backend=self.backend)

    def forward(self,
                inp1: torch.Tensor,
                inp2: torch.Tensor,
                weights1: torch.tensor = None,
                weights2: torch.tensor = None) -> torch.Tensor:
        if weights1 is not None and weights2 is not None:
            if self.normalize_weights is True:
                mass1 = weights1.sum(dim=-1, keepdim=True)
                mass1[mass1 == 0.0] = 1.0
                weights1 = weights1 / mass1
                mass2 = weights2.sum(dim=-1, keepdim=True)
                mass2[mass2 == 0.0] = 1.0
                weights2 = weights2 / mass2
            elif self.normalize_weights == 'weights1':
                mass = weights1.sum(dim=-1, keepdim=True)
                mass[mass == 0.0] = 1.0
                weights1 = weights1 / mass
                weights2 = weights2 / mass
            elif self.normalize_weights == 'weights2':
                mass = weights2.sum(dim=-1, keepdim=True)
                mass[mass == 0.0] = 1.0
                weights1 = weights1 / mass
                weights2 = weights2 / mass

            loss = self._loss_func(weights1, inp1, weights2, inp2)
        else:
            loss = self._loss_func(inp1, inp2)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction is None:
            return loss
        else:
            raise NotImplementedError


class SurrogateLoss(nn.Module):

    def __init__(self,
                 sinkhorn_p: int = 1,
                 sinkhorn_blur: float = 0.05,
                 sinkhorn_normalize: bool = False,
                 sinkhorn_standardize_lims: bool = False,
                 total_weight: float = 1.0,
                 n_rays_loss_weight: float = 0.0) -> None:
        super().__init__()
        self.loss_sinkhorn = SinkhornLoss(p=sinkhorn_p,
                                          blur=sinkhorn_blur,
                                          backend='online',
                                          normalize_weights=sinkhorn_normalize,
                                          reduction='mean')
        self.sinkhorn_standardize_lims = sinkhorn_standardize_lims

        self.total_weight = total_weight
        self.n_rays_loss_weight = n_rays_loss_weight

        self._hist_to_pc = HistToPointCloud()

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        pred_x_lims, pred_y_lims = batch['pred_x_lims'].clone(), batch['pred_y_lims'].clone()
        tar_x_lims, tar_y_lims = batch['tar_x_lims'].clone(), batch['tar_y_lims'].clone()

        if self.sinkhorn_standardize_lims:
            idx_nonzero = batch['tar_n_rays'] > 0.0

            tar_x_scale = (tar_x_lims[idx_nonzero][:, 1] - tar_x_lims[idx_nonzero][:, 0]).abs().mean()
            tar_y_scale = (tar_y_lims[idx_nonzero][:, 1] - tar_y_lims[idx_nonzero][:, 0]).abs().mean()

            pred_x_lims[idx_nonzero] = pred_x_lims[idx_nonzero] / tar_x_scale
            pred_y_lims[idx_nonzero] = pred_y_lims[idx_nonzero] / tar_y_scale
            tar_x_lims[idx_nonzero] = tar_x_lims[idx_nonzero] / tar_x_scale
            tar_y_lims[idx_nonzero] = tar_y_lims[idx_nonzero] / tar_y_scale

        pred_pc_supp, pred_pc_weights = self._hist_to_pc(batch['pred_hist'].flatten(end_dim=1),
                                                         pred_x_lims.flatten(end_dim=1),
                                                         pred_y_lims.flatten(end_dim=1))
        tar_pc_supp, tar_pc_weights = self._hist_to_pc(batch['tar_hist'].flatten(end_dim=1),
                                                       tar_x_lims.flatten(end_dim=1),
                                                       tar_y_lims.flatten(end_dim=1))

        if self.total_weight > 0.0:
            loss = self.total_weight * self.loss_sinkhorn(inp1=pred_pc_supp,
                                                          inp2=tar_pc_supp,
                                                          weights1=pred_pc_weights,
                                                          weights2=tar_pc_weights)
        else:
            loss = torch.tensor(0.0, device=batch['tar_n_rays'].device)

        if self.n_rays_loss_weight > 0.0:
            loss = loss + self.n_rays_loss_weight * (batch['pred_n_rays'] - batch['tar_n_rays']).abs().mean()

        return loss
