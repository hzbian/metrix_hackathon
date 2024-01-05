import numpy as np
import torch
from torch import nn

from ray_nn.metrics.geometric import SinkhornLoss
from ray_nn.utils.ray_processing import HistToPointCloud


class SurrogateLoss(nn.Module):

    def __init__(self,
                 sinkhorn_p: int = 1,
                 sinkhorn_blur: float = 0.05,
                 sinkhorn_normalize: bool = False,
                 sinkhorn_standardize_lims: bool = False,
                 sinkhorn_n_rays_weighting: bool = False,
                 sinkhorn_weight: float = 1.0,
                 mae_lims_weight: float = 1.0,
                 mae_hist_weight: float = 1.0,
                 mae_n_rays_weight: float = 0.0) -> None:
        super().__init__()
        self.loss_sinkhorn = SinkhornLoss(p=sinkhorn_p,
                                          blur=sinkhorn_blur,
                                          backend='online',
                                          normalize_weights=sinkhorn_normalize,
                                          reduction=None)
        self.sinkhorn_n_rays_weighting = sinkhorn_n_rays_weighting

        self.sinkhorn_standardize_lims = sinkhorn_standardize_lims
        self.hist_to_pc = HistToPointCloud()

        self.sinkhorn_weight = sinkhorn_weight
        self.mae_hist_weight = mae_hist_weight
        self.mae_lims_weight = mae_lims_weight
        self.mae_n_rays_weight = mae_n_rays_weight

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        dim_x = dim_y = int(np.sqrt(batch['pred_hist'].shape[1]))

        pred_x_lims, pred_y_lims = batch['pred_x_lims'].clone(), batch['pred_y_lims'].clone()
        tar_x_lims, tar_y_lims = batch['tar_x_lims'].clone(), batch['tar_y_lims'].clone()

        if self.sinkhorn_standardize_lims:
            idx_non_empty = batch['tar_n_rays'] > 1.0

            tar_x_scale = (tar_x_lims[idx_non_empty, 1] - tar_x_lims[idx_non_empty, 0]).abs().mean()
            tar_y_scale = (tar_y_lims[idx_non_empty, 1] - tar_y_lims[idx_non_empty, 0]).abs().mean()

            pred_x_lims[idx_non_empty, ...] = pred_x_lims[idx_non_empty, ...] / tar_x_scale
            pred_y_lims[idx_non_empty, ...] = pred_y_lims[idx_non_empty, ...] / tar_y_scale
            tar_x_lims[idx_non_empty, ...] = tar_x_lims[idx_non_empty, ...] / tar_x_scale
            tar_y_lims[idx_non_empty, ...] = tar_y_lims[idx_non_empty, ...] / tar_y_scale

        pred_pc_supp, pred_pc_weights = self.hist_to_pc(batch['pred_hist'].view(-1, dim_x, dim_y),
                                                        pred_x_lims, pred_y_lims)
        tar_pc_supp, tar_pc_weights = self.hist_to_pc(batch['tar_hist'].view(-1, dim_x, dim_y),
                                                      tar_x_lims, tar_y_lims)
        loss = 0.0
        if self.sinkhorn_weight > 0:
            loss_ = self.loss_sinkhorn(inp1=pred_pc_supp,
                                       inp2=tar_pc_supp,
                                       weights1=pred_pc_weights,
                                       weights2=tar_pc_weights)
            if self.sinkhorn_n_rays_weighting:
                n_rays_weighting = batch['pred_n_rays'].clone() / batch['pred_n_rays'].sum()
                loss_ = (n_rays_weighting.flatten() * loss_.flatten()).sum()
            else:
                loss_ = loss_.mean()

            loss = loss + self.sinkhorn_weight * loss_

        if self.mae_hist_weight > 0:
            loss = loss + self.mae_hist_weight * \
                   ((batch['pred_hist'] - batch['tar_hist']).abs().sum(dim=1) /
                    batch['pred_hist'].shape[1]  # (batch['tar_hist'].abs().sum(dim=1) + 1)
                    ).mean()

        if self.mae_lims_weight > 0:
            loss = loss + self.mae_lims_weight * (
                    (batch['pred_x_lims'] - batch['tar_x_lims']).abs().sum(dim=1).mean() +
                    (batch['pred_y_lims'] - batch['tar_y_lims']).abs().sum(dim=1).mean()
            )

        if self.mae_n_rays_weight > 0:
            loss = loss + self.mae_n_rays_weight * ((batch['pred_n_rays'] - batch['tar_n_rays']).abs() /
                                                    1.0  # batch['tar_n_rays']
                                                    ).mean()

        return loss
