from typing import Dict

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
                 sinkhorn_weight: float = 1.0,
                 mae_lims_weight: float = 1.0,
                 mae_hist_weight: float = 1.0) -> None:
        super().__init__()
        self.loss_sinkhorn = SinkhornLoss(p=sinkhorn_p,
                                          blur=sinkhorn_blur,
                                          backend='online',
                                          normalize_weights=sinkhorn_normalize,
                                          reduction='mean')

        self.hist_to_pc = HistToPointCloud()

        self.sinkhorn_weight = sinkhorn_weight
        self.mae_lims_weight = mae_lims_weight
        self.mae_hist_weight = mae_hist_weight

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        dim_x = dim_y = int(np.sqrt(batch['pred_hist'].shape[1]))
        pred_pc_supp, pred_pc_weights = self.hist_to_pc(batch['pred_hist'].view(-1, dim_x, dim_y),
                                                        batch['pred_x_lims'], batch['pred_y_lims'])
        tar_pc_supp, tar_pc_weights = self.hist_to_pc(batch['tar_hist'].view(-1, dim_x, dim_y),
                                                      batch['tar_x_lims'], batch['tar_y_lims'])
        loss = 0.0
        if self.sinkhorn_weight > 0:
            loss = loss + self.sinkhorn_weight * self.loss_sinkhorn(inp1=pred_pc_supp,
                                                                    inp2=tar_pc_supp,
                                                                    weights1=pred_pc_weights,
                                                                    weights2=tar_pc_weights)
        if self.mae_lims_weight > 0:
            loss = loss + self.mae_lims_weight * (
                    (batch['pred_x_lims'] - batch['tar_x_lims']).abs().sum(dim=1).mean() +
                    (batch['pred_y_lims'] - batch['tar_y_lims']).abs().sum(dim=1).mean()
            )

        if self.mae_hist_weight > 0:
            loss = loss + self.mae_hist_weight * \
                   ((batch['pred_hist'] - batch['tar_hist']).abs().sum(dim=1) /
                    batch['pred_hist'].shape[1]  # (batch['tar_hist'].abs().sum(dim=1) + 1)
                    ).mean()

        return loss
