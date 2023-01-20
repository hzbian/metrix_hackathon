from typing import Dict, Union

import torch
from torch import nn
from geomloss import SamplesLoss

from ..utils.ray_processing import HistToPointCloud


class SinkhornLoss(nn.Module):
    """
    Sinkhorn loss implementation using the geomloss package
    :param p: See :class:`geomloss.SamplesLoss`.
    :param blur: See :class:`geomloss.SamplesLoss`.
    :param normalize_weights: False = no normalization / True = normalize both weights independently to 1 /
        'weights1' = normalize with weights of first input / 'weights2' = normalize with weights of second input.
    :param backend: See :class:`geomloss.SamplesLoss`. Use 'online' for fast computation.
    :param reduction: Can be 'mean' or None.
    """

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
        """
        ``weights1=None`` and ``weights2=None`` means that all points have weight 1.
        """
        if weights1 is not None and weights2 is not None:
            if self.normalize_weights is True:
                mass1 = weights1.sum(dim=-1, keepdim=True)
                mass1[mass1 == 0.0] = 1.0  # handle zero mass
                weights1 = weights1 / mass1
                mass2 = weights2.sum(dim=-1, keepdim=True)
                mass2[mass2 == 0.0] = 1.0  # handle zero mass
                weights2 = weights2 / mass2
            elif self.normalize_weights == 'weights1':
                mass = weights1.sum(dim=-1, keepdim=True)
                mass[mass == 0.0] = 1.0  # handle zero mass
                weights1 = weights1 / mass
                weights2 = weights2 / mass
            elif self.normalize_weights == 'weights2':
                mass = weights2.sum(dim=-1, keepdim=True)
                mass[mass == 0.0] = 1.0  # handle zero mass
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
                 total_weight: float = 1.0,  # weight for sinkhorn loss
                 lims_loss_weight: float = 0.0,  # weight for limit mse loss
                 n_rays_loss_weight: float = 0.0,  # weight for number of rays loss
                 hist_zero_loss_weight: float = 0.0,  # weight for empty histogram loss
                 ) -> None:
        super().__init__()
        self.loss_sinkhorn = SinkhornLoss(p=sinkhorn_p,
                                          blur=sinkhorn_blur,
                                          backend='online',
                                          normalize_weights=sinkhorn_normalize,
                                          reduction='mean')
        self.sinkhorn_standardize_lims = sinkhorn_standardize_lims

        self.total_weight = total_weight
        self.lims_loss_weight = lims_loss_weight
        self.n_rays_loss_weight = n_rays_loss_weight
        self.hist_zero_loss_weight = hist_zero_loss_weight
        self.hist_zero_loss = nn.BCELoss(reduction='mean')

        self._hist_to_pc = HistToPointCloud()

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Note that batch is already with respect to a specific (image) plane.
        """
        # note: cloning is important when self.sinkhorn_standardize_lims=True
        pred_x_lims, pred_y_lims = batch['pred_x_lims'].clone(), batch['pred_y_lims'].clone()
        tar_x_lims, tar_y_lims = batch['tar_x_lims'].clone(), batch['tar_y_lims'].clone()

        # normalize the limits over the given batch (with respect to target limits), ignoring all empty histograms
        # this ensures that distances between data are approximately measured equally on the x- and y-axis
        if self.sinkhorn_standardize_lims:
            # note that empty histograms have "one ray" according to
            # ray_nn.data.transform.SurrogateModelPreparation._process_zero_hist
            idx_nonzero = batch['tar_n_rays'] > 1.0

            tar_x_scale = (tar_x_lims[idx_nonzero][:, 1] - tar_x_lims[idx_nonzero][:, 0]).abs().mean()
            tar_y_scale = (tar_y_lims[idx_nonzero][:, 1] - tar_y_lims[idx_nonzero][:, 0]).abs().mean()

            pred_x_lims[idx_nonzero] = pred_x_lims[idx_nonzero] / tar_x_scale
            pred_y_lims[idx_nonzero] = pred_y_lims[idx_nonzero] / tar_y_scale
            tar_x_lims[idx_nonzero] = tar_x_lims[idx_nonzero] / tar_x_scale
            tar_y_lims[idx_nonzero] = tar_y_lims[idx_nonzero] / tar_y_scale

        # convert histograms to point clouds ("batch dimension" is the number image planes)
        pred_pc_supp, pred_pc_weights = self._hist_to_pc(batch['pred_hist'].flatten(end_dim=1),
                                                         pred_x_lims.flatten(end_dim=1),
                                                         pred_y_lims.flatten(end_dim=1))
        tar_pc_supp, tar_pc_weights = self._hist_to_pc(batch['tar_hist'].flatten(end_dim=1),
                                                       tar_x_lims.flatten(end_dim=1),
                                                       tar_y_lims.flatten(end_dim=1))

        # Sinkhorn loss
        if self.total_weight > 0.0:
            loss = self.total_weight * self.loss_sinkhorn(inp1=pred_pc_supp,
                                                          inp2=tar_pc_supp,
                                                          weights1=pred_pc_weights,
                                                          weights2=tar_pc_weights)
        else:
            loss = torch.tensor(0.0, device=batch['tar_n_rays'].device)

        # Limits loss: measure mse between predicted and target limits
        if self.lims_loss_weight > 0.0:
            loss = loss + self.lims_loss_weight * (
                    torch.nn.functional.mse_loss(pred_x_lims, tar_x_lims, reduction='mean') +
                    torch.nn.functional.mse_loss(pred_y_lims, tar_y_lims, reduction='mean')
            )

        # Number of rays loss: mean over |1 - pred_n_rays / tar_n_rays|
        if self.n_rays_loss_weight > 0.0:
            loss = loss + self.n_rays_loss_weight * (1.0 - batch['pred_n_rays'] / batch['tar_n_rays']).abs().mean()

        # Loss for empty histogram classifier: a simple binary cross entropy criterion is used
        # pred_hist_zero_prob is the predicted "probability" that a histogram is empty
        if self.hist_zero_loss_weight > 0.0:
            # 0-1 label for empty histograms (1 = empty)
            hist_zero_labels = (batch['tar_n_rays'] < 2.0).to(torch.get_default_dtype())
            loss = loss + self.hist_zero_loss_weight * self.hist_zero_loss(batch['pred_hist_zero_prob'],
                                                                           hist_zero_labels)

        return loss


class HistZeroAccuracy(nn.Module):
    """
    Validation metric that measures the accuracy of correct predictions of empty histograms.
    """

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        hist_zero_tar = (batch['tar_n_rays'] < 2.0).to(torch.get_default_dtype())
        hist_zero_pred = (batch['pred_hist_zero_prob'] > 0.5).to(torch.get_default_dtype())
        loss = 1.0 - (hist_zero_pred - hist_zero_tar).abs().mean()
        return loss


class NRaysAccuracy(nn.Module):
    """
    Validation metric that measures the accuracy of correct predictions of the number of rays:
    |1 - pred_n_rays / tar_n_rays|
    """

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return (1.0 - batch['pred_n_rays'] / batch['tar_n_rays']).abs().mean()
