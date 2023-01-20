from typing import Type, Dict, Any, List, Tuple

import torch
from torch import nn
from pytorch_lightning import LightningModule

from .backbones import SurrogateBackbone
from ..metrics.geometric import SurrogateLoss


def freeze(model: nn.Module) -> None:
    """
    Set required_grad of all parameters in ``module`` to False.
    """
    for p in model.parameters():
        p.requires_grad = False


def unfreeze(model: nn.Module) -> None:
    """
    Set required_grad of all parameters in ``module`` to True.
    """
    for p in model.parameters():
        p.requires_grad = True


class SurrogateModel(LightningModule):
    """
    Main PyTorch LightningModule to represent a surrogate model.
    :param planes: List of (image) planes to be considered. The order of the list determines the order of
        beamline components.
    :param backbone: Dictionary of :class:`ray_nn.nn.backbones.SurrogateBackbone` (keys = planes)
    :param backbone_params:
    :param loss_func: Dictionary of :class:`ray_nn.metrics.geometric.SurrogateLoss` (keys = planes)
    :param loss_func_params:
    :param hist_zero_classifier: Dictionary of classifiers to predict of a histogram is empty or not (keys = planes).
        Input is [batch_size, #inp_params] and output [batch_size, 1] (no distinction between different histogram
        layers in a plane).
    :param hist_zero_classifier_params:
    :param n_rays_predictor: Dictionary of nets to predict the number of rays (keys = planes).
        Input is [batch_size, #inp_params] and output [batch_size, #hist_layers], where #hist_layers is the number
        of histograms for this plane.
    :param n_rays_predictor_params:
    :param use_prev_plane_pred: If True, the outputs of the previous plane are used as inputs for the next one.
        If False, random histograms (random noise) are used as input.
    :param n_rays_known: If True, the output histograms are rescaled such that their number of rays is tar_n_rays.
        Note that pred_n_rays is still not overwritten.
    :param optimizer:
    :param optimizer_params:
    :param scheduler:
    :param scheduler_params:
    :param val_metrics: Tuples of validation metrics (name, nn.Module) that are computed in each training and
        validation step.
    :param keep_batch: List of batches to keep until the end of epoch (e.g., for plotting callbacks)
    """

    def __init__(self,
                 planes: List[str],
                 backbone: Dict[str, Type[SurrogateBackbone]],
                 backbone_params: Dict[str, Dict],
                 loss_func: Dict[str, Type[SurrogateLoss]],
                 loss_func_params: Dict[str, Dict],
                 hist_zero_classifier: Dict[str, Type[nn.Module]] = None,
                 hist_zero_classifier_params: Dict[str, Dict] = None,
                 n_rays_predictor: Dict[str, Type[nn.Module]] = None,
                 n_rays_predictor_params: Dict[str, Dict] = None,
                 use_prev_plane_pred: bool = True,
                 n_rays_known: bool = False,
                 optimizer: Type[Any] = torch.optim.Adam,
                 optimizer_params: Dict = {"lr": 2e-4, "eps": 1e-5, "weight_decay": 1e-4},
                 scheduler: Type[Any] = torch.optim.lr_scheduler.StepLR,
                 scheduler_params: Dict = {"step_size": 1, "gamma": 1.0},
                 val_metrics: List[Tuple[str, nn.Module]] = None,
                 keep_batch: List[int] = [0],
                 ):
        super().__init__()
        self.save_hyperparameters()

        self.planes = planes
        self.backbone = nn.ModuleDict({plane: backbone[plane](**backbone_params[plane])
                                       for plane in self.planes})
        self.loss_func = nn.ModuleDict({plane: loss_func[plane](**loss_func_params[plane])
                                        for plane in self.planes})
        self.use_prev_plane_pred = use_prev_plane_pred
        self.n_rays_known = n_rays_known
        self.val_metrics = val_metrics if val_metrics is not None else []

        if hist_zero_classifier is not None:
            self.hist_zero_classifier = nn.ModuleDict(
                {plane: hist_zero_classifier[plane](**hist_zero_classifier_params[plane])
                 for plane in self.planes})
        else:
            self.hist_zero_classifier = None

        if n_rays_predictor is not None:
            self.n_rays_predictor = nn.ModuleDict(
                {plane: n_rays_predictor[plane](**n_rays_predictor_params[plane])
                 for plane in self.planes})
        else:
            self.n_rays_predictor = None

    def forward(self, batch: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Dict[str, torch.Tensor]]:
        # iterate over plane in the right order to write prediction to batch
        for idx, plane in enumerate(self.planes):
            inp_params = batch[plane]['params']
            if idx == 0 or not self.use_prev_plane_pred:
                # use random histograms and limits as input
                inp_hist = torch.randn_like(batch[plane]['tar_hist'])
                inp_x_lims = torch.randn_like(batch[plane]['tar_x_lims'])
                inp_y_lims = torch.randn_like(batch[plane]['tar_y_lims'])
            else:
                # use output of previous plane as input
                inp_hist = batch[self.planes[idx - 1]]['pred_hist']
                inp_x_lims = batch[self.planes[idx - 1]]['pred_x_lims']
                inp_y_lims = batch[self.planes[idx - 1]]['pred_y_lims']

            # apply backbone
            pred_hist, pred_x_lims, pred_y_lims = self.backbone[plane](inp_params, inp_hist, inp_x_lims, inp_y_lims)

            # a histogram must be non-negative
            batch[plane]['pred_hist'] = pred_hist.abs()

            # ensure that the predicted limits are in the right order (0 = lower, 1 = upper)
            pred_x_lims_lo = pred_x_lims.min(dim=-1)[0]
            pred_x_lims_hi = pred_x_lims.max(dim=-1)[0]
            pred_y_lims_lo = pred_y_lims.min(dim=-1)[0]
            pred_y_lims_hi = pred_y_lims.max(dim=-1)[0]
            batch[plane]['pred_x_lims'] = torch.stack([pred_x_lims_lo, pred_x_lims_hi], dim=-1)
            batch[plane]['pred_y_lims'] = torch.stack([pred_y_lims_lo, pred_y_lims_hi], dim=-1)

            if self.n_rays_predictor is not None:
                # use n_rays_predictor to predict the number if rays and rescale histogram accordingly
                pred_n_rays = self.n_rays_predictor[plane](inp_params).abs()
                batch[plane]['pred_hist'] = batch[plane]['pred_hist'] / \
                                            batch[plane]['pred_hist'].sum(dim=(-1, -2), keepdim=True) * \
                                            pred_n_rays.clone().view(inp_params.shape[0], -1, 1, 1)
                batch[plane]['pred_n_rays'] = pred_n_rays
            else:
                # compute number if rays from histogram
                batch[plane]['pred_n_rays'] = batch[plane]['pred_hist'].sum(dim=[-2, -1])

            if self.hist_zero_classifier is not None:
                # compute "probability" of histogram being empty
                pred_hist_zero_prob = torch.sigmoid(self.hist_zero_classifier[plane](inp_params)).flatten(start_dim=1)
                batch[plane]['pred_hist_zero_prob'] = pred_hist_zero_prob
                hist_zero_idx = pred_hist_zero_prob > 0.5  # decision threshold

                # generate empty histograms and insert them where needed
                hist_0, x_lims_0, y_lims_0, n_rays_0 = SurrogateModel._hist_zero_batch(batch[plane]['pred_hist'])
                batch[plane]['pred_hist'][hist_zero_idx, ...] = hist_0[hist_zero_idx, ...]
                batch[plane]['pred_x_lims'][hist_zero_idx, ...] = x_lims_0[hist_zero_idx, ...]
                batch[plane]['pred_y_lims'][hist_zero_idx, ...] = y_lims_0[hist_zero_idx, ...]
                batch[plane]['pred_n_rays'][hist_zero_idx, ...] = n_rays_0[hist_zero_idx, ...]

            if self.n_rays_known:
                # rescale histogram such that number of rays is equal to tar_n_rays (pred_n_rays is NOT overwritten)
                batch[plane]['pred_hist'] = batch[plane]['pred_hist'] / \
                                            batch[plane]['pred_hist'].sum(dim=(-1, -2), keepdim=True) * \
                                            batch[plane]['tar_n_rays'].view(inp_params.shape[0], -1, 1, 1)

        return batch

    @staticmethod
    def _hist_zero_batch(hist_like: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        See :func:`ray_nn.data.transform.SurrogateModelPreparation._process_zero_hist`
        """
        hist = torch.ones_like(hist_like).abs()
        hist = hist / hist.sum(dim=[-2, -1], keepdim=True)
        x_lims = torch.ones(hist_like.shape[0], hist_like.shape[1], 2, device=hist_like.device)
        y_lims = torch.ones(hist_like.shape[0], hist_like.shape[1], 2, device=hist_like.device)
        x_lims[:, :, 0] = y_lims[:, :, 0] = -1e-4
        x_lims[:, :, 1] = y_lims[:, :, 1] = 1e-4
        n_rays = hist.sum(dim=[-2, -1])
        return hist, x_lims, y_lims, n_rays

    def freeze(self, planes: List[str] = None):
        """
        Set required_grad of all components in planes to False.
        """
        planes = self.planes if planes is None else planes
        for plane in planes:
            freeze(self.backbone[plane])
            freeze(self.loss_func[plane])
            if self.hist_zero_classifier is not None:
                freeze(self.hist_zero_classifier[plane])
            if self.n_rays_predictor is not None:
                freeze(self.n_rays_predictor[plane])

    def unfreeze(self, planes: List[str] = None):
        """
        Set required_grad of all components in planes to True.
        """
        planes = self.planes if planes is None else [planes]
        for plane in planes:
            unfreeze(self.backbone[plane])
            unfreeze(self.loss_func[plane])
            if self.hist_zero_classifier is not None:
                unfreeze(self.hist_zero_classifier[plane])
            if self.n_rays_predictor is not None:
                unfreeze(self.n_rays_predictor[plane])

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        loss, batch = self._process_batch(batch)
        # total loss is sum over losses for all planes
        loss_total = torch.stack(list(loss.values())).sum()
        # batch size
        bs = batch[self.planes[0]]['params'].shape[0]
        # log total loss and all individual losses
        self.log(f"train/loss", loss_total.item(), batch_size=bs)
        for plane in self.planes:
            self.log(f"train/loss/{plane}", loss[plane].item(), batch_size=bs)

        # log all validation metrics
        with torch.no_grad():
            for metric, metric_func in self.val_metrics:
                for plane in self.planes:
                    self.log(f"train/metrics/{metric}/{plane}",
                             metric_func(batch[plane]).item(), batch_size=bs)

        if batch_idx in self.hparams.keep_batch:
            out = batch
        else:
            out = None
        return {"loss": loss_total, "out": out}

    def validation_step(self, batch: Dict[str, Dict[str, torch.Tensor]], batch_idx: int):
        with torch.no_grad():
            out = {}
            for dl_name, dl_batch in batch.items():
                loss, dl_batch = self._process_batch(dl_batch)
                # total loss is sum over losses for all planes
                loss_total = torch.stack(list(loss.values())).sum()
                # batch size
                bs = dl_batch[self.planes[0]]['params'].shape[0]
                # log total loss and all individual losses
                self.log(f"val/loss/{dl_name}", loss_total.item(), batch_size=bs)
                for plane in self.planes:
                    self.log(f"val/loss/{dl_name}/{plane}", loss[plane].item(), batch_size=bs)

                # log all validation metrics
                for metric, metric_func in self.val_metrics:
                    for plane in self.planes:
                        self.log(f"val/metrics/{dl_name}/{metric}/{plane}",
                                 metric_func(dl_batch[plane]).item(), batch_size=bs)

            if batch_idx in self.hparams.keep_batch:
                out[dl_name] = dl_batch
            else:
                out[dl_name] = None
        return {"out": out}

    def _process_batch(self, batch: Dict[str, Dict[str, torch.Tensor]]) -> Tuple[Dict, Dict]:
        batch = self(batch)
        loss = {}
        # compute loss for all planes
        for plane in self.planes:
            loss[plane] = self.loss_func[plane](batch[plane])
        return loss, batch

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(self.parameters(), **self.hparams.optimizer_params)
        scheduler = self.hparams.scheduler(optimizer, **self.hparams.scheduler_params)
        return [optimizer], [scheduler]
