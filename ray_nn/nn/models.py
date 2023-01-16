from typing import Type, Dict, Any, List, Tuple

import torch
from torch import nn
from pytorch_lightning import LightningModule

from .backbones import TransformerBackbone
from ..metrics.geometric import SurrogateLoss


class SurrogateModel(LightningModule):

    def __init__(self,
                 planes: List[str],
                 backbone: Dict[str, Type[TransformerBackbone]],
                 backbone_params: Dict[str, Dict],
                 loss_func: Dict[str, Type[SurrogateLoss]],
                 loss_func_params: Dict[str, Dict],
                 use_prev_plane_pred: bool = True,
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
        self.val_metrics = val_metrics if val_metrics is not None else []

    def forward(self, batch: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Dict[str, torch.Tensor]]:
        for idx, plane in enumerate(self.planes):
            inp_params = batch[plane]['params']
            if idx == 0 or not self.use_prev_plane_pred:
                inp_hist = torch.zeros_like(batch[plane]['tar_hist'])
                inp_x_lims = torch.zeros_like(batch[plane]['tar_x_lims'])
                inp_y_lims = torch.zeros_like(batch[plane]['tar_y_lims'])
            else:
                inp_hist = batch[self.planes[idx - 1]]['pred_hist']
                inp_x_lims = batch[self.planes[idx - 1]]['pred_x_lims']
                inp_y_lims = batch[self.planes[idx - 1]]['pred_y_lims']

            pred_hist, pred_x_lims, pred_y_lims = self.backbone[plane](inp_params, inp_hist, inp_x_lims, inp_y_lims)

            pred_x_lims_lo = pred_x_lims.min(dim=-1)[0]
            pred_x_lims_hi = pred_x_lims.max(dim=-1)[0]
            pred_y_lims_lo = pred_y_lims.min(dim=-1)[0]
            pred_y_lims_hi = pred_y_lims.max(dim=-1)[0]

            batch[plane]['pred_hist'] = torch.clamp_min(pred_hist, 0.0)
            batch[plane]['pred_x_lims'] = torch.stack([pred_x_lims_lo, pred_x_lims_hi], dim=-1)
            batch[plane]['pred_y_lims'] = torch.stack([pred_y_lims_lo, pred_y_lims_hi], dim=-1)
            batch[plane]['pred_n_rays'] = batch[plane]['pred_hist'].sum(dim=[-2, -1])

        return batch

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        loss, batch = self._process_batch(batch)
        loss_total = torch.stack(list(loss.values())).sum()

        bs = batch[self.planes[0]]['params'].shape[0]

        self.log(f"train/loss", loss_total.item(), batch_size=bs)
        for plane in self.planes:
            self.log(f"train/loss/{plane}", loss[plane].item(), batch_size=bs)

        if batch_idx in self.hparams.keep_batch:
            out = batch
        else:
            out = None
        return {"loss": loss_total, "out": out}

    def validation_step(self, batch: Dict[str, Dict[str, torch.Tensor]], batch_idx: int):
        out = {}
        for dl_name, dl_batch in batch.items():
            loss, dl_batch = self._process_batch(dl_batch)
            loss_total = torch.stack(list(loss.values())).sum()

            bs = dl_batch[self.planes[0]]['params'].shape[0]
            self.log(f"val/loss/{dl_name}", loss_total.item(), batch_size=bs)
            for plane in self.planes:
                self.log(f"val/loss/{dl_name}/{plane}", loss[plane].item(), batch_size=bs)

            with torch.no_grad():
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
        for plane in self.planes:
            loss[plane] = self.loss_func[plane](batch[plane])
        return loss, batch

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(self.parameters(), **self.hparams.optimizer_params)
        scheduler = self.hparams.scheduler(optimizer, **self.hparams.scheduler_params)
        return [optimizer], [scheduler]
