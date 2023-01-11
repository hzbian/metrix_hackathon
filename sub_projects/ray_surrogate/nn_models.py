from typing import Type, Dict, Any, List, Tuple
from collections import OrderedDict

import torch
from torch import nn
from pytorch_lightning import LightningModule


class SurrogateModel(LightningModule):

    def __init__(self,
                 dim_bottleneck: int,
                 net_bottleneck: Type[nn.Module],
                 net_bottleneck_params: Dict,
                 net_hist: Tuple[Type[nn.Module], Type[nn.Module]],
                 net_hist_params: Tuple[Dict, Dict],
                 net_lims: Tuple[Type[nn.Module], Type[nn.Module]],
                 net_lims_params: Tuple[Dict, Dict],
                 enc_params: Type[nn.Module],
                 enc_params_params: Dict,
                 loss_func: Type[nn.Module],
                 loss_func_params: Dict,
                 optimizer: Type[Any] = torch.optim.Adam,
                 optimizer_params: Dict = {"lr": 2e-4, "eps": 1e-5, "weight_decay": 1e-4},
                 scheduler: Type[Any] = torch.optim.lr_scheduler.StepLR,
                 scheduler_params: Dict = {"step_size": 1, "gamma": 1.0},
                 val_metrics: List[Tuple[str, nn.Module]] = None,
                 keep_batch: List[int] = [0],
                 ):
        super().__init__()
        self.save_hyperparameters()

        self.dim_bottleneck = dim_bottleneck

        self.net_bottleneck = net_bottleneck(**net_bottleneck_params)
        self.bn_net_bottleneck = nn.BatchNorm1d(self.dim_bottleneck)

        self.enc_hist = net_hist[0](**net_hist_params[0])
        self.dec_hist = net_hist[1](**net_hist_params[1])
        self.bn_hist = nn.BatchNorm1d(self.dim_bottleneck)

        self.enc_x_lims = net_lims[0](**net_lims_params[0])
        self.dec_x_lims = net_lims[1](**net_lims_params[1])
        self.bn_x_lims = nn.BatchNorm1d(self.dim_bottleneck)

        self.enc_y_lims = net_lims[0](**net_lims_params[0])
        self.dec_y_lims = net_lims[1](**net_lims_params[1])
        self.bn_y_lims = nn.BatchNorm1d(self.dim_bottleneck)

        self.enc_params = enc_params(**enc_params_params)
        self.bn_params = nn.BatchNorm1d(self.dim_bottleneck)

        self.loss_func = loss_func(**loss_func_params)
        self.val_metrics = val_metrics if val_metrics is not None else []

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        params = batch['params']
        hist = torch.zeros_like(batch['tar_hist'])
        x_lims = torch.zeros_like(batch['tar_x_lims'])
        y_lims = torch.zeros_like(batch['tar_y_lims'])

        params = self.bn_params(self.enc_params(params))
        hist = self.bn_hist(self.enc_hist(hist))
        x_lims = self.bn_x_lims(self.enc_x_lims(x_lims))
        y_lims = self.bn_y_lims(self.enc_y_lims(y_lims))

        bottleneck = sum([params, hist, x_lims, y_lims])
        bottleneck = self.bn_net_bottleneck(self.net_bottleneck(bottleneck))

        batch['pred_hist'] = torch.clamp_min(self.dec_hist(bottleneck), 0.0)
        batch['pred_x_lims'] = self.dec_x_lims(bottleneck)
        batch['pred_y_lims'] = self.dec_y_lims(bottleneck)
        batch['pred_n_rays'] = batch['pred_hist'].sum(dim=1)

        return batch

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        loss, batch = self._process_batch(batch)
        bs = batch['params'].shape[0]
        self.log("train/loss", loss.item(), batch_size=bs)

        if batch_idx in self.hparams.keep_batch:
            out = batch
        else:
            out = None
        return {"loss": loss, "out": out}

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        out = {}
        for dl_name, dl_batch in batch.items():
            loss, dl_batch = self._process_batch(dl_batch)
            bs = dl_batch['params'].shape[0]
            self.log(f"val/loss/{dl_name}", loss.item(), batch_size=bs)

            with torch.no_grad():
                for metric, metric_func in self.val_metrics:
                    self.log(f"val/metrics/{metric}_{dl_name}", metric_func(dl_batch).item(), batch_size=bs)

            if batch_idx in self.hparams.keep_batch:
                out[dl_name] = dl_batch
            else:
                out[dl_name] = None
        return {"out": out}

    def _process_batch(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch = self(batch)
        loss = self.loss_func(batch)
        return loss, batch

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(self.parameters(), **self.hparams.optimizer_params)
        scheduler = self.hparams.scheduler(optimizer, **self.hparams.scheduler_params)
        return [optimizer], [scheduler]


class MLP(nn.Module):

    def __init__(self,
                 dim_in: int, dim_out: int, dim_hidden: List[int],
                 activation: Type[Any] = nn.ReLU,
                 activation_params: Dict = dict(inplace=True)):
        super().__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.activation = activation(**activation_params)

        dims = [self.dim_in] + self.dim_hidden + [self.dim_out]
        self.depth = len(dims) - 1

        modules = []
        for idx in range(1, len(dims)):
            modules += [('layer' + str(idx), nn.Linear(dims[idx - 1], dims[idx], bias=True)),
                        ('act' + str(idx), self.activation)]
        modules.pop()  # no final activation
        self.net = nn.Sequential(OrderedDict(modules))

    def forward(self, x: torch.Tensor):
        return self.net(x.view(x.shape[0], -1))
