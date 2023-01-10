from typing import Type, Dict, Any, List, Tuple
from collections import OrderedDict

import torch
from torch import nn
from pytorch_lightning import LightningModule

from ray_nn.utils.ray_processing import HistSubsampler, HistToPointCloud


class SurrogateModel(LightningModule):

    def __init__(self,
                 base_net: Type[nn.Module],
                 base_net_params: Dict,
                 param_preprocessor: Type[nn.Module],
                 param_preprocessor_params: Dict,
                 hist_subsampler: Type[HistSubsampler],
                 hist_subsampler_params: Dict,
                 hist_to_pc: Type[HistToPointCloud],
                 hist_to_pc_params: Dict,
                 param_dim: int,
                 pc_supp_dim: int,
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
        self.base_net = base_net(**base_net_params)
        self.param_preprocessor = param_preprocessor(**param_preprocessor_params)
        self.hist_subsampler = hist_subsampler(**hist_subsampler_params)
        self.hist_to_pc = hist_to_pc(**hist_to_pc_params)
        self.param_dim = param_dim
        self.pc_supp_dim = pc_supp_dim

        self.loss_func = loss_func(**loss_func_params)
        self.val_metrics = val_metrics if val_metrics is not None else []

        self._param_batch_norm = nn.BatchNorm1d(self.param_dim)

    def forward(self, params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bs = params.shape[0]
        params_emb = self.param_preprocessor(self._param_batch_norm(params))
        pc_inp_supp = torch.zeros(bs, 2 * self.pc_supp_dim, device=params.device)
        pc_inp_weight = torch.zeros(bs, self.pc_supp_dim, device=params.device)
        out = self.base_net(torch.cat([pc_inp_supp, pc_inp_weight, params_emb], dim=1))
        pc_pred_supp = out[:, :2 * self.pc_supp_dim].view(bs, self.pc_supp_dim, 2)
        pc_pred_weight = out[:, 2 * self.pc_supp_dim:]
        return pc_pred_supp.contiguous(), pc_pred_weight.contiguous()

    def training_step(self, batch, batch_idx):
        loss, pred, tar = self._process_batch(batch)
        bs = pred[0].shape[0]
        self.log("train/loss", loss.item(), batch_size=bs)

        if batch_idx in self.hparams.keep_batch:
            out = (pred, tar)
        else:
            out = None
        return {"loss": loss, "out": out}

    def validation_step(self, batch, batch_idx):
        out = {}
        for dl_name, dl_batch in batch.items():
            loss, pred, tar = self._process_batch(dl_batch)
            bs = pred[0].shape[0]
            self.log(f"val/loss/{dl_name}", loss.item(), batch_size=bs)

            with torch.no_grad():
                for metric, metric_func in self.val_metrics:
                    self.log(f"val/metrics/{metric}_{dl_name}",
                             metric_func(pred[0], tar[0], pred[1], tar[1]),
                             batch_size=bs)

            if batch_idx in self.hparams.keep_batch:
                out[dl_name] = (pred, tar)
            else:
                out[dl_name] = None
        return {"out": out}

    def _process_batch(self, batch: Tuple[torch.Tensor, ...]) -> Any:
        params, histogram, x_lim, y_lim, _ = batch
        tar_supp, tar_weights = self.hist_to_pc(self.hist_subsampler(histogram),
                                                x_lim, y_lim)
        pred_supp, pred_weights = self(params)
        loss = self.loss_func(pred_supp, tar_supp, pred_weights, tar_weights)
        return loss, (pred_supp, pred_weights), (tar_supp, tar_weights)

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
