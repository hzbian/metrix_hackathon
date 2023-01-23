from typing import List, Dict, Any

import psutil
from collections import defaultdict

import wandb

import numpy as np
import torch
from pytorch_lightning import Callback, Trainer, LightningModule
from pytorch_lightning.utilities.model_summary import ModelSummary

from matplotlib import pyplot as plt

from .models import SurrogateModel, unfreeze, freeze


class ImagePlaneCallback(Callback):
    """
    Callback to plot images with wandb.
    """

    def __init__(self, plane: str, num_plots: int = 5, overwrite_epoch: bool = True,
                 show_tar_lims_axis: bool = True, show_tar_hist_vminmax: bool = True):
        super().__init__()

        self.plane = plane
        self.num_plots = num_plots
        self.overwrite_epoch = overwrite_epoch
        self.show_tar_lims_axis = show_tar_lims_axis
        self.show_tar_hist_vminmax = show_tar_hist_vminmax

        self.columns = ['epoch', 'pred_hist', 'tar_hist', 'pred_n_rays', 'tar_n_rays']
        self.train_data = []
        self.train_data_cur = []
        self.val_data = defaultdict(list)
        self.val_data_cur = defaultdict(list)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused: int = 0):
        if outputs["out"]:
            batch = outputs["out"]
            batch = {k: v.detach().cpu() for k, v in batch[self.plane].items()}
            bs = min(batch['params'].shape[0], self.num_plots)
            self.train_data_cur += [
                [trainer.current_epoch,
                 wandb.Image(self.plot_data(hist=batch['pred_hist'][idx, 0, ...],
                                            x_lims=batch['pred_x_lims'][idx, 0, ...],
                                            y_lims=batch['pred_y_lims'][idx, 0, ...],
                                            x_lims_show=batch['tar_x_lims'][
                                                idx, 0, ...] if self.show_tar_lims_axis else None,
                                            y_lims_show=batch['tar_y_lims'][
                                                idx, 0, ...] if self.show_tar_lims_axis else None,
                                            hist_show=batch['tar_hist'][
                                                idx, 0, ...] if self.show_tar_hist_vminmax else None)),
                 wandb.Image(self.plot_data(hist=batch['tar_hist'][idx, 0, ...],
                                            x_lims=batch['tar_x_lims'][idx, 0, ...],
                                            y_lims=batch['tar_y_lims'][idx, 0, ...])),
                 batch['pred_n_rays'][idx, 0].item(), batch['tar_n_rays'][idx, 0].item()]
                for idx in range(bs)
            ]

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        for dl_name, dl_batch in outputs["out"].items():
            if dl_batch:
                dl_batch = {k: v.detach().cpu() for k, v in dl_batch[self.plane].items()}
                bs = min(dl_batch['params'].shape[0], self.num_plots)
                self.val_data_cur[dl_name] += [
                    [trainer.current_epoch,
                     wandb.Image(self.plot_data(hist=dl_batch['pred_hist'][idx, 0, ...],
                                                x_lims=dl_batch['pred_x_lims'][idx, 0, ...],
                                                y_lims=dl_batch['pred_y_lims'][idx, 0, ...],
                                                x_lims_show=dl_batch['tar_x_lims'][
                                                    idx, 0, ...] if self.show_tar_lims_axis else None,
                                                y_lims_show=dl_batch['tar_y_lims'][
                                                    idx, 0, ...] if self.show_tar_lims_axis else None,
                                                hist_show=dl_batch['tar_hist'][
                                                    idx, 0, ...] if self.show_tar_hist_vminmax else None)),
                     wandb.Image(self.plot_data(hist=dl_batch['tar_hist'][idx, 0, ...],
                                                x_lims=dl_batch['tar_x_lims'][idx, 0, ...],
                                                y_lims=dl_batch['tar_y_lims'][idx, 0, ...])),
                     dl_batch['pred_n_rays'][idx, 0].item(), dl_batch['tar_n_rays'][idx, 0].item()]
                    for idx in range(bs)
                ]

    def on_train_epoch_end(self, trainer, pl_module):
        if self.train_data_cur:
            self.train_data = self.train_data_cur[:self.num_plots] + self.train_data

        if self.train_data:
            trainer.logger.log_table(key=f'{self.plane} (train)', columns=self.columns, data=self.train_data)

        self.train_data_cur = []
        if self.overwrite_epoch:
            self.train_data = []

    def on_validation_epoch_end(self, trainer, pl_module):
        for dl_name in self.val_data_cur.keys():
            if self.val_data_cur[dl_name]:
                self.val_data[dl_name] = self.val_data_cur[dl_name][:self.num_plots] + self.val_data[dl_name]

            if self.val_data[dl_name]:
                trainer.logger.log_table(key=f'{self.plane} ({dl_name})',
                                         columns=self.columns, data=self.val_data[dl_name])

            self.val_data_cur[dl_name] = []
            if self.overwrite_epoch:
                self.val_data[dl_name] = []

    def plot_data(self,
                  hist: torch.Tensor,
                  x_lims: torch.Tensor,
                  y_lims: torch.Tensor,
                  x_lims_show: torch.Tensor = None,  # x-limits to be used to set xlim in axis
                  y_lims_show: torch.Tensor = None,  # y-limits to be used to set ylim in axis
                  hist_show: torch.Tensor = None,  # Histogram to be used to set vmin and vmax in imshow
                  ):
        fig = plt.figure()
        ax = fig.gca()

        if hist_show is not None:
            vmin = torch.min(hist_show).item()
            vmax = torch.max(hist_show).item()
        else:
            vmin = None
            vmax = None

        im = ax.imshow(hist.T, cmap='turbo', interpolation='none',
                       extent=[x_lims[0].item(), x_lims[1].item() + 1e-8, y_lims[0].item(), y_lims[1].item() + 1e-8],
                       origin='lower', aspect='auto',
                       vmin=vmin, vmax=vmax)
        fig.colorbar(im)

        if x_lims_show is not None and y_lims is not None:
            ax.set_xlim([x_lims_show[0].item(), x_lims_show[1].item() + 1e-8])
            ax.set_ylim([y_lims_show[0].item(), y_lims_show[1].item() + 1e-8])

        fig.tight_layout()

        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        out = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        return out


class PlaneMutator(Callback):

    def __init__(self, mutator_planes: List[List[str]], mutator_epochs: List[int]):
        self.mutator_planes = mutator_planes
        self.mutator_epochs = mutator_epochs

    def on_train_epoch_start(self, trainer: Trainer, pl_module: SurrogateModel) -> None:
        planes_ = self.mutator_planes[0]
        for idx, planes in enumerate(self.mutator_planes):
            if trainer.current_epoch >= self.mutator_epochs[idx]:
                planes_ = planes
        pl_module.planes = planes_
        print('Currently used planes:', pl_module.planes)


class HistNRaysAlternator(Callback):

    def __init__(self, every_epoch: int, start_with_hist_fit: bool = True):
        self.every_epoch = every_epoch
        self.start_with_hist_fit = start_with_hist_fit

    def on_train_epoch_start(self, trainer: Trainer, pl_module: SurrogateModel) -> None:
        if (self.start_with_hist_fit and trainer.current_epoch % (2 * self.every_epoch) < self.every_epoch) or \
                (not self.start_with_hist_fit and trainer.current_epoch % (2 * self.every_epoch) >= self.every_epoch):
            pl_module.unfreeze()
            for plane in pl_module.planes:
                freeze(pl_module.n_rays_predictor[plane])
                pl_module.loss_func[plane].sinkhorn_normalize = True
                pl_module.loss_func[plane].loss_sinkhorn.normalize_weights = True
                pl_module.loss_func[plane].loss_sinkhorn.hist_zero_loss_weight = 1.0
            print('Optimizing (normalized) histograms with fixed number of rays...')
        else:
            pl_module.freeze()
            for plane in pl_module.planes:
                unfreeze(pl_module.n_rays_predictor[plane])
                pl_module.loss_func[plane].sinkhorn_normalize = 'weights2'
                pl_module.loss_func[plane].loss_sinkhorn.normalize_weights = 'weights2'
                pl_module.loss_func[plane].loss_sinkhorn.hist_zero_loss_weight = 0.0
            print('Optimizing number of rays with fixed histograms...')

        for optim in trainer.optimizers:
            optim.param_groups.clear()
            optim.state.clear()
            optim.add_param_group({'params': [p for p in pl_module.parameters() if p.requires_grad is True]})

        print(ModelSummary(pl_module))

    def on_load_checkpoint(self, trainer: Trainer, pl_module: SurrogateModel,
                           checkpoint: Dict[str, Any]) -> None:
        # Hack to reset (possibly) non-matching optimizer states loaded from a checkpoint
        optim = pl_module.configure_optimizers()[0][0]
        checkpoint['optimizer_states'][0]['state'].clear()
        checkpoint['optimizer_states'][0]['param_groups'] = optim.param_groups


class MemoryMonitor(Callback):

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # trainer.train_dataloader.reset()
        # trainer.val_dataloaders[0].reset()
        # print('Dataloaders reset...')
        print('RAM used before train epoch (GB):', psutil.virtual_memory()[3] / (1024 * 1024 * 1024))

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        print('RAM used after train epoch (GB):', psutil.virtual_memory()[3] / (1024 * 1024 * 1024))

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        print('RAM used after val epoch (GB):', psutil.virtual_memory()[3] / (1024 * 1024 * 1024))
