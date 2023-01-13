import os, psutil
from collections import defaultdict
from humanize import naturalsize
import gc

import wandb

import numpy as np
import torch
from pytorch_lightning import Callback

from matplotlib import pyplot as plt


class LogPredictionsCallback(Callback):
    """
    Callback to plot images with wandb.
    """

    def __init__(self, num_plots: int = 5, overwrite_epoch: bool = True):
        super().__init__()

        self.columns = ['epoch', 'pred_hist', 'tar_hist', 'pred_n_rays', 'tar_n_rays']
        self.train_data = []
        self.train_data_cur = []
        self.val_data = defaultdict(list)
        self.val_data_cur = defaultdict(list)
        self.num_plots = num_plots
        self.overwrite_epoch = overwrite_epoch

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused: int = 0):
        if outputs["out"]:
            batch = outputs["out"]
            batch = {k: v.detach().cpu() for k, v in batch.items()}
            bs = batch['params'].shape[0]
            self.train_data_cur += [
                [trainer.current_epoch,
                 wandb.Image(self.plot_data(hist=batch['pred_hist'][idx, ...],
                                            x_lims=batch['pred_x_lims'][idx, ...],
                                            y_lims=batch['pred_y_lims'][idx, ...],
                                            x_lims_show=batch['tar_x_lims'][idx, ...],
                                            y_lims_show=batch['tar_y_lims'][idx, ...],
                                            hist_show=batch['tar_hist'][idx, ...])),
                 wandb.Image(self.plot_data(hist=batch['tar_hist'][idx, ...],
                                            x_lims=batch['tar_x_lims'][idx, ...],
                                            y_lims=batch['tar_y_lims'][idx, ...])),
                 batch['pred_n_rays'][idx].item(), batch['tar_n_rays'][idx].item()]
                for idx in range(bs)
            ]

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        for dl_name, dl_batch in outputs["out"].items():
            if dl_batch:
                dl_batch = {k: v.detach().cpu() for k, v in dl_batch.items()}
                bs = dl_batch['params'].shape[0]
                self.val_data_cur[dl_name] += [
                    [trainer.current_epoch,
                     wandb.Image(self.plot_data(hist=dl_batch['pred_hist'][idx, ...],
                                                x_lims=dl_batch['pred_x_lims'][idx, ...],
                                                y_lims=dl_batch['pred_y_lims'][idx, ...],
                                                x_lims_show=dl_batch['tar_x_lims'][idx, ...],
                                                y_lims_show=dl_batch['tar_y_lims'][idx, ...],
                                                hist_show=dl_batch['tar_hist'][idx, ...])),
                     wandb.Image(self.plot_data(hist=dl_batch['tar_hist'][idx, ...],
                                                x_lims=dl_batch['tar_x_lims'][idx, ...],
                                                y_lims=dl_batch['tar_y_lims'][idx, ...])),
                     dl_batch['pred_n_rays'][idx].item(), dl_batch['tar_n_rays'][idx].item()]
                    for idx in range(bs)
                ]

    def on_train_epoch_end(self, trainer, pl_module):
        if self.train_data_cur:
            self.train_data = self.train_data_cur[:self.num_plots] + self.train_data

        if self.train_data:
            trainer.logger.log_table(key='prediction_table (train)', columns=self.columns, data=self.train_data)

        self.train_data_cur = []
        if self.overwrite_epoch:
            self.train_data = []

    def on_validation_epoch_end(self, trainer, pl_module):
        for dl_name in self.val_data_cur.keys():
            if self.val_data_cur[dl_name]:
                self.val_data[dl_name] = self.val_data_cur[dl_name][:self.num_plots] + self.val_data[dl_name]

            if self.val_data[dl_name]:
                trainer.logger.log_table(key='prediction_table (' + dl_name + ')',
                                         columns=self.columns, data=self.val_data[dl_name])

            self.val_data_cur[dl_name] = []
            if self.overwrite_epoch:
                self.val_data[dl_name] = []

    def plot_data(self, hist: torch.Tensor, x_lims: torch.Tensor, y_lims: torch.Tensor,
                  x_lims_show: torch.Tensor = None, y_lims_show: torch.Tensor = None,
                  hist_show: torch.Tensor = None):
        fig = plt.figure()
        ax = fig.gca()
        dim_x = dim_y = int(np.sqrt(hist.shape[0]))
        if hist_show is not None:
            vmin = torch.min(hist_show).item()
            vmax = torch.max(hist_show).item()
        else:
            vmin = None
            vmax = None

        im = ax.imshow(hist.view(dim_x, dim_y).T, cmap='turbo', interpolation='none',
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


class MemoryManagementCallback(Callback):

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        current_process = psutil.Process(os.getpid())
        mem = current_process.memory_full_info().uss
        for child in current_process.children(recursive=True):
            mem += child.memory_full_info().uss
        print(naturalsize(mem / 1024, format="%.3f", gnu=True))

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        gc.collect()
        print('Garbage collected...')
