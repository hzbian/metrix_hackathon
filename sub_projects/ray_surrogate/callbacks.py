from collections import defaultdict

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
                                            y_lims=batch['pred_y_lims'][idx, ...])),
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
                                                y_lims=dl_batch['pred_y_lims'][idx, ...])),
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

    def plot_data(self, hist: torch.Tensor, x_lims: torch.Tensor, y_lims: torch.Tensor):
        fig = plt.figure()
        ax = fig.gca()
        dim_x = dim_y = int(np.sqrt(hist.shape[0]))
        im = ax.imshow(hist.view(dim_x, dim_y).T, cmap='turbo', interpolation='none',
                       extent=[x_lims[0].item(), x_lims[1].item() + 1e-8, y_lims[0].item(), y_lims[1].item() + 1e-8],
                       origin='lower', aspect='auto')
        fig.colorbar(im)
        fig.tight_layout()

        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        out = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        return out
