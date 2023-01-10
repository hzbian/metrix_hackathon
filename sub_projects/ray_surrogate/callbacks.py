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

        self.columns = ['epoch', 'pred', 'target']
        self.train_data = []
        self.train_data_cur = []
        self.val_data = defaultdict(list)
        self.val_data_cur = defaultdict(list)
        self.num_plots = num_plots
        self.overwrite_epoch = overwrite_epoch

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused: int = 0):
        if outputs["out"]:
            pred_supp, pred_weights = outputs["out"][0]
            tar_supp, tar_weights = outputs["out"][1]
            bs = pred_supp.shape[0]
            self.train_data_cur += [
                [trainer.current_epoch,
                 wandb.Image(self.plot_data(pred_supp[idx, ...], pred_weights[idx, ...])),
                 wandb.Image(self.plot_data(tar_supp[idx, ...], tar_weights[idx, ...]))]
                for idx in range(bs)
            ]

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        for dl_name, output in outputs["out"].items():
            if output:
                pred_supp, pred_weights = output[0]
                tar_supp, tar_weights = output[1]
                bs = pred_supp.shape[0]
                self.val_data_cur[dl_name] += [
                    [trainer.current_epoch,
                     wandb.Image(self.plot_data(pred_supp[idx, ...], pred_weights[idx, ...])),
                     wandb.Image(self.plot_data(tar_supp[idx, ...], tar_weights[idx, ...]))]
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

    def plot_data(self, pc_supp: torch.Tensor, pc_weights: torch.Tensor):
        pc_supp = pc_supp.detach().cpu()
        pc_weights = pc_weights.detach().cpu()

        fig = plt.figure()
        ax = fig.gca()
        ax.scatter(pc_supp[:, 0], pc_supp[:, 1], s=2.0, c=pc_weights)

        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        out = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        return out
