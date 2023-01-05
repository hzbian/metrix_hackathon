import os
import sys
import math

import torch
from torch import optim
import pytorch_lightning as pl
import torch.nn as nn
from pytorch_lightning.loggers import WandbLogger

from ray_tools.simulation.torch_datasets import RayDataset, MemoryDataset

sys.path.insert(0, '../')
from ray_nn.data.lightning_data_module import DefaultDataModule
from ray_nn.data.transform import Select

# Important fix to make custom collate_fn work
# https://forums.fast.ai/t/runtimeerror-received-0-items-of-ancdata/48935
torch.multiprocessing.set_sharing_strategy('file_system')

h5_path = os.path.join('../datasets/metrix_simulation/ray_enhance')
h5_files: list[str] = [os.path.join(h5_path, file) for file in os.listdir(h5_path) if file.endswith('.h5')]


def create_sequential(input_length, output_length, layer_size, blow=0, shrink_factor="log"):
    layers = [input_length]
    blow_disabled = blow == 1 or blow == 0
    if not blow_disabled:
        layers.append(input_length * blow)

    if shrink_factor == "log":
        add_layers = torch.logspace(math.log(layers[-1], 10), math.log(output_length, 10),
                                    steps=layer_size + 2 - len(layers), base=10).long()
        # make sure the last element is correct, even though rounding
        add_layers[-1] = output_length
    elif shrink_factor == "lin":
        add_layers = torch.linspace(layers[-1], output_length, steps=layer_size + 2 - len(layers)).long()
    else:
        shrink_factor = float(shrink_factor)
        new_length = layer_size + 1 - len(layers)
        add_layers = (torch.ones(new_length) * layers[-1] * (
                (torch.ones(new_length) * shrink_factor) ** torch.arange(new_length))).long()
        layers = torch.cat((torch.tensor([input_length]), add_layers))
        layers = torch.cat((layers, torch.tensor([output_length])))

    if not blow_disabled:
        layers = torch.tensor([layers[0]])
        layers = torch.cat((layers, add_layers))
    else:
        layers = add_layers

    nn_layers = []
    for i in range(len(layers) - 1):
        nn_layers.append(nn.Linear(layers[i].item(), layers[i + 1].item()))
        if not i == len(layers) - 2:
            nn_layers.append(nn.ReLU())
            nn_layers.append(nn.BatchNorm1d(layers[i + 1].item()))
    return nn.Sequential(*nn_layers)


class MetrixRayCountPredictor(pl.LightningModule):
    def __init__(self, layer_size: int = 5, blow: float = 100., shrink_factor: str = 'log',
                 learning_rate: float = 0.001, gpus: int = 0, optimizer: str = 'adam'):
        super().__init__()
        self.save_hyperparameters()

        self.net = create_sequential(35, 1, self.hparams.layer_size, blow=self.hparams.blow,
                                     shrink_factor=self.hparams.shrink_factor)
        print(self.net)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.MSELoss()(y, y_hat)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        val_loss = nn.MSELoss()(y_hat, y)
        return {'s_val_loss': val_loss, 'y': y, 'y_hat': y_hat, 'x': x}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x['s_val_loss'] for x in outputs]).mean()
        self.log('val_loss', val_loss)

        # y = torch.cat([x['y'] for x in outputs])
        # y_hat = torch.cat([x['y_hat'] for x in outputs])
        # sample_idx = torch.cat([x['idx'] for x in outputs])

    def configure_optimizers(self):
        if self.hparams.optimizer == 'adam':
            return [torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)]
        elif self.hparams.optimizer == 'sgd':
            return [optim.SGD(self.parameters(), lr=self.hparams.learning_rate, momentum=0.9)]


sub_groups = ['1e6/params',
              '1e6/ray_output/ImagePlane/n_rays']

transform = Select(sub_groups)
dataset = RayDataset(h5_files=h5_files, sub_groups=sub_groups, transform=transform)
dataset = MemoryDataset(dataset)
datamodule = DefaultDataModule(dataset=dataset)
model = MetrixRayCountPredictor()

wandb_logger = WandbLogger()
trainer = pl.Trainer(max_epochs=-1, accelerator="auto", logger=wandb_logger)
trainer.fit(model, datamodule=datamodule)
