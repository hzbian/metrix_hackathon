import math
import torch
from torch import optim, nn, utils
import lightning as L
from lightning.pytorch.loggers import WandbLogger
import matplotlib.pyplot as plt
import wandb

from datasets.metrix_simulation.config_ray_emergency_surrogate import PARAM_CONTAINER_FUNC as params
from ray_nn.data.lightning_data_module import DefaultDataModule
from ray_tools.simulation.torch_datasets import MemoryDataset, RayDataset
from ray_nn.data.transform import Select

# define the LightningModule
class MetrixXYHistSurrogate(L.LightningModule):
    def __init__(self, layer_size:int=2, blow:int=0, shrink_factor:str='log', learning_rate:float=0.001, gpus:int=0, optimizer:str='adam', autoencoder_checkpoint:str='../../data/ae_lrelu_epoch=55-step=52527.ckpt'):
        super(MetrixXYHistSurrogate, self).__init__()
        self.save_hyperparameters()

        self.net = self.create_sequential(34, 100, self.hparams.layer_size, blow=self.hparams.blow, shrink_factor=self.hparams.shrink_factor)
        self.val_loss = []
        self.val_nonempty_loss = []
        self.train_loss = []
        print(self.net)

    def create_sequential(self, input_length, output_length, layer_size, blow=0, shrink_factor="log"):
        layers = [input_length]
        blow_disabled = blow == 1 or blow == 0
        if not blow_disabled:
            layers.append(input_length*blow)

        if shrink_factor == "log":
            add_layers = torch.logspace(math.log(layers[-1], 10), math.log(output_length,10), steps=layer_size+2-len(layers), base=10).long()
            # make sure the first and last element is correct, even though rounding
            add_layers[0] = input_length
            add_layers[-1] = output_length
        elif shrink_factor == "lin":
            add_layers = torch.linspace(layers[-1], output_length, steps=layer_size+2-len(layers)).long()
        else:
            shrink_factor = float(shrink_factor)
            new_length = layer_size+1-len(layers)
            add_layers = (torch.ones(new_length)*layers[-1] * ((torch.ones(new_length) * shrink_factor) ** torch.arange(new_length))).long()
            layers = torch.cat((torch.tensor([input_length]), add_layers))
            layers = torch.cat((layers, torch.tensor([output_length])))
    
        if not blow_disabled:
            layers = torch.tensor([layers[0]])
            layers = torch.cat((layers, add_layers))
        else:
           layers = add_layers

        nn_layers = []
        for i in range(len(layers)-1):
            nn_layers.append(nn.Linear(layers[i].item(), layers[i+1].item()))
            if not i == len(layers)-2:
                nn_layers.append(nn.ReLU())
                #nn_layers.append(nn.BatchNorm1d(layers[i+1].item()))
        return nn.Sequential(*nn_layers)

    def training_step(self, batch, _):
        x, y = batch
        y = y.flatten(start_dim=1)
        y_hat = self.net(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.train_loss.append(loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.flatten(start_dim=1)
        y_hat = self.net(x)
        y = y.cpu()
        y_hat = y_hat.cpu()
        nonempty_mask = y.mean(dim=1) != 0.
        y_nonempty = y[nonempty_mask]
        y_hat_nonempty = y_hat[nonempty_mask]
        if nonempty_mask.sum() > 0. and batch_idx == 0:
            _, ax = plt.subplots(len(y_nonempty), 2, squeeze=False)
            for i, y_element in enumerate(y_nonempty[:5]):
                ax[i, 0].plot(y_element[50:], label='gt')
                ax[i, 0].plot(y_hat_nonempty[i, 50:], label='prediction')
                ax[i, 1].plot(y_element[:50], label='gt')
                ax[i, 1].plot(y_hat_nonempty[i, :50], label='prediction')
            ax[y_nonempty.shape[0]-1, 0].set_xlabel('histogram_x')
            ax[y_nonempty.shape[0]-1, 1].set_xlabel('histogram_y')
            plt.tight_layout()
            plt.legend()
            wandb.log({"xy_hist_plots": wandb.Image(plt)})
            nonempty_loss = nn.functional.mse_loss(y_hat_nonempty, y_nonempty)
            self.val_nonempty_loss.append(nonempty_loss)
        val_loss = nn.functional.mse_loss(y_hat, y)
        self.val_loss.append(val_loss)
        return val_loss

    def on_train_epoch_end(self):
        train_loss = torch.stack(self.train_loss).mean().item()
        self.log("train_loss", train_loss)
        self.train_loss.clear()

    def on_validation_epoch_end(self):
        val_loss = torch.stack(self.val_loss).mean().item()
        val_nonempty_loss = torch.stack(self.val_nonempty_loss).mean().item()
        self.log("val_loss", val_loss)
        self.log("val_nonempty_loss", val_nonempty_loss)
        self.val_loss.clear()
        self.val_nonempty_loss.clear()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


model = MetrixXYHistSurrogate()

dataset = RayDataset(h5_files=['datasets/metrix_simulation/ray_emergency_surrogate/50+50_data_raw_0.h5'],
                     sub_groups=['1e5/params',
                                 '1e5/histogram'], transform=Select(keys=['1e5/params', '1e5/histogram'], search_space=params()))

memory_dataset = MemoryDataset(dataset=dataset)
datamodule = DefaultDataModule(dataset=memory_dataset)
wandb_logger = WandbLogger(project="xy_hist")
trainer = L.Trainer(max_epochs=1000, logger=wandb_logger, log_every_n_steps=100)
trainer.fit(model=model, datamodule=datamodule)
