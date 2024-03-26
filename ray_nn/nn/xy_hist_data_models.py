import glob
import math
import torch
from torch import optim, nn
import lightning as L
from lightning.pytorch.loggers import WandbLogger
import matplotlib.pyplot as plt
import wandb

from datasets.metrix_simulation.config_ray_emergency_surrogate import PARAM_CONTAINER_FUNC as params
from ray_nn.data.lightning_data_module import DefaultDataModule
from ray_tools.simulation.torch_datasets import MemoryDataset, RayDataset
from ray_nn.data.transform import Select

class MetrixXYHistSurrogate(L.LightningModule):
    def __init__(self, layer_size:int=6, blow=4.0, shrink_factor:str='log', learning_rate:float=0.001, optimizer:str='adam', dataset_length: int | None=None, dataset_normalize_outputs:bool=False):
        super(MetrixXYHistSurrogate, self).__init__()
        self.save_hyperparameters()

        self.net = self.create_sequential(34, 100, layer_size, blow=blow, shrink_factor=shrink_factor)
        self.val_loss = []
        self.val_nonempty_loss = []
        self.train_loss = []
        self.validation_plot_len = 5
        self.validation_y_plot_data = torch.tensor([])
        self.validation_y_hat_plot_data = torch.tensor([])
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
            nn_layers.append(nn.Linear(int(layers[i].item()), int(layers[i+1].item())))
            if not i == len(layers)-2:
                nn_layers.append(nn.ReLU())
                #nn_layers.append(nn.BatchNorm1d(layers[i+1].item()))
        return nn.Sequential(*nn_layers)

    def training_step(self, batch):
        x, y = batch
        y = y.flatten(start_dim=1)
        y_hat = self.net(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.train_loss.append(loss)
        return loss
    
    def validation_step(self, batch):
        x, y = batch
        y = y.flatten(start_dim=1)
        y_hat = self.net(x)
        y = y.cpu()
        y_hat = y_hat.cpu()
        nonempty_mask = y.mean(dim=1) != 0.
        y_nonempty = y[nonempty_mask]
        y_hat_nonempty = y_hat[nonempty_mask]
        if nonempty_mask.sum() > 0. and self.validation_y_plot_data.shape[0] < self.validation_plot_len:
            append_len = self.validation_plot_len - self.validation_y_plot_data.shape[0]
            self.validation_y_plot_data = torch.cat([self.validation_y_plot_data, y_nonempty[:append_len]])
            self.validation_y_hat_plot_data = torch.cat([self.validation_y_hat_plot_data, y_hat_nonempty[:append_len]])
        if nonempty_mask.sum() > 0.:
            nonempty_loss = nn.functional.mse_loss(y_hat_nonempty, y_nonempty)
            self.val_nonempty_loss.append(nonempty_loss)
        val_loss = nn.functional.mse_loss(y_hat, y)
        self.val_loss.append(val_loss)
        return val_loss
    
    def test_step(self, batch):
        return self.validation_step(batch)
    
    def on_test_epoch_end(self):
        self.on_validation_epoch_end()

    def on_train_epoch_end(self):
        train_loss = torch.stack(self.train_loss).mean().item()
        self.log("train_loss", train_loss)
        self.train_loss.clear()

    def on_validation_epoch_end(self):
        val_loss = torch.stack(self.val_loss).mean().item()
        if len(self.val_nonempty_loss) != 0:
            val_nonempty_loss = torch.stack(self.val_nonempty_loss).mean().item()
        else:
            val_nonempty_loss = float('nan')
        self.log("val_loss", val_loss)
        self.log("val_nonempty_loss", val_nonempty_loss)
        self.val_loss.clear()
        self.val_nonempty_loss.clear()
        if len(self.validation_y_plot_data) > 0:
            _, ax = plt.subplots(len(self.validation_y_plot_data), 2, squeeze=False)
            for i, y_element in enumerate(self.validation_y_plot_data):
                ax[i, 0].plot(y_element[50:], label='gt')
                ax[i, 0].plot(self.validation_y_hat_plot_data[i, 50:], label='prediction')
                ax[i, 1].plot(y_element[:50], label='gt')
                ax[i, 1].plot(self.validation_y_hat_plot_data[i, :50], label='prediction')
            ax[self.validation_y_plot_data.shape[0]-1, 0].set_xlabel('histogram_x')
            ax[self.validation_y_plot_data.shape[0]-1, 1].set_xlabel('histogram_y')
            plt.tight_layout()
            plt.legend()
            wandb.log({"xy_hist_plots": wandb.Image(plt)})

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

class StandardizeXYHist(torch.nn.Module):
    def forward(self, element):
        return element / 22594.

load_len: int | None =  None
dataset_normalize_outputs = True
h5_files = list(glob.iglob('datasets/metrix_simulation/ray_emergency_surrogate/50+50_data_raw_*.h5')) # ['datasets/metrix_simulation/ray_emergency_surrogate/49+50_data_raw_0.h5']
dataset = RayDataset(h5_files=h5_files,
                     sub_groups=['1e5/params',
                                 '1e5/histogram'], transform=Select(keys=['1e5/params', '1e5/histogram'], search_space=params())) #, non_dict_transform=StandardizeXYHist()))

memory_dataset = MemoryDataset(dataset=dataset, load_len=load_len)
datamodule = DefaultDataModule(dataset=memory_dataset, num_workers=4)
datamodule.prepare_data()
model = MetrixXYHistSurrogate(dataset_length=load_len, dataset_normalize_outputs=dataset_normalize_outputs)
test = False
wandb_logger = WandbLogger(project="xy_hist", save_dir='outputs')
if test:
    datamodule.setup(stage="test")
else:
    datamodule.setup(stage="fit")

trainer = L.Trainer(max_epochs=1000, logger=wandb_logger, log_every_n_steps=100, check_val_every_n_epoch=25)
if test:
    trainer.test(datamodule=datamodule, ckpt_path='outputs/xy_hist/on2yv96j/checkpoints/epoch=999-step=25000000.ckpt', model=model)
else:
    trainer.fit(model=model, datamodule=datamodule)