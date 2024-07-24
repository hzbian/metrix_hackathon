from collections.abc import Iterable
import glob
import math
import psutil
import torch
from torch import optim, nn
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from lightning.pytorch.callbacks import LearningRateMonitor
import matplotlib.pyplot as plt
from torch.nn import Module
import wandb

from datasets.metrix_simulation.config_ray_emergency_surrogate import PARAM_CONTAINER_FUNC as params
from ray_nn.data.lightning_data_module import DefaultDataModule
from ray_tools.base.backend import RayOutput
from ray_tools.base.engine import Engine
from ray_tools.base.parameter import RayParameterContainer
from ray_tools.base.transform import RayTransform
from ray_tools.simulation.torch_datasets import BalancedMemoryDataset, RayDataset
from ray_nn.data.transform import Select

class MetrixXYHistSurrogate(L.LightningModule):
    def __init__(self, standardizer, layer_size:int=4, blow=2.0, shrink_factor:str='log', learning_rate:float=1e-4, optimizer:str='adam', dataset_length: int | None=None, dataset_normalize_outputs:bool=False, last_activation=nn.Sigmoid(), lr_scheduler: str | None = "exp"):
        super(MetrixXYHistSurrogate, self).__init__()
        self.save_hyperparameters(ignore=['last_activation'])

        self.net = self.create_sequential(34, 1, layer_size, blow=blow, shrink_factor=shrink_factor, activation_function=nn.Mish(), last_activation=last_activation)
        self.validation_plot_len = 5
        self.learning_rate = learning_rate
        self.lr_scheduler = lr_scheduler
        self.optimizer = optimizer
        self.standardizer = standardizer
        self.register_buffer("validation_y_plot_data", torch.tensor([]))
        self.register_buffer("validation_y_hat_plot_data", torch.tensor([]))
        self.register_buffer("validation_y_empty_plot_data", torch.tensor([]))
        self.register_buffer("validation_y_hat_empty_plot_data", torch.tensor([]))
        self.register_buffer("train_y_empty_plot_data", torch.tensor([]))
        self.register_buffer("train_y_plot_data", torch.tensor([]))
        self.register_buffer("train_y_hat_empty_plot_data", torch.tensor([]))
        self.register_buffer("train_y_hat_plot_data", torch.tensor([]))
        print(self.net)

    def forward(self, input):
        return self.net(input)

    def create_sequential(self, input_length, output_length, layer_size, blow: int | float =0, shrink_factor="log", activation_function: Module=nn.ReLU(), last_activation: Module | None = None):
        layers = [input_length]
        blow_disabled = blow == 1 or blow == 0
        if not blow_disabled:
            layers.append(input_length*blow)

        if shrink_factor == "log":
            add_layers = torch.logspace(math.log(layers[-1], 10), math.log(output_length,10), steps=layer_size+2-len(layers), base=10).long()
            # make sure the first and last element is correct, even though rounding
            if blow_disabled:
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
                nn_layers.append(activation_function)
                #nn_layers.append(nn.BatchNorm1d(layers[i+1].item()))
            if i == len(layers)-2:
                nn_layers.append(last_activation)
        return nn.Sequential(*nn_layers)

    def training_step(self, batch):
        x, y = batch
        y = y.flatten(start_dim=1)
        y_hat = self.net(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        empty_mask = y.mean(dim=1) == 0.
        if (~empty_mask).sum() > 0. and self.train_y_plot_data.shape[0] < self.validation_plot_len:
            append_len = self.validation_plot_len - self.train_y_plot_data.shape[0]
            self.train_y_plot_data = torch.cat([self.train_y_plot_data,  y[~empty_mask][:append_len]])
            self.train_y_hat_plot_data = torch.cat([self.train_y_hat_plot_data,  y_hat[~empty_mask][:append_len]])
        if empty_mask.sum() > 0. and self.train_y_empty_plot_data.shape[0] < self.validation_plot_len:
            append_len = self.validation_plot_len - self.train_y_empty_plot_data.shape[0]
            self.train_y_empty_plot_data = torch.cat([self.train_y_empty_plot_data,  y[empty_mask][:append_len]])
            self.train_y_hat_empty_plot_data = torch.cat([self.train_y_hat_empty_plot_data,  y_hat[empty_mask][:append_len]])
        return loss
    
    def validation_step(self, batch):
        x, y = batch
        y = y.flatten(start_dim=1)
        y_hat = self.net(x)
        empty_mask = y.mean(dim=1) == 0.
        y_nonempty = y[~empty_mask]
        y_hat_nonempty = y_hat[~empty_mask]
        if (~empty_mask).sum() > 0. and self.validation_y_plot_data.shape[0] < self.validation_plot_len:
            append_len = self.validation_plot_len - self.validation_y_plot_data.shape[0]
            self.validation_y_plot_data = torch.cat([self.validation_y_plot_data, y_nonempty[:append_len]])
            self.validation_y_hat_plot_data = torch.cat([self.validation_y_hat_plot_data, y_hat_nonempty[:append_len]])
        if empty_mask.sum() > 0. and self.validation_y_empty_plot_data.shape[0] < self.validation_plot_len:
            append_len = self.validation_plot_len - self.validation_y_empty_plot_data.shape[0]
            self.validation_y_empty_plot_data = torch.cat([self.validation_y_empty_plot_data,  y[empty_mask][:append_len]])
            self.validation_y_hat_empty_plot_data = torch.cat([self.validation_y_hat_empty_plot_data,  y_hat[empty_mask][:append_len]])
        if (~empty_mask).sum() > 0.:
            nonempty_loss = nn.functional.mse_loss(y_hat_nonempty, y_nonempty)
            self.log("val_nonempty_loss", nonempty_loss, prog_bar=True, logger=True)
        val_loss = nn.functional.mse_loss(y_hat, y)
        self.log("val_loss", val_loss, prog_bar=True, logger=True)
        return val_loss
    
    def test_step(self, batch):
        return self.validation_step(batch)
    
    def on_test_epoch_end(self):
        self.on_validation_epoch_end()

    @staticmethod
    def plot_data(prediction, ground_truth):
        fig, ax = plt.subplots(len(ground_truth), 2, squeeze=False)
        for i, y_element in enumerate(ground_truth):
            ax[i, 0].plot(y_element[50:], label='gt')
            ax[i, 0].plot(prediction[i, 50:], label='prediction')
            ax[i, 1].plot(y_element[:50], label='gt')
            ax[i, 1].plot(prediction[i, :50], label='prediction')
        ax[ground_truth.shape[0]-1, 0].set_xlabel('histogram_x')
        ax[ground_truth.shape[0]-1, 1].set_xlabel('histogram_y')
        plt.tight_layout()
        plt.legend()
        return fig
    @staticmethod
    def create_plot(label: str, y_hat, y):
        if len(y) > 0:
            plt.clf()
            fig = MetrixXYHistSurrogate.plot_data(y_hat.cpu().detach().numpy(), y.cpu().detach().numpy())
            wandb.log({label: wandb.Image(fig)})
            plt.close(fig)

    def on_validation_epoch_end(self):
        MetrixXYHistSurrogate.create_plot('validation', self.validation_y_hat_plot_data, self.validation_y_plot_data)
        MetrixXYHistSurrogate.create_plot('train', self.train_y_hat_plot_data, self.train_y_plot_data)
        MetrixXYHistSurrogate.create_plot('validation_empty', self.validation_y_hat_empty_plot_data, self.validation_y_empty_plot_data)
        MetrixXYHistSurrogate.create_plot('train_empty', self.train_y_hat_empty_plot_data, self.train_y_empty_plot_data)
        self.train_y_hat_plot_data = torch.tensor([]).to(self.train_y_plot_data)
        self.train_y_hat_empty_plot_data = torch.tensor([]).to(self.train_y_empty_plot_data)
        self.train_y_empty_plot_data = torch.tensor([]).to(self.train_y_hat_empty_plot_data)
        self.train_y_plot_data = torch.tensor([]).to(self.train_y_hat_plot_data)
        self.validation_y_hat_plot_data = torch.tensor([]).to(self.validation_y_plot_data)
        self.validation_y_plot_data = torch.tensor([]).to(self.validation_y_hat_plot_data)
        self.validation_y_hat_empty_plot_data = torch.tensor([]).to(self.validation_y_empty_plot_data)
        self.validation_y_empty_plot_data = torch.tensor([]).to(self.validation_y_hat_empty_plot_data)
        special_sample_input = torch.tensor([[0.6176527143, 0.2370701432, 0.5204789042, 0.3861027360, 0.6581171155,
         0.6619033217, 0.6320477128, 0.2780615389, 0.5915879607, 0.2885326445,
         0.4318543971, 0.5335806012, 0.4898788631, 0.5250989199, 0.5505525470,
         0.5152570605, 0.4368085861, 0.4815735817, 0.5060048699, 0.4239439666,
         0.6532316208, 0.5210996866, 0.4563755095, 0.3020407259, 0.6783920527,
         0.4192821085, 0.2460880578, 0.4803712368, 0.6794303656, 0.6803815365,
         0.6727091074, 0.4795180857, 0.4443074763, 0.5825657845]], device=self.device)
        special_sample_simulation_output = torch.tensor([[   0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,
           0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,  156.,
         918.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,
           0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,
           0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,
           0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,
           0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,
           0.,    0.,    0.,    0.,    0.,   13., 1048.,   13.,    0.,    0.,
           0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,
           0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.]])
        #MetrixXYHistSurrogate.create_plot('special_sample', self.standardizer.destandardize(self(special_sample_input)), special_sample_simulation_output)

    def configure_optimizers(self):
        if self.optimizer == "adam_w":
            optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        else:
            optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        if self.lr_scheduler == "exp":
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": ExponentialLR(optimizer, gamma=0.999),
                    "frequency": 1,
                },
            }
        if self.lr_scheduler == "plateau":
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": ReduceLROnPlateau(optimizer, patience=5),
                    "monitor": "val_loss",
                    "frequency": 1,
                },
            }
        if self.lr_scheduler is not None:
            raise Exception("Defined LR scheduler not found.")

        return optimizer

class StandardizeXYHist():
    def __init__(self, divisor=22594., log=False):
        self.divisor = divisor
        self.log = log
    def __call__(self, element):
        if self.log:
            return torch.log((element+1)/self.divisor)
        else:
            return element / self.divisor
    def destandardize(self, element):
        if self.log:
            return (torch.exp(element) * self.divisor) - 1
        else:
            return element * self.divisor
class Maximator():
    def __call__(self, element):
        return element.max().unsqueeze(0)
    
if __name__ == '__main__':
    load_len: int | None = None
    dataset_normalize_outputs = True
    h5_files_original = list(glob.iglob('datasets/metrix_simulation/ray_emergency_surrogate/data_raw_*.h5'))
    h5_files_selected = list(glob.iglob('datasets/metrix_simulation/ray_emergency_surrogate/selected/data_raw_*.h5'))
    assert len(h5_files_original) == len(h5_files_selected)
    original_ratio = 0.2
    amount_original = int(len(h5_files_original) * original_ratio)
    h5_files = h5_files_original[:amount_original]+h5_files_selected[amount_original:]
    #standardizer = StandardizeXYHist(divisor=1., log=True)
    standardizer = Maximator()
    dataset = RayDataset(h5_files=h5_files,
                        sub_groups=['1e5/params',
                                    '1e5/ray_output/ImagePlane/histogram', '1e5/ray_output/ImagePlane/n_rays'], transform=Select(keys=['1e5/params', '1e5/ray_output/ImagePlane/histogram', '1e5/ray_output/ImagePlane/n_rays'], search_space=params(), non_dict_transform={'1e5/ray_output/ImagePlane/histogram': standardizer}))

    memory_dataset = BalancedMemoryDataset(dataset=dataset, load_len=load_len, min_n_rays=10)
    split_swap_epochs = 1000
    workers = psutil.Process().cpu_affinity()
    num_workers = len(workers) if workers is not None else 0
    datamodule = DefaultDataModule(dataset=memory_dataset, num_workers=num_workers, split_training=0, split_swap_epochs=split_swap_epochs)
    datamodule.prepare_data()
    model = MetrixXYHistSurrogate(dataset_length=load_len, dataset_normalize_outputs=dataset_normalize_outputs, standardizer=standardizer)
    test = False
    wandb_logger = WandbLogger(name="ref", project="xy_hist_maximator", save_dir='outputs')
    #wandb_logger = None
    if test:
        datamodule.setup(stage="test")
    else:
        datamodule.setup(stage="fit")

    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = L.Trainer(max_epochs=10000, logger=wandb_logger, log_every_n_steps=100000, check_val_every_n_epoch=1, callbacks=[lr_monitor], reload_dataloaders_every_n_epochs=split_swap_epochs)
    trainer.init_module()

    if test:
        trainer.test(datamodule=datamodule, ckpt_path='outputs/xy_hist/50f8si6i/checkpoints/epoch=29-step=5842830.ckpt', model=model)
    else:
        trainer.fit(model=model, datamodule=datamodule)

class HistSurrogateEngine(Engine):
    def __init__(self, module=MetrixXYHistSurrogate, checkpoint_path: str="outputs/xy_hist/14mxibq2/checkpoints/epoch=102-step=19603784.ckpt"):
        super().__init__()
        self.model = module.load_from_checkpoint(checkpoint_path)
        self.model.to(torch.device('cpu'))
        self.model.compile()
        self.model.eval()
        self.select = Select(keys=['1e5/params'], omit_ray_params=['U41_318eV.numberRays'], search_space=params(), non_dict_transform={'1e5/ray_output/ImagePlane/histogram': self.model.standardizer})

    def run(self, param_containers: list[RayParameterContainer], transforms: RayTransform | dict[str, RayTransform] | Iterable[RayTransform | dict[str, RayTransform]] | None = None) -> list[dict]:
        param_containers_tensor = torch.vstack([self.select({"1e5/params":param_container})[0] for param_container in param_containers])
        with torch.no_grad():
            output = self.model(param_containers_tensor)
            hist_len = int(output.shape[-1] / 2)
            ray_dict_list = [{'x_hist': output_element[:hist_len], 'y_hist': output_element[output.shape[-1]-hist_len:]} for output_element in output]
            return [{'ray_output': {'ImagePlane': {'xy_hist': RayOutput(x_loc=ray_dict['x_hist'], y_loc=ray_dict['y_hist'], z_loc=torch.Tensor(), x_dir=torch.Tensor(), y_dir=torch.Tensor(), z_dir=torch.Tensor(), energy=torch.Tensor())}}} for ray_dict in ray_dict_list]
