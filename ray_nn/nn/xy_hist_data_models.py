from collections.abc import Iterable
from contextlib import nullcontext
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
from matplotlib.ticker import MultipleLocator
from torch.nn import Module
import wandb
import gc

from datasets.metrix_simulation.config_ray_emergency_surrogate import PARAM_CONTAINER_FUNC as params
from ray_nn.data.lightning_data_module import DefaultDataModule
from ray_tools.base.backend import RayOutput
from ray_tools.base.engine import Engine
from ray_tools.base.parameter import MutableParameter, NumericalOutputParameter, RandomOutputParameter, RayParameterContainer
from ray_tools.base.transform import RayTransform
from ray_tools.base.utils import RandomGenerator
from ray_tools.simulation.torch_datasets import BalancedMemoryDataset, RayDataset, HistDataset
from ray_nn.data.transform import Select

SMALL_SIZE = 10
MEDIUM_SIZE = 16
BIGGER_SIZE = 25

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title



class MetrixXYHistSurrogate(L.LightningModule):
    def __init__(self, standardizer, input_parameter_container:RayParameterContainer, histogram_lims, total_bin_count:int=100, layer_size:int=4, blow=2.0, shrink_factor:str='log', learning_rate:float=1e-4, optimizer:str='adam',  dataset_length: int | None=None, dataset_normalize_outputs:bool=False, last_activation=nn.Sigmoid(), lr_scheduler: str | None = "exp"):
        super(MetrixXYHistSurrogate, self).__init__()
        self.save_hyperparameters(ignore=['last_activation'])

        self.input_parameter_container = input_parameter_container
        self.total_bin_count = total_bin_count
        self.mutable_parameter_count = len([item for item in self.input_parameter_container.values() if isinstance(item, MutableParameter)])
        self.net = self.create_sequential(self.mutable_parameter_count, self.total_bin_count, layer_size, blow=blow, shrink_factor=shrink_factor, activation_function=nn.Mish(), last_activation=last_activation)
        self.validation_plot_len = 4
        self.learning_rate = learning_rate
        self.lr_scheduler = lr_scheduler
        self.optimizer = optimizer
        self.standardizer = standardizer
        self.criterion = torch.nn.MSELoss()
        self.histogram_lims = histogram_lims
        self.register_buffer("validation_y_plot_data", torch.full((self.validation_plot_len, self.total_bin_count), torch.nan))
        self.register_buffer("validation_y_hat_plot_data", torch.full((self.validation_plot_len, self.total_bin_count), torch.nan))
        self.register_buffer("validation_y_empty_plot_data", torch.full((self.validation_plot_len, self.total_bin_count), torch.nan))
        self.register_buffer("validation_y_hat_empty_plot_data", torch.full((self.validation_plot_len, self.total_bin_count), torch.nan))
        special_sample_input = torch.tensor([[0.6176527143, 0.2370701432, 0.5204789042, 0.3861027360, 0.6581171155,
         0.6619033217, 0.6320477128, 0.2780615389, 0.5915879607, 0.2885326445,
         0.4318543971, 0.5335806012, 0.4898788631, 0.5250989199, 0.5505525470,
         0.5152570605, 0.4368085861, 0.4815735817, 0.5060048699, 0.4239439666,
         0.6532316208, 0.5210996866, 0.4563755095, 0.3020407259, 0.6783920527,
         0.4192821085, 0.2460880578, 0.4803712368, 0.6794303656, 0.6803815365,
         0.6727091074, 0.4795180857, 0.4443074763, 0.5825657845]], device=self.device)
        shape_diff = self.mutable_parameter_count - special_sample_input.shape[1]
        self.register_buffer("special_sample_input", torch.nn.functional.pad(special_sample_input, (shape_diff,0), "constant", 0.))
        self.register_buffer("special_sample_simulation_output", torch.tensor([[   0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,
           0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,  156.,
         918.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,
           0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,
           0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,
           0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,
           0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,
           0.,    0.,    0.,    0.,    0.,   13., 1048.,   13.,    0.,    0.,
           0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,
           0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.]]))
        print(self.net)

    def forward(self, input):
        return self.net(input)

    def create_sequential(self, input_length, output_length, layer_size, blow: int | float = 0, shrink_factor="log", activation_function: Module=nn.ReLU(), last_activation: Module | None = None, batch_norm=False, layer_norm=False):
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
                if layer_norm:
                    nn_layers.append(nn.LayerNorm(int(layers[i+1].item())))
                nn_layers.append(activation_function)
                if batch_norm:
                    nn_layers.append(nn.BatchNorm1d(int(layers[i+1].item())))
            if i == len(layers)-2:
                nn_layers.append(last_activation)
        return nn.Sequential(*nn_layers)

    def training_step(self, batch):
        x, y = batch
        y = y.flatten(start_dim=1)
        y_hat = self.net(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch):
        x, y = batch
        y = y.flatten(start_dim=1)
        y_hat = self.net(x)
        empty_mask = y.mean(dim=1) == 0.
        y_nonempty = y[~empty_mask]
        y_hat_nonempty = y_hat[~empty_mask]  
        if len(y_nonempty) > 0:
            still_available_mask = torch.isnan(self.validation_y_plot_data).all(dim=1)
            available_indices = torch.arange(len(still_available_mask), device=y_nonempty.device)[still_available_mask]
            self.validation_y_plot_data[available_indices[:len(y_nonempty)]] = y_nonempty[:still_available_mask.sum()]
            self.validation_y_hat_plot_data[available_indices[:len(y_hat_nonempty)]] = y_hat_nonempty[:still_available_mask.sum()]
            nonempty_loss = self.criterion(y_hat_nonempty, y_nonempty)
            self.log("val_nonempty_loss", nonempty_loss, prog_bar=True, logger=True)
        if empty_mask.sum() > 0:
            still_available_mask = torch.isnan(self.validation_y_empty_plot_data).all(dim=1)
            available_indices = torch.arange(len(still_available_mask), device=y.device)[still_available_mask]
            self.validation_y_empty_plot_data[available_indices[:len(y[empty_mask])]] = y[empty_mask][:still_available_mask.sum()]
            self.validation_y_hat_empty_plot_data[available_indices[:len(y_hat[empty_mask])]] = y_hat[empty_mask][:still_available_mask.sum()]
        val_loss = self.criterion(y_hat, y)
        self.log("val_loss", val_loss, prog_bar=True, logger=True)
        return val_loss

    def test_step(self, batch):
        return self.validation_step(batch)
    
    def on_test_epoch_end(self):
        self.on_validation_epoch_end()

    @staticmethod
    def plot_data(prediction, ground_truth):
        fig, ax = plt.subplots(len(ground_truth), 2, squeeze=False, sharex=True, sharey=True, layout='constrained', figsize=(4.905, 4.434))
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for i, y_element in enumerate(ground_truth):
            ax[i, 0].plot(y_element[50:], label='simulation', c=colors[3])
            ax[i, 0].plot(prediction[i, 50:], label='surrogate', c=colors[4])
            ax[i, 1].plot(y_element[:50], label='simulation', c=colors[3])
            ax[i, 1].plot(prediction[i, :50], label='surrogate', c=colors[4])
            ax[i, 1].xaxis.set_major_locator(MultipleLocator(25))
        ax[ground_truth.shape[0]-1, 0].set_xlabel('$x$ histogram')
        ax[ground_truth.shape[0]-1, 1].set_xlabel('$y$ histogram')
        plt.legend()
        fig.supylabel('Normalized ray counts')
        return fig

    @staticmethod
    def create_plot(label: str, y_hat, y):
        if len(y) > 0:
            plt.clf()
            fig = MetrixXYHistSurrogate.plot_data(y_hat.cpu().detach().numpy(), y.cpu().detach().numpy())
            plt.savefig("outputs/"+label+".pdf")
            wandb.log({label: wandb.Image(fig)})
            fig.clf()
            plt.close("all")
            gc.collect()

    def on_validation_epoch_end(self):
        MetrixXYHistSurrogate.create_plot('validation', self.validation_y_hat_plot_data, self.validation_y_plot_data)
        MetrixXYHistSurrogate.create_plot('validation_empty', self.validation_y_hat_empty_plot_data, self.validation_y_empty_plot_data)
        self.validation_y_hat_plot_data = torch.full((self.validation_plot_len, self.total_bin_count), torch.nan).to(self.validation_y_plot_data)
        self.validation_y_plot_data = torch.full((self.validation_plot_len, self.total_bin_count), torch.nan).to(self.validation_y_hat_plot_data)
        self.validation_y_hat_empty_plot_data = torch.full((self.validation_plot_len, self.total_bin_count), torch.nan).to(self.validation_y_empty_plot_data)
        self.validation_y_empty_plot_data = torch.full((self.validation_plot_len, self.total_bin_count), torch.nan).to(self.validation_y_hat_empty_plot_data)
        MetrixXYHistSurrogate.create_plot('special_sample', self.standardizer.destandardize(self(self.special_sample_input)), self.special_sample_simulation_output)
        gc.collect()
        
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

    
if __name__ == '__main__':
    load_len: int | None = None
    standardizer = StandardizeXYHist()
    h5_files = list(glob.iglob('datasets/metrix_simulation/ray_emergency_surrogate_50+50+z+-30/histogram_*.h5'))
    sub_groups = ['parameters', 'histogram/ImagePlane', 'n_rays/ImagePlane']
    transforms=[lambda x: x[1:].float(), lambda x: standardizer(x.flatten().float()), lambda x: x.int()]
    dataset = HistDataset(h5_files, sub_groups, transforms, normalize_sub_groups=['parameters'])
    memory_dataset = BalancedMemoryDataset(dataset=dataset, load_len=load_len, min_n_rays=10, debug_mode=False)
    workers = psutil.Process().cpu_affinity()
    num_workers = len(workers) if workers is not None else 0
    datamodule = DefaultDataModule(dataset=memory_dataset, num_workers=num_workers)
    datamodule.prepare_data()
    model = MetrixXYHistSurrogate(dataset_length=load_len, standardizer=standardizer,  input_parameter_container=HistDataset.retrieve_parameter_container(h5_files[0]), layer_size=7, histogram_lims=HistDataset.retrieve_xy_lims(h5_files[0]))
    test = False
    if not test:
        wandb_logger = WandbLogger(name="ref2_bal_10_sch_.999_std_log_mish_z+-30_7_l", project="xy_hist", save_dir='outputs')
    else:
        wandb_logger =  WandbLogger(name="test", project="xy_hist", save_dir='outputs', offline=True)
    if test:
        datamodule.setup(stage="test")
    else:
        datamodule.setup(stage="fit")

    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = L.Trainer(max_epochs=10000, logger=wandb_logger, log_every_n_steps=100000, check_val_every_n_epoch=1, callbacks=[lr_monitor])
    trainer.init_module()

    if test:
        model = MetrixXYHistSurrogate.load_from_checkpoint(
        checkpoint_path="outputs/xy_hist/qhmpdasi/checkpoints/epoch=295-step=118652488.ckpt",
        #hparams_file="/path/to/experiment/version/hparams.yaml",
        map_location=None,
        )
        trainer.test(datamodule=datamodule, ckpt_path='outputs/xy_hist/qhmpdasi/checkpoints/epoch=295-step=118652488.ckpt', model=model)
    else:
        trainer.fit(model=model, datamodule=datamodule)

class Model:
    def __init__(self, path):
        model_orig = MetrixXYHistSurrogate.load_from_checkpoint(path)
        if torch.cuda.is_available():
            model_orig = model_orig.to('cuda')
        #model_orig.compile()
        model_orig.eval()
        self.model_orig = model_orig
        self.device = model_orig.device
        self.histogram_lims = model_orig.histogram_lims
        self.input_parameter_container = model_orig.input_parameter_container
        new_entries = [('ImagePlane.translationXerror', RandomOutputParameter(value_lims=(-7,7), rg=RandomGenerator(42))),
            ('ImagePlane.translationYerror', RandomOutputParameter(value_lims=(-3,3), rg=RandomGenerator(42)))]
        x_lims = dict(new_entries)['ImagePlane.translationXerror'].value_lims
        y_lims = dict(new_entries)['ImagePlane.translationYerror'].value_lims
        hist_x_lims = model_orig.histogram_lims[0]
        hist_y_lims = model_orig.histogram_lims[1]
        self.x_factor = (x_lims[1]-x_lims[0])/(hist_x_lims[1]-hist_x_lims[0]).item() # translationX_interval length [mm] / histogram_interval length [mmm]
        self.y_factor = (y_lims[1]-y_lims[0])/(hist_y_lims[1]-hist_y_lims[0]).item()  # translationY_interval length [mm] / histogram_interval length [mmm]
        self.input_parameter_container = Model.add_entries_to_pc(self.model_orig.input_parameter_container, new_entries)
        self.mutable_parameter_count = model_orig.mutable_parameter_count + 2
        self.standardizer = self.model_orig.standardizer
        self.offset_space_overrides = {'ImagePlane.translationXerror': RandomOutputParameter(value_lims=(-3.,3.), rg=RandomGenerator(42)), 'ImagePlane.translationYerror': RandomOutputParameter(value_lims=(-2.0,2.0), rg=RandomGenerator(42)), 'ImagePlane.translationZerror': RandomOutputParameter(value_lims=(-3,3), rg=RandomGenerator(42))}
        self.max_offset_share = 0.2
        self.offset_space = Model.calculate_offset_space(self.input_parameter_container, self.max_offset_share, self.offset_space_overrides)
        self.rescale_multiplier, self.rescale_addend = Model.mutable_parameter_offset_conversion_factor(self.input_parameter_container, self.offset_space, device=self.device)

    @staticmethod
    def add_entries_to_pc(pc: RayParameterContainer, new_entries: list):
        new_parameter_container = []
        done = False
        for key, value in pc.items():
            if key == 'ImagePlane.translationZerror':
                new_parameter_container += new_entries
                done = True
            new_parameter_container.append((key, value))
        if not done:
            new_parameter_container += new_entries
        return RayParameterContainer(new_parameter_container)
    @staticmethod
    def calculate_offset_space(input_parameter_container: RayParameterContainer, max_offset_share: float, offset_space_overrides={}):
        new_parameter_container = []
        for key, value in input_parameter_container.items():
            if key in offset_space_overrides.keys():
                new_parameter_container.append((key, offset_space_overrides[key]))
            else:
                if isinstance(value, MutableParameter):
                    value_copy = value.clone()
                    old_size = value_copy.value_lims[1] - value_copy.value_lims[0]
                    new_min = - old_size * max_offset_share / 2.
                    new_max = old_size * max_offset_share / 2.
                    value_copy.value_lims = (new_min, new_max)
                else:
                    value_copy = value
                new_parameter_container.append((key, value_copy))
        return RayParameterContainer(new_parameter_container)
    
    @staticmethod
    def mutable_parameter_offset_conversion_factor(model_space: RayParameterContainer, offset_space: RayParameterContainer, device):
        model_min_list = []
        model_max_list = []
        offset_min_list = []
        offset_max_list = []
        for key, value in model_space.items():
            if isinstance(value, MutableParameter):
                model_min_list.append(value.value_lims[0])
                model_max_list.append(value.value_lims[1])
                offset_value = offset_space[key]
                assert isinstance(offset_value, MutableParameter)
                offset_min_list.append(offset_value.value_lims[0])
                offset_max_list.append(offset_value.value_lims[1])
        model_min = torch.tensor(model_min_list, device=device)
        model_max = torch.tensor(model_max_list, device=device)
        offset_min = torch.tensor(offset_min_list, device=device)
        offset_max = torch.tensor(offset_max_list, device=device)
        rescale_scaler = 1. / (model_max - model_min)
        rescale_multiplier = (offset_max- offset_min) * rescale_scaler
        rescale_addend = (offset_min) * rescale_scaler
        return rescale_multiplier.float(), rescale_addend.float()
    def rescale_offset(self, offset):
        return offset * self.rescale_multiplier + self.rescale_addend

    def __call__(self, x, offset=None, clone_output=False, grad=False):
        assert x.shape[-1] == 37
        if offset is not None:
            x = x + self.rescale_offset(offset)
        inp = torch.cat((x[..., :-3],x[..., -1:]), dim=-1)
            
        with torch.no_grad() if not grad else nullcontext():
            output = self.model_orig(inp)

        output = output.view(*(output.size()[:-1]), 2, -1)
        if clone_output:
            output_copy = output.clone()
        else: 
            output_copy = output
        translation_x = x[..., -3]
        translation_y = x[..., -2]
        output_copy[..., 0,:] = Model.batch_translate_histograms(output[..., 0, :], (translation_x-0.5)*self.x_factor)
        output_copy[..., 1, :] = Model.batch_translate_histograms(output[..., 1, :], (translation_y-0.5) *self.y_factor)
        return output_copy.flatten(start_dim=-2)
        
    @staticmethod
    def batch_translate_histograms(hist_batch, shift_tensors):
        """
        Translate a batch of histograms in the x-direction based on the batch of shift tensors.
        The histograms have defined ranges on both x and y axes.
    
        Parameters:
        hist_batch (torch.Tensor): A tensor of shape [batch_size, 2, 50] representing a batch of histograms.
        shift_tensors (torch.Tensor): A tensor of shape [batch_size, 1] representing the shift proportions for each histogram between -0.5 and 0.5.
        x_range (tuple): The range of the x-axis as (min, max).
        y_range (tuple): The range of the y-axis as (min, max).
    
        Returns:
        torch.Tensor: The translated histograms with out-of-bounds values ignored and zeros filled.
        """
        num_bins = hist_batch.shape[-1]
        shift_tensors = shift_tensors
    
        # Calculate the number of bins to shift for each histogram in the batch
        translation_bins = torch.round(shift_tensors * num_bins).long() # Shape: [batch_size]  
        translated_hist_batch = torch.zeros_like(hist_batch)
        bin_indices = torch.arange(num_bins, device=hist_batch.device).unsqueeze(0)  # Shape: [1, num_bins]
        # Calculate the valid indices after translation for each histogram
        valid_indices = bin_indices - translation_bins.unsqueeze(-1)  # Shape: [batch_size, num_bins]
        
        valid_mask = (valid_indices >= 0) & (valid_indices < num_bins)  # Mask for valid positions
        valid_indices = torch.where(valid_mask, valid_indices, 0)
        # Translate the x-axis values (first row of histograms)
        translated_hist_batch = torch.gather(hist_batch, -1, valid_indices)
        translated_hist_batch = torch.where(valid_mask, translated_hist_batch, torch.zeros_like(translated_hist_batch))
        return translated_hist_batch

class HistSurrogateEngine(Engine):
    def __init__(self, module=MetrixXYHistSurrogate, checkpoint_path: str="outputs/xy_hist/14mxibq2/checkpoints/epoch=102-step=19603784.ckpt"):
        super().__init__()
        self.model = Model(checkpoint_path)
        self.select = Select(keys=['1e5/params'], omit_ray_params=['U41_318eV.numberRays'], search_space=self.model.input_parameter_container, non_dict_transform={'1e5/ray_output/ImagePlane/histogram': self.model.model_orig.standardizer}, device=self.model.device)

    def run(self, param_containers: list[RayParameterContainer], transforms: RayTransform | dict[str, RayTransform] | Iterable[RayTransform | dict[str, RayTransform]] | None = None) -> list[dict]:
        param_containers_tensor = torch.vstack([self.select({"1e5/params":param_container})[0] for param_container in param_containers])
        with torch.no_grad():
            output = self.model(param_containers_tensor)
            hist_len = int(output.shape[-1] / 2)
            ray_dict_list = [{'x_hist': output_element[:hist_len], 'y_hist': output_element[output.shape[-1]-hist_len:]} for output_element in output]
            return [{'ray_output': {'ImagePlane': {'xy_hist': RayOutput(x_loc=ray_dict['x_hist'], y_loc=ray_dict['y_hist'], z_loc=torch.Tensor(), x_dir=torch.Tensor(), y_dir=torch.Tensor(), z_dir=torch.Tensor(), energy=torch.Tensor())}}} for ray_dict in ray_dict_list]
