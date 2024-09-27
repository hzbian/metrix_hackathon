import glob
import pickle
import tqdm
import torch
import matplotlib.pyplot as plt

from ray_optim.plot import Plot
from ray_tools.base.transform import MultiLayer
from ray_nn.data.lightning_data_module import DefaultDataModule
from ray_nn.nn.xy_hist_data_models import MetrixXYHistSurrogate
from datasets.metrix_simulation.config_ray_emergency_surrogate import PARAM_CONTAINER_FUNC as params
from datasets.metrix_simulation.config_ray_emergency_surrogate import TRANSFORMS as cfg_transforms
from ray_tools.base.parameter import NumericalParameter, NumericalOutputParameter, RayParameterContainer
from ray_tools.simulation.torch_datasets import BalancedMemoryDataset, MemoryDataset, RayDataset
from sub_projects.ray_optimization.utils import ray_output_to_tensor
from sub_projects.ray_optimization.ray_optimization import RayOptimization
from ray_optim.ray_optimizer import RayOptimizer

from ray_tools.base.utils import RandomGenerator
from sub_projects.ray_optimization.real_data import import_data
from ray_tools.base.transform import XYHistogram
from ray_nn.data.transform import Select


def tensor_to_param_container(ten):
    param_dict = {}
    for i, (label, entry) in enumerate(params().items()):
        if label == 'U41_318eV.numberRays':
            param_dict[label] = entry
        else:
            value = ten[i-1]*(entry.value_lims[1]-entry.value_lims[0])+entry.value_lims[0]
            param_dict[label] = NumericalParameter(value.item())
            if value.item() < entry.value_lims[0] or value.item() > entry.value_lims[1]:
                if value.item() < entry.value_lims[0]:
                    value = torch.ones_like(value) * entry.value_lims[0]
                elif value.item() > entry.value_lims[1]:
                    value = torch.ones_like(value) * entry.value_lims[1]
                #raise Exception("Out of range. Minimum was {}, maximum {} but value {}. Tensor value was {}.".format(entry.value_lims[0], entry.value_lims[1], value.item(), ten[i-1].item()))
    return RayParameterContainer(param_dict)

def mse_engines_comparison(engine, surrogate_engine, param_container_list: list[RayParameterContainer], transforms):
    out = engine.run(param_container_list, transforms)
    out_surrogate = surrogate_engine.run(param_container_list, transforms)
    std_backward = surrogate_engine.model.standardizer.destandardize
    x_simulation_hist_list = []
    y_simulation_hist_list = []
    mse_list = []
    for i in range(len(out_surrogate)):
        surrogate_hist = out_surrogate[i]['ray_output']['ImagePlane']['xy_hist']
        out_simulation = out[i]['ray_output']['ImagePlane']['0.0']
        x_simulation_hist, _ = torch.histogram(out_simulation.x_loc,bins=50, range=[-10, 10])
        x_simulation_hist_list.append(x_simulation_hist)
        y_simulation_hist, _ = torch.histogram(out_simulation.y_loc,bins=50, range=[-3, 3])
        y_simulation_hist_list.append(y_simulation_hist)
        mse = ((torch.stack([std_backward(surrogate_hist.x_loc), std_backward(surrogate_hist.y_loc)]) - torch.stack([x_simulation_hist, y_simulation_hist]))**2).mean()
        mse_list.append(mse)
    return torch.stack(mse_list), x_simulation_hist_list, y_simulation_hist_list

class Model:
    def __init__(self, path):
        model_orig = MetrixXYHistSurrogate.load_from_checkpoint(path)
        if torch.cuda.is_available():
            model_orig = model_orig.to('cuda')
        #model_orig.compile()
        model_orig.eval()
        self.x_factor = 0./20.
        self.y_factor = 0./4.
        self.model_orig = model_orig
        self.device = model_orig.device
    def __call__(self, x, clone_output=False):
        output = self.model_orig(x[..., :34])
        #print("Original Histogram Batch:")
        #print(output.shape)
        #output = output.view(*(output.size()[:-1]), 2, -1)
        if clone_output:
            output = output.clone()
        translation_x = x[..., -2]
        translation_y = x[..., -1]
        #print("output", output.shape)
        #print("tx", translation_x.shape)
        ##output[..., 0, :] = Model.batch_translate_histograms(output[..., 0, :], translation_x*self.x_factor*0.5+0.5)
        ##output[..., 1, :] = Model.batch_translate_histograms(output[..., 1, :], translation_y*self.y_factor*0.5+0.5)
        #print("\nTranslated Histogram Batch:")
        #print(output.flatten(start_dim=-2).shape)
        
        return output#output.flatten(start_dim=-2)
        
    @staticmethod
    def batch_translate_histograms(hist_batch, shift_tensors):
        #print(hist_batch.shape, shift_tensors.shape)
        """
        Translate a batch of histograms in the x-direction based on the batch of shift tensors.
        The histograms have defined ranges on both x and y axes.
    
        Parameters:
        hist_batch (torch.Tensor): A tensor of shape [batch_size, 2, 50] representing a batch of histograms.
        shift_tensors (torch.Tensor): A tensor of shape [batch_size, 1] representing the shift proportions for each histogram.
        x_range (tuple): The range of the x-axis as (min, max).
        y_range (tuple): The range of the y-axis as (min, max).
    
        Returns:
        torch.Tensor: The translated histograms with out-of-bounds values ignored and zeros filled.
        """
        num_bins = hist_batch.shape[-1]
        #bin_width = (lim_min - lim_max) / num_bins

        shift_tensors = shift_tensors * 2 - 1
    
        # Calculate the number of bins to shift for each histogram in the batch
        translation_bins = (shift_tensors * num_bins).long() # Shape: [batch_size]
        
        translated_hist_batch = torch.zeros_like(hist_batch)
        bin_indices = torch.arange(num_bins, device=hist_batch.device).unsqueeze(0)  # Shape: [1, num_bins]
        #print("translation_bins", translation_bins.shape)
        #print("bin_indices", bin_indices.shape)
        #print("translation_bins.unsqueeze_1", translation_bins.unsqueeze(1).shape)
        # Calculate the valid indices after translation for each histogram
        valid_indices = bin_indices - translation_bins.unsqueeze(-1)  # Shape: [batch_size, num_bins]
        l
        
        valid_mask = (valid_indices >= 0) & (valid_indices < num_bins)  # Mask for valid positions
        valid_indices = torch.where(valid_mask, valid_indices, 0)
        #print("valid_indices", valid_indices, valid_indices.shape)
        #print(valid_indices, valid_mask)
        #print(translated_hist_batch.shape, valid_mask.shape, valid_indices.shape, valid_indices[valid_mask])
        # Translate the x-axis values (first row of histograms)
        #print("hist_batch shape", hist_batch.shape)
        #translated_hist_batch = hist_batch[:,valid_indices] # hist_batch[valid_indices[valid_mask]]
        #print(hist_batch.shape, valid_indices.shape)
        #print("valid_indices", valid_indices)
        #print("hist_batch.shape", hist_batch.shape)
        translated_hist_batch = torch.gather(hist_batch, -1, valid_indices)
        #print("uh oh no gather")
        translated_hist_batch = torch.where(valid_mask, translated_hist_batch, torch.zeros_like(translated_hist_batch))
        #print("kay we got through")
        # Copy the y-axis values (second row) without modification
        #translated_hist_batch[:, 1, :] = hist_batch[:, 1, :]
    
        return translated_hist_batch


#batch_size = 2
#hist_batch = torch.rand(batch_size, 5)  # Create a batch of 4 random 2x50 histograms
#shift_tensors = torch.rand(batch_size)  # Shifts by 10%, 50%, 75%, and 30%
#print("in", hist_batch)
#translated_hist_batch = Model.batch_translate_histograms(hist_batch, shift_tensors)
#print("out", translated_hist_batch)
#indices = torch.tensor([[-4, -3, -2, -1,  0], [-4, -3, -2, -1,  0]])
#print(indices.shape)
#print(hist_batch.T[indices].shape)

#model(torch.randn(4, 5, 36, device=model.device))
#a = torch.arange(3).repeat(2,1)
#print(a)
#print("\nTranslated Histogram Batch:")

#print(translated_hist_batch)
#indices = torch.tensor([[1,0,1],[1,1,0]])
#print("indices", indices, indices.shape)

#print(a[indices], a[indices].shape)

#torch.index_select(input, dim, indices,


def find_good_offset_problem(model, iterations=10000, offset_trials=100, max_offset=0.2, beamline_trials=1000, fixed_parameters=[8, 14, 20, 21, 27, 28]):    
    for i in tqdm.tqdm(range(iterations)):
        offsets = (torch.rand(1, offset_trials, 34, device=model.device) * max_offset * 2) - max_offset
        uncompensated_parameters = torch.rand(beamline_trials, 1, offsets.shape[-1], device=model.device)
        uncompensated_parameters[:,:,fixed_parameters] = uncompensated_parameters[0,0,fixed_parameters].unsqueeze(0).unsqueeze(1)
        tensor_sum = offsets + uncompensated_parameters
        tensor_sum = torch.clamp(tensor_sum, 0, 1)
        uncompensated_parameters = tensor_sum - offsets
        with torch.no_grad():
            compensated_rays = model(tensor_sum)
            #condition = compensated_rays.sum(dim=-1) > 1.3
            condition = (compensated_rays.sum(dim=-1)>0.5).sum(dim=0)>15
            if condition.any():
                result = (compensated_rays.sum(dim=-1)>0.5).sum(dim=0)[condition]
                print(str(len(result))+" results.")
                condition_args = torch.arange(len(condition), device=model.device)[condition][:1]
                mask = compensated_rays[:, condition_args[0]].sum(dim=-1)>0.5
                to_plot_tensor = tensor_sum[:, condition_args][mask]
                uncompensated_parameters_selected = uncompensated_parameters[:, condition_args][mask]
                offsets_selected = offsets[:, condition_args]
                break
    compensated_parameters_selected = uncompensated_parameters_selected+offsets_selected
    return offsets_selected, uncompensated_parameters_selected, compensated_parameters_selected

def optimize_brute(model, observed_rays, uncompensated_parameters, iterations, offset_trials=1000000, max_offset=0.2):
    observed_rays = observed_rays.to(model.device)
    loss_min = float('inf')
    pbar = tqdm.trange(iterations)
    loss_min_params = None
    
    with torch.no_grad():
        loss_min_list = []
        for i in pbar:
            offsets = (torch.rand((1, offset_trials, uncompensated_parameters.shape[-1]), device=model.device) * max_offset * 2) - max_offset
            tensor_sum = offsets + uncompensated_parameters
            
            compensated_rays = model(tensor_sum)
            loss_orig = ((compensated_rays - observed_rays) ** 2).mean(0).mean(-1)
            loss = loss_orig.min()
            if loss < loss_min:
                loss_min = loss
                loss_min_params = tensor_sum[:, loss_orig.argmin(), :]
                pbar.set_postfix({"loss": loss_min.item()})
            loss_min_list.append(loss_min)
    return loss_min_params, loss_min, loss_min_list

def optimize_smart_walker(model, observed_rays, uncompensated_parameters, iterations, num_candidates=100000, max_offset=0.2, step_width=0.02):
    loss_min = float('inf')
    loss_min_list = []
    offsets = (torch.rand(1, num_candidates, uncompensated_parameters.shape[-1], device=model.device) * max_offset * 2) - max_offset
    
    pbar = tqdm.trange(iterations)
    with torch.no_grad():
        for i in pbar:
            offsets = offsets + (torch.randn(offsets.shape[0], num_candidates, offsets.shape[-1], device=model.device) * step_width)
            offsets = torch.clamp(offsets, -max_offset, max_offset)
            tensor_sum = offsets + uncompensated_parameters
            
            compensated_rays = model(tensor_sum)
            compensated_rays = compensated_rays.flatten(start_dim=2)
            loss_orig = ((compensated_rays - observed_rays) ** 2).mean(0).mean(-1)
        
            loss = loss_orig.min()
            offsets = offsets[:, loss_orig.argmin(), :].unsqueeze(dim=1)
            if loss < loss_min:
                loss_min = loss.item()
                loss_min_params = tensor_sum[:, loss_orig.argmin(), :]
                pbar.set_postfix({"loss": loss.item()})
            loss_min_list.append(loss_min)
    return loss_min_params, loss_min, loss_min_list

# get mean of best solutions, std of best solutions, tensor of means of progress, tensor of std of progress
def evaluate_evaluation_method(method, model, observed_rays, uncompensated_parameters, iterations=1000, repetitions=10):
    loss_list = []
    loss_min_tens_list = []
    for i in range(repetitions):
        loss_min_params, loss, loss_min_list = method(model, observed_rays, uncompensated_parameters, iterations=iterations)
        #print(loss_min_list[0].shape)
        loss_min_tens_list.append(torch.tensor(loss_min_list, device=model.device))
        loss_list.append(loss)
    losses = torch.tensor(loss_list)
    loss_min_tens_tens = torch.vstack(loss_min_tens_list)
    return losses.mean().item(), losses.std().item(), loss_min_tens_tens.mean(dim=0), loss_min_tens_tens.std(dim=0)

def simulate_param_tensor(input_param_tensor, engine):
    #print("simulating", input_param_tensor.shape)
    pc = tensor_list_to_param_container_list(input_param_tensor[...,:34])
    param_container_list = []
    for i in input_param_tensor:
        i = i.squeeze()
        param_container_list.append(
            RayParameterContainer([
            ('ImagePlane.translationXerror', NumericalOutputParameter(value=(i[-2].item()-0.5)*20)),
            ('ImagePlane.translationYerror', NumericalOutputParameter(value=(i[-1].item()-0.5)*4)),
            ]))
    #print("pcl",param_container_list)

    #ray_parameter_container_list = []
    #for key in ("translationXerror", "translationYerror"):
    #    ray_parameter_container_list.append(key

    #RayParameterContainer(['U41_318eV.numberRays', OutputNumericalParameter(value=33333)])
        
    exported_plain_transforms = RayOptimizer.translate_exported_plain_transforms(
        exported_plane="ImagePlane",
        param_container_list=param_container_list,
        transform = MultiLayer([0.]),
    )
    #print("ept",exported_plain_transforms)
    engine_output = engine.run(pc, MultiLayer([0.])) ###exported_plain_transforms
    return ray_output_to_tensor(engine_output, 'ImagePlane')

def plot_param_tensors(best_parameters, uncompensated_parameters, engine, observed_rays_point_cloud=None, compensated_parameters=None):
    assert observed_rays_point_cloud is not None or compensated_parameters is not None # you should provide one of two
    if compensated_parameters is not None:
        observed_rays = simulate_param_tensor(compensated_parameters, engine)
    else:
        observed_rays = observed_rays_point_cloud
    best_rays = simulate_param_tensor(best_parameters, engine)
    uncompensated_rays = simulate_param_tensor(uncompensated_parameters, engine)

    x_means = []
    y_means = []
    for entry in observed_rays:
        x_means.append(entry[0, :, 0].mean().item())
        y_means.append(entry[0, :, 1].mean().item())
    y_lim_min = [entry - 1.0 for entry in y_means]
    y_lim_max = [entry + 1.0 for entry in y_means]

    x_lim_min = [entry - 1.0 for entry in x_means]
    x_lim_max = [entry + 1.0 for entry in x_means]
       
    
    compensation_plot = Plot.fixed_position_plot(
    #compensation_plot = Plot.compensation_plot(
        best_rays,
        observed_rays,
        uncompensated_rays,
        xlim=(x_lim_min, x_lim_max),#(-10.,10.),#
        ylim=(y_lim_min, y_lim_max),#(-2.,2.),#
    )
    return compensation_plot

def tensor_list_to_param_container_list(input_param_tensor):
    return [tensor_to_param_container(input_param_tensor[i].squeeze()) for i in range(input_param_tensor.shape[0])]
