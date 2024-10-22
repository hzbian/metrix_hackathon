import glob
import pickle
import tqdm
import torch
import matplotlib.pyplot as plt

from ray_optim.plot import Plot
from ray_tools.base.transform import MultiLayer
from ray_nn.data.lightning_data_module import DefaultDataModule
from ray_nn.nn.xy_hist_data_models import MetrixXYHistSurrogate
from geomloss import SamplesLoss
from ray_tools.base.parameter import NumericalParameter, OutputParameter, NumericalOutputParameter, MutableParameter
from datasets.metrix_simulation.config_ray_emergency_surrogate import TRANSFORMS as cfg_transforms
from ray_tools.base.parameter import NumericalParameter, NumericalOutputParameter, RayParameterContainer
from ray_tools.simulation.torch_datasets import BalancedMemoryDataset, MemoryDataset, RayDataset
from sub_projects.ray_optimization.utils import ray_dict_to_tensor, ray_output_to_tensor
from sub_projects.ray_optimization.ray_optimization import RayOptimization
from ray_optim.ray_optimizer import RayOptimizer

from ray_tools.base.utils import RandomGenerator
from sub_projects.ray_optimization.real_data import import_data
from ray_tools.base.transform import XYHistogram
from ray_nn.data.transform import Select


def tensor_to_param_container(ten, ray_parameter_container: RayParameterContainer):
    assert sum([1 if isinstance(value, MutableParameter) else 0 for key, value in ray_parameter_container.items()]) == ten.shape[0]
    #assert ten.min() >= 0. and ten.max() <= 1.
    param_dict_list = []
    i=0
    for label, entry in ray_parameter_container.items():
        if not isinstance(entry, MutableParameter):
            param_dict_list.append((label, entry))
        else:
            value = ten[i]*(entry.value_lims[1]-entry.value_lims[0])+entry.value_lims[0]
            if not isinstance(entry, OutputParameter):
                param_dict_list.append((label, NumericalParameter(value.item())))
            else:
                param_dict_list.append((label, NumericalOutputParameter(value.item())))
            #if value.item() < entry.value_lims[0] or value.item() > entry.value_lims[1]:
            #    if value.item() < entry.value_lims[0]:
            #        value = torch.ones_like(value) * entry.value_lims[0]
            #    elif value.item() > entry.value_lims[1]:
            #        value = torch.ones_like(value) * entry.value_lims[1]
                #raise Exception("Out of range. Minimum was {}, maximum {} but value {}. Tensor value was {}.".format(entry.value_lims[0], entry.value_lims[1], value.item(), ten[i-1].item()))
            i = i+1

    return RayParameterContainer(param_dict_list)

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



def find_good_offset_problem(model, iterations=10000, offset_trials=100, max_offset=0.2, beamline_trials=1000, fixed_parameters=[8, 14, 20, 21, 27, 28], z_array=[-33., 0., 5., 10.], z_array_label='ImagePlane.translationZerror'):
    mutable_parameter_count = model.mutable_parameter_count
    assert z_array_label in model.input_parameter_container.keys()
    z_array_min, z_array_max = model.input_parameter_container[z_array_label].value_lims
    normalized_z_array = torch.tensor((z_array - z_array_min) / (z_array_max - z_array_min), device=model.device).float()
    
    for i in tqdm.tqdm(range(iterations)):
        offsets = (torch.rand(1, offset_trials, mutable_parameter_count, device=model.device) * max_offset * 2) - max_offset
        uncompensated_parameters = torch.rand(beamline_trials, 1, 1, offsets.shape[-1], device=model.device)
        uncompensated_parameters[:,:,:,fixed_parameters] = uncompensated_parameters[0,0,0,fixed_parameters].unsqueeze(0).unsqueeze(1)
        uncompensated_parameters = torch.repeat_interleave(uncompensated_parameters, len(z_array), dim=1)
        print(offsets.shape, uncompensated_parameters.shape)
        print(uncompensated_parameters[0, :, 0, -1])
        print(uncompensated_parameters[0, :, 0, -1]+normalized_z_array)
        return None, None, None
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

def optimize_brute(model, observed_rays, uncompensated_parameters, iterations, num_candidates=1000000, max_offset=0.2):
    loss_min = float('inf')
    pbar = tqdm.trange(iterations)
    loss_min_params = None
    
    with torch.no_grad():
        loss_min_list = []
        for i in pbar:
            offsets = (torch.rand((1, num_candidates, uncompensated_parameters.shape[-1]), device=model.device) * max_offset * 2) - max_offset
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
def evaluate_evaluation_method(method, model, observed_rays, uncompensated_parameters, offsets, max_offset=0.2, num_candidates=1000000, iterations=1000, repetitions=10):
    loss_list = []
    loss_min_tens_list = []
    for i in range(repetitions):
        loss_min_params, loss, loss_min_list = method(model, observed_rays, uncompensated_parameters, iterations=iterations, num_candidates=num_candidates, max_offset=max_offset, )
        loss_min_tens_list.append(torch.tensor(loss_min_list, device=model.device))
        if i == 0:
           loss_min_params_tens = torch.empty((0, loss_min_params[0].shape[-1]), device=model.device)
        loss_min_params_tens = torch.vstack((loss_min_params_tens, loss_min_params[0]))
        loss_list.append(loss)
    losses = torch.tensor(loss_list)
    loss_min_tens_tens = torch.vstack(loss_min_tens_list)
    return losses.mean().item(), losses.std().item(), loss_min_tens_tens.mean(dim=0), loss_min_tens_tens.std(dim=0), loss_min_params_tens

def simulate_param_tensor(input_param_tensor, engine, ray_parameter_container):
    #print("simulating", input_param_tensor.shape)
    pc = tensor_list_to_param_container_list(input_param_tensor, ray_parameter_container)
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

def plot_param_tensors(best_parameters, uncompensated_parameters, engine, ray_parameter_container, observed_rays_point_cloud=None, compensated_parameters=None):
    assert observed_rays_point_cloud is not None or compensated_parameters is not None # you should provide one of two
    if compensated_parameters is not None:
        observed_rays = simulate_param_tensor(compensated_parameters, engine, ray_parameter_container)
    else:
        observed_rays = observed_rays_point_cloud
    best_rays = simulate_param_tensor(best_parameters, engine, ray_parameter_container)
    uncompensated_rays = simulate_param_tensor(uncompensated_parameters, engine, ray_parameter_container)

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

def tensor_list_to_param_container_list(input_param_tensor, ray_parameter_container):
    return [tensor_to_param_container(input_param_tensor[i].squeeze(), ray_parameter_container) for i in range(input_param_tensor.shape[0])]


''' input shape
iterations x samples x parameters
output shape
ray output list of iterations of lists of samples
'''
def param_tensor_to_ray_outputs(input_tens, engine, ray_paramer_container):
    out_list = []
    for entry in input_tens:
        param_container_list = tensor_list_to_param_container_list(entry, ray_paramer_container)
        out = engine.run(param_container_list, MultiLayer([0.]))
        out_list.append(out)
    return out_list

def compare_with_reference(reference_ray_outputs, compensated_parameters_selected_ray_outputs, output_plane='ImagePlane'):
    loss_fn = SamplesLoss("sinkhorn", blur=0.1)
    distances_list = []
    for repetition in tqdm.tqdm(compensated_parameters_selected_ray_outputs):
        distances_rep_list = []
        for i in range(len(reference_ray_outputs[0])):
            sinkhorn_distance = loss_fn(ray_dict_to_tensor(repetition[i], output_plane).contiguous(), ray_dict_to_tensor(reference_ray_outputs[0][i], output_plane).contiguous())
            distances_rep_list.append(sinkhorn_distance)
        distances_rep = torch.concat(distances_rep_list)
        distances_list.append(distances_rep)
    distances = torch.stack(distances_list)
    return distances.mean().item(), distances.std().item()
