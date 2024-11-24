import glob
from tqdm.auto import tqdm, trange
import torch
import optuna
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



def find_good_offset_problem(model, iterations=10000, offset_trials=1, beamline_trials=10000, fixed_parameters=[8, 14, 20, 21, 27, 28], z_array=[-25., -20., -15., -10., -5., 0., 5., 10., 15., 20., 25., 30.], z_array_label='ImagePlane.translationZerror'):
    mutable_parameter_count = model.mutable_parameter_count
    assert z_array_label in model.input_parameter_container.keys()
    z_array_min, z_array_max = model.input_parameter_container[z_array_label].value_lims
    normalized_z_array = torch.tensor((z_array - z_array_min) / (z_array_max - z_array_min), device=model.device).float()
    
    for i in range(iterations):
        offsets = torch.rand(offset_trials, mutable_parameter_count, device=model.device)
        uncompensated_parameters = torch.rand(beamline_trials, 1, 1, offsets.shape[-1], device=model.device)
        uncompensated_parameters[:,:,:,fixed_parameters] = uncompensated_parameters[0,0,0,fixed_parameters].unsqueeze(0).unsqueeze(1)
        uncompensated_parameters = torch.repeat_interleave(uncompensated_parameters, len(z_array), dim=1)
        uncompensated_parameters[:, :, 0, -1] = normalized_z_array
        scaled_offsets = model.rescale_offset(offsets)
        tensor_sum = scaled_offsets + uncompensated_parameters
        mask = (tensor_sum > 0.) & (tensor_sum  < 1.)
        all_params_true = mask.all(dim=(-1, -2,-3))
        uncompensated_parameters = uncompensated_parameters[all_params_true]
        tensor_sum = tensor_sum[all_params_true]
        if i % 50 == 0 and i != 0:
            print(f"Epoch {i}: Still searching.")
        if tensor_sum.shape[0] == 0:
            continue
        uncompensated_parameters = tensor_sum - scaled_offsets
        with torch.no_grad():
            compensated_rays = model(tensor_sum)
            sufficient_sum_per_hist = compensated_rays.sum(dim=-1)>0.5
            beamline_trials_with_sufficient_hist = sufficient_sum_per_hist.sum(dim=0)>15.
            all_z_trials_sufficient_hist = beamline_trials_with_sufficient_hist.sum(dim=0)==len(z_array)
            condition = all_z_trials_sufficient_hist
            if all_z_trials_sufficient_hist.any():
                condition_args = torch.arange(len(condition), device=model.device)[condition]
                condition_args = condition_args[:1]
                sufficient_sum_per_hist_selected = compensated_rays[:, :, condition_args[0]].sum(dim=-1)>0.5
                mask = sufficient_sum_per_hist_selected.sum(dim=-1) == len(z_array)
                uncompensated_parameters_selected = uncompensated_parameters[:,:, condition_args][mask]
                offsets_selected = offsets[condition_args]
                scaled_offsets_selected = scaled_offsets[condition_args]
                break
    compensated_parameters_selected = uncompensated_parameters_selected+scaled_offsets_selected
    return offsets_selected, uncompensated_parameters_selected, compensated_parameters_selected

def evaluate_evaluation_method(method, model, num_candidates=1000000, iterations=1000, repetitions=10):
    loss_list = []
    loss_min_tens_list = []
    _, uncompensated_parameters, _ = find_good_offset_problem(model, fixed_parameters = [8, 14, 20, 21, 27, 28]) # only for getting the shape
    loss_min_params_tens = torch.full((repetitions, uncompensated_parameters.shape[1], uncompensated_parameters.shape[-1]), float('nan'), device=model.device)
    offset_rmse_tens = torch.full((repetitions,), float('nan'), device=model.device)
    for i in range(repetitions):
        offsets, uncompensated_parameters, compensated_parameters = find_good_offset_problem(model, fixed_parameters = [8, 14, 20, 21, 27, 28])
        with torch.no_grad():
            observed_rays = model(compensated_parameters)
        loss_min_params, loss, loss_min_list = method(model, observed_rays, uncompensated_parameters, iterations=iterations, num_candidates=num_candidates)
        predicted_offsets = loss_min_params[0, 0] - uncompensated_parameters[0, 0, 0]
        normalized_predicted_offsets = model.unscale_offset(predicted_offsets)
        offset_rmse = ((offsets-normalized_predicted_offsets)**2).mean().sqrt()
        offset_rmse_tens[i] = offset_rmse
        loss_min_tens_list.append(torch.tensor(loss_min_list, device=model.device))
        loss_min_params_tens[i] = loss_min_params[0]
        loss_list.append(loss)
    losses = torch.tensor(loss_list)
    loss_min_tens_tens = torch.vstack(loss_min_tens_list)
    return losses, loss_min_tens_tens.mean(dim=0), loss_min_tens_tens.std(dim=0), loss_min_params_tens, offset_rmse_tens

''' input shape
iterations x samples x parameters
output shape
ray output list of iterations of lists of samples
'''
def simulate_param_tensor(input_param_tensor, engine, ray_parameter_container, exported_plane='ImagePlane'):
    assert len(input_param_tensor.shape) == 3
    pc = tensor_list_to_param_container_list(input_param_tensor[:,0], ray_parameter_container)
    for entry in pc:
        entry[exported_plane+'.translationZerror'].value = 0.
        entry['U41_318eV.numberRays'].value=10000

    z_min, z_max = ray_parameter_container[exported_plane+'.translationZerror'].value_lims
    z_layers = input_param_tensor[0,:,-1] * (z_max-z_min) + z_min
    z_layer_list = z_layers.tolist()
    
    engine_output = engine.run(pc, MultiLayer(z_layer_list))
    return [ray_dict_to_tensor(entry, exported_plane, to_cpu=True) for entry in engine_output]
    
def fancy_plot_param_tensors(best_parameters, uncompensated_parameters, engine, ray_parameter_container, observed_rays_point_cloud=None, compensated_parameters=None):
    assert observed_rays_point_cloud is not None or compensated_parameters is not None # you should provide one of two
    if compensated_parameters is not None:
        observed_rays = simulate_param_tensor(compensated_parameters, engine, ray_parameter_container)
    else:
        observed_rays = observed_rays_point_cloud
    best_rays = simulate_param_tensor(best_parameters, engine, ray_parameter_container)
    uncompensated_rays = simulate_param_tensor(uncompensated_parameters, engine, ray_parameter_container)
    return Plot.fancy_ray(
        [uncompensated_rays, observed_rays, best_rays],
        [ "Uncompensated", "Experiment", "Compensated"],
        z_index=[-25., -20., -15., -10., -5., 0., 5., 10., 15., 20., 25., 30.]
    )

    

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

def compare_with_reference(reference_ray_outputs, loss_min_ray_outputs):
    loss_fn = SamplesLoss("sinkhorn", blur=0.1)
    distances_list = []
    
    for i in range(len(loss_min_ray_outputs)):
        distances_list.append(loss_fn(loss_min_ray_outputs[i].contiguous(), reference_ray_outputs[i].contiguous()))
    distances = torch.concat(distances_list)
    return distances.mean(), distances.std()

def compare_with_reference_bak(reference_ray_outputs, compensated_ray_outputs, output_plane='ImagePlane'):
    loss_fn = SamplesLoss("sinkhorn", blur=0.1)
    distances_list = []
    for repetition in tqdm(compensated_ray_outputs, leave=False):
        distances_rep_list = []
        for i in range(len(reference_ray_outputs)):
            sinkhorn_distance = loss_fn(repetition.contiguous(), reference_ray_outputs[i].contiguous())
            distances_rep_list.append(sinkhorn_distance)
        distances_rep = torch.concat(distances_rep_list)
        distances_list.append(distances_rep)
    distances = torch.stack(distances_list)
    return distances.mean().item(), distances.std().item()

def optimize_brute(model, observed_rays, uncompensated_parameters, iterations, num_candidates=1000000):
    loss_min = float('inf')
    pbar = trange(iterations, leave=False)
    loss_min_params = None
    
    with torch.no_grad():
        loss_min_list = []
        for i in pbar:
            offsets = torch.rand((1, num_candidates, uncompensated_parameters.shape[-1]), device=model.device)
            scaled_offsets = model.rescale_offset(offsets)
            tensor_sum = uncompensated_parameters+scaled_offsets
            compensated_rays = model(tensor_sum)
            loss_orig = ((compensated_rays - observed_rays) ** 2).mean(dim=(0, 1, -1))
            loss = loss_orig.min()
            if loss < loss_min:
                loss_min = loss
                loss_min_params = tensor_sum[:, :, loss_orig.argmin(), :]
                pbar.set_postfix({"loss": loss_min.item()})
            loss_min_list.append(loss_min)
    return loss_min_params, loss_min, loss_min_list

def optimize_smart_walker(model, observed_rays, uncompensated_parameters, iterations, num_candidates=100000, step_width=0.02):
    loss_min = float('inf')
    loss_min_list = []
    offsets = torch.rand(1, num_candidates, uncompensated_parameters.shape[-1], device=model.device)
    
    pbar = trange(iterations, leave=False)
    with torch.no_grad():
        for i in pbar:
            offsets = offsets + (torch.randn(offsets.shape[0], num_candidates, offsets.shape[-1], device=model.device) * step_width)
            offsets = torch.clamp(offsets, 0, 1)
            scaled_offsets = model.rescale_offset(offsets)
            compensated_rays = model(uncompensated_parameters+scaled_offsets)
            loss_orig = ((compensated_rays - observed_rays) ** 2).mean(0).mean(0).mean(-1)
        
            loss = loss_orig.min()
            offsets = offsets[:, loss_orig.argmin(), :].unsqueeze(dim=1)
            if loss < loss_min:
                loss_min = loss.item()
                tensor_sum = scaled_offsets + uncompensated_parameters
                loss_min_params = tensor_sum[:,:, loss_orig.argmin(), :]
                pbar.set_postfix({"loss": loss.item()})
            loss_min_list.append(loss_min)
    return loss_min_params, loss_min, loss_min_list

def optimize_tpe(model, observed_rays, uncompensated_parameters, iterations, num_candidates=None):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def objective_function(offsets, model, observed_rays, uncompensated_parameters):
        # Evaluate the model's loss with these offsets
        scaled_offsets = model.rescale_offset(offsets)
        compensated_rays = model(uncompensated_parameters + scaled_offsets)
        loss = ((compensated_rays - observed_rays) ** 2).mean().item()
        return loss
    
    def objective(trial):
        param_list = torch.tensor([trial.suggest_float(str(i), 0, 1.) for i in range(uncompensated_parameters.shape[-1])], device=model.device)
        return objective_function(param_list, model, observed_rays, uncompensated_parameters)
    
    study = optuna.create_study()
    
    study.optimize(objective, n_trials=iterations, show_progress_bar=True)
    
    losses = torch.tensor([trial.value for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE], device=model.device)
    best_losses = torch.cummin(losses, dim=0).values
    global_best_offset = torch.tensor(list(study.best_params.values()), device=model.device)
    loss_min_params = model.rescale_offset(global_best_offset) + uncompensated_parameters
    return loss_min_params.squeeze(-2), study.best_value, best_losses.tolist()
    
def optimize_pso(model, observed_rays, uncompensated_parameters, iterations, num_candidates=100000, step_width=0.02):
    loss_min = float('inf')
    loss_min_list = []
    
    # Initialize particles randomly in the parameter space
    offsets = torch.rand(1, num_candidates, uncompensated_parameters.shape[-1], device=model.device)
    velocities = torch.randn_like(offsets) * step_width  # Initialize velocities
    
    # Initialize personal and global bests
    personal_best_offsets = offsets.clone()
    personal_best_loss = torch.full((1, num_candidates), float('inf'), device=model.device)
    global_best_offset = offsets[:, 0, :].unsqueeze(dim=1)  # Start with first particle
    global_best_loss = float('inf')
    
    pbar = trange(iterations, leave=False)
    with torch.no_grad():
        for i in pbar:
            # Rescale and evaluate the current offsets
            scaled_offsets = model.rescale_offset(offsets)
            compensated_rays = model(uncompensated_parameters + scaled_offsets)
            losses = ((compensated_rays - observed_rays) ** 2).mean(dim=(0, 1, -1))
            
            # Update personal bests
            improved_mask = losses < personal_best_loss
            personal_best_loss[improved_mask] = losses[improved_mask.flatten()]
            personal_best_offsets[:, improved_mask.flatten(), :] = offsets[ :,improved_mask.flatten(), :]
            
            # Update global best
            min_loss, min_index = personal_best_loss.min(dim=1)
            if min_loss < global_best_loss:
                global_best_loss = min_loss.item()
                global_best_offset = personal_best_offsets[:, min_index, :]
                
            # Update particle velocities and positions
            inertia = 0.5
            cognitive = 1.5
            social = 1.5
            velocities = (inertia * velocities
                          + cognitive * torch.rand_like(offsets) * (personal_best_offsets - offsets)
                          + social * torch.rand_like(offsets) * (global_best_offset - offsets))
            
            offsets = offsets + velocities
            offsets = torch.clamp(offsets, 0, 1)  # Keep within bounds
            
            # Track the best loss for progress reporting
            loss_min_list.append(global_best_loss)
            pbar.set_postfix({"loss": global_best_loss})
    
    # Return the best parameters found
    loss_min_params = model.rescale_offset(global_best_offset) + uncompensated_parameters
    return loss_min_params.squeeze(-2), global_best_loss, loss_min_list


def optimize_ea(
    model, observed_rays, uncompensated_parameters, iterations, 
    num_candidates=100, mutation_rate=0.1, crossover_rate=0.7
):
    # Initialize population
    population = torch.rand(num_candidates, uncompensated_parameters.shape[-1], device=model.device)
    best_individual = None
    best_loss = float('inf')
    loss_history = []

    pbar = trange(iterations, leave=False)
    with torch.no_grad():
        for _ in pbar:
            # Rescale offsets and evaluate fitness
            scaled_population = model.rescale_offset(population.unsqueeze(0).unsqueeze(0))  # Shape: (1, pop_size, params_dim)
            compensated_rays = model(uncompensated_parameters + scaled_population)
            losses = ((compensated_rays - observed_rays) ** 2).mean(0).mean(0).mean(-1)  # Shape: (pop_size,)
            
            # Update best solution
            min_loss, min_idx = losses.min(dim=0)
            if min_loss < best_loss:
                best_loss = min_loss.item()
                best_individual = population[min_idx].clone()
                pbar.set_postfix({"best_loss": best_loss})
            
            loss_history.append(best_loss)

            # Selection (Tournament Selection)
            tournament_size = 3
            selected_indices = torch.randint(0, num_candidates, (num_candidates, tournament_size), device=model.device)
            tournament_fitness = losses[selected_indices]  # Shape: (num_candidates, tournament_size)
            best_indices_in_tournaments = torch.argmin(tournament_fitness, dim=1)  # Shape: (num_candidates,)
            selected = selected_indices[torch.arange(num_candidates), best_indices_in_tournaments] 

            # Crossover

            parent1 = population[selected[::2]]
            parent2 = population[selected[1::2]]
            
            if parent1.shape[0] > parent2.shape[0]:
                parent2 = torch.cat([parent2, parent2[:1]], dim=0)
            
            # Random crossover decisions
            crossover_mask = (torch.rand(parent1.shape[0], 1, device=model.device) < crossover_rate).expand(-1, parent1.shape[1])
            
            # Random crossover points
            crossover_points = torch.randint(
                0, parent1.shape[1], 
                (parent1.shape[0], 1), 
                device=model.device
            ).expand(-1, parent1.shape[1])
            
            # Generate masks for slicing
            indices = torch.arange(parent1.shape[1], device=model.device)
            crossover_mask = indices < crossover_points
            
            # Create offspring
            offspring1 = torch.where(crossover_mask, parent1, parent2)
            offspring2 = torch.where(crossover_mask, parent2, parent1)
            
            # Combine offspring
            offspring = torch.cat([offspring1, offspring2], dim=0)

            # Mutation
            #offspring = torch.stack(offspring)
            mutation_mask = torch.rand_like(offspring) < mutation_rate
            mutations = torch.randn_like(offspring) * 0.1  # Scale mutations
            offspring = offspring + mutation_mask * mutations
            offspring = torch.clamp(offspring, 0, 1)  # Ensure valid range

            # Replace population
            population = offspring
    # Return the best solution found
    loss_min_params = model.rescale_offset(best_individual) + uncompensated_parameters
    return loss_min_params.squeeze(-2), best_loss, loss_history
