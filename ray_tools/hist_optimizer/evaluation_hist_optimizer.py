#!/usr/bin/env python
# coding: utf-8

# In[17]:


import torch
import sys
import glob
import os
from tqdm.auto import tqdm, trange
sys.path.append("..")
sys.path.append("../..")
import matplotlib.pyplot as plt
from ray_tools.hist_optimizer.hist_optimizer import tensor_to_param_container, mse_engines_comparison, find_good_offset_problem, optimize_tpe, optimize_smart_walker, optimize_brute, optimize_pso, optimize_ea, evaluate_evaluation_method, plot_param_tensors, tensor_list_to_param_container_list, simulate_param_tensor, compare_with_reference, fancy_plot_param_tensors
from scipy.stats import ttest_rel
from ray_tools.base.transform import MultiLayer
from ray_tools.base.engine import RayEngine
from ray_nn.data.lightning_data_module import DefaultDataModule
from ray_nn.nn.xy_hist_data_models import MetrixXYHistSurrogate, StandardizeXYHist, HistSurrogateEngine, Model
from ray_tools.base.backend import RayBackendDockerRAYUI
from ray_tools.simulation.torch_datasets import BalancedMemoryDataset, MemoryDataset, RayDataset
from ray_optim.plot import Plot
from ray_tools.base.transform import Histogram, RayTransformConcat, XYHistogram
from ray_nn.data.transform import Select
from ray_tools.base.parameter import NumericalParameter, OutputParameter, NumericalOutputParameter, MutableParameter, RayParameterContainer
from sub_projects.ray_optimization.real_data import import_data
from sub_projects.ray_optimization.utils import ray_dict_to_tensor, ray_output_to_tensor
from scipy.stats import kstest
from torch.utils import benchmark
import plotly.io as pio
torch.manual_seed(42)


# In[18]:


file_root = ''
outputs_dir = os.path.join(file_root, 'outputs/')
engine = RayEngine(rml_basefile=os.path.join(file_root,'rml_src/METRIX_U41_G1_H1_318eV_PS_MLearn_1.15.rml'),
                                exported_planes=["ImagePlane"],
                                ray_backend=RayBackendDockerRAYUI(docker_image='ray-ui-service',
                                                                  docker_container_name='ray-ui-service-test',
                                                                  dockerfile_path=os.path.join(file_root, 'ray_docker/rayui'),
                                                                  ray_workdir='/dev/shm/ray-workdir',
                                                                  verbose=False),
                                num_workers=-1,
                                as_generator=False)


model_path = os.path.join(file_root, "outputs/xy_hist/ft1rr9h0/checkpoints/epoch=70-step=67568996.ckpt")
surrogate_engine = HistSurrogateEngine(checkpoint_path=model_path)

model = Model(path=model_path)


# In[3]:


offsets_selected, uncompensated_parameters_selected, compensated_parameters_selected = find_good_offset_problem(model, fixed_parameters = [8, 14, 20, 21, 27, 28])

with torch.no_grad():
    observed_rays = model(compensated_parameters_selected)


# In[4]:


offsets_list = []
uncompensated_parameters_list = []
compensated_parameters_list = []
for i in trange(10000):
    offsets_trial, uncompensated_parameters_trial, compensated_parameters_trial = find_good_offset_problem(model)
    offsets_list.append(offsets_trial)
    uncompensated_parameters_list.append(uncompensated_parameters_trial)
    compensated_parameters_list.append(compensated_parameters_trial)

offsets_list = torch.stack(offsets_list)


# In[37]:


def correlation_matrix(data, model, label):
    data = data.squeeze().cpu()
    # Standardize the data (optional but often necessary)
    mean = data.mean(dim=0, keepdim=True)
    std = data.std(dim=0, keepdim=True)
    data_standardized = (data - mean) / std
    
    # Compute the correlation matrix
    correlation_matrix = torch.mm(data_standardized.T, data_standardized) / (data_standardized.size(0) - 1)
    
    # Convert the correlation matrix to a numpy array for plotting
    correlation_matrix_np = correlation_matrix.numpy()
    print("min-entry", correlation_matrix.min(), "max-entry", (correlation_matrix-torch.eye(correlation_matrix.shape[0])).max())
    # Plot the heatmap using matplotlib
    plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
    cax = plt.imshow(correlation_matrix_np, cmap='RdYlBu_r', vmin=-1, vmax=1)
    
    # Add color bar
    plt.colorbar(cax, label="Correlation")
    
    # Set title and labels
    plt.title("Correlation matrix "+label)
    plt.xlabel("Features")
    plt.ylabel("Features")
    plt.grid(True, alpha=0.3)
    
    # Disable the axis ticks (both x and y)
    labels = [key for key, value in model.input_parameter_container.items() if isinstance(value, MutableParameter)]
    labels = labels[:data.shape[-1]] # omit translationZerror if not in data
    plt.yticks(ticks = torch.arange(len(labels)), labels=labels)  # Round bin edges for readability
    plt.xticks(ticks = torch.arange(len(labels)), labels=labels, rotation=90)  # Round bin edges for readability
    
    # Show the plot
    plt.savefig(os.path.join(outputs_dir,'cor_mat_'+label.replace(" ", "_")+'.pdf'), bbox_inches='tight')
    return plt.gcf()

correlation_matrix(offsets_list, model, "offsets")


# In[38]:


stacked_uncompensated_parameters = torch.vstack([entry[:, 0, 0, :36] for entry in uncompensated_parameters_list])
correlation_matrix(stacked_uncompensated_parameters, model, "uncompensated parameters")


# In[39]:


def correlation_plot(data, model, label, n_bins=15):
    # Example: Creating a random 1000x37 tensor (1000 samples, 37 features)
    data = data.squeeze().cpu() #torch.randn(1000, 37)  # Your tensor with 1000 samples and 37 features
    
    # Initialize a 2D array to store histogram values (one column per feature)
    hist_matrix = []
    
    # Loop through each feature (column) and compute its histogram
    bin_edges = None  # Initialize bin_edges to store the first feature's bin edges
    
    for i in range(data.shape[1]):  # Iterate over each feature
        feature_data = data[:, i].cpu()#.numpy()  # Convert the feature to a NumPy array
        counts, bin_edges = torch.histogram(feature_data, bins=n_bins, range=(feature_data.min().item(), feature_data.max().item()))
        hist_matrix.append(counts)
    
    # Convert the list of histograms to a NumPy array and transpose it (features as columns)
    
    # Plot the 2D histogram using imshow
    plt.figure(figsize=(8, 8))
    plt.imshow(hist_matrix, aspect='auto', vmin=0, cmap='hot', interpolation='none')
    
    # Add color bar
    plt.colorbar(label='Count [#]')
    
    # Label axes
    plt.ylabel('Parameter')
    plt.xlabel('Bin value [normalized]')
    plt.title(label+' distribution')
    labels = [key for key, value in model.input_parameter_container.items() if isinstance(value, MutableParameter)]
    labels = labels[:data.shape[-1]] # omit translationZerror if not in data
    plt.yticks(ticks = torch.arange(len(labels)), labels=labels)  # Round bin edges for readability
    plt.xticks(ticks=torch.arange(n_bins), labels=torch.round(bin_edges[:-1], decimals=2).numpy())  # Round bin edges for readability
    
    # Show the plot
    plt.savefig(os.path.join(outputs_dir,'cor_mat_'+label.replace(" ", "_")+'.pdf'), bbox_inches='tight')
correlation_plot(offsets_list, model, label="Offsets")


# In[40]:


uncompensated_parameters_stack = torch.vstack([entry[:, 0, 0, :36] for entry in uncompensated_parameters_list])
correlation_plot(uncompensated_parameters_stack, model, label="Uncompensated parameters")


# In[9]:


method_dict = {"Smart Walker": (optimize_smart_walker, 10000), "Brute Force": (optimize_brute, 10000), "TPE": (optimize_tpe, None), "PSO": (optimize_pso, 10000), "EA": (optimize_ea, 10000)}
method_evaluation_dict = {}

for key, entry in tqdm(method_dict.items(), desc="Evaluating methods"):
    loss_best, mean_progress, std_progress, loss_min_params_tens, offset_rmse = evaluate_evaluation_method(entry[0], model, repetitions=20, num_candidates=entry[1], iterations=1000)
    method_evaluation_dict[key]= (loss_best, mean_progress, std_progress, offset_rmse)

def run_optimizer(optimizer, model, observed_rays, uncompensated_parameters, iterations, num_candidates):
    return optimizer(model, observed_rays, uncompensated_parameters, iterations, num_candidates)

repetitions= 10
iterations = 1000
for key, (optimizer, num_candidates) in method_dict.items():
    t0 = benchmark.Timer(
        stmt='run_optimizer(optimizer, model, observed_rays, uncompensated_parameters_selected, iterations=iterations, num_candidates=num_candidates)',
        setup='from __main__ import run_optimizer',
        globals={'optimizer': optimizer, 'model': model, 'observed_rays': observed_rays, 'iterations': iterations, 'uncompensated_parameters_selected': uncompensated_parameters_selected, 'num_candidates': num_candidates},
        num_threads=1,
        label='optimize '+key,
        sub_label=None)
    timing_results = t0.blocked_autorange()
    timings = torch.tensor(timing_results.times)
    execution_time = torch.min(timings).item()
    method_evaluation_dict[key] += (execution_time,)


# In[45]:


plt.figure(figsize = (6.905, 4.434))
ax = plt.gca()
i = 0
plot_list = []
for key, (_, mean_progress, std_progress, _, _) in method_evaluation_dict.items():
    color = plt.rcParams["axes.prop_cycle"].by_key()["color"][i+3]
    plt.fill_between(torch.arange(len(mean_progress)), (mean_progress-std_progress).cpu(), (mean_progress+std_progress).cpu(), color=color, alpha=0.2)
    plot, = plt.plot(torch.arange(len(mean_progress)), mean_progress.cpu(), alpha = 1., c = color)
    plot_list.append(plot)
    i = i+1
ax.legend(plot_list, [key for key in method_dict.keys()], prop={'size': 11})
ax.tick_params(axis='both', which='major', labelsize=11)
plt.xlabel('Iteration [#]', fontsize=16)
plt.ylabel('Normalized MSE [log scale]', fontsize=16)
ax.set_yscale('log')
plt.tight_layout()
plt.savefig(os.path.join(outputs_dir,'bl_optimizer_iterations.pdf'), bbox_inches='tight', pad_inches = 0)


# In[11]:


@staticmethod
def significant_confidence_levels(group_A, group_B, confidence=0.99):
    ci = ttest_rel(group_A.flatten().cpu(), group_B.flatten().cpu()).confidence_interval(confidence_level=confidence)
    confidence_interval = (ci.low.item(), ci.high.item())
    return not (confidence_interval[0] < 0. and confidence_interval[1] > 0.), confidence_interval


def statistics(method_evaluation_dict):
    min_mean = float('inf')
    statistics_dict = {}
    for key, (loss_best, mean_progress, std_progress, _, _) in method_evaluation_dict.items():
        loss_best_mean = loss_best.mean()
        statistics_dict[key] = (loss_best_mean.item(), loss_best.std().item())
        if loss_best_mean < min_mean:
            min_mean_key = key
            min_mean = loss_best_mean

    for key, (loss_best, mean_progress, std_progress, offset_rmse, execution_time) in method_evaluation_dict.items():
         statistics_dict[key] =  statistics_dict[key] + (key==min_mean_key,) + significant_confidence_levels(loss_best, method_evaluation_dict[min_mean_key][0]) + (offset_rmse.mean().item(), execution_time)
         #diff = (result_dict[key] - result_dict[min_mean_key]).flatten().abs().cpu()
         #mean = torch.mean(diff)
         #std_dev = torch.std(diff)
    return statistics_dict
statistics_dict = statistics(method_evaluation_dict)


# In[12]:


def generate_latex_table(data):
    table = "\\begin{tabular}{l|ccccc}\n"
    table += "\\hline\n"
    table += "Method & \\acs{MSE} $\\pm \\sigma$ & (\\acs{CI}) & Mean Offset \\acs{RMSE} & Execution Time (s) \\\\ \n"
    table += "\\hline\n"
    
    for method, (mean, std_dev, is_best, is_significant, ci, offset_rmse, execution_time) in data.items():
        # Format the mean and standard deviation in scientific notation
        mean_str = f"{mean:.2e}".replace("e+0", "e+").replace("e-0", "e-")
        std_dev_str = f"$\\pm${std_dev:.2e}".replace("e+0", "e+").replace("e-0", "e-")
        
        # If the method is the best, make the mean bold
        if is_best:
            mean_str = "\\textbf{" + mean_str + "}"
        
        # If it's significant, add the dagger symbol
        if is_significant and not is_best:
            significant_str = "$\\dagger$"
        else:
            significant_str = ""
        
        # Combine mean, std_dev, and significance markers
        mean_with_std_dev = f"{mean_str} {std_dev_str} {significant_str}".strip()
        
        # Format confidence interval as (lower, upper)
        if is_best:
            ci_str = "~"
        else:
            ci_str = f"({ci[0]:.3f}, {ci[1]:.3f})"
        
        # Add the row to the table
        table += f"{method} & {mean_with_std_dev} & {ci_str} & {offset_rmse:.2f} & {execution_time:.2f} \\\\ \n"
    
    table += "\\hline\n"
    table += "\\end{tabular}\n"
    
    return table
# Generate the LaTeX table
latex_table = generate_latex_table(statistics_dict)

# Output the LaTeX table
print(latex_table)


# In[48]:


loss_min_params, loss, loss_min_list = optimize_pso(model, observed_rays, uncompensated_parameters_selected, iterations=1000, num_candidates=10000)
fig = fancy_plot_param_tensors(loss_min_params[:2], uncompensated_parameters_selected[:2].squeeze(), engine = engine, ray_parameter_container=model.input_parameter_container, compensated_parameters=compensated_parameters_selected[:2].squeeze())
pio.write_html(fig, os.path.join(outputs_dir,'fancy.html'))


# In[49]:


loss_min_ray_outputs = simulate_param_tensor(loss_min_params[:2, :3], engine, model.input_parameter_container, exported_plane='ImagePlane')
reference_ray_outputs = simulate_param_tensor(compensated_parameters_selected[:2, :3].squeeze(-2), engine, model.input_parameter_container, exported_plane='ImagePlane')
reference_ray_outputs_2 = simulate_param_tensor(compensated_parameters_selected[:2, :3].squeeze(-2), engine, model.input_parameter_container, exported_plane='ImagePlane')

out = compare_with_reference(reference_ray_outputs, loss_min_ray_outputs)
print("deviation best to ref", out[0].item(), "±", out[1])
out = compare_with_reference(reference_ray_outputs, reference_ray_outputs_2)
print("deviation ref to ref", out[0].item(), "±", out[1])


# In[51]:


fig = plot_param_tensors(loss_min_params[:5, :1], uncompensated_parameters_selected[:5, :1].squeeze(-2), engine = engine, ray_parameter_container=model.input_parameter_container, compensated_parameters=compensated_parameters_selected[:5, :1].squeeze(-2))
plt.savefig(os.path.join(outputs_dir,'fixed_plot.png'), bbox_inches='tight', pad_inches = 0)


# In[ ]:




