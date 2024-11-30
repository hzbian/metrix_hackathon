#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import os
import pickle

sys.path.append("../..")

import matplotlib.pyplot as plt
import plotly.io as pio
from tqdm.auto import tqdm, trange
import torch

from ray_tools.hist_optimizer.hist_optimizer import generate_latex_table, statistics, generate_n_offset_problems, tensor_to_param_container, evaluate_method_dict, mse_engines_comparison, correlation_plot, correlation_matrix, find_good_offset_problem, plot_optimizer_iterations, optimize_tpe, optimize_evotorch, optimize_smart_walker, optimize_brute, optimize_pso, optimize_ea, evaluate_evaluation_method, plot_param_tensors, tensor_list_to_param_container_list, simulate_param_tensor, compare_with_reference, optimize_evotorch_cmaes, fancy_plot_param_tensors
from ray_tools.base.engine import RayEngine
from ray_nn.nn.xy_hist_data_models import HistSurrogateEngine, Model, StandardizeXYHist
from ray_tools.base.backend import RayBackendDockerRAYUI

torch.manual_seed(42)


# In[ ]:


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


model_path = os.path.join(file_root, "outputs/xy_hist/s021yw7n/checkpoints/epoch=235-step=70000000.ckpt")
surrogate_engine = HistSurrogateEngine(checkpoint_path=model_path)

model = Model(path=model_path)


# In[ ]:


offsets_selected, uncompensated_parameters_selected, compensated_parameters_selected = find_good_offset_problem(model, fixed_parameters = [8, 14, 20, 21, 27, 28, 34])

with torch.no_grad():
    observed_rays = model(compensated_parameters_selected)


# # Examine 10000 problems

# In[ ]:


offsets_list, uncompensated_parameters_list, compensated_parameters_list = generate_n_offset_problems(model, 10000)


# In[ ]:


correlation_matrix(offsets_list, model, "offsets", outputs_dir=outputs_dir)


# In[ ]:


stacked_uncompensated_parameters = torch.vstack([entry[:, 0, 0, :36] for entry in uncompensated_parameters_list])
correlation_matrix(stacked_uncompensated_parameters, model, "uncompensated parameters", outputs_dir=outputs_dir)


# In[ ]:


correlation_plot(offsets_list, model, label="Offsets", outputs_dir=outputs_dir)


# In[ ]:


uncompensated_parameters_stack = torch.vstack([entry[:, 0, 0, :36] for entry in uncompensated_parameters_list])
correlation_plot(uncompensated_parameters_stack, model, label="Uncompensated parameters", outputs_dir=outputs_dir)


# # Examine best optimizer

# In[ ]:


loss_min_params, loss, loss_min_list = optimize_evotorch_cmaes(model, observed_rays, uncompensated_parameters_selected, iterations=2000, num_candidates=1000)
fig = fancy_plot_param_tensors(loss_min_params[:], uncompensated_parameters_selected[:].squeeze(), engine = engine, ray_parameter_container=model.input_parameter_container, compensated_parameters=compensated_parameters_selected[:].squeeze())
pio.write_html(fig, os.path.join(outputs_dir,'fancy.html'))


# In[ ]:


loss_min_ray_outputs = simulate_param_tensor(loss_min_params[:, :], engine, model.input_parameter_container, exported_plane='ImagePlane')
reference_ray_outputs = simulate_param_tensor(compensated_parameters_selected[:, :].squeeze(-2), engine, model.input_parameter_container, exported_plane='ImagePlane')
reference_ray_outputs_2 = simulate_param_tensor(compensated_parameters_selected[:, :].squeeze(-2), engine, model.input_parameter_container, exported_plane='ImagePlane')

out = compare_with_reference(reference_ray_outputs, loss_min_ray_outputs)
print("deviation best to ref", out[0].item(), "±", out[1].item())
out = compare_with_reference(reference_ray_outputs, reference_ray_outputs_2)
print("deviation ref to ref", out[0].item(), "±", out[1].item())


# In[ ]:


fig = plot_param_tensors(loss_min_params[:5, :1], uncompensated_parameters_selected[:5, :1].squeeze(-2), engine = engine, ray_parameter_container=model.input_parameter_container, compensated_parameters=compensated_parameters_selected[:5, :1].squeeze(-2))
plt.savefig(os.path.join(outputs_dir,'fixed_plot.png'), bbox_inches='tight', pad_inches = 0)


# # Compare optimizers

# In[ ]:


method_dict = {"Smart Walker": (optimize_smart_walker, 1000), "Brute Force": (optimize_brute, 1000), "TPE": (optimize_tpe, None), "PSO": (optimize_pso, 1000), "CMA-ES": (optimize_evotorch_cmaes, 1000)}

method_evaluation_dict = evaluate_method_dict(method_dict, model, observed_rays, uncompensated_parameters_selected, iterations=2000, repetitions=30, benchmark_repetitions=10)
with open(os.path.join(outputs_dir, "compare_optimizers.pkl"), "wb") as f:
    pickle.dump(method_evaluation_dict, f)


# In[ ]:


plot_optimizer_iterations(method_evaluation_dict, outputs_dir)


# In[ ]:


statistics_dict = statistics(method_evaluation_dict)

# Generate the LaTeX table
latex_table = generate_latex_table(statistics_dict)

# Output the LaTeX table
print(latex_table)


# In[ ]:





# In[ ]:




