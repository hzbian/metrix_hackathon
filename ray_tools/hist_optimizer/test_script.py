import torch
import sys
sys.path.append("..")
sys.path.append("../..")
from ray_tools.hist_optimizer.hist_optimizer import tensor_to_param_container, mse_engines_comparison, Model, find_good_offset_problem, optimize_smart_walker, optimize_brute, evaluate_evaluation_method, plot_param_tensors, tensor_list_to_param_container_list

from ray_nn.nn.xy_hist_data_models import MetrixXYHistSurrogate, StandardizeXYHist, HistSurrogateEngine
from ray_tools.base.engine import RayEngine
from ray_tools.base.backend import RayBackendDockerRAYUI
from ray_optim.plot import Plot
import matplotlib.pyplot as plt

model = Model(path="outputs/xy_hist/qhmpdasi/checkpoints/epoch=295-step=118652488.ckpt")

offsets_selected, uncompensated_parameters_selected, compensated_parameters_selected = find_good_offset_problem(model)

with torch.no_grad():
    observed_rays = model(compensated_parameters_selected)

num_candidates=100000
loss_min_params, loss, loss_min_list = optimize_smart_walker(model, observed_rays, uncompensated_parameters_selected, iterations=10, num_candidates=num_candidates)
loss_min_params, loss, loss_min_list = optimize_brute(model, observed_rays, uncompensated_parameters_selected, iterations=10, num_candidates=num_candidates)