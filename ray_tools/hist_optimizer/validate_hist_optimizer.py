import torch
from ray_tools.hist_optimizer.hist_optimizer import tensor_to_param_container, mse_engines_comparison, Model, find_good_offset_problem, optimize_smart_walker, optimize_brute, evaluate_evaluation_method, plot_param_tensors, tensor_list_to_param_container_list
import matplotlib.pyplot as plt
from ray_nn.nn.xy_hist_data_models import MetrixXYHistSurrogate, StandardizeXYHist, HistSurrogateEngine
from ray_tools.base.engine import RayEngine
from ray_tools.base.backend import RayBackendDockerRAYUI
from ray_optim.plot import Plot
from torch.utils import benchmark
torch.manual_seed(42)

engine = RayEngine(rml_basefile='rml_src/METRIX_U41_G1_H1_318eV_PS_MLearn_1.15.rml',
                                exported_planes=["ImagePlane"],
                                ray_backend=RayBackendDockerRAYUI(docker_image='ray-ui-service',
                                                                  docker_container_name='ray-ui-service-test',
                                                                  dockerfile_path='ray_docker/rayui',
                                                                  ray_workdir='/dev/shm/ray-workdir',
                                                                  verbose=False),
                                num_workers=-1,
                                as_generator=False)
surrogate_engine = HistSurrogateEngine(checkpoint_path="outputs/xy_hist/qhmpdasi/checkpoints/epoch=295-step=118652488.ckpt")

model = Model(path="outputs/xy_hist/qhmpdasi/checkpoints/epoch=295-step=118652488.ckpt")

offsets_selected, uncompensated_parameters_selected, compensated_parameters_selected = find_good_offset_problem(model)

with torch.no_grad():
    observed_rays = model(compensated_parameters_selected)
    
loss_min_params, loss, loss_min_list = optimize_smart_walker(model, observed_rays, uncompensated_parameters_selected, iterations=200)

fig = plot_param_tensors(loss_min_params[[1,2,4,8]], uncompensated_parameters_selected[[1,2,4,8]], engine = engine, compensated_parameters=compensated_parameters_selected[[1,2,4,8]])
fig.savefig('outputs/offset_compensation.png', bbox_inches='tight')

with torch.no_grad():
    observed_rays = model(compensated_parameters_selected)

method_dict = {"smart walker": optimize_smart_walker, "brute force": optimize_brute}
method_evaluation_list = []

for key, entry in method_dict.items():
    mean_best, std_best, mean_progress, std_progress, loss_min_params_tens = evaluate_evaluation_method(entry, model, observed_rays, uncompensated_parameters_selected, offsets_selected, repetitions=2, iterations=100)
    method_evaluation_list.append((key, mean_best, std_best, mean_progress, std_progress))

    # calculate deviations from target offset
    normalized_offsets = (offsets_selected + max_offset) / (max_offset + max_offset)
    predicted_offsets = (loss_min_params_tens[:,0] - uncompensated_parameters_selected[0].squeeze())
    normalized_predicted_offsets = (predicted_offsets + max_offset) / (max_offset + max_offset)
    rmse = ((normalized_offsets-predicted_offsets)**2).mean().sqrt().item()
    print(key, ":", mean_best, "±", std_best, "RMSE from target offset:", rmse)

plt.figure(figsize = (4.905, 4.434))
ax = plt.gca()
i = 0
plot_list = []
for key, mean_best, std_best, mean_progress, std_progress in method_evaluation_list:
    color = plt.rcParams["axes.prop_cycle"].by_key()["color"][i+3]
    plt.fill_between(torch.arange(len(mean_progress)), (mean_progress-std_progress).cpu(), (mean_progress+std_progress).cpu(), color=color, alpha=0.2)
    plot, = plt.plot(torch.arange(len(mean_progress)), mean_progress.cpu(), alpha = 1., c = color)
    plot_list.append(plot)
    i = i+1
ax.legend(plot_list, [key for key in method_dict.keys()], prop={'size': 11})
ax.tick_params(axis='both', which='major', labelsize=11)
plt.xlabel('Iteration', fontsize=16)
plt.ylabel('MSE (log)', fontsize=16)
ax.set_yscale('log')
plt.tight_layout()
plt.savefig('outputs/bl_optimizer_iterations.pdf', bbox_inches='tight', pad_inches = 0)
plt.show()

repetitions=10
t0 = benchmark.Timer(
    stmt='optimize_smart_walker(model, observed_rays, uncompensated_parameters_selected, 200)',
    setup='from __main__ import optimize_smart_walker',
    globals={'model': model, 'observed_rays': observed_rays, 'uncompensated_parameters_selected': uncompensated_parameters_selected},
    num_threads=1,
    label='optimize smart walker',
    sub_label='optimize smart walker')
print(t0.timeit(repetitions))
