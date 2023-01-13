import sys
import os

from ax.service.ax_client import AxClient

import torch
from collections import OrderedDict

from ax import optimize
from ax.service.ax_client import AxClient

from ray_nn.data.transform import Select
from ray_nn.metrics.geometric import SinkhornLoss

sys.path.insert(0, '../../')
from ray_tools.base.parameter import RayParameterContainer, NumericalParameter, RandomParameter, MutableParameter, \
    GridParameter, build_parameter_grid
from ray_tools.base.utils import RandomGenerator
from ray_tools.base.engine import RayEngine
from ray_tools.base.transform import RayTransformConcat, ToDict
from ray_tools.base.backend import RayBackendDockerRAYUI
import wandb
import matplotlib.pyplot as plt
import numpy as np

wandb.init(entity='hzb-aos',
           project='metrix_hackathon_optimization',
           name='4-parameter-no-normalize',
           mode='disabled',  # 'disabled' or 'online'
           )

root_dir = '../../'

rml_basefile = os.path.join(root_dir, 'rml_src', 'METRIX_U41_G1_H1_318eV_PS_MLearn.rml')
ray_workdir = os.path.join(root_dir, 'ray_workdir', 'optimization')

n_rays = ['1e4']

exported_plane = "Spherical Grating"

transforms = [
    RayTransformConcat({
        'raw': ToDict(),
    }),
]

engine = RayEngine(rml_basefile=rml_basefile,
                   exported_planes=[exported_plane],
                   ray_backend=RayBackendDockerRAYUI(docker_image='ray-ui-service',
                                                     ray_workdir=ray_workdir,
                                                     verbose=True),
                   num_workers=1,
                   as_generator=False)

rg = RandomGenerator(seed=42)

param_func = lambda: RayParameterContainer([
    (engine.template.U41_318eV.numberRays, NumericalParameter(value=1e4)),
    (engine.template.U41_318eV.translationXerror, RandomParameter(value_lims=(-0.25, 0.25), rg=rg)),
    (engine.template.U41_318eV.translationYerror, RandomParameter(value_lims=(-0.25, 0.25), rg=rg)),
    (engine.template.U41_318eV.rotationXerror, RandomParameter(value_lims=(-0.05, 0.05), rg=rg)),
    (engine.template.U41_318eV.rotationYerror, RandomParameter(value_lims=(-0.05, 0.05), rg=rg)),
    (engine.template.ASBL.totalWidth, RandomParameter(value_lims=(1.9, 2.1), rg=rg)),
    (engine.template.ASBL.totalHeight, RandomParameter(value_lims=(0.9, 1.1), rg=rg)),
    (engine.template.ASBL.translationXerror, RandomParameter(value_lims=(-0.2, 0.2), rg=rg)),
    (engine.template.ASBL.translationYerror, RandomParameter(value_lims=(-0.2, 0.2), rg=rg)),
    (engine.template.M1_Cylinder.radius, RandomParameter(value_lims=(174.06, 174.36), rg=rg)),
    (engine.template.M1_Cylinder.rotationXerror, RandomParameter(value_lims=(-0.25, 0.25), rg=rg)),
    (engine.template.M1_Cylinder.rotationYerror, RandomParameter(value_lims=(-1., 1.), rg=rg)),
    (engine.template.M1_Cylinder.rotationZerror, RandomParameter(value_lims=(-1., 1.), rg=rg)),
    (engine.template.M1_Cylinder.translationXerror, RandomParameter(value_lims=(-0.15, 0.15), rg=rg)),
    (engine.template.M1_Cylinder.translationYerror, RandomParameter(value_lims=(-1., 1.), rg=rg)),
    (engine.template.SphericalGrating.radius, RandomParameter(value_lims=(109741., 109841.), rg=rg)),
    (engine.template.SphericalGrating.rotationYerror, RandomParameter(value_lims=(-1., 1.), rg=rg)),
    (engine.template.SphericalGrating.rotationZerror, RandomParameter(value_lims=(-2.5, 2.5), rg=rg)),
    (engine.template.ExitSlit.totalHeight, RandomParameter(value_lims=(0.009, 0.011), rg=rg)),
    (engine.template.ExitSlit.translationZerror, RandomParameter(value_lims=(-29., 31.), rg=rg)),
    (engine.template.ExitSlit.rotationZerror, RandomParameter(value_lims=(-0.3, 0.3), rg=rg)),
    (engine.template.E1.longHalfAxisA, RandomParameter(value_lims=(20600., 20900.), rg=rg)),
    (engine.template.E1.shortHalfAxisB, RandomParameter(value_lims=(300.721702601, 304.721702601), rg=rg)),
    (engine.template.E1.rotationXerror, RandomParameter(value_lims=(-0.5, 0.5), rg=rg)),
    (engine.template.E1.rotationYerror, RandomParameter(value_lims=(-7.5, 7.5), rg=rg)),
    (engine.template.E1.rotationZerror, RandomParameter(value_lims=(-4, 4), rg=rg)),
    (engine.template.E1.translationYerror, RandomParameter(value_lims=(-1, 1), rg=rg)),
    (engine.template.E1.translationZerror, RandomParameter(value_lims=(-1, 1), rg=rg)),
    (engine.template.E2.longHalfAxisA, RandomParameter(value_lims=(4325., 4425.), rg=rg)),
    (engine.template.E2.shortHalfAxisB, RandomParameter(value_lims=(96.1560870104, 98.1560870104), rg=rg)),
    (engine.template.E2.rotationXerror, RandomParameter(value_lims=(-0.5, 0.5), rg=rg)),
    (engine.template.E2.rotationYerror, RandomParameter(value_lims=(-7.5, 7.5), rg=rg)),
    (engine.template.E2.rotationZerror, RandomParameter(value_lims=(-4, 4), rg=rg)),
    (engine.template.E2.translationYerror, RandomParameter(value_lims=(-1, 1), rg=rg)),
    (engine.template.E2.translationZerror, RandomParameter(value_lims=(-1, 1), rg=rg)),
])

criterion = SinkhornLoss(normalize_weights=False)

# optimize only some params
params = param_func()
fixed = params.keys() - ['E2.translationZerror', 'E2.rotationYerror', 'M1_Cylinder.translationYerror', 'ASBL.translationXerror']# , , 'ASBL.totalWidth', 'M1_Cylinder.radius', 'M1_Cylinder.rotationXerror', 'M1_Cylinder.rotationYerror', 'M1_Cylinder.translationXerror']

for key in params:
    old_param = params[key]
    if isinstance(old_param, MutableParameter) and key in fixed:
        params[key] = NumericalParameter((old_param.value_lims[1] + old_param.value_lims[0]) / 2)

print(params)


def plot_data(pc_supp: torch.Tensor, pc_weights=None):
    pc_supp = pc_supp.detach().cpu()

    fig = plt.figure()
    ax = fig.gca()
    ax.scatter(pc_supp[:, 0], pc_supp[:, 1], s=2.0, c=pc_weights)

    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    out = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return out


def ray_output_to_tensor(ray_output):
    if isinstance(ray_output, list):
        return [ray_output_to_tensor(element) for element in ray_output]
    x_loc = ray_output['ray_output'][exported_plane].x_loc
    y_loc = ray_output['ray_output'][exported_plane].y_loc
    x_loc = torch.tensor(x_loc)
    y_loc = torch.tensor(y_loc)
    return torch.vstack((x_loc, y_loc)).T


def loss(trial_params, engine, secret_sample_rays, param_container):
    if not isinstance(trial_params, RayParameterContainer): #len(trial_params) > 1:#isinstance(next(iter(trial_params.keys())), int):
        trial_params_first_key = min(trial_params.keys())
        param_container_list = []

        for i in range(trial_params_first_key, trial_params_first_key + len(trial_params)):
            for param_key in trial_params[trial_params_first_key].keys():
                param_container.__setitem__(param_key, NumericalParameter(trial_params[i][param_key]))
            param_container_list.append(param_container.copy())
        param_container = param_container_list
    else:
        for k, v in trial_params.items():
            param_container.__setitem__(k, NumericalParameter(v))
    output = engine.run(param_container)
    if isinstance(output, list):
        return {key+trial_params_first_key: calculate_loss(secret_sample_rays, element) for key, element in enumerate(output)}
    return calculate_loss(secret_sample_rays, output)


def calculate_loss(y, y_hat):
    y = ray_output_to_tensor(y)
    y_hat = ray_output_to_tensor(y_hat)
    if y_hat.shape[0] == 0:
        y_hat = torch.ones((1, 2)) * -1
    loss_out = criterion(y, y_hat, torch.ones_like(y[:,1]) / y.shape[0], torch.ones_like(y_hat[:,1]) / y_hat.shape[0])
    image = wandb.Image(plot_data(y_hat))
    wandb.log({"loss": loss_out, "ray_count": y_hat.shape[0], "plot": image})
    return loss_out.item()


secret_sample_params = RayParameterContainer()
for key, value in params.items():
    if isinstance(value, MutableParameter):
        value = (value.value_lims[1] + value.value_lims[0]) / 2
    if isinstance(value, NumericalParameter):
        value = value.get_value()
    secret_sample_params[key] = NumericalParameter(value)

secret_sample_rays = engine.run(secret_sample_params)

# secret_sample_params['E2.translationYerror'] = GridParameter(np.arange(-1,1,0.1))

# output = engine.run(build_parameter_grid(secret_sample_params))
# for element in output:
#    y_hat = ray_output_to_tensor(element)
#
#    y = ray_output_to_tensor(secret_sample_rays)
#    y_hat_filled = torch.zeros_like(y) - 10.
#    y_hat_filled[:y_hat.shape[0]] = y_hat[:y.shape[0]]
#    out = criterion(y, y_hat_filled)
#    wandb.log({"loss": out, "ray_count": y_hat.shape[0]})

### Bayesian Optimization
experiment_parameters = []
for (key, value) in params.items():
    if isinstance(value, MutableParameter):
        experiment_parameters.append(
            {"name": key, "type": "range", 'value_type': 'float', "bounds": list(value.value_lims)})

ax_client = AxClient()
ax_client.create_experiment(
    name="metrix_experiment",
    parameters=experiment_parameters,
    objective_name="metrix",
    minimize=True,
)

for _ in range(1000):
    trials_to_evaluate = ax_client.get_next_trials(max_trials=2)
    results = loss(trials_to_evaluate[0], engine, secret_sample_rays, params)

    if isinstance(trials_to_evaluate, list):
        for trial_index in results:
            ax_client.complete_trial(trial_index, results[trial_index])

best_parameters, metrics = ax_client.get_best_parameters()

print(best_parameters, metrics)
