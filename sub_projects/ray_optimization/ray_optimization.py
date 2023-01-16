import sys
sys.path.insert(0, '../../')
import os

from ax.service.ax_client import AxClient

import torch
from collections import OrderedDict

from ax import optimize
from ax.service.ax_client import AxClient

from ray_nn.data.transform import Select
from ray_nn.metrics.geometric import SinkhornLoss


from sub_projects.ray_surrogate.ray_engine_surrogate import RayEngineSurrogate

from ray_tools.base.parameter import RayParameterContainer, NumericalParameter, RandomParameter, MutableParameter, \
    GridParameter, build_parameter_grid
from ray_tools.base.utils import RandomGenerator
from ray_tools.base.engine import RayEngine
from ray_tools.base.transform import RayTransformConcat, ToDict
from ray_tools.base.backend import RayBackendDockerRAYUI
import wandb
import matplotlib.pyplot as plt
import numpy as np
import time
wandb.init(entity='hzb-aos',
           project='metrix_hackathon_optimization',
           name='1-parameter-rayui-test',
           mode='online',  # 'disabled' or 'online'
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
                   num_workers=-1,
                   as_generator=False)

PARAMS_INFO = [
    ('U41_318eV.translationXerror', (-0.25, 0.25)),
    ('U41_318eV.translationYerror', (-0.25, 0.25)),
    ('U41_318eV.rotationXerror', (-0.05, 0.05)),
    ('U41_318eV.rotationYerror', (-0.05, 0.05)),
    ('ASBL.totalWidth', (1.9, 2.1)),
    ('ASBL.totalHeight', (0.9, 1.1)),
    ('ASBL.translationXerror', (-0.2, 0.2)),
    ('ASBL.translationYerror', (-0.2, 0.2)),
    ('M1_Cylinder.radius', (174.06, 174.36)),
    ('M1_Cylinder.rotationXerror', (-0.25, 0.25)),
    ('M1_Cylinder.rotationYerror', (-1., 1.)),
    ('M1_Cylinder.rotationZerror', (-1., 1.)),
    ('M1_Cylinder.translationXerror', (-0.15, 0.15)),
    ('M1_Cylinder.translationYerror', (-1., 1.)),
    ('SphericalGrating.radius', (109741., 109841.)),
    ('SphericalGrating.rotationYerror', (-1., 1.)),
    ('SphericalGrating.rotationZerror', (-2.5, 2.5)),
]

nn_engine = RayEngineSurrogate(
    ckpt_path='/scratch/meier/metrix_hackathon/sub_projects/ray_surrogate/training/results/sg_v1_nothing_given/best_val.ckpt',
    params_info=PARAMS_INFO, hist_dim=1024, gpu_id=0)

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

criterion = SinkhornLoss(normalize_weights=False, p=1, backend='online')

# optimize only some params
params = param_func()
fixed = params.keys() - ['U41_318eV.translationYerror'] #['U41_318eV.translationYerror', 'U41_318eV.rotationXerror', 'U41_318eV.rotationYerror', 'ASBL.totalWidth', 'ASBL.totalHeight', 'ASBL.translationXerror', 'ASBL.translationYerror', 'M1_Cylinder.radius', 'M1_Cylinder.rotationXerror', 'M1_Cylinder.rotationYerror', 'M1_Cylinder.rotationZerror', 'M1_Cylinder.translationXerror', 'M1_Cylinder.translationYerror', 'SphericalGrating.radius', 'SphericalGrating.rotationYerror', 'SphericalGrating.rotationZerror', 'ExitSlit.totalHeight', 'ExitSlit.translationZerror', 'ExitSlit.rotationZerror', 'E1.longHalfAxisA', 'E1.shortHalfAxisB', 'E1.rotationXerror', 'E1.rotationYerror', 'E1.rotationZerror', 'E1.translationYerror', 'E1.translationZerror', 'E2.longHalfAxisA', 'E2.shortHalfAxisB', 'E2.rotationXerror', 'E2.rotationYerror', 'E2.rotationZerror', 'E2.translationYerror', 'E2.translationZerror']

# Out[3]: odict_keys(['U41_318eV.numberRays', 'U41_318eV.translationXerror', 'U41_318eV.translationYerror', 'U41_318eV.rotationXerror', 'U41_318eV.rotationYerror', 'ASBL.totalWidth', 'ASBL.totalHeight', 'ASBL.translationXerror', 'ASBL.translationYerror', 'M1_Cylinder.radius', 'M1_Cylinder.rotationXerror', 'M1_Cylinder.rotationYerror', 'M1_Cylinder.rotationZerror', 'M1_Cylinder.translationXerror', 'M1_Cylinder.translationYerror', 'SphericalGrating.radius', 'SphericalGrating.rotationYerror', 'SphericalGrating.rotationZerror', 'ExitSlit.totalHeight', 'ExitSlit.translationZerror', 'ExitSlit.rotationZerror', 'E1.longHalfAxisA', 'E1.shortHalfAxisB', 'E1.rotationXerror', 'E1.rotationYerror', 'E1.rotationZerror', 'E1.translationYerror', 'E1.translationZerror', 'E2.longHalfAxisA', 'E2.shortHalfAxisB', 'E2.rotationXerror', 'E2.rotationYerror', 'E2.rotationZerror', 'E2.translationYerror', 'E2.translationZerror'])
for key in params:
    old_param = params[key]
    if isinstance(old_param, MutableParameter) and key in fixed:
        params[key] = NumericalParameter((old_param.value_lims[1] + old_param.value_lims[0]) / 2)


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
    begin_total_time = time.time()
    trial_params_first_key = min(trial_params.keys())
    param_container_list = []

    for i in range(trial_params_first_key, trial_params_first_key + len(trial_params)):
        for param_key in trial_params[trial_params_first_key].keys():
            param_container.__setitem__(param_key, NumericalParameter(trial_params[i][param_key]))
        param_container_list.append(param_container.copy())
    param_container = param_container_list
    begin_time = time.time()
    print("parameter container", param_container)
    output = engine.run(param_container)
    print("Execution took ", time.time() - begin_time, "s")
    if not isinstance(output, list):
        output = [output]
    output_loss = {key + trial_params_first_key: calculate_loss(secret_sample_rays, element) for key, element in
            enumerate(output)}
    print("Total took", time.time() - begin_total_time, "s")
    return output_loss


def calculate_loss(y, y_hat):
    begin_time = time.time()
    y = ray_output_to_tensor(y).cuda()
    y_hat = ray_output_to_tensor(y_hat).cuda()
    if y_hat.shape[0] == 0:
        y_hat = torch.ones((1, 2)) * -1
    loss_out = criterion(y.contiguous(), y_hat.contiguous(), torch.ones_like(y[:, 1]) / y.shape[0], torch.ones_like(y_hat[:, 1]) / y_hat.shape[0])
    print("Loss took ", time.time() - begin_time, "s")
    image = wandb.Image(plot_data(y_hat))
    image_2 = wandb.Image(plot_data(y))
    wandb.log({"loss": loss_out.cpu(), "ray_count": y_hat.shape[0], "plot": image, "plot2": image_2})
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

ax_client = AxClient(early_stopping_strategy=None)
ax_client.create_experiment(
    name="metrix_experiment",
    parameters=experiment_parameters,
    objective_name="metrix",
    minimize=True,
)

for i in range(1000):
    trials_to_evaluate = ax_client.get_next_trials(max_trials=10)
    results = loss(trials_to_evaluate[0], engine, secret_sample_rays, params)

    for trial_index in results:
        ax_client.complete_trial(trial_index, results[trial_index])

best_parameters, metrics = ax_client.get_best_parameters()

print(best_parameters, metrics)
