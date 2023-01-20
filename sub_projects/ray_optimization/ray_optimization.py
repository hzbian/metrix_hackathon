import sys
from typing import List

sys.path.insert(0, '../../')
import os

import torch

from ax.service.ax_client import AxClient

from ray_nn.metrics.geometric import SinkhornLoss

from abc import ABCMeta, abstractmethod, ABC
from sub_projects.ray_surrogate.ray_engine_surrogate import RayEngineSurrogate

from ray_tools.base.parameter import RayParameterContainer, NumericalParameter, RandomParameter, MutableParameter
from ray_tools.base.utils import RandomGenerator
from ray_tools.base.engine import RayEngine
from ray_tools.base.transform import RayTransformConcat, ToDict
from ray_tools.base.backend import RayBackendDockerRAYUI
import wandb
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import trange
import optuna
from optuna.samplers import TPESampler


class OptimizerBackend(metaclass=ABCMeta):
    @abstractmethod
    def setup_optimization(self):
        pass

    @abstractmethod
    def optimize(self, objective, iterations):
        pass


class OptimizerBackendOptuna(OptimizerBackend):
    def __init__(self, optuna_study, all_parameters: RayParameterContainer):
        self.all_parameters = all_parameters
        self.optuna_study = optuna_study

    def setup_optimization(self):
        pass

    def optuna_objective(self, objective):
        def output_objective(trial):
            optimize_parameters = self.all_parameters.copy()
            for key, value in optimize_parameters.items():
                if isinstance(value, MutableParameter):
                    optimize_parameters[key] = NumericalParameter(trial.suggest_float(key, value.value_lims[0],
                                                                                      value.value_lims[1]))
            output = objective(optimize_parameters)
            return output[min(output.keys())]

        return output_objective

    def optimize(self, objective, iterations):
        self.optuna_study.optimize(self.optuna_objective(objective), n_trials=iterations)
        return self.optuna_study.best_params, {}


class OptimizerBackendAx(OptimizerBackend):
    def __init__(self, ax_client: AxClient, all_parameters: RayParameterContainer):
        self.all_parameters = all_parameters
        self.ax_client = ax_client

    def optimizer_parameter_to_container_list(self, optimizer_parameter) -> List[RayParameterContainer]:
        trial_params_first_key = min(optimizer_parameter.keys())
        param_container_list = []

        for i in range(trial_params_first_key, trial_params_first_key + len(optimizer_parameter)):
            all_parameters_copy = self.all_parameters.copy()
            for param_key in optimizer_parameter[trial_params_first_key].keys():
                all_parameters_copy.__setitem__(param_key, NumericalParameter(optimizer_parameter[i][param_key]))
            param_container_list.append(all_parameters_copy)
        return param_container_list

    def setup_optimization(self):
        experiment_parameters = []
        for key, value in self.all_parameters.items():
            if isinstance(value, MutableParameter):
                experiment_parameters.append(
                    {"name": key, "type": "range", 'value_type': 'float', "bounds": list(value.value_lims)})

        self.ax_client.create_experiment(
            name="metrix_experiment",
            parameters=experiment_parameters,
            objective_name="metrix",
            minimize=True,
        )

    def optimize(self, objective, iterations):
        ranger = trange(iterations)
        for _ in ranger:
            optimization_time = time.time()
            trials_to_evaluate = ax_client.get_next_trials(max_trials=10)[0]
            print("Optimization took {:.2f}s".format(time.time() - optimization_time))

            ray_parameter_container_list = self.optimizer_parameter_to_container_list(trials_to_evaluate)
            results = objective(ray_parameter_container_list)

            for trial_index in results:
                ax_client.complete_trial(trial_index, results[trial_index])

        best_parameters, metrics = ax_client.get_best_parameters()
        return best_parameters, metrics


class RayOptimizer:
    def __init__(self, optimizer_backend: OptimizerBackend, criterion: torch.nn.Module, engine: RayEngine,
                 target_engine: RayEngine = None, verbose=False):
        self.optimizer_backend = optimizer_backend
        self.target_engine = target_engine
        self.engine = engine
        self.criterion = criterion
        if target_engine is None:
            target_engine = engine
        self.target_rays = target_engine.run(target_params)
        self.optimizer_backend.setup_optimization()
        self.evaluation_counter = 0
        self.verbose = verbose

    @staticmethod
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

    @staticmethod
    def ray_output_to_tensor(ray_output):
        if isinstance(ray_output, list):
            return [RayOptimizer.ray_output_to_tensor(element) for element in ray_output]
        x_loc = ray_output['ray_output'][exported_plane].x_loc
        y_loc = ray_output['ray_output'][exported_plane].y_loc
        x_loc = torch.tensor(x_loc)
        y_loc = torch.tensor(y_loc)
        return torch.vstack((x_loc, y_loc)).T

    def evaluation_function(self, parameters):
        if self.verbose:
            begin_total_time = time.time()

        if self.verbose:
            begin_execution_time = time.time()
        output = engine.run(parameters)
        if self.verbose:
            print("Execution took {:.2f}s".format(time.time() - begin_execution_time))
        if not isinstance(output, list):
            output = [output]
        output_loss = {
            key + self.evaluation_counter: self.calculate_loss(self.target_rays, element) for
            key, element in
            enumerate(output)}

        log_dict = {}  # "epoch": self.evaluation_counter, "loss": loss_out.cpu(), "ray_count": y_hat.shape[0]}
        for key, value in output_loss.items():
            log_dict["epoch"] = key
            log_dict["loss"] = value
            # ray count
        if True in [i % 100 == 0 for i in range(self.evaluation_counter, self.evaluation_counter + len(output))]:
            image = wandb.Image(self.plot_data(self.ray_output_to_tensor(output[0])))
            log_dict = {**log_dict, **{"plot": image}}
        if self.evaluation_counter == 0:
            target = wandb.Image(self.plot_data(self.ray_output_to_tensor(self.target_rays)))
            log_dict = {**log_dict, **{"target": target}}
        wandb.log(log_dict)
        self.evaluation_counter += len(parameters)
        if self.verbose:
            print("Total took {:.2f}s".format(time.time() - begin_total_time))
        return output_loss

    def calculate_loss(self, y, y_hat):
        if self.verbose:
            begin_time = time.time()
        y = self.ray_output_to_tensor(y).cuda()
        y_hat = self.ray_output_to_tensor(y_hat).cuda()
        if y_hat.shape[0] == 0:
            y_hat = torch.ones((1, 2), device=y_hat.device, dtype=y_hat.dtype) * -1
        loss_out = criterion(y.contiguous(), y_hat.contiguous(), torch.ones_like(y[:, 1]) / y.shape[0],
                             torch.ones_like(y_hat[:, 1]) / y_hat.shape[0])
        if self.verbose:
            print("Loss took {:.2f}s".format(time.time() - begin_time))
        return loss_out.item()

    def optimize(self, iterations):
        best_parameters, metrics = self.optimizer_backend.optimize(objective=self.evaluation_function, iterations=iterations)
        return best_parameters, metrics


wandb.init(entity='hzb-aos',
           project='metrix_hackathon_optimization',
           name='14-parameter-rayui-TPE',
           mode='online',  # 'disabled' or 'online'
           )

root_dir = '../../'

rml_basefile = os.path.join(root_dir, 'rml_src', 'METRIX_U41_G1_H1_318eV_PS_MLearn.rml')
ray_workdir = os.path.join(root_dir, 'ray_workdir', 'optimization')

n_rays = ['1e4']

exported_plane = "ImagePlane" #"Spherical Grating"

transforms = [
    RayTransformConcat({
        'raw': ToDict(),
    }),
]
verbose = False
engine = RayEngine(rml_basefile=rml_basefile,
                   exported_planes=[exported_plane],
                   ray_backend=RayBackendDockerRAYUI(docker_image='ray-ui-service',
                                                     ray_workdir=ray_workdir,
                                                     verbose=verbose),
                   num_workers=-1,
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

criterion = SinkhornLoss(normalize_weights='weights1', p=1, backend='online')

# optimize only some all_params
all_params = param_func()
fixed = [] #all_params.keys()[]
#   - [ 'U41_318eV.translationXerror']  # ['U41_318eV.translationYerror', 'U41_318eV.rotationXerror', 'U41_318eV.rotationYerror', 'ASBL.totalWidth', 'ASBL.totalHeight', 'ASBL.translationXerror', 'ASBL.translationYerror', 'M1_Cylinder.radius', 'M1_Cylinder.rotationXerror', 'M1_Cylinder.rotationYerror', 'M1_Cylinder.rotationZerror', 'M1_Cylinder.translationXerror', 'M1_Cylinder.translationYerror', 'SphericalGrating.radius', 'SphericalGrating.rotationYerror', 'SphericalGrating.rotationZerror', 'ExitSlit.totalHeight', 'ExitSlit.translationZerror', 'ExitSlit.rotationZerror', 'E1.longHalfAxisA', 'E1.shortHalfAxisB', 'E1.rotationXerror', 'E1.rotationYerror', 'E1.rotationZerror', 'E1.translationYerror', 'E1.translationZerror', 'E2.longHalfAxisA', 'E2.shortHalfAxisB', 'E2.rotationXerror', 'E2.rotationYerror', 'E2.rotationZerror', 'E2.translationYerror', 'E2.translationZerror']

# Out[3]: odict_keys(['U41_318eV.numberRays', 'U41_318eV.translationXerror', 'U41_318eV.translationYerror', 'U41_318eV.rotationXerror', 'U41_318eV.rotationYerror', 'ASBL.totalWidth', 'ASBL.totalHeight', 'ASBL.translationXerror', 'ASBL.translationYerror', 'M1_Cylinder.radius', 'M1_Cylinder.rotationXerror', 'M1_Cylinder.rotationYerror', 'M1_Cylinder.rotationZerror', 'M1_Cylinder.translationXerror', 'M1_Cylinder.translationYerror', 'SphericalGrating.radius', 'SphericalGrating.rotationYerror', 'SphericalGrating.rotationZerror', 'ExitSlit.totalHeight', 'ExitSlit.translationZerror', 'ExitSlit.rotationZerror', 'E1.longHalfAxisA', 'E1.shortHalfAxisB', 'E1.rotationXerror', 'E1.rotationYerror', 'E1.rotationZerror', 'E1.translationYerror', 'E1.translationZerror', 'E2.longHalfAxisA', 'E2.shortHalfAxisB', 'E2.rotationXerror', 'E2.rotationYerror', 'E2.rotationZerror', 'E2.translationYerror', 'E2.translationZerror'])
for key in all_params:
    old_param = all_params[key]
    if isinstance(old_param, MutableParameter) and key in fixed:
        all_params[key] = NumericalParameter((old_param.value_lims[1] + old_param.value_lims[0]) / 2)

target_params = RayParameterContainer()
for key, value in all_params.items():
    if isinstance(value, MutableParameter):
        value = (value.value_lims[1] + value.value_lims[0]) / 2
    if isinstance(value, NumericalParameter):
        value = value.get_value()
    target_params[key] = NumericalParameter(value)

# target_params['E2.translationYerror'] = GridParameter(np.arange(-1,1,0.1))

# output = engine.run(build_parameter_grid(target_params))
# for element in output:
#    y_hat = ray_output_to_tensor(element)
#
#    y = ray_output_to_tensor(secret_sample_rays)
#    y_hat_filled = torch.zeros_like(y) - 10.
#    y_hat_filled[:y_hat.shape[0]] = y_hat[:y.shape[0]]
#    out = criterion(y, y_hat_filled)
#    wandb.log({"loss": out, "ray_count": y_hat.shape[0]})

### Bayesian Optimization

ax_client = AxClient(early_stopping_strategy=None, verbose_logging=verbose)

optimizer_backend_ax = OptimizerBackendAx(ax_client, all_parameters=all_params)

optuna_study = optuna.create_study(sampler=TPESampler())
optimizer_backend_optuna = OptimizerBackendOptuna(optuna_study, all_parameters=all_params)
ray_optimizer = RayOptimizer(optimizer_backend=optimizer_backend_optuna, criterion=criterion, engine=engine, verbose=verbose)

best_parameters, metrics = ray_optimizer.optimize(iterations=1000)
print(best_parameters, metrics)
