from abc import ABCMeta, abstractmethod
import time
from typing import List, Iterable, Union, Dict, Optional

import numpy as np
import torch
from ax.service.ax_client import AxClient
from matplotlib import pyplot as plt
from tqdm import trange

from ray_tools.base import RayTransform, RayOutput
from ray_tools.base.engine import RayEngine
from ray_tools.base.parameter import RayParameterContainer, MutableParameter, NumericalParameter
import wandb


class OptimizerBackend(metaclass=ABCMeta):
    @abstractmethod
    def setup_optimization(self):
        pass

    @abstractmethod
    def optimize(self, objective, iterations, target_rays, target_params=None):
        pass


class OptimizerBackendOptuna(OptimizerBackend):
    def __init__(self, optuna_study, search_space: RayParameterContainer):
        self.search_space = search_space
        self.optuna_study = optuna_study

    def setup_optimization(self):
        pass

    def optuna_objective(self, objective, target_rays, target_params=None):
        def output_objective(trial):
            optimize_parameters = self.search_space.copy()
            for key, value in optimize_parameters.items():
                if isinstance(value, MutableParameter):
                    optimize_parameters[key] = NumericalParameter(trial.suggest_float(key, value.value_lims[0],
                                                                                      value.value_lims[1]))
            output = objective(optimize_parameters, target_rays=target_rays, target_params=target_params)
            return output[min(output.keys())]

        return output_objective

    def optimize(self, objective, iterations, target_rays, target_params=None):
        self.optuna_study.optimize(self.optuna_objective(objective, target_rays, target_params), n_trials=iterations,
                                   show_progress_bar=True)
        return self.optuna_study.best_params, {}


class OptimizerBackendAx(OptimizerBackend):
    def __init__(self, ax_client: AxClient, search_space: RayParameterContainer):
        self.search_space = search_space
        self.ax_client = ax_client

    def optimizer_parameter_to_container_list(self, optimizer_parameter) -> List[RayParameterContainer]:
        trial_params_first_key = min(optimizer_parameter.keys())
        param_container_list = []

        for i in range(trial_params_first_key, trial_params_first_key + len(optimizer_parameter)):
            all_parameters_copy = self.search_space.copy()
            for param_key in optimizer_parameter[trial_params_first_key].keys():
                all_parameters_copy.__setitem__(param_key, NumericalParameter(optimizer_parameter[i][param_key]))
            param_container_list.append(all_parameters_copy)
        return param_container_list

    def setup_optimization(self):
        experiment_parameters = []
        for key, value in self.search_space.items():
            if isinstance(value, MutableParameter):
                experiment_parameters.append(
                    {"name": key, "type": "range", 'value_type': 'float', "bounds": list(value.value_lims)})

        self.ax_client.create_experiment(
            name="metrix_experiment",
            parameters=experiment_parameters,
            objective_name="metrix",
            minimize=True,
        )

    def optimize(self, objective, iterations, target_rays, target_params=None):
        ranger = trange(iterations)
        for _ in ranger:
            optimization_time = time.time()
            trials_to_evaluate = self.ax_client.get_next_trials(max_trials=10)[0]
            print("Optimization took {:.2f}s".format(time.time() - optimization_time))

            ray_parameter_container_list = self.optimizer_parameter_to_container_list(trials_to_evaluate)
            results = objective(ray_parameter_container_list, target_rays, target_params=target_params)

            for trial_index in results:
                self.ax_client.complete_trial(trial_index, results[trial_index])

        best_parameters, metrics = self.ax_client.get_best_parameters()
        return best_parameters, metrics


class LoggingBackend(metaclass=ABCMeta):
    def __init__(self):
        self.log_dict = {}

    def add_to_log(self, add_to_log: dict):
        self.log_dict = {**self.log_dict, **add_to_log}

    def empty_log(self):
        self.log_dict = {}

    def log(self):
        self._log()
        self.empty_log()

    @abstractmethod
    def _log(self):
        pass

    @abstractmethod
    def image(self, key: Union[str, int], image: torch.Tensor):
        pass


class WandbLoggingBackend(LoggingBackend):
    def _log(self):
        wandb.log(self.log_dict)

    def image(self, key: Union[str, int], image: torch.Tensor):
        image = wandb.Image(image)
        self.add_to_log({key: image})


class RayOptimizer:
    def __init__(self, optimizer_backend: OptimizerBackend, criterion: torch.nn.Module, exported_plane: str,
                 engine: RayEngine, search_space: RayParameterContainer,
                 logging_backend: LoggingBackend, transforms: Optional[RayTransform] = None,
                 log_times=False):
        self.optimizer_backend = optimizer_backend
        self.engine = engine
        self.criterion = criterion
        self.exported_plane = exported_plane
        self.search_space = search_space
        self.transforms = transforms
        self.logging_backend = logging_backend
        self.optimizer_backend.setup_optimization()
        self.evaluation_counter = 0
        self.log_times = log_times
        self.plot_interval_best_params: RayParameterContainer = RayParameterContainer()
        self.plot_interval_best_rays: Union[None, torch.Tensor] = None
        self.plot_interval_best_loss: float = float('inf')

    @staticmethod
    def fig_to_image(fig):
        fig.canvas.draw()

        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        out = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return out

    @staticmethod
    def plot_data(pc_supp: torch.Tensor, pc_weights=None):
        pc_supp = pc_supp.detach().cpu()

        fig = plt.figure()
        ax = fig.gca()
        ax.scatter(pc_supp[:, 0], pc_supp[:, 1], s=2.0, c=pc_weights)

        return RayOptimizer.fig_to_image(fig)

    def normalize_parameters(self, parameters: RayParameterContainer):
        normalized_parameters = RayParameterContainer()
        for key, value in self.search_space.items():
            if isinstance(value, MutableParameter):
                normalized_parameters[key] = NumericalParameter((parameters[key].get_value() -
                                                                 value.value_lims[0]) / (value.value_lims[1] -
                                                                                         value.value_lims[0]))
        return normalized_parameters

    def plot_param_comparison(self, predicted_params: RayParameterContainer,
                              real_params: Optional[RayParameterContainer] = None):
        fig, ax = plt.subplots(1, 1, figsize=(16, 9))
        if real_params is not None:
            ax.plot([param.get_value() for param in self.normalize_parameters(real_params).values()], 'bo',
                    markersize=20,
                    label='real parameters')
        ax.plot([param.get_value() for param in self.normalize_parameters(predicted_params).values()], 'm*',
                markersize=20,
                label='predicted parameters')
        param_labels = [param_key for param_key, param_value in real_params.items() if
                        isinstance(self.search_space[param_key], MutableParameter)]
        ax.set_xticks(range(len(param_labels)))
        ax.set_xticklabels(param_labels, rotation=90)
        plt.subplots_adjust(bottom=0.3)
        return RayOptimizer.fig_to_image(fig)

    def ray_output_to_tensor(self, ray_output: Union[Dict, List[Dict], Iterable[Dict]]):
        if not isinstance(ray_output, Dict):
            return [self.ray_output_to_tensor(element) for element in ray_output]
        else:
            rays: dict = ray_output['ray_output'][self.exported_plane]
            x_locs = torch.stack([torch.tensor(value.x_loc) for value in rays.values()])
            y_locs = torch.stack([torch.tensor(value.y_loc) for value in rays.values()])
            return torch.stack((x_locs, y_locs), -1)

    def evaluation_function(self, parameters, target_rays, target_params=None):
        begin_total_time: float = time.time() if self.log_times else None
        if not isinstance(parameters, list):
            parameters = [parameters]
        begin_execution_time: float = time.time() if self.log_times else None
        output = self.engine.run(parameters, transforms=self.transforms)
        if not isinstance(output, list):
            output = [output]

        if self.log_times:
            self.logging_backend.add_to_log({"execution_time": time.time() - begin_execution_time})

        begin_loss_time: float = time.time() if self.log_times else None
        output_loss = {
            key + self.evaluation_counter: self.calculate_loss(target_rays, element) for
            key, element in enumerate(output)}
        if self.log_times:
            self.logging_backend.add_to_log({"loss_time": time.time() - begin_loss_time})

        for epoch, (loss, ray_count) in output_loss.items():
            self.logging_backend.add_to_log({"epoch": epoch, "loss": loss, "ray_count": ray_count})
            if loss < self.plot_interval_best_loss:
                self.plot_interval_best_loss = loss
                self.plot_interval_best_params = parameters[epoch - self.evaluation_counter]
            if True in [i % 100 == 0 for i in range(self.evaluation_counter, self.evaluation_counter + len(output))]:
                image = self.plot_data(self.plot_interval_best_rays[0])
                self.logging_backend.image("footprint", image)
                parameter_comparison_image = self.plot_param_comparison(self.plot_interval_best_params, target_params)
                self.logging_backend.image("parameter_comparison", parameter_comparison_image)
                self.plot_interval_best_loss = float('inf')

            if self.evaluation_counter == 0:
                target_tensor = self.ray_output_to_tensor(target_rays)
                for counter, target_tensor_sample in enumerate(target_tensor):
                    target_image = self.plot_data(target_tensor_sample)
                    self.logging_backend.image("target_footprint_" + str(counter), target_image)
            if self.log_times:
                self.logging_backend.add_to_log({"total_time": time.time() - begin_total_time})
        self.logging_backend.log()
        self.evaluation_counter += len(output)
        return {epoch: loss for epoch, (loss, _) in output_loss.items()}

    def calculate_loss(self, y, y_hat):
        y = self.ray_output_to_tensor(y).cuda()
        y_hat = self.ray_output_to_tensor(y_hat).cuda()
        if y_hat.shape[1] == 0 or y_hat.shape[1] == 1:
            y_hat = torch.ones((y_hat.shape[0], 2, 2), device=y_hat.device, dtype=y_hat.dtype) * -1
        loss_out = self.criterion(y.contiguous(), y_hat.contiguous(), torch.ones_like(y[..., 1]),
                                  torch.ones_like(y_hat[..., 1]))
        loss = loss_out.mean().item()

        if loss < self.plot_interval_best_loss:
            self.plot_interval_best_rays = y_hat
        return loss, y_hat.shape[1]

    def optimize(self, target_rays: Union[Dict, Iterable[Dict], List[Dict]], iterations: int,
                 target_params: Optional[RayParameterContainer] = None):
        best_parameters, metrics = self.optimizer_backend.optimize(objective=self.evaluation_function,
                                                                   iterations=iterations, target_rays=target_rays,
                                                                   target_params=target_params)
        return best_parameters, metrics
