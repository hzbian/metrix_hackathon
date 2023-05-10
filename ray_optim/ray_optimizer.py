import copy
import time
from abc import ABCMeta, abstractmethod
from typing import List, Iterable, Union, Dict, Optional, Callable

import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt
from optuna import Study

from ray_tools.base import RayTransform
from ray_tools.base.engine import RayEngine
from ray_tools.base.parameter import RayParameterContainer, MutableParameter, NumericalParameter, RandomParameter, \
    RandomOutputParameter, NumericalOutputParameter, OutputParameter
from ray_tools.base.transform import RayTransformCompose, MultiLayer, Translation

# from ax.service.ax_client import AxClient

plt.switch_backend('Agg')


class OptimizationTarget:
    def __init__(self, perturbed_parameters_rays: Union[dict, Iterable[dict], List[dict]],
                 search_space: RayParameterContainer,
                 target_params: Optional[RayParameterContainer] = None):
        self.perturbed_parameters_rays = perturbed_parameters_rays
        self.search_space = search_space
        self.target_params = target_params


class OffsetOptimizationTarget(OptimizationTarget):
    def __init__(self, perturbed_parameters_rays: Union[dict, Iterable[dict], List[dict]],
                 search_space: RayParameterContainer,
                 initial_parameters: List[RayParameterContainer],
                 initial_parameters_rays: Union[dict, Iterable[dict], List[dict]],
                 offset: Optional[RayParameterContainer] = None):
        super().__init__(perturbed_parameters_rays, search_space, offset)
        self.initial_parameters: List[RayParameterContainer] = initial_parameters
        self.initial_parameters_rays = initial_parameters_rays


class OptimizerBackend(metaclass=ABCMeta):
    @abstractmethod
    def setup_optimization(self):
        pass

    @abstractmethod
    def optimize(self, objective: Callable, iterations: int, optimization_target: OptimizationTarget):
        pass


class OptimizerBackendBasinhopping(OptimizerBackend):
    def __init__(self, basinhopping_function):
        self.basinhopping_function = basinhopping_function

    def setup_optimization(self):
        pass

    @staticmethod
    def basinhopping_objective(objective, optimization_target: OptimizationTarget):
        def output_objective(input: np.ndarray):
            optimize_parameters = optimization_target.search_space.copy()
            for i, (key, value) in enumerate(optimize_parameters.items()):
                if isinstance(value, MutableParameter):
                    optimize_parameters[key] = NumericalParameter(input[i])
            output = objective(optimize_parameters, optimization_target=optimization_target)
            return tuple(value.mean().item() for value in output[min(output.keys())])

        return output_objective

    def optimize(self, objective: Callable, iterations: int, optimization_target: OptimizationTarget):
        optimize_parameters = optimization_target.search_space.copy()
        x0 = []
        bounds = []
        for key, value in optimize_parameters.items():
            if isinstance(value, MutableParameter):
                bounds.append([value.value_lims[0], value.value_lims[1]])
                x0.append((value.value_lims[1] - value.value_lims[0]) / 2. + value.value_lims[0])
        ret = self.basinhopping_function(self.basinhopping_objective(objective, optimization_target), x0,
                                         niter=iterations, minimizer_kwargs={"bounds": bounds}, disp=True)
        return ret.x, ret.fun


class OptimizerBackendOptuna(OptimizerBackend):
    def __init__(self, optuna_study: Study):
        self.optuna_study: Study = optuna_study

    def setup_optimization(self):
        pass

    @staticmethod
    def optuna_objective(objective, optimization_target: OptimizationTarget):
        def output_objective(trial):
            optimize_parameters = optimization_target.search_space.copy()
            for key, value in optimize_parameters.items():
                if isinstance(value, MutableParameter):
                    optimize_parameters[key] = NumericalParameter(trial.suggest_float(key, value.value_lims[0],
                                                                                      value.value_lims[1]))

            output = objective(optimize_parameters, optimization_target=optimization_target)
            return tuple(value.mean().item() for value in output[min(output.keys())])

        return output_objective

    def optimize(self, objective, iterations, optimization_target: OptimizationTarget):
        self.optuna_study.optimize(
            self.optuna_objective(objective, optimization_target),
            n_trials=iterations,
            show_progress_bar=True)
        return self.optuna_study.best_params, {}


# class optimizerbackendax(optimizerbackend):
#    def __init__(self, ax_client: axclient, search_space: rayparametercontainer):
#        self.search_space = search_space
#        self.ax_client = ax_client
#
#    def optimizer_parameter_to_container_list(self, optimizer_parameter) -> list[rayparametercontainer]:
#        trial_params_first_key = min(optimizer_parameter.keys())
#        param_container_list = []
#
#        for i in range(trial_params_first_key, trial_params_first_key + len(optimizer_parameter)):
#            all_parameters_copy = self.search_space.copy()
#            for param_key in optimizer_parameter[trial_params_first_key].keys():
#                all_parameters_copy.__setitem__(param_key, numericalparameter(optimizer_parameter[i][param_key]))
#            param_container_list.append(all_parameters_copy)
#        return param_container_list
#
#    def setup_optimization(self):
#        experiment_parameters = []
#        for key, value in self.search_space.items():
#            if isinstance(value, MutableParameter):
#                experiment_parameters.append(
#                    {"name": key, "type": "range", 'value_type': 'float', "bounds": list(value.value_lims)})
#
#        self.ax_client.create_experiment(
#            name="metrix_experiment",
#            parameters=experiment_parameters,
#            objective_name="metrix",
#            minimize=True,
#        )
#
#    def optimize(self, objective, iterations, optimization_target: OptimizationTarget):
#        ranger = trange(iterations)
#        for _ in ranger:
#            optimization_time = time.time()
#            trials_to_evaluate = self.ax_client.get_next_trials(max_trials=10)[0]
#            print("Optimization took {:.2f}s".format(time.time() - optimization_time))
#
#            ray_parameter_container_list = self.optimizer_parameter_to_container_list(trials_to_evaluate)
#            results = objective(ray_parameter_container_list, optimization_target.target_rays,
#                                target_params=optimization_target.target_params)
#
#            for trial_index in results:
#                self.ax_client.complete_trial(trial_index, results[trial_index])
#
#        best_parameters, metrics = self.ax_client.get_best_parameters()
#        return best_parameters, metrics


class LoggingBackend(metaclass=ABCMeta):
    def __init__(self):
        self.log_dict: dict = {}

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


class BestSample:

    def __init__(self):
        self._params: RayParameterContainer = RayParameterContainer()
        self._rays: Union[None, torch.Tensor] = None
        self._loss: float = float('inf')
        self._epoch: int = 0

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value: RayParameterContainer):
        self._params = value

    @property
    def rays(self):
        return self._rays

    @rays.setter
    def rays(self, rays: Union[None, torch.Tensor]):
        self._rays = rays

    @property
    def loss(self):
        return self._loss

    @loss.setter
    def loss(self, loss: float):
        self._loss = loss

    @property
    def epoch(self):
        return self._epoch

    @epoch.setter
    def epoch(self, epoch: int):
        self._epoch = epoch


class RayOptimizer:
    def __init__(self, optimizer_backend: OptimizerBackend, criterion, exported_plane: str,
                 engine: RayEngine, logging_backend: LoggingBackend, transforms: Optional[RayTransform] = None,
                 log_times: bool = False, iterations=1000):
        self.optimizer_backend: OptimizerBackend = optimizer_backend
        self.engine: RayEngine = engine
        self.criterion = criterion
        self.exported_plane: str = exported_plane
        self.transforms: Optional[RayTransform] = transforms
        self.logging_backend: LoggingBackend = logging_backend
        self.log_times: bool = log_times
        self.evaluation_counter: int = 0
        self.iterations: int = iterations
        self.plot_interval_best: BestSample = BestSample()
        self.overall_best: BestSample = BestSample()

    @staticmethod
    def fig_to_image(fig: plt.Figure) -> np.array:
        fig.canvas.draw()

        image_from_plot: np.array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return image_from_plot

    @staticmethod
    def plot_data(pc_supp: list[torch.Tensor], pc_weights: Optional[list[torch.Tensor]] = None,
                  epoch: Optional[int] = None) -> np.array:
        pc_supp = [v.detach().cpu() for v in pc_supp]
        pc_weights = None if pc_weights is None else [v.detach().cpu() for v in pc_weights]
        fig, axs = plt.subplots(len(pc_supp), pc_supp[0].shape[0], squeeze=False)
        for i, column in enumerate(pc_supp):
            for j, line in enumerate(column):
                axs[i, j].scatter(line[:, 0], line[:, 1], s=2.0)
                axs[i, j].yaxis.set_major_locator(plt.NullLocator())
                axs[i, j].xaxis.set_major_locator(plt.NullLocator())
        if epoch is not None:
            fig.suptitle('Epoch ' + str(epoch))
        return RayOptimizer.fig_to_image(fig)

    @staticmethod
    def compensation_plot(compensated: list[torch.Tensor], target: list[torch.Tensor],
                          without_compensation: list[torch.Tensor],
                          epoch: Optional[int] = None) -> np.array:
        compensated = [v.detach().cpu() for v in compensated]
        target = [v.detach().cpu() for v in target]
        fig, axs = plt.subplots(3, len(compensated), squeeze=False)
        for i, data in enumerate(compensated):
            axs[1, i].scatter(target[i][0, :, 0], target[i][0, :, 1], s=2.0)
            axs[1, i].xaxis.set_major_locator(plt.NullLocator())
            axs[1, i].yaxis.set_major_locator(plt.NullLocator())
            xlim, ylim = axs[1, i].get_xlim(), axs[1, i].get_ylim()
            axs[0, i].scatter(without_compensation[i][0, :, 0], without_compensation[i][0, :, 1], s=2.0)
            axs[0, i].xaxis.set_major_locator(plt.NullLocator())
            axs[0, i].yaxis.set_major_locator(plt.NullLocator())
            axs[0, i].set_xlim(xlim)
            axs[0, i].set_ylim(ylim)
            axs[2, i].scatter(data[0, :, 0], data[0, :, 1], s=2.0)
            axs[2, i].set_xlim(xlim)
            axs[2, i].set_ylim(ylim)
            axs[2, i].xaxis.set_major_locator(plt.NullLocator())
            axs[2, i].yaxis.set_major_locator(plt.NullLocator())
        axs[0, 0].set_ylabel('w/o compensation')
        axs[1, 0].set_ylabel('target')
        axs[2, 0].set_ylabel('compensated')
        if epoch is not None:
            fig.suptitle('Epoch ' + str(epoch))
        return RayOptimizer.fig_to_image(fig)

    @staticmethod
    def fixed_position_plot(compensated: list[torch.Tensor], target: list[torch.Tensor],
                            without_compensation: list[torch.Tensor], xlim,
                            ylim,
                            epoch: Optional[int] = None) -> np.array:
        fig, axs = plt.subplots(3, len(compensated), squeeze=False, gridspec_kw={'wspace': 0, 'hspace': 0})
        for beamline_idx in range(len(compensated)):
            axs[0, beamline_idx].scatter(without_compensation[beamline_idx][0, :, 0],
                                         without_compensation[beamline_idx][0, :, 1], s=2.0)
            axs[1, beamline_idx].scatter(target[beamline_idx][0, :, 0], target[beamline_idx][0, :, 1], s=2.0)
            axs[2, beamline_idx].scatter(compensated[beamline_idx][0, :, 0], compensated[beamline_idx][0, :, 1], s=2.0)

            for i in range(3):
                axs[i, beamline_idx].set_xlim(xlim)
                axs[i, beamline_idx].set_ylim(ylim)
                axs[i, beamline_idx].xaxis.set_major_locator(plt.NullLocator())
                axs[i, beamline_idx].yaxis.set_major_locator(plt.NullLocator())
                axs[i, beamline_idx].set_aspect('equal')
                axs[i, beamline_idx].set_xticklabels([])
                axs[i, beamline_idx].set_yticklabels([])

        for i in range(3):
            axs[i, 0].set_ylabel(['w/o comp.', 'target', 'compensated'][i])
        if epoch is not None:
            fig.suptitle('Epoch ' + str(epoch))
        fig.set_size_inches(len(compensated) + 2, 3)
        fig.set_dpi(200)
        return RayOptimizer.fig_to_image(fig)

    @staticmethod
    def normalize_parameters(parameters: RayParameterContainer,
                             search_space: RayParameterContainer) -> RayParameterContainer:
        normalized_parameters = RayParameterContainer()
        for key, value in search_space.items():
            if isinstance(value, MutableParameter):
                normalized_parameters[key] = NumericalParameter((parameters[key].get_value() -
                                                                 value.value_lims[0]) / (value.value_lims[1] -
                                                                                         value.value_lims[0]))
        return normalized_parameters

    @staticmethod
    def translate_transform(transform: RayTransform, xyz_translation: tuple[float, float, float]) -> RayTransform:
        transforms_copy = copy.deepcopy(transform)
        if isinstance(transform, MultiLayer):
            transform.dist_layers = [element + xyz_translation[1] for element in transform.dist_layers]
        translation_transform = Translation(xyz_translation[-1], xyz_translation[1])
        return RayTransformCompose(transforms_copy, translation_transform)

    @staticmethod
    def get_exported_plane_translation(exported_plane: str, param_container: RayParameterContainer):
        x_translation: float = 0.
        y_translation: float = 0.
        z_translation: float = 0.
        for key, param in param_container.items():
            if isinstance(param, OutputParameter) and isinstance(param, RandomParameter):
                if key.split('.')[0] == exported_plane:
                    param_entry = key.split('.')[-1]
                    if param_entry == 'translationXerror':
                        x_translation = param.value
                    if param_entry == 'translationYerror':
                        y_translation = param.value
                    if param_entry == 'translationZerror':
                        z_translation = param.value
        return x_translation, y_translation, z_translation

    @staticmethod
    def translate_exported_plain_transforms(exported_plane: str, param_container_list: List[RayParameterContainer],
                                            transform: RayTransform):
        exported_plane_translations = [RayOptimizer.get_exported_plane_translation(exported_plane, param_container_entry) for
                                       param_container_entry in param_container_list]
        transforms = [RayOptimizer.translate_transform(transform, exported_plane_translation) for exported_plane_translation in
                      exported_plane_translations]
        print([exported_plane_translation for exported_plane_translation in exported_plane_translations])
        return transforms

    def plot_param_comparison(self, predicted_params: RayParameterContainer, search_space: RayParameterContainer,
                              real_params: Optional[RayParameterContainer] = None,
                              omit_labels: Optional[List[str]] = None):
        if omit_labels is None:
            omit_labels = []
        fig, ax = plt.subplots(1, 1, figsize=(16, 9))
        ax.set_ylim([0., 1.])
        if real_params is not None:
            ax.plot([param.get_value() for param in self.normalize_parameters(real_params, search_space).values()],
                    'bo',
                    markersize=20,
                    label='real parameters')
        ax.plot([param.get_value() for param in self.normalize_parameters(predicted_params, search_space).values()],
                'm*',
                markersize=20,
                label='predicted parameters')
        param_labels = [param_key for param_key, param_value in predicted_params.items() if
                        param_key not in omit_labels]
        ax.set_xticks(range(len(param_labels)))
        ax.set_xticklabels(param_labels, rotation=90)
        plt.subplots_adjust(bottom=0.3)
        fig.suptitle('Epoch ' + str(self.plot_interval_best.epoch))
        return RayOptimizer.fig_to_image(fig)

    def ray_output_to_tensor(self, ray_output: Union[Dict, List[Dict], Iterable[Dict]]):
        if not isinstance(ray_output, Dict):
            return [self.ray_output_to_tensor(element) for element in
                    ray_output]
        else:
            rays: dict = ray_output['ray_output'][self.exported_plane]
            x_locs = torch.stack([torch.tensor(value.x_loc) for value in rays.values()])
            y_locs = torch.stack([torch.tensor(value.y_loc) for value in rays.values()])
            return torch.stack((x_locs, y_locs), -1)

    def evaluation_function(self, parameters, optimization_target: OptimizationTarget) -> dict:
        begin_total_time: float = time.time() if self.log_times else None
        if not isinstance(parameters, list):
            parameters = [parameters]

        num_combinations = len(optimization_target.perturbed_parameters_rays)  # TODO might be adapted for multi arm
        initial_parameters = [element.copy() for element in parameters]

        if isinstance(optimization_target, OffsetOptimizationTarget):
            evaluation_parameters = [element.clone() for element in optimization_target.initial_parameters]
            for i, perturbed_parameters in enumerate(optimization_target.initial_parameters):
                for k, v in parameters[0].items():
                    if isinstance(v, RandomParameter):
                        value = perturbed_parameters[k].get_value() - v.get_value()
                        if isinstance(v, OutputParameter):
                            evaluation_parameters[i][k] = NumericalOutputParameter(value)
                        else:
                            evaluation_parameters[i][k] = NumericalParameter(value)

            parameters = evaluation_parameters

        begin_execution_time: float = time.time() if self.log_times else None
        transforms = RayOptimizer.translate_exported_plain_transforms(self.exported_plane, parameters, self.transforms)
        output = self.engine.run(parameters, transforms=transforms)
        if self.log_times:
            self.logging_backend.add_to_log({"execution_time": time.time() - begin_execution_time})

        if not isinstance(output, list):
            output = [output]

        begin_loss_time: float = time.time() if self.log_times else None
        output_loss_dict = self.calculate_loss_from_output(output, optimization_target.perturbed_parameters_rays)
        if self.log_times:
            self.logging_backend.add_to_log({"loss_time": time.time() - begin_loss_time})

        for epoch, (loss, ray_count, loss_mean) in output_loss_dict.items():
            self.logging_backend.add_to_log({"epoch": epoch, "loss": loss_mean, "ray_count": ray_count.mean()})
            if loss_mean < self.plot_interval_best.loss:
                self.plot_interval_best.loss = loss_mean
                self.plot_interval_best.params = initial_parameters[epoch - self.evaluation_counter]
                self.plot_interval_best.epoch = epoch

            if loss_mean < self.overall_best.loss:
                self.overall_best.loss = loss_mean
                self.overall_best.params = initial_parameters[epoch - self.evaluation_counter]
                self.overall_best.epoch = epoch
                best_rays_list = self.overall_best.rays
                target_perturbed_parameters_rays_list = self.ray_output_to_tensor(
                    optimization_target.perturbed_parameters_rays)
                target_initial_parameters_rays_list = self.ray_output_to_tensor(
                    optimization_target.initial_parameters_rays)
                fixed_position_plot = self.fixed_position_plot(best_rays_list, target_perturbed_parameters_rays_list,
                                                               target_initial_parameters_rays_list,
                                                               epoch=self.overall_best.epoch,
                                                               xlim=[-2, 2],
                                                               ylim=[-2, 2])
                self.logging_backend.image("overall_fixed_position_plot", fixed_position_plot)

            current_range = range(self.evaluation_counter, self.evaluation_counter + len(output) // num_combinations)
            if True in [i % 10 == 0 for i in current_range]:
                image = self.plot_data(self.plot_interval_best.rays, epoch=self.plot_interval_best.epoch)
                self.logging_backend.image("footprint", image)
                if isinstance(optimization_target, OffsetOptimizationTarget):
                    compensation_image = self.compensation_plot(self.plot_interval_best.rays, self.ray_output_to_tensor(
                        optimization_target.perturbed_parameters_rays), self.ray_output_to_tensor(
                        optimization_target.initial_parameters_rays), epoch=self.plot_interval_best.epoch)
                    self.logging_backend.image("compensation", compensation_image)
                max_ray_index = torch.argmax(torch.Tensor([self.ray_output_to_tensor(element).shape[1] for element in
                                                           optimization_target.perturbed_parameters_rays])).item()
                fixed_position_plot = self.fixed_position_plot([self.plot_interval_best.rays[max_ray_index]],
                                                               [self.ray_output_to_tensor(
                                                                   optimization_target.perturbed_parameters_rays[
                                                                       max_ray_index])], [self.ray_output_to_tensor(
                        optimization_target.initial_parameters_rays[max_ray_index])],
                                                               epoch=self.plot_interval_best.epoch, xlim=[-2, 2],
                                                               ylim=[-2, 2])
                self.logging_backend.image("fixed_position_plot", fixed_position_plot)
                parameter_comparison_image = self.plot_param_comparison(predicted_params=self.plot_interval_best.params,
                                                                        search_space=optimization_target.search_space,
                                                                        real_params=optimization_target.target_params)
                self.logging_backend.image("parameter_comparison", parameter_comparison_image)
                self.plot_interval_best = BestSample()
            if self.evaluation_counter == 0:
                target_tensor = self.ray_output_to_tensor(optimization_target.perturbed_parameters_rays)
                if isinstance(target_tensor, torch.Tensor):
                    target_tensor = [target_tensor]
                target_image = self.plot_data(target_tensor)
                self.logging_backend.image("target_footprint", target_image)
        if self.log_times:
            self.logging_backend.add_to_log({"total_time": time.time() - begin_total_time})
        self.logging_backend.log()
        self.evaluation_counter += len(output) // num_combinations
        return {epoch: loss for epoch, (loss, _, _) in output_loss_dict.items()}

    def calculate_loss_from_output(self, output, target_rays):
        # if isinstance(output, List):
        #    return [self.calculate_loss_from_output(output_element, target_rays[i]) for i, output_element in
        #            enumerate(output)]
        if isinstance(output, dict):
            if 'ray_output' in output.keys():
                output = [output]

        if isinstance(target_rays, dict):
            if 'ray_output' in target_rays.keys():
                target_rays = [target_rays]

        if isinstance(output, List):
            output = {0: output}

        output_loss = {
            key + self.evaluation_counter: self.calculate_loss_epoch(element, target_rays) for
            key, element in enumerate(output.values())}
        return output_loss

    def calculate_loss_epoch(self, output, target_rays):
        output = self.ray_output_to_tensor(output)
        target_rays = self.ray_output_to_tensor(target_rays)
        num_rays = torch.tensor([element.shape[1] for element in output], dtype=target_rays[0].dtype,
                                device=target_rays[0].device)
        losses = [self.criterion(target_rays[i], output[i]) for i in range(len(output))]
        if isinstance(losses[0], tuple):
            output_losses = []
            for i in range(len(losses[0])):
                if isinstance(losses[0][i], torch.Tensor):
                    output_losses.append(torch.stack([loss[i] for loss in losses]))
                else:
                    output_losses.append(torch.Tensor([loss[i] for loss in losses]))
            losses = tuple(output_losses)
            losses_mean = torch.tensor([loss.mean() for loss in losses]).mean().item()
        else:
            losses = torch.stack(losses),
            losses_mean = losses[0].mean().item()

        if losses_mean < self.plot_interval_best.loss:
            self.plot_interval_best.rays = output
        if losses_mean < self.overall_best.loss:
            self.overall_best.rays = output
        return losses, num_rays, losses_mean

    def optimize(self, optimization_target: OptimizationTarget):
        self.optimizer_backend.setup_optimization()
        best_parameters, metrics = self.optimizer_backend.optimize(objective=self.evaluation_function,
                                                                   iterations=self.iterations,
                                                                   optimization_target=optimization_target)
        return best_parameters, metrics
