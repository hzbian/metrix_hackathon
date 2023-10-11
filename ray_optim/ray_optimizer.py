from collections import OrderedDict
import copy
import time
import os
from abc import ABCMeta, abstractmethod
from math import sqrt
from typing import Any, Dict, List, Iterable, Union, Optional, Callable
import multiprocessing as mp

import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt

from ray_tools.base import RayTransform
from ray_tools.base.engine import Engine
from ray_tools.base.parameter import (
    RayParameterContainer,
    MutableParameter,
    NumericalParameter,
    RandomParameter,
    NumericalOutputParameter,
    OutputParameter,
)
from ray_tools.base.transform import RayTransformCompose, MultiLayer, Translation
from sub_projects.ray_optimization.losses.losses import RayLoss
from sub_projects.ray_optimization.utils import ray_output_to_tensor

plt.switch_backend("Agg")


class RayScan:
    def __init__(
        self,
        uncompensated_parameters: List[RayParameterContainer],
        uncompensated_rays: Union[dict, Iterable[dict], List[dict]],
        observed_rays: Union[dict, Iterable[dict], List[dict]],
    ):
        self.uncompensated_parameters: List[
            RayParameterContainer
        ] = uncompensated_parameters
        self.uncompensated_rays: Union[
            dict, Iterable[dict], List[dict]
        ] = uncompensated_rays
        self.observed_rays: Union[dict, Iterable[dict], List[dict]] = observed_rays


class Target:
    def __init__(
        self,
        observed_rays: Union[dict, Iterable[dict], List[dict]],
        search_space: RayParameterContainer,
        target_params: Optional[RayParameterContainer] = None,
    ):
        self.observed_rays = observed_rays
        self.search_space = search_space
        self.target_params = target_params


class OffsetTarget(Target):
    def __init__(
        self,
        observed_rays: Union[dict, Iterable[dict], List[dict]],
        offset_search_space: RayParameterContainer,
        uncompensated_parameters: List[RayParameterContainer],
        uncompensated_rays: Union[dict, Iterable[dict], List[dict]],
        target_compensation: Optional[RayParameterContainer] = None,
        validation_scan: Optional[RayScan] = None,
    ):
        super().__init__(observed_rays, offset_search_space, target_compensation)
        self.uncompensated_parameters: List[
            RayParameterContainer
        ] = uncompensated_parameters
        self.uncompensated_rays: Union[
            dict, Iterable[dict], List[dict]
        ] = uncompensated_rays
        self.validation_scan: Optional[RayScan] = validation_scan


class OptimizerBackend(metaclass=ABCMeta):
    @abstractmethod
    def setup_optimization(self):
        pass

    @abstractmethod
    def optimize(
        self,
        objective: Callable,
        iterations: int,
        target: Target,
    ):
        pass


class Logger:
    def __init__(self, queue):
        self.logged_until_index = 0
        self.log_list = []
        self.queue = queue

    def log(self):
        while not self.queue.empty():
            self.log_list.append(self.queue.get())
        log_dict = OrderedDict(self.log_list)
        log_dict.sorted_keys = lambda: sorted(log_dict.keys())
        for element in log_dict.sorted_keys():
            if element == self.logged_until_index:
                self.logged_until_index += 1
                print(log_dict[element])
                del log_dict[element]
                self.log_list = list(log_dict.items())
            else:
                break


class LoggingBackend(metaclass=ABCMeta):
    def __init__(self):
        super().__init__()
        self.log_dict: Dict[str, Any] = {}
        self.logged_until_index = 0

    def _add_to_log(self, add_to_log: Dict[str, Any]):
        self.log_dict = {**self.log_dict, **add_to_log}

    def empty_log(self):
        self.log_dict = {}

    def log(self, log: Dict[str, Any]):
        self._add_to_log(log)
        self._log()
        self.empty_log()

    @abstractmethod
    def _log(self):
        pass

    @abstractmethod
    def image(self, key: Union[str, int], image: torch.Tensor):
        pass


class WandbLoggingBackend(LoggingBackend):
    def __init__(
        self, logging_entity: str, project_name: str, study_name: str, logging: bool
    ):
        super().__init__()
        os.environ["WANDB__SERVICE_WAIT"] = "300"
        self.handle = wandb.init(
            entity=logging_entity,
            project=project_name,
            name=study_name,
            mode="online" if logging else "disabled",
        )

    def _log(self):
        self.handle.log(self.log_dict)

    def image(self, key: Union[str, int], image: torch.Tensor):
        image = wandb.Image(image)
        return {key: image}


class Sample:
    def __init__(
        self,
        params=RayParameterContainer(),
        rays: Optional[torch.Tensor] = None,
        loss: float = float("inf"),
        epoch: int = 0,
    ):
        self._params: RayParameterContainer = params
        self._rays: Optional[torch.Tensor] = rays
        self._loss: float = loss
        self._epoch: int = epoch

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
    def rays(self, rays: Optional[torch.Tensor]):
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
    def __init__(
        self,
        optimizer_backend: OptimizerBackend,
        criterion: RayLoss,
        exported_plane: str,
        engine: Engine,
        logging_backend: LoggingBackend,
        transforms: Optional[RayTransform] = None,
        log_times: bool = False,
        plot_interval: int = 10,
        iterations: int = 1000,
    ):
        self.optimizer_backend: OptimizerBackend = optimizer_backend
        self.engine: Engine = engine
        self.criterion: RayLoss = criterion
        self.exported_plane: str = exported_plane
        self.transforms: Optional[RayTransform] = transforms
        self.logging_backend: LoggingBackend = logging_backend
        self.log_times: bool = log_times
        self.evaluation_counter: int = 0
        self.iterations: int = iterations
        self.plot_interval_best: Sample = Sample()
        self.overall_best: Sample = Sample()
        self.plot_interval: int = plot_interval
        self.logger_process: Optional[mp.Process] = None

    @staticmethod
    def fig_to_image(fig: plt.Figure) -> np.array:
        fig.canvas.draw()

        image_from_plot: np.array = np.frombuffer(
            fig.canvas.tostring_rgb(), dtype=np.uint8
        )
        image_from_plot = image_from_plot.reshape(
            fig.canvas.get_width_height()[::-1] + (3,)
        )
        plt.close(fig)
        return image_from_plot

    @staticmethod
    def plot_data(
        pc_supp: list[torch.Tensor],
        pc_weights: Optional[list[torch.Tensor]] = None,
        epoch: Optional[int] = None,
    ) -> np.array:
        pc_supp = [v.detach().cpu() for v in pc_supp]
        pc_weights = (
            None if pc_weights is None else [v.detach().cpu() for v in pc_weights]
        )
        fig, axs = plt.subplots(len(pc_supp), pc_supp[0].shape[0], squeeze=False)
        for i, column in enumerate(pc_supp):
            for j, line in enumerate(column):
                axs[i, j].scatter(line[:, 0], line[:, 1], s=2.0)
                axs[i, j].yaxis.set_major_locator(plt.NullLocator())
                axs[i, j].xaxis.set_major_locator(plt.NullLocator())
        if epoch is not None:
            fig.suptitle("Epoch " + str(epoch))
        return RayOptimizer.fig_to_image(fig)

    @staticmethod
    def compensation_plot(
        compensated: list[torch.Tensor],
        target: list[torch.Tensor],
        without_compensation: list[torch.Tensor],
        epoch: Optional[int] = None,
    ) -> np.array:
        compensated = [v.detach().cpu() for v in compensated]
        target = [v.detach().cpu() for v in target]
        fig, axs = plt.subplots(3, len(compensated), squeeze=False)
        for i, data in enumerate(compensated):
            axs[1, i].scatter(target[i][0, :, 0], target[i][0, :, 1], s=2.0)
            axs[1, i].xaxis.set_major_locator(plt.NullLocator())
            axs[1, i].yaxis.set_major_locator(plt.NullLocator())
            xlim, ylim = axs[1, i].get_xlim(), axs[1, i].get_ylim()
            axs[0, i].scatter(
                without_compensation[i][0, :, 0],
                without_compensation[i][0, :, 1],
                s=2.0,
            )
            axs[0, i].xaxis.set_major_locator(plt.NullLocator())
            axs[0, i].yaxis.set_major_locator(plt.NullLocator())
            axs[0, i].set_xlim(xlim)
            axs[0, i].set_ylim(ylim)
            axs[2, i].scatter(data[0, :, 0], data[0, :, 1], s=2.0)
            axs[2, i].set_xlim(xlim)
            axs[2, i].set_ylim(ylim)
            axs[2, i].xaxis.set_major_locator(plt.NullLocator())
            axs[2, i].yaxis.set_major_locator(plt.NullLocator())
        axs[0, 0].set_ylabel("w/o compensation")
        axs[1, 0].set_ylabel("target")
        axs[2, 0].set_ylabel("compensated")
        if epoch is not None:
            fig.suptitle("Epoch " + str(epoch))
        return RayOptimizer.fig_to_image(fig)

    @staticmethod
    def fixed_position_plot(
        compensated: list[torch.Tensor],
        target: list[torch.Tensor],
        without_compensation: list[torch.Tensor],
        xlim,
        ylim,
        epoch: Optional[int] = None,
    ) -> np.array:
        y_label = ["w/o comp.", "observed", "compensated"]
        suptitle = "Epoch" + str(epoch) if epoch is not None else None
        return RayOptimizer.fixed_position_plot_base(
            [without_compensation, target, compensated], xlim, ylim, y_label, suptitle
        )

    @staticmethod
    def fixed_position_plot_base(
        tensor_list_list: list[list[torch.Tensor]],
        xlim,
        ylim,
        ylabel,
        suptitle: Optional[str] = None,
    ):
        fig, axs = plt.subplots(
            len(tensor_list_list),
            len(tensor_list_list[0]),
            squeeze=False,
            gridspec_kw={"wspace": 0, "hspace": 0},
        )
        for idx_list_list in range(len(tensor_list_list)):
            for beamline_idx in range(len(tensor_list_list[0])):
                element = tensor_list_list[idx_list_list][beamline_idx]
                axs[idx_list_list, beamline_idx].scatter(
                    element[0, :, 0], element[0, :, 1], s=2.0
                )
                axs[idx_list_list, beamline_idx].set_xlim(xlim)
                axs[idx_list_list, beamline_idx].set_ylim(ylim)
                axs[idx_list_list, beamline_idx].xaxis.set_major_locator(
                    plt.NullLocator()
                )
                axs[idx_list_list, beamline_idx].yaxis.set_major_locator(
                    plt.NullLocator()
                )
                axs[idx_list_list, beamline_idx].set_aspect("equal")
                axs[idx_list_list, beamline_idx].set_xticklabels([])
                axs[idx_list_list, beamline_idx].set_yticklabels([])

            axs[idx_list_list, 0].set_ylabel(ylabel[idx_list_list])
            if suptitle is not None:
                fig.suptitle(suptitle)
        fig.set_size_inches(len(tensor_list_list[0]) + 2, 3)
        fig.set_dpi(200)
        return fig

    @staticmethod
    def normalize_parameters(
        parameters: RayParameterContainer, search_space: RayParameterContainer
    ) -> RayParameterContainer:
        normalized_parameters = RayParameterContainer()
        for key, value in search_space.items():
            if isinstance(value, MutableParameter):
                normalized_parameters[key] = NumericalParameter(
                    (parameters[key].get_value() - value.value_lims[0])
                    / (value.value_lims[1] - value.value_lims[0])
                )
        return normalized_parameters

    @staticmethod
    def parameters_rmse(
        parameters_a: RayParameterContainer,
        parameters_b: RayParameterContainer,
        search_space: RayParameterContainer,
    ):
        counter: int = 0
        mse: float = 0.0
        normalized_parameters_a: RayParameterContainer = (
            RayOptimizer.normalize_parameters(parameters_a, search_space)
        )
        normalized_parameters_b: RayParameterContainer = (
            RayOptimizer.normalize_parameters(parameters_b, search_space)
        )
        for key in search_space.keys():
            if key in normalized_parameters_b.keys() and key in normalized_parameters_a:
                mse += (
                    normalized_parameters_a[key].get_value()
                    - normalized_parameters_b[key].get_value()
                ) ** 2
                counter += 1
        if counter != 0:
            return sqrt(mse / counter)
        else:
            return 0

    @staticmethod
    def parameters_list_rmse(
        parameters_a: RayParameterContainer,
        parameters_b_list: List[RayParameterContainer],
        search_space: RayParameterContainer,
    ):
        rmse_sum: float = 0.0
        for parameters_b in parameters_b_list:
            rmse_sum += RayOptimizer.parameters_rmse(
                parameters_a, parameters_b, search_space
            )
        return rmse_sum / len(parameters_b_list)

    @staticmethod
    def translate_transform(
        transform: RayTransform, xyz_translation: tuple[float, float, float]
    ) -> RayTransform:
        transforms_copy = copy.deepcopy(transform)
        if isinstance(transform, MultiLayer):
            transforms_copy.dist_layers = [
                element + xyz_translation[2] for element in transform.dist_layers
            ]
        translation_transform = Translation(xyz_translation[0], xyz_translation[1])
        return RayTransformCompose(transforms_copy, translation_transform)

    @staticmethod
    def get_exported_plane_translation(
        exported_plane: str, param_container: RayParameterContainer
    ):
        x_translation: float = 0.0
        y_translation: float = 0.0
        z_translation: float = 0.0
        for key, param in param_container.items():
            if isinstance(param, OutputParameter) and isinstance(
                param, NumericalParameter
            ):
                if key.split(".")[0] == exported_plane:
                    param_entry = key.split(".")[-1]
                    if param_entry == "translationXerror":
                        x_translation = param.value
                    if param_entry == "translationYerror":
                        y_translation = param.value
                    if param_entry == "translationZerror":
                        z_translation = param.value
        return x_translation, y_translation, z_translation

    @staticmethod
    def translate_exported_plain_transforms(
        exported_plane: str,
        param_container_list: List[RayParameterContainer],
        transform: RayTransform,
    ):
        exported_plane_translations = [
            RayOptimizer.get_exported_plane_translation(
                exported_plane, param_container_entry
            )
            for param_container_entry in param_container_list
        ]
        transforms = [
            RayOptimizer.translate_transform(transform, exported_plane_translation)
            for exported_plane_translation in exported_plane_translations
        ]
        return transforms

    @staticmethod
    def plot_param_comparison(
        predicted_params: RayParameterContainer,
        search_space: RayParameterContainer,
        epoch: int,
        real_params: Optional[RayParameterContainer] = None,
        omit_labels: Optional[List[str]] = None,
    ):
        if omit_labels is None:
            omit_labels = []
        fig, ax = plt.subplots(1, 1, figsize=(16, 9))
        ax.set_ylim([-0.5, 0.5])
        if real_params is not None:
            ax.stem(
                [
                    param.get_value() - 0.5
                    for param in RayOptimizer.normalize_parameters(
                        real_params, search_space
                    ).values()
                ],
                label="real parameters",
            )
        ax.stem(
            [
                param.get_value() - 0.5
                for param in RayOptimizer.normalize_parameters(
                    predicted_params, search_space
                ).values()
            ],
            linefmt="g",
            markerfmt="o",
            label="predicted parameters",
        )
        param_labels = [
            param_key
            for param_key, param_value in predicted_params.items()
            if param_key not in omit_labels
        ]
        ax.set_xticks(range(len(param_labels)))
        ax.set_xticklabels(param_labels, rotation=90)
        plt.subplots_adjust(bottom=0.3)
        fig.suptitle("Epoch " + str(epoch))
        return RayOptimizer.fig_to_image(fig)

    @staticmethod
    def tensor_list_to_cpu(tensor_list: List[torch.Tensor]):
        return [element.cpu() for element in tensor_list]

    @staticmethod
    def compensate_parameters_list(
        uncompensated_parameters_list: List[RayParameterContainer],
        compensation: RayParameterContainer,
    ) -> List[RayParameterContainer]:
        evaluation_parameters_list = [
            element.clone() for element in uncompensated_parameters_list
        ]
        for i, uncompensated_parameters in enumerate(uncompensated_parameters_list):
            for offsets_k, offsets_value in compensation.items():
                if isinstance(uncompensated_parameters[offsets_k], RandomParameter):
                    value = (
                        uncompensated_parameters[offsets_k].get_value()
                        + offsets_value.get_value()
                    )
                    if isinstance(uncompensated_parameters[offsets_k], OutputParameter):
                        evaluation_parameters_list[i][
                            offsets_k
                        ] = NumericalOutputParameter(value)
                    else:
                        evaluation_parameters_list[i][offsets_k] = NumericalParameter(
                            value
                        )
        return evaluation_parameters_list

    @staticmethod
    def compensate_list_parameters_list(
        uncompensated_parameters_list: List[RayParameterContainer],
        compensation_list: List[RayParameterContainer],
    ):
        return [
            RayOptimizer.compensate_parameters_list(
                uncompensated_parameters_list, compensation
            )
            for compensation in compensation_list
        ]

    @staticmethod
    def flatten_list_list(list_list: List[List[Any]]):
        flattened_list = []
        for list in list_list:
            for element in list:
                flattened_list.append(element)
        return flattened_list

    @staticmethod
    def reshape_list(list: List[Any], list_list_length: int) -> List[List[Any]]:
        if len(list) % list_list_length != 0:
            raise ValueError(
                "Lenght of list is not dividable by the length of the list lists."
            )
        new_list_length = len(list) // list_list_length
        return [
            list[i * list_list_length : (i + 1) * list_list_length]
            for i in range(new_list_length)
        ]

    @staticmethod
    def current_epochs(evaluation_counter: int, length: int) -> List[int]:
        """This function returns all current epochs. This can be multiple values for multi-arm optimization methods.

        Args:
            evaluation_counter (int): The number of already performed evaluations, this is the first value of current epochs.
            length (int): This is the amount of new current epochs.

        Returns:
            List[int]: A list with integers of current evaluation epochs.
        """
        return [i for i in range(evaluation_counter, evaluation_counter + length)]

    @staticmethod
    def evaluation_parameters(
        target: Target, compensations: List[RayParameterContainer]
    ) -> List[List[RayParameterContainer]]:
        if isinstance(target, OffsetTarget):
            evaluation_parameters = RayOptimizer.compensate_list_parameters_list(
                target.uncompensated_parameters, compensations
            )
        else:
            evaluation_parameters = [compensations]
        return evaluation_parameters

    @staticmethod
    def run_engine(
        engine: Engine,
        evaluation_parameters: List[List[RayParameterContainer]],
        exported_plane: str,
        transforms: RayTransform,
        log_times: bool,
    ):
        flattened_evaluation_parameters: List[
            RayParameterContainer
        ] = RayOptimizer.flatten_list_list(evaluation_parameters)

        transforms = RayOptimizer.translate_exported_plain_transforms(
            exported_plane, flattened_evaluation_parameters, transforms
        )

        begin_execution_time: float = time.time() if log_times else None
        output = engine.run(flattened_evaluation_parameters, transforms=transforms)
        execution_time = time.time() - begin_execution_time if log_times else None
        reshaped_output = RayOptimizer.reshape_list(
            output, len(evaluation_parameters[0])
        )
        return reshaped_output, execution_time

    def evaluation_function(
        self, compensations: List[RayParameterContainer], target: Target
    ) -> List[Any]:
        assert isinstance(compensations, list)
        assert isinstance(compensations[0], RayParameterContainer)

        begin_total_time: Optional[float] = time.time() if self.log_times else None

        log_dict: Dict[str, Any] = {}
        current_epochs = RayOptimizer.current_epochs(
            self.evaluation_counter, len(compensations)
        )

        evaluation_parameters = RayOptimizer.evaluation_parameters(
            target, compensations
        )

        output, execution_time = RayOptimizer.run_engine(
            self.engine,
            evaluation_parameters,
            self.exported_plane,
            self.transforms,
            self.log_times,
        )

        if self.log_times:
            log_dict["System/execution_time"] = execution_time

        self.evaluation_counter += len(output)
        begin_loss_time: float = time.time() if self.log_times else None
        trials = self.calculate_loss_from_output(
            output, target.observed_rays, current_epochs, self.criterion, self.exported_plane, compensations
        )
        if self.log_times:
            log_dict["System/loss_time"] = time.time() - begin_loss_time

        for sample in trials:
            if sample.loss < self.plot_interval_best.loss:
                self.plot_interval_best = sample
            if sample.loss < self.overall_best.loss:
                self.overall_best = sample
        if self.log_times:
            log_dict["System/total_time"]: time.time() - begin_total_time
        if True in [i % self.plot_interval == 0 for i in current_epochs]:
            if self.logger_process is not None:
                self.logger_process.join()
            self.logger_process = mp.Process(
                target=self.new_logger,
                kwargs={
                    "log_dict": log_dict,
                    "trials": trials,
                    "target": target,
                    "exported_plane": self.exported_plane,
                    "plot_interval_best": self.plot_interval_best,
                    "engine": self.engine,
                    "logging_backend": self.logging_backend,
                },
            )
            self.logger_process.start()
            #self.logger_process.join()
        return [sample.loss for sample in trials]

    @staticmethod
    def new_logger(
        log_dict: Dict[str, Any],
        trials,
        target: Target,
        exported_plane: str,
        plot_interval_best: Sample,
        logging_backend: LoggingBackend,
        engine: Engine,
    ):
        for sample in trials:
            log_dict["epoch"]: sample.epoch
            log_dict["ray_count"] = (
                torch.tensor(
                    [layer.shape[1] for layer in sample.rays], dtype=torch.float
                )
                .mean()
                .item()
            )
            log_dict["loss"] = sample.loss
            if target.target_params is not None:
                log_dict["params_rmse"] = RayOptimizer.parameters_rmse(
                    target.target_params,
                    sample.params,
                    target.search_space,
                )
            plots = RayOptimizer.plot(target, exported_plane, plot_interval_best)
            if sample.epoch == 0:
                initial_plots = RayOptimizer.plot_initial_plots(target, exported_plane)
            else:
                initial_plots = {}
            logging_backend.log({**log_dict, **plots, **initial_plots})

    @staticmethod
    def on_better_solution_found(
        target: Target,
        validation_parameters,
        transforms: List[RayTransform],
        overall_best: Sample,
        exported_plane: str,
        engine: Engine,
    ):
        output_dict = {}
        if target.validation_scan is not None:
            validation_parameters = RayOptimizer.compensate_parameters(
                target.validation_scan.uncompensated_parameters, compensations
            )

        best_rays_list = RayOptimizer.tensor_list_to_cpu(overall_best.rays)
        target_perturbed_parameters_rays_list = ray_output_to_tensor(
            target.observed_rays, exported_plane, to_cpu=True
        )
        target_initial_parameters_rays_list = ray_output_to_tensor(
            target.uncompensated_rays, exported_plane, to_cpu=True
        )
        fixed_position_plot = RayOptimizer.fixed_position_plot(
            best_rays_list,
            target_perturbed_parameters_rays_list,
            target_initial_parameters_rays_list,
            epoch=overall_best.epoch,
            xlim=[-2, 2],
            ylim=[-2, 2],
        )
        fixed_position_plot = RayOptimizer.fig_to_image(fixed_position_plot)
        output_dict["overall_fixed_position_plot"] = fixed_position_plot

        if target.validation_scan is not None:
            validation_scan = target.validation_scan
            validation_rays_list = ray_output_to_tensor(
                validation_scan.observed_rays, exported_plane=exported_plane
            )
            validation_parameters_rays_list = ray_output_to_tensor(
                validation_scan.uncompensated_rays, exported_plane
            )
            compensated_rays_list = engine.run(validation_parameters, transforms)
            compensated_rays_list = ray_output_to_tensor(
                compensated_rays_list, exported_plane=exported_plane
            )
            validation_fixed_position_plot = RayOptimizer.fixed_position_plot(
                compensated_rays_list,
                validation_rays_list,
                validation_parameters_rays_list,
                epoch=overall_best.epoch,
                xlim=[-2, 2],
                ylim=[-2, 2],
            )

            validation_fixed_position_plot = RayOptimizer.fig_to_image(
                validation_fixed_position_plot
            )
            output_dict["validation_fixed_position"] = validation_fixed_position_plot
        return output_dict

    @staticmethod
    def plot_initial_plots(target: Target, exported_plane: str):
        output_dict = {}
        target_tensor = ray_output_to_tensor(target.observed_rays, exported_plane)
        if isinstance(target_tensor, torch.Tensor):
            target_tensor = [target_tensor]
            target_image = RayOptimizer.plot_data(target_tensor)
            output_dict["target_footprint"] = target_image
        return output_dict

    @staticmethod
    def plot(target: Target, exported_plane: str, plot_interval_best: Sample):
        output_dict = {}
        interval_best_rays = RayOptimizer.tensor_list_to_cpu(plot_interval_best.rays)
        image = RayOptimizer.plot_data(
            interval_best_rays, epoch=plot_interval_best.epoch
        )
        output_dict["footprint"] = image
        if isinstance(target, OffsetTarget):
            compensation_image = RayOptimizer.compensation_plot(
                interval_best_rays,
                ray_output_to_tensor(target.observed_rays, exported_plane, to_cpu=True),
                ray_output_to_tensor(
                    target.uncompensated_rays,
                    exported_plane,
                    to_cpu=True,
                ),
                epoch=plot_interval_best.epoch,
            )
            output_dict["compensation"] = compensation_image
        max_ray_index = torch.argmax(
            torch.Tensor(
                [
                    ray_output_to_tensor(element, exported_plane).shape[1]
                    for element in target.observed_rays
                ]
            )
        ).item()
        fixed_position_plot = RayOptimizer.fixed_position_plot(
            [interval_best_rays[max_ray_index]],
            [
                ray_output_to_tensor(
                    target.observed_rays[max_ray_index],
                    exported_plane,
                    to_cpu=True,
                )
            ],
            [
                ray_output_to_tensor(
                    target.uncompensated_rays[max_ray_index],
                    exported_plane,
                    to_cpu=True,
                )
            ],
            epoch=plot_interval_best.epoch,
            xlim=[-2, 2],
            ylim=[-2, 2],
        )
        fixed_position_plot = RayOptimizer.fig_to_image(fixed_position_plot)
        output_dict["fixed_position_plot"] = fixed_position_plot
        parameter_comparison_image = RayOptimizer.plot_param_comparison(
            predicted_params=plot_interval_best.params,
            epoch=plot_interval_best.epoch,
            search_space=target.search_space,
            real_params=target.target_params,
        )
        output_dict["parameter_comparison"] = parameter_comparison_image
        return output_dict

    @staticmethod
    def calculate_loss_from_output(
        output: List[List[dict]], target_rays, current_epochs: List[int], criterion: RayLoss, exported_plane: str, compensations: List[RayParameterContainer]
    ) -> List[Sample]:
        list = [
            RayOptimizer.loss_from_epoch(current_epochs[i], element, target_rays, criterion.loss_fn, exported_plane, compensations[i])
            for i, element in enumerate(output)
        ]
        return list
    @staticmethod
    def loss_from_epoch(epoch, output, target_rays, loss_fn, exported_plane: str, compensation: RayParameterContainer):
        output_tensor = ray_output_to_tensor(output, exported_plane)
        new_loss = 0
        for i in range(len(output)):
            new_loss += loss_fn(
                target_rays[i], output[i], exported_plane=exported_plane
            )
        mean_loss = new_loss / len(output)
        new_sample = Sample(
            params=compensation,
            rays=output_tensor,
            loss=mean_loss,
            epoch=epoch,
        )
        return new_sample

    def calculate_loss_epoch(self, output, target_rays):
        output_tensor = ray_output_to_tensor(output, self.exported_plane)
        # target_rays = self.ray_output_to_tensor(target_rays, self.exported_plane)
        num_rays = torch.tensor(
            [element.shape[1] for element in output_tensor],
            dtype=output_tensor[0].dtype,
            device=output_tensor[0].device,
        )
        losses = [
            self.criterion.loss_fn(
                target_rays[i], output[i], exported_plane=self.exported_plane
            )
            for i in range(len(output))
        ]
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
            losses = (torch.stack(losses),)
            losses_mean = losses[0].mean().item()

        if losses_mean < self.plot_interval_best.loss:
            self.plot_interval_best.rays = output_tensor
        if losses_mean < self.overall_best.loss:
            self.overall_best.rays = output_tensor
        return losses, num_rays, losses_mean

    def optimize(self, target: Target):
        self.optimizer_backend.setup_optimization()
        best_parameters, metrics = self.optimizer_backend.optimize(
            objective=self.evaluation_function,
            iterations=self.iterations,
            target=target,
        )
        return best_parameters, metrics
