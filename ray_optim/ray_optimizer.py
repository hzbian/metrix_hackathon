import copy
from operator import itemgetter
import time
from abc import ABCMeta, abstractmethod
from math import sqrt
from typing import Any, Dict, List, Iterable, Union, Optional, Callable
from multiprocessing import JoinableQueue, Process

import torch
from ray_optim.logging import LoggingBackend
from ray_optim.plot import Plot

from ray_tools.base import RayTransform
from ray_tools.base.engine import Engine
from ray_tools.base.parameter import (
    RayParameterContainer,
    NumericalParameter,
    RandomParameter,
    NumericalOutputParameter,
    OutputParameter,
)
from ray_tools.base.transform import RayTransformCompose, MultiLayer, Translation
from sub_projects.ray_optimization.losses.losses import RayLoss
from sub_projects.ray_optimization.utils import ray_output_to_tensor


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
        self.logger_queue: JoinableQueue = JoinableQueue()
        self.logger_consumer_process: Optional[Process] = None

    @staticmethod
    def parameters_rmse(
        parameters_a: RayParameterContainer,
        parameters_b: RayParameterContainer,
        search_space: RayParameterContainer,
    ):
        counter: int = 0
        mse: float = 0.0
        normalized_parameters_a: RayParameterContainer = (
            Plot.normalize_parameters(parameters_a, search_space)
        )
        normalized_parameters_b: RayParameterContainer = (
            Plot.normalize_parameters(parameters_b, search_space)
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
    def single_parameters_rmse(
        parameters_a: RayParameterContainer,
        parameters_b: RayParameterContainer,
        search_space: RayParameterContainer,
    ):
        output_dict = {}
        normalized_parameters_a: RayParameterContainer = (
            Plot.normalize_parameters(parameters_a, search_space)
        )
        normalized_parameters_b: RayParameterContainer = (
            Plot.normalize_parameters(parameters_b, search_space)
        )
        for key in search_space.keys():
            if key in normalized_parameters_b.keys() and key in normalized_parameters_a:
                mse = (
                    normalized_parameters_a[key].get_value()
                    - normalized_parameters_b[key].get_value()
                ) ** 2
            output_dict["params_rmse/"+key] = sqrt(mse)
        return output_dict

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

    @staticmethod
    def log_consumer(q: JoinableQueue, logging_backend: LoggingBackend):
        internal_list = []
        push_counter = 0
        while True:
            if q.empty():
                time.sleep(1)
            if not q.empty():
                res = q.get(block=False)
                internal_list.append(res)
                internal_list = sorted(internal_list, key=itemgetter("epoch"))
                for i in internal_list:
                    if i["epoch"] == push_counter:
                        logging_backend.log(i)
                        push_counter += 1
                q.task_done()

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
            output,
            target.observed_rays,
            current_epochs,
            self.criterion,
            self.exported_plane,
            compensations,
        )
        if self.log_times:
            log_dict["System/loss_time"] = time.time() - begin_loss_time

        if self.log_times:
            log_dict["System/total_time"]: time.time() - begin_total_time

        if self.logger_consumer_process is None:
            self.logger_consumer_process = Process(
                target=RayOptimizer.log_consumer,
                args=(self.logger_queue, self.logging_backend),
                daemon=True,
            )
            self.logger_consumer_process.start()
        logger_process = Process(
            target=self.new_logger,
            kwargs={
                "queue": self.logger_queue,
                "log_dict": log_dict,
                "trials": trials,
                "target": target,
                "exported_plane": self.exported_plane,
                "plot_interval_best": self.plot_interval_best,
                "overall_best": self.overall_best,
                "plot_interval": self.plot_interval,
                "engine": self.engine,
                "transforms": self.transforms,
                "logging_backend": self.logging_backend,
                "compensations": compensations,
            },
        )

        logger_process.start()

        for sample in trials:
            if RayOptimizer.is_new_interval(self.plot_interval_best.epoch, sample.epoch, self.plot_interval):
                self.plot_interval_best = Sample()
            if sample.loss < self.plot_interval_best.loss:
                self.plot_interval_best = sample
            if sample.loss < self.overall_best.loss:
                self.overall_best = sample
        return [sample.loss for sample in trials]

    @staticmethod
    def is_new_interval(epoch_old: int, epoch_new: int, plot_interval: int):
        old_plot_interval = epoch_old // plot_interval
        new_plot_interval = epoch_new // plot_interval
        return old_plot_interval != new_plot_interval
    
    @staticmethod
    def new_logger(
        queue: JoinableQueue,
        log_dict: Dict[str, Any],
        trials,
        target: Target,
        exported_plane: str,
        plot_interval_best: Sample,
        overall_best: Sample,
        plot_interval: int,
        engine: Engine,
        transforms: RayTransform,
        logging_backend: LoggingBackend,
        compensations,
    ):
        for sample in trials:
            log_dict["epoch"] = sample.epoch
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
                single_params_rmse = RayOptimizer.single_parameters_rmse(
                    target.target_params,
                    sample.params,
                    target.search_space
                )
                log_dict = {**log_dict, **single_params_rmse}
            if sample.epoch % plot_interval == 0 and sample.epoch != 0:
                plots = RayOptimizer.plot(target, exported_plane, plot_interval_best)
            else:
                plots = {}
            if sample.epoch == 0:
                initial_plots = RayOptimizer.plot_initial_plots(target, exported_plane)
            else:
                initial_plots = {}
            if sample.loss < overall_best.loss:
                better_plots = RayOptimizer.on_better_solution_found(
                    target,
                    transforms,
                    sample,
                    exported_plane,
                    engine,
                    compensations,
                )
            else:
                better_plots = {}
            log_plots = {**plots, **initial_plots, **better_plots}
            log_plots = {key: logging_backend.figure_to_image(value) for key, value in log_plots.items()}
            queue.put({**log_dict, **log_plots})
            queue.join()

    @staticmethod
    def on_better_solution_found(
        target: Target,
        transforms: List[RayTransform],
        overall_best: Sample,
        exported_plane: str,
        engine: Engine,
        compensations,
    ):
        output_dict = {}
        if target.validation_scan is not None:
            validation_parameters = RayOptimizer.compensate_parameters_list(
                target.validation_scan.uncompensated_parameters, compensations
            )

        best_rays_list = RayOptimizer.tensor_list_to_cpu(overall_best.rays)
        target_observed_rays_list = ray_output_to_tensor(
            target.observed_rays, exported_plane, to_cpu=True
        )
        target_uncompensated_rays_list = ray_output_to_tensor(
            target.uncompensated_rays, exported_plane, to_cpu=True
        )
        xlim, ylim = Plot.switch_lims_if_out_of_lim(target_observed_rays_list, lims_x=(-2., 2.), lims_y=(-2., 2.))
        fixed_position_plot = Plot.fixed_position_plot(
            best_rays_list,
            target_observed_rays_list,
            target_uncompensated_rays_list,
            epoch=overall_best.epoch,
            xlim=xlim,
            ylim=ylim,
        )
        output_dict["overall_fixed_position_plot"] = fixed_position_plot

        if target.validation_scan is not None:
            validation_scan = target.validation_scan
            validation_rays_list = ray_output_to_tensor(
                validation_scan.observed_rays, exported_plane=exported_plane
            )
            validation_parameters_rays_list = ray_output_to_tensor(
                validation_scan.uncompensated_rays, exported_plane
            )
            transforms = RayOptimizer.translate_exported_plain_transforms(
                exported_plane, validation_parameters, transforms
            )
            compensated_rays_list = engine.run(validation_parameters, transforms)
            compensated_rays_list = ray_output_to_tensor(
                compensated_rays_list, exported_plane=exported_plane
            )
            xlim, ylim = Plot.switch_lims_if_out_of_lim(validation_rays_list, lims_x=(-2., 2.), lims_y=(-2., 2.))
            validation_fixed_position_plot = Plot.fixed_position_plot(
                compensated_rays_list,
                validation_rays_list,
                validation_parameters_rays_list,
                epoch=overall_best.epoch,
                xlim=xlim,
                ylim=ylim,
            )

            output_dict["validation_fixed_position"] = validation_fixed_position_plot
        return output_dict

    @staticmethod
    def plot_initial_plots(target: Target, exported_plane: str):
        output_dict = {}
        target_tensor = ray_output_to_tensor(target.observed_rays, exported_plane)
        if isinstance(target_tensor, torch.Tensor):
            target_tensor = [target_tensor]
        target_plot = Plot.plot_data(target_tensor)
        output_dict["target_footprint"] = target_plot
        fancy_plot = Plot.fancy_ray([target_tensor])
        output_dict["fancy_footprint"] = fancy_plot
        return output_dict

    @staticmethod
    def plot(target: Target, exported_plane: str, plot_interval_best: Sample):
        output_dict = {}
        interval_best_rays = RayOptimizer.tensor_list_to_cpu(plot_interval_best.rays)
        plot = Plot.plot_data(interval_best_rays, epoch=plot_interval_best.epoch)
        output_dict["footprint"] = plot
        if isinstance(target, OffsetTarget):
            observed_rays = ray_output_to_tensor(target.observed_rays, exported_plane, to_cpu=True)
            uncompensated_rays = ray_output_to_tensor(
                    target.uncompensated_rays,
                    exported_plane,
                    to_cpu=True,
                )
            compensation_plot = Plot.compensation_plot(
                interval_best_rays,
                observed_rays,
                uncompensated_rays,
                epoch=plot_interval_best.epoch,
            )
            output_dict["compensation"] = compensation_plot
            fancy_ray_plot = Plot.fancy_ray([uncompensated_rays, observed_rays, interval_best_rays])
            output_dict["fancy_ray"] = fancy_ray_plot
        max_ray_index = torch.argmax(
            torch.Tensor(
                [
                    ray_output_to_tensor(element, exported_plane).shape[1]
                    for element in target.observed_rays
                ]
            )
        ).item()
        target_observed_rays_list = [ray_output_to_tensor(
                    target.observed_rays[max_ray_index],
                    exported_plane,
                    to_cpu=True,
                )]
        xlim, ylim = Plot.switch_lims_if_out_of_lim(target_observed_rays_list, lims_x=(-2., 2.), lims_y=(-2., 2.))
        fixed_position_plot = Plot.fixed_position_plot(
            [interval_best_rays[max_ray_index]],
            target_observed_rays_list,
            [
                ray_output_to_tensor(
                    target.uncompensated_rays[max_ray_index],
                    exported_plane,
                    to_cpu=True,
                )
            ],
            epoch=plot_interval_best.epoch,
            xlim=xlim,
            ylim=ylim,
        )
        output_dict["fixed_position_plot"] = fixed_position_plot
        parameter_comparison_plot = Plot.plot_param_comparison(
            predicted_params=plot_interval_best.params,
            epoch=plot_interval_best.epoch,
            search_space=target.search_space,
            real_params=target.target_params,
        )
        output_dict["parameter_comparison"] = parameter_comparison_plot
        return output_dict

    @staticmethod
    def calculate_loss_from_output(
        output: List[List[dict]],
        target_rays,
        current_epochs: List[int],
        criterion: RayLoss,
        exported_plane: str,
        compensations: List[RayParameterContainer],
    ) -> List[Sample]:
        list = [
            RayOptimizer.loss_from_epoch(
                current_epochs[i],
                element,
                target_rays,
                criterion.loss_fn,
                exported_plane,
                compensations[i],
            )
            for i, element in enumerate(output)
        ]
        return list

    @staticmethod
    def loss_from_epoch(
        epoch,
        output,
        target_rays,
        loss_fn,
        exported_plane: str,
        compensation: RayParameterContainer,
    ):
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
