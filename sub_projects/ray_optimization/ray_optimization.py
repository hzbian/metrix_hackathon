import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Tuple, Dict, Union, Iterable

from hydra.core.config_store import ConfigStore

import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, '../../')
from ray_tools.base import RayTransform
from ray_tools.base.utils import RandomGenerator

from ray_tools.base.engine import Engine
from sub_projects.ray_optimization.losses import RayLoss
from sub_projects.ray_optimization.real_data import import_data
from ray_optim.ray_optimizer import RayOptimizer, WandbLoggingBackend, \
    OffsetOptimizationTarget, RayScan, OptimizerBackend

from ray_tools.base.parameter import RayParameterContainer, NumericalParameter, MutableParameter, \
    RayParameter, RandomParameter


class RealDataConfiguration:
    def __init__(self, real_data_dir: str, real_data_train_set: List[str],
                 real_data_validation_set: Optional[List[str]] = None):
        self.real_data_dir: str = real_data_dir
        self.real_data_train_set: List[str] = real_data_train_set
        self.real_data_validation_set: Optional[List[str]] = real_data_validation_set


class RayOptimization:
    def __init__(self, engine: Engine, param_func: Callable, optimizer_backend: OptimizerBackend, exported_plane: str,
                 criterion: RayLoss, wandb_entity: str, wandb_project: str, study_name: str, rg: RandomGenerator,
                 logging: bool = True,
                 plot_interval: int = 10, iterations: int = 1000, num_beamline_samples: int = 20,
                 overwrite_offset_func: Optional[Callable] = None, max_offset_search_deviation: float = 0.3,
                 max_target_deviation: float = 0.3, fixed_params: Optional[List[str]] = (),
                 z_layers: List[float] = 0., transforms: Optional[RayTransform] = None,
                 real_data_configuration: Optional[RealDataConfiguration] = None):
        self.rg: RandomGenerator = rg
        self.engine: Engine = engine
        self.real_data_configuration: RealDataConfiguration = real_data_configuration
        self.optimizer_backend: OptimizerBackend = optimizer_backend
        self.exported_plane: str = exported_plane
        self.criterion: RayLoss = criterion
        self.wandb_entity: str = wandb_entity
        self.wandb_project: str = wandb_project
        self.study_name: str = study_name
        self.logging: bool = logging
        self.plot_interval: int = plot_interval
        self.iterations: int = iterations
        self.num_beamline_samples: int = num_beamline_samples
        self.z_layers: List[float] = z_layers
        self.fixed_params: Optional[Tuple[str]] = fixed_params
        self.transforms: Optional[RayTransform] = transforms
        self.param_func: Callable = param_func
        self.max_target_deviation = max_target_deviation
        self.overwrite_offset: Optional = overwrite_offset_func() if overwrite_offset_func is not None else None
        self.max_offset_search_deviation = max_offset_search_deviation
        os.environ["WANDB__SERVICE_WAIT"] = "300"
        wandb.init(entity=self.wandb_entity,
                   project=self.wandb_project,
                   name=self.study_name,
                   mode='online' if self.logging else 'disabled',
                   )

        # optimize only some all_params
        self.all_params = param_func()
        for key in self.all_params:
            old_param = self.all_params[key]
            if isinstance(old_param, MutableParameter) and key in self.fixed_params:
                self.all_params[key] = NumericalParameter((old_param.value_lims[1] + old_param.value_lims[0]) / 2)

        self.ray_optimizer = RayOptimizer(optimizer_backend=self.optimizer_backend, criterion=self.criterion,
                                          engine=self.engine,
                                          log_times=True, exported_plane=self.exported_plane,
                                          transforms=self.transforms,
                                          logging_backend=WandbLoggingBackend(), plot_interval=self.plot_interval,
                                          iterations=self.iterations)

        if self.real_data_configuration is None:
            target_offset = offset_search_space(self.all_params, self.max_target_deviation, self.rg, None)
            uncompensated_parameters = [self.param_func() for _ in range(self.num_beamline_samples)]
            compensated_parameters: list[RayParameterContainer[str, RayParameter]] = [v.clone() for v in
                                                                                      uncompensated_parameters]
            for configuration in compensated_parameters:
                configuration.perturb(target_offset)
            compensated_transforms = RayOptimizer.translate_exported_plain_transforms(self.exported_plane,
                                                                                      compensated_parameters,
                                                                                      self.transforms)
            observed_rays = self.engine.run(compensated_parameters, transforms=compensated_transforms)
            uncompensated_validation_parameters = None
            observed_validation_rays = None
        else:
            observed_rays = import_data(self.real_data_configuration.real_data_dir,
                                        self.real_data_configuration.real_data_train_set, self.z_layers,
                                        self.param_func(),
                                        check_value_lims=True)
            uncompensated_parameters = [element['param_container_dict'] for element in observed_rays]
            target_offset = None
            observed_validation_rays = import_data(self.real_data_configuration.real_data_dir,
                                                   self.real_data_configuration.real_data_validation_set, self.z_layers,
                                                   self.param_func(), check_value_lims=False)
            uncompensated_validation_parameters = [element['param_container_dict'] for element in
                                                   observed_validation_rays]

        initial_transforms = RayOptimizer.translate_exported_plain_transforms(self.exported_plane,
                                                                              uncompensated_parameters,
                                                                              self.transforms)
        uncompensated_rays = self.engine.run(uncompensated_parameters, transforms=initial_transforms)

        if self.real_data_configuration is not None:
            validation_parameters_rays = self.engine.run(uncompensated_validation_parameters,
                                                         transforms=initial_transforms)
            validation_scan = RayScan(uncompensated_parameters=uncompensated_validation_parameters,
                                      uncompensated_rays=validation_parameters_rays,
                                      observed_rays=observed_validation_rays)
        else:
            validation_scan = None

        offset_optimization_target = OffsetOptimizationTarget(observed_rays=observed_rays,
                                                              offset_search_space=offset_search_space(self.all_params,
                                                                                                      self.max_offset_search_deviation,
                                                                                                      self.rg,
                                                                                                      self.overwrite_offset),
                                                              uncompensated_parameters=uncompensated_parameters,
                                                              uncompensated_rays=uncompensated_rays,
                                                              target_offset=target_offset,
                                                              validation_scan=validation_scan)

        self.ray_optimizer.optimize(optimization_target=offset_optimization_target)


# if len(sys.argv) > 1:
#    big_parameter = int(sys.argv[1])
#    key, value = list(all_params.items())[big_parameter]
#    value_lim_center = (value.value_lims[1] + value.value_lims[0]) / 2
#    old_interval_half = (value.value_lims[1] - value.value_lims[0]) / 2
#    old_interval_half *= 3
#    all_params[key] = type(value)(
#        value_lims=(value_lim_center - old_interval_half, value_lim_center + old_interval_half), rg=CFG.RG)
#    CFG.STUDY_NAME += '_' + key


# Bayesian Optimization
# ax_client = AxClient(early_stopping_strategy=None, verbose_logging=verbose)

# optimizer_backend_ax = OptimizerBackendAx(ax_client, search_space=all_params)

def offset_search_space(input_parameter_container: RayParameterContainer, max_deviation: float, rg: RandomGenerator,
                        overwrite_offset: Optional[RayParameterContainer] = None):
    ray_parameters = []
    for k, v in input_parameter_container.items():
        if not isinstance(v, RandomParameter):
            continue  # Numerical parameters do not need offset search
        if overwrite_offset is not None and k in overwrite_offset:
            ray_parameter = (k, overwrite_offset[k].clone())
        else:
            new_min = -max_deviation * (v.value_lims[1] - v.value_lims[0])
            if v.enforce_lims and new_min < v.value_lims[0]:
                new_min = v.value_lims[0]
            new_max = max_deviation * (v.value_lims[1] - v.value_lims[0])
            if v.enforce_lims and new_max > v.value_lims[1]:
                new_max = v.value_lims[1]
            ray_parameter = (k, type(v)(value_lims=(new_min, new_max), rg=rg))
        ray_parameters.append(ray_parameter)
    return RayParameterContainer(ray_parameters)


def params_to_func(parameters, rg: Optional[RandomGenerator] = None, enforce_lims_keys: List[str] = ()):
    elements = []
    for k, v in parameters.items():
        if hasattr(v, '__getitem__'):
            elements.append((k, RandomParameter(value_lims=(v[0], v[1]), rg=rg, enforce_lims=k in enforce_lims_keys)))
        else:
            elements.append((k, NumericalParameter(value=v)))

    def output_func():
        return RayParameterContainer(elements)

    return output_func


