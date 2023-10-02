import os
import sys
import uuid
from typing import Optional, Callable, List, Tuple
from collections import OrderedDict

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

sys.path.insert(0, '../../')
from sub_projects.ray_optimization.losses.losses import RayLoss
from ray_tools.base import RayTransform
from ray_tools.base.utils import RandomGenerator

from ray_tools.base.engine import Engine
from sub_projects.ray_optimization.real_data import import_data
from ray_optim.ray_optimizer import LoggingBackend, RayOptimizer, OffsetOptimizationTarget, RayScan, OptimizerBackend

from ray_tools.base.parameter import NumericalOutputParameter, RayParameterContainer, NumericalParameter, MutableParameter, \
    RayParameter, RandomParameter, RandomOutputParameter


class RealDataConfiguration:
    def __init__(self, real_data_dir: str, real_data_train_set: List[str],
                 real_data_validation_set: Optional[List[str]] = None):
        self.real_data_dir: str = real_data_dir
        self.real_data_train_set: List[str] = real_data_train_set
        self.real_data_validation_set: Optional[List[str]] = real_data_validation_set


class OptimizationTargetConfiguration:
    def __init__(self, param_func: Callable, engine: Engine, exported_plane: str, num_beamline_samples: int = 20,
                 max_target_deviation: float = 0.3, max_offset_search_deviation: float = 0.3,
                 logging_project: Optional[str] = None,
                 z_layers: List[float] = (0.),
                 transforms: Optional[RayTransform] = None,
                 real_data_configuration: Optional[RealDataConfiguration] = None):
        self.max_offset_search_deviation: float = max_offset_search_deviation
        self.z_layers = z_layers
        self.transforms: RayTransform = transforms
        self.num_beamline_samples: int = num_beamline_samples
        self.exported_plane: str = exported_plane
        self.engine: Engine = engine
        self.max_target_deviation: float = max_target_deviation
        self.param_func: Callable = param_func
        self.real_data_configuration = real_data_configuration
        self.logging_project = logging_project


class RayOptimization:
    def __init__(self, optimization_target_configuration: OptimizationTargetConfiguration, ray_optimizer: RayOptimizer,
                 rg: RandomGenerator, logging_backend: LoggingBackend,
                 logging: bool = True):
        self.optimization_target_configuration = optimization_target_configuration
        self.rg: RandomGenerator = rg
        self.ray_optimizer = ray_optimizer
        self.real_data_configuration: RealDataConfiguration = optimization_target_configuration.real_data_configuration
        self.logging_backend: LoggingBackend = logging_backend
        self.logging: bool = logging
        self.z_layers: List[float] = optimization_target_configuration.z_layers
        self.all_params = self.optimization_target_configuration.param_func()
        if self.real_data_configuration is None:
            target_offset = offset_search_space(self.all_params,
                                                self.optimization_target_configuration.max_target_deviation, self.rg,
                                                None)
            uncompensated_parameters = [self.optimization_target_configuration.param_func() for _ in
                                        range(self.optimization_target_configuration.num_beamline_samples)]
            compensated_parameters: list[RayParameterContainer[str, RayParameter]] = [v.clone() for v in
                                                                                      uncompensated_parameters]
            for configuration in compensated_parameters:
                configuration.perturb(target_offset)
            compensated_transforms = RayOptimizer.translate_exported_plain_transforms(
                self.optimization_target_configuration.exported_plane,
                compensated_parameters,
                self.optimization_target_configuration.transforms)
            observed_rays = self.optimization_target_configuration.engine.run(compensated_parameters,
                                                                              transforms=compensated_transforms)
            uncompensated_validation_parameters = None
            observed_validation_rays = None
        else:
            observed_rays = import_data(self.real_data_configuration.real_data_dir,
                                        self.real_data_configuration.real_data_train_set, self.z_layers,
                                        self.optimization_target_configuration.param_func(),
                                        check_value_lims=True)
            uncompensated_parameters = [element['param_container_dict'] for element in observed_rays]
            target_offset = None
            observed_validation_rays = import_data(self.real_data_configuration.real_data_dir,
                                                   self.real_data_configuration.real_data_validation_set, self.z_layers,
                                                   self.optimization_target_configuration.param_func(),
                                                   check_value_lims=False)
            uncompensated_validation_parameters = [element['param_container_dict'] for element in
                                                   observed_validation_rays]

        initial_transforms = RayOptimizer.translate_exported_plain_transforms(
            self.optimization_target_configuration.exported_plane,
            uncompensated_parameters,
            self.optimization_target_configuration.transforms)
        uncompensated_rays = self.optimization_target_configuration.engine.run(uncompensated_parameters,
                                                                               transforms=initial_transforms)

        if self.real_data_configuration is not None:
            validation_parameters_rays = self.optimization_target_configuration.engine.run(
                uncompensated_validation_parameters,
                transforms=initial_transforms)
            validation_scan = RayScan(uncompensated_parameters=uncompensated_validation_parameters,
                                      uncompensated_rays=validation_parameters_rays,
                                      observed_rays=observed_validation_rays)
        else:
            validation_scan = None

        offset_optimization_target = OffsetOptimizationTarget(observed_rays=observed_rays,
                                                              offset_search_space=offset_search_space(self.all_params,
                                                                                                      self.optimization_target_configuration.max_offset_search_deviation,
                                                                                                      self.rg),
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


def params_to_func(parameters, rg: Optional[RandomGenerator] = None, enforce_lims_keys: List[str] = (),
                   output_parameters: List[str] = (), fixed_parameters: List[str] = ()):
    elements = []
    for k, v in parameters.items():
        if hasattr(v, '__getitem__'):
            if k in output_parameters:
                typ = RandomOutputParameter
            else:
                typ = RandomParameter

            elements.append((k, typ(value_lims=(v[0], v[1]), rg=rg, enforce_lims=k in enforce_lims_keys)))
        else:
            if k in output_parameters:
                typ = NumericalOutputParameter
            else:
                typ = NumericalParameter
           
            elements.append((k, typ(value=v)))
        
    elements = OrderedDict(elements)
    # do not optimize the fixed parameters, set them to the center of interval
    for key in fixed_parameters:
        old_param = elements[key]
        if isinstance(old_param, MutableParameter) and key in fixed_parameters:
            if key in output_parameters:
                typ = NumericalOutputParameter
            else:
                typ = NumericalParameter
  
            elements[key] = typ((old_param.value_lims[1] + old_param.value_lims[0]) / 2)


    def output_func():
        return RayParameterContainer(elements)

    return output_func


def build_study_name(param_func: Callable, max_target_deviation: float, max_offset_search_deviation: float,
                     loss: Optional[RayLoss] = None, optimizer_backend: Optional[OptimizerBackend] = None, appendix: Optional[str] = None) -> str:
    var_count: int = sum(isinstance(x, RandomParameter) for x in param_func().values())
    string_list = [str(var_count), 'target', str(max_target_deviation), 'search', str(max_offset_search_deviation)]

    if appendix is not None:
        string_list.append(appendix)
    if RayLoss is not None:
        string_list.append(loss.__class__.__name__.replace('Loss', ''))
    if OptimizerBackend is not None:
        string_list.append(optimizer_backend.__class__.__name__.replace('OptimizerBackend', ''))
    return '-'.join(string_list)


def build_ray_workdir_path(parent_path: str):
    return os.path.join(parent_path, str(uuid.uuid4()))


os.environ["HYDRA_FULL_ERROR"] = "1"


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def my_app(cfg):
    #wandb.config = OmegaConf.to_container(
    #    cfg, resolve=True, throw_on_missing=True
    #)
    print(OmegaConf.to_yaml(cfg))
    _ = instantiate(cfg)


if __name__ == "__main__":
    my_app()
