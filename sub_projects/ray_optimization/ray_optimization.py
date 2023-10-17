import os
from typing import Optional, List

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from sub_projects.ray_optimization.configuration import RealDataConfiguration, TargetConfiguration

from ray_tools.base.utils import RandomGenerator

from sub_projects.ray_optimization.real_data import import_data
from ray_optim.ray_optimizer import LoggingBackend, RayOptimizer, OffsetTarget, RayScan

from ray_tools.base.parameter import RayParameterContainer, RayParameter, RandomParameter


class RayOptimization:
    def __init__(self, target_configuration: TargetConfiguration, ray_optimizer: RayOptimizer,
                 rg: RandomGenerator, logging_backend: LoggingBackend,
                 logging: bool = True):
        self.target_configuration = target_configuration
        self.rg: RandomGenerator = rg
        self.ray_optimizer = ray_optimizer
        self.real_data_configuration: RealDataConfiguration = target_configuration.real_data_configuration
        self.logging_backend: LoggingBackend = logging_backend
        self.logging: bool = logging
        self.z_layers: List[float] = target_configuration.z_layers
        self.all_params = self.target_configuration.param_func()
        if self.real_data_configuration is None:
            target_offset = offset_search_space(self.all_params,
                                                self.target_configuration.max_target_deviation, self.rg,
                                                None)
            uncompensated_parameters = [self.target_configuration.param_func() for _ in
                                        range(self.target_configuration.num_beamline_samples)]
            compensated_parameters: list[RayParameterContainer[str, RayParameter]] = [v.clone() for v in
                                                                                      uncompensated_parameters]
            for configuration in compensated_parameters:
                configuration.perturb(target_offset)
            compensated_transforms = RayOptimizer.translate_exported_plain_transforms(
                self.target_configuration.exported_plane,
                compensated_parameters,
                self.target_configuration.transforms)
            observed_rays = self.target_configuration.engine.run(compensated_parameters,
                                                                              transforms=compensated_transforms)
            uncompensated_validation_parameters = None
            observed_validation_rays = None
        else:
            observed_rays = import_data(self.real_data_configuration.path,
                                        self.real_data_configuration.train_set, self.z_layers,
                                        self.target_configuration.param_func(),
                                        check_value_lims=True)
            uncompensated_parameters = [element['param_container_dict'] for element in observed_rays]
            target_offset = None
            observed_validation_rays = import_data(self.real_data_configuration.path,
                                                   self.real_data_configuration.validation_set, self.z_layers,
                                                   self.target_configuration.param_func(),
                                                   check_value_lims=False)
            uncompensated_validation_parameters = [element['param_container_dict'] for element in
                                                   observed_validation_rays]

        initial_transforms = RayOptimizer.translate_exported_plain_transforms(
            self.target_configuration.exported_plane,
            uncompensated_parameters,
            self.target_configuration.transforms)
        uncompensated_rays = self.target_configuration.engine.run(uncompensated_parameters,
                                                                               transforms=initial_transforms)

        if self.real_data_configuration is not None:
            validation_parameters_rays = self.target_configuration.engine.run(
                uncompensated_validation_parameters,
                transforms=initial_transforms)
            validation_scan = RayScan(uncompensated_parameters=uncompensated_validation_parameters,
                                      uncompensated_rays=validation_parameters_rays,
                                      observed_rays=observed_validation_rays)
        else:
            validation_scan = None

        offset_target = OffsetTarget(observed_rays=observed_rays,
                                                              offset_search_space=offset_search_space(self.all_params,
                                                                                                      self.target_configuration.max_offset_search_deviation,
                                                                                                      self.rg),
                                                              uncompensated_parameters=uncompensated_parameters,
                                                              uncompensated_rays=uncompensated_rays,
                                                              target_compensation=target_offset,
                                                              validation_scan=validation_scan)

        self.ray_optimizer.optimize(target=offset_target)


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



os.environ["HYDRA_FULL_ERROR"] = "1"


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def optimization(cfg):
    #wandb.config = OmegaConf.to_container(
    #    cfg, resolve=True, throw_on_missing=True
    #)
    print(OmegaConf.to_yaml(cfg))
    _ = instantiate(cfg)


if __name__ == "__main__":
    optimization()
