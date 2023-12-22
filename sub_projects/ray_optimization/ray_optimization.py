import os
from typing import Dict, Optional, List

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from sub_projects.ray_optimization.configuration import (
    RealDataConfiguration,
    TargetConfiguration,
)

from ray_tools.base.utils import RandomGenerator

from sub_projects.ray_optimization.real_data import import_data
from ray_optim.ray_optimizer import LoggingBackend, RayOptimizer, OffsetTarget, RayScan

from ray_tools.base.parameter import (
    RayParameterContainer,
    RayParameter,
    RandomParameter,
)
from sub_projects.ray_optimization.utils import ray_output_to_tensor


class RayOptimization:
    def __init__(
        self,
        target_configuration: TargetConfiguration,
        ray_optimizer: RayOptimizer,
        rg: RandomGenerator,
        logging_backend: LoggingBackend,
        logging: bool = True,
    ):
        self.target_configuration = target_configuration
        self.rg: RandomGenerator = rg
        self.ray_optimizer = ray_optimizer
        self.real_data_configuration: RealDataConfiguration | None = (
            target_configuration.real_data_configuration
        )
        self.logging_backend: LoggingBackend = logging_backend
        self.logging: bool = logging
        self.z_layers: list[float] = target_configuration.z_layers
        self.ray_parameter_container: RayParameterContainer = (
            self.target_configuration.param_func()
        )

    @staticmethod
    def limited_search_space(
        input_parameter_container: RayParameterContainer,
        rg: RandomGenerator,
        max_deviation: float = 1.0,
        overwrite_offset: Optional[RayParameterContainer] = None,
        random_parameters_only: bool = True,
    ):
        ray_parameters = []
        for k, v in input_parameter_container.items():
            if not isinstance(v, RandomParameter):
                if random_parameters_only:
                    continue  # Numerical parameters do not need offset search
                else:
                    ray_parameter = (k, v.clone())
            else:
                if overwrite_offset is not None and k in overwrite_offset:
                    ray_parameter = (k, overwrite_offset[k].clone())
                else:
                    new_min = -max_deviation * (v.value_lims[1] - v.value_lims[0]) / 2
                    if v.enforce_lims and new_min < v.value_lims[0]:
                        new_min = v.value_lims[0]
                    new_max = max_deviation * (v.value_lims[1] - v.value_lims[0]) / 2
                    if v.enforce_lims and new_max > v.value_lims[1]:
                        new_max = v.value_lims[1]
                    ray_parameter = (k, type(v)(value_lims=(new_min, new_max), enforce_lims=v.enforce_lims, rg=rg))
            ray_parameters.append(ray_parameter)
        return RayParameterContainer(ray_parameters)
    
    @staticmethod
    def get_mean_parameter_container(
        input_parameter_container: RayParameterContainer,
    ):
        ray_parameters = {}
        for k, v in input_parameter_container.items():
            ray_parameters[k] = v.clone()
            if isinstance(v, RandomParameter):
                ray_parameters[k].value = (v.value_lims[1] + v.value_lims[0]) / 2
        return RayParameterContainer(ray_parameters)


    def create_target_compensation(self):
        target_compensation = self.limited_search_space(
            self.ray_parameter_container,
            self.rg,
            self.target_configuration.max_target_deviation,
            None,
        )
        return target_compensation

    def create_uncompensated_parameters(self):
        sample_parameter_func = lambda: self.limited_search_space(
            self.target_configuration.param_func(),
            self.rg,
            max_deviation=self.target_configuration.max_sample_generation_deviation,
            random_parameters_only=False,
        )
        uncompensated_parameters = [
            sample_parameter_func()
            for _ in range(self.target_configuration.num_beamline_samples)
        ]
        mean_parameter_container = RayOptimization.get_mean_parameter_container(self.target_configuration.param_func())
        for configuration in uncompensated_parameters:
            configuration.perturb(mean_parameter_container)
    
        return uncompensated_parameters

    def create_compensated_parameters(self, uncompensated_parameters, target_compensation):
        compensated_parameters = [v.clone() for v in uncompensated_parameters]
        for configuration in compensated_parameters:
            configuration.perturb(target_compensation)
        return compensated_parameters

    def create_compensated_transforms(self, compensated_parameters):
        compensated_transforms = RayOptimizer.translate_exported_plain_transforms(
            self.target_configuration.exported_plane,
            compensated_parameters,
            self.target_configuration.transforms,
        )
        return compensated_transforms

    def create_observed_rays(self, uncompensated_parameters, target_compensation):
        compensated_parameters: list[
            RayParameterContainer
        ] = self.create_compensated_parameters(
            uncompensated_parameters, target_compensation
        )
        compensated_transforms = self.create_compensated_transforms(
            compensated_parameters
        )
        observed_rays = self.target_configuration.engine.run(
            compensated_parameters, transforms=compensated_transforms
        )
        return observed_rays

    def create_offset_search_space(self):
        return self.limited_search_space(
            self.ray_parameter_container,
            self.rg,
            self.target_configuration.max_offset_search_deviation,
        )

    def create_initial_transforms(self, uncompensated_parameters):
        initial_transforms = RayOptimizer.translate_exported_plain_transforms(
            self.target_configuration.exported_plane,
            uncompensated_parameters,
            self.target_configuration.transforms,
        )
        return initial_transforms

    def create_uncompensated_rays(self, uncompensated_parameters):
        uncompensated_rays = self.target_configuration.engine.run(
            uncompensated_parameters,
            transforms=self.create_initial_transforms(uncompensated_parameters),
        )
        return uncompensated_rays

    def create_simulated_target(self):
        target_compensation = self.create_target_compensation()
        uncompensated_parameters = self.create_uncompensated_parameters()
        training_scan = RayScan(
            uncompensated_rays=self.create_uncompensated_rays(uncompensated_parameters),
            uncompensated_parameters=uncompensated_parameters,
            observed_rays=self.create_observed_rays(
                uncompensated_parameters, target_compensation
            ),
        )
        return OffsetTarget(
            training_scan=training_scan,
            offset_search_space=self.create_offset_search_space(),
            target_compensation=target_compensation,
        )

    def import_set(self, validation_set: bool = False):
        if self.real_data_configuration is None:
            raise Exception("Real data configuration is not set but trying to import.")
        
        import_set: list[str] | None = (
            self.real_data_configuration.train_set
            if not validation_set
            else self.real_data_configuration.validation_set
        )
        return import_data(
            self.real_data_configuration.path,
            import_set,
            self.z_layers,
            self.target_configuration.param_func(),
            check_value_lims=not validation_set,
        )

    @staticmethod
    def prune_param_container(container_list):
        return [element["param_container_dict"] for element in container_list]

    def create_real_target(self):
        observed_rays = self.import_set(validation_set=False)
        uncompensated_parameters = RayOptimization.prune_param_container(observed_rays)

        training_scan = RayScan(
            uncompensated_parameters=uncompensated_parameters,
            uncompensated_rays=self.create_uncompensated_rays(uncompensated_parameters),
            observed_rays=observed_rays,
        )

        observed_validation_rays = self.import_set(validation_set=True)
        uncompensated_validation_parameters = RayOptimization.prune_param_container(
            observed_validation_rays
        )

        validation_parameters_rays = self.target_configuration.engine.run(
            uncompensated_validation_parameters,
            transforms=self.create_initial_transforms(uncompensated_parameters),
        )
        validation_scan = RayScan(
            uncompensated_parameters=uncompensated_validation_parameters,
            uncompensated_rays=validation_parameters_rays,
            observed_rays=observed_validation_rays,
        )

        return OffsetTarget(
            training_scan=training_scan,
            offset_search_space=self.create_offset_search_space(),
            validation_scan=validation_scan,
        )

    def setup_target(self):
        if self.real_data_configuration is None:
            self.target = self.create_simulated_target()
        else:
            self.target = self.create_real_target()

    def optimize(self):
        try:
            if self.is_output_ray_list_empty(self.target.observed_rays):
                raise Exception("Refusing to optimize an empty target.")
            self.ray_optimizer.optimize(target=self.target, starting_point=self.ray_optimizer.starting_point)
        except NameError:
            raise Exception("You need to run setup_target() first.")

    def is_output_ray_list_empty(self, input: List[Dict]) -> bool:
        input = ray_output_to_tensor(
            input, exported_plane=self.target_configuration.exported_plane
        )
        return not True in [len(entry.reshape(-1)) != 0 for entry in input]


os.environ["HYDRA_FULL_ERROR"] = "1"


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def optimization(cfg):
    print(OmegaConf.to_yaml(cfg))
    ray_optimization: RayOptimization = instantiate(cfg)
    ray_optimization.logging_backend.log_config(
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    )
    ray_optimization.setup_target()
    ray_optimization.optimize()


if __name__ == "__main__":
    optimization()
