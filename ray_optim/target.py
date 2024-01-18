import copy
import torch
from ray_tools.base.parameter import MutableParameter, NumericalParameter, RayParameterContainer
from sub_projects.ray_optimization.utils import ray_output_to_tensor


class RayScan:
    def __init__(
        self,
        uncompensated_parameters: list[RayParameterContainer],
        uncompensated_rays: list[dict],
        observed_rays: list[dict],
    ):
        self.uncompensated_parameters: list[
            RayParameterContainer
        ] = uncompensated_parameters
        self.uncompensated_rays = uncompensated_rays
        self.observed_rays = observed_rays
        self.observed_rays_cpu_tensor = None
        self.uncompensated_rays_cpu_tensor = None
        self.dicts_have_changed: bool = True

    @property
    def uncompensated_rays(self) -> list[dict]:
        return self._uncompensated_rays

    @uncompensated_rays.setter
    def uncompensated_rays(self, value: list[dict]):
        self.dicts_have_changed = True
        self._uncompensated_rays = value

    @property
    def observed_rays(self) -> list[dict]:
        return self._observed_rays

    @observed_rays.setter
    def observed_rays(self, value: list[dict]):
        self.dicts_have_changed = True
        self._observed_rays = value

    @property
    def uncompensated_rays_cpu_tensor(self) -> list[torch.Tensor]:
        if self.dicts_have_changed or self._uncompensated_rays_cpu_tensor is None:
            raise Exception("Please run recalculate cpu tensors first.")
        return self._uncompensated_rays_cpu_tensor

    @uncompensated_rays_cpu_tensor.setter
    def uncompensated_rays_cpu_tensor(self, value: list[torch.Tensor] | None):
        self._uncompensated_rays_cpu_tensor = value

    @property
    def observed_rays_cpu_tensor(self) -> list[torch.Tensor]:
        if self.dicts_have_changed or self._observed_rays_cpu_tensor is None:
            raise Exception("Please run recalculate cpu tensors first.")
        return self._observed_rays_cpu_tensor

    @observed_rays_cpu_tensor.setter
    def observed_rays_cpu_tensor(self, value: list[torch.Tensor] | None):
        self._observed_rays_cpu_tensor = value


    def recalculate_cpu_tensors(self, exported_plane: str):
        self.dicts_have_changed = False
        self.observed_rays_cpu_tensor = ray_output_to_tensor(
            self.observed_rays, exported_plane, to_cpu=True
        )
        self.uncompensated_rays_cpu_tensor = ray_output_to_tensor(
            self.uncompensated_rays, exported_plane, to_cpu=True
        )


class Target:
    def __init__(
        self,
        observed_rays: list[dict],
        search_space: RayParameterContainer,
        target_params: RayParameterContainer | None = None,
    ):
        self.observed_rays = observed_rays
        self.search_space = search_space
        self.target_params: RayParameterContainer | None = target_params
        self.dicts_have_changed: bool = True
        self.unscaled_search_space: RayParameterContainer = search_space.clone()
        self.is_normalized: bool = False
    
    @staticmethod
    def _normalize_parameters(input_parameter_container: RayParameterContainer):
        normalized = input_parameter_container.clone()
        for v in normalized.values():
            if isinstance(v, MutableParameter):
                v.value = (v.value - v.value_lims[0]) / (
                    v.value_lims[1] - v.value_lims[0]
                )
                v.value_lims = (0.0, 1.0)
        return normalized
    
    def denormalize_parameter_container(self, input: RayParameterContainer) -> RayParameterContainer:
        input_copy = input.clone()
        for k, v in input_copy.items():
            original_value = self.unscaled_search_space[k]
            if not isinstance(original_value, MutableParameter) or not isinstance(v, NumericalParameter):
                continue
            if isinstance(v, MutableParameter):
                v.value_lims = original_value.value_lims
            v.value = v.value * (original_value.value_lims[1] - original_value.value_lims[0]) + original_value.value_lims[0]
        return input_copy
    
    def normalize_dict(self, input: dict[str, float]) -> dict[str, float]:
        input_copy: dict[str, float] = copy.deepcopy(input)
        for k, v in input_copy.items():
            original_value = self.unscaled_search_space[k]
            if isinstance(original_value, MutableParameter):
                input_copy[k] = (v - original_value.value_lims[0]) / (original_value.value_lims[1] - original_value.value_lims[0])
        return input_copy
             
    def denormalize_dict(self, input: dict[str, float]) -> dict[str, float]:
        input_copy: dict[str, float] = copy.deepcopy(input)
        for k, v in input_copy.items():
            original_value = self.unscaled_search_space[k]
            if isinstance(original_value, MutableParameter):
                input_copy[k] = v * (original_value.value_lims[1] - original_value.value_lims[0]) + original_value.value_lims[0]
        return input_copy
    
    def normalize(self):
        self.search_space = Target._normalize_parameters(self.search_space)
        self.is_normalized = True

    def recalculate_cpu_tensors(self, exported_plane: str):
        self.dicts_have_changed = False
        self.observed_rays_cpu_tensor = ray_output_to_tensor(
            self.observed_rays, exported_plane, to_cpu=True
        )
    @property
    def search_space(self) -> RayParameterContainer:
        return self._search_space
    
    @search_space.setter
    def search_space(self, value: RayParameterContainer):
        self.is_normalized = False
        self._search_space = value
    @property
    def observed_rays(self) -> list[dict]:
        return self._observed_rays

    @observed_rays.setter
    def observed_rays(self, value: list[dict]):
        self.dicts_have_changed = True
        self._observed_rays = value

    @property
    def observed_rays_cpu_tensor(self) -> list[torch.Tensor]:
        if self.dicts_have_changed or self._observed_rays_cpu_tensor is None:
            raise Exception("Please run recalculate cpu tensors first.")
        return self._observed_rays_cpu_tensor

    @observed_rays_cpu_tensor.setter
    def observed_rays_cpu_tensor(self, value: list[torch.Tensor] | None):
        self._observed_rays_cpu_tensor = value    
    
class OffsetTarget(Target):
    def __init__(
        self,
        training_scan: RayScan,
        offset_search_space: RayParameterContainer,
        target_compensation: RayParameterContainer | None = None,
        validation_scan: RayScan | None = None,
    ):
        super().__init__(
            training_scan.observed_rays, offset_search_space, target_compensation
        )
        self.uncompensated_parameters: list[
            RayParameterContainer
        ] = training_scan.uncompensated_parameters
        self.uncompensated_rays = training_scan.uncompensated_rays
        self.validation_scan: RayScan | None = validation_scan

    @property
    def uncompensated_rays(self) -> list[dict]:
        return self._uncompensated_rays

    @uncompensated_rays.setter
    def uncompensated_rays(self, value: list[dict]):
        self.dicts_have_changed = True
        self._uncompensated_rays = value
    
    @property
    def uncompensated_rays_cpu_tensor(self) -> list[torch.Tensor]:
        if self.dicts_have_changed or self._uncompensated_rays_cpu_tensor is None:
            raise Exception("Please run recalculate cpu tensors first.")
        return self._uncompensated_rays_cpu_tensor

    @uncompensated_rays_cpu_tensor.setter
    def uncompensated_rays_cpu_tensor(self, value: list[torch.Tensor] | None):
        self._uncompensated_rays_cpu_tensor = value

    def recalculate_cpu_tensors(self, exported_plane: str):
        super().recalculate_cpu_tensors(exported_plane=exported_plane)

        if self.validation_scan is not None:
            self.validation_scan.recalculate_cpu_tensors(exported_plane=exported_plane)
 
        self.uncompensated_rays_cpu_tensor = ray_output_to_tensor(
            self.uncompensated_rays, exported_plane, to_cpu=True
        )
       