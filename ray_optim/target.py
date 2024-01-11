from ray_optim.ray_optimizer import RayScan
from ray_tools.base.parameter import RayParameterContainer


class Target:
    def __init__(
        self,
        observed_rays: list[dict],
        search_space: RayParameterContainer,
        target_params: RayParameterContainer | None = None,
    ):
        self.observed_rays = observed_rays
        self.search_space = search_space
        self.target_params = target_params


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
        self.uncompensated_rays: list[dict] = training_scan.uncompensated_rays
        self.validation_scan: RayScan | None = validation_scan

