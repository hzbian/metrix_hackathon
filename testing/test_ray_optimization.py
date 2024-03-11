from typing import Callable
import unittest
from unittest.mock import Mock

import omegaconf
from hydra.utils import instantiate
from tqdm import trange
from ray_optim.logging import DebugPlotBackend
from ray_optim.optimizer_backend.base import OptimizerBackend
from ray_optim.ray_optimizer import OffsetTarget, RayOptimizer
from ray_optim.target import Target
from ray_tools.base.engine import GaussEngine
from hydra import initialize, compose
from ray_tools.base.parameter import (
    MutableParameter,
    NumericalParameter,
    RandomParameter,
    RayParameterContainer,
)
from ray_tools.base.transform import MultiLayer

from ray_tools.base.utils import RandomGenerator
from sub_projects.ray_optimization.configuration import (
    TargetConfiguration,
    params_to_func,
)
from sub_projects.ray_optimization.ray_optimization import RayOptimization

class OptimizerBackendTest(OptimizerBackend):
    def optimize(self, objective: Callable[[list[RayParameterContainer], Target], list[float]], iterations: int, target: Target, starting_point: dict[str, float] | None = None) -> tuple[dict[str, float], dict[str, float]]:
        optimize_parameters: RayParameterContainer = target.search_space.clone()
        if starting_point is not None:
            for key in starting_point.keys():
                current_parameter = optimize_parameters[key]
                if isinstance(current_parameter, NumericalParameter):
                    current_parameter.value = starting_point[key]
        #mutable_parameters_keys = [key for key, value in optimize_parameters.items() if isinstance(value, MutableParameter)] 
        current_parameters = optimize_parameters
        for _ in trange(iterations):
            distance = objective([current_parameters], target)
        return current_parameters.to_value_dict(), {"loss": distance[0]}



class TestRayOptimization(unittest.TestCase):
    def setUp(self):
        self.parameters = {
            "number_rays": 1e2,
            "x_var": [1, 2],
            "y_var": [1, 2],
            "x_mean": [-1, 1],
            "y_mean": [-3, -2],
            "x_dir": 0.0,
            "y_dir": 0.0,
            "z_dir": 1.0,
            "direction_spread": 0.0,
        }
        self.rg = RandomGenerator(42)
        self.param_func = params_to_func(self.parameters, rg=self.rg)
        self.engine = GaussEngine()
        self.transforms = MultiLayer([0.0, 1.0, 2.0, 3.0])
        self.exported_plane: str = "ImagePlane"
        self.max_offset_search_deviation: float = 0.3
        self.max_target_deviation: float = 0.3
        mock = Mock()
        mock.__class__ = RayOptimizer
        target_configuration = TargetConfiguration(
            engine=self.engine,
            exported_plane="ImagePlane",
            logging_project="test",
            max_offset_search_deviation=self.max_offset_search_deviation,
            max_target_deviation=self.max_target_deviation,
            param_func=self.param_func,
            num_beamline_samples=5,
            transforms=self.transforms,
        )
        self.ray_optimization = RayOptimization(
            ray_optimizer=mock,
            target_configuration=target_configuration,
            rg=self.rg,
            logging_backend=DebugPlotBackend(),
        )

    def test_create_target_compensation(self):
        target_compensation = self.ray_optimization.create_target_compensation()
        for key, value in target_compensation.items():
            assert isinstance(value, MutableParameter)
            interval_size = (
                (self.parameters[key][1] - self.parameters[key][0])
                * self.max_target_deviation
                / 2
            )
            # check if all chosen values are within the intervals
            self.assertTrue(value.get_value() >= -interval_size)
            self.assertTrue(value.get_value() <= interval_size)
            # check if intervals match
            self.assertTrue(value.value_lims[0] == -interval_size)
            self.assertTrue(value.value_lims[1] == interval_size)
            # this is the desired behavior
            if key == "x_mean":
                self.assertTrue(value.value_lims[0] == -self.max_target_deviation)
                self.assertTrue(value.value_lims[1] == self.max_target_deviation)

    def test_create_uncompensated_parameters(self):
        uncompensated_parameters = (
            self.ray_optimization.create_uncompensated_parameters()
        )
        self.assertEqual(
            len(uncompensated_parameters),
            self.ray_optimization.target_configuration.num_beamline_samples,
        )
        for sample in uncompensated_parameters:
            for key, value in sample.items():
                if not isinstance(value, MutableParameter):
                    self.assertEqual(value.get_value(), self.parameters[key])
                else:
                    self.assertEqual(value.value_lims, tuple(self.parameters[key]))

    def test_create_observed_rays(self):
        target_compensation = self.ray_optimization.create_target_compensation()
        uncompensated_parameters = (
            self.ray_optimization.create_uncompensated_parameters()
        )
        observed_rays = self.ray_optimization.create_observed_rays(
            uncompensated_parameters, target_compensation
        )
        # TODO checks required

    def test_limited_search_space(self):
        max_deviation = 0.1
        test_output = RayOptimization.limited_search_space(
            self.param_func(),
            rg=self.rg,
            max_deviation=0.1,
            random_parameters_only=False,
        )
        for k, v in self.param_func().items():
            current_output = test_output[k]
            if isinstance(current_output, RandomParameter):
                # if current output is random, the according value in param_func should be also
                assert isinstance(v, RandomParameter)
                if not current_output.enforce_lims:
                    self.assertTrue(
                        current_output.get_value()
                        >= -max_deviation * (v.value_lims[1] - v.value_lims[0]) / 2
                    )
                    self.assertTrue(
                        current_output.get_value()
                        <= max_deviation * (v.value_lims[1] - v.value_lims[0]) / 2
                    )
            else:
                self.assertTrue(k in test_output.keys())

    def test_setup(self):
        self.ray_optimization.setup_target()
        self.assertTrue(isinstance(self.ray_optimization.target, OffsetTarget))

    def test_with_initialize(self) -> None:
        with initialize(version_base=None, config_path="../sub_projects/ray_optimization/conf"):
            # config is relative to a module
            cfg = compose(config_name="config", overrides=["ray_optimizer=test"])
            print(omegaconf.OmegaConf.to_yaml(cfg))
            ray_optimization = instantiate(cfg)
            ray_optimization.setup_target()
