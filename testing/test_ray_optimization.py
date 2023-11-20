from typing import OrderedDict
import unittest
from unittest.mock import Mock
from ray_optim.logging import DebugPlotBackend
from ray_optim.ray_optimizer import OffsetTarget, RayOptimizer
from ray_tools.base.engine import GaussEngine
from ray_tools.base.parameter import (
    MutableParameter,
    NumericalOutputParameter,
    NumericalParameter,
    RandomOutputParameter,
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
            interval_size = (
                (self.parameters[key][1] - self.parameters[key][0])
                * self.max_target_deviation
                / 2
            )
            # check if all chosen values are within the intervals
            self.assertTrue(value.value >= -interval_size)
            self.assertTrue(value.value <= interval_size)
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
        self.assertTrue(
            len(uncompensated_parameters)
            == self.ray_optimization.target_configuration.num_beamline_samples
        )
        for sample in uncompensated_parameters:
            for key, value in sample.items():
                if not isinstance(value, MutableParameter):
                    self.assertTrue(value.value == self.parameters[key])
                else:
                    self.assertTrue(value.value_lims == tuple(self.parameters[key]))
    
    def test_create_observed_rays(self):
        target_compensation = self.ray_optimization.create_target_compensation()
        uncompensated_parameters = self.ray_optimization.create_uncompensated_parameters()
        observed_rays = self.ray_optimization.create_observed_rays(uncompensated_parameters, target_compensation)
        # TODO checks required
    
    def test_limited_search_space(self):
        max_deviation = 0.1
        test_output = RayOptimization.limited_search_space(self.param_func(), rg=self.rg, max_deviation=0.1, random_parameters_only=False)
        for k, v in self.param_func().items():
            if isinstance(test_output[k], RandomParameter):
                if not test_output[k].enforce_lims:
                    self.assertTrue(test_output[k].value >= - max_deviation * (v.value_lims[1]- v.value_lims[0]) / 2)
                    self.assertTrue(test_output[k].value <= max_deviation * (v.value_lims[1]- v.value_lims[0]) / 2)
            else:
                self.assertTrue(k in test_output.keys())


    def test_setup(self):
        self.ray_optimization.setup_target()
        self.assertTrue(isinstance(self.ray_optimization.target, OffsetTarget))
