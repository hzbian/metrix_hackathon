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
from sub_projects.ray_optimization.configuration import TargetConfiguration, params_to_func
from sub_projects.ray_optimization.ray_optimization import RayOptimization


class TestRayOptimization(unittest.TestCase):
    def setUp(self):
        self.parameters = {
            "number_rays": 1e2,
            "x_var": [1, 2],
            "y_var": [1, 2],
            "x_mean": [0, 1],
            "y_mean": [0, 1],
            "x_dir": 0.,
            "y_dir": 0.,
            "z_dir": 1.,
            "direction_spread": 0.,
        }
        rg = RandomGenerator(42)
        self.param_func = params_to_func(self.parameters, rg=rg)
        self.engine = GaussEngine()
        self.transforms = MultiLayer([0., 1., 2., 3.])
        mock = Mock()
        mock.__class__ = RayOptimizer
        target_configuration = TargetConfiguration(engine=self.engine, exported_plane="ImagePlane", logging_project="test", max_offset_search_deviation=0.3, max_target_deviation=0.1, param_func=self.param_func, transforms=self.transforms )
        self.ray_optimization = RayOptimization(ray_optimizer=mock, target_configuration=target_configuration, rg=rg, logging_backend=DebugPlotBackend())

    def test_setup(self):
        self.ray_optimization.setup_target()
        self.assertTrue(isinstance(self.ray_optimization.target, OffsetTarget))
