from typing import OrderedDict
import unittest
from unittest.mock import Mock
from ray_optim.ray_optimizer import RayOptimizer
from ray_tools.base.engine import GaussEngine
from ray_tools.base.parameter import (
    MutableParameter,
    NumericalOutputParameter,
    NumericalParameter,
    RandomOutputParameter,
    RandomParameter,
    RayParameterContainer,
)

from ray_tools.base.utils import RandomGenerator
from sub_projects.ray_optimization.configuration import params_to_func
from sub_projects.ray_optimization.ray_optimization import RayOptimization


class TestRayOptimization(unittest.TestCase):
    def setUp(self):
        self.parameters = {
            "number_rays": 1e2,
            "x_var": [0.001, 0.01],
            "y_var": [0.001, 0.01],
        }
        rg = RandomGenerator(42)
        self.param_func = params_to_func(self.parameters, rg=rg)
        self.engine = GaussEngine()
        mock = Mock()
        mock.__class__ = RayOptimizer
        self.ray_optimization = RayOptimization(ray_optimizer=mock, )

    def test_target_configuration(self):
        self.evaluation_parameters = [self.param_func() for _ in range(5)]
        equal_list = [
            parameter["number_rays"].get_value() == self.parameters["number_rays"]
            for parameter in self.evaluation_parameters
        ]
        self.assertFalse(False in equal_list)
        x_vars = [
            parameter["x_var"].get_value() for parameter in self.evaluation_parameters
        ]
        self.assertTrue(len(set(x_vars)) > 1)
