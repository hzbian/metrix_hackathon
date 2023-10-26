from typing import OrderedDict
import unittest
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


class TestRayOptimizer(unittest.TestCase):
    def setUp(self):
        self.parameters = {
            "number_rays": 1e2,
            "x_var": [0.001, 0.01],
            "y_var": [0.001, 0.01],
        }

    def test_params_to_func(self):
        rg = RandomGenerator()
        self.param_func = params_to_func(self.parameters, rg=rg)
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
