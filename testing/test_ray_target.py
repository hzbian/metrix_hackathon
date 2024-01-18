import unittest
from unittest.mock import Mock

from ray_optim.target import Target
from ray_tools.base.parameter import (
    MutableParameter,
)
from ray_tools.base.utils import RandomGenerator
from sub_projects.ray_optimization.configuration import params_to_func


class TargetTest(unittest.TestCase):
    def setUp(self):
        parameters = {
            "number_rays": 1e2,
            "x_mean": [-1.0, 1.0],
        }
        self.param_func = params_to_func(parameters, rg = RandomGenerator(42))
        self.target = Target(observed_rays=Mock(), search_space=self.param_func())
        self.target2 = Target(Mock(), self.param_func())

    def test_normalize(self):
        normalized = self.target.normalize_parameters(self.target.search_space)
        for k, v in self.target.search_space.items():
            if isinstance(v, MutableParameter):
                self.assertAlmostEqual(
                    normalized[k].get_value(),
                    ((v.value - v.value_lims[0]) / (v.value_lims[1] - v.value_lims[0])),
                    places=8
                )

    def test_denormalize(self):
        normalized = self.target.normalize_parameters(self.target.search_space)
        original = self.target.search_space
        self.assertAlmostEqual(self.target.search_space['x_mean'].get_value(), 0.278853596, places=4)
        self.target.normalize()
        self.assertAlmostEqual(self.target.search_space['x_mean'].get_value(), 0.639426798, places=4)
        denormalized = self.target.denormalize_parameter_container(normalized)
        self.assertAlmostEqual(denormalized['x_mean'].get_value(), 0.278853596, places=4)
        for k, v in original.items():
            self.assertAlmostEqual(v.get_value(), denormalized[k].get_value(), places=8)
            original_value = original[k]
            denormalized_value = denormalized[k]
            if isinstance(original_value, MutableParameter) and isinstance(denormalized_value, MutableParameter):
                self.assertEqual(original_value.value_lims, denormalized_value.value_lims)

if __name__ == "__main__":
    unittest.main()
