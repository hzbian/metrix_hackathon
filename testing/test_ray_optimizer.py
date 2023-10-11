from typing import List
import unittest

from ray_optim.ray_optimizer import RayOptimizer, Sample
from ray_tools.base.engine import GaussEngine
from ray_tools.base.transform import MultiLayer
from sub_projects.ray_optimization.configuration import params_to_func
from sub_projects.ray_optimization.losses.geometric import SinkhornLoss
from sub_projects.ray_optimization.losses.losses import RayLoss


class TestRayOptimizer(unittest.TestCase):
    def setUp(self):
        self.engine = GaussEngine()
        parameters = {
            "number_rays": 1e2,
            "x_dir": 0.,
            "y_dir": 0.,
             "z_dir": 1.,
             "direction_spread": 0.,
             "correlation_factor": [-0.8, 0.8],
            "x_mean": [-2., 2.],
            "y_mean": [-2., 2.],
            "x_var": [0.001, 0.01],
            "y_var": [0.001, 0.01],
            }
        self.param_func = params_to_func(parameters)
        self.evaluation_parameters = [[self.param_func() for _ in range(5)] for _ in range(3)]
    def test_current_epochs(self):
        evaluation_counter = 10
        length = 3
        current_epochs = RayOptimizer.current_epochs(evaluation_counter, length)
        for i in range(evaluation_counter, evaluation_counter + length):
            self.assertTrue(i in current_epochs)

    def test_run_engine(self):
        output, timer = RayOptimizer.run_engine(self.engine, self.evaluation_parameters, "ImagePlane", MultiLayer([0., 1., 2.]), True)
        self.assertTrue(timer > 0.)
        self.assertTrue(len(output[0]) == 5)
        self.assertTrue(len(output) == 3)

    def test_reshape_list(self):
        list = [i for i in range(10)]
        reshaped_list = RayOptimizer.reshape_list(list, 2)
        self.assertEqual(len(reshaped_list), 10 // 2)
        self.assertRaises(ValueError, lambda: RayOptimizer.reshape_list(list, 3))

    def test_loss_from_output(self):
        output, timer = RayOptimizer.run_engine(self.engine, self.evaluation_parameters, "ImagePlane", MultiLayer([0., 1., 2.]), True)
        target_rays = self.engine.run([self.param_func() for _ in range(5)], MultiLayer([0., 1., 2.]))
        criterion: RayLoss = SinkhornLoss()
        
        loss = RayOptimizer.calculate_loss_from_output(output, target_rays, [0, 1, 2], criterion, "ImagePlane")
        self.assertTrue(isinstance(loss[0], Sample))


if __name__ == "__main__":
    unittest.main()
