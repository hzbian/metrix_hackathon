import random
from typing import List, Optional
import unittest
from unittest.mock import Mock
import optuna
from ray_optim.ax import OptimizerBackendAx
from ray_optim.optuna import OptimizerBackendOptuna
from ax.service.ax_client import AxClient
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import ModelRegistryBase, Models

from ray_optim.ray_optimizer import RayOptimizer, Sample, Target
from ray_tools.base.engine import GaussEngine
from ray_tools.base.parameter import RayParameterContainer
from ray_tools.base.transform import MultiLayer
from sub_projects.ray_optimization.configuration import params_to_func
from sub_projects.ray_optimization.losses.torch import BoxIoULoss
from sub_projects.ray_optimization.losses.losses import RayLoss


class TestOptimizerBackend(unittest.TestCase):
    def setUp(self):
        self.engine = GaussEngine()
        parameters = {
            "number_rays": 1e2,
            "x_dir": 0.0,
            "y_dir": 0.0,
            "z_dir": 1.0,
            "direction_spread": 0.0,
            "correlation_factor": [-0.8, 0.8],
            "x_mean": [-2.0, 2.0],
            "y_mean": [-2.0, 2.0],
            "x_var": [0.001, 0.01],
            "y_var": [0.001, 0.01],
        }
        self.param_func = params_to_func(parameters)
        self.evaluation_parameters = [
            [self.param_func() for _ in range(5)] for _ in range(4)
        ]

    def mock_objective(
        self, param_container: list[RayParameterContainer], target: Target
    ) -> List[float]:
        self.assertIsInstance(param_container, list)
        self.assertIsInstance(param_container[0], RayParameterContainer)
        self.assertIsInstance(target, Target)
        self.assertIsInstance(target.search_space, RayParameterContainer)
        self.assertIsInstance(target.target_params, Optional[RayParameterContainer])
        return [random.random() for _ in range(len(param_container))]

    def test_current_epochs(self):
        evaluation_counter = 10
        length = 4
        current_epochs = RayOptimizer.current_epochs(evaluation_counter, length)
        for i in range(evaluation_counter, evaluation_counter + length):
            self.assertTrue(i in current_epochs)

        output, timer = RayOptimizer.run_engine(
            self.engine,
            self.evaluation_parameters,
            "ImagePlane",
            MultiLayer([0.0, 1.0, 2.0]),
            True,
        )

    def test_optuna(self):
        study = optuna.create_study()
        obo = OptimizerBackendOptuna(optuna_study=study)
        target = Target(Mock(), self.param_func(), target_params=self.param_func())
        obo.optimize(self.mock_objective, 10, target)
        target = Target(Mock(), self.param_func(), target_params=None)
        obo.optimize(self.mock_objective, 10, target)

    def test_ax(self):
        gs = GenerationStrategy(
            steps=[
                # 1. Initialization step (does not require pre-existing data and is well-suited for
                # initial sampling of the search space)
                GenerationStep(
                    model=Models.SOBOL,
                    num_trials=5,  # How many trials should be produced from this generation step
                    min_trials_observed=3,  # How many trials need to be completed to move to next model
                    max_parallelism=5,  # Max parallelism for this step
                    model_kwargs={
                        "seed": 999
                    },  # Any kwargs you want passed into the model
                    model_gen_kwargs={},  # Any kwargs you want passed to `modelbridge.gen`
                ),
                # 2. Bayesian optimization step (requires data obtained from previous phase and learns
                # from all data available at the time of each new candidate generation call)
                GenerationStep(
                    model=Models.BOTORCH_MODULAR,
                    num_trials=-1,  # No limitation on how many trials should be produced from this step
                    max_parallelism=3,  # Parallelism limit for this step, often lower than for Sobol
                    # More on parallelism vs. required samples in BayesOpt:
                    # https://ax.dev/docs/bayesopt.html#tradeoff-between-parallelism-and-total-number-of-trials
                ),
            ]
        )
        search_space=self.param_func()
        abo = OptimizerBackendAx()
        abo.setup_optimization(target)
        target = Target(Mock(), search_space, target_params=self.param_func())
        abo.optimize(self.mock_objective, 10, target)
