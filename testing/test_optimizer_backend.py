import random
import unittest
from unittest.mock import Mock
import optuna
import scipy
from ray_optim.optimizer_backend.ax import OptimizerBackendAx
from ray_optim.optimizer_backend.basinhopping import OptimizerBackendBasinhopping
from ray_optim.optimizer_backend.evotorch import OptimizerBackendEvoTorch
from ray_optim.optimizer_backend.optuna import OptimizerBackendOptuna
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.service.ax_client import AxClient

from ray_optim.ray_optimizer import OptimizerBackend, RayOptimizer, Target
from ray_tools.base.engine import GaussEngine
from ray_tools.base.parameter import RayParameterContainer
from ray_tools.base.transform import MultiLayer
from sub_projects.ray_optimization.configuration import params_to_func


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
        self.starting_point_func = params_to_func(parameters, mutable_only=True)
        self.evaluation_parameters = [
            [self.param_func() for _ in range(5)] for _ in range(4)
        ]
        self.target = Target(Mock(), self.param_func(), target_params=self.param_func())

    def mock_objective(
        self, param_container: list[RayParameterContainer], target: Target
    ) -> list[float]:
        self.assertIsInstance(param_container, list)
        self.assertIsInstance(param_container[0], RayParameterContainer)
        self.assertIsInstance(target, Target)
        self.assertIsInstance(target.search_space, RayParameterContainer)
        self.assertIsInstance(target.target_params, RayParameterContainer | None)
        return [random.random() for _ in range(len(param_container))]
    
    def check_optimization_from_backend(self, backend: OptimizerBackend):
        best_parameters, metrics = backend.optimize(self.mock_objective, 10, self.target)
        self.assertGreater(len(best_parameters), 0)
        self.assertIsInstance(best_parameters, dict)
        self.assertGreater(len(metrics), 0)
        self.assertIsInstance(metrics, dict)
        best_parameters, metrics = backend.optimize(self.mock_objective, 10, self.target, starting_point=best_parameters)
        self.assertGreater(len(best_parameters), 0)
        self.assertIsInstance(best_parameters, dict)
        self.assertGreater(len(metrics), 0)
        self.assertIsInstance(metrics, dict)
    
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
        backend = OptimizerBackendOptuna(optuna_study=study)
        self.check_optimization_from_backend(backend)

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
                    model=Models.SAASBO,
                    num_trials=-1,  # No limitation on how many trials should be produced from this step
                    max_parallelism=3,  # Parallelism limit for this step, often lower than for Sobol
                    # More on parallelism vs. required samples in BayesOpt:
                    # https://ax.dev/docs/bayesopt.html#tradeoff-between-parallelism-and-total-number-of-trials
                ),
            ]
        )
        ax_client = AxClient(generation_strategy=gs)
        backend = OptimizerBackendAx(ax_client=ax_client)
        self.check_optimization_from_backend(backend)
    
    def test_basinhopping(self):
        backend = OptimizerBackendBasinhopping(scipy.optimize.basinhopping)
        self.check_optimization_from_backend(backend)

    def test_evotorch(self):
        backend = OptimizerBackendEvoTorch()
        self.check_optimization_from_backend(backend)