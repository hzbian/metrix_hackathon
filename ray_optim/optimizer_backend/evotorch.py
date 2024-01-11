from collections.abc import Callable

from evotorch import Problem
from evotorch.algorithms import SNES
from evotorch.core import BoundsPair
from evotorch.logging import StdOutLogger
import torch

from ray_optim.ray_optimizer import Target, OptimizerBackend
from ray_tools.base.parameter import MutableParameter, NumericalParameter


class OptimizerBackendEvoTorch(OptimizerBackend):
    def optimize(self, objective: Callable, iterations: int, target: Target):
        optimize_parameters = target.search_space.copy()
        bounds = []
        for value in optimize_parameters.values():
            if isinstance(value, MutableParameter):
                bounds.append(BoundsPair(value.value_lims[0], value.value_lims[1]))
        problem = Problem("min", self.evotorch_objective(objective, target), solution_length=len(bounds),
                          initial_bounds=(-1, 1), vectorized=True)
        searcher = SNES(problem, popsize=1, stdev_init=10.0)
        _ = StdOutLogger(searcher, interval=50)
        searcher.run(iterations)

    def evotorch_objective(self, objective, target: Target):
        def output_objective(input: torch.Tensor):
            optimize_parameters = target.search_space.copy()
            for i, (key, value) in enumerate(optimize_parameters.items()):
                if isinstance(value, MutableParameter):
                    optimize_parameters[key] = NumericalParameter(input[0][i].item())
            output = objective(optimize_parameters, target=target)
            return output[min(output.keys())][0].mean().unsqueeze(0)

        return output_objective
