from collections.abc import Callable

from evotorch import Problem
from evotorch.algorithms import SNES, CMAES
from evotorch.core import BoundsPair
from evotorch.logging import StdOutLogger
import torch
from ray_optim.optimizer_backend.base import OptimizerBackend

from ray_optim.ray_optimizer import Target
from ray_tools.base.parameter import MutableParameter, NumericalParameter


class OptimizerBackendEvoTorch(OptimizerBackend):
    def optimize(
        self,
        objective: Callable,
        iterations: int,
        target: Target,
        starting_point: dict[str, float] | None = None,
    ) -> tuple[dict[str, float], dict[str, float]]:
        optimize_parameters = target.search_space.copy()
        lb = []
        ub = []
        for value in optimize_parameters.values():
            if isinstance(value, MutableParameter):
                lb.append(value.value_lims[0])
                ub.append(value.value_lims[1])
        bounds = BoundsPair(torch.tensor(lb), torch.tensor(ub))
        problem = Problem(
            "min",
            self.evotorch_objective(objective, target),
            solution_length=len(lb),
            vectorized=True,
            bounds=bounds
        )
        searcher = CMAES(problem, popsize=100, stdev_init=.001)
        _ = StdOutLogger(searcher, interval=10)
        searcher.run(iterations)

        x_dict = {}
        x_val: torch.Tensor = searcher.status["best"].values
        mutable_index: int = 0
        for key in optimize_parameters.keys():
            if isinstance(optimize_parameters[key], MutableParameter):
                x_dict[key] = x_val[mutable_index].item()
                mutable_index += 1
        return x_dict, {"loss": searcher.status["best_eval"]}

    def evotorch_objective(self, objective, target: Target):
        def output_objective(input: torch.Tensor):
            optimize_parameters = target.search_space.copy()
            mutable_index = 0
            optimize_parameters_list = []
            for entry in input:
                for key, value in optimize_parameters.items():
                    if isinstance(value, MutableParameter):
                        optimize_parameters[key] = NumericalParameter(
                            entry[mutable_index].item()
                        )
                        mutable_index += 1
                optimize_parameters_list.append(optimize_parameters)
            output = objective(optimize_parameters_list, target=target)
            return torch.tensor(output, dtype=input.dtype, device=input.device)

        return output_objective
