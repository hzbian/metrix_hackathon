from typing import Any
from collections.abc import Callable
from optuna import Study, Trial
import torch
from ray_optim.ray_optimizer import Target, OptimizerBackend
from ray_tools.base.parameter import MutableParameter, NumericalParameter, RayParameterContainer


class OptimizerBackendOptuna(OptimizerBackend):
    def __init__(self, optuna_study: Study):
        self.optuna_study: Study = optuna_study

    def setup_optimization(self, target: Target):
        pass

    @staticmethod
    def optuna_objective(objective: Callable[[list[RayParameterContainer], Target], list[float]], target: Target) -> Callable[[Trial], float]:
        def output_objective(trial: Trial) -> float:
            optimize_parameters = target.search_space.copy()
            for key, value in optimize_parameters.items():
                if isinstance(value, MutableParameter):
                    optimize_parameters[key] = NumericalParameter(trial.suggest_float(key, value.value_lims[0],
                                                                                      value.value_lims[1]))

            output: list[float] = objective([optimize_parameters], target)
            return torch.tensor(output).mean().item()

        return output_objective

    def optimize(self, objective: Callable[[list[RayParameterContainer], Target], list[float]], iterations: int, target: Target, starting_point: dict[str, Any] | None = None):
        self.optuna_study.optimize(
            OptimizerBackendOptuna.optuna_objective(objective, target),
            n_trials=iterations,
            show_progress_bar=True)
        return self.optuna_study.best_params, {}
