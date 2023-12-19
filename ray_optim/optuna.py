from typing import Callable, Dict, List
from optuna import Study
import torch
from ray_optim.ray_optimizer import Target, OptimizerBackend
from ray_tools.base.parameter import MutableParameter, NumericalParameter, RayParameterContainer


class OptimizerBackendOptuna(OptimizerBackend):
    def __init__(self, optuna_study: Study):
        self.optuna_study: Study = optuna_study

    def setup_optimization(self, target: Target):
        pass

    @staticmethod
    def optuna_objective(objective: Callable[[List[RayParameterContainer], Target], List[float]], target: Target):
        def output_objective(trial):
            optimize_parameters = target.search_space.copy()
            for key, value in optimize_parameters.items():
                if isinstance(value, MutableParameter):
                    optimize_parameters[key] = NumericalParameter(trial.suggest_float(key, value.value_lims[0],
                                                                                      value.value_lims[1]))

            output = objective([optimize_parameters], target=target)
            return torch.tensor(output).mean()

        return output_objective

    def optimize(self, objective: Callable[[List[RayParameterContainer], Target], List[float]], iterations: int, target: Target):
        self.optuna_study.optimize(
            self.optuna_objective(objective, target),
            n_trials=iterations,
            show_progress_bar=True)
        return self.optuna_study.best_params, {}
