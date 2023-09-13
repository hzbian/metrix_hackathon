from optuna import Study
from ray_optim.ray_optimizer import OptimizationTarget, OptimizerBackend
from ray_tools.base.parameter import MutableParameter, NumericalParameter


class OptimizerBackendOptuna(OptimizerBackend):
    def __init__(self, optuna_study: Study):
        self.optuna_study: Study = optuna_study

    def setup_optimization(self):
        pass

    @staticmethod
    def optuna_objective(objective, optimization_target: OptimizationTarget):
        def output_objective(trial):
            optimize_parameters = optimization_target.search_space.copy()
            for key, value in optimize_parameters.items():
                if isinstance(value, MutableParameter):
                    optimize_parameters[key] = NumericalParameter(trial.suggest_float(key, value.value_lims[0],
                                                                                      value.value_lims[1]))

            output = objective(optimize_parameters, optimization_target=optimization_target)
            return tuple(value.mean().item() for value in output[min(output.keys())])

        return output_objective

    def optimize(self, objective, iterations, optimization_target: OptimizationTarget):
        self.optuna_study.optimize(
            self.optuna_objective(objective, optimization_target),
            n_trials=iterations,
            show_progress_bar=True)
        return self.optuna_study.best_params, {}
