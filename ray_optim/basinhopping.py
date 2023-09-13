from typing import Callable
import numpy as np
from ray_optim.ray_optimizer import OptimizationTarget, OptimizerBackend
from ray_tools.base.parameter import MutableParameter, NumericalParameter


class OptimizerBackendBasinhopping(OptimizerBackend):
    def __init__(self, basinhopping_function):
        self.basinhopping_function = basinhopping_function

    def setup_optimization(self):
        pass

    @staticmethod
    def basinhopping_objective(objective, optimization_target: OptimizationTarget):
        def output_objective(input: np.ndarray):
            optimize_parameters = optimization_target.search_space.copy()
            for i, (key, value) in enumerate(optimize_parameters.items()):
                if isinstance(value, MutableParameter):
                    optimize_parameters[key] = NumericalParameter(input[i].item())
            output = objective(optimize_parameters, optimization_target=optimization_target)
            return tuple(value.mean().item() for value in output[min(output.keys())])

        return output_objective

    def optimize(self, objective: Callable, iterations: int, optimization_target: OptimizationTarget):
        optimize_parameters = optimization_target.search_space.copy()
        x0 = []
        bounds = []
        for _, value in optimize_parameters.items():
            if isinstance(value, MutableParameter):
                bounds.append([value.value_lims[0], value.value_lims[1]])
                x0.append((value.value_lims[1] - value.value_lims[0]) / 2. + value.value_lims[0])
        ret = self.basinhopping_function(self.basinhopping_objective(objective, optimization_target), x0,
                                         niter=iterations, interval=iterations, stepsize=1, T=0.01,
                                         minimizer_kwargs={"bounds": bounds}, disp=True)
        return ret.x, ret.fun

