import abc
from collections.abc import Callable
import numpy as np
import scipy
from ray_optim.optimizer_backend.base import OptimizerBackend
from ray_optim.ray_optimizer import Target
from ray_tools.base.parameter import MutableParameter, NumericalParameter


class OptimizerBackendSciPy(OptimizerBackend):
    @abc.abstractmethod
    def optimizer_function(
        self, fun, x0, iterations, bounds
    ) -> scipy.optimize.OptimizeResult:
        return

    @staticmethod
    def scipy_objective(objective, target: Target):
        def output_objective(input: np.ndarray):
            optimize_parameters = target.search_space.clone()
            mutable_index: int = 0
            for key, value in optimize_parameters.items():
                if isinstance(value, MutableParameter):
                    optimize_parameters[key] = NumericalParameter(
                        input[mutable_index].item()
                    )
                    mutable_index += 1
            output = objective([optimize_parameters], target=target)
            return output

        return output_objective

    def optimize(
        self,
        objective: Callable,
        iterations: int,
        target: Target,
        starting_point: dict[str, float] | None = None,
    ) -> tuple[dict[str, float], dict[str, float]]:
        optimize_parameters = target.search_space.clone()
        x0 = []
        bounds = []
        for key, value in optimize_parameters.items():
            if isinstance(value, MutableParameter):
                bounds.append([value.value_lims[0], value.value_lims[1]])
                if starting_point is not None:
                    x0.append(starting_point[key])
                else:
                    x0.append(
                        (value.value_lims[1] - value.value_lims[0]) / 2.0
                        + value.value_lims[0]
                    )
        ret = self.optimizer_function(
            self.scipy_objective(objective, target), x0, iterations, bounds
        )

        x_dict = {}
        mutable_index: int = 0
        for key in optimize_parameters.keys():
            if isinstance(optimize_parameters[key], MutableParameter):
                x_dict[key] = ret.x[mutable_index]
                mutable_index += 1
        return x_dict, {"loss": ret.fun}


class OptimizerBackendBasinhopping(OptimizerBackendSciPy):
    def __init__(self, interval=1, stepsize=1, T=0.01):
        self.interval = interval
        self.stepsize = stepsize
        self.T = T

    def optimizer_function(self, fun, x0, iterations, bounds):
        return scipy.optimize.basinhopping(
            fun,
            x0,
            niter=iterations,
            interval=self.interval,
            stepsize=self.stepsize,
            T=self.T,
            minimizer_kwargs={"bounds": bounds},
            disp=True,
        )


class OptimizerBackendNelderMead(OptimizerBackendSciPy):
    def optimizer_function(self, fun, x0, iterations, bounds):
        return scipy.optimize.minimize(
            fun,
            x0,
            method="Nelder-Mead",
            options={"disp": True, "maxiter": iterations},
            bounds=bounds,
        )
        


class OptimizerBackendPowell(OptimizerBackendSciPy):
    def optimizer_function(self, fun, x0, iterations, bounds):
        return scipy.optimize.minimize(
            fun,
            x0,
            method="Powell",
            options={"disp": True, "maxiter": iterations},
            bounds=bounds,
        )
