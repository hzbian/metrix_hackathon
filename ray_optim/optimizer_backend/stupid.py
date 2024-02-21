
from collections.abc import Callable
from typing import Any
import random
import math
from tqdm import tqdm
from ray_optim.optimizer_backend.base import OptimizerBackend
from ray_optim.target import Target
from ray_tools.base.parameter import MutableParameter, NumericalParameter, RayParameterContainer


class OptimizerBackendStupid(OptimizerBackend):
    def __init__(self, step_size: float = 0.1, cosine_annealing: bool = True):
        self.step_size = step_size
        self.cosine_annealing: bool = cosine_annealing
    def optimize(self, objective: Callable[[list[RayParameterContainer], Target], list[float]], iterations: int, target: Target, starting_point: dict[str, float] | None = None) -> tuple[dict[str, float], dict[str, float]]:
        optimize_parameters: RayParameterContainer = target.search_space.clone()
        if starting_point is not None:
            for key in starting_point.keys():
                current_parameter = optimize_parameters[key]
                if isinstance(current_parameter, NumericalParameter):
                    current_parameter.value = starting_point[key]
        mutable_parameters_keys = [key for key, value in optimize_parameters.items() if isinstance(value, MutableParameter)] 
        current_parameters = optimize_parameters
        all_time_best_loss: float = objective([current_parameters], target)[0]
        all_time_best_parameters: RayParameterContainer = current_parameters
        current_best_loss: float = all_time_best_loss
        q = tqdm(range(iterations))
        for i in q:
            q.set_postfix({"best_loss": all_time_best_loss, "cur_loss": current_best_loss})
            perturbation_key = random.choice(mutable_parameters_keys)
            perturbation_sign = 1 if random.random() < 0.5 else -1
            perturbation_parameters = current_parameters[perturbation_key]
            assert isinstance(perturbation_parameters, MutableParameter)
            if self.cosine_annealing:
                annealing_factor = math.cos(i*iterations)
            perturbation_value: float = perturbation_sign * (perturbation_parameters.value_lims[1]-perturbation_parameters.value_lims[0]) / 2 * self.step_size * annealing_factor
            current_parameters_copy: RayParameterContainer = current_parameters.clone()
            current_parameters_copy.perturb(RayParameterContainer({perturbation_key: NumericalParameter(perturbation_value)}))
            losses: list[float] = objective([current_parameters_copy], target)
            current_best_loss = losses[0]
            if current_best_loss < all_time_best_loss:
                current_parameters = current_parameters_copy
                all_time_best_loss = current_best_loss
                all_time_best_parameters = current_parameters
        return all_time_best_parameters.to_value_dict(), {"loss": all_time_best_loss}