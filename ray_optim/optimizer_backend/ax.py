from collections.abc import Callable
import time
from typing import cast
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.service.ax_client import AxClient
import torch
from tqdm import trange
from ray_optim.optimizer_backend.base import OptimizerBackend
from ray_optim.ray_optimizer import Target
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ray_tools.base.parameter import MutableParameter, NumericalParameter, RayParameterContainer
from ax.modelbridge.registry import Models
from ax.modelbridge.generation_strategy import GenerationStep
from ax.core.types import TParameterization, TModelPredictArm

class OptimizerBackendAx(OptimizerBackend):
   def __init__(self, ax_client: AxClient | None = None):
        if ax_client is not None:
           self.ax_client = ax_client
        else:
            gs = GenerationStrategy(
                steps=[
                    # 1. Initialization step (does not require pre-existing data and is well-suited for
                    # initial sampling of the search space)
                    GenerationStep(
                        model=Models.SOBOL,
                        num_trials=5,  # How many trials should be produced from this generation step
                        min_trials_observed=3,  # How many trials need to be completed to move to next model
                        max_parallelism=5,  # Max parallelism for this step
                        model_kwargs={"seed": 999},  # Any kwargs you want passed into the model
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
            self.ax_client = AxClient(generation_strategy=gs)

   def optimizer_parameter_to_container_list(self, optimizer_parameter, search_space) -> tuple[list[int], list[RayParameterContainer]]:
       trial_index_list: list[int] = []
       param_container_list: list[RayParameterContainer] = []
       for trial_index, param_container in optimizer_parameter.items():
           all_parameters_copy = search_space.clone()
           for param_key in param_container.keys():
               all_parameters_copy.__setitem__(param_key, NumericalParameter(param_container[param_key]))
           param_container_list.append(all_parameters_copy)
           trial_index_list.append(trial_index)
       return trial_index_list, param_container_list
   
   def _setup_optimization(self, target: Target):
       experiment_parameters = []
       for key, value in target.search_space.items():
           if isinstance(value, MutableParameter):
               experiment_parameters.append(
                   {"name": key, "type": "range", 'value_type': 'float', "bounds": list(value.value_lims)})

       self.ax_client.create_experiment(
           name="metrix_experiment",
           parameters=experiment_parameters,
           objectives={"mse": ObjectiveProperties(minimize=True)},
           overwrite_existing_experiment=True,
       )

   def optimize(self, objective: Callable, iterations: int, target: Target, starting_point: dict[str, float] | None = None) -> tuple[dict[str, float], dict[str, float]]:
       self._setup_optimization(target)
       start: TParameterization = cast(TParameterization, starting_point)
       if starting_point is not None:
           self.ax_client.attach_trial(parameters=start, arm_name="initial_trial")
       ranger = trange(iterations)
       for _ in ranger:
           optimization_time = time.time()
           trials_to_evaluate = self.ax_client.get_next_trials(max_trials=10)[0]
           print("Optimization took {:.2f}s".format(time.time() - optimization_time))

           trial_index_list, ray_parameter_container_list = self.optimizer_parameter_to_container_list(trials_to_evaluate, target.search_space)
           results = objective(ray_parameter_container_list, target=target)

           for i in range(len(results)):
               result = results[i].item() if isinstance(results[i], torch.Tensor) else results[i]
               self.ax_client.complete_trial(trial_index_list[i], {'mse': result})
       best_parameters_metrics = self.ax_client.get_best_parameters()
       assert best_parameters_metrics is not None
       best_parameters, metrics = best_parameters_metrics
       best_parameters_dict: dict[str, float] = cast(dict[str, float], best_parameters)
       assert metrics is not None
       metrics_dict: dict[str, float] = metrics[0]
       return best_parameters_dict, metrics_dict
   
def get_model(selection: str):
    if selection == "SOBOL":
       return Models.SOBOL
    else:
        raise Exception("This name does not exist.")
       
def get_step(model_name: str, max_parallelism: int, num_trials: int):
    return GenerationStep(
        model = get_model(model_name),
        max_parallelism=max_parallelism,
        num_trials=num_trials,
    )