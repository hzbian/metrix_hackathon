# this needs some major work
import time
from typing import List, Optional
import ax
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.service.ax_client import AxClient
from tqdm import trange
from ray_optim.ray_optimizer import OptimizerBackend, Target
from ray_tools.base.parameter import MutableParameter, NumericalParameter, RayParameterContainer
from ax.modelbridge.registry import Cont_X_trans, Models
from ax.modelbridge.generation_strategy import GenerationStep

class OptimizerBackendAx(OptimizerBackend):
   def __init__(self, ax_client: Optional[AxClient] = None):
       if ax_client is not None:
           self.ax_client = ax_client
       else:
           self.ax_client = AxClient()

   def optimizer_parameter_to_container_list(self, optimizer_parameter, search_space) -> tuple[list[int], list[RayParameterContainer]]:
       trial_index_list: list[int] = []
       param_container_list: list[RayParameterContainer] = []
       for trial_index, param_container in optimizer_parameter.items():
           all_parameters_copy = search_space.copy()
           for param_key in param_container.keys():
               all_parameters_copy.__setitem__(param_key, NumericalParameter(param_container[param_key]))
           param_container_list.append(all_parameters_copy)
           trial_index_list.append(trial_index)
       return trial_index_list, param_container_list
   
   def setup_optimization(self, target: Target):
       experiment_parameters = []
       for key, value in target.search_space.items():
           if isinstance(value, MutableParameter):
               experiment_parameters.append(
                   {"name": key, "type": "range", 'value_type': 'float', "bounds": list(value.value_lims)})

       self.ax_client.create_experiment(
           name="metrix_experiment",
           parameters=experiment_parameters,
           objectives={"mse": ObjectiveProperties(minimize=True)},
       )

   def optimize(self, objective, iterations, target: Target):
       ranger = trange(iterations)
       for _ in ranger:
           optimization_time = time.time()
           trials_to_evaluate = self.ax_client.get_next_trials(max_trials=10)[0]
           print("Optimization took {:.2f}s".format(time.time() - optimization_time))

           trial_index_list, ray_parameter_container_list = self.optimizer_parameter_to_container_list(trials_to_evaluate, target.search_space)
           results = objective(ray_parameter_container_list, target=target)

           for i in range(len(results)):
               self.ax_client.complete_trial(trial_index_list[i], {'mse': results[i].item()})

       best_parameters, metrics = self.ax_client.get_best_parameters()
       return best_parameters, metrics
   
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