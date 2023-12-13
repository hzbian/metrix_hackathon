# this needs some major work
import time
from typing import List

from ax.service.ax_client import AxClient
from tqdm import trange
from ray_optim.ray_optimizer import OptimizerBackend, Target
from ray_tools.base.parameter import MutableParameter, NumericalParameter, RayParameterContainer


class OptimizerBackendAx(OptimizerBackend):
   def __init__(self, ax_client: AxClient, search_space: RayParameterContainer):
       self.search_space = search_space
       self.ax_client = ax_client

   def optimizer_parameter_to_container_list(self, optimizer_parameter) -> List[RayParameterContainer]:
       trial_params_first_key = min(optimizer_parameter.keys())
       param_container_list = []

       for i in range(trial_params_first_key, trial_params_first_key + len(optimizer_parameter)):
           all_parameters_copy = self.search_space.copy()
           for param_key in optimizer_parameter[trial_params_first_key].keys():
               all_parameters_copy.__setitem__(param_key, NumericalParameter(optimizer_parameter[i][param_key]))
           param_container_list.append(all_parameters_copy)
       return param_container_list

   def setup_optimization(self):
       experiment_parameters = []
       for key, value in self.search_space.items():
           if isinstance(value, MutableParameter):
               experiment_parameters.append(
                   {"name": key, "type": "range", 'value_type': 'float', "bounds": list(value.value_lims)})

       self.ax_client.create_experiment(
           name="metrix_experiment",
           parameters=experiment_parameters,
           objective_name="metrix",
           minimize=True,
       )

   def optimize(self, objective, iterations, target: Target):
       ranger = trange(iterations)
       for _ in ranger:
           optimization_time = time.time()
           trials_to_evaluate = self.ax_client.get_next_trials(max_trials=10)[0]
           print("Optimization took {:.2f}s".format(time.time() - optimization_time))

           ray_parameter_container_list = self.optimizer_parameter_to_container_list(trials_to_evaluate)
           results = objective(ray_parameter_container_list, target.target_rays,
                               target_params=target.target_params)

           for trial_index in results:
               self.ax_client.complete_trial(trial_index, results[trial_index])

       best_parameters, metrics = self.ax_client.get_best_parameters()
       return best_parameters, metrics
