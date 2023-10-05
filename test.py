from ray_tools.base.engine import GaussEngine
from ray_tools.base.parameter import RayParameterContainer, RandomParameter, NumericalParameter
from ray_tools.base.transform import Histogram, MultiLayer, RayTransformCompose
from sub_projects.ray_optimization.utils import ray_output_to_tensor
from ray_tools.base.utils import RandomGenerator
import optuna
import torch
import matplotlib.pyplot as plt

rg = RandomGenerator(42)
param_func = lambda: RayParameterContainer([
    ("number_rays", NumericalParameter(value=1000)),
    ("x_dir", NumericalParameter(value=0.)),
    ("y_dir", NumericalParameter(value=0.)),
    ("z_dir", NumericalParameter(value=1.)),
    ("direction_spread", NumericalParameter(value=0.)),
    ("correlation_factor", NumericalParameter(value=0.)),
    ("x_mean", RandomParameter(value_lims=(-2., 2.), rg=rg)),
    ("y_mean", RandomParameter(value_lims=(-2., 2.), rg=rg)),
    #("x_var", RandomParameter(value_lims=(1e-10, 0.01), rg=rg)),
    #("y_var", RandomParameter(value_lims=(1e-10, 0.01), rg=rg)),
    ("x_var", RandomParameter(value_lims=(0.15, 0.3), rg=rg)),
    ("y_var", RandomParameter(value_lims=(0.15, 0.3), rg=rg)),
])

target_params = [param_func()]
engine = GaussEngine()
#transforms = RayTransformCompose(MultiLayer([0.]), Histogram(n_bins=10))
transforms = Histogram(n_bins=25, x_lims=(-2., 2.), y_lims= (-2., 2.))
target_output =  engine.run(target_params, transforms=transforms)[0]
target_hist = target_output['ray_output']['ImagePlane']['histogram']

def objective(trial):
    trial_params = RayParameterContainer([
    ("number_rays", NumericalParameter(value=1000)),
    ("x_dir", NumericalParameter(value=0.)),
    ("y_dir", NumericalParameter(value=0.)),
    ("z_dir", NumericalParameter(value=1.)),
    ("direction_spread", NumericalParameter(value=0.)),
    ("correlation_factor", NumericalParameter(value=0.)),
    ("x_mean", NumericalParameter(value=trial.suggest_float("x_mean", -2, 2))),
    ("y_mean", NumericalParameter(value=trial.suggest_float("y_mean", -2, 2))),
    ("x_var", NumericalParameter(value=trial.suggest_float("x_var", target_params[0]['x_var'].value_lims[0], target_params[0]['x_var'].value_lims[1]))),
    ("y_var", NumericalParameter(value=trial.suggest_float("y_var", target_params[0]['y_var'].value_lims[0], target_params[0]['y_var'].value_lims[1]))),
    ])
    trial_hist = engine.run(trial_params, transforms=transforms)[0]['ray_output']['ImagePlane']['histogram']
    return ((target_hist - trial_hist)**2).mean()


study = optuna.create_study()
study.optimize(objective, n_trials=1000, show_progress_bar=True)
study.best_params['x_var']

selected_target_params = torch.tensor([target_params[0][key].value for key in study.best_params.keys()])
print("Target", selected_target_params)
selected_best_params = torch.tensor([val for val in study.best_params.values()])
print("Best", selected_best_params)
plt.imshow(target_hist)
plt.savefig("target_hist.png")

best_params = RayParameterContainer([
    ("number_rays", NumericalParameter(value=1000)),
    ("x_dir", NumericalParameter(value=0.)),
    ("y_dir", NumericalParameter(value=0.)),
    ("z_dir", NumericalParameter(value=1.)),
    ("direction_spread", NumericalParameter(value=0.)),
    ("correlation_factor", NumericalParameter(value=0.)),
    ("x_mean", NumericalParameter(value=study.best_params['x_mean'])),
    ("y_mean", NumericalParameter(value=study.best_params['y_mean'])),
    ("x_var", NumericalParameter(value=study.best_params['x_var'])),
    ("y_var", NumericalParameter(value=study.best_params['y_var'])),
    ])
best_hist = engine.run(best_params, transforms=transforms)[0]['ray_output']['ImagePlane']['histogram']
plt.imshow(best_hist)
plt.savefig("best_hist.png")