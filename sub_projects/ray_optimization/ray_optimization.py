import sys

import optuna

import wandb
from losses import multi_objective_loss, sinkhorn_loss
from sub_projects.ray_optimization.real_data import import_data

sys.path.insert(0, '../../')
from ray_optim.ray_optimizer import OptimizerBackendOptuna, RayOptimizer, WandbLoggingBackend, \
    OffsetOptimizationTarget, OptimizerBackendBasinhopping, OptimizerBackendEvoTorch, RayScan

from ray_tools.base.parameter import RayParameterContainer, NumericalParameter, RandomParameter, MutableParameter, \
    RayParameter
from scipy.optimize import basinhopping
import config.config_gauss as CFG

wandb.init(entity=CFG.WANDB_ENTITY,
           project=CFG.WANDB_PROJECT,
           name=CFG.STUDY_NAME,
           mode='online' if CFG.LOGGING else 'disabled',
           )

engine = CFG.ENGINE
# optimize only some all_params
all_params = CFG.PARAM_FUNC()

for key in all_params:
    old_param = all_params[key]
    if isinstance(old_param, MutableParameter) and key in CFG.FIXED_PARAMS:
        all_params[key] = NumericalParameter((old_param.value_lims[1] + old_param.value_lims[0]) / 2)

target_params = RayParameterContainer()
for key, value in all_params.items():
    if isinstance(value, MutableParameter):
        value = (value.value_lims[1] + value.value_lims[0]) / 2
    if isinstance(value, NumericalParameter):
        value = value.get_value()
    target_params[key] = NumericalParameter(value)

# Bayesian Optimization
# ax_client = AxClient(early_stopping_strategy=None, verbose_logging=verbose)

# optimizer_backend_ax = OptimizerBackendAx(ax_client, search_space=all_params)

directions = CFG.MULTI_OBJECTIVE_DIRECTIONS if CFG.MULTI_OBJECTIVE else None
if CFG.OPTIMIZER == 'optuna':
    optuna_storage_path = CFG.OPTUNA_STORAGE_PATH if CFG.LOGGING else None
    optuna_study = optuna.create_study(directions=directions, sampler=CFG.SAMPLER,
                                       pruner=optuna.pruners.HyperbandPruner(),
                                       storage=optuna_storage_path, study_name=CFG.STUDY_NAME, load_if_exists=True)
    optimizer_backend = OptimizerBackendOptuna(optuna_study)
elif CFG.OPTIMIZER == 'evotorch':
    optimizer_backend = OptimizerBackendEvoTorch()
else:
    optimizer_backend = OptimizerBackendBasinhopping(basinhopping)

criterion = multi_objective_loss if CFG.MULTI_OBJECTIVE else sinkhorn_loss

ray_optimizer = RayOptimizer(optimizer_backend=optimizer_backend, criterion=criterion, engine=engine,
                             log_times=True, exported_plane=CFG.EXPORTED_PLANE,
                             transforms=CFG.TRANSFORMS,
                             logging_backend=WandbLoggingBackend(), plot_interval=CFG.PLOT_INTERVAL, iterations=CFG.ITERATIONS)

# target_rays = engine.run(target_params, transforms=transforms)
# best_parameters, metrics = ray_optimizer.optimize(target_rays, search_space=all_params, target_params=target_params,
#                                                  iterations=100)
# print(best_parameters, metrics)

offset_search_space = lambda: RayParameterContainer(
    [(k, type(v)(
        value_lims=(
            -CFG.MAX_DEVIATION * (v.value_lims[1] - v.value_lims[0]),
            CFG.MAX_DEVIATION * (v.value_lims[1] - v.value_lims[0])),
        rg=CFG.RG)) for
     k, v in
     all_params.items() if isinstance(v, RandomParameter)]
)

if CFG.REAL_DATA_DIR is None:
    target_offset = offset_search_space()
    uncompensated_parameters = [CFG.PARAM_FUNC() for _ in range(CFG.NUM_BEAMLINE_PARAM_SAMPLES)]
    compensated_parameters: list[RayParameterContainer[str, RayParameter]] = [v.clone() for v in uncompensated_parameters]
    for configuration in compensated_parameters:
        configuration.perturb(target_offset)
    compensated_transforms = RayOptimizer.translate_exported_plain_transforms(CFG.EXPORTED_PLANE, compensated_parameters, CFG.TRANSFORMS)
    observed_rays = engine.run(compensated_parameters, transforms=compensated_transforms)
    validation_scan = None
else:
    observed_rays = import_data(CFG.REAL_DATA_DIR, CFG.REAL_DATA_TRAIN_SET, CFG.Z_LAYERS, CFG.PARAM_FUNC(), check_value_lims=True)
    uncompensated_parameters = [element['param_container_dict'] for element in observed_rays]
    target_offset = None
    observed_validation_rays = import_data(CFG.REAL_DATA_DIR, CFG.REAL_DATA_VALIDATION_SET, CFG.Z_LAYERS, CFG.PARAM_FUNC(), check_value_lims=False)
    uncompensated_validation_parameters = [element['param_container_dict'] for element in observed_validation_rays]


initial_transforms = RayOptimizer.translate_exported_plain_transforms(CFG.EXPORTED_PLANE, uncompensated_parameters, CFG.TRANSFORMS)
uncompensated_rays = engine.run(uncompensated_parameters, transforms=initial_transforms)

if CFG.REAL_DATA_DIR is not None:
    validation_parameters_rays = engine.run(uncompensated_validation_parameters, transforms=initial_transforms)
    validation_scan = RayScan(uncompensated_parameters=uncompensated_validation_parameters, uncompensated_rays=validation_parameters_rays, observed_rays=observed_validation_rays)
else:
    validation_scan = None
offset_optimization_target = OffsetOptimizationTarget(observed_rays=observed_rays,
                                                      offset_search_space=offset_search_space(),
                                                      uncompensated_parameters=uncompensated_parameters,
                                                      uncompensated_rays=uncompensated_rays, target_offset=target_offset, validation_scan=validation_scan)


ray_optimizer.optimize(optimization_target=offset_optimization_target)
