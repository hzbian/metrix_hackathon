import sys
import os
import optuna

import wandb
from losses import multi_objective_loss, sinkhorn_loss
from sub_projects.ray_optimization.real_data import import_data

sys.path.insert(0, '../../')
from ray_optim.ray_optimizer import OptimizerBackendOptuna, RayOptimizer, WandbLoggingBackend, \
    OffsetOptimizationTarget, OptimizerBackendBasinhopping

from ray_tools.base.parameter import RayParameterContainer, NumericalParameter, RandomParameter, MutableParameter, \
    RayParameter
from ray_tools.base.engine import RayEngine
from ray_tools.base.backend import RayBackendDockerRAYUI
from scipy.optimize import basinhopping
import config.config_optimization_tpe as CFG

wandb.init(entity=CFG.WANDB_ENTITY,
           project=CFG.WANDB_PROJECT,
           name=CFG.STUDY_NAME,
           mode='online' if CFG.LOGGING else 'disabled',
           )

engine = RayEngine(rml_basefile=CFG.RML_BASEFILE,
                   exported_planes=[CFG.EXPORTED_PLANE],
                   ray_backend=RayBackendDockerRAYUI(docker_image='ray-ui-service',
                                                     docker_container_name=CFG.STUDY_NAME,
                                                     ray_workdir=os.path.join(CFG.RAY_WORKDIR, CFG.STUDY_NAME),
                                                     verbose=CFG.VERBOSE),
                   num_workers=-1,
                   as_generator=False)

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
else:
    optimizer_backend = OptimizerBackendBasinhopping(basinhopping)

criterion = multi_objective_loss if CFG.MULTI_OBJECTIVE else sinkhorn_loss

ray_optimizer = RayOptimizer(optimizer_backend=optimizer_backend, criterion=criterion, engine=engine,
                             log_times=True, exported_plane=CFG.EXPORTED_PLANE,
                             transforms=CFG.TRANSFORMS,
                             logging_backend=WandbLoggingBackend(), iterations=CFG.ITERATIONS)

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
    offset = offset_search_space()
    initial_parameters = [CFG.PARAM_FUNC() for _ in range(CFG.NUM_BEAMLINE_PARAM_SAMPLES)]
    perturbed_parameters: list[RayParameterContainer[str, RayParameter]] = [v.clone() for v in initial_parameters]
    for configuration in perturbed_parameters:
        configuration.perturb(offset)
    perturbed_parameters_rays = engine.run(perturbed_parameters, transforms=CFG.TRANSFORMS)
else:
    perturbed_parameters_rays = import_data(CFG.REAL_DATA_DIR, CFG.Z_LAYERS, CFG.PARAM_FUNC())
    initial_parameters = [element['param_container_dict'] for element in perturbed_parameters_rays]
    offset = None

initial_parameters_rays = engine.run(initial_parameters, transforms=CFG.TRANSFORMS)
offset_optimization_target = OffsetOptimizationTarget(perturbed_parameters_rays=perturbed_parameters_rays,
                                                      search_space=offset_search_space(),
                                                      initial_parameters=initial_parameters,
                                                      initial_parameters_rays=initial_parameters_rays, offset=offset)

ray_optimizer.optimize(optimization_target=offset_optimization_target)
