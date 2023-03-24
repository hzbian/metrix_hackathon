import sys

import optuna

import wandb
from losses import multi_objective_loss, sinkhorn_loss

sys.path.insert(0, '../../')
from ray_optim.ray_optimizer import OptimizerBackendOptuna, RayOptimizer, WandbLoggingBackend, \
    OffsetOptimizationTarget

from ray_tools.base.parameter import RayParameterContainer, NumericalParameter, RandomParameter, MutableParameter, \
    RayParameter
from ray_tools.base.engine import RayEngine
from ray_tools.base.backend import RayBackendDockerRAYUI
import config.config_optimization_tpe as CFG

wandb.init(entity=CFG.WANDB_ENTITY,
           project=CFG.WANDB_PROJECT,
           name=CFG.STUDY_NAME,
           mode='online' if CFG.LOGGING else 'disabled',
           )

engine = RayEngine(rml_basefile=CFG.RML_BASEFILE,
                   exported_planes=[CFG.EXPORTED_PLANE],
                   ray_backend=RayBackendDockerRAYUI(docker_image='ray-ui-service',
                                                     ray_workdir=CFG.RAY_WORKDIR,
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
optuna_storage_path = CFG.OPTUNA_STORAGE_PATH if CFG.LOGGING else None

optuna_study = optuna.create_study(directions=directions, sampler=CFG.SAMPLER, pruner=optuna.pruners.HyperbandPruner(),
                                   storage=optuna_storage_path, study_name=CFG.STUDY_NAME)
optimizer_backend_optuna = OptimizerBackendOptuna(optuna_study)

criterion = multi_objective_loss if CFG.MULTI_OBJECTIVE else sinkhorn_loss

ray_optimizer = RayOptimizer(optimizer_backend=optimizer_backend_optuna, criterion=criterion, engine=engine,
                             log_times=True, exported_plane=CFG.EXPORTED_PLANE,
                             transforms=CFG.TRANSFORMS,
                             logging_backend=WandbLoggingBackend())

# target_rays = engine.run(target_params, transforms=transforms)
# best_parameters, metrics = ray_optimizer.optimize(target_rays, search_space=all_params, target_params=target_params,
#                                                  iterations=100)
# print(best_parameters, metrics)
target_parameters = [CFG.PARAM_FUNC() for _ in range(CFG.NUM_BEAMLINE_PARAM_SAMPLES)]

offset_search_space = lambda: RayParameterContainer(
    [(k, RandomParameter(
        value_lims=(
            -CFG.MAX_DEVIATION * (v.value_lims[1] - v.value_lims[0]),
            CFG.MAX_DEVIATION * (v.value_lims[1] - v.value_lims[0])),
        rg=CFG.RG)) for
     k, v in
     all_params.items() if isinstance(v, RandomParameter)]
)

offset = offset_search_space()
perturbed_parameters: list[RayParameterContainer[str, RayParameter]] = [v.clone() for v in target_parameters]
for configuration in perturbed_parameters:
    configuration.perturb(offset)

offset_target_rays = engine.run(perturbed_parameters, transforms=CFG.TRANSFORMS)
target_rays_without_offset = engine.run(target_parameters, transforms=CFG.TRANSFORMS)
offset_optimization_target = OffsetOptimizationTarget(target_rays=offset_target_rays, target_offset=offset,
                                                      search_space=offset_search_space(),
                                                      perturbed_parameters=target_parameters,
                                                      target_rays_without_offset=target_rays_without_offset)

ray_optimizer.optimize(optimization_target=offset_optimization_target, iterations=CFG.ITERATIONS)
