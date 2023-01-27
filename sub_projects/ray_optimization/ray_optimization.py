import sys
import os

import wandb
import optuna
from optuna.samplers import TPESampler

from ax.service.ax_client import AxClient

sys.path.insert(0, '../../')
from ray_optim.ray_optimizer import OptimizerBackendAx, OptimizerBackendOptuna, RayOptimizer, WandbLoggingBackend

from ray_nn.metrics.geometric import SinkhornLoss

from ray_tools.base.parameter import RayParameterContainer, NumericalParameter, RandomParameter, MutableParameter
from ray_tools.base.utils import RandomGenerator
from ray_tools.base.engine import RayEngine
from ray_tools.base.transform import RayTransformConcat, ToDict, MultiLayer
from ray_tools.base.backend import RayBackendDockerRAYUI

wandb.init(entity='hzb-aos',
           project='metrix_hackathon_optimization',
           name='34-parameter-rayui-TPE-12-Layer-LongRun',
           mode='disabled',  # 'disabled' or 'online'
           )

root_dir = '../../'

rml_basefile = os.path.join(root_dir, 'rml_src', 'METRIX_U41_G1_H1_318eV_PS_MLearn.rml')
ray_workdir = os.path.join(root_dir, 'ray_workdir', 'optimization')

n_rays = ['1e4']

exported_plane = "ImagePlane"  # "Spherical Grating"

# transforms = [
#    RayTransformConcat({
#        'raw': ToDict(),
#    }),
# ]
transforms = MultiLayer([-26, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30], copy_directions=False)
verbose = False
engine = RayEngine(rml_basefile=rml_basefile,
                   exported_planes=[exported_plane],
                   ray_backend=RayBackendDockerRAYUI(docker_image='ray-ui-service',
                                                     ray_workdir=ray_workdir,
                                                     verbose=verbose),
                   num_workers=-1,
                   as_generator=False)

rg = RandomGenerator(seed=42)

param_func = lambda: RayParameterContainer([
    (engine.template.U41_318eV.numberRays, NumericalParameter(value=1e4)),
    (engine.template.U41_318eV.translationXerror, RandomParameter(value_lims=(-0.25, 0.25), rg=rg)),
    (engine.template.U41_318eV.translationYerror, RandomParameter(value_lims=(-0.25, 0.25), rg=rg)),
    (engine.template.U41_318eV.rotationXerror, RandomParameter(value_lims=(-0.05, 0.05), rg=rg)),
    (engine.template.U41_318eV.rotationYerror, RandomParameter(value_lims=(-0.05, 0.05), rg=rg)),
    (engine.template.ASBL.totalWidth, RandomParameter(value_lims=(1.9, 2.1), rg=rg)),
    (engine.template.ASBL.totalHeight, RandomParameter(value_lims=(0.9, 1.1), rg=rg)),
    (engine.template.ASBL.translationXerror, RandomParameter(value_lims=(-0.2, 0.2), rg=rg)),
    (engine.template.ASBL.translationYerror, RandomParameter(value_lims=(-0.2, 0.2), rg=rg)),
    (engine.template.M1_Cylinder.radius, RandomParameter(value_lims=(174.06, 174.36), rg=rg)),
    (engine.template.M1_Cylinder.rotationXerror, RandomParameter(value_lims=(-0.25, 0.25), rg=rg)),
    (engine.template.M1_Cylinder.rotationYerror, RandomParameter(value_lims=(-1., 1.), rg=rg)),
    (engine.template.M1_Cylinder.rotationZerror, RandomParameter(value_lims=(-1., 1.), rg=rg)),
    (engine.template.M1_Cylinder.translationXerror, RandomParameter(value_lims=(-0.15, 0.15), rg=rg)),
    (engine.template.M1_Cylinder.translationYerror, RandomParameter(value_lims=(-1., 1.), rg=rg)),
    (engine.template.SphericalGrating.radius, RandomParameter(value_lims=(109741., 109841.), rg=rg)),
    (engine.template.SphericalGrating.rotationYerror, RandomParameter(value_lims=(-1., 1.), rg=rg)),
    (engine.template.SphericalGrating.rotationZerror, RandomParameter(value_lims=(-2.5, 2.5), rg=rg)),
    (engine.template.ExitSlit.totalHeight, RandomParameter(value_lims=(0.009, 0.011), rg=rg)),
    (engine.template.ExitSlit.translationZerror, RandomParameter(value_lims=(-29., 31.), rg=rg)),
    (engine.template.ExitSlit.rotationZerror, RandomParameter(value_lims=(-0.3, 0.3), rg=rg)),
    (engine.template.E1.longHalfAxisA, RandomParameter(value_lims=(20600., 20900.), rg=rg)),
    (engine.template.E1.shortHalfAxisB, RandomParameter(value_lims=(300.721702601, 304.721702601), rg=rg)),
    (engine.template.E1.rotationXerror, RandomParameter(value_lims=(-0.5, 0.5), rg=rg)),
    (engine.template.E1.rotationYerror, RandomParameter(value_lims=(-7.5, 7.5), rg=rg)),
    (engine.template.E1.rotationZerror, RandomParameter(value_lims=(-4, 4), rg=rg)),
    (engine.template.E1.translationYerror, RandomParameter(value_lims=(-1, 1), rg=rg)),
    (engine.template.E1.translationZerror, RandomParameter(value_lims=(-1, 1), rg=rg)),
    (engine.template.E2.longHalfAxisA, RandomParameter(value_lims=(4325., 4425.), rg=rg)),
    (engine.template.E2.shortHalfAxisB, RandomParameter(value_lims=(96.1560870104, 98.1560870104), rg=rg)),
    (engine.template.E2.rotationXerror, RandomParameter(value_lims=(-0.5, 0.5), rg=rg)),
    (engine.template.E2.rotationYerror, RandomParameter(value_lims=(-7.5, 7.5), rg=rg)),
    (engine.template.E2.rotationZerror, RandomParameter(value_lims=(-4, 4), rg=rg)),
    (engine.template.E2.translationYerror, RandomParameter(value_lims=(-1, 1), rg=rg)),
    (engine.template.E2.translationZerror, RandomParameter(value_lims=(-1, 1), rg=rg)),
])

criterion = SinkhornLoss(normalize_weights='weights1', p=1, backend='online')

# optimize only some all_params
all_params = param_func()
fixed = []
#   - [ 'U41_318eV.translationXerror']  # ['U41_318eV.translationYerror', 'U41_318eV.rotationXerror', 'U41_318eV.rotationYerror', 'ASBL.totalWidth', 'ASBL.totalHeight', 'ASBL.translationXerror', 'ASBL.translationYerror', 'M1_Cylinder.radius', 'M1_Cylinder.rotationXerror', 'M1_Cylinder.rotationYerror', 'M1_Cylinder.rotationZerror', 'M1_Cylinder.translationXerror', 'M1_Cylinder.translationYerror', 'SphericalGrating.radius', 'SphericalGrating.rotationYerror', 'SphericalGrating.rotationZerror', 'ExitSlit.totalHeight', 'ExitSlit.translationZerror', 'ExitSlit.rotationZerror', 'E1.longHalfAxisA', 'E1.shortHalfAxisB', 'E1.rotationXerror', 'E1.rotationYerror', 'E1.rotationZerror', 'E1.translationYerror', 'E1.translationZerror', 'E2.longHalfAxisA', 'E2.shortHalfAxisB', 'E2.rotationXerror', 'E2.rotationYerror', 'E2.rotationZerror', 'E2.translationYerror', 'E2.translationZerror']

# Out[3]: odict_keys(['U41_318eV.numberRays', 'U41_318eV.translationXerror', 'U41_318eV.translationYerror', 'U41_318eV.rotationXerror', 'U41_318eV.rotationYerror', 'ASBL.totalWidth', 'ASBL.totalHeight', 'ASBL.translationXerror', 'ASBL.translationYerror', 'M1_Cylinder.radius', 'M1_Cylinder.rotationXerror', 'M1_Cylinder.rotationYerror', 'M1_Cylinder.rotationZerror', 'M1_Cylinder.translationXerror', 'M1_Cylinder.translationYerror', 'SphericalGrating.radius', 'SphericalGrating.rotationYerror', 'SphericalGrating.rotationZerror', 'ExitSlit.totalHeight', 'ExitSlit.translationZerror', 'ExitSlit.rotationZerror', 'E1.longHalfAxisA', 'E1.shortHalfAxisB', 'E1.rotationXerror', 'E1.rotationYerror', 'E1.rotationZerror', 'E1.translationYerror', 'E1.translationZerror', 'E2.longHalfAxisA', 'E2.shortHalfAxisB', 'E2.rotationXerror', 'E2.rotationYerror', 'E2.rotationZerror', 'E2.translationYerror', 'E2.translationZerror'])
for key in all_params:
    old_param = all_params[key]
    if isinstance(old_param, MutableParameter) and key in fixed:
        all_params[key] = NumericalParameter((old_param.value_lims[1] + old_param.value_lims[0]) / 2)

target_params = RayParameterContainer()
for key, value in all_params.items():
    if isinstance(value, MutableParameter):
        value = (value.value_lims[1] + value.value_lims[0]) / 2
    if isinstance(value, NumericalParameter):
        value = value.get_value()
    target_params[key] = NumericalParameter(value)

# Bayesian Optimization
ax_client = AxClient(early_stopping_strategy=None, verbose_logging=verbose)

optimizer_backend_ax = OptimizerBackendAx(ax_client, search_space=all_params)

optuna_study = optuna.create_study(sampler=TPESampler(), pruner=optuna.pruners.HyperbandPruner())
optimizer_backend_optuna = OptimizerBackendOptuna(optuna_study, search_space=all_params)
ray_optimizer = RayOptimizer(optimizer_backend=optimizer_backend_optuna, criterion=criterion, engine=engine,
                             log_times=True, exported_plane=exported_plane, search_space=all_params,
                             transforms=transforms,
                             logging_backend=WandbLoggingBackend())

target_rays = engine.run(target_params, transforms=transforms)
#best_parameters, metrics = ray_optimizer.optimize(target_rays, target_params=target_params, iterations=100)
#print(best_parameters, metrics)
target_configurations = [param_func() for _ in range(22)]

offset = RayParameterContainer(
    [(k, NumericalParameter(v.get_value() * rg.rg_random.uniform(0, 0.1))) for k, v in param_func().items() if isinstance(v, RandomParameter)])

perturbed_configurations = target_configurations.copy()
for configuration in perturbed_configurations:
    configuration.perturb(offset)
