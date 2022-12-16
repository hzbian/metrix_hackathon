from ray_tools.parameter_builder import build_parameter_grid
from ray_tools.engine import RayEngine, RayParameterDict, RayBackendRayX
from ray_tools.parameter import ConstantParameter, RandomParameter, GridParameter

engine = RayEngine(rml_basefile='METRIX_U41_G1_H1_318eV_PS_MLearn.rml', ray_backend=RayBackendRayX(),
                   num_workers=10, as_generator=False)

param_func = lambda: RayParameterDict([
    (engine.template.U41_318eV.numberRays, ConstantParameter(value=1e4)),
    (engine.template.U41_318eV.translationXerror, RandomParameter(value_lims=(-0.25, 0.25))),
    (engine.template.U41_318eV.translationYerror, RandomParameter(value_lims=(-0.25, 0.25))),
    (engine.template.U41_318eV.rotationXerror, RandomParameter(value_lims=(-0.05, 0.05))),
    (engine.template.U41_318eV.rotationYerror, RandomParameter(value_lims=(-0.05, 0.05))),
])

params = [param_func() for _ in range(25)]
result = engine.run(params)

param_dict = RayParameterDict([
    (engine.template.U41_318eV.numberRays, ConstantParameter(value=1e4)),
    (engine.template.U41_318eV.translationXerror, GridParameter(value=[0.0, 1.0, 2.0])),
    (engine.template.U41_318eV.translationYerror, GridParameter(value=[3.0, 4.0, 5.0])),
    (engine.template.U41_318eV.rotationXerror, RandomParameter(value_lims=(-0.05, 0.05))),
    (engine.template.U41_318eV.rotationYerror, RandomParameter(value_lims=(-0.05, 0.05))),
])

params = build_parameter_grid(param_dict)
result = engine.run(params)