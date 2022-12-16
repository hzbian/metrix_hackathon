from ray_tools.parameter_builder import build_parameter_grid
from ray_tools.engine import RayEngine
from ray_tools.backend import RayBackendDockerRAYX
from ray_tools.parameter import ConstantParameter, RandomParameter, GridParameter, RayParameterContainer

engine = RayEngine(rml_basefile='../rml_src/METRIX_U41_G1_H1_318eV_PS_MLearn.rml',
                   workdir='../ray_workdir',
                   ray_backend=RayBackendDockerRAYX(docker_image='ray-service',
                                                    ray_workdir='../ray_workdir',
                                                    verbose=True),
                   num_workers=-1, as_generator=False)

param_func = lambda: RayParameterContainer([
    (engine.template.U41_318eV.numberRays, ConstantParameter(value=1e4)),
    (engine.template.U41_318eV.translationXerror, RandomParameter(value_lims=(-0.25, 0.25))),
    (engine.template.U41_318eV.translationYerror, RandomParameter(value_lims=(-0.25, 0.25))),
    (engine.template.U41_318eV.rotationXerror, RandomParameter(value_lims=(-0.05, 0.05))),
    (engine.template.U41_318eV.rotationYerror, RandomParameter(value_lims=(-0.05, 0.05))),
])

params = [param_func() for _ in range(100)]
result = engine.run(params)

param_container = RayParameterContainer([
    (engine.template.U41_318eV.numberRays, ConstantParameter(value=1e4)),
    (engine.template.U41_318eV.translationXerror, GridParameter(value=[0.0, 1.0, 2.0])),
    (engine.template.U41_318eV.translationYerror, GridParameter(value=[3.0, 4.0, 5.0])),
    (engine.template.U41_318eV.rotationXerror, RandomParameter(value_lims=(-0.05, 0.05))),
    (engine.template.U41_318eV.rotationYerror, RandomParameter(value_lims=(-0.05, 0.05))),
])

params = build_parameter_grid(param_container)
result = engine.run(params)

engine.ray_backend.kill()
