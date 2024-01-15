import unittest

import hydra
import omegaconf
from hydra.utils import instantiate
from hydra import initialize, compose
from ray_optim.ray_optimizer import OffsetTarget, RayOptimizer

from ray_tools.base.backend import RayBackendDockerRAYUI
from ray_tools.base.engine import RayEngine
from ray_tools.base.parameter import NumericalParameter, RandomParameter, RayParameterContainer
from ray_tools.base.transform import Histogram, RayTransformConcat, MultiLayer
from ray_tools.base.utils import RandomGenerator
from sub_projects.ray_optimization.ray_optimization import RayOptimization


class SimulationBackendTest(unittest.TestCase):
    def test_with_initialize(self) -> None:
            with initialize(version_base=None, config_path="../sub_projects/ray_optimization/conf"):
                # config is relative to a module
                cfg = compose(config_name="config", overrides=["logging_backend=",
                "+logging_backend._target_=ray_optim.logging.DebugPlotBackend", "target_configuration=metrixs_fixed_crit", "target_configuration.max_target_deviation=0.02", "target_configuration.max_sample_generation_deviation=0.2"],)
                print(omegaconf.OmegaConf.to_yaml(cfg))
                ray_optimization: RayOptimization = instantiate(cfg)
                ray_optimization.setup_target()
                target: OffsetTarget = ray_optimization.target
                target.recalculate_cpu_tensors("ImagePlane")
                out_dict = RayOptimizer.plot_initial_plots(target, "ImagePlane")
                #out_dict['fancy_footprint'].write_html('out.html')
