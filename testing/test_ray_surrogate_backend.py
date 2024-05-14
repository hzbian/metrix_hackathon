import unittest

import omegaconf
from hydra.utils import instantiate
from hydra import initialize, compose

from ray_nn.nn.xy_hist_data_models import HistSurrogateEngine, MetrixXYHistSurrogate
#from ray_tools.base.backend import RayBackendDockerRAYUI
from ray_tools.base.engine import RayEngine
from ray_tools.base.parameter import NumericalParameter, RandomParameter, RayParameterContainer
from ray_tools.base.transform import Histogram, RayTransformConcat, MultiLayer
from ray_tools.base.utils import RandomGenerator

from datasets.metrix_simulation.config_ray_emergency_surrogate import PARAM_CONTAINER_FUNC as params

class RayBackendTest(unittest.TestCase):

    def setUp(self):
        self.exported_planes = ["ImagePlane"]
        self.n_rays = 100

        self.rg = RandomGenerator(seed=42)
        self.dist_layers = [-25., -20., -15., -10., -5., 0., 5., 10., 15., 20., 25., 30.]
        self.transform = RayTransformConcat({
            'ml': MultiLayer(dist_layers=self.dist_layers,
                             copy_directions=False,
                             transform=Histogram(n_bins=256)),
            'hist': Histogram(n_bins=1024),

        })
        self.engine = HistSurrogateEngine(MetrixXYHistSurrogate)
        

    def param_func(self):
        return RayParameterContainer([
            ("U41_318eV.numberRays", NumericalParameter(value=self.n_rays)),
            (
                "U41_318eV.translationXerror",
                RandomParameter(value_lims=(-0.25, 0.25), rg=self.rg))
        ])

    def test_output_type_single_output(self):
        # For single output the output should be a list
        n_examples = 10

        result = self.engine.run(param_containers=[params() for _ in range(n_examples)],
                                 transforms={exported_plane: self.transform for exported_plane in self.exported_planes})
        print(result)
        #result = list()
        self.assertTrue(type(result) is list)

if __name__ == '__main__':
    unittest.main()
