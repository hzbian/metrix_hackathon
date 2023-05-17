import unittest

import optuna
import os

from ray_optim.ray_optimizer import OptimizerBackend, RayOptimizer, OptimizerBackendOptuna
from ray_tools.base.backend import RayBackendDockerRAYUI
from ray_tools.base.engine import RayEngine
from ray_tools.base.parameter import RayParameterContainer, RandomParameter, NumericalParameter
from ray_tools.base.utils import RandomGenerator
from sub_projects.ray_optimization.losses import sinkhorn_loss


class RayOptimizerTest(unittest.TestCase):

    def test_optimizer_backend(self):
        optuna_study = optuna.create_study(study_name="test")
        ROOT_DIR = '../../'
        RML_BASEFILE = os.path.join(ROOT_DIR, 'rml_src', 'METRIX_U41_G1_H1_318eV_PS_MLearn.rml')
        engine = RayEngine(rml_basefile=RML_BASEFILE,
                           exported_planes=['ImagePlane'],
                           ray_backend=RayBackendDockerRAYUI(docker_image='ray-ui-service',
                                                             docker_container_name='test',
                                                             ray_workdir='/tmp',
                                                             verbose=True),
                           num_workers=-1,
                           as_generator=False)
        #opti: RayOptimizer = RayOptimizer(optimizer_backend=OptimizerBackendOptuna(optuna_study=optuna_study), criterion=sinkhorn_loss, engine=engine, exported_plane='ImagePlane')

        self.assertEqual('foo'.upper(), 'FOO')

    def test_perturb_parameter(self):
        rg = RandomGenerator(seed=42)
        param_func = lambda: RayParameterContainer([
            ("U41_318eV.numberRays", NumericalParameter(value=1e4)),
            ("U41_318eV.translationXerror", RandomParameter(value_lims=(-0.25, 0.25), rg=rg))])

        parameters = RayParameterContainer([
            ("U41_318eV.translationXerror", NumericalParameter(value=0.1))])
        initial_params = param_func()
        param_container_list = [initial_params]
        evaluation_parameters = RayOptimizer.get_evaluation_parameters(param_container_list, [parameters])
        self.assertEqual(evaluation_parameters[0]['U41_318eV.translationXerror'].get_value(), initial_params['U41_318eV.translationXerror'].get_value() - 0.1)
        

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)


if __name__ == '__main__':
    unittest.main()
