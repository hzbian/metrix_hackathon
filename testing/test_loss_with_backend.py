from collections import OrderedDict
import pickle
import unittest

import omegaconf
from hydra.utils import instantiate
from hydra import initialize, compose
import torch
import matplotlib.pyplot as plt
from tqdm import trange

from ray_optim.target import OffsetTarget
from ray_tools.base.backend import RayBackendDockerRAYUI
from ray_tools.base.engine import RayEngine
from ray_tools.base.parameter import NumericalOutputParameter, NumericalParameter, OutputParameter, RandomParameter, RayParameter, RayParameterContainer
from ray_tools.base.transform import Histogram, RayTransformConcat, MultiLayer, RayTransformDummy
from ray_tools.base.utils import RandomGenerator
from sub_projects.ray_optimization.losses.geometric import SinkhornLoss
from sub_projects.ray_optimization.losses.torch import MSELoss


class RayBackendTest():

    def setUp(self):
        self.exported_planes = ["ImagePlane"]
        self.n_rays = 100
        n_examples = 2
        load_target_path = 'datasets/metrix_simulation/shrinked_offset_target.pkl'

        self.rg = RandomGenerator(seed=42)
        self.dist_layers = [-25., -20., -15., -10., -5., 0., 5., 10., 15., 20., 25., 30.]
        self.transform = MultiLayer(dist_layers = self.dist_layers, copy_directions=False)
        self.engine = RayEngine(rml_basefile='rml_src/METRIX_U41_G1_H1_318eV_PS_MLearn.rml',
                                exported_planes=self.exported_planes,
                                ray_backend=RayBackendDockerRAYUI(docker_image='ray-ui-service',
                                                                  docker_container_name='ray-ui-service-test',
                                                                  dockerfile_path='ray_docker/rayui',
                                                                  ray_workdir='/dev/shm/ray-workdir',
                                                                  verbose=False),
                                num_workers=-1,
                                as_generator=False)
        with open(load_target_path, "rb") as input_file:
            offset_target: OffsetTarget = pickle.load(input_file)
            observed_ray_dicts = [offset_target.observed_rays[i]['param_container_dict'] for i in range(len(offset_target.observed_rays))]
            observed_ray_parameter_containers: list[RayParameterContainer] = []
            for ray_dict in observed_ray_dicts:
                ray_parameter_container: RayParameterContainer = RayParameterContainer([(key, NumericalParameter(value)) if key[:10] != 'ImagePlane' else (key, NumericalOutputParameter(value)) for (key, value) in ray_dict.items()])
                observed_ray_parameter_containers.append(ray_parameter_container)
            observed_rays_params = observed_ray_parameter_containers
            calculated_offset_dict = [("U41_318eV.translationXerror", -0.024764858682925786), ("U41_318eV.translationYerror", 0.02732677138397433), ("U41_318eV.rotationXerror", 0.0016287544291003647), ("U41_318eV.rotationYerror", 0.0006257437223599462), ("ASBL.totalWidth", 0.005123893915722495), ("ASBL.totalHeight", 0.01598461125088929), ("ASBL.translationXerror", -0.040478761707858565), ("ASBL.translationYerror", 0.043094028456024745), ("M1_Cylinder.radius", 0.0252677364683167), ("M1_Cylinder.rotationXerror", -0.04120951080392725), ("M1_Cylinder.rotationYerror", 0.03223799719614692), ("M1_Cylinder.rotationZerror", -0.21381077783560865), ("M1_Cylinder.translationXerror", -0.2999468185230245), ("M1_Cylinder.translationYerror", 0.20542171867572506), ("SphericalGrating.radius", -13.5814957318526), ("SphericalGrating.rotationYerror", -0.16507163595664348), ("SphericalGrating.rotationZerror", -0.5648506506558534), ("ExitSlit.totalHeight", -0.00022103662980157044), ("ExitSlit.translationZerror", -1.1610050915913965), ("ExitSlit.rotationZerror", 0.017630379943383054), ("E1.longHalfAxisA", -37.51539174219665), ("E1.shortHalfAxisB", 0.47290954445986627), ("E1.rotationXerror", -0.3160329053034297), ("E1.rotationYerror", 1.6889366163618402), ("E1.rotationZerror", 0.8014513502752655), ("E1.translationYerror", -0.2806174446100981), ("E1.translationZerror", -0.23220537930106014), ("E2.longHalfAxisA", -0.709850015637429), ("E2.shortHalfAxisB", -0.2597385343671303), ("E2.rotationXerror", -0.05708293253550112), ("E2.rotationYerror", 1.344323627256875), ("E2.rotationZerror", 0.10824054743562761), ("E2.translationYerror", 0.03337644997345139), ("E2.translationZerror", 0.038329621347237004), ("ImagePlane.translationXerror", NumericalOutputParameter(0.23412067286167831)), ("ImagePlane.translationYerror", NumericalOutputParameter(-0.06793630544955398)), ("ImagePlane.translationZerror", NumericalOutputParameter(0.5393918432202326))]
            calculated_offset_dict = [(key, NumericalParameter(value)) if not isinstance(value, RayParameter) else (key, value) for (key, value) in calculated_offset_dict]
            calculated_offset_params = RayParameterContainer(calculated_offset_dict)
            perturbed_list = [element.clone() for element in offset_target.uncompensated_parameters]
            for entry in perturbed_list:
                entry.perturb(calculated_offset_params)
            compensated_rays_params = perturbed_list

            sinkhorn_loss_list = []
            sinkhorn_self_loss_list = []
            mse_loss_list = []
            mse_self_loss_list = []
            for _ in trange(n_examples):
                observed_rays = self.engine.run(param_containers=observed_rays_params,
                                transforms={exported_plane: self.transform for exported_plane in self.exported_planes})
                observed_rays_2 = self.engine.run(param_containers=observed_rays_params,
                                transforms={exported_plane: self.transform for exported_plane in self.exported_planes})
                compensated_rays = self.engine.run(param_containers=compensated_rays_params,
                                transforms={exported_plane: self.transform for exported_plane in self.exported_planes})
                sinkhorn_loss_list.append(torch.Tensor([SinkhornLoss().loss_fn(observed_rays[i], compensated_rays[i], 'ImagePlane') for i in range(len(observed_rays))]))
                sinkhorn_self_loss_list.append(torch.Tensor([SinkhornLoss().loss_fn(observed_rays[i], observed_rays_2[i], 'ImagePlane') for i in range(len(observed_rays))]))
                mse_loss_list.append(torch.Tensor([MSELoss().loss_fn(observed_rays[i], compensated_rays[i], 'ImagePlane') for i in range(len(observed_rays))]))
                mse_self_loss_list.append(torch.Tensor([MSELoss().loss_fn(observed_rays[i], observed_rays_2[i], 'ImagePlane') for i in range(len(observed_rays))]))
            
            sinkhorn_loss_tensor = torch.vstack(sinkhorn_loss_list).T
            sinkhorn_self_loss_tensor = torch.vstack(sinkhorn_self_loss_list).T
            mse_loss_tensor = torch.vstack(mse_loss_list).T
            mse_self_loss_tensor = torch.vstack(mse_self_loss_list).T
            torch.save(sinkhorn_loss_tensor, 'outputs/sinkhorn_loss.pt') 
            torch.save(sinkhorn_self_loss_tensor, 'outputs/sinkhorn_self_loss.pt') 
            torch.save(mse_loss_tensor, 'outputs/mse_loss.pt') 
            torch.save(mse_self_loss_tensor, 'outputs/mse_self_loss.pt') 
            for i in [0, 5, 14]:
                plt.clf()
                plt.hist(mse_loss_tensor[i])
                plt.hist(mse_self_loss_tensor[i])
                plt.savefig('outputs/loss_dist_hist_mse_'+str(i)+'.png')
                plt.clf()
                plt.hist(sinkhorn_loss_tensor[i])
                plt.hist(sinkhorn_self_loss_tensor[i])
                plt.savefig('outputs/loss_dist_hist_sinkhorn_'+str(i)+'.png')
            
            
    def test_print_bla(self):
        print("dummy")

if __name__ == '__main__':
#    unittest.main()
    RayBackendTest().setUp()
