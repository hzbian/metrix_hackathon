import os
from typing import Tuple

import optuna
import pandas as pd
import torch
import numpy as np
import torchvision
from PIL import Image, ImageChops
import wandb
from optuna.samplers import TPESampler
from tqdm import tqdm
import matplotlib.pyplot as plt

from ray_nn.metrics.geometric import SinkhornLoss
from ray_nn.utils.ray_processing import HistToPointCloud
from ray_optim.ray_optimizer import OffsetOptimizationTarget, OptimizerBackendOptuna, RayOptimizer, WandbLoggingBackend
from ray_tools.base.engine import RayEngine
from ray_tools.base.parameter import RayParameterContainer, NumericalParameter, RandomParameter, MutableParameter
from ray_tools.base.transform import MultiLayer
from ray_tools.base.utils import RandomGenerator
from ray_tools.base.backend import RayOutput, RayBackendDockerRAYUI

real_data_dir = '../../datasets/metrix_real_data/2021_march_selected'

rg = RandomGenerator(seed=42)

param_func = lambda: RayParameterContainer([
    ("U41_318eV.numberRays", NumericalParameter(value=1e4)),
    ("U41_318eV.translationXerror", RandomParameter(value_lims=(-0.25, 0.25), rg=rg)),
    ("U41_318eV.translationYerror", RandomParameter(value_lims=(-0.25, 0.25), rg=rg)),
    ("U41_318eV.rotationXerror", RandomParameter(value_lims=(-0.05, 0.05), rg=rg)),
    ("U41_318eV.rotationYerror", RandomParameter(value_lims=(-0.05, 0.05), rg=rg)),
    ("ASBL.totalWidth", RandomParameter(value_lims=(1.9, 2.1), rg=rg)),
    ("ASBL.totalHeight", RandomParameter(value_lims=(0.9, 1.1), rg=rg)),
    ("ASBL.translationXerror", RandomParameter(value_lims=(-0.2, 0.2), rg=rg)),
    ("ASBL.translationYerror", RandomParameter(value_lims=(-0.2, 0.2), rg=rg)),
    ("M1_Cylinder.radius", RandomParameter(value_lims=(174.06, 174.36), rg=rg)),
    ("M1_Cylinder.rotationXerror", RandomParameter(value_lims=(-0.25, 0.25), rg=rg)),
    ("M1_Cylinder.rotationYerror", RandomParameter(value_lims=(-1., 1.), rg=rg)),
    ("M1_Cylinder.rotationZerror", RandomParameter(value_lims=(-1., 1.), rg=rg)),
    ("M1_Cylinder.translationXerror", RandomParameter(value_lims=(-1., 1.), rg=rg)),
    ("M1_Cylinder.translationYerror", RandomParameter(value_lims=(-1., 1.), rg=rg)),
    ("SphericalGrating.radius", RandomParameter(value_lims=(109741., 109841.), rg=rg)),
    ("SphericalGrating.rotationYerror", RandomParameter(value_lims=(-1., 1.), rg=rg)),
    ("SphericalGrating.rotationZerror", RandomParameter(value_lims=(-2.5, 2.5), rg=rg)),
    ("ExitSlit.totalHeight", RandomParameter(value_lims=(0.009, 0.011), rg=rg)),
    ("ExitSlit.translationZerror", RandomParameter(value_lims=(-31., 31.), rg=rg)),
    ("ExitSlit.rotationZerror", RandomParameter(value_lims=(-0.3, 0.3), rg=rg)),
    ("E1.longHalfAxisA", RandomParameter(value_lims=(20600., 20900.), rg=rg)),
    ("E1.shortHalfAxisB", RandomParameter(value_lims=(300.721702601, 304.721702601), rg=rg)),
    ("E1.rotationXerror", RandomParameter(value_lims=(-0.5, 0.5), rg=rg)),
    ("E1.rotationYerror", RandomParameter(value_lims=(-7.5, 7.5), rg=rg)),
    ("E1.rotationZerror", RandomParameter(value_lims=(-4, 4), rg=rg)),
    ("E1.translationYerror", RandomParameter(value_lims=(-1, 1), rg=rg)),
    ("E1.translationZerror", RandomParameter(value_lims=(-1, 1), rg=rg)),
    ("E2.longHalfAxisA", RandomParameter(value_lims=(4325., 4425.), rg=rg)),
    ("E2.shortHalfAxisB", RandomParameter(value_lims=(96.1560870104, 98.1560870104), rg=rg)),
    ("E2.rotationXerror", RandomParameter(value_lims=(-0.5, 0.5), rg=rg)),
    ("E2.rotationYerror", RandomParameter(value_lims=(-7.5, 7.5), rg=rg)),
    ("E2.rotationZerror", RandomParameter(value_lims=(-4, 4), rg=rg)),
    ("E2.translationYerror", RandomParameter(value_lims=(-1, 1), rg=rg)),
    ("E2.translationZerror", RandomParameter(value_lims=(-1, 1), rg=rg)),
])


class SampleWeightedHist(torch.nn.Module):
    """
    Converts a histogram into a point cloud.
    Output is a tensor of shape [batch size, #pixels in hist, 2];
    first dimension are x-coordinates and second are y-coordinates.
    The ray coordinates are computed by a meshgrid according to the size of hist and given limits.
    Each ray is endowed with a weights, which is the corresponding entry of the histogram.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, hist: torch.Tensor, pc_weights: torch.Tensor, num_rays: int) -> Tuple[torch.Tensor, ...]:
        rays_per_weights = num_rays / pc_weights.sum(dim=1)
        out = torch.repeat_interleave(hist, (pc_weights * rays_per_weights).round().int()[0], dim=1)
        return out


def pandas_to_param_container(input_pd, param_container: RayParameterContainer):
    output_param_container = param_container.clone()
    for key, entry in output_param_container.items():
        if isinstance(entry, MutableParameter):
            value = input_pd[key]
            if value < entry.value_lims[0] or value > entry.value_lims[1]:
                raise Exception("The provided value %.2f for %s is not within its defined range [%.2f, %.2f] of the ray"
                                "parameter container." % (value, key, entry.value_lims[0], entry.value_lims[1]))
            entry.value = value

    return output_param_container


def import_data(real_data_dir, included_z_layers):
    transform = HistToPointCloud()
    transform_weight = SampleWeightedHist()
    parameters = pd.read_csv(os.path.join(real_data_dir, 'parameters.csv'), index_col=0)
    xy_dilation = pd.read_csv(os.path.join(real_data_dir, 'xy_dilation_mm.csv'))
    black = Image.open(os.path.join(real_data_dir, 'black.bmp'))
    output_list = []
    for subdir, dirs, files in tqdm(os.walk(real_data_dir)):
        measurement_name = os.path.basename(os.path.normpath(subdir))[:3]
        if measurement_name not in parameters.keys():
            continue
        output_dict = {'param_container_dict': pandas_to_param_container(parameters[measurement_name], param_func())}
        z_direction_dict = {}
        for file in files:
            if file.lower().endswith('.bmp') and not file.lower().endswith('black.bmp'):
                path = os.path.join(subdir, file)
                sample = Image.open(path)
                sample = ImageChops.subtract(sample, black)
                sample = torchvision.transforms.ToTensor()(sample)
                sample_xy_dilation = xy_dilation[measurement_name]
                x_lims = torch.tensor((sample_xy_dilation[0], sample_xy_dilation[0] + 768 * 1.6 / 1000)).unsqueeze(0)
                y_lims = torch.tensor((sample_xy_dilation[1], sample_xy_dilation[1] + 576 * 1.6 / 1000)).unsqueeze(0)
                image, intensity = transform(sample, x_lims, y_lims)
                cleaned_indices = intensity[0] > 0.02
                cleaned_intensity = intensity[0][cleaned_indices]

                scatter = transform_weight(hist=image[:, cleaned_indices], pc_weights=cleaned_intensity.unsqueeze(0),
                                           num_rays=1000)
                ray_output = RayOutput(x_loc=scatter[0, :, 0].numpy(), y_loc=scatter[0, :, 1].numpy(),
                                       z_loc=np.array([]), y_dir=np.array([]),
                                       x_dir=np.array([]), z_dir=np.array([]), energy=np.array([]))
                z_layer_id = file[:-4]
                if int(z_layer_id) in included_z_layers:
                    z_direction_dict[z_layer_id] = ray_output
        for key in included_z_layers:
            if str(key) not in z_direction_dict.keys():
                raise Exception("Layer %s missing in measurement %s." % (key, measurement_name))
        output_dict['ray_output'] = {'ImagePlane': z_direction_dict}
        output_list.append(output_dict)
    return output_list


study_name = 'real_data_tpe'  # -100-startup-trials-100-ei-samples'
wandb.init(entity='hzb-aos',
           project='metrix_hackathon_offsets',
           name=study_name,
           mode='disabled',  # 'disabled' or 'online'
           )

sinkhorn_function = SinkhornLoss(normalize_weights='weights1', p=1, backend='online', reduction=None)


def sinkhorn_loss(y: torch.Tensor, y_hat: torch.Tensor):
    y = y.cuda()

    if y.shape[1] == 0 or y.shape[1] == 1:
        y = torch.ones((y.shape[0], 2, 2), device=y.device, dtype=y.dtype) * -2

    y_hat = y_hat.cuda()
    if y_hat.shape[1] == 0 or y_hat.shape[1] == 1:
        y_hat = torch.ones((y_hat.shape[0], 2, 2), device=y_hat.device, dtype=y_hat.dtype) * -1
    loss = sinkhorn_function(y.contiguous(), y_hat.contiguous(), torch.ones_like(y[..., 1]),
                             torch.ones_like(y_hat[..., 1]))
    # loss = torch.tensor((y.shape[1] - y_hat.shape[1]) ** 2 / 2)

    return loss


def multi_objective_loss(y: torch.Tensor, y_hat: torch.Tensor):
    y = y.cuda()
    y_hat = y_hat.cuda()
    ray_count_loss = (y.shape[1] - y_hat.shape[1]) ** 2 / 2
    return sinkhorn_loss(y, y_hat), ray_count_loss


root_dir = '../../'

rml_basefile = os.path.join(root_dir, 'rml_src', 'METRIX_U41_G1_H1_318eV_PS_MLearn.rml')
ray_workdir = os.path.join(root_dir, 'ray_workdir', 'optimization')

n_rays = ['1e4']
max_deviation = 0.1

exported_plane = "ImagePlane"  # "Spherical Grating"

multi_objective = False

directions = ['minimize', 'minimize'] if multi_objective else None
storage_path = None# "sqlite:////dev/shm/db.sqlite3"
sampler = TPESampler()  # n_startup_trials=100, n_ei_candidates=100) #optuna.samplers.CmaEsSampler()
optuna_study = optuna.create_study(directions=directions, sampler=sampler, pruner=optuna.pruners.HyperbandPruner(),
                                   storage=storage_path, study_name=study_name)
optimizer_backend_optuna = OptimizerBackendOptuna(optuna_study)

criterion = multi_objective_loss if multi_objective else sinkhorn_loss
included_z_layers = [-15, -10, -5, 0, 5, 10, 15, 20, 25, 30]
transforms = MultiLayer(included_z_layers, copy_directions=False)
verbose = False
engine = RayEngine(rml_basefile=rml_basefile,
                   exported_planes=[exported_plane],
                   ray_backend=RayBackendDockerRAYUI(docker_image='ray-ui-service',
                                                     ray_workdir=ray_workdir,
                                                     verbose=verbose),
                   num_workers=-1,
                   as_generator=False)

ray_optimizer = RayOptimizer(optimizer_backend=optimizer_backend_optuna, criterion=criterion, engine=engine,
                             log_times=True, exported_plane=exported_plane,
                             transforms=transforms,
                             logging_backend=WandbLoggingBackend())

real_data = import_data(real_data_dir, included_z_layers)

offset_search_space = lambda: RayParameterContainer(
    [(k, RandomParameter(
        value_lims=(
            -max_deviation * (v.value_lims[1] - v.value_lims[0]), max_deviation * (v.value_lims[1] - v.value_lims[0])),
        rg=rg)) for
     k, v in
     param_func().items() if isinstance(v, RandomParameter)]
)

perturbed_parameters = [element['param_container_dict'] for element in real_data]
target_rays_without_offset = engine.run(perturbed_parameters, transforms=transforms)

offset_optimization_target = OffsetOptimizationTarget(target_rays=real_data,
                                                      search_space=offset_search_space(),
                                                      perturbed_parameters=perturbed_parameters,
                                                      target_rays_without_offset=target_rays_without_offset)

ray_optimizer.optimize(optimization_target=offset_optimization_target, iterations=1000)
