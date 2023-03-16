import os
from typing import Tuple

import pandas as pd
import torch
import numpy as np
import torchvision
from PIL import Image, ImageChops
from tqdm import tqdm
import matplotlib.pyplot as plt

from ray_nn.utils.ray_processing import HistToPointCloud
from ray_tools.base.parameter import RayParameterContainer, NumericalParameter, RandomParameter, MutableParameter
from ray_tools.base.utils import RandomGenerator
from ray_tools.base.backend import RayOutput

root_dir = '../../datasets/metrix_real_data/2021_march_selected'

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

def import_data(root_dir):
    transform = HistToPointCloud()
    transform_weight = SampleWeightedHist()
    parameters = pd.read_csv(os.path.join(root_dir, 'parameters.csv'), index_col=0)
    xy_dilation = pd.read_csv(os.path.join(root_dir, 'xy_dilation_mm.csv'))
    black = Image.open(os.path.join(root_dir, 'black.bmp'))
    for subdir, dirs, files in tqdm(os.walk(root_dir)):
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
                # plt.imshow(sample[0])
                z_id = file[:-4]
                sample_xy_dilation = xy_dilation[measurement_name]
                x_lims = torch.tensor((sample_xy_dilation[0], sample_xy_dilation[0] + 768 * 1.6 / 1000)).unsqueeze(0)
                y_lims = torch.tensor((sample_xy_dilation[1], sample_xy_dilation[1] + 576 * 1.6 / 1000)).unsqueeze(0)
                image, intensity = transform(sample, x_lims, y_lims)
                # plt.scatter(x=image[0,:,0], y=image[0,:,1], c=intensity[0])
                #            plt.scatter(x=image[0,:,0], y=image[0,:,1], c=intensity[0].ceil())
                cleaned_indices = intensity[0] > 0.02
                cleaned_intensity = intensity[0][cleaned_indices]

                # plt.scatter(x=image[0,:,0][cleaned_indices], y=image[0,:,1][cleaned_indices], c=cleaned_intensity)
                # plt.show()
                scatter = transform_weight(hist=image[:, cleaned_indices], pc_weights=cleaned_intensity.unsqueeze(0),
                                           num_rays=1000)
                ray_output = RayOutput(x_loc=scatter[0, :, 0].numpy(), y_loc=scatter[0, :, 1].numpy(), z_loc=np.array([]), y_dir=np.array([]),
                                       x_dir=np.array([]), z_dir=np.array([]), energy=np.array([]))
                z_direction_dict[file[:-4]] = ray_output
                # plt.scatter(x=scatter[0, :, 0], y=scatter[0, :, 1])
                # plt.show()
                ##print(sample_parameters)

        output_dict['ray_output'] = {'ImagePlane': z_direction_dict}


output_dict = import_data(root_dir)
offset_optimization_target = OffsetOptimizationTarget(target_rays=offset_target_rays,
                                                      target_offset=offset,
                                                      search_space=offset_search_space(),
                                                      perturbed_parameters=target_parameters)


ray_optimizer.optimize(optimization_target=offset_optimization_target, iterations=1000)