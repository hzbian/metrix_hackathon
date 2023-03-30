import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image, ImageChops
from tqdm import tqdm

from ray_nn.utils.ray_processing import HistToPointCloud
from ray_tools.base.backend import RayOutput
from ray_tools.base.parameter import RayParameterContainer, MutableParameter


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
        repetitions = (pc_weights * rays_per_weights).floor().int()[0]
        residuum = (pc_weights * rays_per_weights)[0] - repetitions
        _, ordered_indices = residuum.sort()
        out = torch.repeat_interleave(hist, repetitions, dim=1)
        still_required = num_rays - out.shape[1]
        added_from_hist = hist[:, ordered_indices[:still_required]]
        out = torch.hstack((out, added_from_hist))
        if out.shape[1] != num_rays:
            raise Exception("The amount of rays is %i but should be %i." % (out.shape[1], num_rays))
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


def import_data(real_data_dir, included_z_layers, param_container):
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
        output_dict = {'param_container_dict': pandas_to_param_container(parameters[measurement_name], param_container)}
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
                ray_output = RayOutput(x_loc=scatter[0, :, 0].float().numpy(), y_loc=scatter[0, :, 1].float().numpy(),
                                       z_loc=np.array([], dtype=float), y_dir=np.array([], dtype=float),
                                       x_dir=np.array([], dtype=float), z_dir=np.array([], dtype=float), energy=np.array([], dtype=float))
                z_layer_id = file[:-4]
                if int(z_layer_id) in included_z_layers:
                    z_direction_dict[z_layer_id] = ray_output
        for key in included_z_layers:
            if str(key) not in z_direction_dict.keys():
                raise Exception("Layer %s missing in measurement %s." % (key, measurement_name))
        output_dict['ray_output'] = {'ImagePlane': z_direction_dict}
        output_list.append(output_dict)
    return output_list
