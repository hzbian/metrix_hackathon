import os
from typing import List, OrderedDict, Tuple

import pandas as pd
import torch
import torchvision
from PIL import Image, ImageChops
from tqdm import tqdm

from ray_nn.utils.ray_processing import HistToPointCloud
from ray_tools.base.backend import RayOutput
from ray_tools.base.parameter import RayParameterContainer, MutableParameter, OutputParameter


class SampleGridWeightedHist(torch.nn.Module):
    """
    Converts a histogram into a point cloud.
    Output is a tensor of shape [batch size, #pixels in hist, 2];
    first dimension are x-coordinates and second are y-coordinates.
    The ray coordinates are computed by a meshgrid according to the size of hist and given limits.
    Each ray is endowed with a weights, which is the corresponding entry of the histogram.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, hist: torch.Tensor, pc_weights: torch.Tensor, num_rays: int, fill=True) -> Tuple[
        torch.Tensor, ...]:
        rays_per_weights = num_rays / pc_weights.sum(dim=1)
        repetitions = (pc_weights * rays_per_weights).floor().int()[0]
        residuum = (pc_weights * rays_per_weights)[0] - repetitions
        _, ordered_indices = residuum.sort(descending=True)
        out = torch.repeat_interleave(hist, repetitions, dim=1)
        if fill:
            still_required = num_rays - out.shape[1]
            added_from_hist = hist[:, ordered_indices[:still_required]]
            out = torch.hstack((out, added_from_hist))
            if out.shape[1] != num_rays:
                raise Exception("The amount of rays is %i but should be %i." % (out.shape[1], num_rays))
        return out


class SampleRandomWeightedHist(torch.nn.Module):
    """
    Converts a histogram into a point cloud.
    Output is a tensor of shape [batch size, #pixels in hist, 2];
    first dimension are x-coordinates and second are y-coordinates.
    The ray coordinates are computed by a meshgrid according to the size of hist and given limits.
    Each ray is endowed with a weights, which is the corresponding entry of the histogram.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, hist: torch.Tensor, pc_weights: torch.Tensor, num_rays: int) -> torch.Tensor:
        pdf = pc_weights / pc_weights.sum()
        random_indices = torch.multinomial(pdf.flatten(), num_rays, replacement=True)
        random_points = hist[:, random_indices] 
        return random_points

def pandas_to_param_container(input_pd, param_container: RayParameterContainer, check_value_lims: bool = True):
    output_param_container = param_container.clone()
    for key, entry in output_param_container.items():
        if isinstance(entry, MutableParameter):
            if key not in input_pd.keys():
                raise Exception("Could not find an entry for the defined parameter %s in the parameter file." % key)
            value = input_pd[key]
            if check_value_lims:
                if value < entry.value_lims[0] or value > entry.value_lims[1]:
                    raise Exception(
                        "The provided value %.2f for %s is not within its defined range [%.2f, %.2f] of the ray"
                        "parameter container." % (value, key, entry.value_lims[0], entry.value_lims[1]))
            entry.value = value

    return output_param_container


def import_data(real_data_dir, imported_measurements, included_z_layers: list[float], param_container=None, check_value_lims=True, mm_per_pixel: float = 1.6 / 1000.):
    parameters = pd.read_csv(os.path.join(real_data_dir, 'parameters.csv'), index_col=0)
    x_dilation = parameters.T['ImagePlane.translationXerror']  # TODO check if this is added twice now
    y_dilation = parameters.T['ImagePlane.translationYerror']
    black = get_image(os.path.join(real_data_dir, 'black.bmp'))
    transform = HistToPointCloud()
    transform_weight = SampleRandomWeightedHist()
    output_list = []
    for subdir, _, files in tqdm(os.walk(real_data_dir)):
        measurement_name = os.path.basename(os.path.normpath(subdir))[:3]
        if measurement_name not in imported_measurements:
            continue
        if measurement_name not in parameters.keys():
            continue
        if param_container is not None:
            output_dict = {'param_container_dict': pandas_to_param_container(parameters[measurement_name], param_container,
                                                                          check_value_lims=check_value_lims)}
        else:
            output_dict = {}
        z_direction_dict = {}
        for file in files:
            if file.lower().endswith('.bmp') and not file.lower().endswith('black.bmp'):
                path: str = os.path.join(subdir, file)
                sample_image: Image.Image = get_image(path)
                sample_image: Image.Image = subtract_black(sample_image, black)
                sample: torch.Tensor = to_tensor(sample_image)
                x_lims: torch.Tensor = get_lims(x_dilation[measurement_name], sample.shape[2], mm_per_pixel)
                y_lims: torch.Tensor = get_lims(y_dilation[measurement_name], sample.shape[1], mm_per_pixel)
                grid, intensity = transform(sample, x_lims, y_lims)
                cleaned_intensity, cleaned_indices = clean_intensity(intensity[0], threshold=0.02)
                scatter = transform_weight(hist=grid[:, cleaned_indices], pc_weights=cleaned_intensity.unsqueeze(0),
                                           num_rays=1000)
                ray_output: RayOutput = tensor_to_ray_output(scatter)
                z_layer_id = float(file[:-4])
                if z_layer_id in included_z_layers:
                    z_direction_dict[str(z_layer_id)] = ray_output
        z_direction_ordered = []
        for key in included_z_layers:
            if str(key) not in z_direction_dict.keys():
                raise Exception("Layer %s missing in measurement %s." % (key, measurement_name))
            else:
                z_direction_ordered.append((key, z_direction_dict[str(key)]))
        z_direction_ordered_dict = OrderedDict(z_direction_ordered)    
        output_dict['ray_output'] = {'ImagePlane': z_direction_ordered_dict}
        output_list.append(output_dict)
    return output_list

def get_image(path: str) -> Image.Image:
    return Image.open(path)

def subtract_black(image: Image.Image, black: Image.Image) -> Image.Image:
    return ImageChops.subtract(image, black)

def to_tensor(image) -> torch.Tensor:
    return torchvision.transforms.ToTensor()(image)

def read_parameter_csv(real_data_dir: str, csv_name: str = 'parameter.csv') -> pd.DataFrame:
    return pd.read_csv(os.path.join(real_data_dir, csv_name), index_col=0)

def get_lims(dimension_dilation_mm: float, dimension_size_pixel: int, mm_per_pixel: float) -> torch.Tensor:
    return torch.tensor(
                    (dimension_dilation_mm, dimension_dilation_mm + dimension_size_pixel * mm_per_pixel)).unsqueeze(0)

def clean_intensity(intensity: torch.Tensor, threshold: float) -> Tuple[torch.Tensor, torch.Tensor]:
                cleaned_indices = intensity > threshold
                cleaned_intensity = intensity[cleaned_indices]
                return cleaned_intensity, cleaned_indices

def tensor_to_ray_output(scatter: torch.Tensor, index: int = 0) -> RayOutput:
    ray_output = RayOutput(x_loc=scatter[index, :, 0].float(), y_loc=scatter[index, :, 1].float(),
                                       z_loc=torch.tensor([], dtype=torch.float32, device=scatter.device),
                                       y_dir=torch.tensor([], dtype=torch.float32, device=scatter.device),
                                       x_dir=torch.tensor([], dtype=torch.float32, device=scatter.device),
                                       z_dir=torch.tensor([], dtype=torch.float32, device=scatter.device),
                                       energy=torch.tensor([], dtype=torch.float32, device=scatter.device))
    return ray_output