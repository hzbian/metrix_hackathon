import os
import sys
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt

sys.path.insert(0, '../')

from ray_tools.simulation.torch_datasets import RayDataset, extract_field

import torch
from torch.utils.data import WeightedRandomSampler

# import pykeops
#
# pykeops.test_numpy_bindings()
# pykeops.test_torch_bindings()

from geomloss import SamplesLoss


# Important fix to make custom collate_fn work
# https://forums.fast.ai/t/runtimeerror-received-0-items-of-ancdata/48935
torch.multiprocessing.set_sharing_strategy('file_system')

h5_path = os.path.join('../datasets/metrix_simulation/ray_test')

h5_files = [os.path.join(h5_path, file) for file in os.listdir(h5_path) if file.endswith('.h5')]

dataset = RayDataset(h5_files=h5_files[0:1],
                     nested_groups=False,
                     sub_groups=['1e4/params',
                                 '1e4/ray_output/ImagePlane/hist',
                                 '1e4/ray_output/ImagePlane/hist/n_rays',
                                 '1e4/ray_output/ImagePlane/raw',
                                 '1e6/ray_output/ImagePlane/hist',
                                 ])

weights = extract_field(dataset, '1e4/ray_output/ImagePlane/hist/n_rays')

sampler = iter(WeightedRandomSampler(weights=weights,
                                     num_samples=len(dataset),
                                     replacement=True))

idx1 = next(sampler)
idx2 = next(sampler)


def plot_data(data):
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], s=0.1)
    plt.show()


s1_raw = torch.tensor(np.c_[
                          dataset[idx1]['1e4/ray_output/ImagePlane/raw']['x_loc'],
                          dataset[idx1]['1e4/ray_output/ImagePlane/raw']['y_loc']])
s1_img = torch.tensor(dataset[idx1]['1e6/ray_output/ImagePlane/hist']['histogram'])
s1_x_lims = torch.tensor(dataset[idx1]['1e6/ray_output/ImagePlane/hist']['x_lims'])
s1_y_lims = torch.tensor(dataset[idx1]['1e6/ray_output/ImagePlane/hist']['y_lims'])

s2_raw = torch.tensor(np.c_[
                          dataset[idx2]['1e4/ray_output/ImagePlane/raw']['x_loc'],
                          dataset[idx2]['1e4/ray_output/ImagePlane/raw']['y_loc']])
s2_img = torch.tensor(dataset[idx2]['1e6/ray_output/ImagePlane/hist']['histogram'])
s2_x_lims = torch.tensor(dataset[idx2]['1e6/ray_output/ImagePlane/hist']['x_lims'])
s2_y_lims = torch.tensor(dataset[idx2]['1e6/ray_output/ImagePlane/hist']['y_lims'])

# plt.figure(figsize=(10, 10))
# plt.imshow(s1_img, cmap='Greys', vmin=0.0, vmax=1.0)
# plt.show()

plot_data(s1_raw)
plot_data(s2_raw)

loss = SamplesLoss("sinkhorn", p=2, backend='auto')


def hist_to_pc(hist: torch.Tensor, x_lims: Tuple[float, float], y_lims: Tuple[float, float]):
    dim_x, dim_y = hist.shape
    coord_x_width = (x_lims[1] - x_lims[0]) / dim_x
    coord_y_width = (y_lims[1] - y_lims[0]) / dim_y
    coord_x = torch.linspace(x_lims[0] + coord_x_width / 2., x_lims[1] - coord_x_width / 2., dim_x)
    coord_y = torch.linspace(y_lims[0] + coord_y_width / 2., y_lims[1] - coord_y_width / 2., dim_y)
    grid_x, grid_y = torch.meshgrid(coord_x, coord_y, indexing='ij')
    grid_x = grid_x * torch.clamp(hist, 0.0, 1.0)
    grid_y = grid_y * torch.clamp(hist, 0.0, 1.0)
    pc_x = grid_x[hist != 0].flatten()
    pc_y = grid_y[hist != 0].flatten()
    pc_weights = hist[hist != 0].flatten
    return torch.stack([pc_x, pc_y], dim=1), pc_weights


s1_raw_recov, s1_weights = hist_to_pc(s1_img, s1_x_lims, s1_y_lims)
plot_data(s1_raw_recov.detach())
s2_raw_recov, s2_weights = hist_to_pc(s2_img, s2_x_lims, s2_y_lims)
plot_data(s2_raw_recov.detach())

print(loss(s1_raw, s2_raw))
print(loss(s1_raw_recov.cuda(), s2_raw_recov.cuda()))
