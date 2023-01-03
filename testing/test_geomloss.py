import os
import sys

import numpy as np
from matplotlib import pyplot as plt

sys.path.insert(0, '../')

from ray_tools.simulation.torch_datasets import RayDataset, extract_field

import torch
from torch.utils.data import WeightedRandomSampler

from geomloss import SamplesLoss, sinkhorn_divergence

# Important fix to make custom collate_fn work
# https://forums.fast.ai/t/runtimeerror-received-0-items-of-ancdata/48935
torch.multiprocessing.set_sharing_strategy('file_system')

h5_path = os.path.join('../datasets/metrix_simulation/ray_enhance_v2')

h5_files = [os.path.join(h5_path, file) for file in os.listdir(h5_path) if file.endswith('.h5')]

dataset = RayDataset(h5_files=h5_files[0:1],
                     nested_groups=False,
                     sub_groups=['1e4/params',
                                 '1e4/ray_output/ImagePlane/hist',
                                 '1e4/ray_output/ImagePlane/hist/n_rays',
                                 '1e4/ray_output/ImagePlane/raw'
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
s1_img = torch.tensor(dataset[idx1]['1e4/ray_output/ImagePlane/hist']['histogram'])
s1_x_lims = torch.tensor(dataset[idx1]['1e4/ray_output/ImagePlane/hist']['x_lims'])
s1_y_lims = torch.tensor(dataset[idx1]['1e4/ray_output/ImagePlane/hist']['y_lims'])

s2_raw = torch.tensor(np.c_[
                          dataset[idx2]['1e4/ray_output/ImagePlane/raw']['x_loc'],
                          dataset[idx2]['1e4/ray_output/ImagePlane/raw']['y_loc']])
s2_img = torch.tensor(dataset[idx2]['1e4/ray_output/ImagePlane/hist']['histogram'])
s2_x_lims = torch.tensor(dataset[idx2]['1e4/ray_output/ImagePlane/hist']['x_lims'])
s2_y_lims = torch.tensor(dataset[idx2]['1e4/ray_output/ImagePlane/hist']['y_lims'])

plot_data(s1_raw)
plot_data(s2_raw)

loss = SamplesLoss("sinkhorn", p=2)

print(loss(s1_raw, s2_raw))
# print(sinkhorn_divergence)
