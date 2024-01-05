import os
import sys

import numpy as np
from matplotlib import pyplot as plt

sys.path.insert(0, '../../')

from ray_tools.simulation.torch_datasets import RayDataset, extract_field

import torch
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler

# import pykeops
#
# pykeops.test_numpy_bindings()
# pykeops.test_torch_bindings()

from geomloss import SamplesLoss

# Important fix to make custom collate_fn work
# https://forums.fast.ai/t/runtimeerror-received-0-items-of-ancdata/48935
torch.multiprocessing.set_sharing_strategy('file_system')

h5_path = os.path.join('../../datasets/metrix_simulation/ray_test')

h5_files = [os.path.join(h5_path, file) for file in os.listdir(h5_path) if file.endswith('.h5')]

dataset = RayDataset(h5_files=h5_files[0:1],
                     nested_groups=False,
                     sub_groups=['1e4/params',
                                 '1e4/ray_output/ImagePlane/hist',
                                 '1e4/ray_output/ImagePlane/hist/n_rays',
                                 '1e4/ray_output/ImagePlane/raw',
                                 '1e6/ray_output/ImagePlane/hist',
                                 '1e6/ray_output/ImagePlane/ml/0',
                                 ])

weights = extract_field(dataset, '1e4/ray_output/ImagePlane/hist/n_rays')

sampler = iter(WeightedRandomSampler(weights=weights,
                                     num_samples=len(dataset),
                                     replacement=True))

idx1 = next(sampler)
idx2 = next(sampler)


def plot_data(data, weights=None):
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], s=2.0, c=weights)
    plt.show()


s1_raw = torch.tensor(np.c_[
                          dataset[idx1]['1e4/ray_output/ImagePlane/raw']['x_loc'],
                          dataset[idx1]['1e4/ray_output/ImagePlane/raw']['y_loc']])
s1_img = torch.tensor(dataset[idx1]['1e6/ray_output/ImagePlane/hist']['histogram'])
s1_x_lims = torch.tensor(dataset[idx1]['1e6/ray_output/ImagePlane/hist']['x_lims'])
s1_y_lims = torch.tensor(dataset[idx1]['1e6/ray_output/ImagePlane/hist']['y_lims'])

pooler = torch.nn.AvgPool2d(kernel_size=8, divisor_override=1)
s1_img_small = pooler(
    torch.tensor(dataset[idx1]['1e6/ray_output/ImagePlane/ml/0']['histogram']).unsqueeze(0)).squeeze(0)
s1_x_lims_small = torch.tensor(dataset[idx1]['1e6/ray_output/ImagePlane/ml/0']['x_lims'])
s1_y_lims_small = torch.tensor(dataset[idx1]['1e6/ray_output/ImagePlane/ml/0']['y_lims'])

s2_raw = torch.tensor(np.c_[
                          dataset[idx2]['1e4/ray_output/ImagePlane/raw']['x_loc'],
                          dataset[idx2]['1e4/ray_output/ImagePlane/raw']['y_loc']])
s2_img = torch.tensor(dataset[idx2]['1e6/ray_output/ImagePlane/hist']['histogram'])
s2_x_lims = torch.tensor(dataset[idx2]['1e6/ray_output/ImagePlane/hist']['x_lims'])
s2_y_lims = torch.tensor(dataset[idx2]['1e6/ray_output/ImagePlane/hist']['y_lims'])

# plt.figure(figsize=(10, 10))
# plt.imshow(s1_img_small.T, cmap='Greys')
# plt.show()

plot_data(s1_raw)
plot_data(s2_raw)

loss = SamplesLoss("sinkhorn", p=2, backend='auto')


def hist_to_pc(hist: torch.Tensor, x_lims: tuple[float, float], y_lims: tuple[float, float]):
    dim_x, dim_y = hist.shape
    coord_x_width = (x_lims[1] - x_lims[0]) / dim_x
    coord_y_width = (y_lims[1] - y_lims[0]) / dim_y
    coord_x = torch.linspace(x_lims[0] + coord_x_width / 2., x_lims[1] - coord_x_width / 2., dim_x)
    coord_y = torch.linspace(y_lims[0] + coord_y_width / 2., y_lims[1] - coord_y_width / 2., dim_y)
    grid_x, grid_y = torch.meshgrid(coord_x, coord_y, indexing='ij')
    grid_x = grid_x.to(hist.device) * torch.clamp(hist, 0.0, 1.0)
    grid_y = grid_y.to(hist.device) * torch.clamp(hist, 0.0, 1.0)
    pc_x = grid_x[hist != 0].flatten()
    pc_y = grid_y[hist != 0].flatten()
    pc_weights = hist[hist != 0].flatten()
    return torch.stack([pc_x, pc_y], dim=1), pc_weights


s1_raw_recov, s1_weights = hist_to_pc(s1_img.cuda(), s1_x_lims, s1_y_lims)
plot_data(s1_raw_recov.detach().cpu(), weights=s1_weights.detach().cpu())
s1_small, s1_weights_small = hist_to_pc(s1_img_small.cuda(), s1_x_lims_small, s1_y_lims_small)
plot_data(s1_small.detach().cpu(), weights=s1_weights_small.detach().cpu())

s2_raw_recov, s2_weights = hist_to_pc(s2_img.cuda(), s2_x_lims, s2_y_lims)
plot_data(s2_raw_recov.detach().cpu(), weights=s2_weights.detach().cpu())

s1_com = s1_raw_recov.sum(dim=0) / s1_raw_recov.shape[0]
s2_com = s2_raw_recov.sum(dim=0) / s2_raw_recov.shape[0]

# s1_weights_small = torch.zeros_like(s1_weights_small)
# s1_weights_small[0] = 1.0

print(loss(s1_raw, s2_raw))
print(loss(s1_raw_recov, s2_raw_recov))
print(loss(s1_weights / s1_weights.sum(), s1_raw_recov, s2_weights / s2_weights.sum(), s2_raw_recov))
print(loss(s1_weights / s1_weights.sum(), s1_raw_recov, s1_weights_small / s1_weights_small.sum(), s1_small))
print(loss(s1_weights / s1_weights.sum(), s1_raw_recov,
           torch.ones(len(s1_raw), dtype=torch.float64).cuda() / len(s1_raw), s1_raw.cuda()))
print(loss(s1_raw_recov, (s2_raw_recov - s2_com + s1_com)))


def sample_from_hist(hist: torch.Tensor, x_lims: tuple[float, float], y_lims: tuple[float, float], n_samples: int):
    coords, weights = hist_to_pc(hist, x_lims, y_lims)
    return F.gumbel_softmax(weights.cuda().unsqueeze(0).expand(n_samples, -1), tau=1, hard=True) @ coords

# s1_resample = sample_from_hist(s1_img.cuda(), s1_x_lims, s1_y_lims, n_samples=1000)
# s2_resample = sample_from_hist(s2_img.cuda(), s2_x_lims, s2_y_lims, n_samples=1000)
#
# plot_data(s1_resample.cpu())
# plot_data(s2_resample.cpu())
