import os
import sys

from matplotlib import pyplot as plt

sys.path.insert(0, '../')

from ray_tools.simulation.torch_datasets import RayDataset, extract_field
from ray_nn.utils.ray_processing import HistSubsampler, HistToPointCloud
from ray_nn.metrics.geometric import SinkhornLoss

import torch
from torch.utils.data import WeightedRandomSampler, DataLoader

# Important fix to make custom collate_fn work
# https://forums.fast.ai/t/runtimeerror-received-0-items-of-ancdata/48935
torch.multiprocessing.set_sharing_strategy('file_system')

h5_path = os.path.join('../datasets/metrix_simulation/ray_test')

h5_files = [os.path.join(h5_path, file) for file in os.listdir(h5_path) if file.endswith('.h5')]

dataset = RayDataset(h5_files=h5_files,
                     nested_groups=False,
                     sub_groups=['1e4/params',
                                 '1e4/ray_output/ImagePlane/hist',
                                 '1e4/ray_output/ImagePlane/hist/n_rays',
                                 '1e6/ray_output/ImagePlane/hist',
                                 ])

weights = extract_field(dataset, '1e4/ray_output/ImagePlane/hist/n_rays')

sampler = WeightedRandomSampler(weights=weights,
                                num_samples=len(dataset),
                                replacement=True)

batch_size = 100
dataloader = DataLoader(dataset,
                        sampler=sampler,
                        batch_size=batch_size,
                        num_workers=10)

item = next(iter(dataloader))


def plot_data(data, weights=None):
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], s=2.0, c=weights)
    plt.show()


sample_hist = item['1e6/ray_output/ImagePlane/hist']['histogram'].cuda()
sample_x_lims = item['1e6/ray_output/ImagePlane/hist']['x_lims'].cuda()
sample_y_lims = item['1e6/ray_output/ImagePlane/hist']['y_lims'].cuda()

sample_supp, sample_weights = HistToPointCloud()(
    hist=sample_hist,
    x_lims=sample_x_lims,
    y_lims=sample_y_lims)

sample_weights = sample_weights / sample_weights.sum(dim=1, keepdim=True)

sample_supp_ = [sample_supp[idx, sample_weights[idx, ...] != 0] for idx in range(batch_size)]
sample_weights_ = [sample_weights[idx, sample_weights[idx, ...] != 0] for idx in range(batch_size)]

sample_supp_small, sample_weights_small = HistToPointCloud()(
    hist=HistSubsampler(factor=32)(sample_hist),
    x_lims=sample_x_lims,
    y_lims=sample_y_lims)

sample_weights_small = sample_weights_small / sample_weights_small.sum(dim=1, keepdim=True)

sample_supp_small_ = [sample_supp_small[idx, sample_weights_small[idx, ...] != 0] for idx in range(batch_size)]
sample_weights_small_ = [sample_weights_small[idx, sample_weights_small[idx, ...] != 0] for idx in range(batch_size)]

sample_supp_1e4, sample_weights_1e4 = HistToPointCloud()(
    hist=HistSubsampler(factor=32)(item['1e4/ray_output/ImagePlane/hist']['histogram'].cuda()),
    x_lims=item['1e4/ray_output/ImagePlane/hist']['x_lims'].cuda(),
    y_lims=item['1e4/ray_output/ImagePlane/hist']['y_lims'].cuda())

sample_weights_1e4 = sample_weights_1e4 / sample_weights_1e4.sum(dim=1, keepdim=True)

sample_supp_1e4_ = [sample_supp_1e4[idx, sample_weights_1e4[idx, ...] != 0] for idx in range(batch_size)]
sample_weights_1e4_ = [sample_weights_1e4[idx, sample_weights_1e4[idx, ...] != 0] for idx in range(batch_size)]

# ------------

plot_data(sample_supp_[0].detach().cpu(), weights=sample_weights_[0].detach().cpu())
plot_data(sample_supp_small_[0].detach().cpu(), weights=sample_weights_small_[0].detach().cpu())
plot_data(sample_supp_1e4_[0].detach().cpu(), weights=sample_weights_1e4_[0].detach().cpu())

plot_data(sample_supp_[1].detach().cpu(), weights=sample_weights_[1].detach().cpu())
plot_data(sample_supp_small_[1].detach().cpu(), weights=sample_weights_small_[1].detach().cpu())

# ------------

loss = SinkhornLoss(p=2, backend='online', reduction=None)

sample_com = torch.unsqueeze(torch.stack([
    sample_supp_small_[idx].sum(dim=0) / sample_supp_small_[idx].shape[0] for idx in range(batch_size)]), dim=1)

print(loss(sample_supp_small,
           sample_supp_small,
           sample_weights_small,
           sample_weights_small))

print(loss(sample_supp_small,
           sample_supp_1e4,
           sample_weights_small,
           sample_weights_1e4))

# for idx in range(len(sample_supp)):
#     print(loss(sample_supp[idx, ...],
#                sample_supp_small[idx, ...],
#                sample_weights[idx, ...],
#                sample_weights_small[idx, ...]))
#     print(loss(sample_supp_[idx],
#                sample_supp_small_[idx],
#                sample_weights_[idx],
#                sample_weights_small_[idx]))

print(loss(sample_supp_small[0:1, ...].expand(batch_size, -1, -1).clone(),
           sample_supp_small,
           sample_weights_small[0:1, ...].expand(batch_size, -1).clone(),
           sample_weights_small))

sample_supp_small = sample_supp_small - sample_com

print(loss(sample_supp_small[0:1, ...].expand(batch_size, -1, -1).clone(),
           sample_supp_small,
           sample_weights_small[0:1, ...].expand(batch_size, -1).clone(),
           sample_weights_small))
