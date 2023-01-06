import os
import sys
from typing import Tuple

sys.path.insert(0, '../../')

from tqdm import tqdm

import numpy as np
import torch
from torch import nn
from torch.utils.data import WeightedRandomSampler, DataLoader

from ray_tools.simulation.torch_datasets import RayDataset, extract_field
from ray_nn.utils.ray_processing import HistSubsampler, HistToPointCloud
from ray_nn.metrics.geometric import SinkhornLoss

# Important fix to make custom collate_fn work
# https://forums.fast.ai/t/runtimeerror-received-0-items-of-ancdata/48935
torch.multiprocessing.set_sharing_strategy('file_system')

torch.manual_seed(0)  # keep the random seed of torch fixed
np.random.seed(0)  # keep the random seed of numpy fixed

h5_path = os.path.join('../datasets/metrix_simulation/ray_test')
h5_files = [os.path.join(h5_path, file) for file in os.listdir(h5_path) if file.endswith('.h5')]

dataset = RayDataset(h5_files=h5_files,
                     nested_groups=False,
                     sub_groups=['1e4/ray_output/ImagePlane/ml/0',
                                 '1e6/ray_output/ImagePlane/ml/0'])

weights = extract_field(dataset, '1e6/ray_output/ImagePlane/ml/0/n_rays')

sampler = WeightedRandomSampler(weights=weights,
                                num_samples=len(dataset),
                                replacement=True)

batch_size = 100
dataloader = DataLoader(dataset,
                        sampler=sampler,
                        batch_size=batch_size,
                        num_workers=10)

device = torch.device('cuda:1')

hist_to_pc = HistToPointCloud(normalize_weights=True)
subsampler = HistSubsampler(factor=8)

tar_shift = torch.tensor([1.0, 1.0])


class Shifter(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.shift = nn.Parameter(torch.zeros(2), requires_grad=True)

    def forward(self, inp_supp) -> torch.Tensor:
        inp_supp = inp_supp + self.shift.view(1, 1, -1)
        return inp_supp


model = Shifter().to(device)
loss_func = SinkhornLoss(p=2, backend='online', reduction='mean').to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

num_epochs = 10

for epoch in range(num_epochs):
    t = tqdm(enumerate(dataloader),
             desc="epoch {} / {}".format(epoch + 1, num_epochs),
             total=len(dataloader))
    for idx, data in t:
        inp_supp, inp_weights = hist_to_pc(
            hist=subsampler(data['1e4/ray_output/ImagePlane/ml/0']['histogram'].to(device)),
            x_lims=data['1e4/ray_output/ImagePlane/ml/0']['x_lims'].to(device),
            y_lims=data['1e4/ray_output/ImagePlane/ml/0']['y_lims'].to(device))

        tar_supp, tar_weights = hist_to_pc(
            hist=subsampler(data['1e6/ray_output/ImagePlane/ml/0']['histogram'].to(device)),
            x_lims=data['1e6/ray_output/ImagePlane/ml/0']['x_lims'].to(device),
            y_lims=data['1e6/ray_output/ImagePlane/ml/0']['y_lims'].to(device))
        tar_supp = tar_supp + tar_shift.to(device)

        optimizer.zero_grad()

        out_supp = model(inp_supp)
        loss = loss_func(out_supp, tar_supp, inp_weights, tar_weights)

        loss.backward()
        optimizer.step()

        t.set_postfix(loss=loss.item(), shift_x=model.shift[0].item(), shift_y=model.shift[1].item())
