import os
import sys
from typing import Tuple

sys.path.insert(0, '../../')

from tqdm import trange

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

h5_path = os.path.join('../../datasets/metrix_simulation/ray_test')
h5_files = [os.path.join(h5_path, file) for file in os.listdir(h5_path) if file.endswith('.h5')]

dataset = RayDataset(h5_files=h5_files,
                     nested_groups=False,
                     sub_groups=['1e6/ray_output/ImagePlane/ml/0'])

weights = extract_field(dataset, '1e6/ray_output/ImagePlane/ml/0/n_rays')

sampler = WeightedRandomSampler(weights=weights,
                                num_samples=len(dataset),
                                replacement=True)

batch_size = 100
dataloader = DataLoader(dataset,
                        sampler=sampler,
                        batch_size=batch_size,
                        num_workers=10)

data = next(iter(dataloader))

target_supp, target_weights = HistToPointCloud()(
    hist=HistSubsampler(factor=8)(data['1e6/ray_output/ImagePlane/ml/0']['histogram'].cuda()),
    x_lims=data['1e6/ray_output/ImagePlane/ml/0']['x_lims'].cuda(),
    y_lims=data['1e6/ray_output/ImagePlane/ml/0']['y_lims'].cuda())

target_shift = torch.tensor([1.0, 1.0]).view(1, 1, -1).cuda()
target_supp = target_supp + target_shift


class Shifter(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.shift = nn.Parameter(torch.zeros(1, 1, 2), requires_grad=True)
        self._hist_to_pc = HistToPointCloud()
        self._subsampler = HistSubsampler(factor=8)

    def forward(self, inp) -> Tuple[torch.Tensor, torch.Tensor]:
        inp_supp, inp_weights = self._hist_to_pc(
            hist=self._subsampler(inp['1e6/ray_output/ImagePlane/ml/0']['histogram'].cuda()),
            x_lims=inp['1e6/ray_output/ImagePlane/ml/0']['x_lims'].cuda(),
            y_lims=inp['1e6/ray_output/ImagePlane/ml/0']['y_lims'].cuda())

        inp_supp = inp_supp + self.shift

        return inp_supp, inp_weights


model = Shifter().cuda()
loss_func = SinkhornLoss(p=2, normalize_weights=True, backend='online', reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

n_iterations = 1000

t = trange(n_iterations)

for _ in t:
    running_loss = 0

    optimizer.zero_grad()

    out_supp, out_weights = model(data)
    loss = loss_func(out_supp, target_supp, out_weights, target_weights)

    loss.backward()
    optimizer.step()

    with torch.no_grad():
        t.set_postfix(loss=loss.item(), shift_x=model.shift[0, 0, 0].item(), shift_y=model.shift[0, 0, 1].item())
