import os
import sys

sys.path.insert(0, '../../')

from tqdm import tqdm

from ray_tools.simulation.torch_datasets import RayDataset, extract_field

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

# Important fix to make custom collate_fn work
# https://forums.fast.ai/t/runtimeerror-received-0-items-of-ancdata/48935
torch.multiprocessing.set_sharing_strategy('file_system')

h5_path = os.path.join('../../datasets/metrix_simulation/ray_enhance_v2')

h5_files = [os.path.join(h5_path, file) for file in os.listdir(h5_path) if file.endswith('.h5')]

dataset = RayDataset(h5_files=h5_files,
                     nested_groups=False,
                     sub_groups=['1e6/params',
                                 '1e6/ray_output/ImagePlane/ml/0/n_rays',
                                 '1e6/ray_output/ImagePlane/ml/0'])

weights = extract_field(dataset, '1e6/ray_output/ImagePlane/ml/0/n_rays')

batch_size = 100
data_loader = DataLoader(dataset,
                         sampler=WeightedRandomSampler(weights=weights,
                                                       num_samples=len(dataset),
                                                       replacement=True),
                         batch_size=batch_size,
                         num_workers=100)

for idx, item in tqdm(enumerate(data_loader)):
    print(item['1e6/ray_output/ImagePlane/ml/0/n_rays'])
