import os
import sys

sys.path.insert(0, '../../')

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from ray_tools.simulation.torch_datasets import RayDataset

# Important fix to make custom collate_fn work
# https://forums.fast.ai/t/runtimeerror-received-0-items-of-ancdata/48935
torch.multiprocessing.set_sharing_strategy('file_system')

h5_path = os.path.join('/scratch/metrix-hackathon/datasets/metrix_simulation/ray_enhance_final')
h5_files = [os.path.join(h5_path, file) for file in os.listdir(h5_path) if file.endswith('.h5')]

dataset = RayDataset(h5_files=h5_files,
                     nested_groups=False,
                     sub_groups=['1e6/ray_output/ImagePlane/ml/0/n_rays'])

dataloader = DataLoader(dataset,
                        shuffle=False,
                        batch_size=1000,
                        num_workers=10)

weights = [item for idx, item in tqdm(enumerate(dataloader))]
weights = torch.cat([list(w.values())[0] for w in weights], dim=0)
torch.save(weights, '/scratch/metrix-hackathon/datasets/metrix_simulation/n_rays_ray_enhance_final.pt')
