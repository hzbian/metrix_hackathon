from tqdm import tqdm
import os

import torch
from torch.utils.data import DataLoader

from ray_tools.simulation.torch_datasets import RandomDataset

# Important fix to make custom collate_fn work
# https://forums.fast.ai/t/runtimeerror-received-0-items-of-ancdata/48935
torch.multiprocessing.set_sharing_strategy('file_system')

h5_path = os.path.join('../datasets/metrix_simulation')

h5_files = [os.path.join(h5_path, file) for file in os.listdir(h5_path) if file.endswith('.h5')]

dataset = RandomDataset(h5_files=h5_files, exclude_idx_sub=['1e5'])

batch_size = 100
data_loader = DataLoader(dataset,
                         shuffle=True,
                         batch_size=batch_size,
                         num_workers=100)

d = []
for idx, item in tqdm(enumerate(data_loader)):
    pass
