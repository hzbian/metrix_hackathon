import torch
import glob

from ray_nn.data.transform import Select
from ray_tools.simulation.torch_datasets import MemoryDataset, RayDataset

load_len: int | None =  100
dataset_normalize_outputs = True
h5_files = list(glob.iglob('datasets/metrix_simulation/ray_emergency_surrogate/50+50_data_raw_*.h5')) # ['datasets/metrix_simulation/ray_emergency_surrogate/49+50_data_raw_0.h5']
dataset = RayDataset(h5_files=h5_files,
                     sub_groups=['1e5/params',
                                 '1e5/n_rays'], transform=Select(keys=['1e5/params', '1e5/n_rays']))

memory_dataset = MemoryDataset(dataset=dataset, load_len=load_len)
new_min = float('inf')
new_max = float('-inf')
for x,y in memory_dataset:
    new_min = min(new_min, y)
    new_max = max(new_max, y)
print(new_min, new_max)
