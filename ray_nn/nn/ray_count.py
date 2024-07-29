import torch
import glob

from ray_nn.data.transform import Select
from ray_tools.simulation.torch_datasets import MemoryDataset, RayDataset

load_len: int | None = None
h5_files = list(glob.iglob('datasets/metrix_simulation/ray_emergency_surrogate/data_raw_*.h5'))
dataset = RayDataset(h5_files=h5_files,
                        sub_groups=['1e5/params',
                                    '1e5/ray_output/ImagePlane/histogram', '1e5/ray_output/ImagePlane/n_rays'], transform=Select(keys=['1e5/params', '1e5/ray_output/ImagePlane/histogram', '1e5/ray_output/ImagePlane/n_rays']))

memory_dataset = MemoryDataset(dataset=dataset, load_len=load_len)
new_min = float('inf')
new_max = float('-inf')
for params,hist,n_rays in memory_dataset:
    new_min = min(new_min, hist.min())
    new_max = max(new_max, hist.max())
print(new_min, new_max)
