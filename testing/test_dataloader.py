import sys

import numpy as np

sys.path.insert(0, '../')

from tqdm import tqdm
import os

import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from ray_tools.simulation.torch_datasets import RayDataset

# Important fix to make custom collate_fn work
# https://forums.fast.ai/t/runtimeerror-received-0-items-of-ancdata/48935
torch.multiprocessing.set_sharing_strategy('file_system')

h5_path = os.path.join('../datasets/metrix_simulation')

h5_files = [os.path.join(h5_path, file) for file in os.listdir(h5_path) if file.endswith('.h5')]

dataset = RayDataset(h5_files=h5_files)

batch_size = 100
data_loader = DataLoader(dataset,
                         shuffle=True,
                         batch_size=batch_size,
                         num_workers=100)

for idx, item in tqdm(enumerate(data_loader)):
    pass

exported_planes = ["U41_318eV",
                   "ASBL",
                   "M1-Cylinder",
                   "Spherical Grating",
                   "Exit Slit",
                   "E1",
                   "E2",
                   "ImagePlane"]

show_examples = [7]  # range(n_examples)

item = {idx: dataset.__getitem__(idx) for idx in show_examples}

for idx in show_examples:
    for exported_plane in exported_planes:
        # plt.figure()
        # plt.title(exported_plane + ' ' + str(idx))
        # plt.scatter(item[idx]['1e5']['ray_output'][exported_plane]['raw'].x_loc,
        #             item[idx]['1e5']['ray_output'][exported_plane]['raw'].y_loc,
        #             s=0.01)
        # plt.show()

        plt.figure(figsize=(10, 10))
        plt.title(exported_plane + ' ' + str(idx))
        plt.imshow(np.fliplr(item[idx]['1e5']['ray_output'][exported_plane]['hist']['histogram'].T),
                   cmap='Greys')
        print(item[idx]['1e5']['ray_output'][exported_plane]['hist']['n_rays'],
              item[idx]['1e5']['ray_output'][exported_plane]['hist']['x_lims'],
              item[idx]['1e5']['ray_output'][exported_plane]['hist']['y_lims'])
        plt.show()
