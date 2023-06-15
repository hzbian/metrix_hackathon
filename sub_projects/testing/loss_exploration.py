import sys
from typing import List

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
from torch.utils.data import Dataset

sys.path.insert(0, '../../')
from ray_optim.ray_optimizer import RayOptimizer

from ray_tools.base import RayOutput
from ray_tools.base.transform import MultiLayer, ToDict


class GaussBlobbdataset(Dataset):

    def __init__(self, size: int = 100, spread_factor: (int, int) = (5, 20), x_mean: (int, int) = (10, 20),
                 y_mean: (int, int) = (10, 90)
                 , x_dir: int = 1, y_dir: int = 1, z_dir: int = 1, n_rays: int = 10000,
                 dist_layers: List[float] = [0, 10, 20]):
        self.size = size
        self.spread_factor = spread_factor
        self.x_dir = x_dir
        self.y_dir = y_dir
        self.z_dir = z_dir
        self.x_mean = x_mean
        self.y_mean = y_mean
        self.n_rays = n_rays
        self.dist_layers = dist_layers
        self.trans = MultiLayer(self.dist_layers)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        x_mean_value = np.random.randint(self.x_mean[0], self.x_mean[1])
        y_mean_value = np.random.randint(self.x_mean[0], self.x_mean[1])
        spread_factor_value = np.random.randint(self.spread_factor[0], self.spread_factor[1])
        dis = multivariate_normal([x_mean_value, y_mean_value],
                                  [[10 * spread_factor_value, 0], [0, 10 * spread_factor_value]])
        x, y = np.mgrid[0:100, 0:100]
        pos = np.dstack((x, y))
        data = dis.pdf(pos)
        draw = dis.rvs(size=self.n_rays)
        draw[draw < 0] -= 1
        draw = draw.astype(int)
        # draw = np.stack(list(filter(lambda x: 0 <= x[0] < 100 and 0 <= x[1] < 100, draw)))
        x_loc = np.empty(draw.shape[0])
        y_loc = np.empty(draw.shape[0])
        energy = np.ones(draw.shape[0])
        for i in range(draw.shape[0]):
            x_loc[i] = draw[i][0]
            y_loc[i] = draw[i][1]
            # energy[i] = data[draw[i][0]][draw[i][1]]
        x_dir_values = np.full_like(x_loc, self.x_dir)
        y_dir_values = np.full_like(x_loc, self.y_dir)
        z_dir_values = np.full_like(x_loc, self.z_dir)
        z_loc = np.zeros(draw.shape[0])
        ray_out = RayOutput(x_loc, y_loc, z_loc, x_dir_values, y_dir_values, z_dir_values, energy)
        ray_layers = self.trans(ray_out)
        params = {'x_center': torch.Tensor([x_mean_value]), 'y_center': y_mean_value,
                  'spread_factor': spread_factor_value, 'n_rays': torch.Tensor([draw.shape[0]])}
        to_dict_trans = ToDict()

        #for key, layer in ray_layers.items():
        #    ray_layers[key] = to_dict_trans(layer)
        return {'ray_output': {'ImagePlane': ray_layers}, 'params': params}


gbd = GaussBlobbdataset(100)

for ray_output in gbd:
    print(ray_output)
    ray_output = RayOptimizer.ray_output_to_tensor(ray_output=ray_output, exported_plane='ImagePlane')
    print(ray_output.shape)
    fig = RayOptimizer.fixed_position_plot([ray_output], [ray_output], [ray_output], xlim=[-2, 2], ylim=[-2, 2])
    plt.plot()
    plt.show()
    plt.savefig('out.png')
    break
