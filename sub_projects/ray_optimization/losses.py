import sys
from typing import Union, Dict, List, Iterable

import torch

from ray_optim.ray_optimizer import RayOptimizer

sys.path.insert(0, '../../')
from ray_nn.metrics.geometric import SinkhornLoss

sinkhorn_function = SinkhornLoss(normalize_weights='weights1', p=1, backend='online', reduction=None)


def sinkhorn_loss(a: Union[Dict, List[Dict], Iterable[Dict]], b: Union[Dict, List[Dict], Iterable[Dict]],
                  exported_plane: str):
    a = RayOptimizer.ray_output_to_tensor(a, exported_plane=exported_plane)
    b = RayOptimizer.ray_output_to_tensor(b, exported_plane=exported_plane)
    if torch.cuda.is_available():
        a = a.cuda()

    if a.shape[1] == 0 or a.shape[1] == 1:
        a = torch.ones((a.shape[0], 2, 2), device=a.device, dtype=a.dtype) * -2

    if torch.cuda.is_available():
        b = b.cuda()
    if b.shape[1] == 0 or b.shape[1] == 1:
        b = torch.ones((b.shape[0], 2, 2), device=b.device, dtype=b.dtype) * -1
    loss = sinkhorn_function(a.contiguous(), b.contiguous(), torch.ones_like(a[..., 1]),
                             torch.ones_like(b[..., 1]))
    return loss


def ray_count_loss(a: Union[Dict, List[Dict], Iterable[Dict]], b: Union[Dict, List[Dict], Iterable[Dict]],
                   exported_plane: str):
    a_tensor = RayOptimizer.ray_output_to_tensor(a, exported_plane=exported_plane)
    b_tensor = RayOptimizer.ray_output_to_tensor(b, exported_plane=exported_plane)
    return (a_tensor.shape[1] - b_tensor.shape[1]) ** 2 / 2


def multi_objective_loss(a: Union[Dict, List[Dict], Iterable[Dict]], b: Union[Dict, List[Dict], Iterable[Dict]],
                         exported_plane: str):
    return sinkhorn_loss(a, b, exported_plane=exported_plane), ray_count_loss(a, b, exported_plane=exported_plane)
