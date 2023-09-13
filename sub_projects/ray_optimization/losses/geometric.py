from typing import Dict, Iterable, List, Union

import torch
from ray_nn.metrics.geometric import SinkhornLoss
from sub_projects.ray_optimization.losses.losses import RayLoss
from sub_projects.ray_optimization.utils import ray_output_to_tensor

sinkhorn_function = SinkhornLoss(normalize_weights='weights1', p=1, backend='online', reduction=None)


class SinkhornLoss(RayLoss):
    def loss_fn(self, a: Union[Dict, List[Dict], Iterable[Dict]], b: Union[Dict, List[Dict], Iterable[Dict]],
                exported_plane: str) -> torch.Tensor:
        a = ray_output_to_tensor(a, exported_plane=exported_plane)
        b = ray_output_to_tensor(b, exported_plane=exported_plane)
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
        return loss.mean()

