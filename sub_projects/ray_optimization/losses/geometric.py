import torch
from ray_nn.metrics.geometric import SinkhornLoss as SHLoss
from sub_projects.ray_optimization.losses.losses import RayLoss
from sub_projects.ray_optimization.utils import ray_dict_to_tensor

sinkhorn_function = SHLoss(
    normalize_weights="weights1", p=1, backend="online", reduction=None
)


class SinkhornLoss(RayLoss):
    def loss_fn(
        self,
        a: dict,
        b: dict,
        exported_plane: str,
    ) -> torch.Tensor:
        a_tensor = ray_dict_to_tensor(a, exported_plane=exported_plane)
        b_tensor = ray_dict_to_tensor(b, exported_plane=exported_plane)
        if torch.cuda.is_available():
            a_tensor = a_tensor.cuda()

        if a_tensor.shape[1] == 0 or a_tensor.shape[1] == 1:
            a_tensor = (
                torch.ones(
                    (a_tensor.shape[0], 2, 2),
                    device=a_tensor.device,
                    dtype=a_tensor.dtype,
                )
                * -2
            )

        if torch.cuda.is_available():
            b_tensor = b_tensor.cuda()
        if b_tensor.shape[1] == 0 or b_tensor.shape[1] == 1:
            b_tensor = (
                torch.ones(
                    (b_tensor.shape[0], 2, 2),
                    device=b_tensor.device,
                    dtype=b_tensor.dtype,
                )
                * -1
            )
        loss = sinkhorn_function(
            a_tensor.contiguous(),
            b_tensor.contiguous(),
            torch.ones_like(a_tensor[..., 1]),
            torch.ones_like(b_tensor[..., 1]),
        )
        return loss.mean().item()
