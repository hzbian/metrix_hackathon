import sys
import torch

sys.path.insert(0, '../../')
from ray_nn.metrics.geometric import SinkhornLoss

sinkhorn_function = SinkhornLoss(normalize_weights='weights1', p=1, backend='online', reduction=None)


def sinkhorn_loss(y: torch.Tensor, y_hat: torch.Tensor):
    y = y.cuda()

    if y.shape[1] == 0 or y.shape[1] == 1:
        y = torch.ones((y.shape[0], 2, 2), device=y.device, dtype=y.dtype) * -2

    y_hat = y_hat.cuda()
    if y_hat.shape[1] == 0 or y_hat.shape[1] == 1:
        y_hat = torch.ones((y_hat.shape[0], 2, 2), device=y_hat.device, dtype=y_hat.dtype) * -1
    loss = sinkhorn_function(y.contiguous(), y_hat.contiguous(), torch.ones_like(y[..., 1]),
                             torch.ones_like(y_hat[..., 1]))
    # loss = torch.tensor((y.shape[1] - y_hat.shape[1]) ** 2 / 2)

    return loss


def multi_objective_loss(y: torch.Tensor, y_hat: torch.Tensor):
    y = y.cuda()
    y_hat = y_hat.cuda()
    ray_count_loss = (y.shape[1] - y_hat.shape[1]) ** 2 / 2
    return sinkhorn_loss(y, y_hat), ray_count_loss
