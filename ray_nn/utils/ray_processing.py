from typing import Tuple

import torch
from torch import nn


class HistSubsampler(nn.Module):

    def __init__(self, factor: int) -> None:
        super().__init__()
        self.factor = factor
        self._subsampler = nn.AvgPool2d(kernel_size=factor, divisor_override=1)

    def forward(self, hist: torch.Tensor) -> torch.Tensor:
        return self._subsampler(hist)


class HistToPointCloud(nn.Module):

    def __init__(self, as_sequence: bool = True) -> None:
        super().__init__()
        self.as_sequence = as_sequence

    def forward(self, hist: torch.Tensor, x_lims: torch.Tensor, y_lims: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        bs, dim_x, dim_y = hist.shape

        coord_x_width = (x_lims[:, 1] - x_lims[:, 0]) / dim_x
        coord_y_width = (y_lims[:, 1] - y_lims[:, 0]) / dim_y

        # # Non-vectorized generation of coordinates
        # coord_x = []
        # coord_y = []
        # for idx in range(bs):
        #     coord_x.append(torch.linspace(x_lims[idx, 0] + coord_x_width[idx] / 2.,
        #                                   x_lims[idx, 1] - coord_x_width[idx] / 2.,
        #                                   dim_x, device=hist.device))
        #     coord_y.append(torch.linspace(y_lims[idx, 0] + coord_y_width[idx] / 2.,
        #                                   y_lims[idx, 1] - coord_y_width[idx] / 2.,
        #                                   dim_y, device=hist.device))
        # coord_x = torch.stack(coord_x)
        # coord_y = torch.stack(coord_y)

        coord_x = torch.linspace(0.0, 1.0, dim_x, device=hist.device).view(1, -1)
        coord_x = coord_x * coord_x_width.view(-1, 1) * (dim_x - 1)
        coord_x = coord_x + x_lims[:, 0].view(-1, 1) + coord_x_width.view(-1, 1) / 2.

        coord_y = torch.linspace(0.0, 1.0, dim_y, device=hist.device).view(1, -1)
        coord_y = coord_y * coord_y_width.view(-1, 1) * (dim_y - 1)
        coord_y = coord_y + y_lims[:, 0].view(-1, 1) + coord_y_width.view(-1, 1) / 2.

        mesh_x = coord_x.unsqueeze(2)
        mesh_y = coord_y.unsqueeze(1)

        if self.as_sequence:
            grid_x = mesh_x * torch.clamp(hist, 0.0, 1.0)
            grid_y = mesh_y * torch.clamp(hist, 0.0, 1.0)

            idx_nonzero = hist != 0

            pc_x = [grid_x[idx, ...][idx_nonzero[idx, ...]].flatten() for idx in range(bs)]
            pc_y = [grid_y[idx, ...][idx_nonzero[idx, ...]].flatten() for idx in range(bs)]
            pc_weights = [hist[idx, ...][idx_nonzero[idx, ...]].flatten() for idx in range(bs)]
            pc_lens = [len(w) for w in pc_weights]

            pc_x = torch.nn.utils.rnn.pad_sequence(pc_x, batch_first=True)
            pc_y = torch.nn.utils.rnn.pad_sequence(pc_y, batch_first=True)
            pc_weights = torch.nn.utils.rnn.pad_sequence(pc_weights, batch_first=True)

            return torch.stack([pc_x, pc_y], dim=-1), pc_weights, pc_lens
        else:
            pc_x = (mesh_x * torch.ones_like(mesh_y)).flatten(start_dim=1)
            pc_y = (mesh_y * torch.ones_like(mesh_x)).flatten(start_dim=1)
            pc_weights = hist.flatten(start_dim=1)

            return torch.stack([pc_x, pc_y], dim=-1), pc_weights
