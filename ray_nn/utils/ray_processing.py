
import torch
from torch import nn


class HistSubsampler(nn.Module):
    """
    Downsamples a (batch of) histogram by a given factor.
    Downsampling is done by pooling such that the number of rays (sum of histogram) is preserved.
    """

    def __init__(self, factor: int) -> None:
        super().__init__()
        self.factor = factor
        self._subsampler = nn.AvgPool2d(kernel_size=factor, divisor_override=1)

    def forward(self, hist: torch.Tensor) -> torch.Tensor:
        return self._subsampler(hist)


class HistToPointCloud(nn.Module):
    """
    Converts a histogram into a point cloud.
    Output is a tensor of shape [batch size, #pixels in hist, 2];
    first dimension are x-coordinates and second are y-coordinates.
    The ray coordinates are computed by a meshgrid according to the size of hist and given limits.
    Each ray is endowed with a weights, which is the corresponding entry of the histogram.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, hist: torch.Tensor, x_lims: torch.Tensor, y_lims: torch.Tensor) -> tuple[torch.Tensor, ...]:
        _, dim_x, dim_y = hist.shape

        coord_x_width = (x_lims[:, 1] - x_lims[:, 0]) / dim_x
        coord_y_width = (y_lims[:, 1] - y_lims[:, 0]) / dim_y

        unit_grid_x = torch.linspace(0.0, 1.0, dim_x, device=hist.device, dtype=hist.dtype).unsqueeze(0)
        coord_x = unit_grid_x * coord_x_width.unsqueeze(0) * (dim_x - 1)
        coord_x = coord_x + x_lims[:, 0].unsqueeze(0) + coord_x_width.unsqueeze(0) / 2.
        unit_grid_y = torch.linspace(1.0, 0.0, dim_y, device=hist.device, dtype=hist.dtype).unsqueeze(0)
        coord_y = unit_grid_y * coord_y_width.unsqueeze(0) * (dim_y - 1)
        coord_y = coord_y + y_lims[:, 0].unsqueeze(0) + coord_y_width.unsqueeze(0) / 2.

        mesh_x = coord_x.unsqueeze(2)
        mesh_y = coord_y.unsqueeze(1)
        mesh_x = mesh_x * torch.ones_like(mesh_y)
        mesh_y = mesh_y * torch.ones_like(mesh_x)

        pc_x = mesh_x.flatten(start_dim=1)
        pc_y = mesh_y.flatten(start_dim=1)
        pc_weights = hist.flatten(start_dim=1)

        return torch.stack([pc_x, pc_y], dim=-1), pc_weights
