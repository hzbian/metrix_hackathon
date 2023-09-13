from typing import Dict, Iterable, List, Union
import ignite
import torch
from ray_tools.base.transform import Histogram
from sub_projects.ray_optimization.losses.losses import RayLoss


class SSIMHistogramLoss(RayLoss):
    """
    Calculates the histograms of two ray_outputs and the SSIM between those two histograms.
    :param n_bins The amount of bins the histograms are generated with.
    """

    def __init__(self, n_bins: int):
        self.n_bins: int = n_bins
        self.ssim_fun = ignite.metrics.SSIM(1.0)

    def loss_fn(self, a: Union[Dict, List[Dict], Iterable[Dict]], b: Union[Dict, List[Dict], Iterable[Dict]],
                exported_plane: str) -> torch.Tensor:
        a_dict = a['ray_output'][exported_plane]
        b_dict = b['ray_output'][exported_plane]
        hist_a_list = []
        hist_b_list = []

        for key, a in a_dict.items():
            b = b_dict[key]
            x_min = min(a.x_loc.min(), b.x_loc.min())
            x_max = max(a.x_loc.max(), b.x_loc.max())
            y_min = min(a.y_loc.min(), b.y_loc.min())
            y_max = max(a.y_loc.max(), b.y_loc.max())
            hist_a_list.append(Histogram(self.n_bins, (x_min, x_max), (y_min, y_max))(a)['histogram'])
            hist_b_list.append(Histogram(self.n_bins, (x_min, x_max), (y_min, y_max))(b)['histogram'])
        stack_a = torch.stack(hist_a_list).unsqueeze(1).float()
        stack_b = torch.stack(hist_b_list).unsqueeze(1).float()
        self.ssim_fun.update((stack_a, stack_b))
        return torch.Tensor([self.ssim_fun.compute()])