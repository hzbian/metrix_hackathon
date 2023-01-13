from typing import Dict, List, Tuple

import numpy as np
import torch

from ray_nn.utils.ray_processing import HistToPointCloud
from ray_tools.base import RayOutput
from ray_tools.base.parameter import RayParameterContainer
from sub_projects.ray_surrogate.nn_models import SurrogateModel


class RayEngineSurrogate:

    def __init__(self,
                 ckpt_path: str,
                 hist_dim: int,
                 params_info: List[Tuple[str, Tuple[float, float]]],
                 gpu_id: int = None):

        self.ckpt_path = ckpt_path
        self.hist_dim = hist_dim
        self.params_info = params_info
        self.gpu_id = gpu_id

        self.hist_to_pc = HistToPointCloud()

        self.surrogate_model = SurrogateModel.load_from_checkpoint(self.ckpt_path)
        self.surrogate_model.eval()
        if self.gpu_id is not None:
            self.surrogate_model.cuda(self.gpu_id)

        for param in self.surrogate_model.parameters():
            param.requires_grad = False

    def run(self, param_containers: List[RayParameterContainer]) -> List[Dict]:
        bs = len(param_containers)
        params = torch.zeros(bs, len(self.params_info))
        for idx, container in enumerate(param_containers):
            params_cur = torch.tensor([[container[key].get_value(), lo, hi] for key, (lo, hi) in self.params_info],
                                      dtype=torch.get_default_dtype())
            params_cur[:, 0] = params_cur[:, 0] - params_cur[:, 1]
            params_cur[:, 0] = params_cur[:, 0] / (params_cur[:, 2] - params_cur[:, 1] + 1e-8)
            params[idx, :] = params_cur[:, 0]

        hist = torch.zeros(bs, self.hist_dim, dtype=torch.get_default_dtype())
        x_lims = torch.zeros(bs, 2, dtype=torch.get_default_dtype())
        y_lims = torch.zeros(bs, 2, dtype=torch.get_default_dtype())
        n_rays = torch.zeros(bs, dtype=torch.get_default_dtype())

        batch = dict(params=params,
                     tar_hist=hist,
                     tar_x_lims=x_lims,
                     tar_y_lims=y_lims,
                     tar_n_rays=n_rays)

        if self.gpu_id is not None:
            for k, v in batch.items():
                batch[k] = v.cuda(self.gpu_id)

        batch = self.surrogate_model(batch)

        dim_x = dim_y = int(np.sqrt(self.hist_dim))
        pred_pc_supp, pred_pc_weights = self.hist_to_pc(batch['pred_hist'].view(-1, dim_x, dim_y),
                                                        batch['pred_x_lims'], batch['pred_y_lims'])

        out = []
        for idx_bs in range(bs):
            x_loc, y_loc = [], []
            for idx_coord in range(self.hist_dim):
                if int(pred_pc_weights[idx_bs, idx_coord]) > 0:
                    x_loc += int(pred_pc_weights[idx_bs, idx_coord]) * [pred_pc_supp[idx_bs, idx_coord, 0]]
                    y_loc += int(pred_pc_weights[idx_bs, idx_coord]) * [pred_pc_supp[idx_bs, idx_coord, 1]]
            if len(x_loc) > 0:
                x_loc = torch.stack(x_loc).flatten().cpu().numpy()
                y_loc = torch.stack(y_loc).flatten().cpu().numpy()
            else:
                x_loc = np.array([])
                y_loc = np.array([])

            out.append({'ray_output': {'Spherical Grating': RayOutput(x_loc=x_loc, y_loc=y_loc, z_loc=None,
                                                x_dir=None, y_dir=None, z_dir=None, energy=None)}})

        return out
