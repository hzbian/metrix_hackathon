from typing import List, Tuple

import torch

from ray_nn.utils.ray_processing import HistSubsampler


class Select(torch.nn.Module):
    """
     Torch transform for selecting the specified entries in input dict
    """

    def __init__(self, keys):
        super().__init__()
        self.keys = keys

    def forward(self, batch):
        outputs = []
        for key in self.keys:
            new_element = batch[key]
            if isinstance(new_element, dict):
                # TODO: might need to be extended for recursion
                new_element = torch.hstack([torch.tensor(i) for i in new_element.values()]).float()
            else:
                new_element = torch.tensor(batch[key]).float().unsqueeze(-1)
            outputs.append(new_element)
        return tuple(outputs)


class SurrogatePreparation:

    def __init__(self,
                 params_key: str,
                 params_info: List[Tuple[str, Tuple[float, float]]],
                 hist_key: str,
                 hist_subsampler: HistSubsampler,
                 inp_hist_key: str = None
                 ):
        super().__init__()
        self.params_key = params_key
        self.params_info = params_info
        self.hist_key = hist_key
        self.hist_subsampler = hist_subsampler
        self.inp_hist_key = inp_hist_key

    def __call__(self, inp):
        params = inp[self.params_key]
        params = torch.tensor([[params[key], lo, hi] for key, (lo, hi) in self.params_info],
                              dtype=torch.get_default_dtype())
        params[:, 0] = params[:, 0] - params[:, 1]
        params[:, 0] = params[:, 0] / (params[:, 2] - params[:, 1] + 1e-8)
        params = params[:, 0]

        hist = torch.tensor(inp[self.hist_key]['histogram'], dtype=torch.get_default_dtype())
        hist = self.hist_subsampler(hist.unsqueeze(0)).flatten()

        x_lims = torch.tensor(inp[self.hist_key]['x_lims'], dtype=torch.get_default_dtype())
        y_lims = torch.tensor(inp[self.hist_key]['y_lims'], dtype=torch.get_default_dtype())
        n_rays = torch.tensor(inp[self.hist_key]['n_rays'], dtype=torch.get_default_dtype())

        if n_rays.item() == 0.0:
            x_lims[0] = x_lims[1] = y_lims[0] = y_lims[1] = 10.0
            hist[0] = 1.0
            n_rays = torch.tensor(1.0, dtype=torch.get_default_dtype())

        out = dict(params=params,
                   tar_hist=hist,
                   tar_x_lims=x_lims,
                   tar_y_lims=y_lims,
                   tar_n_rays=n_rays)

        if self.inp_hist_key is not None:
            inp_hist = torch.tensor(inp[self.inp_hist_key]['histogram'], dtype=torch.get_default_dtype())
            inp_hist = self.hist_subsampler(inp_hist.unsqueeze(0)).flatten()

            inp_x_lims = torch.tensor(inp[self.inp_hist_key]['x_lims'], dtype=torch.get_default_dtype())
            inp_y_lims = torch.tensor(inp[self.inp_hist_key]['y_lims'], dtype=torch.get_default_dtype())
            inp_n_rays = torch.tensor(inp[self.inp_hist_key]['n_rays'], dtype=torch.get_default_dtype())

            if inp_n_rays.item() == 0.0:
                inp_x_lims[0] = inp_x_lims[1] = inp_y_lims[0] = inp_y_lims[1] = 10.0
                inp_hist[0] = 1.0
                inp_n_rays = torch.tensor(1.0, dtype=torch.get_default_dtype())

            out.update(dict(inp_hist=inp_hist,
                            inp_x_lims=inp_x_lims,
                            inp_y_lims=inp_y_lims,
                            inp_n_rays=inp_n_rays))

        return out
