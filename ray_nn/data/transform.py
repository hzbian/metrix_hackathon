import torch

from ray_nn.utils.ray_processing import HistSubsampler
from ray_optim.plot import Plot
from ray_tools.base.parameter import MutableParameter, NumericalParameter, RayParameterContainer


class Select(torch.nn.Module):
    """
     Torch transform for selecting the specified entries in input dict. If you supply a search space, it will normalize the selected entries, that are dicts. If you want to 
    """

    def __init__(self, keys, search_space=None, non_dict_transform=None, omit_ray_params=[]):
        super().__init__()
        self.keys = keys
        self.search_space = search_space
        self.non_dict_transform = non_dict_transform
        self.omit_ray_params = omit_ray_params

    def forward(self, batch):
        outputs = []
        for key in self.keys:
            new_element = batch[key]
            if isinstance(new_element, dict):
                if self.search_space is not None:
                    if not isinstance(new_element, RayParameterContainer):
                        parameters = RayParameterContainer(
                            {k: NumericalParameter(v) for k, v in new_element.items()}
                        )
                    else:
                        parameters = new_element
                    normalized_parameter_container = Plot.normalize_parameters(parameters, search_space=self.search_space)
                    new_element = normalized_parameter_container
                if isinstance(new_element, RayParameterContainer):
                    new_element = new_element.to_value_dict()
                new_element = torch.hstack([torch.tensor(i) for k, i in new_element.items() if k not in self.omit_ray_params]).float()
            else:
                new_element = torch.tensor(batch[key]).float().unsqueeze(-1)
                if self.non_dict_transform is not None:
                    if key in self.non_dict_transform:
                        new_element = self.non_dict_transform[key](new_element)

            outputs.append(new_element)
        return tuple(outputs)

    
class SurrogateModelPreparation:
    """
    Transform to prepare data to work with :class:`ray_nn.nn.models.SurrogateModel`.
    Histograms with no rays are processed in a special ways, see ``_process_zero_hist``.
    :param planes_info: Dictionary with info about (image) planes to be considered in the surrogate model.
        Key = names of planes / value[0] = dataset keys to one or more histograms corresponding to a plane /
        value[1] = parameter names to be considered for a plane.
    :param params_key: Dataset keys to all recorded parameters
    :param params_info: Dictionary with info about all parameters to be considered. Key = parameter name /
        value[0] = lower bound for parameter values / value[0] = upper bound for parameter values.
    :param hist_subsampler: :class:`ray_nn.utils.ray_processing.HistSubsampler` to downsample raw histograms.
    """

    def __init__(self,
                 planes_info: dict[str, tuple[list[str], list[str]]],
                 params_key: str,
                 params_info: dict[str, tuple[float, float]],
                 hist_subsampler: HistSubsampler | None = None):
        super().__init__()
        self.planes_info = planes_info
        self.params_key = params_key
        self.params_info = params_info
        self.hist_subsampler = hist_subsampler

    def __call__(self, data):
        # normalize parameters to the interval [-1, 1] according to self.params_info
        params = self._process_params(data[self.params_key])

        out = {}
        for plane, (hist_keys, param_names) in self.planes_info.items():
            # Add all parameters for this plane with zero padding to len(params)
            out[plane] = {}
            out[plane]['params'] = torch.zeros(len(params))
            out[plane]['params'][:len(param_names)] = torch.tensor([params[name] for name in param_names])

            # Read all histograms and limits for this plane (e.g., multiple image plane layers)
            hist = torch.stack([torch.tensor(data[key]['histogram']) for key in hist_keys], dim=0)
            if self.hist_subsampler is not None:
                hist = self.hist_subsampler(hist)
            x_lims = torch.stack([torch.tensor(data[key]['x_lims']) for key in hist_keys], dim=0)
            y_lims = torch.stack([torch.tensor(data[key]['y_lims']) for key in hist_keys], dim=0)
            n_rays = torch.stack([torch.tensor(data[key]['n_rays']) for key in hist_keys], dim=0)

            # Replace all empty histograms with special histogram (with non-empty mass)
            self._process_zero_hist(hist, x_lims, y_lims, n_rays)

            out[plane].update(dict(tar_hist=hist.to(torch.get_default_dtype()),
                                   tar_x_lims=x_lims.to(torch.get_default_dtype()),
                                   tar_y_lims=y_lims.to(torch.get_default_dtype()),
                                   tar_n_rays=n_rays.to(torch.get_default_dtype())))
        return out

    def _process_zero_hist(self,
                           hist: torch.Tensor,
                           x_lims: torch.Tensor,
                           y_lims: torch.Tensor,
                           n_rays: torch.Tensor) -> None:
        """
        Replaces all empty histograms with special histogram: use constant values in the square [-1e4, 1e4]^2 and
        normalize them to have mass 1.
        """
        idx_zeros = (n_rays <= 1.0)
        hist[idx_zeros, ...] = torch.ones_like(hist[idx_zeros, ...]).abs()
        hist[idx_zeros, ...] = hist[idx_zeros, ...] / hist[idx_zeros, ...].sum(dim=[-2, -1], keepdim=True)
        x_lims[idx_zeros, 0] = y_lims[idx_zeros, 0] = -1e-4
        x_lims[idx_zeros, 1] = y_lims[idx_zeros, 1] = 1e-4
        n_rays[idx_zeros] = hist[idx_zeros, ...].sum(dim=[-2, -1]).to(n_rays.dtype)

    def _process_params(self, params: dict[str, float]) -> dict[str, float]:
        """
        Normalize parameters to the interval [-1, 1] according to ``params_info``.
        """
        params_processed = {}
        for key, (lo, hi) in self.params_info.items():
            value = params[key]
            # The offset 1e-8 is to handle constant parameters (e.g., number of rays)
            params_processed[key] = 2.0 * (value - lo) / (hi - lo + 1e-8) - 1.0
        return params_processed


class SurrogatePreparation:
    """
    Deprecated
    """

    def __init__(self,
                 params_key: str,
                 params_info: list[tuple[str, tuple[float, float]]],
                 hist_key: str,
                 hist_subsampler: HistSubsampler,
                 inp_hist_key: str | None = None
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
