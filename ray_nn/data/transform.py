from typing import List

import torch


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

    def __init__(self, key_params: str, list_params: List[str], key_hist: str):
        super().__init__()
        self.key_params = key_params
        self.list_params = list_params
        self.key_hist = key_hist

    def __call__(self, inp):
        params = inp[self.key_params]
        params = torch.tensor([params[key] for key in self.list_params], dtype=torch.get_default_dtype())
        histogram = torch.tensor(inp[self.key_hist]['histogram'], dtype=torch.get_default_dtype())
        x_lims = torch.tensor(inp[self.key_hist]['x_lims'], dtype=torch.get_default_dtype())
        y_lims = torch.tensor(inp[self.key_hist]['y_lims'], dtype=torch.get_default_dtype())
        n_rays = torch.tensor(inp[self.key_hist]['n_rays'], dtype=torch.get_default_dtype())
        return params, histogram, x_lims, y_lims, n_rays
