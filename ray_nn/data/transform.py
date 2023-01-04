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
                new_element = torch.tensor(batch[key]).float().unsqueeze(-2)
            outputs.append(new_element)
        return tuple(outputs)
