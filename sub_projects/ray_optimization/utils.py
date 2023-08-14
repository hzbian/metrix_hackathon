from typing import Union, List, Dict, Iterable
import torch


def ray_output_to_tensor(ray_output: Union[Dict, List[Dict], Iterable[Dict]], exported_plane: str):
    if not isinstance(ray_output, Dict):
        return [ray_output_to_tensor(element, exported_plane) for element in
                ray_output]
    else:
        rays: dict = ray_output['ray_output'][exported_plane]
        x_locs = torch.stack([torch.tensor(value.x_loc) for value in rays.values()])
        y_locs = torch.stack([torch.tensor(value.y_loc) for value in rays.values()])
        return torch.stack((x_locs, y_locs), -1)
