from typing import Union, List, Dict, Iterable
import torch


def ray_output_to_tensor(ray_output: Union[Dict, List[Dict], Iterable[Dict]], exported_plane: str, to_cpu=True):
    if not isinstance(ray_output, Dict):
        return [ray_output_to_tensor(element, exported_plane) for element in
                ray_output]
    else:
        rays: dict = ray_output['ray_output'][exported_plane]
        x_locs = torch.stack([value.x_loc.clone().detach() for value in rays.values()])
        y_locs = torch.stack([value.y_loc.clone().detach() for value in rays.values()])
        output = torch.stack((x_locs, y_locs), -1)
        if to_cpu:
            return output.cpu()
        else:
            return output
