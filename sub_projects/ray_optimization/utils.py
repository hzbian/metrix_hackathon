import torch
torch.set_num_threads(1)

def ray_output_to_tensor(ray_output: list[dict], exported_plane: str, to_cpu=False) -> list[torch.Tensor]:
    assert isinstance(ray_output, list)
    return [ray_dict_to_tensor(element, exported_plane, to_cpu=to_cpu) for element in
                ray_output]

def ray_dict_to_tensor(ray_dict: dict, exported_plane: str, to_cpu=False) -> torch.Tensor:
        rays: dict = ray_dict['ray_output'][exported_plane]
        x_locs = torch.stack([value.x_loc.clone().detach() for value in rays.values()])
        y_locs = torch.stack([value.y_loc.clone().detach() for value in rays.values()])
        output = torch.stack((x_locs, y_locs), dim=-1)
        if to_cpu:
            return output.cpu()
        else:
            return output
