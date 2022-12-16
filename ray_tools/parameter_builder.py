from itertools import product

from .engine import RayParameterDict
from .parameter import GridParameter


def build_parameter_grid(param_dict: RayParameterDict):
    param_list = list(param_dict.items())
    param_list_expanded = []
    for param in param_list:
        if isinstance(param[1], GridParameter):
            param_list_expanded.append([(param[0], p) for p in param[1].expand()])
        else:
            param_list_expanded.append([param])

    return [RayParameterDict(list(param_list)) for param_list in product(*param_list_expanded)]