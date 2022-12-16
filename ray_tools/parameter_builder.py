from itertools import product

from .parameter import GridParameter, RayParameterContainer


def build_parameter_grid(param_container: RayParameterContainer):
    param_list = list(param_container.items())
    param_list_expanded = []
    for param in param_list:
        if isinstance(param[1], GridParameter):
            param_list_expanded.append([(param[0], p) for p in param[1].expand()])
        else:
            param_list_expanded.append([param])

    return [RayParameterContainer(list(param_list)) for param_list in product(*param_list_expanded)]