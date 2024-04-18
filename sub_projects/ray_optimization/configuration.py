
from collections import OrderedDict
from collections.abc import Callable

import os
import uuid

from ray_tools.base.engine import Engine
from ray_tools.base.transform import RayTransform
from ray_optim.ray_optimizer import OptimizerBackend
from ray_tools.base.parameter import MutableParameter, NumericalOutputParameter, NumericalParameter, RandomOutputParameter, RandomParameter, RayParameterContainer
from ray_tools.base.utils import RandomGenerator
from sub_projects.ray_optimization.losses.losses import RayLoss


class RealDataConfiguration:
    def __init__(self, path: str, train_set: list[str],
                 validation_set: list[str] | None = None):
        self.path: str = path
        self.train_set: list[str] = train_set
        self.validation_set: list[str] | None = validation_set


class TargetConfiguration:
    def __init__(self, param_func: Callable, engine: Engine, exported_plane: str, num_beamline_samples: int = 20,
                 max_target_deviation: float = 0.3, max_offset_search_deviation: float = 0.3, max_sample_generation_deviation: float = 1.0,
                 logging_project: str | None = None,
                 z_layers: list[float] = [0.,],
                 transforms: RayTransform | None = None,
                 real_data_configuration: RealDataConfiguration | None = None,
                 load_target_path: str | None = None):
        self.max_offset_search_deviation: float = max_offset_search_deviation
        self.z_layers = z_layers
        self.transforms: RayTransform | None = transforms
        self.num_beamline_samples: int = num_beamline_samples
        self.max_sample_generation_deviation: float = max_sample_generation_deviation
        self.exported_plane: str = exported_plane
        self.engine: Engine = engine
        self.max_target_deviation: float = max_target_deviation
        self.param_func: Callable = param_func
        self.real_data_configuration = real_data_configuration
        self.logging_project = logging_project
        self.load_target_path = load_target_path

def params_to_func(parameters, rg: RandomGenerator | None = None, enforce_lims_keys: list[str] = [],
                   output_parameters: list[str] = [], fixed_parameters: list[str] = [], mutable_only: bool = False) -> Callable[[], RayParameterContainer]:
    def output_func() -> RayParameterContainer:
        elements = []
        for k, v in parameters.items():
            if hasattr(v, '__getitem__'):
                if mutable_only and k in fixed_parameters:
                    continue

                if k in output_parameters:
                    typ = RandomOutputParameter
                else:
                    typ = RandomParameter

                elements.append((k, typ(value_lims=(v[0], v[1]), rg=rg, enforce_lims=k in enforce_lims_keys)))
            else:
                if mutable_only:
                    continue
                if k in output_parameters:
                    typ = NumericalOutputParameter
                else:
                    typ = NumericalParameter
               
                elements.append((k, typ(value=v)))
            
        elements = OrderedDict(elements)
        # do not optimize the fixed parameters, set them to the center of interval
        for key in fixed_parameters:
            old_param = elements[key]
            if isinstance(old_param, MutableParameter) and key in fixed_parameters:
                if key in output_parameters:
                    typ = NumericalOutputParameter
                else:
                    typ = NumericalParameter
      
                elements[key] = typ((old_param.value_lims[1] + old_param.value_lims[0]) / 2)
        return RayParameterContainer(elements)
    return output_func


def build_study_name(param_func: Callable, max_target_deviation: float, max_offset_search_deviation: float,
                     loss: RayLoss | None = None, optimizer_backend: OptimizerBackend | None = None, appendix: str | None = None) -> str:
    var_count: int = sum(isinstance(x, RandomParameter) for x in param_func().values())
    string_list = [str(var_count), 'target', str(max_target_deviation), 'search', str(max_offset_search_deviation)]

    if appendix is not None:
        string_list.append(appendix)
    if RayLoss is not None:
        string_list.append(loss.__class__.__name__.replace('Loss', ''))
    if OptimizerBackend is not None:
        string_list.append(optimizer_backend.__class__.__name__.replace('OptimizerBackend', ''))
    return '-'.join(string_list)


def build_ray_workdir_path(parent_path: str):
    return os.path.join(parent_path, str(uuid.uuid4()))

