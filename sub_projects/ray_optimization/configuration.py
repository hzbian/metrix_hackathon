
from typing import Optional, Callable, List

from ray_tools.base.engine import Engine
from ray_tools.base.transform import RayTransform

class RealDataConfiguration:
    def __init__(self, path: str, train_set: List[str],
                 validation_set: Optional[List[str]] = None):
        self.path: str = path
        self.train_set: List[str] = train_set
        self.validation_set: Optional[List[str]] = validation_set


class TargetConfiguration:
    def __init__(self, param_func: Callable, engine: Engine, exported_plane: str, num_beamline_samples: int = 20,
                 max_target_deviation: float = 0.3, max_offset_search_deviation: float = 0.3,
                 logging_project: Optional[str] = None,
                 z_layers: List[float] = (0.),
                 transforms: Optional[RayTransform] = None,
                 real_data_configuration: Optional[RealDataConfiguration] = None):
        self.max_offset_search_deviation: float = max_offset_search_deviation
        self.z_layers = z_layers
        self.transforms: RayTransform = transforms
        self.num_beamline_samples: int = num_beamline_samples
        self.exported_plane: str = exported_plane
        self.engine: Engine = engine
        self.max_target_deviation: float = max_target_deviation
        self.param_func: Callable = param_func
        self.real_data_configuration = real_data_configuration
        self.logging_project = logging_project

