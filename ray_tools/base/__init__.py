from typing import Union, Dict

from .backend import RayOutput
from .transform import RayTransform

RayTransformType = Union[RayTransform, Dict[str, RayTransform]]

