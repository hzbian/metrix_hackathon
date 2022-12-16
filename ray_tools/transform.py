from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Any

from ray_tools.backend import RayOutput


class RayTransform(metaclass=ABCMeta):

    @abstractmethod
    def __call__(self, ray_output: RayOutput) -> Any:
        pass
