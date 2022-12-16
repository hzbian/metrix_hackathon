from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Tuple, Union, List, Any

import numpy as np

from .utils import RandomGenerator


class RayParameter(metaclass=ABCMeta):

    @abstractmethod
    def get_value(self) -> Any:
        pass

    @abstractmethod
    def clone(self) -> RayParameter:
        pass

    def __repr__(self) -> str:
        return self.__class__.__name__ + ': ' + str(self.get_value())


class ConstantParameter(RayParameter):

    def __init__(self, value: float):
        self.value = value

    def get_value(self) -> float:
        return self.value

    def clone(self) -> ConstantParameter:
        return ConstantParameter(value=self.value)


class GridParameter(RayParameter):

    def __init__(self, value: Union[List[float], np.ndarray]):
        self.value = np.array(value).flatten()

    def get_value(self) -> float:
        return self.value[0]

    def expand(self) -> List[ConstantParameter]:
        return [ConstantParameter(value=value) for value in self.value]

    def clone(self) -> GridParameter:
        return GridParameter(value=self.value)


class MutableParameter(RayParameter):

    def __init__(self, value: float, value_lims: Tuple[float, float] = None):
        self.value = value
        self.value_lims = value_lims

    def get_value(self) -> float:
        return self.value

    def clone(self) -> MutableParameter:
        return MutableParameter(value=self.value, value_lims=self.value_lims)


class RandomParameter(MutableParameter):

    def __init__(self, value_lims: Tuple[float, float] = None, rg: RandomGenerator = None):
        self.rg = rg if rg is not None else RandomGenerator()
        value = self.rg.rg_random.uniform(*value_lims)
        super().__init__(value, value_lims)

    def resample(self) -> None:
        self.value = self.rg.rg_random.uniform(*self.value_lims)

    def clone(self) -> RandomParameter:
        param = RandomParameter(self.value_lims, rg=self.rg)
        param.value = self.value
        return param
