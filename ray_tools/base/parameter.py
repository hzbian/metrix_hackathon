from __future__ import annotations

from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import Tuple, Union, List, Any

import numpy as np
from raypyng.xmltools import XmlElement

from .utils import RandomGenerator


class RayParameter(metaclass=ABCMeta):

    @abstractmethod
    def get_value(self) -> Any:
        """
        Important: This method should have a deterministic output (calling it twice should return the same value)
        :return:
        """
        pass

    @abstractmethod
    def clone(self) -> RayParameter:
        pass

    def __repr__(self) -> str:
        return self.__class__.__name__ + ': ' + str(self.get_value())


class NumericalParameter(RayParameter):

    def __init__(self, value: float):
        self.value = value

    def get_value(self) -> float:
        return self.value

    def clone(self) -> NumericalParameter:
        return NumericalParameter(value=self.value)


class MutableParameter(NumericalParameter):

    def __init__(self, value: float, value_lims: Tuple[float, float] = None):
        super().__init__(value)
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


class GridParameter(RayParameter):

    def __init__(self, value: Union[List[float], np.ndarray]):
        self.value = np.array(value).flatten()

    def get_value(self) -> np.ndarray:
        return self.value

    def expand(self) -> List[NumericalParameter]:
        return [NumericalParameter(value=value) for value in self.value]

    def clone(self) -> GridParameter:
        return GridParameter(value=self.value)


class RayParameterContainer(OrderedDict[str, RayParameter]):

    def __setitem__(self, k: Union[str, XmlElement], v: RayParameter) -> None:
        if isinstance(k, XmlElement):
            k = self._element_to_key(k)
        super().__setitem__(k, v)

    def __getitem__(self, k: Union[str, XmlElement]) -> RayParameter:
        if isinstance(k, XmlElement):
            k = self._element_to_key(k)
        return super().__getitem__(k)

    def clone(self) -> RayParameterContainer:
        dict_copy = self.copy()
        for key, param in self.items():
            dict_copy[key] = param.clone()
        return dict_copy

    def to_value_dict(self):
        value_dict = dict()
        for key, param in self.items():
            value_dict[key] = param.get_value()
        return value_dict

    @staticmethod
    def _element_to_key(element: XmlElement) -> str:
        return '.'.join(element.get_full_path().split('.')[2:])
