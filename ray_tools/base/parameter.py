from __future__ import annotations

from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from itertools import product
from typing import Tuple, Union, List, Any, Dict

import numpy as np

from raypyng.xmltools import XmlElement

from .utils import RandomGenerator


class RayParameter(metaclass=ABCMeta):
    """
    Base class for parameters used in RayParameterContainer.
    Warning: ``get_value()`` should have a deterministic output,
    i.e., calling it twice should return the same value.
    """

    @abstractmethod
    def get_value(self) -> Any:
        """
        :return: Parameter value.
        """
        pass

    @abstractmethod
    def clone(self) -> RayParameter:
        """
        :return: Deep copy of this parameter.
        """
        pass

    def __repr__(self) -> str:
        return self.__class__.__name__ + ': ' + str(self.get_value())


class NumericalParameter(RayParameter):
    """
    Base class for all numerical parameters, represented as a float.
    """

    def __init__(self, value: float):
        self.value = value

    def get_value(self) -> float:
        return self.value

    def clone(self) -> NumericalParameter:
        return NumericalParameter(value=self.value)


class MutableParameter(NumericalParameter):
    """
    Extends :class:`NumericalParameter` by value_lims, specifying lower and upper limits for value.
    """

    def __init__(self, value: float, value_lims: Tuple[float, float] = None):
        super().__init__(value)
        self.value_lims = value_lims

    def get_value(self) -> float:
        return self.value

    def clone(self) -> MutableParameter:
        return MutableParameter(value=self.value, value_lims=self.value_lims)


class RandomParameter(MutableParameter):
    """
    Draws a random value in the interval value_lims.
    """

    def __init__(self, value_lims: Tuple[float, float] = None, rg: RandomGenerator = None):
        self.rg = rg if rg is not None else RandomGenerator()
        value = self.rg.rg_random.uniform(*value_lims)
        super().__init__(value, value_lims)

    def resample(self) -> None:
        """
        Resample value from value_lims.
        """
        self.value = self.rg.rg_random.uniform(*self.value_lims)

    def clone(self) -> RandomParameter:
        # Note: the value of the cloned parameter is not resampled but also copied.
        param = RandomParameter(self.value_lims, rg=self.rg)
        param.value = self.value
        return param


class GridParameter(RayParameter):
    """
    Represents a grid parameter, containing a numpy-array of values.
    Can be used to build a grid of :class:`RayParameterContainer`.
    """

    def __init__(self, value: Union[List[float], np.ndarray]):
        self.value = np.array(value).flatten()

    def get_value(self) -> np.ndarray:
        return self.value

    def expand(self) -> List[NumericalParameter]:
        """
        Expands values into list of individual :class:`NumericalParameter`.
        """
        return [NumericalParameter(value=value) for value in self.value]

    def clone(self) -> GridParameter:
        return GridParameter(value=self.value)


class RayParameterContainer(OrderedDict[str, RayParameter]):
    """
    Container class for :class:`RayParameter`. Is actually an :class:`OrderedDict`,
    and extends it by a few useful features.
    Key format: ``beamline_component.parameter`` (internal format) or corresponding :class:`XmlElement`.
    """

    def __setitem__(self, k: Union[str, XmlElement], v: RayParameter) -> None:
        if isinstance(k, XmlElement):
            k = self._element_to_key(k)  # get string-key
        super().__setitem__(k, v)

    def __getitem__(self, k: Union[str, XmlElement]) -> RayParameter:
        if isinstance(k, XmlElement):
            k = self._element_to_key(k)  # get string-key
        return super().__getitem__(k)

    def clone(self) -> RayParameterContainer:
        """
        :return: Deep copy of container.
        """
        dict_copy = self.copy()
        for key, param in self.items():
            dict_copy[key] = param.clone()
        return dict_copy

    def perturb(self, perturbation_dict: Dict[Union[str, XmlElement], RayParameter]):
        for k, v in perturbation_dict.items():
            if isinstance(self[k], NumericalParameter) and isinstance(v, NumericalParameter):
                self[k].value += v.get_value()

    def to_value_dict(self) -> Dict[str, Any]:
        """
        Converts container into an ordinary dictionary with values.
        """
        value_dict = dict()
        for key, param in self.items():
            value_dict[key] = param.get_value()
        return value_dict

    @staticmethod
    def _element_to_key(element: XmlElement) -> str:
        """
        Builds string-key from a given :class:`XmlElement`.
        """
        return '.'.join(element.get_full_path().split('.')[2:])


def build_parameter_grid(param_container: RayParameterContainer):
    """
    Builds a list of :class:`RayParameterContainer` based on a container with (multiple) :class:`GridParameter`.
    Each GridParameter is expanded and every individual parameter is combined with every other.
    See :func:`itertools.product`.
    """
    param_list = list(param_container.items())
    param_list_expanded = []
    for param in param_list:
        if isinstance(param[1], GridParameter):
            # expand grid parameter and create sublist of individual parameters
            param_list_expanded.append([(param[0], p) for p in param[1].expand()])
        else:
            # wrap non-grid parameters into sublist (needed for itertools.product)
            param_list_expanded.append([param])

    return [RayParameterContainer(list(param_list)) for param_list in product(*param_list_expanded)]
