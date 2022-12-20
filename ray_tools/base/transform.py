from __future__ import annotations

import functools
from abc import ABCMeta, abstractmethod
from typing import Any, Tuple, Dict, List

import numpy as np

from . import RayOutput


class RayTransform(metaclass=ABCMeta):

    @abstractmethod
    def __call__(self, ray_output: RayOutput) -> Any:
        pass


class RayTransformCompose(RayTransform):

    def __init__(self, *functions):
        def _compose2(f1, f2):
            return lambda x: f1(f2(x))

        self._transform = functools.reduce(_compose2, functions, lambda x: x)

    def __call__(self, ray_output: RayOutput) -> Any:
        return self._transform(ray_output)


class ToDict(RayTransform):

    def __init__(self, ignore: List[str] = None):
        self.ignore = ignore if ignore else []

    def __call__(self, ray_output: RayOutput) -> Dict:
        out = dict(name=ray_output.name,
                   x_loc=ray_output.x_loc,
                   y_loc=ray_output.y_loc,
                   z_loc=ray_output.z_loc,
                   x_dir=ray_output.x_dir,
                   y_dir=ray_output.y_dir,
                   z_dir=ray_output.z_dir,
                   energy=ray_output.energy)

        for key in self.ignore:
            del out[key]

        return out


class Crop(RayTransform):

    def __init__(self,
                 x_lims: Tuple[float, float] = None,
                 y_lims: Tuple[float, float] = None,
                 z_lims: Tuple[float, float] = None):
        self.x_lims = x_lims if x_lims else (-float('inf'), float('inf'))
        self.y_lims = y_lims if y_lims else (-float('inf'), float('inf'))
        self.z_lims = z_lims if z_lims else (-float('inf'), float('inf'))

    def __call__(self, ray_output: RayOutput) -> RayOutput:
        idx = np.logical_and.reduce([self.x_lims[0] < ray_output.x_loc,
                                     ray_output.x_loc < self.x_lims[1],
                                     self.y_lims[0] < ray_output.y_loc,
                                     ray_output.y_loc < self.y_lims[1],
                                     self.z_lims[0] < ray_output.z_loc,
                                     ray_output.z_loc < self.z_lims[1]
                                     ])

        ray_output.x_loc = ray_output.x_loc[idx]
        ray_output.y_loc = ray_output.y_loc[idx]
        ray_output.z_loc = ray_output.z_loc[idx]
        ray_output.x_dir = ray_output.x_dir[idx]
        ray_output.y_dir = ray_output.y_dir[idx]
        ray_output.z_dir = ray_output.z_dir[idx]
        ray_output.energy = ray_output.energy[idx]

        return ray_output


class Histogram(RayTransform):

    def __init__(self,
                 n_bins: int,
                 x_lims: Tuple[float, float],
                 y_lims: Tuple[float, float]):
        self.n_bins = n_bins
        # TODO: 'auto' option to crop to center of mass
        self.x_lims = x_lims
        self.y_lims = y_lims

    def __call__(self, ray_output: RayOutput) -> Dict:
        histogram = np.histogram2d(ray_output.x_loc, ray_output.y_loc,
                                   bins=(self.n_bins, self.n_bins),
                                   range=[[self.x_lims[0], self.x_lims[1]], [self.y_lims[0], self.y_lims[1]]])[0]
        return dict(name=ray_output.name,
                    histogram=histogram,
                    x_lims=self.x_lims,
                    y_lims=self.y_lims)
