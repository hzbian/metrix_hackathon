from __future__ import annotations

import functools
from abc import ABCMeta, abstractmethod
from typing import Any, Tuple, Dict, List

import numpy as np

from . import RayOutput


class RayTransform(metaclass=ABCMeta):
    """
    Base class for all transformations of :class:`RayOutput`.
    """

    @abstractmethod
    def __call__(self, ray_output: RayOutput) -> Any:
        pass


class RayTransformCompose(RayTransform):
    """
    Creates a composition of multiple :class:`RayTransform`.
    """

    def __init__(self, *functions):
        def _compose2(f1, f2):
            return lambda x: f1(f2(x))

        self._transform = functools.reduce(_compose2, functions, lambda x: x)

    def __call__(self, ray_output: RayOutput) -> Any:
        return self._transform(ray_output)


class RayTransformConcat(RayTransform):
    """
    Applies multiple :class:`RayTransform` to the same :class:`RayOutput`.
    ``transforms`` is given as a dictionary of RayTransform, and the keys are used to generate the output dictionary.
    """

    def __init__(self, transforms=Dict[str, List[RayTransform]]):
        self.transforms = transforms

    def __call__(self, ray_output: RayOutput) -> Dict:
        return {k: transform(ray_output) for k, transform in self.transforms.items()}


class RayTransformDummy(RayTransform):
    """
    Dummy transform, just the identity.
    """

    def __call__(self, ray_output: RayOutput) -> RayOutput:
        return ray_output


class ToDict(RayTransform):
    """
    Transforms a :class:`RayOutput` instance into an ordinary dictionary.
    :param ignore: Fields of :class:`RayOutput` to be ignored.
    """

    def __init__(self, ignore: List[str] = None):
        self.ignore = ignore if ignore else []

    def __call__(self, ray_output: RayOutput) -> Dict:
        out = dict(x_loc=ray_output.x_loc,
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
    """
    Crops :class:`RayOutput` according to limits in x, y, z, i.e., all outside rays are discarded.
    :param x_lims: Interval for x-coordinate.
    :param y_lims: Interval for y-coordinate.
    :param z_lims: Interval for z-coordinate.
    """

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
    """
    Transforms :class:`RayOutput` into a 2D-histogram. Ray directions and z-coordinate is discarded.
    :param n_bins: Number of bins (same for x- and y-coordinate)
    :param x_lims: Interval for x-coordinate. If None, `x_lims` and `y_lims` are automatically computed.
    :param y_lims: Interval for y-coordinate.
    :param auto_center: If True, the histogram will be computed with respect to the center of mass of the rays.
        ``x_lims`` and ``y_lims`` are shifted by the center of mass.
    """

    def __init__(self,
                 n_bins: int,
                 x_lims: Tuple[float, float] = None,
                 y_lims: Tuple[float, float] = None,
                 auto_center: bool = False):
        self.n_bins = n_bins
        self.x_lims = x_lims
        self.y_lims = y_lims
        self.auto_center = auto_center

    def __call__(self, ray_output: RayOutput) -> Dict:
        return self.compute_histogram(ray_output.x_loc, ray_output.y_loc)

    def compute_histogram(self, x_loc: np.ndarray, y_loc: np.ndarray) -> Dict:
        # store number of rays
        out = {'n_rays': x_loc.size}

        # if number of rays is zero, create a trivial histogram
        if out['n_rays'] == 0:
            out['x_lims'] = (0.0, 0.0)
            out['y_lims'] = (0.0, 0.0)
            out['histogram'] = np.zeros((self.n_bins, self.n_bins))
            return out

        # if x_lims is not given, compute histogram with automatic limits
        if self.x_lims is None:
            out['histogram'], x_lims, y_lims = np.histogram2d(x_loc, y_loc, bins=(self.n_bins, self.n_bins))
            out['x_lims'] = (x_lims[0], x_lims[-1])
            out['y_lims'] = (y_lims[0], y_lims[-1])
        else:
            if self.auto_center:
                # compute center of mass and adapt limits
                x_com = np.sum(x_loc) / x_loc.size
                y_com = np.sum(y_loc) / y_loc.size
                x_lims = (self.x_lims[0] + x_com, self.x_lims[1] + x_com)
                y_lims = (self.y_lims[0] + y_com, self.y_lims[1] + y_com)
            else:
                x_lims = self.x_lims
                y_lims = self.y_lims

            # compute histogram with given limits
            out['x_lims'] = x_lims
            out['y_lims'] = y_lims
            out['histogram'] = np.histogram2d(x_loc, y_loc,
                                              bins=(self.n_bins, self.n_bins),
                                              range=[[x_lims[0], x_lims[1]], [y_lims[0], y_lims[1]]])[0]

        return out


class MultiLayer(RayTransform):
    """
    Creates (virtual) image plane layers (as :class:`RayOutput`) at given distances from an RayOutput.
    Transforms are stored in dictionary of RayOutput, with each key indicating the distance.
    :param dist_layers: List of layer distances, relative to the original image plane layer (which is at 0).
    :param copy_directions: If True, directions and energy fields are deeply copied (increases memory).
    :param transform: Optional :class:`RayTransform` applied to each created layer.
    """

    def __init__(self, dist_layers: List[float], copy_directions: bool = True, transform: RayTransform = None):
        self.dist_layers = dist_layers
        self.copy_directions = copy_directions
        self.transform = transform

    def __call__(self, ray_output: RayOutput) -> Dict[str, RayOutput]:
        # compute x and y direction for normalized z direction (zz_dir would be 1)
        xz_dir = ray_output.x_dir / ray_output.z_dir
        yz_dir = ray_output.y_dir / ray_output.z_dir

        layers = {}
        # compute x, y, z coordinates at each layer
        for dist in self.dist_layers:
            x_cur = ray_output.x_loc + xz_dir * dist
            y_cur = ray_output.y_loc + yz_dir * dist
            z_cur = ray_output.z_loc + dist
            layers[str(dist)] = RayOutput(
                x_loc=x_cur, y_loc=y_cur, z_loc=z_cur,
                x_dir=ray_output.x_dir.copy() if self.copy_directions else ray_output.x_dir,
                y_dir=ray_output.y_dir.copy() if self.copy_directions else ray_output.y_dir,
                z_dir=ray_output.z_dir.copy() if self.copy_directions else ray_output.z_dir,
                energy=ray_output.energy.copy() if self.copy_directions else ray_output.energy)

        # apply transform if given
        if self.transform:
            return {key: self.transform(ray_output_) for key, ray_output_ in layers.items()}
        else:
            return layers
