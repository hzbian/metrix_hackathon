import os
from typing import Callable, List, Dict, Union, Any

import h5py
import numpy as np

from ..base import RayTransformType
from ..base.engine import RayEngine
from ..base.parameter import RayParameterContainer


class RandomRayDatasetGenerator:
    """
    Creates a raytracing dataset based on a :class:`RayParameterContainer` sampling function.
    Each dataset sample if based on a fresh call of ``param_container_sampler``.
    :param ray_engine: :class:`RayEngine` used to perform the raytracing simulations.
    :param param_container_sampler: Callable that returns a dictionary of the following format:
        ``param_container`` is a list of RayParameterContainer, specifying configs for the present sample /
        ``idx_sub`` is a list of strings providing a name for each returned RayParameterContainer
        (e.g., nunmber of rays) /
        ``transform`` is a list of RayTransform, which are applied to the output of each returned RayParameterContainer.
        See also :func:`build_param_container_sampler`.
    :param h5_datadir: Directory where h5-files are written to.
    :param h5_basename: Base string for the h5-file names.
    :param h5_max_size: Number of dataset samples written into each h5-file.
    """

    def __init__(self,
                 ray_engine: RayEngine,
                 param_container_sampler: Callable[..., Dict],
                 h5_datadir: str,
                 h5_basename: str = 'raw',
                 h5_max_size: int = 1000,
                 ):
        self.ray_engine = ray_engine
        self.param_container_sampler = param_container_sampler
        self.h5_basename = h5_basename
        self.h5_max_size = h5_max_size
        self.h5_datadir = h5_datadir

        self.ray_engine.as_generator = False

    def generate(self, h5_idx: int, batch_size: int = -1) -> None:
        """
        Generates a single h5-file with index ``h5_idx``. ``batch_size`` specifies how many configs are processed in a
        row by the ``ray_engine`` before writing them to the h5-file. Larger batch sizes increase memory.
        ``batch_size = -1`` means ``batch_size = h5_max_size``.
        """
        if batch_size == -1:
            batch_size = self.h5_max_size

        # create and open h5-file
        h5_file_obj = h5py.File(os.path.join(self.h5_datadir, f'{self.h5_basename}_{h5_idx}.h5'), "w")

        # index for total sample count
        idx_total = 0
        while idx_total < self.h5_max_size:
            # a single batch is processed
            params = dict(idx_sample=[], param_container=[], idx_sub=[], transform=[])
            for idx_sample in range(idx_total, min(idx_total + batch_size, self.h5_max_size)):
                # sample parameter config(s) and add them to the list to be processed by the ray_engine
                params_cur = self.param_container_sampler()
                params['idx_sample'] += len(params_cur['idx_sub']) * [idx_sample]
                params['param_container'] += params_cur['param_container']
                params['idx_sub'] += params_cur['idx_sub']
                params['transform'] += params_cur['transform']

            # perform raytracing simulations
            ray_results = list(self.ray_engine.run(param_containers=params['param_container'],
                                                   transforms=params['transform']))

            idx_total += batch_size

            # write raytracing results to h5-file
            params_len = len(ray_results)
            for idx in range(params_len):
                idx_sample = params['idx_sample'][idx]
                idx_sub = params['idx_sub'][idx]
                # build base h5-group /idx_sample/idx_sub
                sample_grp = h5_file_obj.create_group(f'/{idx_sample}/{idx_sub}')

                # write parameter container values to /idx_sample/idx_sub/params
                params_grp = sample_grp.create_group('params')
                dict_to_h5(params_grp, ray_results[idx]['param_container_dict'])

                # write raytracing outputs to /idx_sample/idx_sub/ray_output
                ray_output_grp = sample_grp.create_group('ray_output')
                dict_to_h5(ray_output_grp, ray_results[idx]['ray_output'], compress_numpy=True)

                print(f'Sample {idx_sample} / {idx_sub} written to {h5_file_obj.filename}')

        h5_file_obj.close()

    @staticmethod
    def build_param_container_sampler(param_container_func: Callable[..., List[RayParameterContainer]],
                                      idx_sub: List[str],
                                      transform: List[RayTransformType]) -> Callable[..., Dict]:
        """
        Helper function to conveniently build a ``param_container_sampler``.
        """
        return lambda: dict(idx_sub=idx_sub, transform=transform, param_container=param_container_func())


def dict_to_h5(h5_grp: h5py.Group, d: Dict, compress_numpy=False) -> None:
    """
    Helper to write a dictionary into a :class:`h5py.Group`.
    If a dictionary value is again a dictionary, this function is called recursively and a new h5-group is created.
    :param h5_grp: h5-group to write into.
    :param d: Dictionary to be serialized.
    :param compress_numpy: Use compression if True.
    """
    for k, v in d.items():
        if isinstance(v, dict):
            # recursive call if v is again a dictionary
            sub_grp = h5_grp.create_group(str(k))
            dict_to_h5(sub_grp, d=v, compress_numpy=compress_numpy)
        elif isinstance(v, np.ndarray):
            # write numpy array
            h5_grp.create_dataset(name=str(k), data=v, compression='lzf' if compress_numpy else None)
        elif isinstance(v, str):
            # write string
            h5_grp.create_dataset(name=str(k), data=v, dtype=h5py.string_dtype(encoding='utf-8'))
        else:
            # write other
            h5_grp.create_dataset(name=str(k), data=v)


def h5_to_dict(h5_obj: Union[h5py.Group, h5py.Dataset]) -> Any:
    """
    Helper to create a dictionary from a :class:`h5py.Group` or :class:`h5py.Dataset`.
    This function works recursively.
    If the top-level h5_obj is a dataset, it is returned directly without creating a dictionary.
    """
    if isinstance(h5_obj, h5py.Dataset):
        if h5py.check_string_dtype(h5_obj.dtype) is not None:
            return h5_obj.asstr()[()]
        else:
            return h5_obj[()]
    # if h5_obj is a h5py.Group, create a dictionary according to its keys and apply helper recursively
    d = {}
    for k, v in h5_obj.items():
        d[k] = h5_to_dict(v)
    return d
