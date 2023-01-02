import os
from typing import Callable, List, Dict
import posixpath as pp

import h5py
import numpy as np

from ..base import RayTransformType
from ..base.engine import RayEngine
from ..base.parameter import RayParameterContainer


class RandomRayDatasetGenerator:

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

        if batch_size == -1:
            batch_size = self.h5_max_size

        h5_file_obj = h5py.File(os.path.join(self.h5_datadir, f'{self.h5_basename}_{h5_idx}.h5'), "w")

        idx_total = 0
        while idx_total < self.h5_max_size:
            params = dict(idx_sample=[], param_container=[], idx_sub=[], transform=[])
            for idx_sample in range(idx_total, min(idx_total + batch_size, self.h5_max_size)):
                params_cur = self.param_container_sampler()
                params['idx_sample'] += len(params_cur['idx_sub']) * [idx_sample]
                params['param_container'] += params_cur['param_container']
                params['idx_sub'] += params_cur['idx_sub']
                params['transform'] += params_cur['transform']

            ray_results = list(self.ray_engine.run(param_containers=params['param_container'],
                                                   transforms=params['transform']))
            idx_total += batch_size

            params_len = len(ray_results)
            for idx in range(params_len):
                idx_sample = params['idx_sample'][idx]
                idx_sub = params['idx_sub'][idx]
                sample_grp = h5_file_obj.create_group(f'/{idx_sample}/{idx_sub}')

                params_grp = sample_grp.create_group('params')
                dict_to_h5(params_grp, ray_results[idx]['param_container_dict'])

                ray_output_grp = sample_grp.create_group('ray_output')
                dict_to_h5(ray_output_grp, ray_results[idx]['ray_output'], compress_numpy=True)

                print(f'Sample {idx_sample} / {idx_sub} written to {h5_file_obj.filename}')

        h5_file_obj.close()

    @staticmethod
    def build_param_container_sampler(param_container_func: Callable[..., List[RayParameterContainer]],
                                      idx_sub: List[str],
                                      transform: List[RayTransformType]) -> Callable[..., Dict]:
        return lambda: dict(idx_sub=idx_sub, transform=transform, param_container=param_container_func())


def dict_to_h5(h5_grp: h5py.Group, d: Dict, compress_numpy=False):
    for k, v in d.items():
        if isinstance(v, dict):
            sub_grp = h5_grp.create_group(str(k))
            dict_to_h5(sub_grp, d=v, compress_numpy=compress_numpy)
        elif isinstance(v, np.ndarray):
            h5_grp.create_dataset(name=str(k), data=v, compression='lzf' if compress_numpy else None)
        elif isinstance(v, str):
            h5_grp.create_dataset(name=str(k), data=v, dtype=h5py.string_dtype(encoding='utf-8'))
        else:
            h5_grp.create_dataset(name=str(k), data=v)


def h5_to_dict(h5_grp: h5py.Group) -> Dict:
    if isinstance(h5_grp, h5py.Dataset):
        return {pp.basename(pp.normpath(h5_grp.name)): h5_grp[()]}

    d = {}

    for k, v in h5_grp.items():
        if isinstance(v, h5py.Group):
            d[k] = h5_to_dict(v)
        elif isinstance(v, h5py.Dataset):
            if h5py.check_string_dtype(v.dtype) is not None:
                d[k] = v.asstr()[()]
            else:
                d[k] = v[()]
    return d
