import os
from itertools import compress
from typing import Callable, List, Dict

import h5py
import numpy as np

from ..base.backend import RayBackend
from ..base.engine import RayEngine
from ..base.parameter import RayParameterContainer
from ..base.transform import RayTransform


class RandomDatasetGenerator:

    def __init__(self,
                 rml_basefile: str,
                 ray_workdir: str,
                 ray_backend: RayBackend,
                 random_param_container: Callable[..., Dict],
                 h5_datadir: str,
                 h5_basename: str = 'raw',
                 h5_max_size: int = 1000,
                 num_workers: int = 1,
                 ):
        self.rml_basefile = rml_basefile
        self.ray_workdir = ray_workdir
        self.ray_backend = ray_backend
        self.random_param_container = random_param_container
        self.h5_basename = h5_basename
        self.h5_max_size = h5_max_size
        self.h5_datadir = h5_datadir
        self.num_workers = num_workers

        self._ray_engine = RayEngine(rml_basefile=self.rml_basefile,
                                     ray_backend=self.ray_backend,
                                     workdir=self.ray_workdir,
                                     num_workers=self.num_workers,
                                     as_generator=False)

    def generate(self, h5_idx: int, batch_size: int = -1) -> None:

        if batch_size == -1:
            batch_size = self.h5_max_size

        h5_file = h5py.File(os.path.join(self.h5_datadir, self.h5_basename + str(h5_idx) + '.h5'), "w")

        idx_abs = 0
        while idx_abs < self.h5_max_size:
            params = dict(ids_sample=[], param_containers=[], ids=[], transforms=[])
            for idx_sample in range(idx_abs, min(idx_abs + batch_size, self.h5_max_size)):
                params_cur = self.random_param_container()
                params['ids_sample'] += len(params_cur['ids']) * [idx_sample]
                params['param_containers'] += params_cur['param_containers']
                params['ids'] += params_cur['ids']
                params['transforms'] += params_cur['transforms']

            ray_results = list(self._ray_engine.run(param_containers=params['param_containers'],
                                                    transforms=params['transforms']))
            idx_abs += batch_size

            params_len = len(ray_results)
            for idx in range(params_len):
                id_sample = params['ids_sample'][idx]
                id_ = params['ids'][idx]
                param_container: RayParameterContainer = params['param_containers'][idx]
                ray_outputs = ray_results[idx]['ray_output']

                sample_grp = h5_file.create_group(f'/{id_sample}/{id_}')
                param_container_grp = sample_grp.create_group('params')
                dict_to_h5(param_container_grp, param_container.to_value_dict())

                for ray_output in ray_outputs:
                    ray_output_name = ray_output['name']
                    ray_output_grp = sample_grp.create_group(f'ray_output/{ray_output_name}')
                    dict_to_h5(ray_output_grp, ray_output)

                print(f'Sample {id_sample} / {id_} written')

        h5_file.close()


def dict_to_h5(h5_grp: h5py.Group, d: Dict):
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            h5_grp.create_dataset(name=str(k), data=v, compression='gzip')
        else:
            h5_grp.create_dataset(name=str(k), data=v)



def build_random_param_container(param_container_func: Callable[..., List[RayParameterContainer]],
                                 ids: List[str],
                                 transforms: List[RayTransform]) -> Callable[..., Dict]:
    return lambda: dict(ids=ids, transforms=transforms, param_containers=param_container_func())
