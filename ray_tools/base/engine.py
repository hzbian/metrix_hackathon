from __future__ import annotations

import os
from typing import Iterable, Union, Dict, List

from joblib import Parallel, delayed

from raypyng import RMLFile
from raypyng.xmltools import XmlElement

from .backend import RayBackend
from .parameter import RayParameterContainer
from .transform import RayTransform


class RayEngine:

    def __init__(self,
                 rml_basefile: str,
                 ray_backend: RayBackend,
                 workdir: str = 'ray_workdir',
                 num_workers: int = 1,
                 as_generator: bool = False,
                 ) -> None:
        super().__init__()
        self.rml_basefile = rml_basefile
        self.ray_backend = ray_backend
        self.workdir = os.path.abspath(workdir)
        self.num_workers = num_workers
        self.as_generator = as_generator

        self._raypyng_rml = RMLFile(self.rml_basefile)
        self.template = self._raypyng_rml.beamline

    def run(self,
            param_containers: Union[RayParameterContainer, Iterable[RayParameterContainer]],
            transforms: Union[RayTransform, Iterable[RayTransform]] = None,
            ) -> Union[Dict, Iterable[Dict], List[Dict]]:
        os.makedirs(self.workdir, exist_ok=True)

        if not isinstance(param_containers, Iterable):
            param_containers = [param_containers]

        if transforms is None or not isinstance(transforms, Iterable):
            transforms = len(param_containers) * [transforms]

        _iter = ((str(run_id), run_params, transform) for run_id, (run_params, transform) in
                 enumerate(zip(param_containers, transforms)))
        if not self.as_generator:
            # TODO: Is multiprocessing possible here?
            worker = Parallel(n_jobs=self.num_workers, verbose=False, backend='threading')
            jobs = (delayed(self._run_func)(*item) for item in _iter)
            result = worker(jobs)
            return result if len(result) > 1 else result[0]
        else:
            return (self._run_func(*item) for item in _iter)

    def _run_func(self,
                  run_id: str,
                  param_container: RayParameterContainer,
                  transform: RayTransform = None,
                  ) -> Dict:
        # TODO: Good idea to return param_container?
        result = {'param_container': param_container.clone(), 'ray_output': None}

        raypyng_rml_work = RMLFile(self.rml_basefile)
        template_work = raypyng_rml_work.beamline
        for key, param in param_container.items():
            value = param.get_value()
            element = self._key_to_element(key, template=template_work)
            element.cdata = str(value)

        rml_workfile = os.path.join(self.workdir, run_id + '.rml')
        raypyng_rml_work.write(rml_workfile)
        result['ray_output'] = self.ray_backend.run(rml_workfile)

        os.remove(rml_workfile)

        if transform is not None:
            result['ray_output'] = list(map(transform, result['ray_output']))
        return result

    def _key_to_element(self, key: str, template: XmlElement = None) -> XmlElement:
        if template is None:
            template = self.template
        component, param = key.split('.')
        return template.__getattr__(component).__getattr__(param)
