from __future__ import annotations

from typing import Iterable, Union, Dict, List

from joblib import Parallel, delayed

from raypyng import RMLFile
from raypyng.xmltools import XmlElement

from . import RayTransformType
from .backend import RayBackend
from .parameter import RayParameterContainer
from .transform import RayTransform



class RayEngine:

    def __init__(self,
                 rml_basefile: str,
                 exported_planes: List[str],
                 ray_backend: RayBackend,
                 num_workers: int = 1,
                 as_generator: bool = False,
                 verbose: bool = False,
                 ) -> None:
        super().__init__()
        self.rml_basefile = rml_basefile
        self.exported_planes = exported_planes
        self.ray_backend = ray_backend
        self.num_workers = num_workers
        self.as_generator = as_generator
        self.verbose = verbose

        self._raypyng_rml = RMLFile(self.rml_basefile)
        self.template = self._raypyng_rml.beamline

    def run(self,
            param_containers: Union[RayParameterContainer, Iterable[RayParameterContainer]],
            transforms: Union[RayTransformType, Iterable[RayTransformType]] = None,
            ) -> Union[Dict, Iterable[Dict], List[Dict]]:

        if isinstance(param_containers, RayParameterContainer):
            param_containers = [param_containers]

        if transforms is None or isinstance(transforms, (RayTransform, dict)):
            transforms = len(param_containers) * [transforms]

        _iter = ((str(run_id), run_params, transform) for run_id, (run_params, transform) in
                 enumerate(zip(param_containers, transforms)))
        if not self.as_generator:
            worker = Parallel(n_jobs=self.num_workers, verbose=self.verbose, backend='threading')
            jobs = (delayed(self._run_func)(*item) for item in _iter)
            result = worker(jobs)
            return result if len(result) > 1 else result[0]
        else:
            return (self._run_func(*item) for item in _iter)

    def _run_func(self,
                  run_id: str,
                  param_container: RayParameterContainer,
                  transform: RayTransformType = None,
                  ) -> Dict:
        result = {'param_container_dict': dict(), 'ray_output': None}

        raypyng_rml_work = RMLFile(self.rml_basefile)
        template_work = raypyng_rml_work.beamline
        for key, param in param_container.items():
            value = param.get_value()
            element = self._key_to_element(key, template=template_work)
            element.cdata = str(value)
            result['param_container_dict'][key] = value

        result['ray_output'] = self.ray_backend.run(raypyng_rml=raypyng_rml_work,
                                                    run_id=run_id,
                                                    exported_planes=self.exported_planes)

        if transform is not None:
            for plane in self.exported_planes:
                t = transform if isinstance(transform, RayTransform) else transform[plane]
                result['ray_output'][plane] = t(result['ray_output'][plane])
        return result

    def _key_to_element(self, key: str, template: XmlElement = None) -> XmlElement:
        if template is None:
            template = self.template
        component, param = key.split('.')
        return template.__getattr__(component).__getattr__(param)
