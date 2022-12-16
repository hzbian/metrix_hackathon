from __future__ import annotations

import os
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Union, Any, Dict

from collections import OrderedDict

from joblib import Parallel, delayed
import subprocess
import h5py

import numpy as np
import pandas as pd

from raypyng import RMLFile
from raypyng.xmltools import XmlElement

from .parameter import RayParameter


@dataclass
class RayOutput:
    x_loc: np.ndarray
    y_loc: np.ndarray
    z_loc: np.ndarray
    x_dir: np.ndarray
    y_dir: np.ndarray
    z_dir: np.ndarray


class RayTransform(metaclass=ABCMeta):

    @abstractmethod
    def __call__(self, ray_output: RayOutput) -> Any:
        pass


class RayBackend(metaclass=ABCMeta):

    @abstractmethod
    def run(self, rml_filename: str) -> RayOutput:
        pass


class RayBackendRayX(RayBackend):

    def __init__(self, ray_x_path='/RAY-X/build/bin/debug/TerminalApp', verbose=True) -> None:
        super().__init__()
        self.ray_x_path = ray_x_path
        self.verbose = verbose

    def run(self, rml_workfile: str) -> RayOutput:
        subprocess.call([self.ray_x_path, "-x", "-i", rml_workfile], stdout=subprocess.DEVNULL)

        work_path = os.path.dirname(rml_workfile)
        ray_output_file = os.path.join(work_path, os.path.splitext(os.path.basename(rml_workfile))[0] + '.h5')

        with h5py.File(ray_output_file, 'r') as h5f:
            # default, Columns are hard-coded, to be changed if necessary
            _keys = [int(idx) for idx in list(h5f.keys())]
            _keys.sort()
            _dfs = []
            for key in _keys:
                dataset = h5f[str(key)]
                _df = pd.DataFrame(dataset, columns=['Xloc', 'Yloc', 'Zloc', 'Weight', 'Xdir', 'Ydir', 'Zdir', 'Energy',
                                                     'Stokes0', 'Stokes1', 'Stokes2', 'Stokes3', 'pathLength', 'order',
                                                     'lastElement', 'extraParam'])
                _dfs.append(_df)
            # concat once done otherwise, too memory intensive
            raw_output = pd.concat(_dfs, axis=0)

        os.remove(ray_output_file)

        # TODO: replace by transform
        raw_output_abs = raw_output.abs()
        raw_output = raw_output[(raw_output_abs['Xloc']) < 1 & (raw_output_abs['Yloc'] < 1)]

        ray_output = RayOutput(x_loc=raw_output['Xloc'].to_numpy(),
                               y_loc=raw_output['Yloc'].to_numpy(),
                               z_loc=raw_output['Zloc'].to_numpy(),
                               x_dir=raw_output['Xdir'].to_numpy(),
                               y_dir=raw_output['Ydir'].to_numpy(),
                               z_dir=raw_output['Zdir'].to_numpy())

        if self.verbose:
            print('Ray output from ' + os.path.basename(rml_workfile) + ' successfully generated')

        return ray_output


class RayParameterDict(OrderedDict[str, RayParameter]):

    def __setitem__(self, k: Union[str, XmlElement], v: RayParameter) -> None:
        # TODO: check if string format is correct
        if isinstance(k, XmlElement):
            k = self._element_to_key(k)
        super().__setitem__(k, v)

    def __getitem__(self, k: Union[str, XmlElement]) -> RayParameter:
        if isinstance(k, XmlElement):
            k = self._element_to_key(k)
        return super().__getitem__(k)

    def clone(self) -> RayParameterDict:
        dict_copy = self.copy()
        for key, param in self.items():
            dict_copy[key] = param.clone()
        return dict_copy

    @staticmethod
    def _element_to_key(element: XmlElement) -> str:
        return '.'.join(element.get_full_path().split('.')[2:])


class RayEngine:

    def __init__(self,
                 rml_basefile: str,
                 ray_backend: RayBackend,
                 work_path: str = 'ray_tmp',
                 transform: RayTransform = None,
                 num_workers: int = 1,
                 as_generator: bool = False,
                 ) -> None:
        super().__init__()
        self.rml_basefile = rml_basefile
        self.ray_backend = ray_backend
        self.work_path = work_path
        self.transform = transform
        self.num_workers = num_workers
        self.as_generator = as_generator

        self._raypyng_rml = RMLFile(self.rml_basefile)
        self.template = self._raypyng_rml.beamline

    def run(self, params: Union[RayParameterDict, Iterable[RayParameterDict]]) -> Union[Dict, Iterable[Dict]]:
        os.makedirs(self.work_path, exist_ok=True)

        if isinstance(params, RayParameterDict):
            params = [params]

        _iter = ((str(run_id), run_params) for run_id, run_params in enumerate(params))
        if not self.as_generator:
            # TODO: Is use of threading safe?
            worker = Parallel(n_jobs=self.num_workers, verbose=False, backend='threading')
            jobs = (delayed(self._run_func)(*item) for item in _iter)
            result = worker(jobs)
            return result if len(result) > 1 else result[0]
        else:
            return (self._run_func(*item) for item in _iter)

    def _run_func(self, run_id: str, param_dict: RayParameterDict) -> Dict:
        # TODO: what other info should be returned?
        result = {'values': OrderedDict(), 'ray_output': None}

        raypyng_rml_work = RMLFile(self.rml_basefile)
        template_work = raypyng_rml_work.beamline
        for key, param in param_dict.items():
            value = param.get_value()
            element = self._key_to_element(key, template=template_work)
            element.cdata = str(value)
            result['values'][key] = value

        rml_workfile = os.path.join(self.work_path, run_id + '.rml')
        raypyng_rml_work.write(rml_workfile)
        result['ray_output'] = self.ray_backend.run(rml_workfile)
        os.remove(rml_workfile)

        if self.transform is not None:
            result['ray_output'] = self.transform(result['ray_output'])
        return result

    def _key_to_element(self, key: str, template: XmlElement = None) -> XmlElement:
        if template is None:
            template = self.template
        component, param = key.split('.')
        return template.__getattr__(component).__getattr__(param)
