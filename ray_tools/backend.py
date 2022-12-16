from __future__ import annotations

import os
import docker
import docker.errors
import docker.types

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

import h5py
import numpy as np
import pandas as pd


@dataclass
class RayOutput:
    x_loc: np.ndarray
    y_loc: np.ndarray
    z_loc: np.ndarray
    x_dir: np.ndarray
    y_dir: np.ndarray
    z_dir: np.ndarray


class RayBackend(metaclass=ABCMeta):

    @abstractmethod
    def run(self, rml_filename: str) -> RayOutput:
        pass


class RayBackendDockerRAYX(RayBackend):

    def __init__(self,
                 docker_image: str,
                 ray_workdir: str,
                 docker_container_name: str = None,
                 verbose=True) -> None:
        super().__init__()
        self.docker_image = docker_image
        self.ray_workdir = os.path.abspath(ray_workdir)
        self.docker_container_name = docker_container_name if docker_container_name else self.docker_image + '_backend'
        self.verbose = verbose

        self._rayx_workdir = '/RAY-X-workdir'
        self._rayx_path = '/RAY-X/build/bin/debug/TerminalApp'

        self.client = docker.from_env()
        try:
            self.docker_container = self.client.containers.get(self.docker_container_name)
            print(f'Docker container {self.docker_container_name} already exists.\n' + 'Stopping and recreating...')
            self.docker_container.stop()
            self.docker_container.remove()
        except docker.errors.NotFound:
            pass

        self.docker_container = self.client.containers.run(
            self.docker_image,
            name=self.docker_container_name,
            volumes={self.ray_workdir: {'bind': self._rayx_workdir, 'mode': 'rw'}},
            detach=True,
            auto_remove=True,
            # device_requests=[docker.types.DeviceRequest(device_ids=["0", "1"], capabilities=[['gpu']])]
        )

    def kill(self):
        # TODO: better way to do that?
        try:
            self.docker_container.kill()
        except docker.errors.NotFound:
            pass

    def __del__(self):
        self.kill()

    def run(self, rml_workfile: str) -> RayOutput:
        docker_rml_workfile = os.path.join(self._rayx_workdir, os.path.basename(rml_workfile))
        self.docker_container.exec_run(
            cmd=f"{self._rayx_path} -x -i {docker_rml_workfile}",
            stdout=True,
        )

        ray_output_file = os.path.splitext(rml_workfile)[0] + '.h5'

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
            print(f'Ray output from {os.path.basename(rml_workfile)} successfully generated')

        return ray_output
