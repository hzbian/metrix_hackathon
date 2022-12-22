from __future__ import annotations

import os
import shutil
import time
import string
import random
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import List, Dict

import docker
import docker.errors
import docker.types

import h5py
import numpy as np
import pandas as pd

from raypyng import RMLFile


@dataclass
class RayOutput:
    x_loc: np.ndarray
    y_loc: np.ndarray
    z_loc: np.ndarray
    x_dir: np.ndarray
    y_dir: np.ndarray
    z_dir: np.ndarray

    energy: np.ndarray


class RayBackend(metaclass=ABCMeta):

    @abstractmethod
    def run(self, raypyng_rml: RMLFile, exported_planes: List[str], run_id: str = None) -> Dict[str, RayOutput]:
        pass


class RayBackendDockerRAYX(RayBackend):

    def __init__(self,
                 docker_image: str,
                 ray_workdir: str,
                 docker_container_name: str = None,
                 gpu_ids: List[str] = None,
                 verbose=True) -> None:
        super().__init__()
        self.docker_image = docker_image
        self.ray_workdir = os.path.abspath(ray_workdir)
        self.docker_container_name = docker_container_name if docker_container_name else self.docker_image + '_backend'
        self.gpu_ids = gpu_ids
        self.verbose = verbose

        self._rayx_workdir = '/opt/ray-x-workdir'
        self._rayx_path = '/opt/ray-x/build/bin/debug/TerminalApp'

        self.client = docker.from_env()
        try:
            self.docker_container = self.client.containers.get(self.docker_container_name)
            print(f'Docker container {self.docker_container_name} already exists.\n' + 'Stopping and recreating...')
            self.docker_container.stop()
            self.docker_container.remove()
        except docker.errors.NotFound:
            pass

        if self.gpu_ids:
            _devices = [docker.types.DeviceRequest(device_ids=self.gpu_ids, capabilities=[['gpu']])]
        else:
            _devices = []

        self.docker_container = self.client.containers.run(
            self.docker_image,
            name=self.docker_container_name,
            volumes={self.ray_workdir: {'bind': self._rayx_workdir, 'mode': 'rw'}},
            detach=True,
            auto_remove=True,
            device_requests=_devices,
        )

    def kill(self):
        # TODO: better way to do that?
        try:
            self.docker_container.kill()
        except docker.errors.NotFound:
            pass

    def __del__(self):
        self.kill()

    def run(self,
            raypyng_rml: RMLFile,
            exported_planes: List[str],  # TODO: RAY-X can only generate results for a single image plane
            run_id: str = None) -> Dict[str, RayOutput]:

        os.makedirs(self.ray_workdir, exist_ok=True)

        if run_id is None:
            run_id = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(16))

        time_start = time.time()
        rml_workfile = os.path.join(self.ray_workdir, run_id + '.rml')
        raypyng_rml.write(rml_workfile)

        docker_rml_workfile = os.path.join(self._rayx_workdir, os.path.basename(rml_workfile))
        self.docker_container.exec_run(
            cmd=f"{self._rayx_path} -x -i {docker_rml_workfile}"
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

        os.remove(rml_workfile)
        os.remove(ray_output_file)

        ray_output = {'ImagePlane': RayOutput(x_loc=raw_output['Xloc'].to_numpy(),
                                              y_loc=raw_output['Yloc'].to_numpy(),
                                              z_loc=raw_output['Zloc'].to_numpy(),
                                              x_dir=raw_output['Xdir'].to_numpy(),
                                              y_dir=raw_output['Ydir'].to_numpy(),
                                              z_dir=raw_output['Zdir'].to_numpy(),
                                              energy=raw_output['Energy'].to_numpy())
                      }

        time_end = time.time()
        if self.verbose:
            print(f'Ray output from {os.path.basename(rml_workfile)}' +
                  ' successfully generated in {:.2f}s'.format(time_end - time_start))

        return ray_output


class RayBackendDockerRAYUI(RayBackend):

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

        self._rayui_workdir = '/opt/ray-ui-workdir'

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
            volumes={self.ray_workdir: {'bind': self._rayui_workdir, 'mode': 'rw'}},
            detach=True,
            auto_remove=True,
        )

    def kill(self):
        try:
            self.docker_container.kill()
        except docker.errors.NotFound:
            pass

    def __del__(self):
        self.kill()

    def run(self,
            raypyng_rml: RMLFile,
            exported_planes: List[str],
            run_id: str = None) -> Dict[str, RayOutput]:

        if run_id is None:
            run_id = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(16))

        run_workdir = os.path.join(self.ray_workdir, run_id)
        os.makedirs(run_workdir, exist_ok=True)

        time_start = time.time()
        rml_workfile = os.path.join(run_workdir, 'workfile.rml')
        raypyng_rml.write(rml_workfile)

        docker_rml_workfile = os.path.join(self._rayui_workdir, run_id, os.path.basename(rml_workfile))
        cmd_exported_planes = " ".join("\"" + plane + "\"" for plane in exported_planes)
        self.docker_container.exec_run(
            cmd=f"python /opt/script_rayui_bg.py {docker_rml_workfile} -p {cmd_exported_planes}"
        )

        ray_output = {}
        for exported_plane in exported_planes:
            ray_output_file = os.path.join(run_workdir, exported_plane + '-RawRaysBeam.csv')

            raw_output = pd.read_csv(ray_output_file, sep='\t', skiprows=1,
                                     usecols=[exported_plane + '_OX', exported_plane + '_OY', exported_plane + '_OZ',
                                              exported_plane + '_DX', exported_plane + '_DY', exported_plane + '_DZ',
                                              exported_plane + '_EN', exported_plane + '_PL'])

            ray_output[exported_plane] = RayOutput(x_loc=raw_output[exported_plane + '_OX'].to_numpy(),
                                                   y_loc=raw_output[exported_plane + '_OY'].to_numpy(),
                                                   z_loc=raw_output[exported_plane + '_OZ'].to_numpy(),
                                                   x_dir=raw_output[exported_plane + '_DX'].to_numpy(),
                                                   y_dir=raw_output[exported_plane + '_DY'].to_numpy(),
                                                   z_dir=raw_output[exported_plane + '_DZ'].to_numpy(),
                                                   energy=raw_output[exported_plane + '_EN'].to_numpy())

        shutil.rmtree(run_workdir)

        time_end = time.time()
        if self.verbose:
            print(f'Run ID {run_id}: Ray output from {os.path.basename(rml_workfile)}' +
                  ' successfully generated in {:.2f}s'.format(time_end - time_start))

        return ray_output
