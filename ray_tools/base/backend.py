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

import podman

import h5py
import numpy as np
import pandas as pd

from raypyng import RMLFile


@dataclass
class RayOutput:
    """
    Stores the output of a raytracing simulation.

    ``x_loc``, ``y_loc``, ``z_loc``: position of rays at exported plane.

    ``x_dir``, ``y_dir``, ``z_dir``: direction coordinates of rays at exported plane.

    ``energy``: energy of rays at exported plane.
    """
    x_loc: np.ndarray
    y_loc: np.ndarray
    z_loc: np.ndarray
    x_dir: np.ndarray
    y_dir: np.ndarray
    z_dir: np.ndarray
    energy: np.ndarray


class RayBackend(metaclass=ABCMeta):
    """
    Base class for raytracing backend.
    """

    @abstractmethod
    def run(self, raypyng_rml: RMLFile, exported_planes: List[str], run_id: str = None) -> Dict[str, RayOutput]:
        """
        Run raytracing given an RMLFile instance.
        :param raypyng_rml: RMLFile instance to be processed.
        :param exported_planes: Image planes and component outputs to be exported.
        :param run_id: Run identifier (as string).
        :return: Dict with 'exported_planes' as keys and generated :class:`RayOutput` instances as values.
        """
        pass


class RayBackendDockerRAYUI(RayBackend):
    """
    Creates a Ray-UI backend within a docker container.
    :param docker_image: Tag of docker image to be used.
    :param ray_workdir: Local directory where temporary RML-files and exports are processed.
    :param docker_container_name: Name of corresponding docker container (``docker_image`` + '_backend' is None)
    :param max_retry: Number of retries if Ray-UI fails.
    :param verbose: Show detailed outputs.
    """

    def __init__(self,
                 docker_image: str,
                 ray_workdir: str,
                 docker_container_name: str = None,
                 max_retry: int = 1000,
                 verbose=True) -> None:
        super().__init__()
        self.docker_image = docker_image
        self.ray_workdir = os.path.abspath(ray_workdir)
        self.docker_container_name = docker_container_name if docker_container_name else self.docker_image + '_backend'
        self.max_retry = max_retry
        self.verbose = verbose

        # Ray-UI workdir in docker container
        self._rayui_workdir = '/opt/ray-ui-workdir'

        self.client = docker.from_env()
        # if container already exists, stop and remove it
        try:
            self.docker_container = self.client.containers.get(self.docker_container_name)
            print(f'Docker container {self.docker_container_name} already exists.\n' + 'Stopping and recreating...')
            self.docker_container.stop()
            try:
                self.docker_container.remove()
            except docker.errors.APIError:
                pass
        except docker.errors.NotFound:
            pass



    def kill(self):
        """
        Send kill signal to docker container.
        """
        try:
            #self.docker_container.kill()
            print("Kill")
        except docker.errors.NotFound:
            pass

    def __del__(self):
        self.kill()

    def run(self,
            raypyng_rml: RMLFile,
            exported_planes: List[str],
            run_id: str = None) -> Dict[str, RayOutput]:

        # create random id if not given
        if run_id is None:
            run_id = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(16))

        # create local Ray-UI workdir (should be done before mounting docker directory below)
        os.makedirs(self.ray_workdir, exist_ok=True)

        # create sub-workdir (required for multi-threading with Ray-UI)
        run_workdir = os.path.join(self.ray_workdir, run_id)
        os.makedirs(run_workdir, exist_ok=True)

        tic = time.perf_counter()
       # create RML-file for this run
        rml_workfile = os.path.join(run_workdir, 'workfile.rml')
        raypyng_rml.write(rml_workfile)

        # get corresponding RML-file for in docker container
        docker_rml_workfile = os.path.join(self._rayui_workdir, run_id, os.path.basename(rml_workfile)).replace("\\","/")

        # argument for planes to be exported
        cmd_exported_planes = " ".join("\"" + plane + "\"" for plane in exported_planes)
        # run Ray-UI background mode in docker and retry if it failes


        for run in range(self.max_retry + 1):
            docker_command = "docker run -v "+self.ray_workdir+':'+self._rayui_workdir+':rw ray-ui-service ray-ui -m '+docker_rml_workfile+' '+cmd_exported_planes+' RawRaysBeam'
            #os.system(docker_command)
            print(docker_command)
            self.docker_container = self.client.containers.run(
                self.docker_image,
                name=self.docker_container_name+run_id,
                volumes={self.ray_workdir: {'bind': self._rayui_workdir, 'mode': 'rw'}},  # mount Ray-UI workdir
                detach=False,
                auto_remove=True,  # remove container after kill or stop
                entrypoint=['ray-ui', '-m', docker_rml_workfile, 'ImagePlane', 'RawRaysBeam']
            )
            retry = False
            # fail indicator: any required CSV-file is missing
            for exported_plane in exported_planes:
                if not os.path.isfile(os.path.join(run_workdir, exported_plane + '-RawRaysBeam.csv').replace("\\","/")):
                    print('Could not find' + os.path.join(run_workdir, exported_plane + '-RawRaysBeam.csv'))
                    retry = True
            if not retry:
                break
            else:
                print(f"Run {run + 1} failed. Retry ...")

        # extract Ray-UI output from CSV-files and create RayOutput instances
        ray_output = {}
        for exported_plane in exported_planes:
            ray_output_file = os.path.join(run_workdir, exported_plane + '-RawRaysBeam.csv').replace("\\","/")

            raw_output = pd.read_csv(ray_output_file, sep='\t', skiprows=1, engine='c',
                                     usecols=[exported_plane + '_OX', exported_plane + '_OY', exported_plane + '_OZ',
                                              exported_plane + '_DX', exported_plane + '_DY', exported_plane + '_DZ',
                                              exported_plane + '_EN', exported_plane + '_PL'])

            ray_output[exported_plane] = RayOutput(x_loc=raw_output[exported_plane + '_OX'].to_numpy(dtype=np.float32),
                                                   y_loc=raw_output[exported_plane + '_OY'].to_numpy(dtype=np.float32),
                                                   z_loc=raw_output[exported_plane + '_OZ'].to_numpy(dtype=np.float32),
                                                   x_dir=raw_output[exported_plane + '_DX'].to_numpy(dtype=np.float32),
                                                   y_dir=raw_output[exported_plane + '_DY'].to_numpy(dtype=np.float32),
                                                   z_dir=raw_output[exported_plane + '_DZ'].to_numpy(dtype=np.float32),
                                                   energy=raw_output[exported_plane + '_EN'].to_numpy(dtype=np.float32))

        # remove sub-workdir
        shutil.rmtree(run_workdir)

        toc = time.perf_counter()
        if self.verbose:
            print(f'Run ID {run_id}: Ray-UI output from {os.path.basename(rml_workfile)}' +
                  f' successfully generated in {toc - tic:.2f}s')

        return ray_output

class RayBackendPodmanRAYUI(RayBackend):
    """
    Creates a Ray-UI backend within a docker container.
    :param docker_image: Tag of docker image to be used.
    :param ray_workdir: Local directory where temporary RML-files and exports are processed.
    :param docker_container_name: Name of corresponding docker container (``docker_image`` + '_backend' is None)
    :param max_retry: Number of retries if Ray-UI fails.
    :param verbose: Show detailed outputs.
    """

    def __init__(self,
                 docker_image: str,
                 ray_workdir: str,
                 docker_container_name: str = None,
                 max_retry: int = 1000,
                 verbose=True) -> None:
        super().__init__()
        self.docker_image = docker_image
        self.ray_workdir = os.path.abspath(ray_workdir)
        self.docker_container_name = docker_container_name if docker_container_name else self.docker_image + '_backend'
        self.max_retry = max_retry
        self.verbose = verbose

        # Ray-UI workdir in docker container
        self._rayui_workdir = '/opt/ray-ui-workdir'

        self.client = docker.from_env()
        self.client = podman.PodmanClient(base_url='npipe:////./pipe/podman-machine-default')
        #self.client = podman.PodmanClient(base_url="unix:////run/user/1003/podman/podman.sock")

        # if container already exists, stop and remove it
        try:
            self.docker_container = self.client.containers.get(self.docker_container_name)
            print(f'Docker container {self.docker_container_name} already exists.\n' + 'Stopping and recreating...')
            self.docker_container.stop()
            try:
                self.docker_container.remove()
            except docker.errors.APIError:
                pass
        except docker.errors.NotFound:
            pass

        # create local Ray-UI workdir (should be done before mounting docker directory below)
        os.makedirs(self.ray_workdir, exist_ok=True)

        self.docker_container = self.client.containers.run(
            self.docker_image,
            name=self.docker_container_name,
            volumes={self.ray_workdir: {'bind': self._rayui_workdir, 'mode': 'rw'}},  # mount Ray-UI workdir
            detach=True,  # run in background
            auto_remove=True,  # remove container after kill or stop
        )

    def kill(self):
        """
        Send kill signal to docker container.
        """
        try:
            #self.docker_container.kill()
            print("Kill")
        except docker.errors.NotFound:
            pass

    def __del__(self):
        self.kill()

    def run(self,
            raypyng_rml: RMLFile,
            exported_planes: List[str],
            run_id: str = None) -> Dict[str, RayOutput]:

        # create random id if not given
        if run_id is None:
            run_id = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(16))

        # create sub-workdir (required for multi-threading with Ray-UI)
        run_workdir = os.path.join(self.ray_workdir, run_id)
        os.makedirs(run_workdir, exist_ok=True)

        tic = time.perf_counter()

        # create RML-file for this run
        rml_workfile = os.path.join(run_workdir, 'workfile.rml').replace("\\","/")
        raypyng_rml.write(rml_workfile)

        # get corresponding RML-file for in docker container
        docker_rml_workfile = os.path.join(self._rayui_workdir, run_id, os.path.basename(rml_workfile)).replace("\\","/")

        # argument for planes to be exported
        cmd_exported_planes = " ".join("\"" + plane + "\"" for plane in exported_planes)
        # run Ray-UI background mode in docker and retry if it failes
        for run in range(self.max_retry + 1):
            self.docker_container.exec_run(
                cmd=f"python /opt/script_rayui_bg.py {docker_rml_workfile} -p {cmd_exported_planes}"
            )
            retry = False
            # fail indicator: any required CSV-file is missing
            for exported_plane in exported_planes:
                if not os.path.isfile(os.path.join(run_workdir, exported_plane + '-RawRaysBeam.csv')).replace("\\","/"):
                    retry = True
            if not retry:
                break
            else:
                print(f"Run {run + 1} failed. Retry ...")

        # extract Ray-UI output from CSV-files and create RayOutput instances
        ray_output = {}
        for exported_plane in exported_planes:
            ray_output_file = os.path.join(run_workdir, exported_plane + '-RawRaysBeam.csv').replace("\\","/")

            raw_output = pd.read_csv(ray_output_file, sep='\t', skiprows=1, engine='c',
                                     usecols=[exported_plane + '_OX', exported_plane + '_OY',
                                              exported_plane + '_OZ',
                                              exported_plane + '_DX', exported_plane + '_DY',
                                              exported_plane + '_DZ',
                                              exported_plane + '_EN', exported_plane + '_PL'])

            ray_output[exported_plane] = RayOutput(
                x_loc=raw_output[exported_plane + '_OX'].to_numpy(dtype=np.float32),
                y_loc=raw_output[exported_plane + '_OY'].to_numpy(dtype=np.float32),
                z_loc=raw_output[exported_plane + '_OZ'].to_numpy(dtype=np.float32),
                x_dir=raw_output[exported_plane + '_DX'].to_numpy(dtype=np.float32),
                y_dir=raw_output[exported_plane + '_DY'].to_numpy(dtype=np.float32),
                z_dir=raw_output[exported_plane + '_DZ'].to_numpy(dtype=np.float32),
                energy=raw_output[exported_plane + '_EN'].to_numpy(dtype=np.float32))

        # remove sub-workdir
        shutil.rmtree(run_workdir)

        toc = time.perf_counter()
        if self.verbose:
            print(f'Run ID {run_id}: Ray-UI output from {os.path.basename(rml_workfile)}' +
                  f' successfully generated in {toc - tic:.2f}s')

        return ray_output


class RayBackendDockerRAYX(RayBackend):
    """
    Creates a Ray-X backend within a docker container.
    Currently only a single image plane can be exported.
    This backend is still experimental and needs to be revised.
    """

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
            try:
                self.docker_container.remove()
            except docker.errors.APIError:
                pass
        except docker.errors.NotFound:
            pass

        if self.gpu_ids:
            _devices = [docker.types.DeviceRequest(device_ids=self.gpu_ids, capabilities=[['gpu']])]
        else:
            _devices = []

        os.makedirs(self.ray_workdir, exist_ok=True)
        self.docker_container = self.client.containers.run(
            self.docker_image,
            name=self.docker_container_name,
            volumes={self.ray_workdir: {'bind': self._rayx_workdir, 'mode': 'rw'}},
            detach=True,
            auto_remove=True,
            device_requests=_devices,
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

        os.makedirs(self.ray_workdir, exist_ok=True)

        tic = time.perf_counter()
        rml_workfile = os.path.join(self.ray_workdir, run_id + '.rml').replace("\\","/")
        raypyng_rml.write(rml_workfile)

        docker_rml_workfile = os.path.join(self._rayx_workdir, os.path.basename(rml_workfile)).replace("\\","/")
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

        toc = time.perf_counter()
        if self.verbose:
            print(f'Run ID {run_id}: Ray-X output from {os.path.basename(rml_workfile)}' +
                  f' successfully generated in {toc - tic:.2f}s')

        return ray_output
