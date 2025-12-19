from __future__ import annotations

import os
import shutil
import time
import string
import random
from subprocess import DEVNULL, STDOUT
import subprocess
import shlex
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

import docker
import docker.errors
import docker.types
from docker.models.resource import Model

import h5py
import torch
import pandas as pd
import numpy as np

from .raypyng.rml import RMLFile


@dataclass
class RayOutput:
    """
    Stores the output of a raytracing simulation.

    ``x_loc``, ``y_loc``, ``z_loc``: position of rays at exported plane.

    ``x_dir``, ``y_dir``, ``z_dir``: direction coordinates of rays at exported plane.

    ``energy``: energy of rays at exported plane.
    """
    x_loc: torch.Tensor
    y_loc: torch.Tensor
    z_loc: torch.Tensor
    x_dir: torch.Tensor
    y_dir: torch.Tensor
    z_dir: torch.Tensor
    energy: torch.Tensor


class RayBackend(metaclass=ABCMeta):
    """
    Base class for raytracing backend.
    """

    @abstractmethod
    def run(self, raypyng_rml: RMLFile, exported_planes: list[str], run_id: str | None = None) -> dict[str, RayOutput]:
        """
        Run raytracing given an RMLFile instance.
    :param raypyng_rml: RMLFile instance to be processed.
    :param exported_planes: Image planes and component outputs to be exported.
    :param run_id: Run identifier (as string).
    :return: dict with 'exported_planes' as keys and generated :class:`RayOutput` instances as values.
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
    :param additional_mount_files: Paths to additional files that will be mounted in working directory
    """

    def __init__(self,
                 docker_image: str,
                 ray_workdir: str,
                 dockerfile_path: str | None = None,
                 docker_container_name: str | None = None,
                 max_retry: int = 1000,
                 verbose=True,
                 set_export_plane_in_exec=True,
                 executable='python3 /opt/script_rayui_bg.py',
                 additional_mount_files: list[str] | None = None, device: torch.device = torch.device('cpu')) -> None:
        super().__init__()
        self.docker_image = docker_image
        self.ray_workdir = os.path.abspath(ray_workdir)
        ray_workdir_name: str = os.path.basename(os.path.normpath(self.ray_workdir))
        self.docker_container_name = docker_container_name if docker_container_name else self.docker_image + ray_workdir_name + '_backend'
        self.dockerfile_path = dockerfile_path
        self.max_retry = max_retry
        self.verbose = verbose
        self.container_system = "podman"
        self.print_device = STDOUT if self.verbose else DEVNULL
        self.additional_mount_files = additional_mount_files
        self.device = device
        self.container_executable = "podman"
        self.executable = executable
        self.set_export_plane_in_exec = set_export_plane_in_exec

        # workdir in docker container
        self.docker_workdir = '/opt/ray-workdir'

        # create local Ray-UI workdir (should be done before mounting docker directory below)
        os.makedirs(self.ray_workdir, exist_ok=True)

        if self.container_system == "docker":
            self.client = docker.from_env()

        if dockerfile_path is not None:
            if self.container_system == "docker":
                self.client.images.build(path=dockerfile_path, tag=self.docker_image)
            else:
                try:
                    stop_command = f"{self.container_executable} kill {self.docker_container_name}"
                    if self.verbose:
                        print(stop_command)
                    output = subprocess.check_output(shlex.split(stop_command), stderr=self.print_device)
                    if self.verbose:
                        print(output)
                    rm_command = f"{self.container_executable} rm {self.docker_container_name}"
                    if self.verbose:
                        print(rm_command)
                    output = subprocess.check_output(shlex.split(rm_command), stderr=self.print_device)
                    if self.verbose:
                        print(output)
                except Exception:
                    if self.verbose:
                        print("Could not stop and remove podman.")
                    pass

        # if container already exists, stop and remove it
        if self.container_system == "docker":
            try:
                self.docker_container = self.client.containers.get(self.docker_container_name)
                assert isinstance(self.docker_container, Model)
                print(f'Docker container {self.docker_container_name} already exists.\n' + 'Stopping and recreating...')
                self.docker_container.stop()
                try:
                    self.docker_container.remove()
                except docker.errors.APIError:
                    pass
            except docker.errors.NotFound:
                pass
        else:
            cleanup_command = self.container_executable + " system prune -f"
            if self.verbose:
                print(cleanup_command)
            output = subprocess.check_output(shlex.split(cleanup_command), stderr=self.print_device)
            if self.verbose:
                print(output)
            if dockerfile_path is not None:
                str_path = os.path.abspath(os.path.join(dockerfile_path, 'Dockerfile'))
                build_command = self.container_executable + " build --security-opt label=disable -f {} -t {}".format(
                   str_path, self.docker_image)
                if self.verbose:
                    print(build_command)
                output = subprocess.check_output(shlex.split(build_command), stderr=self.print_device)
                if self.verbose:
                    print(output)

            podman_command = f"{self.container_executable} run -d --cgroups=disabled --security-opt label=disable --name {self.docker_container_name} --mount" \
                 f"=type=bind,src={self.ray_workdir}," \
                 f"dst={self.docker_workdir},relabel=shared -t {self.docker_image} tail -f " \
                 f"/dev/null"
            if self.verbose:
                print(podman_command)
            try:
                output = subprocess.check_output(shlex.split(podman_command), stderr=self.print_device)
                if self.verbose:
                    print(output)
            except Exception:
                if self.verbose:
                    print("Could not run podman.")
                pass

        bind_volumes = [
            {
                'type': 'bind',
                'source': self.ray_workdir,
                'target': self.docker_workdir
            }
        ]
        if self.container_system == 'docker':
            self.docker_container = self.client.containers.run(
                self.docker_image,
                name=self.docker_container_name,
                # volumes={self.ray_workdir: {'bind': self._rayui_workdir, 'mode': 'rw'}},  # mount Ray-UI workdir
                mounts=bind_volumes,
                detach=True,  # run in background
                auto_remove=True,  # remove container after kill or stop
                entrypoint=["tail", "-f", "/dev/null"],
            )

    def kill(self):
        """
        Send kill signal to docker container.
        """
        try:
            if self.container_system == 'docker':
                self.docker_container.kill()
        except docker.errors.NotFound:
            pass

    def __del__(self):
        self.kill()

    def extract_ray_output(self, run_workdir, exported_planes, _):
        ray_output = {}
        for exported_plane in exported_planes:
            ray_output_file = os.path.join(run_workdir, exported_plane + '-RawRaysBeam.csv').replace("\\", "/")
            raw_output = pd.read_csv(ray_output_file, sep='\t', skiprows=1, engine='c', dtype='float32',
                                usecols=[exported_plane + '_OX', exported_plane + '_OY', exported_plane + '_OZ',
                                        exported_plane + '_DX', exported_plane + '_DY', exported_plane + '_DZ',
                                        exported_plane + '_EN', exported_plane + '_PL'])
        
            ray_output[exported_plane] = RayOutput(
            x_loc=torch.tensor(raw_output[exported_plane + '_OX'].values, dtype=torch.float, device=self.device),
            y_loc=torch.tensor(raw_output[exported_plane + '_OY'].values, dtype=torch.float, device=self.device),
            z_loc=torch.tensor(raw_output[exported_plane + '_OZ'].values, dtype=torch.float, device=self.device),
            x_dir=torch.tensor(raw_output[exported_plane + '_DX'].values, dtype=torch.float, device=self.device),
            y_dir=torch.tensor(raw_output[exported_plane + '_DY'].values, dtype=torch.float, device=self.device),
            z_dir=torch.tensor(raw_output[exported_plane + '_DZ'].values, dtype=torch.float, device=self.device),
            energy=torch.tensor(raw_output[exported_plane + '_EN'].values, dtype=torch.float, device=self.device))
        return ray_output
    
    def run(self,
            raypyng_rml: RMLFile,
            exported_planes: list[str],
            run_id: str | None = None) -> dict[str, RayOutput]:

        # create random id if not given
        if run_id is None:
            run_id = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(16))

        # create sub-workdir (required for multi-threading with Ray-UI)
        run_workdir = os.path.join(self.ray_workdir, run_id)
        os.makedirs(run_workdir, exist_ok=True)
        if self.additional_mount_files is not None:
            for file in self.additional_mount_files:
                shutil.copy(file, self.ray_workdir)

        tic = time.perf_counter()

        # create RML-file for this run
        rml_workfile = os.path.join(run_workdir, 'workfile.rml')
        raypyng_rml.write(rml_workfile)

        if not os.path.isfile(rml_workfile):
             raise Exception("RML workfile not found in", rml_workfile)
        
        # get corresponding RML-file for in docker container
        docker_rml_workfile = os.path.join(self.docker_workdir, run_id, os.path.basename(rml_workfile)).replace("\\",
                                                                                                                "/")

        # argument for planes to be exported
        cmd_exported_planes = " ".join("\"" + plane + "\"" for plane in exported_planes)
        # run Ray-UI background mode in docker and retry if it failes
        for run in range(self.max_retry + 1):
            if self.container_system == 'docker':
                self.docker_container.exec_run(
                    cmd=f"{self.executable} {docker_rml_workfile} -p {cmd_exported_planes}"
                )
            else:

                podman_command = f"{self.container_executable} exec {self.docker_container_name} {self.executable} {docker_rml_workfile}"
                if self.set_export_plane_in_exec:
                    podman_command += f" -p {cmd_exported_planes}"
                if self.verbose:
                    print(podman_command)
                output = subprocess.check_output(shlex.split(podman_command), stderr=self.print_device)
                if self.verbose:
                    print(output)
            retry = False
            # fail indicator: any required CSV-file is missing
            for exported_plane in exported_planes:
                if not os.path.isfile(
                        os.path.join(run_workdir, exported_plane + '-RawRaysBeam.csv').replace("\\", "/")) and not os.path.isfile(os.path.splitext(rml_workfile)[0] + '.h5'):
                    retry = True
            if not retry:
                break
            else:
                print(f"Run {run + 1} failed. Retry ...")

        # extract Ray-UI output from CSV-files and create RayOutput instances
        ray_output = self.extract_ray_output(run_workdir, exported_planes, rml_workfile)
        # remove sub-workdir
        #shutil.rmtree(run_workdir)

        toc = time.perf_counter()
        if self.verbose:
            print(f'Run ID {run_id}: Ray-UI output from {os.path.basename(rml_workfile)}' +
                  f' successfully generated in {toc - tic:.2f}s')

        return ray_output


class RayBackendDockerRAYX(RayBackendDockerRAYUI):
    """
    Creates a Ray-X backend within a docker container.
    Currently only a single image plane can be exported.
    This backend is still experimental and needs to be revised.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def extract_ray_output(self, _, exported_planes, rml_workfile):
        ray_output_file = os.path.splitext(rml_workfile)[0] + '.h5'

        with h5py.File(ray_output_file, 'r') as h5f:
            object_names = list(h5f['rayx']['object_names'])
            object_names = [name.decode('utf-8') for name in object_names]
            events_object_id = torch.tensor(h5f['rayx']['events']['object_id'][:])

            x_dir = torch.tensor(h5f['rayx']['events']['direction_x'])
            y_dir = torch.tensor(h5f['rayx']['events']['direction_y'])
            z_dir = torch.tensor(h5f['rayx']['events']['direction_z'])
            x_loc = torch.tensor(h5f['rayx']['events']['position_x'])
            y_loc = torch.tensor(h5f['rayx']['events']['position_y'])
            z_loc = torch.tensor(h5f['rayx']['events']['position_z'])
            energy = torch.tensor(h5f['rayx']['events']['energy'])
            ray_output = {}
            for exported_plane in exported_planes:
                index = object_names.index(exported_plane) 
                current_index_mask = events_object_id == index
                ray_output[exported_plane] = RayOutput(
                    x_loc=x_loc[current_index_mask],
                    y_loc=y_loc[current_index_mask],
                    z_loc=z_loc[current_index_mask],
                    x_dir=x_dir[current_index_mask],
                    y_dir=y_dir[current_index_mask],
                    z_dir=z_dir[current_index_mask],
                    energy=energy[current_index_mask],
                )

        if self.verbose:
            print(f'Ray-X output from {os.path.basename(rml_workfile)}' +
                  f' successfully generated ')
        return ray_output

