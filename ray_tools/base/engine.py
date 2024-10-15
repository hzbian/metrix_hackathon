from __future__ import annotations

import abc
import torch
from collections.abc import Iterable 

from joblib import Parallel, delayed

from raypyng import RMLFile
from raypyng.xmltools import XmlElement

from . import RayTransformType
from .backend import RayBackend, RayOutput
from .parameter import NumericalParameter, RandomParameter, RayParameterContainer, OutputParameter
from .transform import RayTransform

def get_exported_plane_translation(
    exported_plane: str, param_container: RayParameterContainer
):
    x_translation: float = 0.0
    y_translation: float = 0.0
    z_translation: float = 0.0
    for key, param in param_container.items():
        if isinstance(param, OutputParameter) and isinstance(
            param, NumericalParameter
        ):
            if key.split(".")[0] == exported_plane:
                param_entry = key.split(".")[-1]
                if param_entry == "translationXerror":
                    x_translation = param.value
                if param_entry == "translationYerror":
                    y_translation = param.value
                if param_entry == "translationZerror":
                    z_translation = param.value
    return x_translation, y_translation, z_translation


class Engine(abc.ABC):
    @abc.abstractmethod
    def run(self,
            param_containers: list[RayParameterContainer],
            transforms: RayTransformType | Iterable[RayTransformType] | None = None,
            ) -> list[dict]:
        pass


class RayEngine(Engine):
    """
    Creates an engine to run raytracing simulations.
    :param rml_basefile: RML-file to be used as beamline template.
    :param exported_planes: Image planes and component outputs to be exported.
    :param ray_backend: RayBackend object that actually runs the simulation.
    :param num_workers: Number of parallel workers for runs (multi-threading, NOT multi-processing).
        Use 1 for no single-threading.
    :param as_generator: If True, :func:`RayEngine.run` returns a generator so that runs are performs when iterating
        over it.
    :param verbose:
    """

    def __init__(self,
                 rml_basefile: str,
                 exported_planes: list[str],
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

        # internal RMLFile object
        self._raypyng_rml = RMLFile(self.rml_basefile)
        self.template = self._raypyng_rml.beamline

    def run(self,
            param_containers: list[RayParameterContainer],
            transforms: RayTransformType | Iterable[RayTransformType] | None = None,
            ) -> list[dict]:
        """
        Runs simulation for given (Iterable of) parameter containers.
        :param param_containers: (Iterable of) :class:`ray_tools.base.parameter.RayParameterContainer` to be processed.
        :param transforms: :class:`ray_tools.base.transform.RayTransform` to be used.
            If a singleton, the same transform is applied everywhere.
            If Iterable of RayTransform (same length a param_containers), individual transforms are applied to each
            parameter container. Transform can be also dicts of RayTransform specifying with transform to apply to
            which exported planes (keys must be same as ``RayEngine.exported_planes``).
        :return: (Iterable of) dict with ray outputs (field ``ray_output``,
            see also :class:`ray_tools.base.backend.RayBackend`) and used parameters for simulation
            (field ``param_container_dict``, dict with same keys as in ``param_containers``).
        """

        # convert transforms into list if it was a singleton
        if transforms is None or isinstance(transforms, (RayTransform, dict)):
            transforms_list = len(param_containers) * [transforms]
        else:
            transforms_list = transforms

        # Iterable of arguments used for RayEngine._run_func
        _iter = ((run_params, transform) for (run_params, transform) in
                 zip(param_containers, transforms_list))
        # multi-threading (if self.num_workers > 1)
        worker = Parallel(n_jobs=self.num_workers, verbose=self.verbose, backend='threading')
        jobs = (delayed(self._run_func)(*item) for item in _iter)
        result = worker(jobs)
        if not isinstance(result, list):
            raise Exception("The result must be a list if we input a list.")
        # extract only element if param_containers was a singleton
        return result

    def _run_func(self,
                  param_container: RayParameterContainer,
                  transform: RayTransformType | None = None,
                  ) -> dict:
        """
        This method performs the actual simulation run.
        """
        result = {'param_container_dict': dict(), 'ray_output': None}

        # create a copy of RML template to avoid problems with multi-threading
        raypyng_rml_work = RMLFile(self.rml_basefile)
        template_work = raypyng_rml_work.beamline
        # write values in param_container to RML template and param_container_dict
        for key, param in param_container.items():
            value = param.get_value()
            if not isinstance(param, OutputParameter):
                element = self._key_to_element(key, template=template_work)
                element.cdata = str(value)
            result['param_container_dict'][key] = value

        # call the backend to perform the run
        ray_output_all_planes = self.ray_backend.run(raypyng_rml=raypyng_rml_work,
                                                    exported_planes=self.exported_planes)
        for key, ray_output in ray_output_all_planes.items():
            # compute x and y direction for normalized z direction (zz_dir would be 1)
            xz_dir = ray_output.x_dir / ray_output.z_dir
            yz_dir = ray_output.y_dir / ray_output.z_dir
            trans_x, trans_y, trans_z = get_exported_plane_translation(key, param_container=param_container)

            x_cur = ray_output.x_loc + xz_dir * trans_z + trans_x
            y_cur = ray_output.y_loc + yz_dir * trans_z + trans_y
            z_cur = ray_output.z_loc + trans_z

            ray_output_all_planes[key].x_loc = x_cur
            ray_output_all_planes[key].y_loc = y_cur
            ray_output_all_planes[key].z_loc = z_cur

        result['ray_output'] = ray_output_all_planes
        # apply transform (to each exported plane)
        if transform is not None:
            for plane in self.exported_planes:
                t = transform if isinstance(transform, RayTransform) else transform[plane]
                result['ray_output'][plane] = t(result['ray_output'][plane])
        return result

    def _key_to_element(self, key: str, template: XmlElement | None = None) -> XmlElement:
        """
        Helper function that retrieves an XML-subelement given a key (same format as in RayParameterContainer).
        """
        if template is None:
            template = self.template
        component, param = key.split('.')
        if template is None:
            raise Exception("Template cannot be None.")
        element = template.__getattr__(component)
        if not isinstance(element, XmlElement):
            raise Exception("Element must be XmlElement.")
        return element.__getattr__(param)


class GaussEngine(Engine):
    def __init__(self, device: torch.device | None = None) -> None:
        super().__init__()
        self.device = device

    def run(self,
            param_containers: list[RayParameterContainer],
            transforms: RayTransformType | list[RayTransformType] | None = None,
            ) -> list[dict]:

        if isinstance(param_containers, RayParameterContainer):
            param_containers = [param_containers]

        # convert transforms into list if it was a singleton
        if transforms is None or isinstance(transforms, (RayTransform, dict)):
            transforms_list = len(param_containers) * [transforms]
        else:
            transforms_list = transforms

        outputs = []
        for param_container_num, param_container in enumerate(param_containers):
            x_mean = param_container['x_mean'].get_value()
            y_mean = param_container['y_mean'].get_value()
            x_var = param_container['x_var'].get_value()
            y_var = param_container['y_var'].get_value()
            n_rays = int(param_container['number_rays'].get_value())
            if 'correlation_factor' in param_container:
                correlation_factor = torch.tensor(param_container['correlation_factor'].get_value(), device=self.device)
            else:
                correlation_factor = torch.tensor(0., device=self.device)


            m = torch.distributions.multivariate_normal.MultivariateNormal(torch.tensor([0., 0.], device=self.device).float(),
                                                                           torch.diag(torch.tensor([x_var, y_var], device=self.device)).float())
            samples = m.rsample(torch.Size([n_rays]))
            samples @= torch.tensor([[torch.cos(correlation_factor), -torch.sin(correlation_factor)],[torch.sin(correlation_factor), torch.cos(correlation_factor)]], device=self.device)
            samples += (torch.tensor([x_mean, y_mean], device=self.device).float())
            samples_directions = torch.rand([n_rays, 3], device=self.device) * param_container['direction_spread'].get_value()
            ray_out = RayOutput(samples[:, 0], samples[:, 1], torch.zeros_like(samples[:, 0]),
                                param_container['x_dir'].get_value() + samples_directions[:, 0],
                                param_container['y_dir'].get_value() + samples_directions[:, 1],
                                param_container['z_dir'].get_value() + samples_directions[:, 2],
                                torch.ones_like(samples[:, 0]))
            if transforms_list[param_container_num] is not None:
                ray_out = transforms_list[param_container_num](ray_out)

            outputs.append({'ray_output': {'ImagePlane': ray_out}, 'param_container_dict': param_container})
        return outputs

class SurrogateEngine(Engine):
    def __init__(self,  checkpoint_path:str, model: torch.nn.Module, device: torch.device, is_vae:bool = True):
        super(SurrogateEngine, self).__init__()
        self.is_vae = is_vae
        self.device = device
        if is_vae:
            self.latent_size = 200
        else:
            self.latent_size = None
        #     model = CVAE(20 * 20, self.latent_size, 5).to(device)
        # else:
        #     model = Transformer(hist_dim=(20, 20), param_dim=36, transformer_dim=2048,
        #                         n_hist_layers_inp=3, use_inp_template=True, n_hist_layers_out=1,
        #                         transformer_heads=2, transformer_layers=3)
        # checkpoint = torch.load(checkpoint_path)
        # model_weights = checkpoint["state_dict"]
        # for key in list(model_weights):
        #     model_weights[key.replace("backbone.", "")] = model_weights.pop(key)
        # model.load_state_dict(model_weights)
        # model.eval()

        self.model = model


    def run(self,
            param_containers: list[RayParameterContainer],
            transforms: RayTransformType | Iterable[RayTransformType] | None = None,
            ) -> list[dict]:

        if isinstance(param_containers, RayParameterContainer):
            param_containers = [param_containers]

        # convert transforms into list if it was a singleton
        if isinstance(param_containers, RayParameterContainer):
            param_containers = [param_containers]

        # convert transforms into list if it was a singleton
        if transforms is None or isinstance(transforms, (RayTransform, dict)):
            transforms_list = len(param_containers) * [transforms]
        else:
            transforms_list = transforms


        outputs = []
        for param_container_num, param_container in enumerate(param_containers):
            params = param_container
            keys = []
            for key in params.keys():
                current_param = params[key]
                if isinstance(current_param, RandomParameter):
                    value_lims = current_param.value_lims
                    norm_value = (current_param.get_value() - value_lims[0]) / (value_lims[1] - value_lims[0])
                    current_param.value = norm_value
                else:
                    keys.append(key)
            if keys != list(params.keys()):
                for key in keys:
                    params.pop(key)
            params_list = []
            for key in params.keys():
                params_list.append(torch.Tensor([params[key].get_value()]))
            params_tensor = torch.cat(params_list, dim=0)
            params_tensor = torch.unsqueeze(params_tensor,0)
            if self.is_vae and self.latent_size is not None:
                means = torch.zeros(self.latent_size).to(self.device)
                std = torch.ones(self.latent_size).to(self.device)
                draw = torch.normal(means, std).to(self.device)
                draw = torch.reshape(draw, (1, -1))
                model_out = self.model.decode(draw, params_tensor.to(self.device))
                outputs.append(model_out)
            else:
                model_out = self.model(params, 1, 1, 1)
                outputs.append(model_out)
        return outputs
