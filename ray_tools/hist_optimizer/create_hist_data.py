import sys
import torch
import h5py
import numpy as np
import os
from tqdm import trange
from ray_tools.base.engine import RayEngine
from ray_tools.base.parameter import MutableParameter
from ray_tools.base.backend import RayBackendDockerRAYUI
from ray_tools.base.transform import Histogram, RayTransformConcat, XYHistogram, MultiLayer, RayTransformCompose
from datasets.metrix_simulation.config_ray_emergency_surrogate import PARAM_CONTAINER_FUNC as params
from ray_tools.hist_optimizer.hist_optimizer import tensor_to_param_container, mse_engines_comparison, Model, find_good_offset_problem, optimize_smart_walker, optimize_brute, evaluate_evaluation_method, plot_param_tensors, tensor_list_to_param_container_list

def engine_output_to_np_array(out):
    arr_list = []
    n_rays_list = []
    histogram_list = []
    titles = [label for label in out[0]['param_container_dict'].keys()]
    for entry in out:
        arr_list.append(np.array([value for value in entry['param_container_dict'].values()]))
        first_key = next(iter(entry['ray_output']['ImagePlane']))
        n_rays_list.append(entry['ray_output']['ImagePlane'][first_key]['n_rays'])
        histogram_list.append(entry['ray_output']['ImagePlane'][first_key]['histogram'])
    arr = np.stack(arr_list)
    n_rays = np.stack(n_rays_list)
    histogram = np.stack(histogram_list)
    return arr, n_rays, histogram

def create_histogram_file(engine, seed, path="outputs/", total_size=10, batch_size=2, x_lims=(-10., 10.), y_lims=(-3., 3.), z_lims=(-3., 3.)):
    with h5py.File(os.path.join(path, 'histogram_'+str(seed)+'.h5'), 'w') as f:
        torch.manual_seed(seed)
        dset_arr = f.create_dataset("parameters", (total_size,36), dtype='float64', track_order=True)
        dset_n_rays = f.create_dataset("n_rays", (total_size,), dtype='float64')
        dset_hist = f.create_dataset("histogram", (total_size,2,50), dtype='float64')
        dset_hist.attrs['lims'] = x_lims, y_lims, z_lims
        lims_list = []
        for label, value in params().items():
            if isinstance(value, MutableParameter):
                lims_list.append((label, value.value_lims))
            else:
                lims_list.append((label, value.get_value()))
        lims_list.append(('ImagePlane.translationZerror', z_lims))
        dset_arr.attrs.update(lims_list)
        random_params = torch.rand(total_size,35)
    
        for i in trange(0, total_size, batch_size):
            end = min(i + batch_size, total_size)  # Ensure we don't go out of bounds
            
            z_translations = random_params[i:end, 34] * (z_lims[1]-z_lims[0]) + z_lims[0]
            out = engine.run(tensor_list_to_param_container_list(random_params[i:end, :34]), [RayTransformCompose(XYHistogram(50, x_lims, y_lims), MultiLayer([i])) for i in z_translations])
            arr, n_rays, histogram = engine_output_to_np_array(out)
            arr = np.concatenate((arr, z_translations.unsqueeze(0).numpy()), axis=1)
            dset_arr[i:end] = arr
            dset_n_rays[i:end] = n_rays
            dset_hist[i:end] = histogram


engine = RayEngine(rml_basefile='rml_src/METRIX_U41_G1_H1_318eV_PS_MLearn_1.15.rml',
                                exported_planes=["ImagePlane"],
                                ray_backend=RayBackendDockerRAYUI(docker_image='ray-ui-service',
                                                                  docker_container_name='ray-ui-service-test',
                                                                  dockerfile_path='ray_docker/rayui',
                                                                  ray_workdir='/dev/shm/ray-workdir',
                                                                  verbose=False),
                                num_workers=-1,
                                as_generator=False)
create_histogram_file(engine, int(sys.argv[1]), total_size=1000000, batch_size=10000)
