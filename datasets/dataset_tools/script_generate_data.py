import os
import sys

sys.path.insert(0, '../')

from ray_tools.base.engine import RayEngine
from ray_tools.simulation.data_tools import RandomRayDatasetGenerator
from ray_tools.base.parameter import build_parameter_grid
from ray_tools.base.backend import RayBackendDockerRAYUI

import datasets.metrix_simulation.config_ray_enhance_final as CFG

os.makedirs(CFG.H5_DATADIR, exist_ok=True)

param_container_sampler = RandomRayDatasetGenerator.build_param_container_sampler(
    param_container_func=lambda: build_parameter_grid(CFG.PARAM_CONTAINER_FUNC()),
    idx_sub=CFG.N_RAYS,
    transform=[{exported_plane: transform for exported_plane in CFG.EXPORTED_PLANES} for transform in CFG.TRANSFORMS]
)

generator = RandomRayDatasetGenerator(
    ray_engine=RayEngine(rml_basefile=CFG.RML_BASEFILE,
                         exported_planes=CFG.EXPORTED_PLANES,
                         ray_backend=RayBackendDockerRAYUI(docker_image='ray-ui-service', dockerfile_path='ray_docker/rayui',
                                                           ray_workdir=CFG.RAY_WORKDIR,
                                                           verbose=True),
                         num_workers=-1,
                         as_generator=False),
    param_container_sampler=param_container_sampler,
    h5_datadir=CFG.H5_DATADIR,
    h5_basename='data_raw',
    h5_max_size=CFG.H5_MAX_SIZE)

h5_failed = []
for h5_idx in CFG.H5_IDX_RANGE:
    try:
        generator.generate(h5_idx=h5_idx, batch_size=CFG.BATCH_SIZE)
    except:
        h5_failed.append(h5_idx)

print(f'Failed files: {h5_failed}')
