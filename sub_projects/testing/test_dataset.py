import os
import sys

from raypyng import RMLFile
from raypyng.xmltools import XmlElement

sys.path.insert(0, '../../')

import matplotlib.pyplot as plt

from ray_tools.simulation.torch_datasets import RayDataset

import numpy as np
import torch

# Important fix to make custom collate_fn work
# https://forums.fast.ai/t/runtimeerror-received-0-items-of-ancdata/48935
torch.multiprocessing.set_sharing_strategy('file_system')

h5_path = os.path.join('../datasets/metrix_simulation/ray_surrogate')

h5_files = [os.path.join(h5_path, file) for file in os.listdir(h5_path) if file.endswith('.h5')]

exported_planes = [
    "U41_318eV",
    "ASBL",
    "M1-Cylinder",
    "Spherical Grating",
    "Exit Slit",
    "E1",
    "E2",
    "ImagePlane"
]

dataset = RayDataset(h5_files=h5_files,
                     sub_groups=['1e5/params'] +
                                [f'1e5/ray_output/{exported_plane}/hist_small' for exported_plane in exported_planes])

show_examples = [13]  # range(n_examples)

item = {idx: dataset.__getitem__(idx) for idx in show_examples}

for idx in show_examples:
    for exported_plane in exported_planes:
        plt.figure(figsize=(10, 10))
        plt.title(str(idx) + ' ' + exported_plane)
        plt.imshow(np.flipud(item[idx]['1e5']['ray_output'][exported_plane]['hist_small']['histogram'].T),
                   cmap='Greys')
        plt.xlabel(str(item[idx]['1e5']['ray_output'][exported_plane]['hist_small']['n_rays']) + ' ' +
                   str(item[idx]['1e5']['ray_output'][exported_plane]['hist_small']['x_lims']) + ' ' +
                   str(item[idx]['1e5']['ray_output'][exported_plane]['hist_small']['y_lims']))
        plt.show()

params = item[show_examples[-1]]['1e5']['params']


# -------------

def key_to_element(key: str, template: XmlElement) -> XmlElement:
    component, param = key.split('.')
    return template.__getattr__(component).__getattr__(param)


rml_basefile = '../rml_src/METRIX_U41_G1_H1_318eV_PS_MLearn.rml'

raypyng_rml_work = RMLFile(rml_basefile)
for key, val in params.items():
    element = key_to_element(key, raypyng_rml_work.beamline)
    element.cdata = str(val)

raypyng_rml_work.write('test_dataset.rml')
