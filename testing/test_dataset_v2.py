import os
import sys

from raypyng import RMLFile
from raypyng.xmltools import XmlElement

sys.path.insert(0, '../')

import matplotlib.pyplot as plt

from ray_tools.simulation.torch_datasets import RayDataset

import numpy as np
import torch

# Important fix to make custom collate_fn work
# https://forums.fast.ai/t/runtimeerror-received-0-items-of-ancdata/48935
torch.multiprocessing.set_sharing_strategy('file_system')

h5_path = os.path.join('../datasets/metrix_simulation/ray_enhance')

h5_files = [os.path.join(h5_path, file) for file in os.listdir(h5_path) if file.endswith('.h5')]

dataset = RayDataset(h5_files=h5_files,
                     sub_groups=['1e4/params'] +
                                [f'1e4/ray_output/ImagePlane/hist'] +
                                ['1e6/params'] +
                                [f'1e6/ray_output/ImagePlane']
                     )

show_examples = [1]  # range(n_examples)

item = {idx: dataset.__getitem__(idx) for idx in show_examples}

for idx in show_examples:
    plt.figure(figsize=(10, 10))
    plt.title(str(idx))
    pooler = torch.nn.AvgPool2d(kernel_size=8, divisor_override=1)
    img = torch.tensor(np.flipud(item[idx]['1e6']['ray_output']['ImagePlane']['histogram'].T).copy())
    plt.imshow(pooler(img.unsqueeze(0)).squeeze(),
               cmap='Greys')
    plt.xlabel(str(item[idx]['1e6']['ray_output']['ImagePlane']['n_rays']) + ' ' +
               str(item[idx]['1e6']['ray_output']['ImagePlane']['x_lims']) + ' ' +
               str(item[idx]['1e6']['ray_output']['ImagePlane']['y_lims']))
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.title(str(idx))
    img = torch.tensor(np.flipud(item[idx]['1e4']['ray_output']['ImagePlane']['hist']['histogram'].T).copy())
    plt.imshow(pooler(img.unsqueeze(0)).squeeze(),
               cmap='Greys')
    plt.xlabel(str(item[idx]['1e4']['ray_output']['ImagePlane']['hist']['n_rays']) + ' ' +
               str(item[idx]['1e4']['ray_output']['ImagePlane']['hist']['x_lims']) + ' ' +
               str(item[idx]['1e4']['ray_output']['ImagePlane']['hist']['y_lims']))
    plt.show()

params = item[show_examples[-1]]['1e6']['params']


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
