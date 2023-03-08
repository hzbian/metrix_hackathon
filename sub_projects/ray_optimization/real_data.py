from PIL import Image, ImageChops
import numpy as np
import os
import torch
from tqdm import tqdm
import pandas as pd
import subprocess

from ray_nn.utils.ray_processing import HistToPointCloud
from ray_tools.base.backend import RayBackendDockerRAYUI
from ray_tools.base.engine import RayEngine
from ray_tools.base.parameter import RayParameterContainer, NumericalParameter, RandomParameter
from ray_tools.base.transform import MultiLayer
from ray_tools.base.utils import RandomGenerator
import matplotlib.pyplot as plt
import torchvision

root_dir = '../../datasets/metrix_real_data/2021_march_selected'
black = "black.bmp"
xy_dilation = pd.read_csv(os.path.join(root_dir, 'xy_dilation_mm.csv'))

rg = RandomGenerator(seed=42)

root_dir_2 = '../../'

rml_basefile = os.path.join(root_dir_2, 'rml_src', 'METRIX_U41_G1_H1_318eV_PS_MLearn.rml')
ray_workdir = os.path.join(root_dir_2, 'ray_workdir', 'optimization')

n_rays = ['1e4']
max_deviation = 0.1

exported_plane = "ImagePlane"  # "Spherical Grating"

multi_objective = False

# transforms = [
#    RayTransformConcat({
#        'raw': ToDict(),
#    }),
# ]
# transforms = MultiLayer([0], copy_directions=False)
verbose = False
engine = RayEngine(rml_basefile=rml_basefile,
                   exported_planes=[exported_plane],
                   ray_backend=RayBackendDockerRAYUI(docker_image='ray-ui-service',
                                                     ray_workdir=ray_workdir,
                                                     verbose=verbose),
                   num_workers=-1,
                   as_generator=False)


param_func = lambda: RayParameterContainer([
    (engine.template.U41_318eV.numberRays, NumericalParameter(value=1e4)),
    (engine.template.U41_318eV.translationXerror, RandomParameter(value_lims=(-0.25, 0.25), rg=rg)),
    (engine.template.U41_318eV.translationYerror, RandomParameter(value_lims=(-0.25, 0.25), rg=rg)),
    (engine.template.U41_318eV.rotationXerror, RandomParameter(value_lims=(-0.05, 0.05), rg=rg)),
    (engine.template.U41_318eV.rotationYerror, RandomParameter(value_lims=(-0.05, 0.05), rg=rg)),
    (engine.template.ASBL.totalWidth, RandomParameter(value_lims=(1.9, 2.1), rg=rg)),
    (engine.template.ASBL.totalHeight, RandomParameter(value_lims=(0.9, 1.1), rg=rg)),
    (engine.template.ASBL.translationXerror, RandomParameter(value_lims=(-0.2, 0.2), rg=rg)),
    (engine.template.ASBL.translationYerror, RandomParameter(value_lims=(-0.2, 0.2), rg=rg)),
    (engine.template.M1_Cylinder.radius, RandomParameter(value_lims=(174.06, 174.36), rg=rg)),
    (engine.template.M1_Cylinder.rotationXerror, RandomParameter(value_lims=(-0.25, 0.25), rg=rg)),
    (engine.template.M1_Cylinder.rotationYerror, RandomParameter(value_lims=(-1., 1.), rg=rg)),
    (engine.template.M1_Cylinder.rotationZerror, RandomParameter(value_lims=(-1., 1.), rg=rg)),
    (engine.template.M1_Cylinder.translationXerror, RandomParameter(value_lims=(-0.15, 0.15), rg=rg)),
    (engine.template.M1_Cylinder.translationYerror, RandomParameter(value_lims=(-1., 1.), rg=rg)),
    (engine.template.SphericalGrating.radius, RandomParameter(value_lims=(109741., 109841.), rg=rg)),
    (engine.template.SphericalGrating.rotationYerror, RandomParameter(value_lims=(-1., 1.), rg=rg)),
    (engine.template.SphericalGrating.rotationZerror, RandomParameter(value_lims=(-2.5, 2.5), rg=rg)),
    (engine.template.ExitSlit.totalHeight, RandomParameter(value_lims=(0.009, 0.011), rg=rg)),
    (engine.template.ExitSlit.translationZerror, RandomParameter(value_lims=(-29., 31.), rg=rg)),
    (engine.template.ExitSlit.rotationZerror, RandomParameter(value_lims=(-0.3, 0.3), rg=rg)),
    (engine.template.E1.longHalfAxisA, RandomParameter(value_lims=(20600., 20900.), rg=rg)),
    (engine.template.E1.shortHalfAxisB, RandomParameter(value_lims=(300.721702601, 304.721702601), rg=rg)),
    (engine.template.E1.rotationXerror, RandomParameter(value_lims=(-0.5, 0.5), rg=rg)),
    (engine.template.E1.rotationYerror, RandomParameter(value_lims=(-7.5, 7.5), rg=rg)),
    (engine.template.E1.rotationZerror, RandomParameter(value_lims=(-4, 4), rg=rg)),
    (engine.template.E1.translationYerror, RandomParameter(value_lims=(-1, 1), rg=rg)),
    (engine.template.E1.translationZerror, RandomParameter(value_lims=(-1, 1), rg=rg)),
    (engine.template.E2.longHalfAxisA, RandomParameter(value_lims=(4325., 4425.), rg=rg)),
    (engine.template.E2.shortHalfAxisB, RandomParameter(value_lims=(96.1560870104, 98.1560870104), rg=rg)),
    (engine.template.E2.rotationXerror, RandomParameter(value_lims=(-0.5, 0.5), rg=rg)),
    (engine.template.E2.rotationYerror, RandomParameter(value_lims=(-7.5, 7.5), rg=rg)),
    (engine.template.E2.rotationZerror, RandomParameter(value_lims=(-4, 4), rg=rg)),
    (engine.template.E2.translationYerror, RandomParameter(value_lims=(-1, 1), rg=rg)),
    (engine.template.E2.translationZerror, RandomParameter(value_lims=(-1, 1), rg=rg)),
])

parameters = pd.read_csv(os.path.join(root_dir, 'parameters.csv'))

black = Image.open(os.path.join(root_dir, black))

transform = HistToPointCloud()
for subdir, dirs, files in tqdm(os.walk(root_dir)):
    for file in files:
        if file.lower().endswith('.bmp') and not file.lower().endswith('black.bmp'):
            path = os.path.join(subdir, file)
            sample = Image.open(path)
            sample = ImageChops.subtract(sample, black)
            sample = torchvision.transforms.ToTensor()(sample)
            plt.imshow(sample)
            configuration_id = file[:3]
            sample_xy_dilation = xy_dilation[configuration_id]
            x_lims = torch.tensor((sample_xy_dilation[0], sample_xy_dilation[0]+768*1.6/1000)).unsqueeze(0)
            y_lims = torch.tensor((sample_xy_dilation[1], sample_xy_dilation[1]+576*1.6/1000)).unsqueeze(0)
            sample_parameters = parameters[configuration_id]
            image = transform(sample, x_lims, y_lims)[0]
            plt.scatter(x=image[:,0], y=image[:,1])
            ##print(sample_parameters)
            #transform(sample, )
            # we should calculate x and y lims by inferring from xyshifts

            # put it to a histogram

            # get the according parameters

            #save_path = os.path.splitext(path)[0]+'_out.bmp'
            #DiffImage.save(save_path)