from PIL import Image, ImageChops
import numpy as np
import os
from tqdm import tqdm
import subprocess

root_dir = '../../datasets/metrix_real_data/2021_march_selected'
black = "black.bmp"

black = Image.open(os.path.join(root_dir, black))

for subdir, dirs, files in tqdm(os.walk(root_dir)):
    for file in files:
        if file.lower().endswith('.bmp'):
            path = os.path.join(subdir, file)
            sample = Image.open(path)
            DiffImage = ImageChops.subtract(sample, black)
            # we should calculate x and y lims by inferring from xyshifts
            # put it to a histogram

            # get the according parameters

            #save_path = os.path.splitext(path)[0]+'_out.bmp'
            #DiffImage.save(save_path)