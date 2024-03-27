import h5py
import glob
from tqdm import tqdm

if __name__ == '__main__':
    file_list = list(glob.iglob('ray_emergency_surrogate/data_raw_*.h5'))
    x_min = float('inf')
    y_min = float('inf')
    x_max = -float('inf')
    y_max = -float('inf')
    for i, file in tqdm(enumerate(file_list)):
        with h5py.File(file, 'r') as f:
            for key in f.keys():
                for subkey in f[key].keys():                    
                    input_hist = f[key][subkey]['ray_output']['ImagePlane']['hist']['histogram'][:]
                    x_lims = f[key][subkey]['ray_output']['ImagePlane']['hist']['x_lims'][:]
                    y_lims = f[key][subkey]['ray_output']['ImagePlane']['hist']['y_lims'][:]
                    num_rays = f[key][subkey]['ray_output']['ImagePlane']['hist']['n_rays'][()]
                    if num_rays > 0:
                        if x_lims[0] < x_min:
                            x_min = x_lims[0]
                        if x_lims[1] > x_max:
                            x_max = x_lims[1]
                        if y_lims[0] < y_min:
                            y_min = y_lims[0]
                        if y_lims[1] > y_max:
                            y_max = y_lims[1]
                        

    print("x", str(x_min), str(x_max))
    print("y", str(y_min), str(y_max))
