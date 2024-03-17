import h5py
import glob
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

from ray_nn.utils.ray_processing import HistToPointCloud
from sub_projects.ray_optimization.real_data import SampleRandomWeightedHist

def bin_xy(input_hist, x_lims, y_lims, num_rays, output_bins):
    if num_rays > 0:
        grid, intensity =  HistToPointCloud().forward(torch.tensor(input_hist).unsqueeze(0), torch.tensor(x_lims).unsqueeze(0), torch.tensor(y_lims).unsqueeze(0))
        scatter = SampleRandomWeightedHist().forward(hist=grid, pc_weights=intensity, num_rays=num_rays)
        x_hist, _ = torch.histogram(scatter[0, :, 0], bins=output_bins, range=(-10., 10.))
        y_hist, _ = torch.histogram(scatter[0, :, 1], bins=output_bins, range=(-3., 3.))
    else:
        x_hist = torch.empty([0, output_bins])
        y_hist = x_hist 
    #plt.scatter(scatter[0, :, 0], scatter[0, :, 1])
    #plt.show()
    return x_hist.numpy(), y_hist.numpy()


if __name__ == '__main__':
    bins = 50
    file_list = list(glob.iglob('datasets/metrix_simulation/ray_emergency_surrogate/data_raw_*.h5'))
    for i, file in enumerate(tqdm(file_list)):
        with h5py.File(file, 'r') as f:
            output = h5py.File('datasets/metrix_simulation/ray_emergency_surrogate/'+str(bins)+'+'+str(bins)+'_data_raw_'+str(i)+'.h5','w')
            for key in f.keys():
                for subkey in f[key].keys():
                    d = f[key][subkey]['params']
                    output.create_group(key+'/'+subkey)
                    f.copy(d, output[key][subkey])
                    
                    input_hist = f[key][subkey]['ray_output']['ImagePlane']['hist']['histogram'][:]
                    x_lims = f[key][subkey]['ray_output']['ImagePlane']['hist']['x_lims'][:]
                    y_lims = f[key][subkey]['ray_output']['ImagePlane']['hist']['y_lims'][:]
                    num_rays = f[key][subkey]['ray_output']['ImagePlane']['hist']['n_rays'][()]
                    output.create_dataset(key+'/'+subkey+'/histogram', data=bin_xy(input_hist, x_lims, y_lims, num_rays, bins))
                    output.create_dataset(key+'/'+subkey+'/n_rays', data=num_rays)
                    
            output.close()
