import glob
from tqdm import tqdm

from ray_tools.simulation.torch_datasets import RayDataset

if __name__ == '__main__':
    h5_files = list(glob.iglob('datasets/metrix_simulation/ray_emergency_surrogate/50+50_data_raw_*.h5'))
    dataset: RayDataset = RayDataset(h5_files=h5_files,
                     sub_groups=['1e5/params',
                                 '1e5/histogram', '1e5/n_rays'])
    num_rays = 0
    assert dataset is not None

    for i in tqdm(dataset):
        if num_rays < i['1e5/n_rays']:
            num_rays = i['1e5/n_rays']
    
    print(num_rays)