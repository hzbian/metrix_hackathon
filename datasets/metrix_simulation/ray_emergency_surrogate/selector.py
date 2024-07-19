import h5py
import glob
import tqdm
import os

h5_files = list(glob.iglob('data_raw_*.h5')) # datasets/metrix_simulation/ray_emergency_surrogate/
output_path = 'selected'

os.makedirs(output_path, exist_ok=True)

for path in tqdm.tqdm(h5_files):
    if not os.path.isfile(os.path.join(output_path, path)):
        with h5py.File(path, 'r') as fs:
            with h5py.File(os.path.join(output_path,path), 'w') as fd:
                for a in fs.attrs:
                    fd.attrs[a] = fs.attrs[a]
                for key, ds in fs.items():
                    if not fs[key+'/1e5/ray_output/ImagePlane/n_rays'][()] == 0: fs.copy(ds, fd)
