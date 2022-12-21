from typing import List, Callable

import h5py
import torch

from torch.utils.data import Dataset, DataLoader


class RandomDataset(Dataset):
    def __init__(self,
                 h5_files: List[str],
                 exclude_idx_sub: List[str] = None,
                 exclude_ray_output: List[str] = None,
                 transform: Callable = None):

        self.h5_files = h5_files
        self.exclude_idx_sub = exclude_idx_sub if exclude_idx_sub else []
        self.exclude_ray_output = exclude_ray_output if exclude_ray_output else []
        # open all h5_files
        self.h5_files_obj = [h5py.File(f, "r", swmr=True, libver='latest') for f in self.h5_files]

        # map that yields the consecutive index based on (h5file index, index in h5file)
        self.get_idx = {}
        # map that yields a tuple (h5file index, index in h5file) given a (consecutive) index
        self.get_identifier = {}
        # lengths of data in each h5file
        self._n_samples = []
        idx_total = 0
        for idx_h5, h5_file_obj in enumerate(self.h5_files_obj):
            # get length of data in file and check if same for all keys
            self._n_samples.append(len(h5_file_obj))

            for idx_sample in h5_file_obj.keys():
                self.get_idx[(idx_h5, idx_sample)] = idx_total
                self.get_identifier[idx_total] = (idx_h5, idx_sample)
                idx_total += 1

        self._n_samples_total = sum(self._n_samples)
        self.transform = transform

    def __del__(self):
        for h5_file_obj in self.h5_files_obj:
            h5_file_obj.close()

    def __getitem__(self, idx):
        idx_h5, idx_sample = self.get_identifier[idx]

        sample_grp: h5py.Group = self.h5_files_obj[idx_h5][idx_sample]
        data = {}

        for idx_sub in sample_grp.keys():
            if idx_sub not in self.exclude_idx_sub:
                data[idx_sub] = {}
                data[idx_sub]['params'] = {}
                for key, val in sample_grp[idx_sub]['params'].items():
                    data[idx_sub]['params'][key] = val[()]
                data[idx_sub]['ray_output'] = {}
                ray_output_grp: h5py.Group = sample_grp[idx_sub]['ray_output']
                for idx_ray_output in ray_output_grp.keys():
                    if idx_ray_output not in self.exclude_ray_output:
                        data[idx_sub]['ray_output'][idx_ray_output] = {}
                        for key, val in ray_output_grp[idx_ray_output].items():
                            if h5py.check_string_dtype(val.dtype) is not None:
                                data[idx_sub]['ray_output'][idx_ray_output][key] = val.asstr()[()]
                            else:
                                data[idx_sub]['ray_output'][idx_ray_output][key] = torch.tensor(val[:])

        if self.transform is not None:
            data = self.transform(data)
        return data

    def __len__(self):
        return self._n_samples_total


# Important fix to make custom collate_fn work
# https://forums.fast.ai/t/runtimeerror-received-0-items-of-ancdata/48935
torch.multiprocessing.set_sharing_strategy('file_system')

from tqdm import tqdm
import os

h5_path = os.path.join('../../datasets/metrix_simulation')

h5_files = [os.path.join(h5_path, file) for file in os.listdir(h5_path) if file.endswith('.h5')]

dataset = RandomDataset(h5_files=h5_files)

batch_size = 100
data_loader = DataLoader(dataset,
                         shuffle=True,
                         batch_size=batch_size,
                         num_workers=100)

d = []
for idx, item in tqdm(enumerate(data_loader)):
    pass
