from typing import List, Callable

import h5py

from torch.utils.data import Dataset

from .torch_data_tools import h5_to_dict


class RayDataset(Dataset):
    def __init__(self,
                 h5_files: List[str],
                 sub_groups: List[str],
                 transform: Callable = None):

        self.h5_files = h5_files
        self.sub_groups = sub_groups
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
        for grp in self.sub_groups:
            grp_split = grp.split(sep='/')
            sub_dict = data
            for key in grp_split:
                if key not in sub_dict:
                    sub_dict[key] = {}
                sub_dict = sub_dict[key]

            sub_dict.update(h5_to_dict(sample_grp[grp]))

        if self.transform is not None:
            data = self.transform(data)
        return data

    def __len__(self):
        return self._n_samples_total
