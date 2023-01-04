from typing import List, Callable, Any, Dict

from tqdm import tqdm

import h5py

from torch.utils.data import Dataset

from .data_tools import h5_to_dict


class RayDataset(Dataset):
    def __init__(self,
                 h5_files: List[str],
                 sub_groups: List[str],
                 nested_groups: bool = False,
                 transform: Callable = None):

        self.h5_files = h5_files
        self.sub_groups = sub_groups
        self.nested_groups = nested_groups
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

    def __getitem__(self, idx, sub_groups: List[str] = None, nested_groups: bool = None) -> Dict[str, Any]:
        sub_groups = sub_groups if sub_groups is not None else self.sub_groups
        nested_groups = nested_groups if nested_groups is not None else self.nested_groups

        idx_h5, idx_sample = self.get_identifier[idx]
        sample_grp: h5py.Group = self.h5_files_obj[idx_h5][idx_sample]

        data = {}
        for grp in sub_groups:
            if nested_groups:
                grp_split = grp.split(sep='/')
                sub_dict = data
                for key in grp_split[:-1]:
                    if key not in sub_dict:
                        sub_dict[key] = {}
                    sub_dict = sub_dict[key]
                sub_dict[grp_split[-1]] = h5_to_dict(sample_grp[grp])
            else:
                data[grp] = h5_to_dict(sample_grp[grp])

        return self.transform(data) if self.transform else data

    def __len__(self):
        return self._n_samples_total


def extract_field(dataset: RayDataset, field: str) -> List[Any]:
    data = len(dataset) * [None]
    for idx in tqdm(range(len(dataset))):
        data[idx] = list(dataset.__getitem__(idx, sub_groups=[field], nested_groups=False).values())[0]
    return data
