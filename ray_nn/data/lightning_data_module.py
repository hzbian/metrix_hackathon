import lightning as L
import torch
import os
from torch.utils.data import DataLoader, random_split
from ray_tools.simulation.torch_datasets import BalancedMemoryDataset, RayDataset, HistDataset

class DefaultDataModule(L.LightningDataModule):

    def __init__(
        self,
        train_dataset_ids,
        val_dataset_ids,
        test_dataset_ids,
        dataset_path,
        file_pattern,
        transforms = None,
        load_len = None,
        batch_size_train: int = 32,
        batch_size_val: int = 32,
        num_workers: int = 0,
        on_gpu: bool = False,
        seed_split: int = 42,  # fixed seed to be used for random train-val-test split
        seed_train: int = 42,  # fixed seed to be used shuffle in train dataloader
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["dataset"])

        self.test_dataset_ids = test_dataset_ids
        self.val_dataset_ids = val_dataset_ids
        self.train_dataset_ids = train_dataset_ids

        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.num_workers = num_workers
        
        self.sub_groups = ['parameters', 'histogram/ImagePlane', 'n_rays/ImagePlane']
        self.transforms=transforms
        self.normalize_sub_groups = ['parameters']
        self.load_len = load_len
        self.file_pattern = file_pattern.split('*')
        self.parameter_container = None
        self.xy_lims = None
        self.dataset_path = dataset_path

    
        if on_gpu is not None:
            self.pin_memory = True
        else:
            self.pin_memory = False
        self.seed_train = seed_train
        self.seed_split = seed_split
    def generate_dataset(self, ids):
        h5_files = [os.path.join(self.dataset_path, self.file_pattern[0]+str(i)+self.file_pattern[1]) for i in ids]
        dataset = HistDataset(h5_files, self.sub_groups, self.transforms, normalize_sub_groups=self.normalize_sub_groups, load_max=self.load_len)
        if self.parameter_container is None or self.xy_lims is None:
            self.parameter_container = dataset.retrieve_parameter_container(h5_files[0])
            self.xy_lims = dataset.retrieve_xy_lims(h5_files[0])
        memory_dataset = BalancedMemoryDataset(dataset=dataset, load_len=self.load_len, min_n_rays=10)
        del dataset
        return memory_dataset
        
    def get_xy_lims(self):
        return self.xy_lims

    def get_parameter_container(self):
        return self.parameter_container

    def prepare_data(self):
        if self.train_dataset_ids is not None:
            self.train_dataset = self.generate_dataset(self.train_dataset_ids)
        if self.val_dataset_ids is not None:
            self.val_dataset = self.generate_dataset(self.val_dataset_ids)
        if self.test_dataset_ids is not None:
            self.test_dataset = self.generate_dataset(self.test_dataset_ids)

    def train_dataloader(self):
        assert self.train_dataset is not None
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size_train,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            generator=torch.Generator().manual_seed(self.seed_train),
        )

    def val_dataloader(self):
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size_val,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        assert self.test_dataset is not None
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size_val,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
