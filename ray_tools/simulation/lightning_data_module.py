import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from ray_tools.simulation.torch_datasets import RayDataset


class DefaultDataModule(pl.LightningDataModule):

    def __init__(
            self,
            dataset,
            batch_size_train: int = 32,
            batch_size_val: int = 32,
            num_workers: int = 0,
            on_gpu: bool = True,
            split=None,
            seed=None,
    ):
        super().__init__()

        self.test_dataset = None
        self.val_dataset = None
        self.train_dataset = None
        if split is None:
            split = [0.8, 0.1, 0.1]
        self.save_hyperparameters(ignore='dataset')
        self.dataset = dataset
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.num_workers = num_workers
        self.on_gpu = on_gpu
        self.split = split
        self.generator = None if seed is None else torch.Generator().manual_seed(seed)

    def setup(self, stage=None):
        train_size, val_size, _ = self.split
        train_size = int(train_size * len(self.dataset))
        val_size = int(val_size * len(self.dataset))
        test_size = len(self.dataset) - (train_size + val_size)
        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(self.dataset,
                                                                                                [train_size, val_size,
                                                                                                 test_size],
                                                                                                generator=self.generator)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.batch_size_train,
                          num_workers=self.num_workers,
                          pin_memory=self.on_gpu)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, batch_size=self.batch_size_val,
                          num_workers=self.num_workers,
                          pin_memory=self.on_gpu)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, shuffle=False, batch_size=self.batch_size_val,
                          num_workers=self.num_workers,
                          pin_memory=self.on_gpu)


class RayDataModule(DefaultDataModule):
    def __init__(self, h5_files, sub_groups, transform=None):
        dataset = RayDataset(h5_files=h5_files,
                             sub_groups=sub_groups,
                             transform=transform)
        super().__init__(dataset)
