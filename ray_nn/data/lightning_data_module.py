from typing import List

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split


class DefaultDataModule(pl.LightningDataModule):

    def __init__(
            self,
            dataset,
            batch_size_train: int = 32,
            batch_size_val: int = 32,
            split: List[float] = None,
            num_workers: int = 0,
            on_gpu: str = None,
            seed_split: int = None,
            seed_train: int = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['dataset'])

        self.dataset = dataset
        self.test_dataset = None
        self.val_dataset = None
        self.train_dataset = None

        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.split = [0.8, 0.1, 0.1] if split is None else split
        self.num_workers = num_workers
        if on_gpu is not None:
            self.pin_memory = True
            self.pin_memory_device = on_gpu
        else:
            self.pin_memory = False
            self.pin_memory_device = ''
        self.split = split
        self.seed_train = seed_train
        self.seed_split = seed_split

    def setup(self, stage=None):
        train_size, val_size, _ = self.split
        train_size = int(train_size * len(self.dataset))
        val_size = int(val_size * len(self.dataset))
        test_size = len(self.dataset) - (train_size + val_size)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.seed_split)
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          batch_size=self.batch_size_train,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          pin_memory_device=self.pin_memory_device,
                          generator=torch.Generator().manual_seed(self.seed_train))

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          batch_size=self.batch_size_val,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          pin_memory_device=self.pin_memory_device)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          shuffle=False,
                          batch_size=self.batch_size_val,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          pin_memory_device=self.pin_memory_device)
