import lightning as L
import torch
import os
from torch.utils.data import DataLoader, random_split

class DefaultDataModule(L.LightningDataModule):

    def __init__(
        self,
        train_dataset = None,
        val_dataset = None,
        test_dataset = None,                
        batch_size_train: int = 32,
        batch_size_val: int = 32,
        num_workers: int = 0,
        on_gpu: bool = False,
        seed_train: int = 42,  # fixed seed to be used shuffle in train dataloader
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["dataset"])

        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.num_workers = num_workers
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
            
        if on_gpu is not None:
            self.pin_memory = True
        else:
            self.pin_memory = False
        self.seed_train = seed_train

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
