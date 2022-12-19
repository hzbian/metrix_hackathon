import random
import sys

import numpy as np
import torch


class RandomGenerator:

    def __init__(self, seed: int = None):
        # dummies
        self.rg_torch = None
        self.rg_numpy = None
        self.rg_random = None
        self.seed = None

        self.set_seed(seed)

    def set_seed(self, seed: int = None):
        if seed is None:
            self.seed = random.randrange(sys.maxsize)
        else:
            self.seed = seed
        self.rg_random = random.Random()
        self.rg_random.seed(self.seed)
        self.rg_numpy = np.random.default_rng(self.seed)
        self.rg_torch = torch.Generator().manual_seed(self.seed)
