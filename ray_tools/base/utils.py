import random
import sys

import numpy as np
import torch


class RandomGenerator:
    """
    Container for random generators (rg_random: random / rg_numpy: numpy / rg_torch: pytorch)
    :param seed: Random seed to be used for all generators.
    """

    def __init__(self, seed: int = None):
        # dummies
        self.rg_torch = None
        self.rg_numpy = None
        self.rg_random = None
        self.seed = None

        self.set_seed(seed)

    def set_seed(self, seed: int = None):
        """
        Creates random generators with given seed. Use random seed if None.
        """
        if seed is None:
            self.seed = random.randrange(sys.maxsize)
        else:
            self.seed = seed
        self.rg_random = random.Random()
        self.rg_random.seed(self.seed)
        self.rg_numpy = np.random.default_rng(self.seed)
        self.rg_torch = torch.Generator().manual_seed(self.seed)
