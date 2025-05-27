import random
import time
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset

from ttrl.helper.utils import to, pin_memory
from ttrl.env.prime_samples_maker import PrimeSamples


class PrimeSamplesDataset(Dataset):
    def __init__(self, prime_samples_list: List[PrimeSamples]):
        self.samples = prime_samples_list
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]
    

