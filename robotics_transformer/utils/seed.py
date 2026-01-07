# robotic_transformer/utils/seed.py
from __future__ import annotations

import os
import random
import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed) # Set seed for Python's built-in random module
    np.random.seed(seed) # Set seed for NumPy
    torch.manual_seed(seed) # Set seed for PyTorch CPU
    torch.cuda.manual_seed_all(seed) # Set seed for all CUDA devices

    os.environ["PYTHONHASHSEED"] = str(seed) # Set seed for Python hash functions
    if deterministic:
        torch.backends.cudnn.deterministic = True # Ensure deterministic behavior for CuDNN
        torch.backends.cudnn.benchmark = False # Disable CuDNN benchmark for reproducibility
