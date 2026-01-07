# robotic_transformer/utils/shapes.py
from __future__ import annotations

from typing import Sequence
import torch


def assert_shape(x: torch.Tensor, expected: Sequence[int], name: str = "tensor") -> None:
    # Check the number of dimensions
    if x.dim() != len(expected):
        raise ValueError(f"{name} rank mismatch: got {tuple(x.shape)}, expected rank {len(expected)}")
    
    # Check each dimension
    for i, (got, exp) in enumerate(zip(x.shape, expected)):
        if exp != -1 and got != exp:
            raise ValueError(f"{name} shape mismatch at dim {i}: got {got}, expected {exp}. full={tuple(x.shape)}")
