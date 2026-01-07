# robotics_transformer/models/film.py
from __future__ import annotations

from einops import rearrange
import torch
import torch.nn as nn


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation:
      y = (1 + gamma) * x + beta
    
    Identity init: gamma=0, beta=0 at start.
    """
    def __init__(self, text_dim: int, channels: int):
        super().__init__()
        self.to_params = nn.Linear(text_dim, channels * 2)
        nn.init.zeros_(self.to_params.weight)
        nn.init.zeros_(self.to_params.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        cond: (B, D)
        returns: (B, C, H, W)
        """
        if cond.dim() != 2:
            raise ValueError(f"FiLM expects cond=(B, D), got {tuple(cond.shape)}")
        
        params = self.to_params(cond)
        gamma, beta = params.chunk(2, dim=-1)

        # (B, C) -> (B, C, 1, 1)
        gamma = rearrange(gamma, "b c -> b c 1 1")
        beta = rearrange(beta, "b c -> b c 1 1")

        return (1.0 + gamma) * x + beta
