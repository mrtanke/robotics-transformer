from __future__ import annotations

from einops import rearrange
from einops.layers.torch import Reduce
import torch
import torch.nn as nn


class TokenLearner(nn.Module):
    """
    Compress the token sequence from N -> M tokens:
      input tokens:  (B, N, D)
      output tokens: (B, M, D)

    Learns M attention maps over the N tokens and produces weighted sums.
    """
    def __init__(self, token_dim: int, num_tokens_out: int = 8, hidden_dim: int = 256):
        super().__init__()
        self.token_dim = token_dim
        self.num_tokens_out = num_tokens_out

        # (B, N, D) -> (B, N, M)
        self.mlp = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_tokens_out),
        )
        self.token_reduce = Reduce("b m n d -> b m d", "sum")

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() != 3:
            raise ValueError(f"TokenLearner expects (Batch_size, Sequence_length, Token_dim), got {tuple(tokens.shape)}")
        
        _, _, token_dim = tokens.shape
        if token_dim != self.token_dim:
            raise ValueError(f"TokenLearner token_dim mismatch: got Token_dim={token_dim}, expected {self.token_dim}")
        
        attn = self.mlp(tokens).softmax(dim=1)  # (B, N, M)

        # (B, M, N) @ (B, N, D) -> (B, M, D)
        return torch.matmul(attn.transpose(1, 2), tokens)
