# robotics_transformer/models/transformer_decoder.py
from __future__ import annotations

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by n_heads {n_heads}")
        
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = rearrange(q, "b t (h d) -> b h t d", h=self.n_heads)
        k = rearrange(k, "b t (h d) -> b h t d", h=self.n_heads)
        v = rearrange(v, "b t (h d) -> b h t d", h=self.n_heads)

        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.1, is_causal=True)
        y = rearrange(y, "b h t d -> b t (h d)")
        y = self.proj_drop(self.proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, dim: int, ff_mult: int, dropout: float):
        super().__init__()
        hidden_dim = ff_mult * dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, ff_mult: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, n_heads, dropout) # self-attention
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, ff_mult, dropout) # feed-forward network

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, dim: int, n_layers: int, n_heads: int, ff_mult: int, dropout: float, max_seq_len: int):
        super().__init__()
        self.max_seq_len = max_seq_len

        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, dim)) # (1, T, D)
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[
            DecoderBlock(dim, n_heads, ff_mult, dropout) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, T, _ = x.shape # (B, T, D)
        if T > self.max_seq_len:
            raise ValueError(f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}")
        
        x = x + self.pos_emb[:, :T, :]
        x = self.blocks(self.drop(x))
        return self.ln_f(x)
