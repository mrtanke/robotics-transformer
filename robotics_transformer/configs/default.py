# robotics_transformer/configs/default.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class robotic_transformerConfig:
    # data
    image_size: int = 300
    history_len: int = 6  # number of observation frames
    # action dims = 11 (7 arm + 3 base + 1 mode)
    action_dims: int = 11
    action_bins: int = 256

    # vision tokenizer
    vision_token_dim: int = 512     # vision tokenizer output dim
    vision_grid: Tuple[int, int] = (9, 9)  # 81 tokens
    tokens_per_image: int = 8       # TokenLearner output

    # transformer
    n_layers: int = 8
    n_heads: int = 8
    dropout: float = 0.1
    ff_mult: int = 4 # feedforward hidden dim multiplier

    # Transformer input sequence length = 6*8 + 11 = 59 (obs tokens + action tokens)
    max_seq_len: int = 59

    # training
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 0.05
    grad_clip: float = 1.0 # max gradient L2 norm
    epochs: int = 5
    num_workers: int = 4
    log_every: int = 50
