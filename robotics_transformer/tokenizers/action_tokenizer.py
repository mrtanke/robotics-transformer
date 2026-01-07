# robotics_transformer/tokenizers/action_tokenizer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from einops import rearrange
import numpy as np
import torch

from robotics_transformer.tokenizers.action_bounds import ActionBounds, default_action_bounds


@dataclass(frozen=True)
class ActionTokenizerConfig:
    num_bins: int = 256
    bos_id: int = 256  # BOS + previous tokens for the first action (vocab size = bins + 1) [0, 255] -> [0, 256]


class ActionTokenizer:
    """
    Discretize continuous action vectors into per-dimension integer tokens [0..num_bins-1].
    - For the last dim (mode), we treat it as discrete integer in {0,1,2}.
    - For other dims: uniform binning between [low, high].
    """
    def __init__(self, bound: Optional[ActionBounds] = None, cfg: Optional[ActionTokenizerConfig] = None):
        self.bound = bound or default_action_bounds()
        self.cfg = cfg or ActionTokenizerConfig()

        if self.bound.low.shape != self.bound.high.shape:
            raise ValueError("ActionBounds low/high shape mismatch.")
        self.num_action_dims = int(self.bound.low.shape[0]) # total action dims

    @property
    def num_bins(self) -> int:
        return self.cfg.num_bins

    @property
    def vocab_size(self) -> int:
        return self.cfg.num_bins + 1  # + BOS

    @property
    def bos_id(self) -> int:
        return self.cfg.bos_id

    def encode(self, action: torch.Tensor) -> torch.Tensor:
        """
        Continuous actions -> discrete tokens [0, num_bins-1] per dim
        Steps:
        1. clamp to [low, high]
        2. for continuous dims: uniform binning into [0, num_bins-1]
        3. for mode dim: round to nearest integer in {0,1,2}
        4. return integer tokens

        Inputs: action: (B, num_action_dims) float tensor
        Outputs: tokens: (B, num_action_dims) int64 tensor
        """
        if action.dim() != 2 or action.shape[1] != self.num_action_dims:
            raise ValueError(f"Expected action (B,{self.num_action_dims}), got {tuple(action.shape)}")
        
        low = torch.tensor(self.bound.low, device=action.device, dtype=action.dtype)
        high = torch.tensor(self.bound.high, device=action.device, dtype=action.dtype)
        
        # force into bounds
        x = torch.clamp(action, low, high) 

        actions_before_mode = x[:, :-1]
        denominator = torch.clamp(high[:-1] - low[:-1], min=1e-6) # avoid div-by-zero
        u = (actions_before_mode - low[:-1]) / denominator # in [0,1]
        tokens_before_mode = torch.floor(u * self.num_bins).to(torch.int64)
        tokens_before_mode = torch.clamp(tokens_before_mode, 0, self.num_bins - 1)

        mode = torch.round(x[:, -1]).to(torch.int64)
        mode = torch.clamp(mode, 0, 2)

        tokens = torch.zeros((action.shape[0], self.num_action_dims), device=action.device, dtype=torch.int64)
        tokens[:, :-1] = tokens_before_mode
        tokens[:, -1] = mode
        return tokens

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Discrete tokens -> continuous actions
        Steps:
        1. for continuous dims: map token [0, num_bins-1] back to continuous value in [low, high]
        2. for mode dim: keep as integer in {0,1,2}
        3. return continuous actions 

        Inputs: tokens: (B, num_action_dims) int64 tensor
        Outputs: actions: (B, num_action_dims) float tensor
        """
        if tokens.dim() != 2 or tokens.shape[1] != self.num_action_dims:
            raise ValueError(f"Expected tokens (B,{self.num_action_dims}), got {tuple(tokens.shape)}")
        
        low = torch.tensor(self.bound.low, device=tokens.device, dtype=torch.float32)
        high = torch.tensor(self.bound.high, device=tokens.device, dtype=torch.float32)

        tokens_before_mode = torch.clamp(tokens[:, :-1], 0, self.num_bins - 1).to(torch.float32)

        denominator = torch.clamp(high[:-1] - low[:-1], min=1e-6)
        u = (tokens_before_mode + 0.5) / self.num_bins
        actions_before_mode = low[:-1] + u * denominator # dim (B, num_action_dims-1)

        mode = torch.clamp(tokens[:, -1], 0, 2).to(torch.float32)
        mode = rearrange(mode, "b -> b 1") # dim (B,) -> (B,1)
        return torch.cat([actions_before_mode, mode], dim=1)

    def make_action_input_tokens(self, target_tokens: torch.Tensor) -> torch.Tensor:
        """
        Prepare the AutoRegressive action input tokens for the transformer.
        
        Inputs: target_tokens: (B, num_action_dims) int64 tensor
        Outputs: input_tokens: (B, num_action_dims) int64 tensor
          where input_tokens[:,0] = BOS_ID
                input_tokens[:,1:] = target_tokens[:,:-1]
        """
        if target_tokens.dim() != 2 or target_tokens.shape[1] != self.num_action_dims:
            raise ValueError(f"Expected target_tokens (B,{self.num_action_dims}), got {tuple(target_tokens.shape)}")
        
        batch_size = target_tokens.shape[0]
        input = torch.empty((batch_size, self.num_action_dims), device=target_tokens.device, dtype=torch.int64)
        input[:, 0] = self.bos_id
        input[:, 1:] = target_tokens[:, :-1]
        return input