# robotics_transformer/models/film_efficientnet_b3.py
from __future__ import annotations

from typing import List, Tuple
from einops.layers.torch import Rearrange
import torch
import torch.nn as nn
import torchvision

from robotics_transformer.models.film import FiLM


class FiLMEfficientNetB3Tokenizer(nn.Module):
    """
    Minimal, FiLM EfficientNet-B3 tokenizer.
      - produce a (B, 81, D) token sequence from (image, instruction_embedding)
      - preserve the idea of FiLM conditioning with identity initialization

    Steps:
      1. run EfficientNet-B3 `features` stages
      2. apply FiLM after each top-level stage output
      3. project to token_dim via Linear 1x1 Conv
      4. adaptive pool to (9,9)
      5. flatten to 81 tokens
    """
    def __init__(self, image_size: int = 300, text_dim: int = 512, token_dim: int = 512, grid: Tuple[int, int] = (9, 9)):
        super().__init__()
        self.image_size = image_size
        self.text_dim = text_dim
        self.token_dim = token_dim
        self.grid = grid

        self.stages = torchvision.models.efficientnet_b3(weights='DEFAULT').features

        stage_channels = self._infer_stage_channels()
        self.films = nn.ModuleList([FiLM(text_dim=text_dim, channels=channel) for channel in stage_channels])

        last_channel = stage_channels[-1]
        self.proj = nn.Conv2d(last_channel, token_dim, kernel_size=1, stride=1, padding=0)
        self.pool = nn.AdaptiveAvgPool2d(self.grid)
        self.flatten_tokens = Rearrange("b d h w -> b (h w) d") 
    @torch.no_grad()
    def _infer_stage_channels(self) -> List[int]:
        x = torch.zeros(1, 3, self.image_size, self.image_size)
        channels = []
        for stage in self.stages:
            x = stage(x)
            channels.append(int(x.shape[1]))
        
        return channels

    def forward(self, images: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """
        images: (B, 3, H, W)
        text_emb: (B, text_dim)
        returns: (B, 81, token_dim)
        """
        x = images
        for stage, film in zip(self.stages, self.films):
            x = stage(x)
            x = film(x, text_emb)

        x = self.proj(x) # (B, token_dim, h, w)
        x = self.pool(x) # (B, token_dim, 9, 9)
        return self.flatten_tokens(x)
