from __future__ import annotations

from einops import rearrange, reduce
import torch
import torch.nn.functional as F


def action_ce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if targets.dim() != 2:
        raise ValueError(f"targets must be (B,11), got {tuple(targets.shape)}")

    logits_flat = rearrange(logits, "b t n -> (b t) n")
    targets_flat = rearrange(targets, "b t -> (b t)")
    return F.cross_entropy(logits_flat, targets_flat)

def token_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    preds = logits.argmax(dim=-1) # (B, T)
    return reduce((preds == targets).float(), "b t ->", "mean") # scalar
