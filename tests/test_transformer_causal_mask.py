from __future__ import annotations
import torch
from robotics_transformer.models.transformer_decoder import TransformerDecoder


def test_transformer_runs():
    model = TransformerDecoder(dim=64, n_layers=2, n_heads=4, ff_mult=4, dropout=0.0, max_seq_len=32)
    x = torch.randn(2, 16, 64)
    y = model(x)
    assert y.shape == x.shape
