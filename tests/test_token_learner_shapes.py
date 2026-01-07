from __future__ import annotations
import torch
from robotics_transformer.models.token_learner import TokenLearner


def test_token_learner_shapes():
    tl = TokenLearner(token_dim=512, num_tokens_out=8)
    x = torch.randn(2, 81, 512)
    y = tl(x)
    assert y.shape == (2, 8, 512)
