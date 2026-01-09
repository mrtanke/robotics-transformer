from __future__ import annotations
import numpy as np
import torch
from robotics_transformer.tokenizers.action_tokenizer import ActionTokenizer


def test_encode_decode_torch():
    tokenizer = ActionTokenizer()
    a = torch.zeros(4, tokenizer.num_action_dims)
    a[:, -1] = 2.0
    t = tokenizer.encode(a)
    assert t.dtype == torch.int64
    a2 = tokenizer.decode(t)
    assert a2.shape == a.shape
    assert torch.all((a2[:, -1] >= 0) & (a2[:, -1] <= 2))


main = __name__ == "__main__"
if main:
    test_encode_decode_torch()