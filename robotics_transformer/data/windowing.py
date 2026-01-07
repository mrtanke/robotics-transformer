# robotics_transformer/data/windowing.py
from __future__ import annotations
from typing import List


def make_history_indices(t: int, history_len: int) -> List[int]:
    """
    Given current time t and desired history length, return list of indices
    for history frames, padding with the first frame as needed.
    E.g., t=2, history_len=6 -> [0,0,0,1,2]
    """
    idx = []
    for k in range(history_len):
        i = t - (history_len - 1 - k)
        idx.append(max(i, 0))
    return idx
