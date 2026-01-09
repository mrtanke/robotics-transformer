# robotics_transformer/data/windowing.py
from __future__ import annotations
from typing import List


def make_history_indices(t: int, history_len: int) -> List[int]:
    """
    Given current time t and desired history length, return list of indices
    for history frames, padding with the first frame as needed.

    All the cases for history_len=6:
    t=0 -> [0,0,0,0,0,0]
    t=1 -> [0,0,0,0,0,1]
    t=2 -> [0,0,0,0,1,2]
    t=3 -> [0,0,0,1,2,3]
    t=4 -> [0,0,1,2,3,4]
    t=5 -> [0,1,2,3,4,5]
    """
    idx = []
    for k in range(history_len):
        i = t - (history_len - 1 - k)
        idx.append(max(i, 0))
    return idx
