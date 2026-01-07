# robotics_transformer/tokenizers/action_bounds.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass(frozen=True)
class ActionBounds:
    """
    RT-1 action layout (11 dims):
      - 7 arm dims: continuous
      - 3 base dims: continuous
      - 1 mode dim: discrete in {0,1,2} representing arm / base / terminate

    This repo keeps the layout but does not assume a specific robot.
    Set bounds consistent with your dataset.
    """
    low: np.ndarray   # (11,)
    high: np.ndarray  # (11,)
    mode_values: Tuple[int, int, int] = (0, 1, 2)  # arm, base, terminate


def default_action_bounds() -> ActionBounds:
    """
    This is the Default bounds following https://github.com/google-research/robotics_transformer.git.
    Can be replaced with dataset-accurate values before serious training.
    """

    # 11 Dimensions total: 
    # 3 (World) + 3 (Rotation) + 1 (Gripper) + 3 (Base) + 1 (Mode/Terminate)
    low = np.array([
        -1.0, -1.0, -1.0,               # world_vector
        -1.570796, -1.570796, -1.570796,# rotation_delta (-pi/2)
        -1.0,                           # gripper_closedness
        -1.0, -1.0, -1.0,               # base_displacement
        0.0                             # terminate_mode
    ], dtype=np.float32)

    high = np.array([
        1.0, 1.0, 1.0,                  # world_vector
        1.570796, 1.570796, 1.570796,   # rotation_delta (pi/2)
        1.0,                            # gripper_closedness
        1.0, 1.0, 1.0,                  # base_displacement
        2.0                             # terminate_mode (0, 1, 2)
    ], dtype=np.float32)

    return ActionBounds(low=low, high=high)
