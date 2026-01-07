# robotics_transformer/data/tfds_rlds_iterable.py
from __future__ import annotations

from typing import Any, Dict, Iterator, Optional, Sequence, Tuple
from einops import rearrange
import numpy as np
import torch
from torch.utils.data import IterableDataset

from robotics_transformer.data.transforms import build_image_transform
from robotics_transformer.data.windowing import make_history_indices
from robotics_transformer.tokenizers.action_tokenizer import ActionTokenizer


def _first_existing(d: Dict[str, Any], keys: Sequence[str]) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    raise KeyError(f"None of keys exist: {keys}. available={list(d.keys())[:30]} ...")


class TFDSRLDSIterableDataset(IterableDataset):
    """
    Minimal TFDS/RLDS -> PyTorch IterableDataset.

    Expected step fields (heuristics):
    - image: observation.image OR observation.rgb OR observation.camera_image
    - instruction embedding: observation.natural_language_embedding OR observation.language_embedding (512,)
    - action: step.action as float vector (11,)

    If your dataset differs, edit `_extract_step()` only.
    """
    def __init__(
        self,
        tfds_dir: str,
        split: str = "train",
        history_len: int = 6,
        image_size: int = 300,
        shuffle_buffer: int = 0,
    ):
        super().__init__()
        self.tfds_dir = tfds_dir
        self.split = split
        self.history_len = history_len
        self.image_size = image_size
        self.action_tokenizer = ActionTokenizer()
        self.shuffle_buffer = shuffle_buffer
        self.img_tf = build_image_transform(image_size)

    def _load_tfds(self):
        import tensorflow_datasets as tfds
        builder = tfds.builder_from_directory(self.tfds_dir)
        ds = builder.as_dataset(split=self.split, shuffle_files=False)
        return ds

    def _extract_step(self, step: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs = step.get("observation", step)
        
        # Official RT-1 uses 'image' key specifically
        img = _first_existing(obs, ["image"])
        img_np = np.array(img)

        instruction = _first_existing(obs, ["natural_language_embedding", "language_embedding"])
        instruction_np = np.array(instruction, dtype=np.float32).flatten()
        
        action_data = step.get("action", None)
        if action_data is None:
            raise KeyError("Missing step['action']")

        # If action is a dictionary (common in official RLDS), we must flatten it
        if isinstance(action_data, dict):
            action_np = np.concatenate([
                action_data['world_vector'],            # [x, y, z]
                action_data['rotation_delta'],         # [roll, pitch, yaw]
                action_data['gripper_closedness_action'], # [open/close]
                action_data.get('terminate_episode', np.array([1, 0, 0])) # [mode]
            ]).astype(np.float32)
        else:
            action_np = np.array(action_data, dtype=np.float32).flatten()

        if instruction_np.shape[0] != 512:
            raise ValueError(f"Instruction dim mismatch: {instruction_np.shape}")
            
        if action_np.shape[0] != self.action_tokenizer.dims:
            if action_np.shape[0] < self.action_tokenizer.dims:
                padding = np.zeros(self.action_tokenizer.dims - action_np.shape[0])
                action_np = np.concatenate([action_np, padding])

        return img_np, instruction_np, action_np

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        ds = self._load_tfds()

        for episode in ds:
            steps = episode["steps"]

            imgs, instruction, actions = [], [], []
            for step in steps:
                img_np, instruction_np, actions_np = self._extract_step(step)
                
                imgs.append(img_np)
                instruction.append(instruction_np)
                actions.append(actions_np)

            if len(imgs) == 0:
                continue

            # The instruction is constant per episode
            instruction0 = instruction[0]

            for t in range(len(imgs)):
                idxs = make_history_indices(t, self.history_len)
                history_imgs = [imgs[i] for i in idxs]

                imgs = []
                for img in history_imgs:
                    if img.dtype != np.uint8:
                        img = img.astype(np.uint8)
                    imgs.append(self.img_tf(img))
                imgs = torch.stack(imgs, dim=0)  # (6,3,H,W)

                instruction = torch.from_numpy(instruction0).to(torch.float32)

                actions = torch.from_numpy(actions[t]).to(torch.float32)
                actions_batch = rearrange(actions, "d -> 1 d")
                actions_tokens = rearrange(self.action_tokenizer.encode(actions_batch), "1 d -> d")

                # imgs: (6, 3, H, W), instruction: (512,), actions_tokens: (11,)
                yield {"images": imgs, "instr_emb": instruction, "action_tokens": actions_tokens} 