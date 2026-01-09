# robotics_transformer/data/synthetic_dataset.py
from __future__ import annotations

from typing import Dict, Iterator

import torch
from torch.utils.data import IterableDataset

from robotics_transformer.data.windowing import make_history_indices


class SyntheticRTDataset(IterableDataset):
    """Generates lightweight RT-style batches without external data.

    Each episode samples a single instruction embedding, a stream of RGB frames,
    and discrete action tokens so the rest of the stack (tokenizer, policy,
    trainer) can be exercised end-to-end.
    """

    def __init__(
        self,
        *,
        history_len: int = 6,
        image_size: int = 300,
        instruction_emb: int = 512,
        action_dims: int = 11,
        action_bins: int = 256,
        num_episodes: int = 4,
        episode_length: int = 4,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.history_len = history_len
        self.image_size = image_size
        self.instruction_emb = instruction_emb
        self.action_dims = action_dims
        self.action_bins = action_bins
        self.num_episodes = num_episodes
        self.episode_length = episode_length
        self.seed = seed

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        gen = torch.Generator()
        gen.manual_seed(self.seed)

        for _ in range(self.num_episodes):
            instruction = torch.randn(self.instruction_emb, generator=gen)
            frames = []

            for t in range(self.episode_length):
                frame = torch.rand(3, self.image_size, self.image_size, generator=gen)
                frames.append(frame)

                idxs = make_history_indices(t, self.history_len)
                history = torch.stack([frames[i] for i in idxs], dim=0)

                action_tokens = torch.randint(
                    low=0,
                    high=self.action_bins,
                    size=(self.action_dims,),
                    dtype=torch.long,
                    generator=gen,
                )

                yield {
                    "images": history.to(torch.float32), # (history_len, 3, H, W)
                    "instruction_emb": instruction.to(torch.float32), # (instruction_emb,)
                    "action_tokens": action_tokens, # (action_dims,)
                }

    # num_batches = number of episodes * episode length / batch_size
    def __len__(self) -> int: 
        return self.num_episodes * self.episode_length
