from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Dict
import torch
from torch.optim import AdamW
from tqdm import tqdm

from robotics_transformer.training.losses import action_ce_loss, token_accuracy

@dataclass
class TrainState:
    step: int = 0
    epoch: int = 0


def train_bc(
    model: torch.nn.Module,
    dataloader: Iterable[Dict[str, torch.Tensor]],
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    log_every: int = 50,
) -> None:
    model.to(device)
    model.train()
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    state = TrainState()

    for epoch in range(epochs):
        state.epoch = epoch
        pbar = tqdm(dataloader, desc=f"epoch {epoch}", leave=True)
        
        for batch in pbar:
            images = batch["images"].to(device)
            instruction = batch["instr_emb"].to(device)
            action_tokens = batch["action_tokens"].to(device)

            logits = model(images, instruction, action_tokens)
            loss = action_ce_loss(logits, action_tokens)
            acc = token_accuracy(logits, action_tokens)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            opt.step()

            state.step += 1
            if state.step % log_every == 0:
                pbar.set_postfix(loss=float(loss.item()), acc=float(acc.item()))
