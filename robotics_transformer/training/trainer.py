# robotics_transformer/training/trainer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Dict, List
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
) -> Dict[str, object]:
    model.to(device)
    model.train()
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    state = TrainState()
    epoch_stats: List[Dict[str, float | int]] = []
    last_loss = float("nan")
    last_acc = float("nan")

    for epoch in range(epochs):
        state.epoch = epoch
        pbar = tqdm(dataloader, desc=f"epoch {epoch}", leave=True)

        loss_sum = 0.0
        acc_sum = 0.0
        batches = 0
        
        for batch in pbar:
            images = batch["images"].to(device)
            instruction_tensor = batch.get("instr_emb")
            if instruction_tensor is None:
                instruction_tensor = batch["instruction_emb"]
            instruction = instruction_tensor.to(device)
            action_tokens = batch["action_tokens"].to(device)

            logits = model(images, instruction, action_tokens)
            loss = action_ce_loss(logits, action_tokens)
            acc = token_accuracy(logits, action_tokens)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            opt.step()

            state.step += 1
            loss_value = float(loss.item())
            acc_value = float(acc.item())
            last_loss = loss_value
            last_acc = acc_value
            loss_sum += loss_value
            acc_sum += acc_value
            batches += 1
            if state.step % log_every == 0:
                pbar.set_postfix(loss=loss_value, acc=acc_value)

        avg_loss = loss_sum / batches if batches else float("nan")
        avg_acc = acc_sum / batches if batches else float("nan")
        epoch_stats.append({
            "epoch": epoch,
            "loss": avg_loss,
            "acc": avg_acc,
            "batches": batches,
        })

    return {
        "total_steps": state.step,
        "epochs": epochs,
        "last_loss": last_loss,
        "last_acc": last_acc,
        "epoch_stats": epoch_stats,
    }
