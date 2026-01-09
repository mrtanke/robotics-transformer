# scripts/somke_test_trainbc.py
from __future__ import annotations
import torch
from torch.utils.data import DataLoader

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robotics_transformer.configs.default import robotic_transformerConfig
from robotics_transformer.data.synthetic_dataset import SyntheticRTDataset
from robotics_transformer.models.policy import robotic_transformerPolicy
from robotics_transformer.training.trainer import train_bc


def main():
    cfg = robotic_transformerConfig()

    dl = DataLoader(SyntheticRTDataset(), batch_size=4, num_workers=0)

    model = robotic_transformerPolicy(cfg)
    metrics = train_bc(
        model=model,
        dataloader=dl,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        epochs=cfg.epochs,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        grad_clip=cfg.grad_clip,
        log_every=cfg.log_every,
    )

    print("\nTraining summary:")
    print(f"  total steps: {metrics['total_steps']}")
    print(f"  last loss: {metrics['last_loss']:.4f}")
    print(f"  last acc: {metrics['last_acc']:.4f}")
    for stat in metrics["epoch_stats"]:
        print(
            f"  epoch {int(stat['epoch'])}: batches={int(stat['batches'])} "
            f"loss={stat['loss']:.4f} acc={stat['acc']:.4f}"
        )

    


if __name__ == "__main__":
    main()
