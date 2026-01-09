from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robotics_transformer.configs.default import robotic_transformerConfig
from robotics_transformer.data.synthetic_dataset import SyntheticRTDataset
from robotics_transformer.models.policy import robotic_transformerPolicy
from robotics_transformer.training.trainer import train_bc


def parse_args(cfg: robotic_transformerConfig) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RT-1 style policy on synthetic data")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=cfg.epochs, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=cfg.batch_size, help="Batch size")
    parser.add_argument("--lr", type=float, default=cfg.lr, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=cfg.weight_decay, help="AdamW weight decay")
    parser.add_argument("--grad_clip", type=float, default=cfg.grad_clip, help="Gradient clipping norm")
    parser.add_argument("--log_every", type=int, default=cfg.log_every, help="Logging interval in steps")
    parser.add_argument("--num_workers", type=int, default=cfg.num_workers, help="DataLoader worker threads")
    parser.add_argument("--num_episodes", type=int, default=8, help="Synthetic episodes per epoch")
    parser.add_argument("--episode_length", type=int, default=8, help="Synthetic timesteps per episode")
    parser.add_argument("--seed", type=int, default=0, help="Synthetic data RNG seed")
    return parser.parse_args()


def build_dataloader(cfg: robotic_transformerConfig, args: argparse.Namespace) -> DataLoader:
    dataset = SyntheticRTDataset(
        history_len=cfg.history_len,
        image_size=cfg.image_size,
        action_dims=cfg.action_dims,
        action_bins=cfg.action_bins,
        num_episodes=args.num_episodes,
        episode_length=args.episode_length,
        seed=args.seed,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


def main():
    cfg = robotic_transformerConfig()
    args = parse_args(cfg)

    dataloader = build_dataloader(cfg, args)
    model = robotic_transformerPolicy(cfg)

    metrics = train_bc(
        model=model,
        dataloader=dataloader,
        device=torch.device(args.device),
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        log_every=args.log_every,
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
