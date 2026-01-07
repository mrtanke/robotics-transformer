from __future__ import annotations
import argparse
import torch
from torch.utils.data import DataLoader

from robotics_transformer.configs.default import robotic_transformerConfig
from robotics_transformer.data.tfds_rlds_iterable import TFDSRLDSIterableDataset
from robotics_transformer.models.policy import robotic_transformerPolicy
from robotics_transformer.training.trainer import train_bc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfds_dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    cfg = robotic_transformerConfig()
    epochs = args.epochs if args.epochs is not None else cfg.epochs

    ds = TFDSRLDSIterableDataset(tfds_dir=args.tfds_dir, split=args.split, history_len=cfg.history_len, image_size=cfg.image_size)
    dl = DataLoader(ds, batch_size=cfg.batch_size, num_workers=0)

    model = robotic_transformerPolicy(cfg)
    train_bc(
        model=model,
        dataloader=dl,
        device=torch.device(args.device),
        epochs=epochs,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        grad_clip=cfg.grad_clip,
        log_every=cfg.log_every,
    )


if __name__ == "__main__":
    main()
