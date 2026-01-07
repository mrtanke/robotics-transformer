from __future__ import annotations
import argparse
import itertools
from torch.utils.data import DataLoader

from robotics_transformer.data.tfds_rlds_iterable import TFDSRLDSIterableDataset
from robotics_transformer.configs.default import robotic_transformerConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfds_dir", type=str, required=True, help="Local TFDS directory (builder_from_directory).")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--num_batches", type=int, default=2)
    args = ap.parse_args()

    cfg = robotic_transformerConfig()
    ds = TFDSRLDSIterableDataset(
        tfds_dir=args.tfds_dir,
        split=args.split,
        history_len=cfg.history_len,
        image_size=cfg.image_size,
    )
    dl = DataLoader(ds, batch_size=cfg.batch_size, num_workers=0)

    for i, batch in enumerate(itertools.islice(dl, args.num_batches)):
        print(f"batch {i}:")
        print("  images:", tuple(batch["images"].shape))
        print("  instr_emb:", tuple(batch["instr_emb"].shape))
        print("  action_tokens:", tuple(batch["action_tokens"].shape))

    print("smoke_test_dataset finished (schema looked OK)")


if __name__ == "__main__":
    main()
