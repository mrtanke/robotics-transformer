from __future__ import annotations
import argparse
import itertools
from torch.utils.data import DataLoader

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robotics_transformer.data.synthetic_dataset import SyntheticRTDataset
from robotics_transformer.configs.default import robotic_transformerConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_batches", type=int, default=2)
    args = ap.parse_args()

    # cfg = robotic_transformerConfig()
    dl = DataLoader(SyntheticRTDataset(), batch_size=4, num_workers=0)

    for i, batch in enumerate(itertools.islice(dl, args.num_batches)):
        print(f"batch {i}:")
        print("  images:", tuple(batch["images"].shape))
        print("  instruction_emb:", tuple(batch["instruction_emb"].shape))
        print("  action_tokens:", tuple(batch["action_tokens"].shape))
        print("  action_tokens_history:", tuple(batch["action_tokens_history"].shape))

    print("smoke_test_dataset finished (schema looked OK)")


if __name__ == "__main__":
    main()
