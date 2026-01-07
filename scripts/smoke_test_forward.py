from __future__ import annotations
import torch

from robotics_transformer.configs.default import robotic_transformerConfig
from robotics_transformer.models.policy import robotic_transformerPolicy
from robotics_transformer.tokenizers.action_tokenizer import ActionTokenizer


def main():
    cfg = robotic_transformerConfig()
    tokenizer = ActionTokenizer()
    model = robotic_transformerPolicy(cfg)

    B = 2
    images = torch.randn(B, cfg.history_len, 3, cfg.image_size, cfg.image_size) # (B, 6, 3, H, W)
    instruction = torch.randn(B, 512)
    target = torch.randint(0, cfg.action_bins, (B, cfg.action_dims)) # (B, 11)
    target[:, -1] = torch.randint(0, 3, (B,)) # mode dim in {0,1,2}

    with torch.no_grad():
        logits = model(images, instruction, target)
    print("logits:", tuple(logits.shape))
    assert logits.shape == (B, cfg.action_dims, cfg.action_bins)
    print("smoke_test_forward passed")


if __name__ == "__main__":
    main()
