"""Quick verification of interleaved RT-1 shapes."""
from robotics_transformer.configs.default import robotic_transformerConfig
from robotics_transformer.models.policy import robotic_transformerPolicy
import torch

cfg = robotic_transformerConfig()
model = robotic_transformerPolicy(cfg)
B = 2
images = torch.randn(B, cfg.history_len, 3, cfg.image_size, cfg.image_size)
instruction = torch.randn(B, 512)
target = torch.randint(0, cfg.action_bins, (B, cfg.action_dims))
action_history = torch.randint(0, cfg.action_bins, (B, cfg.history_len, cfg.action_dims))

with torch.no_grad():
    logits = model(images, instruction, action_history, target)
    print("forward logits:", tuple(logits.shape))
    assert logits.shape == (B, cfg.action_dims, cfg.action_bins)

    gen = model.generate_action_tokens(images, instruction, action_history)
    print("generate tokens:", tuple(gen.shape))
    assert gen.shape == (B, cfg.action_dims)

print("All shape checks passed!")
