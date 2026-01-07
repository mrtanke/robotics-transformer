from __future__ import annotations

from einops import rearrange, repeat
import torch
import torch.nn as nn

from robotics_transformer.configs.default import robotic_transformerConfig
from robotics_transformer.models.film_efficientnet_b3 import FiLMEfficientNetB3Tokenizer
from robotics_transformer.models.token_learner import TokenLearner
from robotics_transformer.models.transformer_decoder import TransformerDecoder
from robotics_transformer.tokenizers.action_tokenizer import ActionTokenizer


class robotic_transformerPolicy(nn.Module):
    """
    RT-1 policy:
      - tokens from 6 images (FiLM EffNet-B3 -> 81 -> TokenLearner -> 8/image)
      - decoder-only transformer with causal mask
      - autoregressive prediction of 11 action tokens (256 bins each)
    """
    def __init__(self, cfg: robotic_transformerConfig):
        super().__init__()
        self.cfg = cfg
        self.action_tokenizer = ActionTokenizer()

        self.vision = FiLMEfficientNetB3Tokenizer(
            image_size=cfg.image_size,
            text_dim=512,
            token_dim=cfg.vision_token_dim,
            grid=cfg.vision_grid,
        )

        self.token_learner = TokenLearner(
            token_dim=cfg.vision_token_dim,
            num_tokens_out=cfg.tokens_per_image,
            hidden_dim=256,
        )

        self.action_emb = nn.Embedding(self.action_tokenizer.vocab_size, cfg.vision_token_dim) # vocab_size=num_bins+1
        self.transformer = TransformerDecoder(
            dim=cfg.vision_token_dim,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            ff_mult=cfg.ff_mult,
            dropout=cfg.dropout,
            max_seq_len=cfg.max_seq_len,
        )

        self.head = nn.Linear(cfg.vision_token_dim, cfg.action_bins)

    def encode_obs(self, images_history: torch.Tensor, instruction_emb: torch.Tensor) -> torch.Tensor:
        """
        Instruction and image tokenization.

        images_history: (B, 6, 3, H, W) - 6 history images
        instruction_emb: (B, 512)
        returns: (B, 48, D) - encoded observation tokens
        """
        x = rearrange(images_history, "b t c h w -> (b t) c h w")
        text = repeat(instruction_emb, "b d -> (b t) d", t=images_history.shape[1])
        
        tokens_81 = self.vision(x, text) # (B*T, 81, D)
        tokens_8 = self.token_learner(tokens_81) # (B*T, 8, D)
        tokens_8 = rearrange(
            tokens_8,
            "(b t) n d -> b (t n) d",
            t=images_history.shape[1],
        )
        return tokens_8  # (B, 48, D)

    def forward(self, images_history: torch.Tensor, instruction_emb: torch.Tensor, target_action_tokens: torch.Tensor) -> torch.Tensor:
        """
        images_history: (B, 6, 3, H, W)
        instruction_emb: (B, 512)
        target_action_tokens: (B, 11)
        returns: (B, 11, 256)
        """
        observation_tokens = self.encode_obs(images_history, instruction_emb)

        action_in = self.action_tokenizer.make_action_input_tokens(target_action_tokens) # (B, 11)
        action_in_emb = self.action_emb(action_in) # (B, 11, D)

        x = torch.cat([observation_tokens, action_in_emb], dim=1) # (B, 59, D)
        h = self.transformer(x)
        h_action = h[:, -self.cfg.action_dims:, :] # (B, 11, D) = (B, 11, 512)
        logits = self.head(h_action) # (B, 11, 256)
        return logits

    @torch.no_grad()
    def generate_action_tokens(self, images_history: torch.Tensor, instruction_emb: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Autoregressive action token generation.
        images_history: (B, 6, 3, H, W)
        instruction_emb: (B, 512)
        temperature: sampling temperature
        returns: (B, 11) predicted action tokens
        """
        self.eval()
        B = images_history.shape[0]
        observation_tokens = self.encode_obs(images_history, instruction_emb)

        current_sequence = torch.full(
            (B, 1), 
            self.action_tokenizer.bos_id, 
            device=images_history.device, 
            dtype=torch.int64
        )
        predicted_action_tokens = torch.empty((B, 0), device=images_history.device, dtype=torch.int64)

        for _ in range(self.cfg.action_dims):
            action_emb = self.action_emb(current_sequence)
            x = torch.cat([observation_tokens, action_emb], dim=1)
            h = self.transformer(x)
            last_token = h[:, -1, :]

            logits = self.head(last_token) / max(temperature, 1e-6)
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1) # (B,1)
            predicted_action_tokens = torch.cat([predicted_action_tokens, next_token], dim=1)
            current_sequence = torch.cat([current_sequence, next_token], dim=1)

        return predicted_action_tokens
