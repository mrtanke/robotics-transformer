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
    RT-1 policy
      - tokens from T images (FiLM EffNet-B3 -> 81 -> TokenLearner -> 8/image)
      - decoder-only transformer with causal mask
      - per-timestep interleaved input: [obs_1|act_1|obs_2|act_2|...|obs_T|act_T]
      - autoregressive prediction of 11 action tokens (256 bins each) at last timestep
    """
    def __init__(self, cfg: robotic_transformerConfig):
        super().__init__()
        self.cfg = cfg
        self.action_tokenizer = ActionTokenizer()

        self.vision = FiLMEfficientNetB3Tokenizer(
            image_size=cfg.image_size,
            instruction_dim=512,
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


    # Observation encoder
    def encode_obs(self, images_history: torch.Tensor, instruction_emb: torch.Tensor) -> torch.Tensor:
        """
        Instruction and image tokenization, preserving per-timestep grouping.

        images_history: (B, T, 3, H, W)
        instruction_emb: (B, 512)
        returns: (B, T, tokens_per_image, D)
        """
        T = images_history.shape[1]
        x = rearrange(images_history, "b t c h w -> (b t) c h w")
        text = repeat(instruction_emb, "b d -> (b t) d", t=T)

        tokens_81 = self.vision(x, text)         # (B*T, 81, D)
        tokens_8 = self.token_learner(tokens_81)  # (B*T, tokens_per_image, D)
        tokens_8 = rearrange(tokens_8, "(b t) n d -> b t n d", t=T)
        return tokens_8  # (B, T, 8, D)


    # Helpers: build the interleaved [obs_t | act_t] sequence
    def _interleave(self, obs_tokens: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        """
        obs_tokens: (B, T, N_obs, D)
        act_emb:    (B, T, N_act, D)
        returns:    (B, T*(N_obs+N_act), D)
        """
        per_step = torch.cat([obs_tokens, act_emb], dim=2)  # (B, T, N_obs+N_act, D)
        return rearrange(per_step, "b t n d -> b (t n) d")


    # Training forward (teacher-forced)
    def forward(
        self,
        images_history: torch.Tensor,
        instruction_emb: torch.Tensor,
        action_tokens_history: torch.Tensor,
        target_action_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        images_history:       (B, T, 3, H, W)
        instruction_emb:      (B, 512)
        action_tokens_history: (B, T, 11) – ground-truth actions for *every* timestep
        target_action_tokens:  (B, 11)    – target for the current (last) timestep
        returns:               (B, 11, 256) – logits for last timestep only
        """
        obs_tokens = self.encode_obs(images_history, instruction_emb)  # (B, T, 8, D)

        # Build per-timestep action token ids
        #   t < T-1 : ground-truth past actions (used as context)
        #   t = T-1 : teacher-forced (BOS + shifted target)
        act_ids = action_tokens_history.clone()          # (B, T, 11)
        teacher_forced = self.action_tokenizer.make_action_input_tokens(
            target_action_tokens
        )                                                 # (B, 11)
        act_ids[:, -1, :] = teacher_forced

        act_emb = self.action_emb(act_ids)               # (B, T, 11, D)

        x = self._interleave(obs_tokens, act_emb)        # (B, T*19, D)
        h = self.transformer(x)

        # Last T-th group's action positions are the final action_dims tokens
        h_action = h[:, -self.cfg.action_dims:, :]       # (B, 11, D)
        logits = self.head(h_action)                      # (B, 11, 256)
        return logits


    # Inference (autoregressive generation at the last timestep)
    @torch.no_grad()
    def generate_action_tokens(
        self,
        images_history: torch.Tensor,
        instruction_emb: torch.Tensor,
        action_tokens_history: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Autoregressive action-token generation for the *current* (last) timestep.

        images_history:        (B, T, 3, H, W)
        instruction_emb:       (B, 512)
        action_tokens_history: (B, T, 11) – past actions; last-timestep column is ignored
        temperature:           sampling temperature
        returns:               (B, 11)
        """
        self.eval()
        B = images_history.shape[0]
        T = images_history.shape[1]
        device = images_history.device

        obs_tokens = self.encode_obs(images_history, instruction_emb)  # (B, T, 8, D)

        # ---- fixed context: past timesteps (t=0..T-2) ----
        if T > 1:
            past_obs = obs_tokens[:, :-1]                              # (B, T-1, 8, D)
            past_act_emb = self.action_emb(action_tokens_history[:, :-1])  # (B, T-1, 11, D)
            past_context = self._interleave(past_obs, past_act_emb)    # (B, (T-1)*19, D)
            cur_obs = obs_tokens[:, -1]                                 # (B, 8, D)
            context = torch.cat([past_context, cur_obs], dim=1)        # (B, (T-1)*19 + 8, D)
        else:
            context = obs_tokens[:, 0]                                  # (B, 8, D)

        # ---- autoregressive loop for last timestep's action tokens ----
        cur_act = torch.full((B, 1), self.action_tokenizer.bos_id,
                             device=device, dtype=torch.int64)
        predicted = torch.empty((B, 0), device=device, dtype=torch.int64)

        for _ in range(self.cfg.action_dims):
            act_emb = self.action_emb(cur_act)                         # (B, k, D)
            x = torch.cat([context, act_emb], dim=1)
            h = self.transformer(x)
            last_h = h[:, -1, :]                                       # (B, D)

            logits = self.head(last_h) / max(temperature, 1e-6)
            probs = torch.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)         # (B, 1)

            predicted = torch.cat([predicted, next_tok], dim=1)
            cur_act = torch.cat([cur_act, next_tok], dim=1)

        return predicted  # (B, 11)
