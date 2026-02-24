## robotics-transformer
Implementation of **RT1 (Robotic Transformer)** in Pytorch, from [RT-1: Robotics Transformer for Real-World Control at Scale](https://arxiv.org/abs/2212.06817).

This repo focus on core ideas appeared in the paper, like model design: FiLM-conditioned EfficientNet tokens, TokenLearner compression, and a causal decoder that predicts discretized actions. Following the official implementation, the Transformer input uses a **per-timestep interleaved** structure: `[obs_1][act_1][obs_2][act_2]...[obs_T][act_T]`, where each timestep contributes 8 observation tokens + 11 action tokens = 19 tokens, for a total of 6 × 19 = 114 tokens. To keep things lightweight and easy to study, it replaces heavy TFDS pipelines with a small synthetic data generator.

You can find more implementation details in: [Reproducing Robotics Transformer 1](https://mrtanke.github.io/projects/2026-01-10-reproducing-robotics-transformer-1/).

### Install

Create a virtual environment (optional) and install the project in editable mode:

```bash
python -m venv .venv
.\.venv\Scripts\activate  # use source .venv/bin/activate on Unix
pip install -e .
```

### Synthetic dataset

We use [`SyntheticRTDataset`](robotics_transformer/data/synthetic_dataset.py) to emulate RT-1 batches:

- `images`: `(history_len, 3, image_size, image_size)`
- `instruction_emb`: `(512,)` per episode, reused across its timesteps
- `action_tokens`: `(action_dims,)` integers in `[0, action_bins)` — the target for the current timestep
- `action_tokens_history`: `(history_len, action_dims)` — ground-truth actions for all T timesteps in the history window

Default windowing matches RT-1 conventions, in [`robotic_transformerConfig`](robotics_transformer/configs/default.py):

- history frames: `history_len = 6`
- instruction embedding: fixed for each episode
- action dims: `action_dims = 11` with `action_bins = 256`

### Quick sanity checks

1. Inspect the synthetic loader and tensor shapes:
   ```bash
   python scripts/smoke_test_dataset.py
   ```
2. Verify the policy forward pass and logits shape:
   ```bash
   python scripts/smoke_test_forward.py
   ```
3. Run the behavioral cloning smoke test (prints per-epoch stats):
   ```bash
   python scripts/smoke_test_trainbc.py
   ```
4. Execute the unit tests for tokenization, masking, and module shapes:
   ```bash
   pytest
   ```

### Training

```bash
python scripts/smoke_test_trainbc.py
```


### Acknowledgements

This project draws inspiration from RT-1 and related works on robotic transformers.