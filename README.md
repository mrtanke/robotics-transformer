# robotics-transformer
Implementation of **RT1 (Robotic Transformer)** in Pytorch, from [RT-1: Robotics Transformer for Real-World Control at Scale](https://arxiv.org/abs/2212.06817).

This repo focus on core ideas appeared in the paper, like model design: FiLM-conditioned EfficientNet tokens, TokenLearner compression, and a causal decoder that predicts discretized actions. To keep things lightweight and easy to study, it replaces heavy TFDS pipelines with a small synthetic data generator. It mirrors the data shapes of RT-1 (6 image frames + 512-d instruction + 11 action tokens).

More implementation details will land in an upcoming blog post: [RT-1 From-Scratch Notes (coming soon)](https://example.com/blog-post-placeholder)

## Install

Create a virtual environment (optional) and install the project in editable mode:

```bash
python -m venv .venv
.\.venv\Scripts\activate  # use source .venv/bin/activate on Unix
pip install -e .
```

## Synthetic dataset

We use [`SyntheticRTDataset`](robotics_transformer/data/synthetic_dataset.py) to emulate RT-1 batches:

- `images`: `(history_len, 3, image_size, image_size)`
- `instruction_emb`: `(512,)` per episode, reused across its timesteps
- `action_tokens`: `(action_dims,)` integers in `[0, action_bins)`

Default windowing matches RT-1 conventions, in [`robotic_transformerConfig`](robotics_transformer/configs/default.py):

- history frames: `history_len = 6`
- instruction embedding: fixed for each episode
- action dims: `action_dims = 11` with `action_bins = 256`

## Quick sanity checks

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

## Training

```bash
python scripts/smoke_test_trainbc.py --epochs train.py
```


## Acknowledgements

This project draws inspiration from RT-1 and related works on large-scale robotic transformers.