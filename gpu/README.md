# GPU tool environment

Separate Python 3.12 environment for SAM Audio and Judge. Their laion-clap dependency
requires numpy <2; the root project and CI remain independent of GPU dependencies.

```bash
uv sync --project gpu --locked
```

Weights are never installed with this environment. Use an existing read-only HF
cache and `HF_HUB_OFFLINE=1`, `CUDA_VISIBLE_DEVICES=0` for model calls.

PyTorch, torchaudio and torchvision use the official CUDA 12.8 wheel index, with
no version constraints added. This keeps the resolved runtime compatible with
the sv3-a100 driver 570; the default PyPI torch now selects CUDA 13.
