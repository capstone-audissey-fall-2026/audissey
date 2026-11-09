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

Run both tools with one process (inject the token through the environment):

```bash
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 HF_HOME=/home/common/huggingface
export AUDYSSEY_SAM_AUDIO_MODEL=mrfakename/sam-audio-large-tv
export AUDYSSEY_SAM_AUDIO_JUDGE_MODEL=mrfakename/sam-audio-judge
uv run --project gpu --locked audyssey tool-server --tool separation --tool filtering --impl real --host 0.0.0.0 --port 8700
uv run --project gpu --locked python scripts/sam_audio_smoke.py samples/sample.mp4 --description "a person speaking" --out work/jobs/smoke
```

The lock resolves numpy 1.26.4 and torch/torchaudio 2.11.0+cu128. Locked installation
was verified; GPU imports, timing, VRAM and Docker image sizes remain **pending
measurement** because the sandbox launcher refused execution. Do not interpret a
successful installation or `/health` response as verified model inference.
See the [team deployment guide](../README.md) for compose and RunPod settings.
