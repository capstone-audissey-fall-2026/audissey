# GPU tool environment

Separate Python 3.12 environment for SAM Audio, Judge, and SAM3 grounding. Their laion-clap dependency
requires numpy <2; the root project and CI remain independent of GPU dependencies.

```bash
uv sync --project gpu --locked
```

Weights are never installed with this environment. Use an existing read-only HF
cache and `HF_HUB_OFFLINE=1`, `CUDA_VISIBLE_DEVICES=0` for model calls.

PyTorch, torchaudio and torchvision use the official CUDA 12.8 wheel index, with
no version constraints added. This keeps the resolved runtime compatible with
the sv3-a100 driver 570; the default PyPI torch now selects CUDA 13.

Run all three tools with one process (inject the token through the environment):

```bash
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 HF_HOME=/home/common/huggingface
export AUDYSSEY_SAM_AUDIO_MODEL=mrfakename/sam-audio-large-tv
export AUDYSSEY_SAM_AUDIO_JUDGE_MODEL=mrfakename/sam-audio-judge
uv run --project gpu --locked audyssey tool-server --tool separation --tool filtering --tool grounding --impl real --host 0.0.0.0 --port 8700
uv run --project gpu --locked python scripts/sam_audio_smoke.py samples/sample.mp4 --description "a person speaking" --out work/jobs/smoke
```

The lock resolves numpy 1.26.4, torch/torchaudio 2.11.0+cu128, and SAM3 at
`2814fa619404a722d03e9a012e083e4f293a4e53`. Resolution, installation and CPU-side
SAM3 import passed. SAM Audio GPU measurements are recorded in the design;
SAM3 GPU inference and Docker E2E for grounding remain pending operator verification.
No third-party dependency constraints are overridden.

`huggingface-hub<1.0` preserves the loader API used by the pinned SAM Audio revision.
Torchcodec comes from the cu128 index and uses `nvidia-npp-cu12`; set
`LD_LIBRARY_PATH` as shown in the [deployment guide](../README.md).
SAM3 uses existing cached `facebook/sam3.1/sam3.1_multiplex.pt` weights only, with
`local_files_only=True`. Optional FlashAttention 3 and compilation are disabled.
The pinned multiplex model requires CUDA even though its modules can be imported
without constructing a GPU model.
