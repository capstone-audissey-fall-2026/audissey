# Generation tool environment

SelVA FT + SelSync and LAION-CLAP run in this separate Python 3.12 project.
SAM Audio's `gpu/` environment and root dependencies are unchanged.

```bash
uv sync --project gen --locked
HF_HUB_OFFLINE=1 uv run --project gen --locked python -c 'import selva, laion_clap'
export HF_HOME=/home/common/huggingface HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
export AUDYSSEY_TOOL_TOKEN=<private-token>
uv run --project gen --locked audyssey tool-server --tool generation --impl real --host 127.0.0.1 --port 8700
```

The lock uses torch/torchaudio 2.6.0+cu124, torchvision 0.21.0+cu124, numpy 2.0.2,
transformers 4.51.3, open-clip-torch 3.3.0, av 17.0.1, and laion-clap 1.1.5.
PyTorch packages use an explicit official CUDA 12.4 index. CLAP 1.1.5 resolves
with numpy 2 without overriding any third-party dependency constraints; its
embedding methods return numpy arrays and do not accept `use_tensor`.

SelVA is installed from upstream commit
`d7d40a992aab58e7cf246055681a657e5d8b4a4d`. The reference fork's transformers
pin conflicts with the evaluation environment. No SelVA code or model weights
are copied into this repository or image.

Lock, sync and CPU imports have been verified. The shared uv cache had an
unreadable entry during verification; `UV_CACHE_DIR=/tmp/audyssey-selva-uv`
provided a task-owned cache. A transient GitHub fetch failure was retried with
command-scoped `GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=http.version
GIT_CONFIG_VALUE_0=HTTP/1.1`. Imports emit an upstream timm deprecation warning.

Prepare the [offline weights](../docs/running.md#가중치-준비) first; the adapter
requires `HF_HUB_OFFLINE=1` before loading any models. No model downloads happen
at server runtime. GPU inference, 8s/15s quality and memory measurements, Docker
build, and six-container Compose E2E remain pending operator verification.
