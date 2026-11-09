FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04
COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates ffmpeg git \
    && rm -rf /var/lib/apt/lists/*
ENV UV_PYTHON_INSTALL_DIR=/opt/python
RUN uv python install 3.12
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ src/
COPY gpu/pyproject.toml gpu/uv.lock gpu/.python-version gpu/
# Several large git dependencies; GitHub sometimes drops one clone mid-transfer.
# The cache mount keeps finished clones between retries and stays out of the image.
ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv \
    for i in 1 2 3 4 5; do \
      uv sync --project gpu --locked --no-dev --python 3.12 && exit 0; \
      echo "uv sync attempt $i failed, retrying"; sleep 10; \
    done; exit 1
ENV PATH="/app/gpu/.venv/bin:$PATH" HF_HOME=/hf HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
# torchcodec cu128 needs NVIDIA NPP from the nvidia-npp-cu12 wheel at load time.
ENV LD_LIBRARY_PATH="/app/gpu/.venv/lib/python3.12/site-packages/nvidia/npp/lib:$LD_LIBRARY_PATH"
EXPOSE 8700
CMD ["audyssey", "tool-server", "--tool", "separation", "--tool", "filtering", "--impl", "real", "--host", "0.0.0.0", "--port", "8700"]
