FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uv /usr/local/bin/uv
RUN for i in 1 2 3; do \
      apt-get update && apt-get install -y -o Acquire::Retries=5 --no-install-recommends \
        ca-certificates ffmpeg git gcc libc6-dev \
      && rm -rf /var/lib/apt/lists/* && exit 0; \
      echo "apt attempt $i failed, retrying"; sleep 15; \
    done; exit 1
ENV UV_PYTHON_INSTALL_DIR=/opt/python
RUN uv python install 3.12
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ src/
COPY gen/pyproject.toml gen/uv.lock gen/.python-version gen/
ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv \
    for i in 1 2 3 4 5; do \
      uv sync --project gen --locked --no-dev --python 3.12 && exit 0; \
      echo "uv sync attempt $i failed, retrying"; sleep 10; \
    done; exit 1
ENV PATH="/app/gen/.venv/bin:$PATH" HF_HOME=/hf HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN useradd --create-home --uid 10001 app
USER app
EXPOSE 8700
CMD ["audyssey", "tool-server", "--tool", "generation", "--impl", "real", "--host", "0.0.0.0", "--port", "8700"]
