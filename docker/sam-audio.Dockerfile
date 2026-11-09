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
RUN uv sync --project gpu --locked --no-dev --python 3.12 && uv cache clean
ENV PATH="/app/gpu/.venv/bin:$PATH" HF_HOME=/hf HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
EXPOSE 8700
CMD ["audyssey", "tool-server", "--tool", "separation", "--tool", "filtering", "--impl", "real", "--host", "0.0.0.0", "--port", "8700"]
