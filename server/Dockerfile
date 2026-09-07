# syntax=docker/dockerfile:1

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_HTTP_TIMEOUT=600 \
    UV_LINK_MODE=copy \
    GIT_HTTP_VERSION=HTTP/1.1 \
    PATH="/app/server/.venv/bin:$PATH" \
    V2A_SERVER_HOST=0.0.0.0 \
    V2A_SERVER_PORT=8080 \
    V2A_SERVER_UPLOAD_DIR=/data/uploads \
    HF_HOME=/data/model-cache/huggingface \
    TRANSFORMERS_CACHE=/data/model-cache/transformers \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

WORKDIR /app/server


FROM base AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

RUN git config --global http.version HTTP/1.1

COPY server/pyproject.toml server/uv.lock server/README.md ./

RUN --mount=type=cache,target=/root/.cache/uv \
    for attempt in 1 2 3; do \
      uv sync --locked --no-dev --no-install-project && break; \
      if [ "$attempt" = "3" ]; then exit 1; fi; \
      sleep 10; \
    done

COPY server/src ./src

RUN --mount=type=cache,target=/root/.cache/uv \
    for attempt in 1 2 3; do \
      uv sync --locked --no-dev && break; \
      if [ "$attempt" = "3" ]; then exit 1; fi; \
      sleep 10; \
    done


FROM base AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    espeak-ng \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app /app

EXPOSE 8080
VOLUME ["/data"]

CMD ["v2a-inspect-server", "serve"]
