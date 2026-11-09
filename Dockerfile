FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./

RUN uv sync --locked --no-dev --no-install-project --extra ui --extra observability

COPY src ./src

RUN uv sync --locked --no-dev --no-editable --extra ui --extra observability


FROM node:22-bookworm-slim AS frontend

WORKDIR /app/web

COPY web/package.json web/package-lock.json ./
RUN npm ci

COPY web ./
RUN npm run build


FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/venv/bin:$PATH \
    V2A_INSPECT_UI_HOST=0.0.0.0 \
    V2A_INSPECT_UI_PORT=8501 \
    V2A_INSPECT_UI_WORK_DIR=/data/work \
    V2A_INSPECT_UI_STATIC_DIR=/app/web/dist \
    V2A_INSPECT_UI_SERVER_URL= \
    V2A_INSPECT_UI_SCENE_THRESHOLD=27.0 \
    V2A_INSPECT_UI_MAX_KEYFRAMES_PER_SCENE=20 \
    V2A_INSPECT_CLIENT_SERVER_HOST=localhost \
    V2A_INSPECT_CLIENT_SERVER_PORT=8080 \
    V2A_INSPECT_CLIENT_TIMEOUT=300 \
    V2A_INSPECT_LLM_SMALL_MODEL=gemini-3.1-flash-lite \
    V2A_INSPECT_LLM_MEDIUM_MODEL=gemini-3.5-flash \
    V2A_INSPECT_LLM_LARGE_MODEL=gemini-3.5-flash \
    V2A_INSPECT_LLM_THINKING_LEVEL=low \
    V2A_INSPECT_LLM_SMALL_THINKING_LEVEL=medium \
    V2A_INSPECT_LLM_MEDIUM_THINKING_LEVEL=low \
    V2A_INSPECT_LLM_LARGE_THINKING_LEVEL=low \
    V2A_INSPECT_LLM_THINKING_BUDGET= \
    V2A_INSPECT_LLM_SMALL_THINKING_BUDGET= \
    V2A_INSPECT_LLM_MEDIUM_THINKING_BUDGET= \
    V2A_INSPECT_LLM_LARGE_THINKING_BUDGET= \
    V2A_INSPECT_LLM_TEMPERATURE=1 \
    V2A_INSPECT_LLM_TIMEOUT_SECONDS=600 \
    V2A_INSPECT_LLM_MAX_RETRIES=5 \
    V2A_INSPECT_AGENT_SOUND_TIMELINE_RECURSION_LIMIT=500 \
    V2A_INSPECT_AGENT_SOUND_TIMELINE_MAX_WORKERS=3 \
    V2A_INSPECT_AGENT_SOUND_TIMELINE_SEGMENT_SECONDS=30 \
    V2A_INSPECT_VIDEO_ENCODE_USE_NVENC=true \
    V2A_INSPECT_LANGFUSE_BASE_URL=https://langfuse.riverfog7.com \
    V2A_INSPECT_LANGFUSE_ENVIRONMENT=prod

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash appuser && \
    mkdir -p /data && \
    chown -R appuser:appuser /data

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --from=frontend /app/web/dist /app/web/dist

USER appuser

EXPOSE 8501
VOLUME ["/data"]

CMD ["/opt/venv/bin/v2a", "ui"]
