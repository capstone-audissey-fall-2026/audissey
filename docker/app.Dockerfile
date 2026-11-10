FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src/ src/
RUN uv sync --locked --no-dev --extra ui --extra llm
ENV PATH="/app/.venv/bin:$PATH"
# Run as an unprivileged user; jobs are written under /app/work (named volume).
RUN useradd --create-home --uid 10001 app && mkdir -p /app/work/jobs && chown -R app:app /app/work
USER app
EXPOSE 8600
CMD ["audyssey", "serve", "--host", "0.0.0.0", "--port", "8600"]
