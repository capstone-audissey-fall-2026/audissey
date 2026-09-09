# V2A Inspect

V2A Inspect turns a video into an editable sound timeline, routes each track to
the appropriate audio model, and mixes the generated audio back into the video.

## Entrypoints

- `uv run v2a run VIDEO`: analyze a video and export a `VideoAsset` JSON.
- `uv run v2a ui`: open the upload, timeline-editing, generation, and preview UI.
- `uv run v2a synthesize`: generate audio from an exported asset or timeline.
- `uv run --project server v2a-inspect-server serve`: run the inference API.

The commands are one-shot CLI/UI entrypoints. The sound-timeline inference stage
uses LangGraph internally; it is not an interactive CLI loop.

## Local setup

```bash
cp .env.example .env
uv sync --extra ui
uv sync --project server
```

Add a Gemini API key to `.env`, then start the inference server and UI in
the repository-owned virtual environments with one command:

```bash
./scripts/start-local.sh
```

This runs both processes as the invoking host user, using `.venv` for the UI
and `server/.venv` for GPU inference. Press `Ctrl+C` to stop both. Docker
containers use image-local environments instead of these host environments.

Open `http://127.0.0.1:8501`. The UI flow is:

1. Upload a video and run visual/agent inference.
2. Edit sound tracks and events in the timeline.
3. Generate routed audio.
4. Play event/track audio or preview and download the mixed video.

Speech tracks use Kokoro TTS and require `spoken_text`. SFX and ambience default
to text-to-audio; use V2A only for sounds that require precise synchronization
with visible motion. Kokoro downloads its model on the first speech request.

## Docker Compose

```bash
cp .env.example .env
docker compose up --build
```

This starts both services with the UI available only on
`http://127.0.0.1:8501`. The inference API is reachable only by the UI on the
private Compose network. Set `V2A_DOCKER_UID` and `V2A_DOCKER_GID` in `.env`
to `id -u` and `id -g`; the inference process then runs as that host user.
Model files remain in the `v2a_server_data` volume across container restarts.

## Packages

- `src/v2a_inspect/`: main CLI, agent pipeline, editor API, and inference clients.
- `web/`: React timeline editor.
- `server/`: GPU inference API for SAM3, embeddings, Hunyuan V2A, and Kokoro TTS.
- `tests/` and `server/tests/`: lightweight regression tests.
- `scripts/run.sh`: batch wrapper around `v2a run`.

Runtime media, generated outputs, virtual environments, and downloaded model
weights are intentionally ignored by Git. Copy `.env.example` to `.env`; never
commit real credentials.
