# v2a-inspect-server

GPU inference API used by the main V2A Inspect application. Keeping this package
separate prevents client/UI installs from pulling the large model dependencies.

## Run

From the repository root:

```bash
cp .env.example .env
uv sync --project server
uv run --project server v2a-inspect-server serve
```

`V2A_SERVER_HOST` and `V2A_SERVER_PORT` configure the bind address. Other model
and cache options are documented in the root `.env.example`.

## Endpoints

- `GET /healthz`
- `POST /videos/upload`
- `POST /infer/sam3/track-video`
- `POST /infer/sam3/segment-image`
- `POST /infer/dinov2/embed-images`
- `POST /infer/score`
- `POST /infer/hunyuan/generate-v2a`
- `POST /infer/kokoro/generate-speech`

Hunyuan is used for video-synchronized effects. Kokoro is loaded lazily for
speech and downloads its weights on the first request if they are not cached.
