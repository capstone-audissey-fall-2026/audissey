# Audyssey skeleton

GPU와 API 키 없이 영상 업로드 → 플래닝·필터링 검증 루프 → 소스 트랙과 나머지 → 편집 → 영상과 stems 내보내기를 실행합니다. 기본 도구는 모두 stub이며 분리 품질을 평가하는 구현은 아닙니다.

요구사항: Python 3.11+, [uv](https://docs.astral.sh/uv/), `ffmpeg`/`ffprobe` (PATH에 설치). 개발용 화면에는 Node가 필요 없습니다.

```bash
uv sync --extra ui
uv run audyssey serve
# http://127.0.0.1:8600 에서 samples/sample.mp4 업로드
```

개발용 화면에서 트랙을 재생하고 keep/drop, 음량, 피치를 조절합니다. 설명을 바꾸어 해당 소스만 다시 분리하거나 소스를 추가할 수 있습니다. 합치기는 편집된 MP4, mix WAV, keep된 소스의 stems를 만듭니다. 나머지 stem은 선택 사항입니다. 편집 없이 합친 WAV는 원본 추출 오디오와 수치 오차 범위에서 같습니다. AAC 영상 오디오는 손실 압축됩니다.

```bash
uv sync --dev --extra ui --extra observability
uv run python -m unittest discover tests
uv run ruff check src/ server/src/
uv run ruff format --check src/ server/src/
uv run ty check --error-on-warning src/
```

| 환경변수 | 기본값 | 의미 |
|---|---|---|
| `AUDYSSEY_PLANNER` | `stub` | `stub` 또는 `real` |
| `AUDYSSEY_SEPARATOR` | `stub` | `stub` 또는 `real` |
| `AUDYSSEY_FILTER` | `stub` | `stub` 또는 `real` |
| `AUDYSSEY_MAX_PLAN_ROUNDS` | `2` | 플래닝 최대 회차 (1 이상) |
| `AUDYSSEY_MAX_SEPARATION_RETRIES` | `2` | 재분리 횟수, 총 시도 최대 3회 |
| `AUDYSSEY_MAX_DURATION_S` | `30` | 업로드 영상 길이 상한 |
| `AUDYSSEY_WORK_DIR` | `work/jobs` | 작업 폴더 위치 |
| `AUDYSSEY_SAM_AUDIO_MODEL` | `facebook/sam-audio-large-tv` | 향후 분리 모델 ID |
| `AUDYSSEY_SAM_AUDIO_JUDGE_MODEL` | `facebook/sam-audio-judge` | 향후 필터링 모델 ID |

설정은 환경변수로만 읽으며 `.env`는 읽지 않습니다. 작업은 `<WORK_DIR>/<id>/job.json`과 상대 경로 파일로 보존됩니다. 서버 재시작 후 `GET /api/jobs`와 `GET /api/jobs/{id}`로 다시 조회할 수 있습니다. 단일 서버 프로세스로 실행합니다. 실행 중 재분리·추가·합치기는 409를 반환하며 keep/drop과 음량·피치 변경은 허용합니다.

팀원은 담당 `src/audyssey/tools/<모듈>/real.py`의 동일 시그니처 함수를 구현하고 `AUDYSSEY_PLANNER=real`, `AUDYSSEY_SEPARATOR=real`, `AUDYSSEY_FILTER=real` 중 해당 값을 설정합니다. 현재 real 함수는 담당자가 표시된 `NotImplementedError` placeholder입니다.

- 도은: `planning/real.py`의 `plan`, `validate_plan`
- SAM Audio 담당: `separation/real.py`의 `separate`
- 윤석: `filtering/real.py`의 `judge`
- UI 담당: 기존 `web/`를 바탕으로 최종 UI를 결정합니다.

공통 계약 `contracts.py` 변경은 PR로 합의합니다. `audyssey`는 `v2a_inspect`를 import하지 않습니다. 모델은 필요할 때 로드하고 가중치는 `HF_HOME` 캐시를 사용합니다. 뼈대 실행에는 모델 다운로드가 없습니다. 도구 교체 시 API와 오케스트레이터를 수정하지 않습니다.

[설계](docs/superpowers/specs/2026-11-09-audyssey-skeleton-design.md) · [구현 계획](docs/superpowers/plans/2026-11-09-audyssey-skeleton.md)

---

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
(cd web && npm ci && npm run build)
```

The UI serves the built frontend from `web/dist`, so the build step is required
before `v2a ui` shows anything. Use Node 20.19+ or 22.12+ (Vite 7).

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

## Sample data

`samples/` holds a 7-second clip and its analyzed `VideoAsset` (scenes, object
tracks, and an agent-generated sound timeline). To edit without running the
analysis, open the UI and use **Import asset** with `samples/sample.mp4` and
`samples/sample.video-asset.json`. Then edit the timeline and run
`Generate Audio` (needs the inference server and a GPU). Keyframe images and the
SAM3 overlay are not included, so those previews stay empty until you run the
analysis yourself.

## Frontend development

Run the UI backend and the Vite dev server separately:

```bash
uv run v2a ui                  # terminal 1, port 8501
cd web && npm ci && npm run dev   # terminal 2, proxies /api and /events to 8501
```

## Tests

```bash
uv run python -m unittest discover tests
uv run --project server python -m unittest discover server/tests
(cd web && npm test)
```

## Prerequisites

- Linux with an NVIDIA GPU for the inference server (SAM3, Hunyuan V2A).
- A Gemini API key (`GEMINI_API_KEY`) for the analysis and timeline agent.
- Hunyuan weights (about 8 GB) and Kokoro download automatically on first use.
- SAM3 is installed from `facebookresearch/sam3`; its checkpoint may need a
  Hugging Face account with access (set `HF_TOKEN` in `.env`).

## Docker Compose

```bash
cp .env.example .env
docker compose up --build
```

This starts both services with the UI available only on
`http://127.0.0.1:8501`. The inference API is reachable only by the UI on the
private Compose network. Set `V2A_DOCKER_UID` and `V2A_DOCKER_GID` in `.env`
to `id -u` and `id -g`; the inference process then runs as that host user.
Set `V2A_GPU_DEVICE` to the physical GPU index assigned to the inference
container. Only that GPU is exposed, and all CUDA models use it as `cuda:0`.
Model files remain in the `v2a_server_data` volume across container restarts.

## Packages

- `src/v2a_inspect/`: main CLI, agent pipeline, editor API, and inference clients.
- `web/`: React timeline editor.
- `server/`: GPU inference API for SAM3, Hunyuan V2A, and Kokoro TTS.
- `tests/` and `server/tests/`: lightweight regression tests.
- `scripts/run.sh`: batch wrapper around `v2a run`.

Runtime media, generated outputs, virtual environments, and downloaded model
weights are intentionally ignored by Git. Copy `.env.example` to `.env`; never
commit real credentials.
