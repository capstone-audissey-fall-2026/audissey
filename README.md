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
uv sync --dev --extra ui --extra llm
uv run python -m unittest discover tests
uv run ruff check src/
uv run ruff format --check src/
uv run ty check --error-on-warning src/
```

| 환경변수 | 기본값 | 의미 |
|---|---|---|
| `AUDYSSEY_PLANNER` | `stub` | `stub` 또는 `real` |
| `AUDYSSEY_SEPARATOR` | `stub` | `stub`, `real`, `remote` |
| `AUDYSSEY_FILTER` | `stub` | `stub`, `real`, `remote` |
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
- UI 담당: 최종 UI를 결정합니다. 이전 `web/` React UI는 삭제되었고 git 기록과 `legacy` 원격에 남아 있습니다.

공통 계약 `contracts.py` 변경은 PR로 합의합니다. 이전 `v2a_inspect` 코드는 삭제되었고 git 기록과 `legacy` 원격에 남아 있습니다. 모델은 필요할 때 로드하고 가중치는 `HF_HOME` 캐시를 사용합니다. 뼈대 실행에는 모델 다운로드가 없습니다. 도구 교체 시 API와 오케스트레이터를 수정하지 않습니다.

[설계](docs/superpowers/specs/2026-11-09-audyssey-skeleton-design.md) · [구현 계획](docs/superpowers/plans/2026-11-09-audyssey-skeleton.md)


## GPU 서버에서 도구 띄우기

```bash
uv sync --extra ui
AUDYSSEY_TOOL_TOKEN=... uv run audyssey tool-server --tool separation --impl real --host 0.0.0.0 --port 8700
# 다른 프로세스에서 같은 토큰으로:
AUDYSSEY_TOOL_TOKEN=... uv run audyssey tool-server --tool filtering --impl real --host 0.0.0.0 --port 8701
```

현재 real 구현은 placeholder입니다. GPU·가중치 없이 연결을 확인하려면 `--impl stub`을 씁니다. 모델 가중치는 이 작업에서 다운로드하지 않습니다. 서버당 모델 호출은 한 번에 하나씩 실행합니다. `GET /health`는 인증 없이 도구·구현을 보여줍니다. 외부 바인딩에는 토큰이 필수이며, 실제 토큰은 셸 기록에 직접 쓰지 말고 환경에서 주입합니다. 네트워크 구간은 TLS 프록시나 SSH 터널로 보호합니다.

메인 서버에서는 다음을 설정합니다. 두 도구가 같은 서버에 있을 필요는 없습니다.

```bash
export AUDYSSEY_SEPARATOR=remote
export AUDYSSEY_SEPARATOR_URL=http://gpu:8700
export AUDYSSEY_FILTER=remote
export AUDYSSEY_FILTER_URL=http://gpu:8701
export AUDYSSEY_TOOL_TOKEN=...
export AUDYSSEY_REMOTE_TIMEOUT_S=600
uv run audyssey serve
```

URL이 없으면 시작 시 오류가 납니다. 인증 오류(401), 서버 오류, 타임아웃은 해당 소스만 실패시킵니다. 토큰과 응답 본문은 오류 메시지에 넣지 않습니다.

## 플래너를 Gemini/vLLM에 연결하기

```bash
uv sync --extra ui --extra llm
# vLLM (로컬 서버가 인증을 요구하지 않으면 키 생략 가능)
export AUDYSSEY_PLANNER_BASE_URL=http://localhost:8000/v1
export AUDYSSEY_PLANNER_MODEL=your-served-model
export AUDYSSEY_PLANNER_API_KEY=...
# Gemini OpenAI 호환 API를 쓸 때는 위 값을 다음으로 교체:
# AUDYSSEY_PLANNER_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
# AUDYSSEY_PLANNER_MODEL=<사용 가능한 Gemini 모델 ID>
# AUDYSSEY_PLANNER_API_KEY=<Gemini API 키>
```

`planning/real.py` 담당자는 `chat_client(settings)`를 가져와 `client.chat.completions.create(model=settings.planner_model, ...)`로 호출합니다. `AUDYSSEY_PLANNER=real`은 real 본체를 구현한 뒤 설정합니다. 이번 변경은 클라이언트 진입점만 제공하며 real 본체는 placeholder입니다. 영상은 프레임 이미지로, 오디오는 텍스트로 바꿔 넣어 Gemini와 vLLM에서 같은 코드가 돌게 합니다.

## LangSmith로 보기

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=...
export LANGSMITH_PROJECT=audyssey
uv run audyssey serve
```

키와 tracing 환경변수를 설정하지 않은 기본 상태에서는 아무것도 보내지 않습니다. 호스팅된 LangSmith로 보낼 때는 세 변수를 모두 설정합니다. tracing만 켜고 키를 생략해도 전송 시도는 일어날 수 있으므로 끌 때는 `LANGSMITH_TRACING=false`를 씁니다. 그래프, 도구, 명령과 래핑된 LLM 호출의 토큰 사용량을 볼 수 있습니다. 명령 입력은 `job_id`와 명령만 남기고 Job 전체·설정은 제외합니다. 오디오·영상 바이트와 키·토큰은 올리지 않고 경로와 메타데이터만 기록합니다.
