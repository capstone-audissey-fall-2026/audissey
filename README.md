# Audyssey

짧은 영상(10초 안팎)의 오디오를 소리 나는 대상별 트랙으로 나눕니다.
트랙을 들어 보고 켜고 끄거나 음량·피치를 바꾼 뒤 다시 합쳐, 편집된 영상과 트랙별 오디오(stems)를 내보냅니다.

**현재 단계: 뼈대(skeleton).** 실제 모델로 처음부터 끝까지 도는 것까지 확인했고, 품질은 아직 보지 않았습니다.
담당자별로 모듈을 교체해 나갑니다.

## 바로 해 보기 (GPU 없이)

```bash
uv sync --extra ui
uv run audyssey serve
# 브라우저에서 http://127.0.0.1:8600 을 열고 samples/sample.mp4 업로드
```

모든 모델 자리에 가짜(stub) 구현이 들어가 있어 GPU, API 키, 모델 다운로드가 필요 없습니다.
필요한 것은 Python 3.11 이상, [uv](https://docs.astral.sh/uv/), ffmpeg입니다.

## 목차

1. [구조](#1-구조)
2. [실행 방법](#2-실행-방법)
3. [작업 규칙과 CI/CD](#3-작업-규칙과-cicd)
4. [모듈별 담당](#4-모듈별-담당)
5. [모듈별 남은 과제](#5-모듈별-남은-과제)
6. [실측 기록](#6-실측-기록)

설계 원문: [docs/superpowers/specs/2026-11-09-audyssey-skeleton-design.md](docs/superpowers/specs/2026-11-09-audyssey-skeleton-design.md)

---

## 1. 구조

### 1.1 처리 흐름

```mermaid
flowchart LR
  UP["영상 업로드"] --> PRE["전처리<br/>48kHz 오디오 · 모노 · 파형"]
  PRE --> PLAN["플래닝<br/>plan"]
  PLAN --> CHECK{"플래닝 검증<br/>형식 · validate_plan<br/>(시각 힌트 켜면 SAM3 locate)"}
  CHECK -- "문제 있음, 한도 전" --> PLAN
  CHECK -- "통과 또는 한도 도달" --> SRC
  subgraph SRC["소스마다"]
    MASK["그라운딩 마스크<br/>시각 힌트를 켰을 때만"] --> SEP["분리<br/>separate"]
    SEP --> JUDGE{"필터링 검증<br/>judge"}
    JUDGE -- "불합격, 한도 전" --> SEP
  end
  SRC --> RES["나머지 계산<br/>residual"]
  RES --> TL["타임라인"]
  TL -- "keep · drop · 음량 · 피치" --> TL
  TL -- "reseparate · add_source<br/>그 소스만" --> SRC
  TL -- "render" --> OUT["편집된 MP4<br/>mix WAV · stems"]
```

- 단계 순서는 코드에 고정되어 있습니다(LangGraph). LLM은 다음 단계를 고르지 않습니다.
- 업로드하면 타임라인까지 자동으로 진행합니다. 그다음 편집은 사용자가 명령을 보내서 합니다.
- 플래닝과 분리 뒤에는 검증이 있고, 실패하면 정해진 횟수만큼 다시 합니다([1.4](#14-검증과-재시도)).

### 1.2 서버 구성

```mermaid
flowchart LR
  UI["브라우저 / UI"] -- "HTTP :8600<br/>API · SSE" --> MAIN
  subgraph MAIN["main 서버 · audyssey-app · GPU 없음"]
    GRAPH["LangGraph 흐름<br/>명령 처리"] --- STORE[("work/jobs<br/>job.json")]
  end
  MAIN -- "OpenAI 호환 API" --> LLM["플래너 · 검증기 LLM<br/>Gemini 또는 vLLM"]
  MAIN -- "HTTP + Bearer 토큰 :8700" --> TOOLS
  subgraph TOOLS["도구 서버 · audyssey-tools · GPU"]
    T1["분리<br/>SAM Audio"]
    T2["필터링<br/>SAM Audio Judge"]
    T3["그라운딩<br/>SAM3.1"]
  end
  TOOLS -. "읽기 전용 · 오프라인" .-> HF[("HF 캐시<br/>가중치")]
  MAIN -. "LANGSMITH_*" .-> LS["LangSmith"]
```

| 구성 | 하는 일 | 실행 환경 |
|---|---|---|
| main 서버 | 흐름 실행, 명령 처리, 작업 저장, 개발용 화면. GPU를 쓰지 않음 | `audyssey-app` 이미지 (Python 3.13) |
| 도구 서버 | SAM Audio(분리), Judge(평가), SAM3(위치 찾기)를 HTTP로 제공 | `audyssey-tools` 이미지 (CUDA 12.8, Python 3.12) |
| 플래너 LLM | 소스 계획과 계획 검증. 주소만 바꾸면 Gemini와 vLLM을 오갈 수 있음 | Gemini API 또는 vLLM 서버 |

도구마다 실행 방식을 고릅니다.

- `stub`: 가짜 결과. GPU 없이 흐름을 확인할 때
- `real`: 같은 프로세스에서 모델 실행
- `remote`: 다른 서버에 띄운 도구 서버를 HTTP로 호출

배포 방식은 두 가지입니다. Compose는 도구마다 컨테이너를 띄우고, RunPod는 Pod 1개에서 도구 3개를 함께 띄웁니다.

```mermaid
flowchart TB
  subgraph B["B. GPU 서버 한 대: docker compose up"]
    B1["main"] --> B2["separation"]
    B1 --> B3["judge"]
    B1 --> B4["grounding"]
    B1 --> B5["vllm"]
  end
  subgraph C["C. 노트북 + RunPod"]
    C1["노트북: docker compose up main"] -- "HTTPS :8700" --> C2["RunPod Pod 1개<br/>tools 이미지 기본 CMD<br/>분리 · Judge · 그라운딩"]
    C1 -- "OpenAI 호환 API" --> C3["Gemini API"]
  end
```

### 1.3 용어

| 용어 | 뜻 |
|---|---|
| 소스 | 소리 나는 대상 1개. 같은 대상이 소리를 3번 내면 소스 1개에 구간 3개입니다. 종류가 같아도 대상이 다르면 다른 소스입니다 |
| 분리 조건 | 텍스트 설명(필수), 구간(선택), 시각 힌트(선택: 시점, 0~1 박스, 설명) |
| 시각 힌트 | 플래너가 화면에 보이는 소스마다 만듭니다. **기본값은 사용 안 함**(`AUDYSSEY_VISUAL_PROMPT=false`). 켜면 SAM3가 대상을 찾아 나머지를 검게 칠한 영상을 만들고, 분리할 때 함께 넣습니다 |
| 분리 트랙 | 소스마다 1개. 원본과 길이가 같은 48kHz 모노 WAV |
| 나머지 (residual) | 원본에서 모든 소스 트랙을 뺀 소리. 편집 없이 합치면 원본과 같습니다. `source_id="residual"`로 켜고 끄거나 음량을 바꿀 수 있습니다 |

### 1.4 검증과 재시도

| | 플래닝 검증 | 필터링 검증 |
|---|---|---|
| 검사 | 형식 검사 → `validate_plan` → (시각 힌트를 켰을 때) 힌트마다 SAM3 `locate` | `judge`가 분리 트랙을 평가 |
| 실패하면 | 문제 목록(`PlanIssue`)을 `plan(feedback=…)`에 넘겨 계획 전체를 다시 세움 | `suggestion`이 있으면 그 조건으로, 없으면 `attempt`를 1 올려 그 소스만 다시 분리 |
| 횟수 한도 | `AUDYSSEY_MAX_PLAN_ROUNDS=2` | `AUDYSSEY_MAX_SEPARATION_RETRIES=2` (최대 3번 시도) |
| 한도에 닿으면 | 마지막 계획으로 진행하고 경고 표시 | 마지막 트랙을 쓰고 `filter.passed=False` 경고 표시 |

### 1.5 계약, 함수, 명령, API

**공통 데이터 형식**은 [`src/audyssey/contracts.py`](src/audyssey/contracts.py)에 Pydantic 모델로 있습니다.
주요 모델은 `SourcePlan`, `VisualHint`, `PlanIssue`, `FilterResult`, `GroundResult`, `Job`, `Command`입니다.
오디오는 파일 경로로 주고받고, 시간은 초 단위 float입니다.

**도구 함수**는 `stub`, `real`, `remote` 모두 시그니처가 같습니다.

```python
plan(video_path, audio_path, duration, feedback: list[PlanIssue]) -> list[SourcePlan]
validate_plan(video_path, audio_path, plans: list[SourcePlan]) -> list[PlanIssue]  # 빈 목록 = 통과
separate(mono_path, plan: SourcePlan, masked_video_path | None, out_path, attempt: int) -> out_path  # None = 텍스트만
judge(mono_path, track_path, plan: SourcePlan) -> FilterResult
locate(video: Path, hint: VisualHint) -> GroundResult
mask_video(video: Path, hint: VisualHint, out: Path) -> Path
render(job: Job, job_dir, include_residual_stem: bool) -> RenderOutput
```

**명령.** 작업 상태는 명령으로만 바꿉니다(`graph.handle_command`).
버튼, 나중에 붙일 자연어 입력, 에이전트가 모두 같은 명령을 쓰게 하려는 규칙입니다.

```
keep / drop(source_id)            set_gain(source_id, 0.0~2.0)     set_pitch(source_id, -12~+12)
reseparate(source_id, description?, spans?)   add_source(plan)     render(include_residual_stem)
```

**main 서버 API.** UI는 아래 API만 쓰면 됩니다.

```
POST /api/jobs                    multipart(video) → Job (업로드 후 자동 진행)
GET  /api/jobs                    → Job 요약 목록
GET  /api/jobs/{id}               → Job
GET  /api/jobs/{id}/events        SSE: 상태가 바뀔 때마다 Job 전송
POST /api/jobs/{id}/commands      Command → Job | 409(실행 중) | 422(형식 오류)
GET  /api/jobs/{id}/files/{path}  작업 폴더의 오디오·영상·파형 파일
```

**도구 서버 API.** `/health`를 빼면 모두 `Authorization: Bearer $AUDYSSEY_TOOL_TOKEN`이 필요합니다.

```
GET  /health                                    인증 없음
POST /separate  multipart mono, plan, attempt (+ video)     → WAV   video는 시각 힌트를 켰을 때만
POST /judge     multipart mono, track, plan                  → FilterResult
POST /locate    multipart video, hint                        → GroundResult
POST /mask      multipart video, hint                        → MP4
```

### 1.6 코드 위치

```
src/audyssey/
  contracts.py   팀 공통 계약 (변경은 PR로 합의)
  settings.py    AUDYSSEY_* 환경변수 (pydantic-settings)
  graph.py       LangGraph 흐름, 검증 루프, 명령 처리
  store.py       Job ↔ work/jobs/<id>/job.json
  api.py         HTTP API + 개발용 화면
  toolserver.py  도구 서버
  cli.py         audyssey serve | audyssey tool-server
  media.py       ffmpeg 추출·프레임·WAV 입출력
  devui/         빌드 없는 개발용 화면 (최종 UI 아님)
  tools/
    planning/    stub.py real.py llm.py      plan, validate_plan
    separation/  stub.py real.py remote.py   separate (SAM Audio)
    filtering/   stub.py real.py remote.py   judge (SAM Audio Judge)
    grounding/   stub.py real.py remote.py   locate, mask_video (SAM3)
    editing/     render.py                   residual, pitch, render
gpu/             GPU 도구 전용 uv 프로젝트 (Python 3.12, numpy 1.26, torch cu128)
docker/          app.Dockerfile, tools.Dockerfile
compose.yaml     main + separation + judge + grounding + vllm
tests/           unittest (GPU 코드는 가짜 torch/sam_audio로 검사)
```

---

## 2. 실행 방법

| 상황 | 필요한 것 | 안내 |
|---|---|---|
| A. GPU 없이 흐름만 확인 | Python 3.11+, uv, ffmpeg | [2.1](#21-a-gpu-없이-stub) |
| B. GPU 서버 한 대에서 전부 실행 | Docker, NVIDIA Container Toolkit, 가중치 캐시 | [2.2](#22-b-gpu-서버-한-대-docker-compose) |
| C. 노트북 + RunPod GPU | 노트북 Docker, RunPod Pod 1개 | [2.3](#23-c-노트북--runpod) |
| D. 내 모듈만 GPU로 개발 | GPU, `gpu/` 환경 | [2.4](#24-d-내-모듈만-gpu로-개발) |
| 플래너를 Gemini나 vLLM으로 | API 키 또는 vLLM 주소 | [2.5](#25-플래너-연결-gemini-vllm) |
| LangSmith로 실행 기록 보기 | LangSmith 키 | [2.6](#26-langsmith) |
| 모델 가중치 받기 | Hugging Face 승인과 토큰 | [2.7](#27-가중치-준비) |

### 2.1 A. GPU 없이 (stub)

```bash
uv sync --extra ui
uv run audyssey serve
# http://127.0.0.1:8600 에서 samples/sample.mp4 업로드
```

- 모든 도구가 stub이고, 두 검증 루프가 한 번씩 돕니다.
- 개발용 화면에서 재생, 켜고 끄기, 음량, 피치, 설명을 고쳐 다시 분리, 소스 추가, 합치기를 해 볼 수 있습니다.
- 편집 없이 합친 WAV는 원본과 수치 오차 범위에서 같습니다.

### 2.2 B. GPU 서버 한 대 (Docker Compose)

```bash
cp .env.example .env      # .env는 커밋하지 않음
# .env에서 AUDYSSEY_TOOL_TOKEN을 긴 임의 값으로, HF_CACHE를 가중치 캐시 경로로 바꿈
docker compose pull       # 직접 빌드하려면 docker compose build
docker compose up
# http://localhost:8600 (원격 서버라면 SSH 터널: ssh -L 8600:localhost:8600 <서버>)
```

main, separation, judge, grounding, vllm 컨테이너 5개가 뜹니다.

**접속과 보안**
- 호스트에는 main의 `127.0.0.1:8600`만 열립니다.
- main에는 인증이 없습니다. `127.0.0.1` 밖으로 열지 말고, 원격에서는 SSH 터널을 씁니다. 외부에 꼭 열어야 하면 인증이 있는 리버스 프록시를 앞에 둡니다.

**GPU 지정**
- 도구별로 `AUDYSSEY_SEPARATION_GPU`, `AUDYSSEY_JUDGE_GPU`, `AUDYSSEY_GROUNDING_GPU`, `AUDYSSEY_VLLM_GPU`에 GPU 번호를 넣습니다. 번호가 같으면 그 GPU를 나눠 씁니다.
- `.env.example`은 GPU 0 한 장을 나눠 쓰는 설정이고, vLLM 메모리 비율은 `AUDYSSEY_VLLM_GPU_UTIL=0.2`입니다(80GB A100에서 확인).
- vLLM에 GPU를 따로 주면 이 값을 `0.85` 정도로 올립니다.

**가중치와 작업 파일**
- 가중치 캐시(`HF_CACHE`)는 컨테이너 안 `/hf`에 읽기 전용으로 붙고, 오프라인(`HF_HUB_OFFLINE=1`)으로 읽습니다.
- 작업 파일은 `jobs` 볼륨에 남습니다.
- 컨테이너는 일반 사용자(uid 10001)로 실행됩니다. 예전 root 이미지로 만든 볼륨은 쓰기 권한이 없으니 `docker volume rm audyssey_jobs` 후 다시 띄웁니다.

**그 밖에**
- 플래너 기본값은 Compose 안의 vLLM(`Qwen/Qwen2.5-VL-3B-Instruct`)입니다. Gemini를 쓰면 vLLM 없이 `docker compose up main separation judge grounding`으로 띄웁니다.
- GPU 없이 main만 띄우려면 `AUDYSSEY_PLANNER=stub`과 도구 stub(또는 원격 주소)을 설정하고 `docker compose up main`을 실행합니다.
- sv3-a100에서는 `HF_CACHE=/home/common/huggingface`를 씁니다. 이 서버의 기본 Docker 빌드 엔진은 설정 문제로 실패하므로 `docker buildx --builder <docker-container 빌더>`로 빌드합니다.

### 2.3 C. 노트북 + RunPod

RunPod Pod 안에서는 Docker Compose를 쓸 수 없습니다.
그래서 Pod에서는 tools 이미지의 기본 명령으로 도구 3개를 한 프로세스에 띄우고, 노트북에서는 main만 띄웁니다.

| RunPod 설정 | 값 |
|---|---|
| Container image | `ghcr.io/capstone-audissey-fall-2026/audyssey-tools:latest` (공개 이미지) |
| GPU | 1장, CUDA 12.8 드라이버. 기본 설정(시각 힌트 끔)에서는 48GB 이상, 시각 힌트를 켜면 80GB 권장 (5장 참고) |
| Expose HTTP ports | `8700` |
| Container start command | 비워 둠 (이미지 기본 명령이 분리·Judge·그라운딩을 함께 실행) |
| Network volume | 가중치를 미리 받아 둔 볼륨을 `/workspace`에 연결. Pod를 반납해도 볼륨은 남으므로 가중치를 다시 받지 않아도 됨 |
| 환경변수 | `HF_HOME=/workspace/huggingface`, `HF_HUB_OFFLINE=1`, `CUDA_VISIBLE_DEVICES=0`, 모델 ID 3개([2.7](#27-가중치-준비)) |
| 비밀값 | `AUDYSSEY_TOOL_TOKEN`은 공유 템플릿에 넣지 말고 RunPod secrets로 넣음 |

템플릿은 Templates → New Template에 저장해 두고, Pod를 만들 때 고릅니다([공식 안내](https://docs.runpod.io/pods/templates/manage-templates)).
노트북 `.env`에는 같은 토큰과 Pod 주소를 넣습니다.

```dotenv
AUDYSSEY_TOOL_TOKEN=<Pod와 같은 토큰>
AUDYSSEY_SEPARATOR_URL=https://YOUR-POD-ID-8700.proxy.runpod.net
AUDYSSEY_FILTER_URL=https://YOUR-POD-ID-8700.proxy.runpod.net
AUDYSSEY_GROUNDING_URL=https://YOUR-POD-ID-8700.proxy.runpod.net
# 플래너는 Gemini 권장 (2.5)
AUDYSSEY_PLANNER_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
AUDYSSEY_PLANNER_MODEL=gemini-2.5-flash
GOOGLE_API_KEY=<비공개>
```

```bash
docker compose up main
# http://localhost:8600
```

### 2.4 D. 내 모듈만 GPU로 개발

SAM Audio가 numpy 1.x를 요구해서 GPU 도구는 별도 uv 프로젝트 `gpu/`에서 돌립니다. 루트 환경과 CI에는 torch가 없습니다.

```bash
uv sync --project gpu --locked
# torchcodec(cu128)가 쓰는 NVIDIA NPP 라이브러리 경로 (Docker 이미지에는 이미 설정됨)
export LD_LIBRARY_PATH="$PWD/gpu/.venv/lib/python3.12/site-packages/nvidia/npp/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME=<가중치 캐시> HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
export AUDYSSEY_TOOL_TOKEN=<임의 토큰>

# 내 도구만 real로 띄우기 (--tool은 separation | filtering | grounding, 반복 가능)
uv run --project gpu --locked audyssey tool-server --tool separation --impl real --host 127.0.0.1 --port 8700

# 다른 터미널: main은 루트 환경에서. 내 도구만 remote, 나머지는 stub
AUDYSSEY_SEPARATOR=remote AUDYSSEY_SEPARATOR_URL=http://127.0.0.1:8700 uv run audyssey serve
```

- 다른 모듈은 stub으로 두면 되므로 GPU에 올리지 않아도 됩니다.
- GPU 없이 HTTP 연결만 확인하려면 루트 환경에서 `--impl stub`으로 띄웁니다.
- `127.0.0.1` 밖 주소로 띄우려면 토큰이 있어야 하고, TLS 프록시나 SSH 터널로 보호합니다.
- `/health`는 `{"tools": [...], "impl": "..."}`를 돌려줍니다.

분리와 Judge만 따로 확인하는 스크립트도 있습니다. 로딩·실행 시간, 최대 GPU 메모리, 판정, 결과 WAV 경로를 JSON으로 출력합니다.

```bash
uv run --project gpu --locked python scripts/sam_audio_smoke.py samples/sample.mp4 --description "a person speaking" --out work/jobs/smoke-text
uv run --project gpu --locked python scripts/sam_audio_smoke.py samples/sample.mp4 --description "a person speaking" --span 0 3 --out work/jobs/smoke-span
```

### 2.5 플래너 연결 (Gemini, vLLM)

플래너 real(`planning/real.py`)은 영상에서 고르게 뽑은 JPEG 프레임(기본 4장)과 피드백을 OpenAI 호환 API로 보냅니다.
오디오는 아직 보내지 않습니다.

| 대상 | 설정 |
|---|---|
| Gemini | `AUDYSSEY_PLANNER=real`, `AUDYSSEY_PLANNER_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/`, `AUDYSSEY_PLANNER_MODEL=gemini-2.5-flash`, 키는 `AUDYSSEY_PLANNER_API_KEY` (Compose에서는 `GOOGLE_API_KEY`를 넣으면 대신 전달) |
| 직접 띄운 vLLM | `AUDYSSEY_PLANNER_BASE_URL=http://<vllm>:8000/v1`, `AUDYSSEY_PLANNER_MODEL=<모델 ID>`. 인증 없는 서버면 키 생략 |
| Compose 안의 vLLM | 기본값. `vllm/vllm-openai:v0.32.0` + `Qwen/Qwen2.5-VL-3B-Instruct` (캐시 필요) |

Python으로 직접 띄울 때(`uv run audyssey serve`)는 `.env`를 읽지 않습니다. 환경변수로 넘기거나 `uv run --env-file .env ...`를 씁니다.

<details>
<summary>플래너만 실제로 한 번 호출해 보기</summary>

`plan`과 `validate_plan`을 한 번씩 호출하고 소스 수, 문제 수, 걸린 시간만 출력합니다. 키와 응답 원문은 출력하지 않습니다.

```bash
cat > /tmp/audyssey-planner-live.py <<'PYTHON'
import json, os, time
from pathlib import Path
from tempfile import TemporaryDirectory
from audyssey.media import extract_audio, probe
from audyssey.tools.planning.real import plan, validate_plan

os.environ["LANGSMITH_TRACING"] = "false"
if os.environ.get("AUDYSSEY_PLANNER_BASE_URL", "").startswith("https://generativelanguage.googleapis.com/"):
    os.environ["AUDYSSEY_PLANNER_API_KEY"] = os.environ.get("AUDYSSEY_PLANNER_API_KEY") or os.environ["GOOGLE_API_KEY"]
video = Path(os.environ.get("AUDYSSEY_SMOKE_VIDEO", "samples/sample.mp4"))
with TemporaryDirectory() as tmp:
    audio = extract_audio(video, Path(tmp) / "audio.wav", False)
    duration, _ = probe(video)
    start = time.monotonic(); plans = plan(video, audio, duration, []); plan_s = time.monotonic() - start
    start = time.monotonic(); issues = validate_plan(video, audio, plans)
    print(json.dumps({"sources": len(plans), "issues": len(issues), "plan_s": plan_s, "validate_s": time.monotonic() - start}))
PYTHON

# Gemini
AUDYSSEY_PLANNER_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/ \
AUDYSSEY_PLANNER_MODEL=gemini-2.5-flash \
uv run --env-file .env python /tmp/audyssey-planner-live.py

# Compose 안의 vLLM
docker compose up -d vllm   # 로그에 API 서버 준비 메시지가 뜰 때까지 기다림
docker compose run --rm -T --entrypoint python \
  -e AUDYSSEY_PLANNER_BASE_URL=http://vllm:8000/v1 -e AUDYSSEY_PLANNER_MODEL=Qwen/Qwen2.5-VL-3B-Instruct \
  -e AUDYSSEY_PLANNER_API_KEY=unused -e AUDYSSEY_SMOKE_VIDEO=/samples/sample.mp4 \
  -v "$PWD/samples:/samples:ro" main - < /tmp/audyssey-planner-live.py
```

</details>

### 2.6 LangSmith

`.env`(Compose) 또는 환경변수에 아래 세 값을 넣으면 그래프 실행, 도구 호출, 명령, LLM 토큰 사용량이 기록됩니다.

```dotenv
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=<개인 키>
LANGSMITH_PROJECT=audyssey
```

- [smith.langchain.com](https://smith.langchain.com) → Tracing에서 `audyssey` 프로젝트를 엽니다. 프로젝트는 첫 기록 때 자동으로 생깁니다.
- 기본은 꺼져 있습니다. 끌 때는 `LANGSMITH_TRACING=false`로 둡니다.
- 오디오·영상·프레임 데이터와 키·토큰은 올리지 않습니다. 파일 경로와 메타데이터만 남깁니다.
- 플래너 LLM 기록에는 모델명과 토큰 사용량만 남습니다. 이미지가 든 프롬프트와 응답 원문은 남기지 않습니다.

### 2.7 가중치 준비

- 가중치는 레포와 이미지에 들어 있지 않습니다.
- 쓸 모델은 환경변수(`.env` 또는 RunPod 템플릿)의 모델 ID로 정합니다.
- 서버는 실행 중에 아무것도 내려받지 않고 `HF_HOME` 캐시만 오프라인으로 읽습니다. 그래서 캐시를 미리 받아 둬야 합니다.

**공식 모델을 씁니다.** 순서는 다음과 같습니다.

1. Hugging Face에서 아래 공식 모델 페이지마다 접근 승인(SAM License)을 요청합니다. 승인에 시간이 걸릴 수 있습니다.
2. Hugging Face 설정 → Access Tokens에서 읽기 토큰을 만듭니다. 토큰은 커밋하거나 공유하지 않습니다.
3. 그 토큰으로 로그인해 캐시에 받습니다. 토큰은 이때만 쓰고, 서버 설정에는 넣지 않습니다.

```bash
uvx --from 'huggingface_hub[cli]<1.0' huggingface-cli login
export HF_HOME=<가중치 캐시>   # RunPod: /workspace/huggingface (네트워크 볼륨)
H="uvx --from huggingface_hub[cli]<1.0 huggingface-cli download"
$H facebook/sam-audio-large-tv
$H t5-base                                                   # SAM Audio 텍스트 인코더
$H facebook/sam-audio-judge --revision sam_audio
$H facebook/sam-audio-judge --include "*.json"               # Judge 토크나이저는 main에서 읽음
$H facebook/sam3.1 config.json sam3.1_multiplex.pt
# 분리 후보를 2개 이상 쓸 때만 (AUDYSSEY_SAM_AUDIO_RERANKING_CANDIDATES>1)
$H lukewys/laion_clap 630k-best.pt
$H roberta-base; $H bert-base-uncased; $H facebook/bart-base
# Compose 안의 vLLM 플래너를 쓸 때만
$H Qwen/Qwen2.5-VL-3B-Instruct
```

| 모델 | 설정 | 캐시에 필요한 것 |
|---|---|---|
| 분리 | `AUDYSSEY_SAM_AUDIO_MODEL` | `facebook/sam-audio-large-tv`, `t5-base` |
| 필터링 | `AUDYSSEY_SAM_AUDIO_JUDGE_MODEL` | `facebook/sam-audio-judge`의 `sam_audio` revision 전체, `main` revision의 JSON(설정·토크나이저, 약 2MB) |
| 그라운딩 | `AUDYSSEY_SAM3_MODEL` | `facebook/sam3.1`의 `config.json`, `sam3.1_multiplex.pt` |
| 후보 고르기 (선택) | 후보 2개 이상 | `lukewys/laion_clap`의 `630k-best.pt`, `roberta-base`, `bert-base-uncased`, `facebook/bart-base` |
| 플래너 (선택) | `AUDYSSEY_VLLM_MODEL` | `Qwen/Qwen2.5-VL-3B-Instruct` |

- 승인을 기다리는 동안에는 비공식 미러 `mrfakename/sam-audio-large-tv`, `mrfakename/sam-audio-judge`를 모델 ID로 쓸 수 있습니다. 미러도 사용 권한을 확인해야 하고, 미러 Judge는 `sam_audio` 브랜치의 `checkpoint.pt`가 캐시에 있어야 합니다.
- sv3-a100 공용 캐시 `/home/common/huggingface`에는 공식 모델과 미러가 모두 있습니다. 가중치를 복사하지 말고 이 경로를 그대로 연결합니다.

### 2.8 설정 (환경변수)

설정은 `AUDYSSEY_*` 환경변수로만 읽습니다(`src/audyssey/settings.py`).

**도구 선택과 연결**

| 환경변수 | 기본값 | 설명 |
|---|---|---|
| `AUDYSSEY_PLANNER` | `stub` | `stub` 또는 `real` (Compose 기본 `real`) |
| `AUDYSSEY_SEPARATOR` / `AUDYSSEY_FILTER` / `AUDYSSEY_GROUNDER` | `stub` | `stub`, `real`, `remote` (Compose 기본 `remote`) |
| `AUDYSSEY_SEPARATOR_URL` / `AUDYSSEY_FILTER_URL` / `AUDYSSEY_GROUNDING_URL` | 빈 값 | remote 도구 서버 주소. remote인데 비어 있으면 시작할 때 오류 |
| `AUDYSSEY_TOOL_TOKEN` | 빈 값 | 도구 서버 Bearer 토큰. main과 도구 서버에 같은 값 |
| `AUDYSSEY_REMOTE_TIMEOUT_S` | `600` | 원격 도구 호출 제한 시간(초) |
| `AUDYSSEY_VISUAL_PROMPT` | `false` | 시각 힌트 사용. 켜면 계획 검증에서 SAM3 `locate`, 분리 전 `mask_video`를 실행하고 마스크 영상을 분리에 넣음 |

**플래너**

| 환경변수 | 기본값 | 설명 |
|---|---|---|
| `AUDYSSEY_PLANNER_BASE_URL` / `AUDYSSEY_PLANNER_MODEL` / `AUDYSSEY_PLANNER_API_KEY` | 빈 값 | OpenAI 호환 플래너 주소, 모델, 키 |
| `AUDYSSEY_PLANNER_FRAMES` | `4` | 플래너와 검증기에 보내는 프레임 수 (1~16) |

**흐름과 작업**

| 환경변수 | 기본값 | 설명 |
|---|---|---|
| `AUDYSSEY_MAX_PLAN_ROUNDS` | `2` | 플래닝 최대 횟수 |
| `AUDYSSEY_MAX_SEPARATION_RETRIES` | `2` | 소스마다 다시 분리하는 횟수 (총 시도 = 1 + 이 값) |
| `AUDYSSEY_JUDGE_PASS_THRESHOLD` | `0.5` | Judge overall 합격 기준 (아직 보정 전) |
| `AUDYSSEY_MAX_DURATION_S` | `30` | 업로드 영상 길이 상한(초) |
| `AUDYSSEY_MAX_UPLOAD_MB` | `200` | 업로드 파일 크기 상한. main과 도구 서버 모두 적용, 넘으면 413 |
| `AUDYSSEY_WORK_DIR` | `work/jobs` | 작업 폴더 |

**모델**

| 환경변수 | 기본값 | 설명 |
|---|---|---|
| `AUDYSSEY_SAM_AUDIO_MODEL` | `facebook/sam-audio-large-tv` | 분리 모델 |
| `AUDYSSEY_SAM_AUDIO_JUDGE_MODEL` | `facebook/sam-audio-judge` | Judge 모델. 필터링과 후보 고르기에 씀 |
| `AUDYSSEY_SAM_AUDIO_DEVICE` | `cuda` | 분리와 Judge를 돌릴 장치 |
| `AUDYSSEY_SAM_AUDIO_RERANKING_CANDIDATES` | `1` | 분리 후보 수. 2 이상이면 그 수만큼 후보를 만들고 CLAP×5 + Judge×1 점수로 하나를 고름 |
| `AUDYSSEY_SAM_AUDIO_CLAP_CHECKPOINT` | 빈 값 | 후보 고르기용 CLAP 파일. 비우면 캐시의 `lukewys/laion_clap` 630k-best |
| `AUDYSSEY_SAM_AUDIO_IMAGEBIND_CHECKPOINT` | 빈 값 | 시각 힌트 소스의 후보 고르기용 ImageBind 파일. 비우면 쓰지 않음. Compose에서는 파일을 separation 컨테이너에 따로 마운트해야 함 |
| `AUDYSSEY_SAM_AUDIO_PREDICT_SPANS` | `false` | 구간이 없을 때 모델이 구간을 예측 (span predictor를 GPU에 추가로 올림) |
| `AUDYSSEY_SAM3_MODEL` / `AUDYSSEY_SAM3_DEVICE` | `facebook/sam3.1` / `cuda` | 그라운딩 모델과 장치 (CUDA 전용) |
| `AUDYSSEY_SAM3_TRACK_FPS` | `2` | 마스크 추적 fps. `0`이면 원본 fps (24GB GPU에서는 메모리 부족 위험) |

Compose에서만 쓰는 값(`.env`): `HF_CACHE`, `AUDYSSEY_{SEPARATION,JUDGE,GROUNDING,VLLM}_GPU`, `AUDYSSEY_VLLM_MODEL`,
`AUDYSSEY_VLLM_GPU_UTIL`, `GOOGLE_API_KEY`, `LANGSMITH_*`.

---

## 3. 작업 규칙과 CI/CD

### 3.1 코드 규칙

**모듈과 계약**
- 담당 모듈 폴더만 고칩니다. `src/audyssey/tools/<모듈>/real.py`의 함수를 같은 시그니처([1.5](#15-계약-함수-명령-api))로 구현합니다.
- 도구를 바꿀 때 `graph.py`, `api.py`, UI를 고쳐야 한다면 설계 문제일 수 있으니 팀에 먼저 공유합니다.
- `contracts.py`는 모든 모듈이 함께 쓰는 파일입니다. 바꿀 때는 PR에서 팀 합의를 받고 머지합니다.
- 작업 상태는 명령으로만 바꿉니다(`handle_command`). 새 편집 기능은 새 명령으로 추가합니다.

**설정**
- 설정은 환경변수로 받습니다. 새 설정을 만들면 `settings.py` 필드, 이 README 2.8 표의 한 줄, 테스트를 함께 추가합니다.
- 서버 경로, 키, 모델 경로를 코드에 직접 쓰지 않습니다.

**테스트와 검사**
- 테스트를 먼저 씁니다. 기능 테스트 → 구현 → 통과 순서입니다.
- CI에는 GPU가 없으므로 GPU 코드는 가짜 모듈로 검사합니다. 예: `tests/sam_audio_fakes.py`.
- 검사를 우회하지 않습니다. `# noqa`, `ty: ignore`, 테스트 skip, assert 완화로 통과시키지 말고 코드를 고칩니다.

**의존성**
- 루트(`pyproject.toml`)에는 torch를 넣지 않습니다.
- GPU 의존성은 `gpu/pyproject.toml`에 넣고 `uv lock --project gpu`로 잠급니다. 다른 패키지가 정한 버전 제약을 덮어쓰지 않습니다.

**비밀값과 공용 서버**
- `.env`는 git에서 제외되어 있습니다. 커밋하거나 로그에 출력하지 않습니다.
- 가중치는 HF 캐시에만 둡니다.
- sv3-a100 같은 공용 서버에서는 다른 사람의 GPU, Docker 이미지, 캐시를 건드리지 않습니다. 지울 때는 내가 만든 이미지와 볼륨만 지우고, 쓸 GPU 번호는 미리 알립니다.

### 3.2 브랜치와 커밋

- 브랜치: `<type>/<짧은-설명>`. 예: `feat/sam3-grounding`, `fix/tools-image-ffmpeg`, `docs/e2e-results`
- 커밋: `<type>(<scope>): <소문자 명령형 요약>` (Conventional Commits). 예:
  - `feat(separation): support reranking with configurable ranker checkpoints`
  - `fix(grounding): track masks at a reduced frame rate`
  - `build(docker): run app, tools and vllm containers as non-root`
  - `docs(readme): add sam3 grounding guide`
- type: `feat`, `fix`, `docs`, `build`(Docker, 의존성), `test`, `refactor`, `chore`
- scope: `planning`, `separation`, `filtering`, `grounding`, `graph`, `api`, `contracts`, `media`, `toolserver`, `docker`, `gpu`, `readme`, `spec` 등
- 커밋 하나에는 변경 하나만 담습니다. 테스트와 구현은 같은 커밋에 넣어도 됩니다.

### 3.3 commit → push → PR → merge → 배포

| 단계 | 자동으로 일어나는 일 | 내가 할 일 |
|---|---|---|
| 1. 브랜치 | 없음 | 최신 `main`에서 브랜치를 만듦 |
| 2. 커밋 | 없음 (로컬 훅 없음) | 커밋 전에 로컬 검사([3.4](#34-로컬-검사))를 통과시킴 |
| 3. push | 없음. push만으로는 CI가 돌지 않음 | `git push -u origin <브랜치>` |
| 4. PR (대상 `main`) | `ci-lint`가 job 2개를 실행. `lint-and-format-check`(필수): ruff lint·format, ty, unittest. `numpy-1x`: Python 3.12 + numpy 1.26 unittest | 설명에 요약과 확인 방법을 적음. `contracts.py`를 바꿨다면 팀 합의를 받음. 실패하면 고쳐서 다시 push |
| 5. merge | `main` 보호 규칙 적용: 필수 검사 통과 전 머지 불가(관리자 포함), 강제 push와 브랜치 삭제 금지 | 필수 승인 수는 0이지만 리뷰를 받고 머지. 머지 커밋 방식을 쓰고, 머지한 브랜치는 직접 지움 |
| 6. 배포 | `main`에 반영되면 `publish-images`가 `audyssey-app`, `audyssey-tools`를 빌드해 GHCR에 `latest`와 커밋 SHA 태그로 올림 (수동 실행 가능, PR에서는 올리지 않음) | GPU 서버: `docker compose pull && docker compose up -d`. RunPod: Pod를 다시 만들어 새 이미지를 받거나 SHA 태그로 고정 |

이미지는 공개되어 있어 로그인 없이 받을 수 있습니다. tools 이미지는 약 23GB라 빌드와 다운로드에 시간이 걸립니다.

### 3.4 로컬 검사

CI와 같은 검사입니다. ffmpeg/ffprobe가 PATH에 있어야 테스트가 돕니다.

```bash
uv sync --dev --locked --extra ui --extra llm
uv run ruff check src/
uv run ruff format --check src/
uv run ty check --error-on-warning src/
uv run python -m unittest discover tests
# numpy-1x job과 같은 검사
uv run --isolated --no-project --python 3.12 --with-editable '.[ui,llm]' --with 'numpy==1.26.4' python -m unittest discover tests
```

비밀값 없이 Compose 파일만 검사하려면:

```bash
env -i PATH="$PATH" docker compose --env-file .env.example config --no-env-resolution --quiet
```

---

## 4. 모듈별 담당

| 모듈 | 위치 | 담당 | 현재 상태 | 할 일 |
|---|---|---|---|---|
| 오케스트레이터·API | `graph.py`, `api.py`, `store.py` | 오케스트레이터 담당 | 고정 흐름, 검증 루프, 명령, SSE 동작 | 소스별 병렬 처리, 자연어 → 명령 번역 |
| UI | `devui/` (임시) | UI 담당 | 개발용 화면만 있음 | [1.5](#15-계약-함수-명령-api)의 API로 최종 화면 설계·구현 |
| 플래닝 | `tools/planning/real.py` | 플래닝 담당 | 연결 확인용 LLM 기본 구현 | `plan`·`validate_plan` 교체, 모델 선택, 오디오 정보 반영, 시각 힌트 검증 기준 |
| 분리 | `tools/separation/real.py` | 분리 담당 | SAM Audio large-tv, 후보 고르기 지원 | 재시도할 때 조건·후보 수를 바꾸는 전략, 시각 조건 품질 확인 |
| 필터링 | `tools/filtering/real.py` | 필터링 담당 | SAM Audio Judge 예시 | 합격 기준 보정, 실패 이유와 `suggestion` 생성 |
| 그라운딩 | `tools/grounding/real.py` | 미정 | SAM3.1 위치 찾기·마스크 | 위치 확인 강화, 추적 실패 대비 |
| 편집·합치기 | `tools/editing/render.py` | 미정 | 음량·피치·합치기 (DSP) | 편집 품질, 편집 명령 추가 |
| 인프라·배포 | `docker/`, `compose.yaml`, `.github/workflows/` | 인프라 담당 | 이미지 2개, Compose, GHCR 자동 배포 | RunPod 템플릿 검증, 캐시 준비 안내 |

---

## 5. 모듈별 남은 과제

뼈대 단계에서 확인한 한계입니다. 측정 기록은 [6장](#6-실측-기록)과 [설계 문서 §13](docs/superpowers/specs/2026-11-09-audyssey-skeleton-design.md#13-미결정--확인-필요)에 있습니다.

**오케스트레이터·API**
- 소스를 차례로 하나씩 분리합니다(`graph.py`의 `ponytail:` 표시). GPU 도구를 늘리면 병렬로 바꿀 수 있습니다.
- 서버 프로세스는 1개이고 작업 큐, DB, 사용자 인증이 없습니다. SSE는 저장된 작업을 주기적으로 읽어 보냅니다.
- 자연어 편집 입력(명령 번역)은 아직 없습니다.

**UI**
- 개발용 최소 화면만 있습니다. 최종 UI는 정해지지 않았습니다.

**플래닝**
- 프레임 이미지만 보고 소스를 추정합니다. 오디오와 전사를 쓰지 않아서, 실제로 들리는 소리인지는 보장하지 못합니다.
- 기본 vLLM 모델(Qwen2.5-VL-3B)은 품질이 낮습니다. 예: 2라운드 뒤 소스가 1개만 남음.
- 검증기는 의심 가는 소스만 보고합니다. 형식 오류가 두 번 나면 문제 없음으로 보고 진행합니다.

**분리**
- 재시도할 때 `attempt`를 시드로만 씁니다. 필터링이 `suggestion`을 주지 않으므로 사실상 같은 조건으로 다시 분리합니다.
- 후보 고르기는 기본으로 꺼져 있습니다(후보 1개). ImageBind 파일은 Compose에 마운트하지 않았습니다.
- 도구 서버마다 모델 1개, 요청 1개씩 처리합니다. 첫 호출 때 모델을 올리므로 첫 요청이 느립니다.

**필터링**
- 합격 기준 0.5는 보정 전입니다.
  - Judge overall 점수는 1~5 범위로 보입니다.
  - 대사와 효과음은 약 1.0, 앰비언스는 약 3.2로 소스 종류에 따라 크게 다릅니다.
  - 지금 기준으로는 거의 다 통과합니다.
- 실패 이유와 수정 조건(`suggestion`)을 만들지 않습니다.

**그라운딩**
- 시각 힌트는 기본으로 꺼져 있습니다. 텍스트 설명이 있으면 마스크가 분리 결과를 거의 바꾸지 못했습니다([6장](#6-실측-기록)). 같은 성별 두 사람처럼 텍스트로 구분하기 어려운 장면에서만 켤 만합니다.
- 힌트 박스를 함께 주기 때문에 SAM3는 박스 안에서 거의 항상 무언가를 찾습니다. 없는 대상도 찾았다고 나와서 검증 효과가 약합니다.
- 마스크는 대상만 남기고 나머지를 검게 칠합니다. SAM Audio의 `mask_videos`는 반대로 대상을 지우지만, 공식 벤치마크 화자 쌍에서 마스크가 효과를 낸 경우는 모두 우리 방식이었습니다. 코드를 `mask_videos` 방식으로 바꾸지 마세요.
- 아주 작은 대상을 추적하다 놓치면 마스크 전체가 실패합니다(`No points are provided`). 그 소스만 실패하고 작업은 계속됩니다.
- 메모리 때문에 2fps로 추적합니다. CUDA에서만 돕니다.

**편집·합치기**
- 피치 변경은 ffmpeg `asetrate`+`atempo`를 쓰는 단순한 방식입니다.
- 영상에 들어가는 오디오는 AAC라 손실 압축됩니다.
- 구간별 편집이나 페이드 기능은 없습니다.

**인프라·배포**
- RunPod 템플릿을 실제 Pod에서 띄워 보지 않았습니다.
  - 필요한 GPU 메모리는 sv3 Compose 측정값에서 vLLM 몫(약 16GB)을 뺀 추정입니다. 기본 설정에서 약 34GB, 시각 힌트를 켜면 약 53GB입니다.
  - 2.7의 다운로드 목록은 sv3에서 그 목록만 담은 캐시로 확인했습니다. 빈 RunPod 볼륨에 처음부터 받아 보지는 않았습니다.
- GPU 이미지는 빌드 중 큰 git 의존성 다운로드가 자주 끊겨 최대 5번 다시 시도합니다.
- CI 필수 검사는 `lint-and-format-check` 하나입니다(`numpy-1x`는 필수 아님). GPU 실측은 CI에 없습니다.

---

## 6. 실측 기록

**Compose 전체 실행** (`samples/sample.mp4`, 1920×1080 60fps, 7.1초)

| 환경 | 구성 | 결과 |
|---|---|---|
| dnn5, RTX A5000 24GB × 4 | 컨테이너 5개, 도구마다 GPU 1장, `sam-audio-small`, vLLM 플래너 | 306초에 ready. 플래닝 루프 2라운드, SAM3 2회, 분리·Judge·render 정상 |
| dnn5 직접 실행, 루프 유도 | Gemini 플래너, Judge 기준 1.5 | 시각 힌트 4개 → SAM3 마스크 → 시각 조건 분리. 재분리 루프 최대 3회 뒤 경고 |
| sv3-a100, A100 80GB × 1 | 컨테이너 5개를 한 장에, 미러 large-tv, 후보 1개 | 468초에 ready, 최대 GPU 62.6GB |
| sv3-a100, A100 80GB × 1 | 컨테이너 5개를 한 장에, 공식 large-tv·Judge·SAM3.1, 후보 4개, vLLM 0.2 | 201초에 ready, Judge 4.95 통과, 최대 GPU 69.1GB |
| sv3-a100, A100 80GB × 1 | GHCR 이미지(`65bd7a3`), 시각 힌트 끔(기본), 후보 4개, vLLM 0.2 | 131초에 ready, SAM3 호출 없음, Judge 5.01 통과, 오류 0, 최대 GPU 49.9GB |

**시각 힌트를 기본으로 끈 뒤** (sv3-a100 A100 80GB × 1, 한 프로세스에서 실제 모델 실행, Gemini 2.5 Flash 플래너, large-tv, 후보 1개.
영상은 SAM Audio 공식 예제 `office.mp4`, 640×360, 15초. 시간은 모델을 올린 뒤 `ready`까지)

| 조건 | 결과 |
|---|---|
| 시각 힌트 끔 | 38.5초. 소스 6개, 루프 반복 없음 |
| 시각 힌트 켬 | 102.9초. `locate` 11회 5.6초, `mask_video` 5회 22.1초. 소스 1개는 SAM3가 추적 중 대상을 놓쳐 실패(`No points are provided`) |
| 시각 힌트 끔 + 루프 강제 (`AUDYSSEY_JUDGE_PASS_THRESHOLD=5`) | 123.0초. 플래닝 2라운드, 소스 4개 × 분리 3회. render 0.4초. 모델 로딩을 포함한 첫 실행은 133.7초 |
| `locate` 1회 (`samples/sample.mp4`, 1080p60) | 영상 전체를 볼 때 16.4초 → 힌트 프레임 한 장만 볼 때 1.9초, 같은 박스 |

**시각 힌트 효과**
- `office.mp4`(말하는 사람 1명): 텍스트 설명이 있으면 마스크를 어떻게 넣어도 텍스트만 넣었을 때와 결과 차이가 2% 안팎이었습니다. 텍스트 없이 마스크만 넣으면 무음이 나왔습니다.
- `facebook/sam-audio-bench`의 같은 성별 화자 쌍 5장면(공식 마스크): 마스크가 지정한 화자를 뽑은 장면은 2개였고, 둘 다 대상만 남기는 방식이었습니다. 나머지 3장면에서는 마스크가 결과를 바꾸지 못했습니다.

단계별 시간, 메모리, 해결한 문제는 [설계 문서 §13](docs/superpowers/specs/2026-11-09-audyssey-skeleton-design.md#13-미결정--확인-필요)에 있습니다.
