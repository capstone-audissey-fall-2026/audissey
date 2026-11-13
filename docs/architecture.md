# 구조 상세

[README](../README.md)의 처리 흐름과 서버 구성 그림을 보충하는 문서입니다. 설계 원문은
[docs/superpowers/specs/2026-11-09-audyssey-skeleton-design.md](superpowers/specs/2026-11-09-audyssey-skeleton-design.md)에 있습니다.

## 서버 구성

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
  subgraph B["GPU 서버 한 대: docker compose up"]
    B1["main"] --> B2["separation"]
    B1 --> B3["judge"]
    B1 --> B4["grounding"]
    B1 --> B5["vllm"]
  end
  subgraph C["노트북 + RunPod"]
    C1["노트북: docker compose up main"] -- "HTTPS :8700" --> C2["RunPod Pod 1개<br/>tools 이미지 기본 CMD<br/>분리 · Judge · 그라운딩"]
    C1 -- "OpenAI 호환 API" --> C3["Gemini API"]
  end
```

## 용어

| 용어 | 뜻 |
|---|---|
| 소스 | 소리 나는 대상 1개. 같은 대상이 소리를 3번 내면 소스 1개에 구간 3개입니다. 종류가 같아도 대상이 다르면 다른 소스입니다 |
| 분리 조건 | 텍스트 설명(필수), 구간(선택), 시각 힌트(선택: 시점, 0~1 박스, 설명) |
| 시각 힌트 | 플래너가 화면에 보이는 소스마다 만듭니다. **기본값은 사용 안 함**(`AUDYSSEY_VISUAL_PROMPT=false`). 켜면 SAM3가 대상을 찾아 나머지를 검게 칠한 영상을 만들고, 분리할 때 함께 넣습니다 |
| 분리 트랙 | 소스마다 1개. 원본과 길이가 같은 48kHz 모노 WAV |
| 생성 트랙 | 영상 길이와 같은 샘플 수(`round(duration * 48000)`)의 48kHz 모노 WAV. 설명과 seed로 생성하며 계획 구간과 시각 힌트는 사용하지 않습니다 |
| 나머지 (residual) | 원본에서 분리 트랙만 keep과 무관하게 뺀 소리. 생성 트랙은 빼지 않으며, 무음 영상에서는 만들지 않습니다. 분리 결과를 편집 없이 합치면 원본과 같습니다. `source_id="residual"`로 켜고 끄거나 음량을 바꿀 수 있습니다 |

## 검증과 재시도

| | 플래닝 검증 | 필터링 검증 |
|---|---|---|
| 검사 | 형식 검사 → `validate_plan` → (시각 힌트를 켰을 때) 힌트마다 SAM3 `locate` | `judge`가 분리 트랙을 평가 |
| 실패하면 | 문제 목록(`PlanIssue`)을 `plan(feedback=…)`에 넘겨 계획 전체를 다시 세움 | `suggestion`이 있으면 그 조건으로, 없으면 `attempt`를 1 올려 그 소스만 다시 분리 |
| 횟수 한도 | `AUDYSSEY_MAX_PLAN_ROUNDS=2` | `AUDYSSEY_MAX_SEPARATION_RETRIES=2` (최대 3번 시도) |
| 한도에 닿으면 | 마지막 계획으로 진행하고 경고 표시 | 마지막 트랙을 쓰고 `filter.passed=False` 경고 표시 |

## 계약, 함수, 명령, API

**공통 데이터 형식**은 [`src/audyssey/contracts.py`](../src/audyssey/contracts.py)에 Pydantic 모델로 있습니다.
주요 모델은 `SourcePlan`, `VisualHint`, `PlanIssue`, `FilterResult`, `GroundResult`, `Job`, `Command`입니다.
오디오는 파일 경로로 주고받고, 시간은 초 단위 float입니다.

**생성 경로 계약** (이전 Job JSON은 기본값으로 읽습니다):

| 모델 | 추가·변경 |
|---|---|
| `Job` | `silent: bool = False`. 오디오 스트림이 없거나 최대 진폭이 −60dBFS 미만이면 무음. 영상 길이의 0 오디오·모노·파형 파일을 만듭니다 |
| `Source` | `origin: "separated" \| "generated" = "separated"`, `replaces: str | None = None` |
| `SourceStatus` | `skipped` 추가. 무음 영상의 대사는 트랙 없이 건너뛰며 `dialogue is not generated`를 기록 |
| `Command` | `Generate(source_id, description?)` 추가, `AddSource.method: "separate" \| "generate" = "separate"` 추가 |

유음 업로드는 기존 분리 흐름을 돌리고, 생성은 명령을 받을 때만 합니다. 무음 업로드는 플래닝·검증 후 비대사 소스를 생성합니다.
`generate`가 분리 소스를 가리키면 `<원래 이름> (generated)` 소스를 만들고 원래 소스는 지우지 않고 `keep=False`로 둡니다.
같은 분리 소스의 대체 생성 소스가 있으면 재사용합니다. 생성 소스 대상이면 그 자리에서 설명을 바꾸고 다시 생성합니다.
`attempts`는 누적되며 seed는 `AUDYSSEY_SELVA_SEED + attempts - 1`입니다. 루프마다 최대 `1 + MAX_GENERATION_RETRIES`번 시도하고, 합격 시 멈추며 최고 `text_margin` 결과만 남깁니다(없으면 0, 동점이면 최신).
단계 키는 `generate:<id>`, `score:<id>`이고, 각 시도의 점수는 score 메시지와 LangSmith에 남깁니다. 채점은 기존 `FilterResult`를 씁니다.
형제 계획은 대사·자기 자신·자기가 대체하는 원본을 제외합니다. 텍스트 TargetSel만으로 합격을 판정합니다. `Source.replaces`는 대체 생성과 원래 분리 소스의 연결을 유지합니다.
real 생성은 기본 15초까지 단일 패스이며, 그보다 길면 8초 창·1초 겹침으로 처리합니다. 마지막 창은 끝에 맞추고 전체 겹침을 선형 크로스페이드합니다.
생성 뒤에는 residual을 다시 계산하지 않습니다. 원래 분리 stem을 다시 켜면 두 stem 모두 합쳐집니다.
원본 코덱 패딩과 생성 길이가 다르면 믹스에 더할 때만 겹치는 길이를 사용하며, 내보내는 stem 길이는 유지합니다.

**도구 함수**는 `stub`, `real`, `remote` 모두 시그니처가 같습니다.

```python
plan(video_path, audio_path, duration, feedback: list[PlanIssue]) -> list[SourcePlan]
validate_plan(video_path, audio_path, plans: list[SourcePlan]) -> list[PlanIssue]  # 빈 목록 = 통과
separate(mono_path, plan: SourcePlan, masked_video_path | None, out_path, attempt: int) -> out_path  # None = 텍스트만
judge(mono_path, track_path, plan: SourcePlan) -> FilterResult
locate(video: Path, hint: VisualHint) -> GroundResult
mask_video(video: Path, hint: VisualHint, out: Path) -> Path
generate(video: Path, plan: SourcePlan, out: Path, seed: int) -> Path
score_generation(track: Path, plan: SourcePlan, siblings: list[SourcePlan]) -> FilterResult
render(job: Job, job_dir, include_residual_stem: bool) -> RenderOutput
```

**명령.** 작업 상태는 명령으로만 바꿉니다(`graph.handle_command`).
버튼, 나중에 붙일 자연어 입력, 에이전트가 모두 같은 명령을 쓰게 하려는 규칙입니다.

```
keep / drop(source_id)            set_gain(source_id, 0.0~2.0)     set_pitch(source_id, -12~+12)
reseparate(source_id, description?, spans?)   generate(source_id, description?)
add_source(plan, method="separate")          render(include_residual_stem)
```

대사·residual 대상 `generate`, 무음 영상의 `add_source(method="separate")`, 대사 `add_source(method="generate")`, 생성 소스·무음 영상의 `reseparate`는 거절합니다(422, 다른 명령 검증 오류와 같음). 없는 소스는 404입니다. `generate`도 실행 잠금을 사용해 바쁜 작업에는 409를 돌려줍니다.

**main 서버 API.** UI는 아래 API만 쓰면 됩니다.

```
POST /api/jobs                    multipart(video) → Job (업로드 후 자동 진행)
GET  /api/jobs                    → Job 요약 목록
GET  /api/jobs/{id}               → Job
GET  /api/jobs/{id}/events        SSE: 상태가 바뀔 때마다 Job 전송
POST /api/jobs/{id}/commands      Command → Job | 404(없는 소스) | 409(실행 중) | 422(형식·검증 오류)
GET  /api/jobs/{id}/files/{path}  작업 폴더의 오디오·영상·파형 파일
```

**도구 서버 API.** `/health`를 빼면 모두 `Authorization: Bearer $AUDYSSEY_TOOL_TOKEN`이 필요합니다.

```
GET  /health                                    인증 없음
POST /separate  multipart mono, plan, attempt (+ video)     → WAV   video는 시각 힌트를 켰을 때만
POST /judge     multipart mono, track, plan                  → FilterResult
POST /locate    multipart video, hint                        → GroundResult
POST /mask      multipart video, hint                        → MP4
POST /generate  multipart video, plan, seed                   → WAV
POST /score     multipart track, plan, siblings → FilterResult
```

생성 도구도 `audyssey tool-server --tool generation --impl stub`으로 실행합니다. `siblings`는 SourcePlan 목록 JSON입니다. 생성 track만 업로드하며 업로드 이름은 저장 경로로 사용하지 않습니다. 기존 Bearer 인증·파일별 업로드 크기 제한·remote 시간 제한을 동일하게 적용합니다. `real` 생성은 SelVA FT·SelSync와 CLAP 채점으로 구현하며 별도 `gen/` 환경에서 실행합니다.

## 코드 위치

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
    generation/  stub.py real.py remote.py   generate, score_generation
    editing/     render.py                   residual, pitch, render
gpu/             GPU 도구 전용 uv 프로젝트 (Python 3.12, numpy 1.26, torch cu128)
docker/          app.Dockerfile, tools.Dockerfile
compose.yaml     main + separation + judge + grounding + vllm
tests/           unittest (GPU 코드는 가짜 torch/sam_audio로 검사)
```
