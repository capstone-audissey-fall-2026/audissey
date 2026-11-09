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
| `AUDYSSEY_SAM_AUDIO_MODEL` | `facebook/sam-audio-large-tv` | 분리 모델 ID |
| `AUDYSSEY_SAM_AUDIO_JUDGE_MODEL` | `facebook/sam-audio-judge` | 필터링 모델 ID |
| `AUDYSSEY_SAM_AUDIO_DEVICE` | `cuda` | 모델 장치 |
| `AUDYSSEY_SAM_AUDIO_RERANKING_CANDIDATES` | `1` | 분리 후보 수 (1 이상) |
| `AUDYSSEY_SAM_AUDIO_PREDICT_SPANS` | `false` | span이 없을 때 모델의 span 예측 |
| `AUDYSSEY_JUDGE_PASS_THRESHOLD` | `0.5` | Judge 예시 기준, 아직 실측 보정 전 |

설정은 환경변수로만 읽으며 `.env`는 읽지 않습니다. 작업은 `<WORK_DIR>/<id>/job.json`과 상대 경로 파일로 보존됩니다. 서버 재시작 후 `GET /api/jobs`와 `GET /api/jobs/{id}`로 다시 조회할 수 있습니다. 단일 서버 프로세스로 실행합니다. 실행 중 재분리·추가·합치기는 409를 반환하며 keep/drop과 음량·피치 변경은 허용합니다.

팀원은 담당 `src/audyssey/tools/<모듈>/real.py`의 동일 시그니처 함수를 구현하고 `AUDYSSEY_PLANNER=real`, `AUDYSSEY_SEPARATOR=real`, `AUDYSSEY_FILTER=real` 중 해당 값을 설정합니다. 분리는 SAM Audio, 필터링은 SAM Audio Judge 예시로 구현되어 있습니다. 플래닝 real은 담당자가 구현할 placeholder입니다.

- 도은: `planning/real.py`의 `plan`, `validate_plan`
- SAM Audio 담당: `separation/real.py`의 `separate`
- 윤석: `filtering/real.py`의 `judge`
- UI 담당: 최종 UI를 결정합니다. 이전 `web/` React UI는 삭제되었고 git 기록과 `legacy` 원격에 남아 있습니다.

공통 계약 `contracts.py` 변경은 PR로 합의합니다. 이전 `v2a_inspect` 코드는 삭제되었고 git 기록과 `legacy` 원격에 남아 있습니다. 모델은 필요할 때 로드하고 가중치는 `HF_HOME` 캐시를 사용합니다. 뼈대 실행에는 모델 다운로드가 없습니다. 도구 교체 시 API와 오케스트레이터를 수정하지 않습니다.

[설계](docs/superpowers/specs/2026-11-09-audyssey-skeleton-design.md) · [구현 계획](docs/superpowers/plans/2026-11-09-audyssey-skeleton.md)


## GPU 실행 차단 사항

현재 `gpu/uv.lock`은 의존성 해결·설치에는 성공했지만 real 모델을 실행할 준비가
완료된 상태는 아닙니다. 정적 리뷰에서 `huggingface-hub==2.4.0`의
`ModelHubMixin.from_pretrained`가 고정 SAM Audio의 `_from_pretrained`에 필요한
`proxies`, `resume_download`를 전달하지 않는 API 불일치를 확인했습니다.
SAM Audio 내부 Judge ranker도 같은 로더를 사용하므로 직접 호출 두 곳만 바꾸는
것으로는 해결이 보장되지 않습니다. 외부 제약 강제 핀, monkeypatch, ranker 비활성화는
하지 않았습니다. 이 호환성 문제를 해결하기 전에는 아래 GPU 배포 절차를 완료된
실행 가이드로 간주하지 마세요. GPU·Docker 실행 자체도 sandbox로 검증 대기입니다.

## 한 GPU 서버에서 전부 띄우기 (sv3-a100 등 Docker 되는 서버)

NVIDIA Container Toolkit과 Docker Compose가 설치된 서버에서 실행합니다.

```bash
cp .env.example .env
# .env의 AUDYSSEY_TOOL_TOKEN을 임의의 긴 토큰으로 교체
# sv3-a100: HF_CACHE=/home/common/huggingface, AUDYSSEY_SEPARATION_GPU·AUDYSSEY_JUDGE_GPU로 GPU 지정
docker compose up
# http://localhost:8600 에서 samples/sample.mp4 업로드
```

main, separation, judge 세 컨테이너가 실행됩니다. main만 `127.0.0.1:8600`에
연결되며 GPU 도구 포트는 호스트에 공개하지 않습니다. 원격 서버 UI는 SSH 터널로
연결합니다. 작업은 `jobs` 볼륨에 남습니다. 플래너는 기본 stub입니다.

이미지는 `ghcr.io/capstone-audissey-fall-2026/audyssey-main:latest`와
`ghcr.io/capstone-audissey-fall-2026/audyssey-sam-audio:latest`입니다.
처음 배포되어 있다면 `docker compose pull`, 직접 빌드하려면 `docker compose build`를
먼저 실행합니다. main 반영 또는 수동 workflow 실행 시 GHCR에 latest와 커밋 SHA
태그를 올립니다. PR에서는 업로드하지 않습니다. 이미지가 비공개면 GHCR 읽기 권한으로
인증해야 합니다. 이 브랜치의 Docker 빌드·GPU 실측은 sandbox 때문에 대기 중입니다.

## 노트북 + RunPod

RunPod에서 GPU **Pod 하나**를 만들고 다음 값을 설정합니다.

| 설정 | 값 |
|---|---|
| Container image | `ghcr.io/capstone-audissey-fall-2026/audyssey-sam-audio:latest` |
| GPU | GPU 1개, CUDA 12.8 지원 드라이버; 필요한 VRAM은 실측 대기 |
| Expose HTTP ports | `8700` |
| Container start command | 비워 두어 이미지 기본 CMD 사용 (분리와 Judge를 함께 실행) |
| Network volume | 기존 가중치 캐시가 있는 볼륨을 `/workspace`에 연결 |
| `HF_HOME` | `/workspace/huggingface` (볼륨 안의 실제 캐시 경로) |
| `HF_HUB_OFFLINE` / `CUDA_VISIBLE_DEVICES` | `1` / `0` |
| `AUDYSSEY_TOOL_TOKEN` | 노트북 .env와 같은 비밀 토큰 |
| 모델 ID 2개 | `.env.example`의 `AUDYSSEY_SAM_AUDIO_MODEL`, `AUDYSSEY_SAM_AUDIO_JUDGE_MODEL` |

Templates → New Template에서 이미지·포트·환경변수·시작 명령을 위 값으로 저장하고,
Pod 배포 시 그 템플릿과 네트워크 볼륨을 고릅니다. 비밀 토큰은 공유 템플릿에 넣지
말고 Pod 환경/RunPod secrets로 주입합니다. [템플릿 공식 안내](https://docs.runpod.io/pods/templates/manage-templates).

노트북에서는 `.env.example`을 `.env`로 복사해 토큰과 두 도구의 주소를 넣습니다.

```dotenv
AUDYSSEY_SEPARATOR_URL=https://YOUR-POD-ID-8700.proxy.runpod.net
AUDYSSEY_FILTER_URL=https://YOUR-POD-ID-8700.proxy.runpod.net
```

```bash
docker compose up main
# http://localhost:8600
```

main에는 GPU가 필요 없습니다. RunPod GPU Pod 안에서는 Docker Compose를 실행할 수
없으므로, 이미지 하나의 기본 CMD로 두 도구를 함께 실행합니다.
[RunPod의 컨테이너 제한](https://docs.runpod.io/tutorials/pods/build-docker-images).

## 가중치와 로컬 GPU 환경

가중치는 이미지에 포함하지 않으며 서버가 내려받지 않도록 `HF_HUB_OFFLINE=1`을
사용합니다. sv3-a100은 `/home/common/huggingface`를 `/hf:ro`로 연결합니다.
RunPod는 사용자가 별도로 미리 준비한 네트워크 볼륨 캐시를 사용합니다. 공식
`facebook/sam-audio-*`에는 HF 승인과 SAM License가 필요하고, 미러를 쓸 때도 사용
권한을 확인해야 합니다. 체크포인트 외에 tokenizer·설정·보조 모델 캐시도 있어야 합니다.
Judge의 고정 upstream은 `sam_audio` revision을 읽으므로 그 revision도 캐시에 필요합니다.
이 작업에서는 가중치를 다운로드하거나 복사하지 않았습니다.

SAM Audio 의존성의 numpy<2 조건 때문에 GPU 도구는 별도 Python 3.12 uv 프로젝트를
씁니다. 루트 환경과 CI에는 torch를 설치하지 않습니다.

```bash
uv sync --project gpu --locked
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 HF_HOME=/home/common/huggingface
export AUDYSSEY_SAM_AUDIO_MODEL=mrfakename/sam-audio-large-tv
export AUDYSSEY_SAM_AUDIO_JUDGE_MODEL=mrfakename/sam-audio-judge
# AUDYSSEY_TOOL_TOKEN은 환경에서 주입
uv run --project gpu --locked audyssey tool-server --tool separation --tool filtering --impl real --host 0.0.0.0 --port 8700
```

한 도구만 실행하려면 `--tool`을 하나만 줍니다. 공개 주소에는 토큰이 필수입니다.
원격 연결은 TLS 프록시나 SSH 터널로 보호합니다. `GET /health`는 인증 없이
`{"tools": ["separation", "filtering"], "impl": "real"}`을 반환합니다. 모델은 첫 호출에
프로세스당 한 번 로드하고 호출을 직렬화합니다. span은 positive anchor로 전달하고,
`attempt`를 시드로 써 재시도를 구분합니다. 출력은 원본 길이의 48 kHz float32 mono
WAV입니다. 시각 힌트/SAM3는 다음 PR 범위입니다. Judge는 점수와 통과 여부만 내며
suggestion은 없습니다. 기준값 0.5는 아직 보정되지 않은 예시입니다.

GPU·가중치 없이 HTTP 연결만 확인하려면 루트 환경에서 `--impl stub`을 사용합니다.
원격 모드는 `AUDYSSEY_SEPARATOR=remote`, `AUDYSSEY_FILTER=remote`, 두 URL과 같은
토큰을 설정합니다. `AUDYSSEY_REMOTE_TIMEOUT_S` 기본값은 600초입니다. URL이 없으면
시작 시 오류가 나며 인증·서버·타임아웃 오류는 해당 소스만 실패시킵니다.

```bash
# 위 offline/GPU 0/모델 ID 환경변수를 설정한 뒤
uv run --project gpu --locked python scripts/sam_audio_smoke.py samples/sample.mp4 --description "a person speaking" --out work/jobs/smoke-text
uv run --project gpu --locked python scripts/sam_audio_smoke.py samples/sample.mp4 --description "a person speaking" --span 0 3 --out work/jobs/smoke-span
# CI와 같은 GPU 없는 환경
uv sync --dev --locked --extra ui --extra llm
uv run --isolated --no-project --python 3.12 --with-editable '.[ui,llm]' --with 'numpy==1.26.4' python -m unittest discover tests
```

스모크 스크립트는 로딩·분리·Judge 시간, 최대 GPU 메모리, torch/CUDA 버전, 판정과
결과 WAV 경로를 JSON으로 출력합니다. [실측 대기 표와 차단 명령](docs/superpowers/specs/2026-11-09-audyssey-skeleton-design.md#sam-audio-실측-2026-11-10)에 현재 상태를 기록합니다.

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
