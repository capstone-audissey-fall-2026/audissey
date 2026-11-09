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
| `AUDYSSEY_GROUNDER` | `stub` | `stub`, `real`, `remote` |
| `AUDYSSEY_GROUNDING_URL` | 빈 문자열 | remote 그라운딩 도구 서버 주소 |
| `AUDYSSEY_SAM3_MODEL` | `facebook/sam3.1` | 기존 캐시의 SAM3.1 모델 ID |
| `AUDYSSEY_SAM3_DEVICE` | `cuda` | SAM3 CUDA 장치 (`cuda:0` 등) |
| `AUDYSSEY_PLANNER_FRAMES` | `4` | 플래너·검증기 JPEG 프레임 수 (1~16) |
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

팀원은 담당 `src/audyssey/tools/<모듈>/real.py`의 동일 시그니처 함수를 구현하고 `AUDYSSEY_PLANNER=real`, `AUDYSSEY_SEPARATOR=real`, `AUDYSSEY_FILTER=real` 중 해당 값을 설정합니다. 분리는 SAM Audio, 필터링은 SAM Audio Judge 예시로 구현되어 있습니다. 플래닝 real은 연결 확인용 LLM 기본 구현이며 도은 담당자가 교체할 수 있습니다.

- 도은: `planning/real.py`의 `plan`, `validate_plan`
- SAM Audio 담당: `separation/real.py`의 `separate`
- 윤석: `filtering/real.py`의 `judge`
- UI 담당: 최종 UI를 결정합니다. 이전 `web/` React UI는 삭제되었고 git 기록과 `legacy` 원격에 남아 있습니다.

공통 계약 `contracts.py` 변경은 PR로 합의합니다. 이전 `v2a_inspect` 코드는 삭제되었고 git 기록과 `legacy` 원격에 남아 있습니다. 모델은 필요할 때 로드하고 가중치는 `HF_HOME` 캐시를 사용합니다. 뼈대 실행에는 모델 다운로드가 없습니다. 도구 교체 시 API와 오케스트레이터를 수정하지 않습니다.

[설계](docs/superpowers/specs/2026-11-09-audyssey-skeleton-design.md) · [구현 계획](docs/superpowers/plans/2026-11-09-audyssey-skeleton.md)


## GPU 실행 상태

dnn5(RTX A5000 24GB × 4)에서 모든 모듈을 real로 연결한 E2E를 확인했습니다.
`docker compose up` 한 번으로 main, 분리, Judge, SAM3, vLLM 컨테이너 5개가 뜨고,
샘플 업로드 → 플래닝·검증 루프 → 분리·판정 → render까지 306초에 끝났습니다.
Gemini 플래너와 높인 Judge 기준으로는 시각 힌트 마스크 분리와 재분리 루프까지 확인했습니다.
실측값과 알려진 한계는
[설계 문서](docs/superpowers/specs/2026-11-09-audyssey-skeleton-design.md#전체-e2e-실측-2026-11-10-dnn5-rtx-a5000-24gb--4)에 있습니다.

## 한 GPU 서버에서 전부 띄우기 (sv3-a100 등 Docker 되는 서버)

NVIDIA Container Toolkit과 Docker Compose가 설치된 서버에서 실행합니다.

```bash
cp .env.example .env
# .env의 AUDYSSEY_TOOL_TOKEN을 임의의 긴 토큰으로 교체
# sv3-a100: HF_CACHE=/home/common/huggingface, AUDYSSEY_SEPARATION_GPU·AUDYSSEY_JUDGE_GPU·AUDYSSEY_GROUNDING_GPU로 GPU 지정
docker compose up
# http://localhost:8600 에서 samples/sample.mp4 업로드
```

main, separation, judge, grounding, vllm 다섯 컨테이너가 실행됩니다. main만 `127.0.0.1:8600`에
연결되며 GPU 도구 포트는 호스트에 공개하지 않습니다. 원격 서버 UI는 SSH 터널로
연결합니다. 작업은 `jobs` 볼륨에 남습니다. Compose 플래너는 기본 real이며 내부 `http://vllm:8000/v1`로
연결합니다. Python 직접 실행의 기본값은 계속 stub입니다. vLLM은
`Qwen/Qwen2.5-VL-3B-Instruct` 캐시를 미리 준비해야 하며 GPU 기본값은 0,
메모리 사용 비율은 0.85입니다. 다른 GPU 도구와 함께 실행할 때는
`AUDYSSEY_VLLM_GPU`와 메모리 여유를 조정합니다.

vLLM 이미지는 [공식 v0.32.0 릴리스](https://github.com/vllm-project/vllm/releases/tag/v0.32.0)로
고정했습니다. [해당 버전의 지원 모델](https://docs.vllm.ai/en/v0.32.0/models/supported_models.html)에
Qwen2.5-VL-3B가 있습니다. 이미지 pull과 GPU 실행은 운영자 확인이 필요합니다.
Gemini 키를 `.env`의 `GOOGLE_API_KEY`로 준비한 경우 다음 두 줄로 전환합니다.
`AUDYSSEY_PLANNER_API_KEY`를 따로 설정하면 그 값이 우선합니다.

```dotenv
AUDYSSEY_PLANNER_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
AUDYSSEY_PLANNER_MODEL=gemini-2.5-flash
```

Gemini에서는 `docker compose up main separation judge`로 vLLM 없이 실행할 수 있습니다.
GPU 없이 Compose main만 실행하려면 `AUDYSSEY_PLANNER=stub`과 분리·필터링 stub
또는 원격 도구 주소를 설정합니다.

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
| Container start command | 비워 두어 이미지 기본 CMD 사용 (분리·Judge·그라운딩을 함께 실행) |
| Network volume | 기존 가중치 캐시가 있는 볼륨을 `/workspace`에 연결 |
| `HF_HOME` | `/workspace/huggingface` (볼륨 안의 실제 캐시 경로) |
| `HF_HUB_OFFLINE` / `CUDA_VISIBLE_DEVICES` | `1` / `0` |
| `AUDYSSEY_TOOL_TOKEN` | 노트북 .env와 같은 비밀 토큰 |
| 모델 ID 3개 | `.env.example`의 `AUDYSSEY_SAM_AUDIO_MODEL`, `AUDYSSEY_SAM_AUDIO_JUDGE_MODEL`, `AUDYSSEY_SAM3_MODEL` |

Templates → New Template에서 이미지·포트·환경변수·시작 명령을 위 값으로 저장하고,
Pod 배포 시 그 템플릿과 네트워크 볼륨을 고릅니다. 비밀 토큰은 공유 템플릿에 넣지
말고 Pod 환경/RunPod secrets로 주입합니다. [템플릿 공식 안내](https://docs.runpod.io/pods/templates/manage-templates).

노트북에서는 `.env.example`을 `.env`로 복사해 토큰과 세 도구의 주소를 넣습니다.

```dotenv
AUDYSSEY_SEPARATOR_URL=https://YOUR-POD-ID-8700.proxy.runpod.net
AUDYSSEY_FILTER_URL=https://YOUR-POD-ID-8700.proxy.runpod.net
AUDYSSEY_GROUNDING_URL=https://YOUR-POD-ID-8700.proxy.runpod.net
```

```bash
docker compose up main
# http://localhost:8600
```

main에는 GPU가 필요 없습니다. RunPod GPU Pod 안에서는 Docker Compose를 실행할 수
없으므로, 이미지 하나의 기본 CMD로 세 도구를 함께 실행합니다.
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
# torchcodec(cu128)가 쓰는 NVIDIA NPP 경로. Docker 이미지는 이미 설정되어 있음
export LD_LIBRARY_PATH="$PWD/gpu/.venv/lib/python3.12/site-packages/nvidia/npp/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 HF_HOME=/home/common/huggingface
export AUDYSSEY_SAM_AUDIO_MODEL=mrfakename/sam-audio-large-tv
export AUDYSSEY_SAM_AUDIO_JUDGE_MODEL=mrfakename/sam-audio-judge
# AUDYSSEY_TOOL_TOKEN은 환경에서 주입
uv run --project gpu --locked audyssey tool-server --tool separation --tool filtering --tool grounding --impl real --host 0.0.0.0 --port 8700
```

한 도구만 실행하려면 `--tool`을 하나만 줍니다. 공개 주소에는 토큰이 필수입니다.
원격 연결은 TLS 프록시나 SSH 터널로 보호합니다. `GET /health`는 인증 없이
`{"tools": ["separation", "filtering", "grounding"], "impl": "real"}`을 반환합니다. 모델은 첫 호출에
프로세스당 한 번 로드하고 호출을 직렬화합니다. span은 positive anchor로 전달하고,
`attempt`를 시드로 써 재시도를 구분합니다. 출력은 원본 길이의 48 kHz float32 mono
WAV입니다. 시각 힌트가 있으면 아래 SAM3 그라운딩을 거친 마스크 영상을 함께 사용합니다. Judge는 점수와 통과 여부만 내며
suggestion은 없습니다. 기준값 0.5는 아직 보정되지 않은 예시입니다.

GPU·가중치 없이 HTTP 연결만 확인하려면 루트 환경에서 `--impl stub`을 사용합니다.
원격 모드는 `AUDYSSEY_SEPARATOR=remote`, `AUDYSSEY_FILTER=remote`, `AUDYSSEY_GROUNDER=remote`, 세 URL과 같은
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
결과 WAV 경로를 JSON으로 출력합니다.

24GB GPU에 올리기 위해 분리 어댑터는 재순위 후보가 1개이고 구간 예측을 끈 기본 설정에서
쓰지 않는 ranker와 span predictor를 GPU에 올리지 않습니다(SAM Audio가 None을 공식 지원).
`AUDYSSEY_SAM_AUDIO_RERANKING_CANDIDATES`를 2 이상으로, `AUDYSSEY_SAM_AUDIO_PREDICT_SPANS`를
`true`로 바꾸면 해당 모듈을 함께 올리며 메모리가 크게 늘어납니다(small 전체 fp32 약 23GB).

## 플래너를 Gemini/vLLM에 연결하기

`planning/real.py`는 도은 담당 모듈의 교체 가능한 기본 구현입니다. 영상 길이에 고르게
뽑은 JPEG 프레임(기본 4장, 가로 최대 512px)과 시각·피드백을 기존 OpenAI 호환
`chat_client(settings)`로 보냅니다. 영상·오디오 파일은 보내지 않습니다. 오디오 전사도
아직 없으므로 실제 들리는 소리를 확인하는 품질 검증용으로 쓰지 않습니다.

소스별 Pydantic 검증 후 잘못된 소스는 버리고, 시간과 박스 범위를 자릅니다.
형식이 깨진 응답은 한 번만 다시 요청하며, 계속 깨지면 플래닝은 실패합니다.
검증기는 의심스러운 소스만 `PlanIssue`로 보고하고, 두 차례 형식 오류이면 경고를
기록한 뒤 이슈 없이 진행합니다. API 연결·인증 오류는 호출자에게 전달됩니다.

```bash
uv sync --dev --extra ui --extra llm
# Gemini: .env에 GOOGLE_API_KEY를 비공개로 준비 (커밋하거나 출력하지 않음)
AUDYSSEY_PLANNER=real \
AUDYSSEY_PLANNER_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/ \
AUDYSSEY_PLANNER_MODEL=gemini-2.5-flash \
uv run --env-file .env python -c 'import os; from audyssey.cli import serve; os.environ["AUDYSSEY_PLANNER_API_KEY"] = os.environ.get("AUDYSSEY_PLANNER_API_KEY") or os.environ["GOOGLE_API_KEY"]; serve()'
```

로컬 vLLM 주소가 이미 있으면 `AUDYSSEY_PLANNER=real`,
`AUDYSSEY_PLANNER_BASE_URL=http://localhost:8000/v1`,
`AUDYSSEY_PLANNER_MODEL=Qwen/Qwen2.5-VL-3B-Instruct`를 설정하고
`uv run audyssey serve`를 실행합니다. 인증 없는 로컬 서버는 키를 생략할 수 있습니다.
Compose의 vLLM 포트는 호스트에 공개하지 않으므로 아래 확인은 Compose 네트워크 안에서 합니다.

### 운영자 실연결 확인 (이 브랜치에서는 실행하지 않음)

같은 스크립트로 `samples/sample.mp4`의 `plan`과 `validate_plan`을 각각 한 번 호출해
소스 수·이슈 수·소요 시간을 출력합니다. JSON 재요청과 SDK의 전송 재시도는 별도로
발생할 수 있으므로 반복 호출하지 않습니다. 키·미디어·전체 모델 응답은 출력하지 않습니다.

```bash
cat > /tmp/audyssey-planner-live.py <<'PYTHON'
import json
import os
import time
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
    start = time.monotonic()
    plans = plan(video, audio, duration, [])
    plan_s = time.monotonic() - start
    start = time.monotonic()
    issues = validate_plan(video, audio, plans)
    print(json.dumps({"sources": len(plans), "issues": len(issues),
                      "plan_s": plan_s, "validate_s": time.monotonic() - start}))
PYTHON

# Gemini: root uv environment; .env is read only by the human operator.
AUDYSSEY_PLANNER_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/ \
AUDYSSEY_PLANNER_MODEL=gemini-2.5-flash \
uv run --env-file .env python /tmp/audyssey-planner-live.py

# vLLM: pre-cache Qwen weights/processor under HF_CACHE before starting.
docker compose pull vllm
docker compose build main
docker compose up -d vllm
# Wait for the vLLM log to report that the API server is ready.
docker compose run --rm -T --entrypoint python \
  -e AUDYSSEY_PLANNER_BASE_URL=http://vllm:8000/v1 \
  -e AUDYSSEY_PLANNER_MODEL=Qwen/Qwen2.5-VL-3B-Instruct \
  -e AUDYSSEY_PLANNER_API_KEY=unused \
  -e AUDYSSEY_SMOKE_VIDEO=/samples/sample.mp4 \
  -v "$PWD/samples:/samples:ro" main - < /tmp/audyssey-planner-live.py
```

Docker/GPU 실측과 Gemini/vLLM 실제 응답은 운영자가 확인해야 합니다.
비밀값 없이 Compose만 검사하려면 다음을 실행합니다. `--no-env-resolution`으로 서비스의
`.env` 로딩을 막고 `--env-file`로 자동 `.env` 보간도 막습니다.

```bash
env -i PATH="$PATH" docker compose --env-file .env.example config --no-env-resolution --quiet
```

## LangSmith로 보기

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=...
export LANGSMITH_PROJECT=audyssey
uv run audyssey serve
```

키와 tracing 환경변수를 설정하지 않은 기본 상태에서는 아무것도 보내지 않습니다. 호스팅된 LangSmith로 보낼 때는 세 변수를 모두 설정합니다. tracing만 켜고 키를 생략해도 전송 시도는 일어날 수 있으므로 끌 때는 `LANGSMITH_TRACING=false`를 씁니다. 그래프, 도구, 명령과 래핑된 LLM 호출의 토큰 사용량을 볼 수 있습니다. 명령 입력은 `job_id`와 명령만 남기고 Job 전체·설정은 제외합니다. 오디오·영상·프레임 바이트와 키·토큰은 올리지 않고 경로와 메타데이터만 기록합니다. 플래너 LLM trace에는 모델명과 토큰 사용량만 남기며 이미지가 포함된 프롬프트와 원문 응답은 기록하지 않습니다.


## SAM3 시각 그라운딩

플래너의 `VisualHint(time, box, description)`가 있는 소스만 처리합니다. 검증 단계에서
`locate`가 대상을 못 찾거나 예외를 내면 `visual_hint_not_found` 문제를 재계획에 전달합니다.
분리 전 `mask_video`가 대상을 앞뒤로 추적해 `masks/<source_id>.mp4`에 저장하고,
오케스트레이터가 그 경로를 분리 도구에 넘깁니다. SAM Audio는 힌트가 있을 때만
`masked_videos`로 받습니다. 마스크 실패는 해당 소스만 실패시킵니다.

기본 `stub`은 고정 박스 밖을 검게 칠하고 `ghost` 설명을 찾지 못한 것으로 처리합니다.
`real`은 `gpu/`의 고정 SAM3 commit `2814fa619404a722d03e9a012e083e4f293a4e53`을
사용합니다. `facebook/sam3.1`의 `sam3.1_multiplex.pt`가 기존 HF 캐시에 있어야 합니다.
로더는 `local_files_only=True`로 캐시만 읽고 다운로드하지 않습니다. SAM3.1 upstream은
CUDA 전용이므로 `AUDYSSEY_SAM3_DEVICE`에는 CUDA 장치를 지정합니다.

```bash
# GPU·가중치 없이 HTTP 테스트
uv run audyssey tool-server --tool grounding --impl stub --port 8702
# 다른 터미널: 메인 서버가 이 도구를 사용
AUDYSSEY_GROUNDER=remote AUDYSSEY_GROUNDING_URL=http://127.0.0.1:8702 uv run audyssey serve
```

`POST /locate`는 multipart `video`와 `hint` JSON을 받아 `GroundResult` JSON을 반환하고,
`POST /mask`는 같은 입력으로 MP4를 반환합니다. 다른 도구와 같은 Bearer 토큰과
600초 기본 타임아웃을 사용합니다. compose는 `AUDYSSEY_GROUNDING_GPU`(기본 0)로
그라운딩 GPU를 지정하며 main의 기본 주소는 `http://grounding:8700`입니다.
RunPod에서는 세 URL에 같은 Pod 주소를 넣습니다.

마스크 MP4는 RGB 무손실이며 원본 해상도를 유지합니다. SAM3는 프레임별 추적 상태를 GPU에 쌓으므로
기본적으로 `AUDYSSEY_SAM3_TRACK_FPS`(기본 2)로 낮춘 영상을 추적하고 그 fps로 마스크 영상을 씁니다.
SAM Audio는 마스크 영상을 시간 기준으로 맞춰 읽으므로 fps가 원본과 달라도 됩니다. `0`이면 원본 fps
그대로 추적합니다(1080p60 기준 4fps부터 24GB GPU에서 메모리 부족). 추적하지 못한 프레임은 검은색입니다.

dnn5(RTX A5000 24GB)에서 1920×1080 60fps 7.1초 샘플로 확인한 값: 위치 찾기 약 27초(첫 호출, 로딩 포함)·
이후 약 6초, 2fps 마스크 영상 25초(14프레임), 최대 GPU 메모리 14.3GB.
한계: 힌트 박스를 함께 주므로 SAM3는 박스 안에서 거의 항상 무언가를 찾습니다(없는 대상 "ghost"도 found).
지금의 위치 확인은 검증으로서 약하며, 기준은 플래닝 담당과 정해야 합니다.
