# 실행 상세

[README](../README.md)의 "실행 방법" 표에서 고른 상황별 안내입니다.

## GPU 없이 (stub)

```bash
uv sync --extra ui
uv run audyssey serve
# http://127.0.0.1:8600 에서 samples/sample.mp4 업로드
```

- 모든 도구가 stub입니다. 유음 영상은 기존 분리·검증, 무음 영상은 대사를 건너뛰고 자동 생성·채점을 돌립니다.
- 개발용 화면에서 재생, 켜고 끄기, 음량, 피치, 설명을 고쳐 다시 분리, 소스 추가, 합치기를 해 볼 수 있습니다.
- 유음 영상을 생성·편집 없이 합친 WAV는 원본과 수치 오차 범위에서 같습니다.
- 소스의 출처(분리/생성)를 확인하고, 비대사 분리 소스는 "생성으로 바꾸기", 생성 소스는 "다시 생성"으로 조작합니다. 소스 추가는 분리/생성을 고를 수 있으며 무음 영상은 생성만 지원합니다.
- 무음 샘플: `ffmpeg -i samples/sample.mp4 -an -c:v copy /tmp/silent.mp4`로 만든 영상을 업로드합니다. 대사는 "대사는 생성하지 않음"으로 표시됩니다.
- 끌 때는 그 터미널에서 Ctrl+C를 누릅니다. 작업 파일은 `work/jobs`에 남습니다.
- 8600 포트를 이미 쓰고 있으면 `uv run audyssey serve --port 8601`처럼 바꿉니다.

**stub이 하는 일**: 결과는 가짜이므로 품질 평가에는 쓰지 않습니다.

| 도구 | stub 동작 |
|---|---|
| 플래너 | 고정된 소스 4개(대사 2, 효과음 2)를 영상 길이 비율로 냄. 검증에서 하나를 지적받아 한 번 다시 계획 |
| 분리 | 계획 구간의 원본 오디오를 그대로 잘라 냄 |
| Judge | `character_a`를 한 번 불합격시켜 재분리 루프를 돌게 함 |
| 생성·채점 | 설명에서 정한 주파수의 사인파를 영상 가운데 최대 1초에 넣음. `sword_clash`만 기본 seed에서 진폭 0.1로 불합격하고 다음 seed는 0.2로 합격. 나머지는 바로 합격. 형제가 없으면 margin을 생략 |
| 그라운딩 | 시각 힌트를 켰을 때만 호출됨. 힌트 상자를 그대로 고정 마스크로 씀 (추적 없음) |

**일부만 실제로 바꾸기**: 바꾸고 싶은 것만 환경변수로 지정하고 나머지는 stub으로 둡니다.
- 플래너만 실제로 (GPU 불필요): 아래 [플래너 연결](#플래너-연결-gemini-vllm)의 Gemini 설정에 `AUDYSSEY_PLANNER=real`
- 내 GPU 모듈만 실제로: [내 모듈만 GPU로 개발](#내-모듈만-gpu로-개발)

코드를 고친 뒤에는 [로컬 검사](contributing.md#로컬-검사)(ruff, ty, 단위 테스트)를 돌립니다. 단위 테스트도 GPU 없이 돕니다.

**생성 remote를 GPU 없이 확인하기** (터미널 두 개):

```bash
AUDYSSEY_TOOL_TOKEN=local-test uv run audyssey tool-server --tool generation --impl stub --port 8701
AUDYSSEY_TOOL_TOKEN=local-test AUDYSSEY_GENERATOR=remote AUDYSSEY_GENERATOR_URL=http://127.0.0.1:8701 uv run audyssey serve
```

real 생성은 SelVA FT + SelSync와 CLAP 채점을 사용하며, numpy 2 / torch 2.6 전용 `gen/` 환경과 `audyssey-generation` 이미지로 분리합니다. 현재 RunPod 구성은 생성 도구를 포함하지 않습니다. RunPod에서 main만 띄울 때는 `AUDYSSEY_GENERATOR=stub`으로 둡니다. 별도 생성 Pod 구성과 검증은 이후에 진행합니다.

## GPU 서버 한 대 (Docker Compose)

Compose는 6개 컨테이너로 구성합니다. 생성 추가 전 5개 구성의 GPU 메모리는 약 50~70GB였습니다([실측](status.md)). 생성 추가 후 한 장에 모두 올렸을 때의 최대 메모리는 **측정 전**이며, Docker 빌드와 GPU E2E는 운영자가 확인해야 합니다.

**1. 준비 (처음 한 번)**
```bash
cp .env.example .env      # .env는 커밋하지 않음
# .env에서 AUDYSSEY_TOOL_TOKEN을 긴 임의 값으로, HF_CACHE를 가중치 캐시 경로로,
# AUDYSSEY_*_GPU를 쓸 GPU 번호로 바꿈 (공용 서버라면 nvidia-smi로 비어 있는 GPU 확인)
```

**2. 띄우기**
```bash
docker compose pull       # 새 이미지 받기 (main 머지 때마다 갱신됨). 직접 빌드하려면 docker compose build
docker compose up -d      # 백그라운드로 실행
docker compose logs -f vllm   # "Application startup complete"가 뜨면 준비 끝 (약 1~2분). Ctrl+C는 로그 보기만 멈춤
```

main, separation, judge, grounding, generation, vllm 컨테이너 6개가 뜹니다. 새 `audyssey-generation` GHCR 패키지는 첫 게시 뒤 공개 상태인지 확인하고, 비공개면 GitHub 패키지 설정에서 공개로 전환합니다. 이미지에는 가중치가 없습니다.
플래너를 vLLM 대신 Gemini로 쓰려면 `.env`에 `GOOGLE_API_KEY`를 넣고 vLLM 없이 띄웁니다.
```bash
AUDYSSEY_PLANNER_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/ \
AUDYSSEY_PLANNER_MODEL=gemini-2.5-flash \
docker compose up -d main separation judge grounding generation
```

**3. 접속**
- 서버에서: http://localhost:8600
- 원격 서버라면 노트북에서 SSH 터널을 열고 같은 주소로 접속합니다: `ssh -L 8600:localhost:8600 <서버>`
- 분리·Judge·그라운딩·생성 모델은 첫 요청 때 GPU에 올라가서 첫 작업만 1~2분 더 걸립니다. 그다음부터는 컨테이너를 내릴 때까지 올라간 채로 재사용합니다.
- 진행 로그: `docker compose logs -f main separation judge generation`

**4. 내리기**
```bash
docker compose down       # 컨테이너를 지우고 GPU 메모리를 돌려줌
nvidia-smi                # 메모리가 비었는지 확인
```
- `up -d`로 띄웠으면 Ctrl+C로는 꺼지지 않습니다. 반드시 `docker compose down`을 실행합니다.
- 작업 결과(`jobs` 볼륨)와 이미지는 남습니다. 다시 띄우면 이전 작업이 그대로 보입니다.
- 빌린 GPU라면 내린 뒤 바로 반납합니다.

**접속과 보안**
- 호스트에는 main의 `127.0.0.1:8600`만 열립니다.
- main에는 인증이 없습니다. `127.0.0.1` 밖으로 열지 말고, 원격에서는 SSH 터널을 씁니다. 외부에 꼭 열어야 하면 인증이 있는 리버스 프록시를 앞에 둡니다.

**GPU 지정**
- 도구별로 `AUDYSSEY_SEPARATION_GPU`, `AUDYSSEY_JUDGE_GPU`, `AUDYSSEY_GROUNDING_GPU`, `AUDYSSEY_GENERATION_GPU`, `AUDYSSEY_VLLM_GPU`에 GPU 번호를 넣습니다. 번호가 같으면 그 GPU를 나눠 씁니다.
- `.env.example`은 GPU 0 한 장을 나눠 쓰는 설정이고, vLLM 메모리 비율은 `AUDYSSEY_VLLM_GPU_UTIL=0.2`입니다(생성 추가 전 80GB A100에서 확인, 생성 포함 메모리는 측정 전).
- vLLM에 GPU를 따로 주면 이 값을 `0.85` 정도로 올립니다.

**가중치와 작업 파일**
- 가중치 캐시(`HF_CACHE`)는 컨테이너 안 `/hf`에 읽기 전용으로 붙고, 오프라인(`HF_HUB_OFFLINE=1`)으로 읽습니다.
- 작업 파일은 `jobs` 볼륨에 남습니다.
- 컨테이너는 일반 사용자(uid 10001)로 실행됩니다. 예전 root 이미지로 만든 볼륨은 쓰기 권한이 없으니 `docker volume rm audyssey_jobs` 후 다시 띄웁니다.

**그 밖에**
- 플래너 기본값은 Compose 안의 vLLM(`Qwen/Qwen2.5-VL-3B-Instruct`)입니다.
- GPU 없이 main만 띄우려면 `AUDYSSEY_PLANNER=stub`과 도구 stub(또는 원격 주소)을 설정하고 `docker compose up main`을 실행합니다.
- sv3-a100에서는 `HF_CACHE=/home/common/huggingface`를 씁니다. 이 서버의 기본 Docker 빌드 엔진은 설정 문제로 실패하므로 `docker buildx --builder <docker-container 빌더>`로 빌드합니다.

## 노트북 + RunPod

RunPod Pod 안에서는 Docker Compose를 쓸 수 없습니다.
그래서 Pod에서는 tools 이미지의 기본 명령으로 도구 3개를 한 프로세스에 띄우고, 노트북에서는 main만 띄웁니다.

| RunPod 설정 | 값 |
|---|---|
| Container image | `ghcr.io/capstone-audissey-fall-2026/audyssey-tools:latest` (공개 이미지) |
| GPU | 1장, CUDA 12.8 드라이버. 기본 설정(시각 힌트 끔)에서는 48GB 이상, 시각 힌트를 켜면 80GB 권장 ([현재 상태](status.md#인프라배포) 참고) |
| Expose HTTP ports | `8700` |
| Container start command | 비워 둠 (이미지 기본 명령이 분리·Judge·그라운딩을 함께 실행) |
| Network volume | 가중치를 미리 받아 둔 볼륨을 `/workspace`에 연결. Pod를 반납해도 볼륨은 남으므로 가중치를 다시 받지 않아도 됨 |
| 환경변수 | `HF_HOME=/workspace/huggingface`, `HF_HUB_OFFLINE=1`, `CUDA_VISIBLE_DEVICES=0`, 모델 ID 3개([가중치 준비](#가중치-준비)) |
| 비밀값 | `AUDYSSEY_TOOL_TOKEN`은 공유 템플릿에 넣지 말고 RunPod secrets로 넣음 |

템플릿은 Templates → New Template에 저장해 두고, Pod를 만들 때 고릅니다([공식 안내](https://docs.runpod.io/pods/templates/manage-templates)).
노트북 `.env`에는 같은 토큰과 Pod 주소를 넣습니다.

```dotenv
AUDYSSEY_TOOL_TOKEN=<Pod와 같은 토큰>
AUDYSSEY_SEPARATOR_URL=https://YOUR-POD-ID-8700.proxy.runpod.net
AUDYSSEY_FILTER_URL=https://YOUR-POD-ID-8700.proxy.runpod.net
AUDYSSEY_GROUNDING_URL=https://YOUR-POD-ID-8700.proxy.runpod.net
AUDYSSEY_GENERATOR=stub  # 현재 RunPod 구성에는 생성 서버가 없음
# 플래너는 Gemini 권장
AUDYSSEY_PLANNER_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
AUDYSSEY_PLANNER_MODEL=gemini-2.5-flash
GOOGLE_API_KEY=<비공개>
```

```bash
docker compose up main
# http://localhost:8600
```

## 내 모듈만 GPU로 개발

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

### 생성 모듈 (SelVA FT + SelSync)

분리 환경과 섞지 않고 [gen/](../gen/README.md)의 잠금 파일을 사용합니다. 가중치는 아래 절의 명령으로 미리 준비합니다.

```bash
uv sync --project gen --locked
export HF_HOME=/home/common/huggingface HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
export AUDYSSEY_TOOL_TOKEN=<임의-토큰>
uv run --project gen --locked audyssey tool-server --tool generation --impl real --host 127.0.0.1 --port 8701
# 다른 터미널: 같은 AUDYSSEY_TOOL_TOKEN을 설정
AUDYSSEY_GENERATOR=remote AUDYSSEY_GENERATOR_URL=http://127.0.0.1:8701 uv run audyssey serve
```

- 기본 설정: `large_44k`, bf16, Euler 25회, CFG 4.5, 부정 프롬프트 없음. 설명만 사용하며 계획 구간·시각 힌트로 자르지 않습니다.
- 10초까지 단일 생성, 그보다 길면 8초 창을 잇습니다. 마지막 창은 끝에 맞추므로 실제 겹침이 1초보다 길 수 있으며 전체 겹침을 선형 크로스페이드합니다. 44.1kHz 결과를 48kHz 모노로 리샘플하고 `round(duration * 48000)` 샘플에 맞춥니다.
- CLAP 텍스트 TargetSel은 자기 설명이 가장 가깝고 margin이 임계값 이상이면 합격합니다. 형제가 없으면 margin 없이 합격합니다.
- 대체 생성은 같은 job의 분리 stem 전체를 비교하여 `ref_correct`와 `ref_margin`도 기록합니다. 비교할 다른 분리 stem이 없으면 `ref_margin`은 생략합니다. 참조 점수는 합격 판정에 쓰지 않습니다.
- 실행 중 다운로드를 막기 위해 real 생성·채점은 `HF_HUB_OFFLINE=1`을 필수로 요구합니다. 모델은 첫 호출 뒤 재사용하며 GPU 호출은 한 번에 하나씩 실행합니다.
- 8초/10초 품질·시간·메모리와 6개 컨테이너 E2E는 측정 전입니다. `GENERATION_MARGIN_THRESHOLD`도 보정 전입니다.

## 플래너 연결 (Gemini, vLLM)

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

## LangSmith

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

## 가중치 준비

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

**생성 가중치** (서버 실행 전 운영자가 받음): 공개 베이스는 승인 없이 받을 수 있습니다. 비공개 FT 저장소는 `sogang-audyssey` 조직 멤버에게만 열리며, 팀원 각자 위 로그인 단계에서 자신의 읽기 토큰을 사용합니다. 토큰과 FT 가중치는 레포·공개 이미지에 넣지 않습니다.

```bash
export HF_HOME=/home/common/huggingface
H="uvx --from huggingface_hub[cli]<1.0 huggingface-cli download"
$H jnwnlee/SelVA weights/video_enc_sup_5.pth ext_weights/v1-44.pth ext_weights/synchformer_state_dict.pth
$H sogang-audyssey/selva-ft-selsync generator/S_O_s43/model_ema_step1000.pth encoder/S_O/best.pt
$H apple/DFN5B-CLIP-ViT-H-14-384
$H google/flan-t5-base
$H nvidia/bigvgan_v2_44khz_128band_512x
$H lukewys/laion_clap 630k-best.pt
$H roberta-base
```

공식 생성기 `weights/generator_large_44khz_sup_5.pth`는 사용하지 않습니다. FT 생성기(EMA)와 SelSync 인코더만 덮어씁니다. SelVA 코드는 라이선스 표시가 없어 복사하지 않고 고정 커밋 의존성으로 설치하며, 가중치 라이선스는 CC-BY-NC-SA-4.0입니다.

| 모델 | 설정 | 캐시에 필요한 것 |
|---|---|---|
| 분리 | `AUDYSSEY_SAM_AUDIO_MODEL` | `facebook/sam-audio-large-tv`, `t5-base` |
| 필터링 | `AUDYSSEY_SAM_AUDIO_JUDGE_MODEL` | `facebook/sam-audio-judge`의 `sam_audio` revision 전체, `main` revision의 JSON(설정·토크나이저, 약 2MB) |
| 그라운딩 | `AUDYSSEY_SAM3_MODEL` | `facebook/sam3.1`의 `config.json`, `sam3.1_multiplex.pt` |
| 생성 | `AUDYSSEY_SELVA_BASE_MODEL` / `AUDYSSEY_SELVA_FT_MODEL` | 위 베이스 3개 + FT 2개, DFN5B CLIP·flan-t5·BigVGAN·CLAP·roberta-base |
| 후보 고르기 (선택) | 후보 2개 이상 | `lukewys/laion_clap`의 `630k-best.pt`, `roberta-base`, `bert-base-uncased`, `facebook/bart-base` |
| 플래너 (선택) | `AUDYSSEY_VLLM_MODEL` | `Qwen/Qwen2.5-VL-3B-Instruct` |

- 승인을 기다리는 동안에는 비공식 미러 `mrfakename/sam-audio-large-tv`, `mrfakename/sam-audio-judge`를 모델 ID로 쓸 수 있습니다. 미러도 사용 권한을 확인해야 하고, 미러 Judge는 `sam_audio` 브랜치의 `checkpoint.pt`가 캐시에 있어야 합니다.
- sv3-a100 공용 캐시 `/home/common/huggingface`에는 공식 모델과 미러가 모두 있습니다. 가중치를 복사하지 말고 이 경로를 그대로 연결합니다.

## 설정 (환경변수)

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

Compose에서만 쓰는 값(`.env`): `HF_CACHE`, `AUDYSSEY_{SEPARATION,JUDGE,GROUNDING,GENERATION,VLLM}_GPU`, `AUDYSSEY_VLLM_MODEL`,
`AUDYSSEY_VLLM_GPU_UTIL`, `GOOGLE_API_KEY`, `LANGSMITH_*`.

**생성**

| 환경변수 | 기본값 | 설명 |
|---|---|---|
| `AUDYSSEY_GENERATOR` / `AUDYSSEY_GENERATOR_URL` | `stub` / 빈 값 | 생성과 채점 도구. `stub`, `real`, `remote`. Compose 기본 `remote` / `http://generation:8700`, remote는 URL 필수 |
| `AUDYSSEY_SELVA_SEED` | `42` | 첫 생성 seed. 다시 생성할 때 누적 attempts로 증가 |
| `AUDYSSEY_MAX_GENERATION_RETRIES` | `2` | 루프마다 최대 재시도 수(0 이상). 총 시도는 1 + 이 값 |
| `AUDYSSEY_GENERATION_MARGIN_THRESHOLD` | `0.0` | 텍스트 TargetSel 합격 임계값. 보정 전(stub은 진폭으로 판정) |
| `AUDYSSEY_SELVA_BASE_MODEL` | `jnwnlee/SelVA` | 베이스 저장소 |
| `AUDYSSEY_SELVA_FT_MODEL` | `sogang-audyssey/selva-ft-selsync` | 비공개 FT 저장소 |
| `AUDYSSEY_SELVA_FT_GENERATOR_FILE` | `generator/S_O_s43/model_ema_step1000.pth` | FT 생성기 파일 |
| `AUDYSSEY_SELVA_FT_ENCODER_FILE` | `encoder/S_O/best.pt` | SelSync 인코더 파일 |
| `AUDYSSEY_SELVA_DEVICE` | `cuda` | 생성 장치 |
| `AUDYSSEY_SELVA_STEPS` / `AUDYSSEY_SELVA_CFG` | `25` / `4.5` | Euler 생성 설정 |
| `AUDYSSEY_SELVA_SINGLE_MAX_S` / `AUDYSSEY_SELVA_WINDOW_S` | `10` / `8` | 단일 생성 최대 길이 / 긴 영상 창 길이(초) |
| `AUDYSSEY_GENERATION_GPU` | `0` | Compose 생성 서비스 GPU 번호 |
