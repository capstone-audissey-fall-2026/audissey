# Audyssey SAM Audio 연결 + Docker 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:**
- 분리 real을 SAM Audio로, 필터링 real 예시를 SAM Audio Judge로 구현한다.
- GPU 도구 서버용 설치 환경을 따로 둔다.
- main/GPU Docker 이미지, compose, GHCR 자동 업로드를 만든다.
- 팀원이 sv3-a100(compose)과 RunPod(Pod 하나)에서 재현할 수 있게 한다.

**Architecture:**
- **환경 분리:** SAM Audio가 끌고 오는 `laion-clap==1.1.6`이 `numpy<2`를 요구한다. 그래서 GPU 도구 서버는 별도 uv 프로젝트 `gpu/`(Python 3.12, numpy 1.26, sam-audio, audyssey는 경로 의존성)로 설치한다. main 서버와 CI는 지금 환경(Python 3.13, numpy 2.x)을 그대로 쓴다.
- **어댑터:** `separation/real.py`와 `filtering/real.py`는 `torch`와 `sam_audio`를 함수 안에서 `importlib.import_module`로 불러온다. 그래서 GPU 의존성이 없는 CI에서도 import·타입 검사·테스트가 돈다.
- **이미지:**
  - `audyssey-main`: Python 3.13, ffmpeg, 우리 코드
  - `audyssey-sam-audio`: CUDA, Python 3.12, `gpu/` 환경
  - SAM Audio와 Judge는 같은 라이브러리를 쓰므로 한 이미지다.
    - 도구별 컨테이너(compose): 같은 이미지로 실행 명령만 바꿔 두 개 띄운다.
    - RunPod Pod 하나: 이미지 기본 명령이 도구 서버 하나에 두 도구를 함께 띄운다.
  - 설치 환경이 다른 도구(SAM3 등)가 생기면 그때 "합친 이미지"(환경 여러 개 + 시작 스크립트)를 추가한다. 이번 범위가 아니다.

**Tech Stack:** sam-audio (facebookresearch/sam-audio), torch, uv, Docker, docker compose, GitHub Actions + GHCR

**Spec:** `docs/superpowers/specs/2026-11-09-audyssey-skeleton-design.md`

## Global Constraints

- **고정할 SAM Audio 버전:** `sam-audio @ git+https://github.com/facebookresearch/sam-audio@bb4c6999d2677c7402360e426afc01ddfad6dce0`
- **의존성 범위:**
  - 남의 패키지 제약은 덮어쓰지 않는다. `override-dependencies`, `constraint-dependencies`, 강제 핀을 쓰지 않는다.
  - 고칠 수 있는 것은 **우리 패키지의 최소 버전**뿐이다.
    - 예: audyssey의 `numpy>=2.4.3`을 `numpy>=1.26`으로 낮춘다.
    - 필요하면 scipy 등 우리 최소 버전도 낮춘다.
    - 낮춘 버전에서 테스트가 통과하는 것을 CI로 증명한다.
  - `gpu/`가 그래도 풀리지 않으면 멈추고 보고한다.
- **main 환경:** core 의존성과 루트 `uv.lock`, CI의 기존 `uv sync` 조합(`--extra ui --extra llm`)은 numpy 최소 버전 외에는 바꾸지 않는다. CI는 torch를 설치하지 않는다.
- **CI 기준을 그대로 지킨다.**
  - 명령: `uv run ruff check src/`, `uv run ruff format --check src/`, `uv run ty check --error-on-warning src/`, `uv run python -m unittest discover tests`
  - 검사를 끄거나 건너뛰는 것(`ty: ignore`, `type: ignore`, `noqa`, `skip`, 약해진 assert, CI 설정 완화)은 금지다.
  - GPU 없는 CI 테스트는 가짜 `torch`/`sam_audio` 모듈을 `sys.modules`에 넣어 우리 어댑터만 검증한다.
- **가중치를 받지 않는다.**
  - 이미지에 가중치를 넣지 않는다. HF 캐시는 볼륨으로 연결한다.
  - sv3-a100에서는 `/home/common/huggingface`를 **읽기 전용**으로 연결한다.
  - GPU 실행은 항상 `HF_HUB_OFFLINE=1`로 한다.
  - 모델 ID는 환경변수로 준다: `AUDYSSEY_SAM_AUDIO_MODEL=mrfakename/sam-audio-large-tv`, `AUDYSSEY_SAM_AUDIO_JUDGE_MODEL=mrfakename/sam-audio-judge`
  - 코드 기본값은 공식 ID 그대로 둔다.
- **GPU 사용:** `CUDA_VISIBLE_DEVICES=0`(Docker는 `device_ids: ["0"]`)만 쓴다. 다른 GPU는 다른 사람이 쓰고 있다.
- **데이터 계약:** 48 kHz float32 mono wav, 입력과 같은 길이다.
- **시각 힌트(`visual_hint`, SAM3)는 이번 범위가 아니다.** `masked_videos`는 넘기지 않는다.
- **커밋과 브랜치:**
  - 커밋은 `type(scope): lowercase subject` 형식으로 최소 단위로 나눈다. 끝에 빈 줄과 `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`를 붙인다.
  - 브랜치는 `feat/sam-audio`이고, 이미 있으며 계획 커밋 `a60cea0`이 올라가 있다.
  - 머지와 force push는 하지 않는다.
- **막히면 우회하지 않는다.** 샌드박스 때문에 GPU나 Docker 명령이 막히면 그 명령과 오류를 Deviations에 적고 다음 Task로 넘어간다. 실측과 이미지 빌드는 사람이 대신 한다.

## Review Focus

1. **spans 처리:** plan에 spans가 있으면 anchors를 `[["+", start, end], ...]`로 넘긴다. spans가 없으면 anchors는 `None`이고, `predict_spans`는 설정값(기본 `False`)을 따른다.
2. **출력 길이:** 모델 출력 길이가 입력과 다르면 입력 길이에 맞춘다(자르거나 0으로 채운다). 결과는 1채널 48 kHz다.
3. **재시도 구분:** `attempt`를 시드로 쓴다(`torch.manual_seed(attempt)`). Judge는 `suggestion`을 주지 않으므로, 재시도할 때 결과가 달라지게 하기 위해서다.
4. **모델 로딩과 동시 호출:** 모델 로딩은 프로세스당 1회다(`functools.cache`). 동시 호출은 모듈 잠금 하나로 한 번에 하나씩 처리한다.
5. **GPU 의존성이 없을 때:** real 도구를 처음 부를 때 `ModuleNotFoundError`가 나면 `RuntimeError("SAM Audio is not installed: install the gpu/ environment")`로 바꿔 알린다. import 시점이 아니라 호출 시점이므로 CI와 stub 모드는 영향받지 않는다.
6. **compose 기본 동작:**
   - 도구 서버 포트는 기본적으로 호스트에 공개하지 않는다. compose 내부 네트워크만 쓴다.
   - main은 `127.0.0.1:8600`에만 연결한다.
   - 토큰은 `.env`에서 읽는다. `.env.example`만 커밋한다.

---

### Task 0: 계획 갱신 커밋

- [ ] **Step 1:** `git switch feat/sam-audio`로 이동한다. 작업 트리가 깨끗하고 이 계획 파일만 바뀌었는지 확인한다.
- [ ] **Step 2:** 커밋한다: `docs: revise sam audio plan for gpu env and docker`

### Task 1: numpy 최소 버전 낮추기 + 두 버전 CI

**Files:** Modify `pyproject.toml`, `uv.lock`, `.github/workflows/ci-lint.yaml`

- [ ] **Step 1:** audyssey의 `numpy>=2.4.3`을 `numpy>=1.26`으로 낮춘다.
  - scipy 등 다른 우리 최소 버전 때문에 numpy 1.26과 Python 3.12 조합이 안 되면, 그 최소 버전도 필요한 만큼만 낮춘다.
  - `uv lock`을 돌린다. 루트 lock에서는 Python 3.13이 numpy 2.x로 풀리는 것이 정상이다.
- [ ] **Step 2:** 로컬에서 Python 3.12 + numpy 1.26으로 전체 테스트를 돌린다.
  - 예: `uv run --isolated --python 3.12 --with 'numpy==1.26.4' --extra ui --extra llm python -m unittest discover tests`. 정확한 명령은 실제로 동작하는 것으로 정한다.
  - 실패하면 우리 코드를 고친다.
- [ ] **Step 3:** CI에 `numpy-1x` 작업을 추가한다. 위 명령으로 Python 3.12 + numpy 1.26에서 unittest를 돌린다. 기존 작업은 그대로 둔다.
- [ ] **Step 4:** 커밋을 둘로 나눈다.
  - `build: allow numpy 1.26 for the gpu tool environment`
  - `ci: run unit tests on python 3.12 with numpy 1.26`

### Task 2: GPU 도구 서버 환경 `gpu/`

**Files:** Create `gpu/pyproject.toml`, `gpu/uv.lock`, `gpu/README.md`(짧게)

- [ ] **Step 1:** `gpu/pyproject.toml`을 만든다.
  - `name = "audyssey-gpu"`, `requires-python = ">=3.11,<3.13"`
  - `dependencies = ["audyssey[ui]", "sam-audio @ git+...@bb4c699..."]`
  - `[tool.uv.sources] audyssey = { path = "..", editable = true }`
  - `package = false`(배포용이 아닌 환경 정의)
- [ ] **Step 2:** `cd gpu && uv lock`을 돌린다. 풀리지 않으면 Global Constraints대로 멈추고 보고한다.
- [ ] **Step 3:** `uv sync --project gpu`를 돌린다. 이어서 `uv run --project gpu python -c "import sam_audio, torch, audyssey; print(torch.__version__, torch.cuda.is_available())"`로 import를 확인한다.
- [ ] **Step 4:** 커밋한다: `build: add gpu tool server environment`

### Task 3: 설정

**Files:** Modify `src/audyssey/settings.py`; Test `tests/test_audyssey_settings.py`

**Produces:**
- `sam_audio_device: str = "cuda"`
- `sam_audio_reranking_candidates: int = Field(default=1, ge=1)`
- `sam_audio_predict_spans: bool = False`
- `judge_pass_threshold: float = 0.5`. 예시 기본값이며, Task 9 실측 후 조정한다.

- [ ] **Step 1:** 실패하는 테스트를 쓴다: `test_sam_audio_settings_from_env`
- [ ] **Step 2:** 실행해서 실패하는지 본다.
- [ ] **Step 3:** 설정을 추가한다.
- [ ] **Step 4:** 실행해서 통과하는지 본다.
- [ ] **Step 5:** 커밋한다: `feat(settings): add sam audio runtime settings`

### Task 4: 분리 real (SAM Audio)

**Files:** Modify `src/audyssey/tools/separation/real.py`; Test `tests/test_audyssey_sam_separation.py`

**Produces:** `separate(mono, plan, video, out, attempt) -> Path`. 시그니처는 그대로다.

- `_load()`: `@cache`로 한 번만 실행한다.
  - `import_module("torch")`, `import_module("sam_audio")`로 불러온다.
  - `SAMAudio.from_pretrained(settings.sam_audio_model)`을 `.to(device).eval()`로 올리고, `SAMAudioProcessor.from_pretrained(...)`를 만든다.
  - `ModuleNotFoundError`는 Review Focus 5대로 바꿔 낸다.
- 처리 흐름:
  1. `read_wav(mono)`
  2. `(1, n)` 텐서로 바꾼다.
  3. `processor(descriptions=[plan.description], audios=[tensor], anchors=...)`
  4. `batch.to(device)`
  5. `torch.inference_mode()` 안에서 `torch.manual_seed(attempt)`를 하고 `model.separate(batch, reranking_candidates=..., predict_spans=...)`를 부른다.
  6. `result.target[0]`을 numpy로 바꾸고 길이를 맞춘 뒤 `write_wav(out, ...)`로 쓴다.
- 정확한 API는 `gpu/` 환경에 설치된 고정 커밋의 소스에서 확인한다.
  - `/home/judejiwoo/_AAAI27_Multitrack_V2A_System/sam-audio/examples/*.ipynb`는 읽기 전용 참고 자료다. 로컬 수정이 있을 수 있으므로 판단 기준으로 삼지 않는다.
- 잠금: 모듈 전역 `threading.Lock`. 주석: `# ponytail: one GPU model per process, one call at a time; batch or queue if throughput matters`

- [ ] **Step 1: 실패하는 테스트.** 가짜 `torch`/`sam_audio`를 `sys.modules`에 넣는다. 가짜 모델은 입력을 0.5배 해서 돌려주고, 받은 인자를 기록한다.
  - `test_separate_passes_description_and_span_anchors`
  - `test_separate_without_spans_uses_no_anchors`
  - `test_separate_writes_same_length_mono_48k`: 출력이 더 길거나 짧아도 입력 길이로 맞춰진다.
  - `test_attempt_sets_seed`
  - `test_model_loaded_once`: 두 번 불러도 `from_pretrained`는 1회만 불린다. 테스트마다 `_load.cache_clear()`를 한다.
  - `test_missing_gpu_env_message`
- [ ] **Step 2:** 실행해서 실패하는지 본다.
- [ ] **Step 3:** 구현한다.
- [ ] **Step 4:** 실행해서 통과하는지 본다. 기존 테스트도 모두 통과해야 한다.
- [ ] **Step 5:** 커밋한다: `feat(separation): separate sources with sam audio`

### Task 5: 필터링 real 예시 (SAM Audio Judge)

**Files:** Modify `src/audyssey/tools/filtering/real.py`; Test `tests/test_audyssey_sam_judge.py`

**Produces:** `judge(mono, track, plan) -> FilterResult`
- 로딩은 Task 4와 같은 패턴이다. 모델은 `SAMAudioJudgeModel`, processor는 `SAMAudioJudgeProcessor`, 설정은 `settings.sam_audio_judge_model`.
- 처리: processor로 `(text=[desc], input_audio=[mono], separated_audio=[track])`를 만들고, `inference_mode` 안에서 모델을 부른다.
- 결과:
  - `scores`: `overall`, `recall`, `precision`, `faithfulness` 중 값이 있는 것을 float로 담는다.
  - `passed`: `overall >= settings.judge_pass_threshold`
  - `reason`: 실패하면 `f"judge overall {x:.2f} < {threshold}"`
  - `suggestion`: `None`
- 모듈 docstring에 "윤석 담당 모듈의 예시 구현. 기준과 구현은 교체 가능"이라고 적는다.

- [ ] **Step 1: 실패하는 테스트** (가짜 모듈 사용)
  - `test_judge_maps_scores_and_threshold`: overall 0.7, 기준 0.5 → 통과. overall 0.3 → 실패이고 reason에 `0.30`이 들어간다.
  - `test_judge_passes_description_and_both_audios`
- [ ] **Step 2:** 실행해서 실패하는지 본다.
- [ ] **Step 3:** 구현한다.
- [ ] **Step 4:** 실행해서 통과하는지 본다.
- [ ] **Step 5:** 커밋한다: `feat(filtering): add sam audio judge example`

### Task 6: 도구 서버 하나로 여러 도구 띄우기

**Files:** Modify `src/audyssey/toolserver.py`, `src/audyssey/cli.py`; Test `tests/test_audyssey_toolserver.py`, `tests/test_audyssey_cli.py`

**Produces:**
- `create_tool_app(tools: list[Literal["separation", "filtering"]], impl, token) -> FastAPI`
  - 받은 도구들의 엔드포인트를 모두 연다.
  - `/health`는 `{"tools": [...], "impl": impl}`를 돌려준다.
  - 기존 단일 도구 호출부와 테스트는 리스트로 옮긴다.
- CLI: `audyssey tool-server --tool separation --tool filtering ...`처럼 `--tool`을 여러 번 받는다. 최소 1개다.

- [ ] **Step 1: 실패하는 테스트**
  - `test_multi_tool_app_serves_both`: 하나의 앱에서 `/separate`와 `/judge`가 모두 200이다(stub).
  - `test_tool_server_accepts_repeated_tool_option`
- [ ] **Step 2:** 실행해서 실패하는지 본다.
- [ ] **Step 3:** 구현한다.
- [ ] **Step 4:** 실행해서 통과하는지 본다. 기존 원격 E2E 테스트도 통과해야 한다.
- [ ] **Step 5:** 커밋한다: `feat(toolserver): serve multiple tools from one process`

### Task 7: GPU 실측 스크립트

**Files:** Create `scripts/sam_audio_smoke.py`

**Produces:** `uv run --project gpu python scripts/sam_audio_smoke.py <video> --description "..." [--span START END] --out <dir>`
- real `separate` 1회, real `judge` 1회를 부른다.
- 출력:
  - 로딩 시간과 호출당 시간
  - `torch.cuda.max_memory_allocated()`(GB)
  - torch, CUDA 버전
  - judge 점수
  - 결과 wav 경로

- [ ] **Step 1:** 작성한다. `uv run ruff check scripts/`와 `uv run ruff format --check scripts/`가 통과해야 한다.
- [ ] **Step 2:** 커밋한다: `feat(scripts): add sam audio gpu smoke script`

### Task 8: Docker 이미지와 compose

**Files:** Create `docker/main.Dockerfile`, `docker/sam-audio.Dockerfile`, `compose.yaml`, `.env.example`, `.dockerignore`

**`docker/main.Dockerfile`:**
- 베이스는 `python:3.13-slim`이다. ffmpeg를 설치한다.
- uv로 `--extra ui --extra llm`을 설치한다(`--locked`, dev 제외).
- `EXPOSE 8600`, `CMD ["audyssey", "serve", "--host", "0.0.0.0", "--port", "8600"]`

**`docker/sam-audio.Dockerfile`:**
- 베이스는 torch 버전에 맞는 CUDA runtime 이미지(예: `nvidia/cuda:12.x-cudnn-runtime-ubuntu22.04`)다. 드라이버 570(CUDA 12.8)에서 돌아야 한다.
- uv로 Python 3.12를 받고, `gpu/` 환경을 `--locked`로 설치한다. ffmpeg를 설치한다.
- `ENV HF_HOME=/hf HF_HUB_OFFLINE=1`
- 기본 명령은 Pod 하나용이다: `CMD ["audyssey", "tool-server", "--tool", "separation", "--tool", "filtering", "--impl", "real", "--host", "0.0.0.0", "--port", "8700"]`
- 토큰이 없으면 기존 CLI 규칙대로 시작을 거부한다.

**`compose.yaml`:**
- 서비스 3개:
  - `main`: `audyssey-main`
  - `separation`: `audyssey-sam-audio`, 명령은 `--tool separation`
  - `judge`: 같은 이미지, 명령은 `--tool filtering`
- 각 서비스에 `image: ghcr.io/capstone-audissey-fall-2026/audyssey-<name>:latest`와 `build:`를 둘 다 적는다. 받거나 직접 빌드하거나 둘 다 된다.
- `main`:
  - `ports: ["127.0.0.1:8600:8600"]`
  - `env_file: .env`
  - 기본 연결 주소: `AUDYSSEY_SEPARATOR=remote`, `AUDYSSEY_SEPARATOR_URL=http://separation:8700`, `AUDYSSEY_FILTER=remote`, `AUDYSSEY_FILTER_URL=http://judge:8700`
  - 이 값들은 `.env`로 덮어쓸 수 있게 한다(RunPod 주소).
  - 작업 폴더는 볼륨으로 둔다.
- GPU 서비스:
  - `deploy.resources.reservations.devices`에 `driver: nvidia`, `device_ids: ["${AUDYSSEY_GPU:-0}"]`, `capabilities: [gpu]`
  - HF 캐시는 `${HF_CACHE:-~/.cache/huggingface}:/hf:ro`
  - 포트는 공개하지 않는다.
- `docker compose up`은 전부 띄운다. `docker compose up main`은 main만 띄운다.

**`.env.example`:** `AUDYSSEY_TOOL_TOKEN`, 모델 ID 2개, `HF_CACHE`, `AUDYSSEY_GPU`, 원격 주소 예시(주석 처리)

- [ ] **Step 1:** 파일을 작성한다. `docker compose config`가 오류 없이 끝나는지 확인한다.
- [ ] **Step 2:** `docker build -f docker/main.Dockerfile .`를 실행한다. 이어서 컨테이너를 stub 모드로 띄워 `/api/jobs`가 응답하는지 확인한다.
- [ ] **Step 3:** `docker build -f docker/sam-audio.Dockerfile .`를 실행하고, 이미지 크기를 기록한다.
- [ ] **Step 4:** Docker가 샌드박스에 막히면 Global Constraints대로 기록하고 넘어간다.
- [ ] **Step 5:** 커밋을 셋으로 나눈다.
  - `build(docker): add main server image`
  - `build(docker): add sam audio tool server image`
  - `build(docker): add compose for main and gpu tool servers`

### Task 9: sv3-a100 실측과 E2E

- [ ] **Step 1: 스모크.** `uv run --project gpu`로 실행한다. span 없이 한 번, `--span 0 3`을 붙여 한 번 돌린다. 환경변수는 Global Constraints대로 준다.
- [ ] **Step 2: compose E2E.**
  - `.env`에 토큰과 모델 ID, `HF_CACHE=/home/common/huggingface`를 넣고 `docker compose up`을 실행한다.
  - `samples/sample.mp4`를 업로드한다. 작업이 `ready`가 되고, 판정 점수가 기록되며, render가 200으로 끝나야 한다.
  - 소스별 시도 횟수와 통과 여부를 기록한다.
- [ ] **Step 3: 결과 기록.** 설계 문서 §13에 실측 표를 적는다.
  - GPU 메모리, 10초 기준 처리 시간, 로딩 시간
  - torch, CUDA, 드라이버(570.124.06) 버전
  - 이미지 크기, Judge 점수 범위, 정한 `judge_pass_threshold`
  - 기준값을 바꿨다면 Task 3 테스트와 함께 고친다.
- [ ] **Step 4:** 샌드박스에 막히면 기록하고 넘어간다. 실측은 사람이 대신 한다.

### Task 10: GHCR 자동 업로드

**Files:** Create `.github/workflows/publish-images.yaml`

- 트리거는 `push` to `main`과 `workflow_dispatch`다. 두 이미지를 빌드해 `ghcr.io/capstone-audissey-fall-2026/audyssey-main`과 `audyssey-sam-audio`에 `latest`와 커밋 SHA 태그로 올린다.
- 권한은 `packages: write`, 인증은 `GITHUB_TOKEN`이다.
- sam-audio 이미지는 크므로 빌드 전에 러너 디스크를 정리한다(사용하지 않는 툴체인 삭제).
- PR에서는 돌지 않는다. 필수 검사에도 넣지 않는다.

- [ ] **Step 1:** 작성한다. YAML 문법을 확인한다(`python -c "import yaml; yaml.safe_load(open(...))"` 등 이미 있는 도구로).
- [ ] **Step 2:** 커밋한다: `ci: publish main and sam audio images to ghcr`

### Task 11: 문서, push, PR

**Files:** Modify `README.md`, `docs/superpowers/specs/2026-11-09-audyssey-skeleton-design.md`, `gpu/README.md`

- [ ] **Step 1: README 팀 가이드**
  - **"한 GPU 서버에서 전부 띄우기 (sv3-a100 등 Docker 되는 서버)"**: `.env` 작성 → `docker compose up` → `localhost:8600`
  - **"노트북 + RunPod"**:
    - RunPod Pod 설정값: 이미지 `ghcr.io/.../audyssey-sam-audio:latest`, 포트 8700, 환경변수(토큰, 모델 ID, `HF_HOME`), 네트워크 볼륨
    - 템플릿 저장 방법
    - 노트북 `.env`에 Pod 주소 넣기 → `docker compose up main`
    - RunPod GPU Pod 안에서는 compose를 쓸 수 없다는 점
  - **"가중치"**: sv3-a100은 공용 캐시를 읽기 전용으로 쓴다. RunPod는 네트워크 볼륨에 받아 둔다. HF 승인 또는 미러가 필요하다는 점.
- [ ] **Step 2: 설계 문서**
  - §7.1에 real 구현 상태를 적는다.
  - §10에 이미지 2개, compose, RunPod 운영 방식을 적는다.
  - §12에 결정을 추가한다(2026-11-10).
    - CI는 GPU 의존성 없이 가짜 모듈로 검증한다.
    - GPU 환경은 `gpu/`로 분리한다(numpy 충돌 때문).
    - 이미지는 설치 환경 단위로 만든다. 도구별 컨테이너는 compose로, RunPod는 이미지 하나 = Pod 하나다.
    - 설치 환경이 다른 도구가 생기면 합친 이미지를 추가한다.
    - 도구 서버는 HTTP 창구가 없는 모델(SAM 계열)용이다. LLM은 Gemini 또는 vLLM 공식 이미지로 따로 둔다.
  - §13에 SAM3 그라운딩(다음 PR)을 적는다.
- [ ] **Step 3:** CI 명령 4개와 numpy 1.26 테스트를 다시 통과시킨다.
- [ ] **Step 4:** 커밋한다.
  - `docs(readme): add docker, runpod and sam audio guide`
  - `docs(spec): record sam audio, gpu env and docker decisions`
- [ ] **Step 5:** `git push -u origin feat/sam-audio`, 이어서 `gh pr create --base main`.
  - PR 본문에 실측 표(또는 "실측 대기")를 넣는다.
  - 끝에 `🤖 Generated with [Claude Code](https://claude.com/claude-code)`를 넣는다.
  - 머지하지 않는다.
