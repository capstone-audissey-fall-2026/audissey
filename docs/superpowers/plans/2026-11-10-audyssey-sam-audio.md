# Audyssey SAM Audio 연결 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 분리 도구의 real 구현을 SAM Audio로 만든다. 필터링 real에는 SAM Audio Judge 연결 예시를 넣는다. 둘 다 같은 컴퓨터(`real`)에서도, 도구 서버(`remote`)에서도 그대로 돈다.

**Architecture:**
- `separation/real.py`와 `filtering/real.py`는 기존 도구 함수 시그니처를 그대로 지킨다.
- `torch`와 `sam_audio`는 함수 안에서 `importlib.import_module`로 불러온다. 그래서 CI처럼 GPU 의존성이 없는 환경에서도 import·타입 검사·테스트가 돈다.
- 모델은 프로세스당 한 번만 GPU에 올리고, 호출은 잠금으로 한 번에 하나씩만 실행한다.

**Tech Stack:** sam-audio (facebookresearch/sam-audio, `[gpu]` extra), torch, numpy

**Spec:** `docs/superpowers/specs/2026-11-09-audyssey-skeleton-design.md` (§4.2 분리 조건, §7.1 도구 함수, §13 실측 항목)

## Global Constraints

- **의존성:** 새 의존성은 `[gpu]` extra 하나다.
  - `sam-audio @ git+https://github.com/facebookresearch/sam-audio@bb4c6999d2677c7402360e426afc01ddfad6dce0`
  - torch 등은 이것이 끌고 온다.
  - core 의존성과 CI의 `uv sync` extras(`ui`, `llm`)는 바꾸지 않는다. CI는 `[gpu]`를 설치하지 않는다.
- **`uv lock`이 `[gpu]` extra를 풀지 못하면 멈춘다.** 의존성을 억지로 고정하거나 다른 패키지로 대체하지 말고, 멈추고 보고한다.
- **CI 기준을 그대로 지킨다.**
  - 명령: `uv run ruff check src/`, `uv run ruff format --check src/`, `uv run ty check --error-on-warning src/`, `uv run python -m unittest discover tests`
  - 검사를 끄거나 건너뛰는 것(`ty: ignore`, `noqa`, `skip`, 약해진 assert, CI 설정 완화)은 금지다.
  - GPU 없는 CI에서 도는 테스트는 가짜 `torch`/`sam_audio` 모듈로 우리 어댑터 코드만 검증한다(`mock.patch.dict(sys.modules, ...)`).
- **가중치를 받지 않는다.**
  - sv3-a100 공용 캐시(`/home/common/huggingface`)의 미러를 읽기만 하고, 복사하지 않는다.
  - GPU 실행은 항상 `HF_HUB_OFFLINE=1`로 한다.
  - 모델 ID는 환경변수로 준다: `AUDYSSEY_SAM_AUDIO_MODEL=mrfakename/sam-audio-large-tv`, `AUDYSSEY_SAM_AUDIO_JUDGE_MODEL=mrfakename/sam-audio-judge`
  - 코드 기본값은 공식 ID(`facebook/...`)를 그대로 둔다.
- **GPU는 `CUDA_VISIBLE_DEVICES=0`만 쓴다.** 다른 GPU는 다른 사람이 쓰고 있다.
- **데이터 계약:** 48 kHz float32 mono wav, 원본과 같은 길이. SAM Audio도 48 kHz가 기본이라 리샘플은 필요 없지만, 길이는 맞춘다(자르거나 0으로 채운다).
- **시각 힌트(`visual_hint`, SAM3 마스크)는 이번 범위가 아니다.** `masked_videos`는 넘기지 않는다.
- **커밋과 브랜치:**
  - 커밋은 `type(scope): lowercase subject` 형식으로 최소 단위로 나눈다. 끝에 빈 줄과 `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`를 붙인다.
  - 브랜치는 `feat/sam-audio`(main에서 시작)이다.
  - 머지와 force push는 하지 않는다.

## Review Focus

1. **spans 처리:** plan에 spans가 있으면 anchors는 `[["+", start, end], ...]`로 넘긴다. spans가 없으면 anchors는 `None`이고, `predict_spans`는 설정값을 따른다(기본 `False`).
2. **출력 길이:** 모델 출력 길이가 입력과 다르면 입력 길이에 맞춘다. 결과 wav는 1채널 48 kHz다.
3. **재시도 구분:** `attempt`를 시드로 쓴다(`torch.manual_seed(attempt)`). Judge가 `suggestion`을 주지 않아 같은 설명으로 다시 분리할 때도 결과가 달라지게 하기 위해서다.
4. **모델 로딩 횟수:** 프로세스당 1회다(`functools.cache`). 동시 호출은 모듈 잠금 하나로 한 번에 하나씩 처리한다.
5. **GPU 의존성이 없을 때:** `real` 도구를 처음 부를 때 `ModuleNotFoundError`가 나면 `RuntimeError("SAM Audio is not installed: uv sync --extra gpu")`로 바꿔 알린다. import 시점이 아니라 호출 시점이므로 CI와 stub 모드는 영향받지 않는다.

---

### Task 0: 브랜치와 의존성

**Files:** Modify `pyproject.toml`, `uv.lock`

- [ ] **Step 1:** `git switch main && git pull --ff-only && git switch -c feat/sam-audio`를 실행한다. 이 계획 파일을 커밋한다: `docs: add sam audio implementation plan`
- [ ] **Step 2:** `[project.optional-dependencies]`에 `gpu = ["sam-audio @ git+https://github.com/facebookresearch/sam-audio@bb4c6999d2677c7402360e426afc01ddfad6dce0"]`를 추가하고 `uv lock`을 돌린다.
  - 풀리지 않으면 Global Constraints대로 멈추고 보고한다.
  - 풀리면 `uv sync --dev --locked --extra ui --extra llm`(CI와 같은 조합)이 GPU 패키지 없이 끝나는지 확인한다.
- [ ] **Step 3:** 커밋한다: `build: add sam-audio gpu extra`

### Task 1: 설정

**Files:** Modify `src/audyssey/settings.py`; Test `tests/test_audyssey_settings.py`

**Produces:**
- `sam_audio_device: str = "cuda"`
- `sam_audio_reranking_candidates: int = Field(default=1, ge=1)`
- `sam_audio_predict_spans: bool = False`
- `judge_pass_threshold: float = 0.5`. 예시 기본값이며, Task 5 실측 후 조정한다.

- [ ] **Step 1:** 실패하는 테스트를 쓴다: `test_sam_audio_settings_from_env`(환경변수 4개를 넣으면 같은 값이 나온다).
- [ ] **Step 2:** 실행해서 실패하는지 본다.
- [ ] **Step 3:** 설정을 추가한다.
- [ ] **Step 4:** 실행해서 통과하는지 본다.
- [ ] **Step 5:** 커밋한다: `feat(settings): add sam audio runtime settings`

### Task 2: 분리 real (SAM Audio)

**Files:** Modify `src/audyssey/tools/separation/real.py`; Test `tests/test_audyssey_sam_separation.py`

**Produces:** `separate(mono, plan, video, out, attempt) -> Path`. 시그니처는 그대로다.

- `_load() -> tuple[model, processor, torch]`: `@cache`로 한 번만 실행한다.
  - `import_module("torch")`, `import_module("sam_audio")`로 불러온다.
  - `SAMAudio.from_pretrained(settings.sam_audio_model)`을 `.to(device).eval()`로 올리고, `SAMAudioProcessor.from_pretrained(...)`를 만든다.
  - `ModuleNotFoundError`는 Review Focus 5대로 바꿔 낸다.
- 처리 흐름:
  1. `read_wav(mono)`
  2. torch `(1, n)` 텐서로 바꾼다.
  3. `processor(descriptions=[plan.description], audios=[tensor], anchors=...)`
  4. `batch.to(device)`
  5. `torch.inference_mode()` 안에서 `torch.manual_seed(attempt)`를 하고 `model.separate(batch, reranking_candidates=..., predict_spans=...)`를 부른다.
  6. `result.target[0]`을 numpy로 바꾸고 길이를 맞춘 뒤 `write_wav(out, ...)`로 쓴다.
- 정확한 API는 `/home/judejiwoo/_AAAI27_Multitrack_V2A_System/sam-audio/examples/*.ipynb`와 고정한 커밋의 소스에서 확인한다.
  - 판단 기준은 고정 커밋의 공식 코드다. judejiwoo 사본은 로컬 수정이 있을 수 있으므로 기준으로 삼지 않는다.
  - 다르게 써야 하면 Deviations에 적는다.
- 잠금: 모듈 전역 `threading.Lock`. 주석: `# ponytail: one GPU model per process, one call at a time; batch or queue if throughput matters`

- [ ] **Step 1: 실패하는 테스트.** 가짜 `torch`와 `sam_audio`를 `sys.modules`에 넣는다. 가짜 모델은 입력 오디오를 0.5배 해서 target으로 돌려주고, 받은 인자를 기록한다.
  - `test_separate_passes_description_and_span_anchors`: spans 2개 → anchors가 `[["+", s, e], ["+", s, e]]`이고, 설명이 그대로 전달된다.
  - `test_separate_without_spans_uses_no_anchors`
  - `test_separate_writes_same_length_mono_48k`: 가짜 모델이 더 길거나 짧게 돌려줘도 입력 길이로 맞춰진다.
  - `test_attempt_sets_seed`
  - `test_model_loaded_once`: 두 번 불러도 `from_pretrained`는 1회만 불린다. 테스트 사이에 `_load.cache_clear()`를 한다.
  - `test_missing_gpu_extra_message`: `sam_audio` import가 실패하면 `"uv sync --extra gpu"`가 담긴 `RuntimeError`가 난다.
- [ ] **Step 2:** 실행해서 실패하는지 본다.
- [ ] **Step 3:** 구현한다.
- [ ] **Step 4:** 실행해서 통과하는지 본다. 기존 테스트도 모두 통과해야 한다.
- [ ] **Step 5:** 커밋한다: `feat(separation): separate sources with sam audio`

### Task 3: 필터링 real 예시 (SAM Audio Judge)

**Files:** Modify `src/audyssey/tools/filtering/real.py`; Test `tests/test_audyssey_sam_judge.py`

**Produces:** `judge(mono, track, plan) -> FilterResult`
- 로딩은 Task 2와 같은 패턴이다. 모델은 `SAMAudioJudgeModel`, processor는 `SAMAudioJudgeProcessor`, 설정은 `settings.sam_audio_judge_model`.
- 처리: processor로 `(text=[plan.description], input_audio=[mono], separated_audio=[track])`를 만들고, `inference_mode` 안에서 모델을 부른다.
- 결과:
  - `scores`: 출력의 `overall`, `recall`, `precision`, `faithfulness` 중 값이 있는 것을 float로 담는다.
  - `passed`: `scores["overall"] >= settings.judge_pass_threshold`
  - `reason`: 실패하면 `f"judge overall {x:.2f} < {threshold}"`
  - `suggestion`: `None`. Judge는 설명을 고쳐 주지 않는다.
- 모듈 docstring에 "윤석 담당 모듈의 예시 구현. 기준과 구현은 교체 가능"이라고 적는다.

- [ ] **Step 1: 실패하는 테스트** (가짜 모듈 사용)
  - `test_judge_maps_scores_and_threshold`: overall 0.7, 기준 0.5 → 통과. overall 0.3 → 실패이고 reason에 `0.30`이 들어간다.
  - `test_judge_passes_description_and_both_audios`
- [ ] **Step 2:** 실행해서 실패하는지 본다.
- [ ] **Step 3:** 구현한다.
- [ ] **Step 4:** 실행해서 통과하는지 본다.
- [ ] **Step 5:** 커밋한다: `feat(filtering): add sam audio judge example`

### Task 4: GPU 실측 스크립트

**Files:** Create `scripts/sam_audio_smoke.py`

**Produces:** `uv run --extra gpu python scripts/sam_audio_smoke.py <video> --description "..." [--span START END] --out <dir>`
- 영상에서 mono 48k를 뽑는다. 기존 `audyssey.media` 함수를 쓴다.
- real `separate`를 1번, real `judge`를 1번 부른다.
- 출력:
  - 모델 로딩 시간과 호출당 시간(초)
  - `torch.cuda.max_memory_allocated()`(GB)
  - torch, CUDA 버전
  - judge 점수
  - 결과 wav 경로
- CI에서는 돌리지 않는다. 테스트 대상이 아닌 수동 도구다.

- [ ] **Step 1:** 작성한다. `uv run ruff check scripts/`와 `uv run ruff format --check scripts/`가 통과해야 한다.
- [ ] **Step 2:** 커밋한다: `feat(scripts): add sam audio gpu smoke script`

### Task 5: sv3-a100 실측과 E2E

- [ ] **Step 1:** `uv sync --dev --extra ui --extra llm --extra gpu`
- [ ] **Step 2: 스모크.** 아래 명령을 2번 실행한다. 한 번은 span 없이, 한 번은 `--span 0 3`을 붙여서 실행한다.
  ```bash
  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
  AUDYSSEY_SAM_AUDIO_MODEL=mrfakename/sam-audio-large-tv \
  AUDYSSEY_SAM_AUDIO_JUDGE_MODEL=mrfakename/sam-audio-judge \
  uv run python scripts/sam_audio_smoke.py samples/sample.mp4 --description "a man speaking" --out work/smoke
  ```
- [ ] **Step 3: 원격 E2E.**
  - 분리 도구 서버(`--impl real`, 포트 8701)와 필터링 도구 서버(`--impl real`, 포트 8702)를 같은 환경변수로 띄운다. 토큰을 설정한다.
  - main 서버는 `AUDYSSEY_SEPARATOR=remote`, `AUDYSSEY_FILTER=remote`로 띄운다(플래너는 stub).
  - `samples/sample.mp4`를 업로드한다. 작업이 `ready`가 되고, 소스별 판정 점수가 기록되며, render가 200으로 끝나야 한다.
  - 각 소스의 시도 횟수와 통과 여부를 기록한다.
- [ ] **Step 4: 결과 기록.** 설계 문서 §13의 "SAM Audio 실측" 항목에 측정값을 표로 적는다.
  - GPU 메모리, 10초 기준 처리 시간, 로딩 시간
  - torch/CUDA 버전과 드라이버(570.124.06, CUDA 12.8)
  - Judge 점수 범위
  - 정한 `judge_pass_threshold` 값
  - 점수를 보고 `judge_pass_threshold` 기본값을 바꿨다면 Task 1의 테스트와 함께 고친다.
- [ ] **Step 5:** GPU 실행이 샌드박스 등으로 막히면 억지로 우회하지 않는다. 막힌 명령과 오류를 Deviations에 적고 다음 Task로 넘어간다. 실측은 사람이 대신 한다.

### Task 6: 문서, push, PR

**Files:** Modify `README.md`, `docs/superpowers/specs/2026-11-09-audyssey-skeleton-design.md`

- [ ] **Step 1: README 팀 가이드**
  - "SAM Audio 도구 서버 띄우기"를 추가한다: `uv sync --extra gpu`, 모델 환경변수, `HF_HUB_OFFLINE`, GPU 지정, 도구 서버 명령.
  - sv3-a100 공용 캐시 미러 사용법과, 다른 GPU(RunPod 등)에서는 HF 승인 또는 미러가 필요하다는 점을 적는다.
- [ ] **Step 2: 설계 문서**
  - §7.1에 real 구현 상태를 적는다.
  - §12에 결정을 추가한다: CI는 GPU 의존성 없이 가짜 모듈로 검증하고, 실제 분리는 GPU 서버에서 실측한다(사용자 결정, 2026-11-10).
- [ ] **Step 3:** CI 명령 4개가 모두 통과하는지 확인한다.
- [ ] **Step 4:** 커밋한다.
  - `docs(readme): add sam audio tool server guide`
  - `docs(spec): record sam audio decisions and measurements`
- [ ] **Step 5:** `git push -u origin feat/sam-audio`, 이어서 `gh pr create --base main`. PR 본문에 실측 표를 넣고, 끝에 `🤖 Generated with [Claude Code](https://claude.com/claude-code)`를 넣는다. 머지하지 않는다.
