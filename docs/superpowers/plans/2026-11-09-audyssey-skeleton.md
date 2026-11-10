# Audyssey 뼈대(Skeleton) 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. 구현은 ponytail(full)로 한다. 계획에 없는 추상화, 설정, 파일은 만들지 않는다.

**Goal**: GPU와 API 키 없이 업로드 → 플래닝 검증 루프 → 소스별 분리와 필터링 검증 루프 → 나머지 계산 → 편집 → 합치기까지 끝까지 도는 Audyssey 뼈대를 만든다. 모든 도구는 stub이다.

**Architecture**
- 새 패키지 `src/audyssey/`를 만든다. 기존 `v2a_inspect`와 같은 레포·같은 uv 프로젝트에 두고, 서로 import하지 않는다.
- 공통 계약은 `contracts.py`(Pydantic)에 둔다.
- 도구는 `tools/<모듈>/{stub,real}.py`의 순수 함수다. `tools.load_tools(settings)`가 설정에 따라 stub이나 real을 고른다.
- 분석 흐름은 LangGraph StateGraph다. 플래닝 루프는 조건부 엣지로, 소스별 분리 루프는 노드 안의 반복으로 구현한다.
- FastAPI가 Job 폴더(`work/jobs/<id>/job.json` + wav 파일)를 읽고 쓴다. 분석은 백그라운드 스레드에서 실행한다.
- 개발용 화면은 빌드가 필요 없는 정적 HTML 한 장이다.

**Tech Stack**
- 기존 의존성: Python 3.11+, uv, pydantic 2, pydantic-settings, langgraph, fastapi, uvicorn, python-multipart, httpx(TestClient), scipy(wav 입출력), numpy, typer
- 시스템 도구: ffmpeg/ffprobe(subprocess)
- 테스트: unittest(CI가 `uv run python -m unittest discover tests`로 실행)

**Spec**: `docs/superpowers/specs/2026-11-09-audyssey-skeleton-design.md`. 실행 전에 반드시 읽는다. 이 계획은 spec을 근거로 하므로 둘을 함께 본다.

## Global Constraints

- **새 의존성을 추가하지 않는다.** wav 입출력은 `scipy.io.wavfile`, pitch와 오디오 추출, 영상 합성은 ffmpeg subprocess로 한다. torch, librosa, soundfile, moviepy는 쓰지 않는다.
- **`audyssey`는 `v2a_inspect`를 import하지 않는다.** 반대 방향도 마찬가지다.
- **오디오 형식**
  - 샘플레이트는 48000Hz, 중간 파일은 float32 wav다.
  - 분리 트랙은 모노이고 원본과 길이가 같다(샘플 수 동일).
  - `audio.wav`는 원본 채널을 유지하고, `mono.wav`는 1채널이다.
- **경로와 시간**
  - Job 안의 모든 파일 경로는 **job 폴더 기준 상대 경로**다. 그래야 다른 기계로 옮겨도 동작한다.
  - 시간은 초 단위 float다.
- **설정값 (env prefix `AUDYSSEY_`)**
  - `PLANNER`, `SEPARATOR`, `FILTER` ∈ {`stub`,`real`}, 기본값 `stub`
  - `MAX_PLAN_ROUNDS=2`
  - `MAX_SEPARATION_RETRIES=2`(재분리 횟수이므로 총 시도는 최대 3회)
  - `MAX_DURATION_S=30`
  - `WORK_DIR=work/jobs`
  - `SAM_AUDIO_MODEL=facebook/sam-audio-large-tv`
  - `SAM_AUDIO_JUDGE_MODEL=facebook/sam-audio-judge`
- **명령 값 범위**
  - gain은 선형 0.0~2.0
  - pitch는 -12~+12 반음이고, residual에는 적용할 수 없다.
- **CI 통과**: ruff check, ruff format --check(`src/`), `ty check --error-on-warning src/`, unittest, 웹 빌드를 모두 통과해야 한다. 기존 v2a_inspect 테스트도 계속 통과해야 한다.
- **커밋 규칙 (sogang-v2a 팀 컨벤션)**
  - 형식: `type(scope): 소문자 명령형 요약`. 끝에 마침표를 찍지 않는다.
  - type: feat, fix, test, refactor, build, ci, docs, chore
  - scope: contracts, settings, preprocess, editing, tools, store, graph, api, devui, cli, ci, docs
  - **한 커밋에는 하나의 변경만 담는다.** 테스트와 그 테스트를 통과시키는 구현은 같은 커밋에 넣는다.
  - 모든 커밋 메시지 끝에 빈 줄을 두고 `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`를 붙인다.
  - 브랜치는 `feat/skeleton`(main에서 분기)이다. main에 직접 push하거나 merge하지 않는다.

## Review Focus

테스트에 꼭 넣어야 하는 실패 상황이다. 각 항목의 테스트는 아래 표에 적힌 태스크에 넣는다.

| 상황 | 기대하는 동작 | 테스트 위치 |
|---|---|---|
| 오디오 트랙이 없는 영상 업로드 | 400 응답과 "audio stream" 메시지. Job은 만들지 않는다 | Task 3, Task 8 |
| 30초 넘는 영상 업로드 | 400 응답과 "30" 포함 메시지 | Task 3, Task 8 |
| `files/{path}`에 `../` 경로 | 404. job 폴더 밖 파일은 절대 제공하지 않는다 | Task 6, Task 8 |
| 존재하지 않는 `source_id`로 명령 | 404, Job은 바뀌지 않는다 | Task 7, Task 8 |
| 모든 소스와 나머지를 drop하고 render | 에러 없이 무음 mix와 원본 길이의 영상이 나온다. stems는 빈 목록 | Task 4 |

---

## 파일 구조

```
src/audyssey/
  __init__.py
  contracts.py          # Task 1
  settings.py           # Task 2
  media.py              # Task 3 — ffprobe/ffmpeg, wav 입출력, peaks
  tools/
    __init__.py         # Task 5 — Tools, load_tools
    planning/   __init__.py stub.py real.py
    separation/ __init__.py stub.py real.py
    filtering/  __init__.py stub.py real.py
    editing/    __init__.py render.py      # Task 4
  store.py              # Task 6
  graph.py              # Task 7
  api.py                # Task 8
  devui/index.html      # Task 9
  cli.py                # Task 9
tests/
  audyssey_media.py     # Task 3 — 테스트용 합성 영상 생성 헬퍼 (test_ 접두어 없음)
  test_audyssey_contracts.py / _settings.py / _media.py / _editing.py
  test_audyssey_tools.py / _store.py / _graph.py / _api.py / _cli.py
```

**Job 폴더 구조**: `input.<ext>`, `audio.wav`, `mono.wav`, `peaks.json`, `tracks/<source_id>.wav`, `residual.wav`, `out/mix.wav`, `out/video.mp4`, `out/stems/<source_id>.wav`

---

### Task 0: 브랜치와 패키지 등록

**Files**
- Modify: `pyproject.toml`
  - `[tool.uv.build-backend] module-name`에 `"audyssey"`를 추가한다.
  - `[tool.uv-build] include`에 `"src/audyssey/devui/*.html"`를 추가한다.
  - `[project.scripts]`에 `audyssey = "audyssey.cli:main"`을 추가한다.
- Create: `src/audyssey/__init__.py`(빈 파일)

- [ ] **Step 1**: `git switch -c feat/skeleton main`
- [ ] **Step 2**: pyproject를 수정하고 빈 `__init__.py`를 만든다. `cli:main`은 Task 9에서 생긴다. 스크립트 항목은 지금 넣어 두며, 설치 단계에서는 문제가 없다.
- [ ] **Step 3**: `uv sync --dev --extra ui --extra observability && uv run python -c "import audyssey"`를 실행한다. 기대 결과: 종료 코드 0
- [ ] **Step 4**: 커밋. `build: register audyssey package`

---

### Task 1: 공통 계약

**Files**
- Create: `src/audyssey/contracts.py`
- Test: `tests/test_audyssey_contracts.py`

**Interfaces**
- 이 태스크가 만드는 것(Produces): spec §6의 모델 전부. 아래는 spec과 다르거나 spec에 없는 부분만 적는다.
  - `Span(start: float >= 0, end: float)`: `end > start`가 아니면 ValidationError
  - `VisualHint(time: float, box: tuple[float, float, float, float], description: str)`: box 값이 0~1 범위를 벗어나면 ValidationError
  - `SourceKind = Literal["dialogue","sfx","music","ambience"]`
  - `SourcePlan(name: str (min_length=1), kind: SourceKind, description: str, spans: list[Span] = [], visual_hint: VisualHint | None = None)`
  - `PlanIssue(source_name: str | None, kind: str, message: str)`
  - `FilterResult(passed: bool, scores: dict[str, float] = {}, reason: str = "", suggestion: SourcePlan | None = None)`
  - `SourceStatus = Literal["pending","running","done","failed"]`
  - `Source(id: str, plan: SourcePlan, track_path: str | None = None, filter: FilterResult | None = None, status: SourceStatus = "pending", attempts: int = 0, keep: bool = True, gain: float = 1.0 (0~2), pitch_semitones: float = 0.0 (-12~12))`
  - `Residual(path: str | None = None, keep: bool = True, gain: float = 1.0 (0~2))`
  - `StepStatus(key: str, status: Literal["pending","running","done","failed","warning"], round: int = 0, max_rounds: int = 0, message: str = "")`
  - `RenderOutput(mix_wav: str, video_mp4: str, stems: list[str])`
  - `Job(id, video_path, duration: float = 0.0, audio_path: str = "", mono_path: str = "", peaks_path: str = "", sources: list[Source] = [], residual: Residual = Residual(), plan_issues: list[PlanIssue] = [], steps: list[StepStatus] = [], outputs: RenderOutput | None = None, state: Literal["running","ready","failed"] = "running", version: int = 0, error: str | None = None)`
    - spec에 없는 필드 두 개를 추가했다. `state`는 잠금과 UI 표시에 쓰고, `version`은 SSE에서 변경을 감지하는 데 쓴다.
  - 명령 모델(모두 `type: Literal[...]` 필드를 가진다)
    - `Keep(source_id)`, `Drop(source_id)`
    - `SetGain(source_id, gain: 0~2)`
    - `SetPitch(source_id, semitones: -12~12)`
    - `Reseparate(source_id, description: str | None = None, spans: list[Span] | None = None)`
    - `AddSource(plan: SourcePlan)`
    - `Render(include_residual_stem: bool = False)`
  - `Command = Annotated[Keep | Drop | SetGain | SetPitch | Reseparate | AddSource | Render, Field(discriminator="type")]`
  - `parse_command(data: dict) -> Command`(내부적으로 `TypeAdapter(Command)`를 쓴다)
  - `Job.find_source(source_id: str) -> Source | None`
- 모든 모델에 `model_config = ConfigDict(extra="forbid")`를 둔다.

- [ ] **Step 1: 실패하는 테스트를 작성한다**
  - `test_span_rejects_end_before_start`: `Span(start=2, end=1)` → `ValidationError`
  - `test_visual_hint_box_must_be_normalized`: box `(0, 0, 1.5, 1)` → `ValidationError`
  - `test_job_roundtrip_json`: Source 하나와 Residual이 있는 Job을 `model_dump_json()` → `Job.model_validate_json()`으로 왕복하면 원래 객체와 같다.
  - `test_parse_command_dispatches_on_type`
    - `parse_command({"type":"set_gain","source_id":"s1","gain":0.5})`가 `SetGain`이다.
    - `{"type":"render"}`의 `include_residual_stem`이 False다.
  - `test_command_bounds`: gain 2.5, semitones 13, 알 수 없는 type은 각각 `ValidationError`다.
  - `test_find_source`: 있으면 해당 Source, 없으면 None을 돌려준다.
- [ ] **Step 2**: `uv run python -m unittest tests.test_audyssey_contracts -v`를 실행한다. 기대 결과: ImportError
- [ ] **Step 3**: `contracts.py`를 구현한다.
- [ ] **Step 4**: 같은 명령을 다시 실행한다. 기대 결과: 전부 PASS. `uv run ty check --error-on-warning src/audyssey`도 통과해야 한다.
- [ ] **Step 5**: 커밋. `feat(contracts): add shared job, source and command models`

---

### Task 2: 설정

**Files**
- Create: `src/audyssey/settings.py`
- Test: `tests/test_audyssey_settings.py`

**Interfaces**
- 만드는 것: `class Settings(BaseSettings)`(env_prefix `AUDYSSEY_`). 필드와 기본값은 Global Constraints와 같다. `work_dir: Path`, 도구 선택 필드는 `Literal["stub","real"]`이다.

- [ ] **Step 1: 테스트를 작성한다**
  - `test_defaults`: planner, separator, filter가 `"stub"`이다. `max_plan_rounds == 2`, `max_separation_retries == 2`, `max_duration_s == 30`
  - `test_env_override`: `unittest.mock.patch.dict(os.environ, {"AUDYSSEY_PLANNER": "real", "AUDYSSEY_MAX_PLAN_ROUNDS": "3"})`을 적용하면 값이 반영된다.
  - `test_rejects_unknown_mode`: `AUDYSSEY_SEPARATOR=gpu` → `ValidationError`
- [ ] **Step 2**: 테스트가 실패하는지 확인한다(ImportError).
- [ ] **Step 3**: 구현한다. `.env` 파일은 읽지 않는다. 환경변수만 쓴다.
- [ ] **Step 4**: 테스트가 통과하는지 확인한다.
- [ ] **Step 5**: 커밋. `feat(settings): add audyssey settings with stub defaults`

---

### Task 3: 미디어 유틸 (ffprobe/ffmpeg, wav, peaks)

**Files**
- Create: `src/audyssey/media.py`, `tests/audyssey_media.py`
- Test: `tests/test_audyssey_media.py`

**Interfaces**
- 만드는 것
  - `class MediaError(ValueError)`: 사용자에게 보여 줄 메시지를 담는다.
  - `probe(video: Path) -> tuple[float, int]`: (길이 초, 오디오 채널 수)를 돌려준다.
    - 오디오 스트림이 없으면 `MediaError("video has no audio stream")`
  - `check_upload(video: Path, max_duration_s: float) -> float`: probe 결과를 검사하고 길이를 돌려준다.
    - 길이가 상한을 넘으면 `MediaError(f"video is {d:.1f}s; maximum is {max:.0f}s")`
  - `extract_audio(video: Path, out: Path, mono: bool) -> Path`: ffmpeg `-vn -ar 48000 -c:a pcm_f32le`. mono이면 `-ac 1`을 추가한다.
  - `read_wav(path) -> np.ndarray`: float32, 모양은 (samples, channels)다. 모노도 2차원 (n, 1)이다.
  - `write_wav(path, data: np.ndarray) -> Path`: float32로 쓰고, 상위 폴더를 만든다.
  - `compute_peaks(wav: Path, buckets: int = 800) -> list[float]`: 모노로 합친 뒤 구간별 최대 절댓값. 0~1이고 길이는 `min(buckets, samples)`다.
- 테스트 헬퍼 `tests/audyssey_media.py`
  - `make_video(path: Path, seconds: float = 3.0, channels: int = 2, audio: bool = True) -> Path`
  - ffmpeg lavfi `testsrc=size=160x120:rate=10`와 `sine=frequency=440`(channels만큼)로 mp4를 만든다. `audio=False`면 영상만 만든다.
- 이 파일의 모든 테스트 클래스에 `@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg required")`를 붙인다. CI에는 Task 10에서 ffmpeg를 설치한다.

- [ ] **Step 1: 테스트를 작성한다**
  - `test_probe_reads_duration_and_channels`: 3초 스테레오 영상 → 길이 ≈ 3.0(±0.1), 채널 2
  - `test_check_upload_rejects_no_audio`: `audio=False` → `MediaError`이고 메시지에 "audio stream" 포함 (Review Focus)
  - `test_check_upload_rejects_too_long`: 3초 영상에 상한 2를 주면 `MediaError`이고 메시지에 "maximum is 2s" 포함 (Review Focus)
  - `test_extract_audio_mono_and_original`: 원본 추출은 shape[1]==2, 모노 추출은 shape[1]==1. 둘 다 샘플레이트 48000이고 샘플 수가 같다.
  - `test_wav_roundtrip_float32`: 랜덤 (100, 2) 배열을 쓰고 읽으면 dtype float32이고 값이 같다.
  - `test_peaks_range`: 길이 ≤ 800, 모든 값이 0~1
- [ ] **Step 2**: 테스트가 실패하는지 확인한다.
- [ ] **Step 3**: 구현한다. ffprobe는 `-show_entries format=duration:stream=codec_type,channels -of json`으로 호출한다. subprocess가 실패하면 stderr 내용을 담아 `MediaError`를 낸다.
- [ ] **Step 4**: 테스트가 통과하는지 확인한다.
- [ ] **Step 5**: 커밋. `feat(preprocess): add ffmpeg audio extraction, wav io and peaks`

---

### Task 4: 나머지(residual)와 render (편집 DSP)

**Files**
- Create: `src/audyssey/tools/__init__.py`(Task 5에서 내용을 채운다. 여기서는 빈 파일), `src/audyssey/tools/editing/__init__.py`, `src/audyssey/tools/editing/render.py`
- Test: `tests/test_audyssey_editing.py`

**Interfaces**
- 이 태스크가 쓰는 것(Consumes): `media.read_wav`, `write_wav`, `contracts.Job/Source/Residual/RenderOutput`
- 만드는 것
  - `compute_residual(audio: Path, tracks: list[Path], out: Path) -> Path`: `residual = original − Σ tracks`. 모노 트랙은 원본의 각 채널에서 뺀다. 계산은 float64, 저장은 float32다.
  - `shift_pitch(src: Path, out: Path, semitones: float) -> Path`
    - ffmpeg 필터 `asetrate=48000*r,aresample=48000,atempo=1/r`(r = 2^(semitones/12)) 를 쓴다.
    - 결과는 입력과 샘플 수가 같아야 한다. 모자라면 0으로 채우고, 넘치면 자른다.
  - `render(job: Job, job_dir: Path, include_residual_stem: bool) -> RenderOutput`
    - 이 태스크에서 Job의 상대 경로를 해석하는 기준은 `job_dir`다. spec의 `out_dir` 인자는 `job_dir/"out"`로 고정했다.
    - **mix**: keep된 소스에 pitch와 gain을 적용하고, 원본 채널 수만큼 복제해 더한다. keep된 residual은 gain만 적용해 더한다. 결과는 `out/mix.wav`다.
    - **stems**: keep된 소스마다 편집값을 적용한 모노 wav를 `out/stems/<id>.wav`에 쓴다. `include_residual_stem`이고 residual이 keep이면 `out/stems/residual.wav`도 쓴다.
    - **video**: `ffmpeg -i <video> -i out/mix.wav -map 0:v -map 1:a -c:v copy -c:a aac -shortest` → `out/video.mp4`
    - 반환하는 경로는 모두 job_dir 기준 상대 경로다.

- [ ] **Step 1: 테스트를 작성한다.** 영상은 Task 3의 `make_video`를 쓰고, 소스 트랙은 원본 모노의 일부 구간만 남기고 나머지를 0으로 채운 배열로 만든다.
  - `test_untouched_render_equals_original`: 소스 2개와 residual을 계산하고 편집 없이 render하면, mix.wav와 audio.wav의 최대 절대 오차가 1e-4 미만이다. (성공 기준 3)
  - `test_drop_removes_exactly_that_source`: 소스 1을 drop하고 render하면 mix와 (원본 − 소스 1 트랙)의 차이가 1e-4 미만이다.
  - `test_shift_pitch_preserves_length`: +5와 −7 반음 모두 샘플 수가 그대로다.
  - `test_stems_only_kept_sources`: 소스 2개 중 하나를 drop하면 stems가 1개다. `include_residual_stem=True`면 2개다.
  - `test_all_dropped_renders_silence`: 모든 소스와 residual을 drop하면 mix의 최대 절댓값이 0이고, `out/video.mp4`가 있으며, stems는 빈 목록이다. (Review Focus)
  - `test_residual_ignores_keep_flag`: `compute_residual`의 결과는 keep 값과 무관하다. 호출하는 쪽이 트랙이 있는 소스를 모두 넘겨야 한다는 사실을 docstring에 적는다.
- [ ] **Step 2**: 테스트가 실패하는지 확인한다.
- [ ] **Step 3**: `compute_residual`과 `shift_pitch`를 구현한다. 여기까지 해당 테스트가 통과하면 커밋한다. `feat(editing): compute residual and duration-preserving pitch shift`
- [ ] **Step 4**: `render`를 구현한다.
- [ ] **Step 5**: `uv run python -m unittest tests.test_audyssey_editing -v`를 실행한다. 기대 결과: 전부 PASS
- [ ] **Step 6**: 커밋. `feat(editing): render edited mix, stems and video`

---

### Task 5: stub 도구와 도구 로더

**Files**
- Create: `src/audyssey/tools/{planning,separation,filtering}/{__init__,stub,real}.py`
- Modify: `src/audyssey/tools/__init__.py`
- Test: `tests/test_audyssey_tools.py`

**Interfaces**
- 쓰는 것: `contracts`, `settings.Settings`, `media.read_wav/write_wav`
- 만드는 것(spec §7.1의 시그니처를 그대로 따른다. 경로는 `Path`다)
  - `planning.stub.plan(video: Path, audio: Path, duration: float, feedback: list[PlanIssue]) -> list[SourcePlan]`
  - `planning.stub.validate_plan(video: Path, audio: Path, plans: list[SourcePlan]) -> list[PlanIssue]`
  - `separation.stub.separate(mono: Path, plan: SourcePlan, video: Path, out: Path, attempt: int) -> Path`
  - `filtering.stub.judge(mono: Path, track: Path, plan: SourcePlan) -> FilterResult`
  - `tools.Tools`: frozen dataclass. 필드는 `plan`, `validate_plan`, `separate`, `judge`이고, 위 시그니처의 `Callable`이다.
  - `tools.load_tools(settings: Settings) -> Tools`
    - 모듈마다 `importlib.import_module(f"audyssey.tools.{module}.{mode}")`로 불러온다.
    - real 모듈은 선택된 경우에만 import된다.
- `real.py` 세 파일은 같은 이름의 함수를 정의한다. 본문은 `raise NotImplementedError("<담당자>: see docs/superpowers/specs/2026-11-09-audyssey-skeleton-design.md §7.1")`이다. 담당자는 planning=플래닝 담당, separation=SAM Audio, filtering=필터링 담당이다.
- **stub 동작.** 상태를 저장하지 않고, 입력만 보고 결정한다. d는 duration이다.
  - `plan(feedback=[])` → 소스 4개
    - `character_a`(dialogue, spans [0, 0.3d])
    - `character_b`(dialogue, [0.35d, 0.6d])
    - `sword_clash`(sfx, [0.65d, 0.9d])
    - `ghost_sound`(sfx, [0.1d, 0.2d])
    - description은 각각 `"voice of character a"` 형식이다.
  - `plan(feedback=비어 있지 않음)` → 위 목록에서 feedback에 이름이 나온 소스를 뺀다.
  - `validate_plan` → 이름이 `ghost_`로 시작하는 소스마다 `PlanIssue(source_name, "not_visible_in_frame", ...)`를 돌려준다. 그 결과 플래닝 루프가 정확히 한 번 돈다.
  - `separate` → 모노 원본에서 spans 구간만 남기고 나머지를 0으로 채운 뒤 out에 쓴다. 길이는 원본과 같다.
  - `judge`
    - `plan.name == "character_a"`이고 description이 `" (refined)"`로 끝나지 않으면 불합격이다. `reason="leakage from other voices"`이고, `suggestion`은 description 끝에 `" (refined)"`를 붙인 plan이다.
    - 그 밖에는 합격이다. 점수는 `{"overall": 0.9, "recall": 0.9, "precision": 0.9}`
    - 그 결과 필터링 루프가 character_a에 대해 정확히 한 번 돈다.

- [ ] **Step 1: 테스트를 작성한다**
  - `test_load_tools_defaults_to_stub`: `load_tools(Settings())`의 각 함수가 stub 모듈의 함수와 같다.
  - `test_load_tools_real_raises_not_implemented`: planner가 real이면 `tools.plan(...)` 호출이 `NotImplementedError`다.
  - `test_stub_plan_loop`: 첫 plan의 validate_plan 결과가 문제 1개(ghost_sound)다. 그 문제를 feedback으로 넘기면 소스 3개가 나오고, 다시 검증하면 빈 목록이다.
  - `test_stub_judge_loop`: character_a 원래 plan은 불합격이고 suggestion이 있다. suggestion으로 다시 judge하면 합격이다.
  - `test_stub_separate_keeps_length`: 출력 샘플 수가 원본 모노와 같고, span 밖은 모두 0이다.
- [ ] **Step 2**: 테스트가 실패하는지 확인한다.
- [ ] **Step 3**: 구현한다.
- [ ] **Step 4**: 테스트가 통과하는지 확인한다.
- [ ] **Step 5**: 커밋한다. 두 개로 나눈다.
  - `feat(tools): add stub planning, separation and filtering tools`(stub, real, 테스트)
  - `feat(tools): select stub or real tools from settings`(load_tools와 해당 테스트)

---

### Task 6: Job 저장소

**Files**
- Create: `src/audyssey/store.py`
- Test: `tests/test_audyssey_store.py`

**Interfaces**
- 만드는 것: `class JobStore`
  - `__init__(root: Path)`
  - `create(filename: str, data: bytes) -> tuple[Job, Path]`
    - 새 id는 `uuid4().hex[:12]`다.
    - `<root>/<id>/input<ext>`를 저장하고 `job.json`을 쓴다. `video_path="input<ext>"`
  - `dir(job_id: str) -> Path`
  - `save(job: Job) -> Job`: `version`을 1 올리고, 임시 파일에 쓴 뒤 이름을 바꾸는 방식으로 `job.json`을 쓴다.
  - `load(job_id: str) -> Job`: 없으면 `KeyError`
  - `list() -> list[Job]`: 수정 시각 최신 순
  - `resolve(job_id: str, rel: str) -> Path`
    - job 폴더 안의 파일 경로를 돌려준다.
    - `resolve()`한 결과가 job 폴더 밖이거나, 파일이 없으면 `KeyError`
  - job_id도 `[0-9a-f]{12}` 형식이 아니면 `KeyError`다.

- [ ] **Step 1: 테스트를 작성한다**
  - `test_create_and_load`
  - `test_save_increments_version`
  - `test_list_newest_first`
  - `test_resolve_rejects_traversal`: `"../x"`, `"/etc/passwd"`, 없는 파일은 모두 `KeyError` (Review Focus)
  - `test_load_rejects_bad_id`: `"../.."` → `KeyError`
- [ ] **Step 2**: 테스트가 실패하는지 확인한다.
- [ ] **Step 3**: 구현한다.
- [ ] **Step 4**: 테스트가 통과하는지 확인한다.
- [ ] **Step 5**: 커밋. `feat(store): persist jobs as job folders`

---

### Task 7: 오케스트레이션 흐름 (LangGraph + 명령)

**Files**
- Create: `src/audyssey/graph.py`
- Test: `tests/test_audyssey_graph.py`

**Interfaces**
- 쓰는 것: Task 1~6 전부
- 만드는 것
  - `@dataclass class Context`: `tools: Tools`, `settings: Settings`, `store: JobStore`
  - `check_plans(plans: list[SourcePlan], duration: float) -> tuple[list[SourcePlan], list[PlanIssue]]`: 형식 검사다.
    - 이름이 같은 소스는 하나로 합친다(spans 합침, 앞의 description 유지).
    - span이 영상 길이를 넘으면 `span_out_of_range` 문제다.
    - description이 비어 있으면 `empty_description` 문제다.
  - `analyze(job: Job, ctx: Context) -> Job`: 업로드 직후 분석 전체를 실행한다.
    - LangGraph `StateGraph(AnalysisState)`이고, `AnalysisState = TypedDict("AnalysisState", {"job": Job, "issues": list[PlanIssue], "round": int})`다.
    - 노드: `preprocess` → `plan` → `validate` →(조건부) `plan` 또는 `separate` → `residual` → END
    - **preprocess**: `extract_audio`로 audio.wav와 mono.wav를 만들고, `compute_peaks`로 peaks.json을 만든다. 결과를 `job.audio_path`, `job.mono_path`, `job.peaks_path`에 상대 경로로 기록한다. `job.duration`이 0이면 `probe` 값으로 채운다.
    - **plan**: `tools.plan(feedback=issues)`를 부르고 `check_plans`를 적용한다.
    - **validate**: 형식 문제가 있으면 그 문제를 쓰고, 없을 때만 `tools.validate_plan` 결과를 쓴다.
    - **조건부 엣지**: 문제가 있고 `round < max_plan_rounds`면 plan으로 돌아간다. 그렇지 않으면 separate로 간다.
      - 한도에 도달했는데 문제가 남아 있으면 `job.plan_issues`에 남기고, plan 단계 상태를 `warning`으로 둔다.
    - **separate**: 소스마다 `process_source`를 부른다.
    - **residual**: 트랙이 있는 모든 소스로 `compute_residual`을 부른다.
    - 노드가 끝날 때마다 `ctx.store.save(job)`를 호출하고, 해당 `StepStatus`(key는 spec §6 형식)를 갱신한다.
    - preprocess나 plan에서 예외가 나면 `state="failed"`로 바꾸고 `error=str(e)`를 남긴 뒤 저장하고 반환한다. 예외를 다시 던지지 않는다.
    - 정상 종료하면 `state="ready"`
  - `process_source(job: Job, source: Source, ctx: Context) -> None`
    - 반복: attempt를 올리고 `separate`를 부른 뒤 `judge`를 부른다.
    - 불합격이고 `attempts <= max_separation_retries`면 suggestion이 있을 때 `source.plan`을 suggestion으로 바꾸고 다시 시도한다.
    - 루프를 마치면 `status="done"`이다. 마지막 결과가 불합격이면 step 상태를 `warning`으로 둔다.
    - 도구에서 예외가 나면 그 소스만 `status="failed"`, `track_path=None`
  - `handle_command(job: Job, cmd: Command, ctx: Context) -> Job`
    - Keep, Drop, SetGain, SetPitch: 값만 바꾼다. `source_id="residual"`이면 residual을 바꾼다. residual에 SetPitch를 주면 `ValueError`다.
    - Reseparate: description이나 spans가 있으면 plan을 바꾸고, `attempts`를 0으로 되돌린 뒤 `process_source`와 residual 재계산을 한다.
    - AddSource: `check_plans`에 문제가 있으면 `ValueError(문제 메시지)`다. 문제가 없으면 새 id(`s<n>`, 기존 최대값 + 1)로 Source를 만들고 `process_source`와 residual 재계산을 한다.
    - Render: `render`를 부르고 결과를 `job.outputs`에 넣는다.
    - 알 수 없는 source_id면 `KeyError(source_id)`이고, Job은 바꾸지 않는다. (Review Focus)
    - 끝나면 저장하고 반환한다.
  - 소스 id는 플래닝 결과 순서대로 `s1`, `s2`, ...다.

- [ ] **Step 1: 테스트를 작성한다.** 모두 stub 도구를 쓰고, 3초 스테레오 합성 영상을 store.create로 만든 뒤 진행한다.
  - `test_analyze_runs_both_validation_loops`
    - state가 `"ready"`다.
    - 소스 3개이고, `ghost_sound`는 없다.
    - plan step의 round가 2다.
    - character_a는 `attempts == 2`이고 `filter.passed`가 True이며, description이 `" (refined)"`로 끝난다.
    - 다른 소스는 `attempts == 1`이다.
    - `residual.path`가 있다. (성공 기준 4)
  - `test_analyze_then_untouched_render_equals_original`: handle_command(Render())의 mix와 audio.wav의 오차가 1e-4 미만이다.
  - `test_plan_round_limit_warns`: `max_plan_rounds=1`이면 `plan_issues`가 1개이고 plan step이 `"warning"`이며, ghost_sound가 남아 있다.
  - `test_check_plans_merges_and_flags`: 같은 이름 두 개는 하나로 합쳐지고 span이 2개다. span 끝이 길이를 넘으면 `span_out_of_range`다.
  - `test_tool_error_marks_only_that_source_failed`: separate가 sword_clash에서만 예외를 내는 Tools로 바꾸면, 그 소스만 `failed`이고 state는 `"ready"`다.
  - `test_commands_update_values`: drop, set_gain(s2, 0.5), keep(residual)가 반영된다. residual에 set_pitch를 주면 `ValueError`다.
  - `test_reseparate_and_add_source`
    - reseparate(s2, description="new")는 plan.description을 "new"로 바꾸고 attempts를 1로 만들며, residual 파일을 다시 쓴다(mtime이 바뀜).
    - add_source는 s4를 만든다.
    - spans가 범위를 넘는 add_source는 `ValueError`다.
  - `test_unknown_source_id`: drop("s99")는 `KeyError`이고, 저장된 version이 그대로다. (Review Focus)
- [ ] **Step 2**: 테스트가 실패하는지 확인한다.
- [ ] **Step 3**: `check_plans`와 `process_source`를 구현한다. 해당 테스트가 통과하면 커밋한다. `feat(graph): add plan format checks and per-source separation loop`
- [ ] **Step 4**: `analyze`를 LangGraph로 구현한다. 소스 처리는 순차 반복으로 하고 `# ponytail: sequential per-source loop, fan out with Send when real GPU tools need parallelism` 주석을 단다. analyze 테스트가 통과하면 커밋한다. `feat(graph): run analysis graph with planning validation loop`
- [ ] **Step 5**: `handle_command`를 구현한다. 명령 테스트가 통과하면 커밋한다. `feat(graph): handle edit, reseparate, add-source and render commands`

---

### Task 8: HTTP API

**Files**
- Create: `src/audyssey/api.py`
- Test: `tests/test_audyssey_api.py`

**Interfaces**
- 쓰는 것: `graph.Context/analyze/handle_command`, `store.JobStore`, `media.check_upload/MediaError`, `contracts.parse_command`
- 만드는 것: `create_app(settings: Settings | None = None, tools: Tools | None = None) -> FastAPI`
  - 상태는 `app.state`에 둔다. Context 하나와 Job별 `threading.Lock` 딕셔너리다.
  - **엔드포인트** (spec §8)
    - `POST /api/jobs` (multipart `file`)
      - 임시 저장 후 `check_upload`를 한다. `MediaError`면 업로드 파일을 지우고 400과 `{"detail": 메시지}`를 돌려준다.
      - 통과하면 store에 Job을 만들고 duration을 기록한다. 그 Job의 잠금을 잡은 상태로 스레드에서 `analyze`를 실행하고, 끝나면 잠금을 푼다.
      - 응답은 201과 Job이다.
    - `GET /api/jobs`: Job 목록
    - `GET /api/jobs/{id}`: Job. 없으면 404
    - `GET /api/jobs/{id}/events`: SSE(`text/event-stream`)
      - 연결 직후 현재 Job을 `data: <json>\n\n`로 한 번 보낸다.
      - 그 뒤 0.25초마다 store를 다시 읽어 `version`이 바뀌면 보낸다.
      - `# ponytail: polling store, push from save() if latency matters`
    - `POST /api/jobs/{id}/commands`
      - body를 `parse_command`로 읽는다. 실패하면 422다.
      - 값 변경 명령(Keep, Drop, SetGain, SetPitch)은 잠금 없이 즉시 처리하고 200과 Job을 돌려준다.
      - 실행 명령(Reseparate, AddSource, Render)
        - 잠금이 잡혀 있으면 409와 `{"detail": "job is busy"}`
        - 아니면 잠금을 잡고 **동기로** 실행한 뒤 200과 Job을 돌려준다. 10초 클립이라 동기 처리로 충분하다.
      - `KeyError`는 404, `ValueError`는 422
    - `GET /api/jobs/{id}/files/{path:path}`: `store.resolve`를 거쳐 `FileResponse`로 보낸다. `KeyError`는 404다.
    - `GET /`: Task 9의 devui/index.html. 이 태스크에서는 아직 없으므로 라우트만 둔다. 파일이 없으면 404다.

- [ ] **Step 1: 테스트를 작성한다.** `fastapi.testclient.TestClient`를 쓰고, `work_dir`는 `tempfile.TemporaryDirectory()`다.
  - `test_upload_runs_analysis_to_ready`: 3초 영상을 업로드하면 201이다. 0.1초 간격으로 최대 30초까지 GET을 반복하면 state가 `"ready"`가 되고 소스가 3개다.
  - `test_upload_rejects_no_audio` → 400, detail에 "audio stream"이 있다. store에 Job이 0개다. (Review Focus)
  - `test_upload_rejects_too_long`: `max_duration_s=2` 설정에 3초 영상을 올리면 400, detail에 "maximum is 2s"가 있다. (Review Focus)
  - `test_command_render_and_download`: ready가 된 뒤 render 명령을 보내면 200이고 `outputs.video_mp4`가 있다. 그 파일을 files 엔드포인트로 받으면 200이고 내용이 비어 있지 않다.
  - `test_command_conflict_while_busy`: 테스트에서 `app.state.locks[id]`를 직접 잡은 상태로 reseparate를 보내면 409다. 같은 상태에서 set_gain은 200이다.
  - `test_command_errors`: 알 수 없는 type은 422, 없는 source_id는 404, 범위를 넘는 add_source는 422 (Review Focus)
  - `test_files_traversal_404`: `/files/..%2F..%2Fetc%2Fpasswd`와 `/files/missing.wav`는 404 (Review Focus)
  - `test_events_first_message_is_job`: `client.stream("GET", .../events)`의 첫 `data:` 줄을 JSON으로 읽으면 id가 맞다.
- [ ] **Step 2**: 테스트가 실패하는지 확인한다.
- [ ] **Step 3**: 구현한다.
- [ ] **Step 4**: 테스트가 통과하는지 확인한다.
- [ ] **Step 5**: 커밋한다. 두 개로 나눈다.
  - `feat(api): add job upload, status, files and events endpoints`
  - `feat(api): add command endpoint with per-job busy lock`
  - 명령 엔드포인트와 그 테스트를 두 번째 커밋에 넣는다.

---

### Task 9: 개발용 화면과 CLI

**Files**
- Create: `src/audyssey/devui/index.html`, `src/audyssey/cli.py`
- Test: `tests/test_audyssey_cli.py`. `/` 테스트는 `tests/test_audyssey_api.py`에 추가한다.

**Interfaces**
- 만드는 것
  - `devui/index.html`: 빌드 없이 쓰는 정적 HTML 한 장이다. vanilla JS를 쓰고, 외부 CDN을 쓰지 않는다. 기능은 spec §9 목록과 같다.
    - 업로드 폼
    - 단계 목록: key, status, round/max_rounds, message를 보여 주고 warning과 failed는 색으로 구분한다.
    - 소스 목록: `<audio>` 재생, keep/drop 토글, gain과 pitch 슬라이더, description 입력 + "다시 분리" 버튼
    - residual 행: keep과 gain
    - 소스 추가 폼: 이름, 종류, 설명, 구간 하나
    - "합치기" 버튼과 결과 영상 `<video>`, stems 링크
    - 상태는 `EventSource(/api/jobs/{id}/events)`로 갱신한다. 명령 응답이 409나 422면 응답의 detail을 화면에 보여 준다.
    - 제목은 "Audyssey dev UI"이고, 화면에 "개발용 화면 — 최종 UI 아님" 문구를 둔다.
  - `api.py`의 `GET /`는 `importlib.resources.files("audyssey.devui") / "index.html"`을 `HTMLResponse`로 돌려준다. 그러려면 `devui/__init__.py`가 필요하다(빈 파일).
  - `cli.py`: typer 앱 `main()`이고 명령은 `serve(host: str = "127.0.0.1", port: int = 8600)`이다. `uvicorn.run(create_app(), host, port)`를 실행한다.

- [ ] **Step 1: 테스트를 작성한다**
  - `test_root_serves_dev_ui`: `GET /`가 200이고 본문에 "Audyssey dev UI"가 있다.
  - `test_cli_help`: `typer.testing.CliRunner().invoke(app, ["serve", "--help"])`의 exit_code가 0이고 출력에 "--port"가 있다.
- [ ] **Step 2**: 테스트가 실패하는지 확인한다.
- [ ] **Step 3**: 구현한다.
- [ ] **Step 4**: 테스트가 통과하는지 확인한다. 이어서 수동으로 확인한다.
  - `uv run audyssey serve --port 8600 &`로 서버를 띄운다.
  - `curl -F file=@samples/sample.mp4 localhost:8600/api/jobs`를 보낸다.
  - 결과로 받은 id로 `GET`을 반복해 `"state":"ready"`가 나오는지 본다.
  - render 명령을 보낸 뒤 `out/video.mp4`를 받아 크기가 0보다 큰지 확인한다. 확인이 끝나면 서버를 종료한다.
- [ ] **Step 5**: 커밋한다. 두 개로 나눈다.
  - `feat(devui): add minimal dev ui for end-to-end checks`
  - `feat(cli): add audyssey serve command`

---

### Task 10: CI와 문서

**Files**
- Modify: `.github/workflows/ci-lint.yaml`. `lint-and-format-check` job의 `unit-tests` 단계 앞에 다음 단계를 추가한다.
  - `- name: install ffmpeg` / `run: sudo apt-get update && sudo apt-get install -y ffmpeg`
  - `ubuntu-slim`에서 실패하면 이 job의 `runs-on`을 `ubuntu-latest`로 바꾼다.
- Modify: `README.md`. 맨 위에 "Audyssey skeleton" 절을 추가한다. 기존 v2a 내용은 그 아래에 그대로 둔다.
  - 요구사항: uv, ffmpeg. Node는 필요 없다.
  - 실행: `uv sync --extra ui` → `uv run audyssey serve` → http://127.0.0.1:8600
  - 테스트: `uv run python -m unittest discover tests`
  - 설정: `AUDYSSEY_*` 표
  - **팀원 가이드**
    - 자기 `tools/<모듈>/real.py`를 구현한다.
    - `AUDYSSEY_<TOOL>=real`로 설정한다.
    - 계약(`contracts.py`) 변경은 PR로 합의한다.
    - 모델 가중치는 `HF_HOME` 캐시를 쓴다.
    - spec과 plan 문서로 가는 링크를 둔다.
- Modify: `docs/superpowers/specs/2026-11-09-audyssey-skeleton-design.md`. 구현 중 확정된 차이를 반영한다.
  - §6 Job에 `state`와 `version`을 추가하고, 경로가 job 폴더 기준 상대 경로라는 점을 적는다.
  - §9, §10, §11: 개발용 화면은 `src/audyssey/devui/index.html` 한 장이고 Node가 필요 없다는 점을 적는다. `web/`는 UI 담당이 쓰도록 그대로 둔다.
  - §7.1: render 인자를 `job_dir`로 바꾼다.

- [ ] **Step 1**: `uv run ruff check src/ server/src/ && uv run ruff format --check src/ server/src/ && uv run ty check --error-on-warning src/ && uv run python -m unittest discover tests`를 실행한다. 기대 결과: 모두 통과
- [ ] **Step 2**: CI를 수정하고 커밋한다. `ci: install ffmpeg for audyssey media tests`
- [ ] **Step 3**: README를 고치고 커밋한다. `docs(readme): add audyssey skeleton quickstart and team guide`
- [ ] **Step 4**: spec을 고치고 커밋한다. `docs(spec): record skeleton implementation decisions`
- [ ] **Step 5**: `git push -u origin feat/skeleton`을 한다. 이어서 `gh pr create --base main --title "feat: add audyssey end-to-end skeleton with stub tools"`로 PR을 연다.
  - 본문: Summary(무엇을 만들었는지), Test plan(실행한 명령과 결과), 담당자별 다음 작업, 마지막 줄 `🤖 Generated with [Claude Code](https://claude.com/claude-code)`
  - **merge하지 않는다.**
- [ ] **Step 6**: `gh pr checks --watch`로 CI 결과를 확인한다. 실패하면 원인을 고치고 `fix(<scope>): ...` 커밋을 추가한다.
