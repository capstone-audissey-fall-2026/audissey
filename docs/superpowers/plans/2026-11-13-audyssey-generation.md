# Audyssey 생성 경로(SelVA + SelSync) 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 지금의 분리 시스템에 SelVA(FT)+SelSync 생성 경로를 붙여, 유음 영상은 "분리 + 요청 시 생성", 무음 영상은 "자동 생성"으로 끝까지 돌게 한다.

**Architecture:**
- 그래프 노드는 그대로 두고, 소스별 처리(`process_source`)가 `source.origin`에 따라 분리 루프 또는 생성 루프를 고른다.
- 생성 도구는 다른 도구처럼 `tools/generation/{stub,real,remote}.py` + 도구 서버 `--tool generation`이다.
- real은 numpy 2 / torch 2.6이 필요해 새 uv 프로젝트 `gen/`과 새 이미지 `audyssey-generation`으로 분리한다.
- residual은 분리 소스만 원본에서 뺀다.

**Tech Stack:** SelVA (`jnwnlee/selva`), laion-clap, torch 2.6, uv, FastAPI, LangGraph, Docker Compose, GHCR

**Spec:** `docs/superpowers/specs/2026-11-13-audyssey-generation-design.md` (이 계획보다 우선한다. 충돌하면 명세를 따르고 Deviations에 적는다.)

## Global Constraints

- PR은 둘로 나눈다. 둘 다 main을 향하고, 머지는 사용자가 한다. main 직접 push, force push 금지.
  - **PR 1** (Task 1~7, 흐름 + stub + 도구 서버 + 화면): 브랜치 `feat/generation` (main에서 시작, 첫 커밋은 명세·계획 문서). GPU와 모델 없이 끝난다. Compose와 이미지는 바꾸지 않는다.
  - **PR 2** (Task 8~10, SelVA real + 배포 + 실측): PR 1이 main에 머지된 뒤 main에서 새 브랜치 `feat/generation-selva`로 시작한다. PR 1이 아직 머지되지 않았으면 시작하지 않고 멈춘다.
- 커밋: 한 변경에 한 커밋, `type(scope): lowercase subject`. 메시지 끝에 빈 줄과 `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- CI 기준: `uv run ruff check src/`, `uv run ruff format --check src/`, `uv run ty check --error-on-warning src/`, `uv run python -m unittest discover tests`, 그리고 기존 numpy-1x 잡. 모두 통과해야 한다.
- 검사 우회 금지: `noqa`, `ty: ignore`, `type: ignore`, skip, 약해진 assert, CI 설정 완화.
- 의존성: 남의 패키지 제약을 덮어쓰지 않는다(`override-dependencies`, `constraint-dependencies` 금지). `gen/`이 풀리지 않으면 멈추고 보고한다. 루트 환경과 `gpu/`에는 의존성을 추가하지 않는다.
- 루트(CI) 환경에는 torch가 없다. real 어댑터는 `torch`, `selva`, `laion_clap`, `huggingface_hub`를 함수 안에서 `importlib.import_module`로 불러오고, 테스트는 가짜 모듈을 `sys.modules`에 넣어 검증한다(`tests/sam_audio_fakes.py` 방식).
- 모델 고정값:
  - SelVA 코드: `selva @ git+https://github.com/jnwnlee/selva@d7d40a992aab58e7cf246055681a657e5d8b4a4d` (fork `gwpark0741/selva@8589976`은 `transformers==4.27.4` 핀만 다르고 평가 환경 4.51.3과 충돌하므로 쓰지 않는다)
  - 베이스 저장소 `jnwnlee/SelVA`: `weights/video_enc_sup_5.pth`, `ext_weights/v1-44.pth`, `ext_weights/synchformer_state_dict.pth`
  - FT 저장소 `sogang-audyssey/selva-ft-selsync`(비공개): `generator/S_O_s43/model_ema_step1000.pth`, `encoder/S_O/best.pt`
  - 생성 설정: variant `large_44k`, `cfg_strength=4.5`, `num_steps=25`, Euler, bf16, 부정 프롬프트 없음
  - CLAP: `lukewys/laion_clap`의 `630k-best.pt` (`amodel` 기본 HTSAT-tiny)
- 서버는 실행 중 내려받지 않는다(`HF_HUB_OFFLINE=1`). 가중치는 `hf_hub_download(..., local_files_only=True)`처럼 캐시에서만 찾는다.
- `contracts.py` 변경은 명세 §5에 적힌 것만 한다. PR 설명에 "계약 변경" 절을 따로 둔다.
- 대사(`kind="dialogue"`)는 생성하지 않는다. 생성에는 `plan.spans`와 `plan.visual_hint`를 쓰지 않고 결과를 구간으로 자르지 않는다.
- 실제 GPU 작업은 Task 10에서만 한다. sv3-a100에서는 GPU 4~6을 쓰지 않는다. 다른 사람의 GPU·이미지·캐시를 건드리지 않는다. 가중치는 `/home/common/huggingface`에만 받고 복사하지 않는다. 비밀값은 출력하거나 커밋하지 않는다.

## Review Focus

- **무음 영상인데 플래너가 대사만 냈을 때**: 생성할 소스가 하나도 없다. 작업은 `ready`로 끝나고 합치기는 무음 영상을 그대로 낸다. → Task 4 `test_silent_job_with_only_dialogue_is_ready`
- **유음 영상에서 같은 분리 소스에 "생성으로 바꾸기"를 두 번**: 생성 소스가 둘이 되지 않고 같은 소스가 다른 seed로 다시 생성된다. → Task 5 `test_generate_twice_reuses_replacement`
- **대체 생성 뒤 사용자가 분리 stem을 다시 켬**: 둘 다 켜진 채 합쳐진다(의도된 동작). residual은 바뀌지 않는다. → Task 5 `test_generate_does_not_change_residual`
- **10초를 넘는 영상**: 창이 영상 끝까지 덮고, 결과 길이가 영상 길이와 같다. → Task 8 `test_windows_cover_long_video`, `test_long_video_output_length`
- **형제 소스가 없는 생성**: margin 없이 합격하고, 재시도하지 않는다. → Task 8 `test_score_without_siblings_passes`

---

## PR 1 (`feat/generation`)

### Task 1: 무음 영상 받기와 판정

**Files:**
- Modify: `src/audyssey/media.py`, `src/audyssey/contracts.py` (`Job.silent`), `src/audyssey/graph.py` (`preprocess`, `residual` 노드)
- Test: `tests/test_audyssey_media.py`, `tests/test_audyssey_graph.py`

**Interfaces:**
- Produces:
  - `media.probe(video: Path) -> tuple[float, int]`: 오디오 스트림이 없으면 채널 수 0을 돌려준다(예외 아님). 비디오 스트림 없음, 길이 오류는 지금처럼 `MediaError`.
  - `media.is_silent(wav: Path, threshold_db: float = -60.0) -> bool`: 최대 절댓값이 `10 ** (threshold_db / 20)` 미만이면 True.
  - `media.write_silence(out: Path, duration: float, channels: int) -> Path`: 48kHz float32 0으로 된 WAV, 샘플 수 `round(duration * 48000)`.
  - `Job.silent: bool = False`

- [ ] **Step 1: 실패하는 테스트 작성**
  - `test_probe_accepts_video_without_audio`: 오디오 없는 영상(ffmpeg `-an`으로 만든 1초 클립)의 `probe`가 `(≈1.0, 0)`.
  - `test_is_silent_threshold`: 진폭 0.0005 사인파 → True, 0.01 → False.
  - `test_preprocess_marks_silent_without_audio_stream`: 오디오 없는 영상으로 분석을 돌리면 `job.silent is True`, `audio.wav`·`mono.wav`·`peaks.json`이 있고 길이가 영상 길이와 같으며, `job.residual.path is None`.
  - `test_preprocess_marks_silent_for_near_silent_audio`: −70 dBFS 오디오 영상 → `job.silent is True`.
  - `test_preprocess_keeps_normal_audio`: 기존 샘플 영상 → `job.silent is False`, residual이 지금처럼 생김.
- [ ] **Step 2: 실행해서 실패 확인** — `uv run python -m unittest tests.test_audyssey_media tests.test_audyssey_graph -v`
- [ ] **Step 3: 구현**
  - `preprocess`: 채널 0이면 `write_silence`로 `audio.wav`(2채널)·`mono.wav`(1채널)를 만들고 `silent=True`. 채널이 있으면 지금처럼 추출한 뒤 `is_silent(mono.wav)`로 판정.
  - `residual` 노드와 `refresh_residual`: `job.silent`이면 아무것도 하지 않고 `job.residual.path = None`.
  - 업로드 API가 오디오 없는 영상을 400으로 거절하지 않는지 확인한다(`check_upload`가 `probe`를 쓰므로 자동으로 풀림).
- [ ] **Step 4: 통과 확인** — 같은 명령. 전체 `uv run python -m unittest discover tests`도 통과.
- [ ] **Step 5: 커밋** — `feat(media): accept silent videos and mark them in preprocess`

### Task 2: 계약 변경

**Files:**
- Modify: `src/audyssey/contracts.py`
- Test: `tests/test_audyssey_contracts.py`

**Interfaces:**
- Produces:
  - `Source.origin: Literal["separated", "generated"] = "separated"`, `Source.replaces: str | None = None`
  - `SourceStatus = Literal["pending", "running", "done", "failed", "skipped"]`
  - `class Generate(Model): type: Literal["generate"] = "generate"; source_id: str; description: str | None = None` — `Command` 유니온에 추가
  - `AddSource.method: Literal["separate", "generate"] = "separate"`

- [ ] **Step 1: 실패하는 테스트** — `test_generate_command_parses`(`parse_command({"type": "generate", "source_id": "s1"})`가 `Generate`), `test_add_source_method_defaults_to_separate`, `test_source_origin_defaults_to_separated`, `test_old_job_json_still_loads`(새 필드가 없는 기존 job JSON이 그대로 읽힘).
- [ ] **Step 2: 실패 확인** — `uv run python -m unittest tests.test_audyssey_contracts -v`
- [ ] **Step 3: 구현** — 위 필드와 타입만 추가한다.
- [ ] **Step 4: 통과 확인** — 같은 명령 + 전체 테스트.
- [ ] **Step 5: 커밋** — `feat(contracts): add source origin and generate command`

### Task 3: 생성 도구 계약, 설정, stub

**Files:**
- Create: `src/audyssey/tools/generation/__init__.py`, `src/audyssey/tools/generation/stub.py`
- Modify: `src/audyssey/tools/__init__.py`, `src/audyssey/settings.py`
- Test: `tests/test_audyssey_tools.py`, `tests/test_audyssey_settings.py`

**Interfaces:**
- Produces:
  - `Tools.generate: Callable[[Path, SourcePlan, Path, int], Path]` — `(video, plan, out_wav, seed)`; 48kHz 모노, 샘플 수 `round(duration * 48000)`
  - `Tools.score_generation: Callable[[Path, SourcePlan, list[SourcePlan], Path | None], FilterResult]` — `(track, plan, siblings, reference_stem)`; `scores`에 `text_correct`(0/1), 형제가 있으면 `text_margin`
  - 설정(`AUDYSSEY_` 접두사, 명세 §8 표와 같은 기본값): `generator: Literal["stub","real","remote"]="stub"`, `generator_url=""`, `selva_base_model="jnwnlee/SelVA"`, `selva_ft_model="sogang-audyssey/selva-ft-selsync"`, `selva_ft_generator_file="generator/S_O_s43/model_ema_step1000.pth"`, `selva_ft_encoder_file="encoder/S_O/best.pt"`, `selva_device="cuda"`, `selva_steps: int = 25`, `selva_cfg: float = 4.5`, `selva_seed: int = 42`, `selva_single_max_s: float = 10`, `selva_window_s: float = 8`, `max_generation_retries: int = Field(2, ge=0)`, `generation_margin_threshold: float = 0.0`
  - `load_tools`: `generator == "remote"`이고 URL이 비면 `ValueError("AUDYSSEY_GENERATOR_URL is required for remote mode")`. 두 함수 모두 LangSmith `traceable`로 감싼다(`generate`, `score_generation`).

- [ ] **Step 1: 실패하는 테스트**
  - `test_stub_generate_length_matches_video`: 결과 WAV 샘플 수가 `round(probe(video)[0] * 48000)`, 모노.
  - `test_stub_score_fails_first_seed_then_passes`: `seed == settings.selva_seed`로 만든 stub 결과는 `passed=False`, `seed + 1`은 `passed=True`.
  - `test_remote_generator_requires_url`
- [ ] **Step 2: 실패 확인**
- [ ] **Step 3: 구현**
  - stub `generate`: 설명 문자열로 정한 주파수(`220 + 55 * (sum(map(ord, description)) % 8)` Hz)의 사인파를 영상 가운데 1초에만 넣는다. 진폭은 `seed == Settings().selva_seed`이면 0.1, 아니면 0.2.
  - stub `score_generation`: 최대 진폭이 0.15 이상이면 합격. `scores={"text_correct": 1.0 or 0.0, "text_margin": peak}`(형제가 없으면 `text_margin` 생략), 불합격 이유 `"stub: first seed"`.
  - stub은 GPU, torch 없이 돈다.
- [ ] **Step 4: 통과 확인**
- [ ] **Step 5: 커밋** — `feat(generation): add generation tool contract, settings and stub`

### Task 4: 생성 루프와 무음 흐름

**Files:**
- Modify: `src/audyssey/graph.py`
- Test: `tests/test_audyssey_graph.py`

**Interfaces:**
- Consumes: Task 1~3
- Produces:
  - `generate_source(job: Job, source: Source, ctx: Context) -> None` — 명세 §6의 생성 루프
  - `process_source(job, source, ctx)`는 `source.origin == "generated"`이면 `generate_source`로 넘긴다
  - `siblings(job: Job, source: Source) -> list[SourcePlan]` — 대사가 아니고, 자기 자신과 `source.replaces` 소스를 뺀 소스들의 계획

- [ ] **Step 1: 실패하는 테스트** (도구는 `Mock`이나 stub)
  - `test_generation_retries_then_keeps_best_margin`: 점수 순서 `[fail margin 0.1, fail margin 0.3, fail margin 0.2]`, 최대 시도 3 → 최종 `track_path`의 내용이 두 번째 시도 결과이고, `source.filter.scores["text_margin"] == 0.3`, 시도 파일은 하나만 남음.
  - `test_generation_seed_counts_attempts`: 세 번 호출된 seed가 `42, 43, 44`.
  - `test_generation_stops_on_pass`: 첫 시도 합격이면 한 번만 호출.
  - `test_silent_job_generates_non_dialogue_sources`: 무음 영상 + stub 플래너 → 플래닝 소스는 모두 `origin="generated"`, 대사 소스는 `status="skipped"`·`track_path is None`이고 단계 메시지에 `dialogue is not generated`, 나머지는 `done`, 분리 도구는 호출되지 않음, residual 없음, `render`가 성공.
  - `test_silent_job_with_only_dialogue_is_ready` (Review Focus)
  - `test_refresh_residual_ignores_generated_sources`: 생성 소스가 있어도 residual = 원본 − 분리 소스들.
  - `test_sound_job_unchanged`: 유음 영상 분석은 생성 도구를 한 번도 부르지 않음.
- [ ] **Step 2: 실패 확인**
- [ ] **Step 3: 구현**
  - 단계 키 `generate:<id>`, `score:<id>`. `score:<id>` 메시지에 시도별 `attempt n: correct=c margin=m`을 이어 붙인다.
  - 최대 시도 = `1 + max_generation_retries`(이 루프 안에서 센다). seed = `selva_seed + source.attempts - 1`(attempts를 먼저 1 올린 뒤).
  - 시도별 파일은 `tracks/<id>.attempt<n>.wav`. 끝나면 가장 높은 `text_margin`(없으면 0)의 파일을 `tracks/<id>.wav`로 옮기고 나머지는 지운다.
  - 무음 영상에서 `plan` 노드가 만드는 소스는 `origin="generated"`. 대사 소스는 `status="skipped"`.
  - 예외 처리와 단계 상태는 기존 분리 루프와 같은 방식(`status="failed"`, `track_path=None`).
- [ ] **Step 4: 통과 확인** — 전체 테스트.
- [ ] **Step 5: 커밋** — `feat(graph): generate sources for silent videos with a scoring loop`

### Task 5: 생성 명령

**Files:**
- Modify: `src/audyssey/graph.py` (`handle_command`)
- Test: `tests/test_audyssey_graph.py`, `tests/test_audyssey_api.py`

**Interfaces:**
- Consumes: Task 2 `Generate`, `AddSource.method`; Task 4 `generate_source`

- [ ] **Step 1: 실패하는 테스트**
  - `test_generate_replaces_separated_source`: 분리 소스 `s1`에 `Generate` → 새 소스 `origin="generated"`, `replaces="s1"`, 이름 `"<s1 이름> (generated)"`, `s1.keep is False`, `s1.track_path` 유지.
  - `test_generate_twice_reuses_replacement` (Review Focus): 두 번째 `Generate(s1)`은 소스 수를 늘리지 않고 같은 생성 소스를 다른 seed로 다시 생성.
  - `test_generate_regenerates_generated_source`: 생성 소스에 `Generate(description="x")` → 설명이 바뀌고 seed가 이전보다 큼(attempts를 되돌리지 않음).
  - `test_generate_does_not_change_residual` (Review Focus): `Generate` 전후 residual 파일 내용이 같고, 분리 stem을 다시 `Keep`해도 residual은 그대로.
  - 거절(`ValueError` → API 422, 기존 명령과 같음): 대사 소스에 `Generate`, `source_id="residual"`, 무음 영상에서 `AddSource(method="separate")`, `AddSource(method="generate", kind="dialogue")`, 생성 소스나 무음 영상에 `Reseparate`.
  - `test_add_source_generate`: 유음 영상에 `AddSource(method="generate")` → `origin="generated"`로 생성, 분리 도구 호출 없음, residual 그대로.
- [ ] **Step 2: 실패 확인**
- [ ] **Step 3: 구현** — 명세 §5 동작 그대로. 생성 명령 뒤에는 `refresh_residual`을 부르지 않는다.
- [ ] **Step 4: 통과 확인**
- [ ] **Step 5: 커밋** — `feat(graph): add generate command and generated add_source`

### Task 6: 원격 호출과 도구 서버

**Files:**
- Create: `src/audyssey/tools/generation/remote.py`
- Modify: `src/audyssey/toolserver.py`, `src/audyssey/cli.py` (`Tool`에 `generation`)
- Test: `tests/test_audyssey_toolserver.py`, `tests/test_audyssey_remote.py`, `tests/test_audyssey_cli.py`

**Interfaces:**
- Produces:
  - `POST /generate`: multipart `video`(파일), form `plan`(SourcePlan JSON), `seed`(int) → `audio/wav`
  - `POST /score`: multipart `track`(파일), `reference`(파일, 선택), form `plan`, `siblings`(SourcePlan 목록 JSON) → `FilterResult` JSON
  - `/health`의 `tools`에 `"generation"`이 들어갈 수 있음

- [ ] **Step 1: 실패하는 테스트** — stub으로 띄운 도구 서버에서 `/generate` 결과 길이, `/score` 결과 형태, 토큰 없으면 401, `AUDYSSEY_MAX_UPLOAD_MB` 초과 시 413, 잘못된 `siblings` JSON은 422. remote 클라이언트가 같은 결과를 돌려주는지(기존 remote 테스트 방식).
- [ ] **Step 2: 실패 확인**
- [ ] **Step 3: 구현** — 기존 `/separate`, `/judge`와 같은 패턴(`save_upload`, `_model_lock`, `parse_plan`). remote는 `tools/remote.post` 재사용.
- [ ] **Step 4: 통과 확인**
- [ ] **Step 5: 커밋** — `feat(toolserver): serve generation tool`

### Task 7: 개발용 화면, 문서, PR 1

**Files:**
- Modify: `src/audyssey/devui/index.html`, `README.md`, `docs/running.md`, `docs/architecture.md`

- [ ] **Step 1: devui** — 명세 §9의 다섯 가지(출처 표시, "생성으로 바꾸기"/"다시 생성", 소스 추가 분리/생성 선택, 무음 안내, 대사 skipped 표시). stub 서버(`uv run audyssey serve`)에서 유음 샘플과 무음 영상(`ffmpeg -an`)으로 직접 눌러 확인한다.
- [ ] **Step 2: 문서** — `docs/architecture.md`에 계약·명령·엔드포인트, `docs/running.md` stub 표에 생성 stub 동작과 설정 표의 §8 설정(real 관련 값은 "PR 2에서 연결"로 표시), `README.md` 구조 그림에 생성 경로.
- [ ] **Step 3: 커밋** — `feat(devui): add generation controls`, `docs: document the generation flow`
- [ ] **Step 4: PR 1** — `feat/generation`을 push하고 `gh pr create`. 제목 `feat: add generation flow with stub tools`. 본문에 요약, **계약 변경** 절, 확인 결과, Deviations, "real SelVA는 PR 2"라는 안내. 끝에 `🤖 Generated with [Claude Code](https://claude.com/claude-code)`. 여기서 멈추고 보고한다.

---

## PR 2 (PR 1 머지 뒤, `feat/generation-selva`)

### Task 8: real 어댑터 (SelVA + SelSync, CLAP 채점)

**Files:**
- Create: `src/audyssey/tools/generation/real.py`, `tests/selva_fakes.py`
- Test: `tests/test_audyssey_selva_generation.py`

**Interfaces:**
- Produces (real.py 안, 테스트 대상):
  - `windows(duration: float, single_max: float, window: float, overlap: float = 1.0) -> list[tuple[float, float]]`
  - `crossfade_join(chunks: list[np.ndarray], spans: list[tuple[float, float]], sample_rate: int) -> np.ndarray`
  - `generate(video, plan, out, seed) -> Path`, `score_generation(track, plan, siblings, reference_stem) -> FilterResult` (Task 3 계약)

- [ ] **Step 1: 실패하는 테스트**
  - `test_windows_short_video`: `windows(9.8, 10, 8) == [(0, 9.8)]`, `windows(10, 10, 8) == [(0, 10)]`
  - `test_windows_cover_long_video` (Review Focus): `windows(20, 10, 8) == [(0, 8), (7, 15), (12, 20)]` — 간격 `window - overlap`, 마지막 창은 끝에 맞춤.
  - `test_crossfade_join_length_and_continuity`: 상수 1.0 조각들을 이으면 길이가 `round(20 * sr)`이고 모든 값이 1.0(선형 크로스페이드 합이 1).
  - `test_loads_ft_generator_and_selsync_encoder`(가짜 모듈): `hf_hub_download`가 FT 저장소의 두 파일과 베이스 저장소 세 파일을 `local_files_only=True`로 요청하고, 인코더는 베이스 `load_weights` 뒤 SelSync `state_dict`를 `strict=False`로 불러오며, 생성기는 FT 파일로 불러옴. 모델 로딩은 두 번째 호출에서 다시 일어나지 않음(`@cache`).
  - `test_generate_uses_eval_settings`: 가짜 `generate`가 `cfg_strength=4.5`, `num_steps=25`, 텍스트 `[plan.description]`, seed로 시드한 `rng`, 부정 프롬프트 없음으로 호출되고, spans를 쓰지 않음.
  - `test_long_video_output_length` (Review Focus): 20초 영상 → 가짜 생성이 3번 호출되고, 출력은 48kHz 모노 `round(20 * 48000)` 샘플.
  - `test_score_text_targetsel`: 가짜 CLAP 임베딩으로 자기 설명이 가장 가까우면 `text_correct=1`, `text_margin = 자기 − 형제 최댓값`, 임계값 미만이면 불합격하고 `reason`이 `"closer to sibling: <이름>"`.
  - `test_score_without_siblings_passes` (Review Focus)
  - `test_score_records_reference_targetsel`: 참조 stem이 있으면 `ref_correct`, `ref_margin`이 `scores`에 들어가고 합격 판정에는 영향 없음.
- [ ] **Step 2: 실패 확인** — `uv run python -m unittest tests.test_audyssey_selva_generation -v`
- [ ] **Step 3: 구현**
  - 기준 코드는 SelVA `demo.py`(`get_my_textsynch('depth1')`, `get_my_mmaudio(model_cfg.model_name)`, `FlowMatching(min_sigma=0, inference_mode='euler', num_steps=...)`, `FeaturesUtils(..., enable_conditions=True, need_vae_encoder=False)`)와 `selva.utils.eval_utils.generate`/`load_video`다. 원본 경로: `/home/gwpark0741/NC/MV2A_compare/audio_eval/models/SelVA/demo.py`.
  - `model_cfg.download_if_needed()`는 부르지 않는다. 파일 경로는 모두 `hf_hub_download`로 캐시에서 얻는다.
  - SelSync 인코더 로딩은 `/home/gwpark0741/NC/mt_eval/experiments/selective_encoder/scr/paper_selva/extract_selva_sync.py:121-123`을 따른다.
  - 창마다 `load_video(video, duration)`를 그 구간으로 잘라 넣는다(영상 자르기는 ffmpeg 임시 파일 또는 프레임 텐서 슬라이스, 빌더 판단). 44.1kHz 결과는 48kHz로 리샘플하고 영상 길이에 맞춰 0 채움 또는 자르기.
  - 채점: `laion_clap.CLAP_Module(enable_fusion=False)` + `load_ckpt(630k-best.pt)`. 오디오는 48kHz 모노 그대로 넣는다. 코사인 유사도.
  - GPU 호출은 모듈 수준 `Lock`으로 한 번에 하나만(다른 real 어댑터와 같음).
- [ ] **Step 4: 통과 확인** — 해당 테스트 + 전체 + ty.
- [ ] **Step 5: 커밋** — `feat(generation): add selva selsync adapter with clap targetsel scoring`

### Task 9: `gen/` 환경, 이미지, Compose, 문서

**Files:**
- Create: `gen/pyproject.toml`, `gen/uv.lock`, `gen/.python-version`, `gen/README.md`, `docker/generation.Dockerfile`
- Modify: `compose.yaml`, `.env.example`, `.github/workflows/publish-images.yaml`, `README.md`, `docs/running.md`
- 확인: Step 4 빌드

- [ ] **Step 1: `gen/` 프로젝트** — `gpu/`와 같은 구조(`package = false`, `audyssey` 경로 의존성, PyTorch 인덱스 explicit). 의존성: `torch`/`torchaudio`/`torchvision` 2.6 계열 CUDA 휠, Global Constraints의 SelVA 커밋, `laion-clap`, `huggingface-hub`. Python은 SelVA가 허용하는 3.12(안 되면 3.11, 이유를 Deviations에). `uv lock --project gen`이 풀리지 않으면 멈추고 보고.
- [ ] **Step 2: Dockerfile** — `docker/tools.Dockerfile` 구조를 따른다(apt 재시도, uv sync 재시도, uid 10001, `HF_HUB_OFFLINE=1`). CMD: `audyssey tool-server --tool generation --impl real --host 0.0.0.0 --port 8700`.
- [ ] **Step 3: Compose와 배포**
  - `generation` 서비스: 이미지 `ghcr.io/capstone-audissey-fall-2026/audyssey-generation:latest`, GPU `${AUDYSSEY_GENERATION_GPU:-0}`, 다른 도구 서비스와 같은 환경·볼륨.
  - main 환경 기본값 `AUDYSSEY_GENERATOR=remote`, `AUDYSSEY_GENERATOR_URL=http://generation:8700`.
  - `.env.example`에 `AUDYSSEY_GENERATION_GPU=0`.
  - `publish-images` 매트릭스에 `generation` 추가(GPU 이미지 디스크 정리 단계 포함). 이미지는 공개로 둔다.
- [ ] **Step 4: 빌드 확인** — sv3에서 `docker buildx build --builder <docker-container 빌더> -f docker/generation.Dockerfile .` 성공, `docker compose config` 성공.
- [ ] **Step 5: 문서**
  - `docs/running.md`: 설정 표에 §8 설정, 가중치 준비에 베이스 3개 + 비공개 FT 2개 내려받기(팀원 각자 토큰, 조직 멤버만 가능), Compose 컨테이너 6개, 생성 관련 GPU 메모리(측정 전이면 "측정 전"), RunPod는 생성 미지원 한계.
  - `README.md` 모듈 표에 생성 담당 행, 실행 방법의 컨테이너 수.
- [ ] **Step 6: 전체 검사 통과 확인 후 커밋** — `build(generation): add gen environment, image and compose service`, `docs: document real generation setup`(두 커밋)

### Task 10: 실제 모델 확인과 PR 2

- [ ] **Step 1: 베이스 가중치 준비** — 공용 캐시에 `jnwnlee/SelVA`의 세 파일이 없으면 받는다(`HF_HOME=/home/common/huggingface`, 토큰은 dnn5의 사용자 토큰을 환경변수로만). FT 저장소 두 파일도 같은 캐시에 받는다. CLAP·DFN5B·flan-t5·BigVGAN이 캐시에 있는지 확인.
- [ ] **Step 2: 단독 실행** — 비어 있는 GPU(4~6 제외)에서 `gen/` 환경으로 real `generate`를 8초·10초 영상에 각각 돌려 시간·최대 GPU 메모리·TargetSel을 기록.
- [ ] **Step 3: Compose E2E** — 6개 컨테이너를 GPU 한 장에 올려, 유음 영상(분리 뒤 `Generate` 1회)과 무음 영상(오디오를 뺀 같은 영상, 자동 생성)을 끝까지 돌리고 `render`까지 확인. 단계별 시간, 최대 GPU 메모리, 점수를 `docs/status.md` 실측 표에 추가하고 커밋(`docs(status): record generation measurements`).
- [ ] **Step 4: PR 2** — `feat/generation-selva`를 push하고 `gh pr create`. 제목 `feat: connect selva selsync generation`. 본문에 요약, 확인 결과, 실측값, Deviations, 새 패키지 `audyssey-generation`이 머지 뒤 공개인지 확인해야 한다는 안내. 끝에 `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.

최종 보고(각 PR마다): PR URL, `git log --oneline main..<브랜치>`, 검사·테스트 결과, CI 상태, (PR 2) 실측값, Deviations.
