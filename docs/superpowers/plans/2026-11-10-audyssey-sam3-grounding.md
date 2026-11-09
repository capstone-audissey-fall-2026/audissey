# Audyssey SAM3 그라운딩 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 플래너가 만든 시각 힌트(`VisualHint`: 시각, 정규화 박스, 설명)를 SAM3로 처리한다.
- **검증:** 그 시각 프레임에서 대상을 찾는다(`locate`). 못 찾으면 플래닝 루프의 문제(`PlanIssue`)가 된다.
- **분리 입력:** 대상을 영상 전체에서 추적해 마스크 영상을 만든다(`mask_video`). 이 영상이 SAM Audio에 `masked_videos`로 들어간다.

**Architecture:**
- **새 도구 종류 `grounding`:** `stub` / `real` / `remote` 세 구현을 둔다. 다른 도구와 같은 구조다.
- **오케스트레이터(graph)가 조합한다.**
  - 검증 노드는 `validate_plan` 결과에 더해, 힌트가 있는 plan마다 `locate`를 부른다.
  - 분리 단계는 힌트가 있으면 `mask_video`로 만든 영상을 `separate`의 `video` 인자로 넘긴다.
  - 분리 real은 plan에 힌트가 있을 때만 그 영상을 `masked_videos=[...]`로 SAM Audio에 넘긴다.
  - 분리 도구는 그라운딩 도구를 직접 부르지 않는다.
- **도구 서버 확장:** `--tool grounding`으로 `/locate`, `/mask`를 연다. GPU 환경(`gpu/`)과 이미지 `audyssey-sam-audio`에 sam3를 추가한다.
  - 예전 v2a_inspect 서버가 같은 torch 2.11 cu128, numpy 1.26.4, huggingface-hub 0.36.2 조합으로 sam3를 썼으므로 같은 환경에 들어갈 것으로 예상한다.
  - 충돌하면 Global Constraints대로 처리한다.

**Spec:** `docs/superpowers/specs/2026-11-09-audyssey-skeleton-design.md` (§4.3 시각 힌트, §5 검증 루프)

## Global Constraints

- **SAM3 고정 버전:** `sam3 @ git+https://github.com/facebookresearch/sam3.git@2814fa619404a722d03e9a012e083e4f293a4e53`(예전 서버 lock과 같은 커밋)
- **가중치:** `facebook/sam3.1`(`sam3.1_multiplex.pt`)
  - sv3 공용 캐시, dnn5 개인 캐시에 이미 있다.
  - 받지 않는다(`HF_HUB_OFFLINE=1`). 복사하지 않는다.
- **참고 코드(읽기 전용):** `git show a405538^1:server/src/v2a_inspect_server/inference/sam3.py`(예전 서버의 SAM3 영상 추적 구현, `build_sam3_multiplex_video_predictor`). API는 설치된 고정 커밋 소스를 기준으로 판단한다.
- **SAM3가 `gpu/`에 함께 풀리지 않을 때:**
  - 남의 패키지 제약을 덮어쓰지 않는다.
  - 대신 **별도 환경 `gpu-sam3/`와 이미지 `audyssey-sam3`**를 만든다.
  - RunPod용으로 두 환경을 한 이미지에 담고 시작 스크립트로 도구 서버 두 개를 띄우는 **합친 이미지 `audyssey-tools`**를 추가한다.
  - Deviations에 기록한다.
- **CI 기준을 그대로 지킨다.**
  - 명령: `uv run ruff check src/`, `uv run ruff format --check src/`, `uv run ty check --error-on-warning src/`, `uv run python -m unittest discover tests`, numpy 1.26 작업
  - 검사를 끄거나 건너뛰는 것은 금지다.
  - GPU 없는 테스트는 가짜 `sam3`/`torch` 모듈로 어댑터만 검증한다.
- **설정과 기본값:** 모든 설정은 `AUDYSSEY_` 환경변수로 받는다. 기본값은 stub이며, GPU·키·네트워크 없이 기존 E2E가 돌아야 한다.
- **커밋과 브랜치:**
  - 커밋은 `type(scope): lowercase subject` 형식으로 최소 단위로 나눈다. 끝에 빈 줄과 `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`를 붙인다.
  - 브랜치는 `feat/sam3-grounding`(main에서 시작)이다.
  - 머지와 force push는 하지 않는다.
- **막히면 우회하지 않는다.** GPU, Docker, 모델 캐시가 샌드박스에 막히면 명령과 오류를 Deviations에 적고 넘어간다. 실측은 사람이 대신 한다.

## Review Focus

1. **힌트 없는 plan:** `locate`와 `mask_video`를 부르지 않는다. 동작은 지금과 완전히 같다(기존 E2E 결과 불변).
2. **`locate` 실패 처리:** 못 찾으면 `PlanIssue(source_name=..., kind="visual_hint_not_found", message=...)`가 된다. 예외가 나면 그 plan의 issue로 기록하고, 작업 전체를 실패시키지 않는다.
3. **`mask_video` 실패 처리:** 예외가 나면 그 소스만 실패한다(기존 분리 실패 처리와 같다). 마스크 영상은 작업 폴더 안에 `masks/<source_id>.mp4`로 저장한다.
4. **마스크 영상 형식:** 원본과 같은 프레임 수·fps·해상도다. 대상 밖은 검은색이다. 대상을 끝까지 추적하지 못한 프레임도 검은색이다.
5. **좌표 변환:** 박스는 0~1 정규화 좌표이고, 픽셀 변환은 그라운딩 구현 안에서만 한다.

---

### Task 0: 브랜치

- [ ] **Step 1:** `git switch main && git pull --ff-only && git switch -c feat/sam3-grounding`를 실행한다. 이 계획 파일을 커밋한다: `docs: add sam3 grounding implementation plan`

### Task 1: 계약과 설정

**Files:** Modify `src/audyssey/contracts.py`, `src/audyssey/settings.py`; Tests 해당 파일

**Produces:**
- `GroundResult(Model)`: `found: bool`, `box: tuple[float, float, float, float] | None`(정규화), `score: float | None`
- 설정
  - `grounder: Literal["stub", "real", "remote"] = "stub"`
  - `grounding_url: str = ""`
  - `sam3_model: str = "facebook/sam3.1"`
  - `sam3_device: str = "cuda"`

- [ ] Step 1 실패하는 테스트 → Step 2 실패 확인 → Step 3 구현 → Step 4 통과 확인
- [ ] Step 5: 커밋한다: `feat(contracts): add grounding result and settings`

### Task 2: 그라운딩 stub과 도구 로더

**Files:**
- Create `src/audyssey/tools/grounding/__init__.py`, `stub.py`
- Modify `src/audyssey/tools/__init__.py`
- Test `tests/test_audyssey_grounding.py`

**Produces:**
- `locate(video: Path, hint: VisualHint) -> GroundResult`
- `mask_video(video: Path, hint: VisualHint, out: Path) -> Path`
- stub 구현
  - `locate`: `hint.description`에 `"ghost"`가 들어 있으면 `found=False`, 아니면 `found=True, box=hint.box`
  - `mask_video`: ffmpeg로 박스 밖을 검게 칠한 영상을 쓴다. 박스는 고정이고 추적하지 않는다.
- `Tools`에 `locate`, `mask_video`를 추가한다. LangSmith `traceable`로 감싸는 것은 다른 도구와 같다. remote인데 URL이 없으면 fail-fast하는 것도 같다.

- [ ] Step 1 실패하는 테스트
  - `test_stub_locate_finds_unless_ghost`
  - `test_stub_mask_video_keeps_frames_and_blackens_outside`: 프레임 수와 해상도가 같고, 박스 밖 픽셀이 0이다.
- [ ] Step 2 실패 확인 → Step 3 구현 → Step 4 통과 확인
- [ ] Step 5: 커밋한다: `feat(grounding): add stub grounding tool`

### Task 3: 오케스트레이터 연결

**Files:** Modify `src/audyssey/graph.py`, `src/audyssey/tools/planning/stub.py`, `src/audyssey/tools/separation/real.py`; Tests `tests/test_audyssey_graph.py`, `tests/test_audyssey_sam_separation.py`

- **검증 노드:** `validate_plan` 결과에 더해, 힌트가 있는 plan마다 `ctx.tools.locate`를 부른다.
  - 못 찾거나 예외가 나면 `PlanIssue(kind="visual_hint_not_found")`를 추가한다(Review Focus 2).
  - 이 issue도 기존 재계획 루프를 탄다.
- **분리:** 힌트가 있으면 `masks/<source_id>.mp4`를 `mask_video`로 만들고, 그 경로를 `separate`의 `video`로 넘긴다. 실패하면 그 소스만 실패한다.
- **분리 real:** `plan.visual_hint`가 있을 때만 `processor(..., masked_videos=[str(video)])`를 넘긴다.
- **stub 플래너:**
  - `character_a` plan에 시각 힌트를 붙인다: `VisualHint(time=1.0, box=(0.0, 0.0, 0.5, 1.0), description="man on the left")`
  - 1라운드의 `ghost_sound`에도 힌트(`description="ghost"`)를 붙인다. 이렇게 하면 stub E2E에서 locate 실패 → 재계획 루프가 실제로 돈다.
  - 기존 issue 동작은 유지한다.
- [ ] Step 1 실패하는 테스트
  - `test_hint_not_found_triggers_replan`
  - `test_masked_video_passed_to_separate_only_with_hint`
  - `test_mask_failure_fails_only_that_source`
  - `test_real_separate_passes_masked_video_only_with_hint`
  - 기존 E2E 테스트도 통과해야 한다.
- [ ] Step 2 실패 확인 → Step 3 구현 → Step 4 통과 확인(전체 테스트)
- [ ] Step 5: 커밋을 둘로 나눈다.
  - `feat(graph): check visual hints and mask video for separation`
  - `feat(planning): add visual hints to stub plans`

### Task 4: 그라운딩 real (SAM3)

**Files:**
- Modify `gpu/pyproject.toml`, `gpu/uv.lock`
- Create `src/audyssey/tools/grounding/real.py`
- Test `tests/test_audyssey_sam3_grounding.py`

- [ ] **Step 1:** `gpu/`에 고정 sam3를 추가하고 `uv lock`을 돌린다. 예전 서버 lock에 있던 런타임 의존성(예: `opencv-python-headless`, `pycocotools`)은 필요할 때만 추가한다.
  - 풀리지 않으면 Global Constraints의 별도 환경 경로로 간다.
  - 커밋한다: `build(gpu): add sam3 to the gpu tool environment`
- [ ] **Step 2: 어댑터**
  - `torch`, `sam3`는 함수 안에서 `import_module`로 부른다. 로딩은 `@cache`로 1회, 호출은 모듈 잠금으로 한 번에 하나씩이다.
  - `locate`: 힌트 시각의 프레임에서 텍스트(`hint.description`)와 박스로 영역을 찾는다. 가장 점수가 높은 객체의 박스를 정규화해서 돌려준다. 객체가 없으면 `found=False`.
  - `mask_video`: 힌트 시각 프레임에 박스를 넣고 영상 전체로 전파(propagate)한다. 대상 밖을 검게 칠한 mp4를 쓴다(Review Focus 4).
  - 정확한 API는 참고 코드와 설치된 sam3 소스로 확인한다.
- [ ] **Step 3: 실패하는 테스트를 먼저 쓴다** (가짜 모듈 사용)
  - `test_locate_returns_normalized_best_box`
  - `test_locate_not_found`
  - `test_mask_video_writes_same_shape_with_black_outside`
  - `test_model_loaded_once`
  - `test_missing_gpu_env_message`
  - 테스트를 돌려 실패하는지 본 다음, 어댑터를 구현하고 통과시킨다.
- [ ] **Step 4:** 커밋한다: `feat(grounding): locate and mask visual hints with sam3`

### Task 5: 그라운딩 remote와 도구 서버

**Files:**
- Create `src/audyssey/tools/grounding/remote.py`
- Modify `src/audyssey/toolserver.py`, `src/audyssey/cli.py`, `compose.yaml`, `.env.example`, `docker/sam-audio.Dockerfile`(기본 CMD에 `--tool grounding` 추가)
- Tests

- **도구 서버:** `--tool grounding`이면 엔드포인트 두 개를 연다.
  - `POST /locate`: multipart `video` + `hint` JSON → `GroundResult` JSON
  - `POST /mask`: multipart `video` + `hint` → mp4 바이트
- **remote 클라이언트:** 기존 `tools/remote.post`를 재사용한다. 설정은 `AUDYSSEY_GROUNDER=remote`, `AUDYSSEY_GROUNDING_URL`.
- **compose:** `grounding` 서비스를 추가한다. 같은 이미지에 `--tool grounding`, GPU는 `${AUDYSSEY_GROUNDING_GPU:-0}`.
  - main 기본값: `AUDYSSEY_GROUNDER=remote`, `AUDYSSEY_GROUNDING_URL=http://grounding:8700`
  - 모델 ID는 `.env.example`에 `AUDYSSEY_SAM3_MODEL=facebook/sam3.1`로 적는다.
- [ ] Step 1 실패하는 테스트: stub 그라운딩 도구 서버를 띄우고 remote로 locate와 mask가 직접 호출과 같은 결과인지 본다. 토큰 없으면 401인지도 본다.
- [ ] Step 2 실패 확인 → Step 3 구현 → Step 4 통과 확인
- [ ] Step 5: 커밋을 둘로 나눈다.
  - `feat(toolserver): serve sam3 grounding`
  - `build(docker): run grounding as a compose service`

### Task 6: 문서, push, PR

- [ ] **README:** 그라운딩 서비스와 GPU 지정(`AUDYSSEY_GROUNDING_GPU`)을 적는다.
- [ ] **설계 문서:**
  - §4.3에 구현 상태를 적는다.
  - §12에 결정을 추가한다: 그라운딩은 도구 서버의 도구이고, 오케스트레이터가 locate·mask를 조합한다.
- [ ] **확인:** CI 명령 전체를 통과시킨다.
- [ ] **커밋:**
  - `docs(readme): add sam3 grounding guide`
  - `docs(spec): record sam3 grounding decisions`
- [ ] **push와 PR:** `git push -u origin feat/sam3-grounding`, 이어서 `gh pr create --base main`. PR 본문 끝에 `🤖 Generated with [Claude Code](https://claude.com/claude-code)`를 넣는다. 머지하지 않는다.
