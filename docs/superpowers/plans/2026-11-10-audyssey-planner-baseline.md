# Audyssey 플래너·검증기 기본 구현 + vLLM/Gemini 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 플래닝 모듈의 real 구현을 "성능은 낮아도 도는" 기본형으로 만든다.
- 이 기본형은 도은님이 나중에 교체한다.
- 같은 코드가 vLLM(Qwen2.5-VL-3B-Instruct; gemma-3-4b-it는 HF 승인이 없어 403)과 Gemini API(OpenAI 호환 주소) 양쪽에서 돌아야 한다.
- vLLM을 compose 서비스로 추가한다.

**Architecture:**
- `planning/real.py`는 기존 `chat_client(settings)`(OpenAI 호환, LangSmith 기록, 키 가림)만 쓴다.
- 영상은 통째로 넣지 않는다. 프레임 N장을 JPEG data URL로 넣는다. 오디오는 넣지 않는다(설계 문서의 플래너 이식성 규칙).
- 응답은 JSON으로 받아 pydantic으로 검증한다.
- 검증기는 같은 LLM에 "이 소스 목록이 프레임과 맞는가"를 물어 `PlanIssue` 목록을 받는다.
- 시각 힌트의 실존 확인은 이미 오케스트레이터가 SAM3로 하므로(PR #그라운딩) 여기서는 하지 않는다.

**Spec:** `docs/superpowers/specs/2026-11-09-audyssey-skeleton-design.md` (§5 검증 루프, §13 플래너 이식성 규칙)

## Global Constraints

- **새 Python 의존성은 없다.** openai는 `[llm]` extra에 이미 있다.
- **vLLM은 공식 이미지를 쓴다:** `vllm/vllm-openai`. 2주 이상 된 정확한 버전 태그로 고정하고, `latest`는 쓰지 않는다. 버전은 실제로 pull해서 확인한다.
- **비밀값:**
  - Gemini 키는 `.env`의 `GOOGLE_API_KEY`(사용자가 넣어 둠, 권한 600)다.
  - compose에서 `AUDYSSEY_PLANNER_API_KEY`로 넘긴다.
  - 키는 커밋, 로그, 예외 메시지에 넣지 않는다. 테스트에는 가짜 키만 쓴다.
- **Gemini 호출은 검증에 필요한 만큼만 한다.**
  - 실제 호출은 Task 5에서만, 총 30회 이하로 한다.
  - 모델은 `gemini-2.5-flash`.
  - 주소는 `https://generativelanguage.googleapis.com/v1beta/openai/`.
  - 반복 테스트는 가짜 클라이언트나 vLLM으로 한다.
- **CI 기준을 그대로 지킨다.** CI 명령 전체와 numpy 1.26 작업이 통과해야 하고, 검사를 끄거나 건너뛰는 것은 금지다. CI 테스트는 네트워크 없이 가짜 클라이언트로 돈다(`chat_client`의 `transport` 인자나 함수 교체).
- **커밋과 브랜치:**
  - 커밋은 `type(scope): lowercase subject` 형식으로 최소 단위로 나눈다. 끝에 빈 줄과 `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`를 붙인다.
  - 브랜치는 `feat/planner-baseline`(main에서 시작)이다.
  - 머지와 force push는 하지 않는다.
- **막히면 우회하지 않는다.** GPU, Docker, 네트워크가 샌드박스에 막히면 명령과 오류를 Deviations에 적고 넘어간다.

## Review Focus

1. **모델이 깨진 JSON을 줄 때:** 한 번만 다시 묻는다. 그래도 깨지면 `RuntimeError("planner returned invalid JSON")`. 응답 원문은 앞 500자만 메시지에 넣는다.
2. **일부 소스만 형식 오류일 때:** 그 소스는 버리고 나머지를 쓴다. 전부 버려지면 오류다.
3. **좌표와 시간 범위:** span은 영상 길이 안으로 자른다. 박스는 0~1로 자른다. 잘라서 무효가 되면 그 항목을 뺀다.
4. **피드백 반영:** 검증에서 나온 피드백(`PlanIssue`)이 다음 플래닝 프롬프트에 그대로 들어간다. 재계획 루프의 근거가 되므로 테스트로 확인한다.
5. **프레임 시각 표시:** 프레임 추출은 영상 길이에 고르게 N장(기본 4)이다. 각 프레임의 시각(초)을 프롬프트에 함께 적는다. 플래너가 span과 힌트 시각을 정할 근거가 된다.

---

### Task 0: 브랜치
- [ ] `git switch main && git pull --ff-only && git switch -c feat/planner-baseline`를 실행한다. 이 계획을 커밋한다: `docs: add planner baseline implementation plan`

### Task 1: 설정과 프레임 추출
**Files:** `src/audyssey/settings.py`, `src/audyssey/media.py`; Tests

**Produces:**
- `planner_frames: int = Field(default=4, ge=1, le=16)`
- `extract_frames(video: Path, count: int, out_dir: Path) -> list[tuple[float, Path]]`: ffmpeg로 JPEG를 뽑는다. 가로는 최대 512px로 줄인다.

- [ ] Step 1 실패하는 테스트: `test_extract_frames_evenly_with_times`(`tests/audyssey_media.make_video` 사용)
- [ ] Step 2 실패 확인 → Step 3 구현 → Step 4 통과 확인
- [ ] Step 5 커밋: `feat(media): extract evenly spaced frames for planners`

### Task 2: 플래너 기본 구현
**Files:** `src/audyssey/tools/planning/real.py`; Test `tests/test_audyssey_planner_real.py`

- `plan(video, audio, duration, feedback)`:
  1. 프레임을 추출한다.
  2. 메시지를 만든다.
     - system: 역할, 소스 정의(소리를 내는 개체 하나, 같은 개체는 쪼개지 않음), `kind` 4종, 출력 JSON 스키마
     - user: 프레임 이미지와 각 시각, 영상 길이, 피드백 목록
  3. `chat_client(settings).chat.completions.create(model=settings.planner_model, messages=..., response_format={"type": "json_object"})`를 부른다.
  4. 응답을 `{"sources": [SourcePlan, ...]}`로 파싱한다.
  5. Review Focus 1~3을 적용한다.
- 모듈 docstring에 다음을 적는다: "도은 담당 모듈의 기본 구현. 성능이 아니라 연결 확인용. 교체 가능."

- [ ] Step 1 실패하는 테스트(가짜 클라이언트 사용)
  - `test_plan_parses_sources_and_clamps`
  - `test_plan_retries_once_on_bad_json`
  - `test_plan_drops_invalid_items_and_errors_when_none_left`
  - `test_plan_includes_feedback_and_frame_times_in_prompt`
  - `test_plan_sends_images_not_video_or_audio`
- [ ] Step 2 실패 확인 → Step 3 구현 → Step 4 통과 확인
- [ ] Step 5 커밋: `feat(planning): add llm planner baseline`

### Task 3: 검증기 기본 구현
**Files:** 같은 파일; 같은 테스트 파일

- `validate_plan(video, audio, plans)`:
  1. 같은 프레임과 소스 목록 JSON을 보낸다.
  2. `{"issues": [{"source_name": str | null, "kind": str, "message": str}]}`를 받는다.
  3. `PlanIssue` 목록으로 바꾼다.
  4. 깨진 JSON이면 한 번 다시 묻고, 그래도 깨지면 issue 없음으로 처리한다. 검증 실패가 작업을 막지 않게 하기 위해서다. 이 경우 로그를 남긴다.
- 프롬프트에 "의심 가는 것만 보고하라"고 적는다. 이 기본 구현은 엄격도를 설정으로 두지 않는다(루프 유도는 E2E에서 다른 방법으로 한다).

- [ ] Step 1 실패하는 테스트
  - `test_validate_maps_issues`
  - `test_validate_bad_json_returns_no_issues`
- [ ] Step 2 실패 확인 → Step 3 구현 → Step 4 통과 확인
- [ ] Step 5 커밋: `feat(planning): add llm plan validator baseline`

### Task 4: vLLM compose 서비스
**Files:** `compose.yaml`, `.env.example`, `README.md`

- 서비스 `vllm`
  - 이미지는 `vllm/vllm-openai:<고정 버전>`
  - 명령은 `--model ${AUDYSSEY_VLLM_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct} --gpu-memory-utilization ${AUDYSSEY_VLLM_GPU_UTIL:-0.85} --max-model-len 8192`
  - GPU는 `${AUDYSSEY_VLLM_GPU:-0}`
  - HF 캐시를 `/root/.cache/huggingface:ro`로 연결하고, `HF_HUB_OFFLINE=1`
  - 포트는 공개하지 않는다.
- `main` 기본값
  - `AUDYSSEY_PLANNER=${AUDYSSEY_PLANNER:-real}`
  - `AUDYSSEY_PLANNER_BASE_URL=${AUDYSSEY_PLANNER_BASE_URL:-http://vllm:8000/v1}`
  - `AUDYSSEY_PLANNER_MODEL=${AUDYSSEY_PLANNER_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}`
  - `AUDYSSEY_PLANNER_API_KEY=${AUDYSSEY_PLANNER_API_KEY:-${GOOGLE_API_KEY:-}}`
- Gemini로 바꾸려면 `.env`에 base_url과 model 두 줄만 넣으면 된다. `.env.example`과 README에 그 예시를 적는다.
- `docker compose config`가 통과하는지 확인한다.
- [ ] 커밋: `build(docker): add vllm planner service`

### Task 5: 실제 연결 확인
- [ ] **Gemini:** 루트 환경에서 `samples/sample.mp4`로 `plan`을 1회, `validate_plan`을 1회 부른다. 키는 `.env`의 `GOOGLE_API_KEY`에서 읽어 환경변수로만 넘긴다. 결과 소스 수와 걸린 시간을 기록한다.
- [ ] **vLLM:** 가능하면 같은 확인을 한다. GPU나 Docker가 막히면 기록만 한다.
- [ ] 샌드박스에 막히면 Deviations에 적는다.

### Task 6: 문서, push, PR
- [ ] **README:** "플래너를 Gemini/vLLM에 연결하기" 절을 실제 명령으로 갱신한다.
- [ ] **설계 문서:** §12에 결정을 추가한다. 플래너·검증기 기본 구현이 들어갔고, 교체 대상이라는 것.
- [ ] **확인:** CI 명령 전체를 통과시킨다.
- [ ] **커밋:**
  - `docs(readme): document planner baseline with gemini and vllm`
  - `docs(spec): record planner baseline decisions`
- [ ] **push와 PR:** `git push -u origin feat/planner-baseline`, 이어서 `gh pr create --base main`. 머지하지 않는다.
