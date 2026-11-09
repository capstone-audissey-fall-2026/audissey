# Audyssey 원격 도구 모드 + LangSmith 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 분리·필터링 도구를 다른 GPU 컴퓨터의 도구 서버에서 돌리고(remote 모드), 플래너는 OpenAI 호환 주소·모델·키만으로 부르게 하며, LangSmith로 그래프·도구·명령·LLM 토큰을 기록한다.

**Architecture:**
- **도구 서버:** `audyssey tool-server`는 기존 `real.py`/`stub.py` 함수를 HTTP로 감싼 작은 FastAPI 앱이다.
- **remote 모드:** `remote.py`는 같은 함수 시그니처를 지키면서 파일을 HTTP로 보내고 결과를 받는다. 오케스트레이터, API, UI는 바꾸지 않는다.
- **LangSmith:** 환경변수로만 켠다. 도구 호출과 명령은 `traceable`로 감싼다.

**Tech Stack:** FastAPI, uvicorn, httpx, langsmith(이미 설치됨), openai(`[llm]` extra)

**Spec:** `docs/superpowers/specs/2026-11-09-audyssey-skeleton-design.md` (이 계획이 §1, §7, §10, §12, §13을 갱신한다)

## Global Constraints

- **의존성:** 새 의존성은 `openai`(`[llm]` extra에만) 하나다. httpx, fastapi, uvicorn, langsmith는 이미 설치된 것을 쓴다.
- **도구 함수 시그니처는 그대로 둔다:**
  - `separate(mono, plan, video, out, attempt) -> Path`
  - `judge(mono, track, plan) -> FilterResult`
  - `plan(...)`, `validate_plan(...)`
- **모든 작업 상태 변경은 명령(`handle_command`)으로만 한다.** 이 계획은 명령 밖에서 Job을 바꾸는 코드를 추가하지 않는다.
- **설정은 전부 `AUDYSSEY_` 환경변수로 받는다.** 기본값은 stub이고, GPU·키·네트워크 없이 기존 E2E가 그대로 돌아야 한다.
- **비밀값을 남기지 않는다.** 토큰과 API 키는 로그, 예외 메시지, LangSmith 기록 어디에도 남기지 않는다. 오디오·영상 바이트는 LangSmith에 올리지 않고, 경로와 메타데이터만 올린다.
- **CI 기준을 그대로 지킨다.**
  - `uv run ruff check src/`
  - `uv run ruff format --check src/`
  - `uv run ty check --error-on-warning src/`
  - `uv run python -m unittest discover tests`
  - 검사를 끄거나 건너뛰는 코드(`ty: ignore`, `noqa`, `skip`)를 추가하지 않는다. 막히면 코드를 고친다.
- **커밋과 브랜치:**
  - 커밋은 `type(scope): lowercase subject` 형식으로 최소 단위로 나눈다. 끝에 빈 줄과 `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`를 붙인다.
  - 브랜치는 `feat/remote-tools`(main에서 시작)이다.
  - 머지와 force push는 하지 않는다.
- **모델 가중치를 받지 않는다.** 모든 테스트는 stub 구현으로 돈다.

## Review Focus

1. **토큰 없이 외부에 열린 도구 서버:** `--host`가 loopback(`127.0.0.1`, `localhost`, `::1`)이 아닌데 `AUDYSSEY_TOOL_TOKEN`이 비어 있으면 서버를 띄우지 않고 오류로 끝낸다.
2. **틀리거나 빠진 토큰:** 401을 돌려준다. 비교는 `secrets.compare_digest`로 한다. remote 쪽에서는 그 소스만 failed가 되고, 메시지에 상태코드 401이 들어가며 토큰 값은 들어가지 않는다.
3. **도구 서버 오류·타임아웃:** 500, 연결 실패, 타임아웃이 나면 그 소스만 실패한다. 기존 실패 처리를 그대로 탄다. 예외 메시지에는 서버 주소와 상태코드가 들어간다.
4. **remote 모드인데 URL이 비어 있음:** `load_tools` 단계에서 바로 `ValueError`를 낸다. 서버 시작 때 실패해야 하며, 작업 도중에 실패하면 안 된다.
5. **클라이언트가 보낸 파일 이름:** 도구 서버는 업로드 파일 이름을 쓰지 않는다. 임시 폴더 안의 고정 이름(`mono.wav`, `video.mp4`, `track.wav`)으로 저장한다.

---

### Task 0: 브랜치

- [ ] **Step 1:** 시작 조건을 확인한다. `git switch main && git pull --ff-only`를 실행하고, `git log -1`이 PR #5(기존 코드 삭제) 머지 이후인지 본다.
- [ ] **Step 2:** `git switch -c feat/remote-tools`를 실행한다. 이 계획 파일을 커밋한다: `docs: add remote tools implementation plan`

### Task 1: 설정

**Files:** Modify `src/audyssey/settings.py`; Test `tests/test_audyssey_settings.py`

**Produces:**
- `planner: Literal["stub", "real"]` (그대로)
- `separator`, `filter`: `Literal["stub", "real", "remote"]`
- `separator_url: str = ""`, `filter_url: str = ""`
- `tool_token: str = ""`
- `remote_timeout_s: float = Field(default=600, gt=0)`
- `planner_base_url: str = ""`, `planner_model: str = ""`, `planner_api_key: str = ""`

- [ ] **Step 1: 실패하는 테스트**
  - `test_remote_modes_and_urls_from_env`: 다음 환경변수를 패치한다(`mock.patch.dict(os.environ, ...)`).
    - `AUDYSSEY_SEPARATOR=remote`
    - `AUDYSSEY_SEPARATOR_URL=http://gpu:8700`
    - `AUDYSSEY_TOOL_TOKEN=t`
    - `AUDYSSEY_PLANNER_BASE_URL=http://llm/v1`
    - `AUDYSSEY_PLANNER_MODEL=m`
  - 그 상태에서 만든 `Settings()`의 각 필드가 같은 값이고, `filter == "stub"`인지 확인한다.
- [ ] **Step 2:** 실행해서 실패하는지 본다.
- [ ] **Step 3:** 필드를 추가한다.
- [ ] **Step 4:** 실행해서 통과하는지 본다.
- [ ] **Step 5:** 커밋한다: `feat(settings): add remote tool and planner endpoint settings`

### Task 2: 도구 서버

**Files:** Create `src/audyssey/toolserver.py`; Test `tests/test_audyssey_toolserver.py`

**Produces:** `create_tool_app(tool: Literal["separation", "filtering"], impl: Literal["stub", "real"], token: str) -> FastAPI`

**엔드포인트:**
- `GET /health` → `{"tool": tool, "impl": impl}`. 인증이 필요 없다.
- `POST /separate`: separation 서버에만 있다.
  - multipart 필드: `mono`(wav 파일), `video`(파일), `plan`(SourcePlan JSON 문자열), `attempt`(int)
  - 응답: 분리된 wav 바이트(`audio/wav`)
- `POST /judge`: filtering 서버에만 있다.
  - multipart 필드: `mono`, `track`(wav 파일), `plan`(JSON 문자열)
  - 응답: `FilterResult` JSON

**동작:**
- 구현 모듈은 `import_module(f"audyssey.tools.{tool}.{impl}")`로 고른다.
- `token`이 비어 있지 않으면 `/health`를 뺀 모든 요청에 `Authorization: Bearer <token>`을 요구한다. 없거나 틀리면 401이다.
- 잘못된 plan JSON은 422로 거절한다.
- 업로드는 `TemporaryDirectory`에 고정 이름으로 저장한 뒤 구현 함수를 부른다.
- 구현 함수 호출은 전역 `threading.Lock` 하나로 한 번에 하나씩만 돌린다. 주석: `# ponytail: one GPU model, one request at a time; add a queue or batching if throughput matters`

- [ ] **Step 1: 실패하는 테스트** (FastAPI `TestClient`)
  - `test_health`
  - `test_separate_stub_matches_direct_call`: 같은 입력으로 stub `separate`를 직접 부른 결과와 서버 응답 wav가 같다(`np.allclose`).
  - `test_judge_stub_returns_filter_result`: `character_a` plan이면 `passed is False`이고 `suggestion.description`이 `" (refined)"`로 끝난다.
  - `test_token_required`: 토큰 `"t"`로 만든 앱에서 헤더 없이 401, `Bearer x`로 401, `Bearer t`로 200이 나온다.
  - `test_wrong_endpoint_for_tool`: separation 서버에서 `/judge`는 404다.
  - `test_bad_plan_json_is_422`
- [ ] **Step 2:** 실행해서 실패하는지 본다.
- [ ] **Step 3:** 구현한다.
- [ ] **Step 4:** 실행해서 통과하는지 본다.
- [ ] **Step 5:** 커밋한다: `feat(toolserver): serve separation and filtering tools over http`

### Task 3: remote 클라이언트와 도구 로더

**Files:**
- Create `src/audyssey/tools/remote.py` (공용 함수)
- Create `src/audyssey/tools/separation/remote.py`, `src/audyssey/tools/filtering/remote.py`
- Modify `src/audyssey/tools/__init__.py`
- Test `tests/test_audyssey_remote.py`

**Produces:**
- `tools/remote.py`: `post(url: str, path: str, files: dict, data: dict, token: str, timeout: float) -> httpx.Response`
  - `Authorization` 헤더를 붙여 httpx로 POST한다.
  - 2xx가 아니거나 연결 실패, 타임아웃이 나면 `RuntimeError(f"tool server {url}{path} failed: {status or error}")`를 낸다. 응답 본문의 `detail`은 덧붙여도 되지만 토큰은 넣지 않는다.
- `separation/remote.py` `separate(...)`:
  - `Settings()`에서 URL, 토큰, 타임아웃을 읽고 `/separate`를 호출한다.
  - 응답 바이트를 `out`에 쓰고 `out`을 돌려준다.
- `filtering/remote.py` `judge(...)`: `/judge`를 호출하고 응답을 `FilterResult.model_validate`로 바꾼다.
- `load_tools`: separator/filter가 `"remote"`인데 해당 URL이 비어 있으면 `ValueError("AUDYSSEY_SEPARATOR_URL is required for remote mode")`(필터는 `FILTER_URL`)를 낸다.

- [ ] **Step 1: 실패하는 테스트**
  - 준비: `tests/test_audyssey_api.py`가 쓰는 방식대로 실제 uvicorn 서버를 스레드에서 띄운다. 빈 소켓을 쓰고, 끝나면 `should_exit`로 끈다.
    - separation(stub) 도구 서버와 filtering(stub) 도구 서버를 각각 하나씩 띄운다. 둘 다 토큰은 `"t"`다.
  - `test_remote_e2e_matches_stub`: 환경변수 `SEPARATOR=remote`, `FILTER=remote`, 두 URL, `TOOL_TOKEN=t`를 설정한 `Settings`로 `graph.analyze`를 돌린다. 기대 결과는 다음과 같다.
    - 작업 상태는 `ready`다.
    - 소스는 3개이고 모두 `filter.passed`가 참이다.
    - `s1`의 `attempts`는 2다.
    - 손대지 않은 렌더 믹스가 원본과 같다(`np.allclose`, atol 1e-6).
  - `test_wrong_token_fails_only_sources`: `TOOL_TOKEN=x`로 돌린다. 기대 결과는 다음과 같다.
    - 작업은 실패하지 않는다.
    - 각 소스의 상태는 failed다.
    - 메시지에 `401`이 들어가고 `x`(토큰)는 들어가지 않는다. 토큰을 `"secret-token-value"`처럼 긴 값으로 바꿔 검사한다.
  - `test_unreachable_server_fails_sources`: 닫힌 포트의 URL을 쓴다. 소스는 failed이고 메시지에 그 URL이 들어간다.
  - `test_remote_without_url_raises`: `load_tools(Settings(separator="remote"))`가 `ValueError`를 낸다.
- [ ] **Step 2:** 실행해서 실패하는지 본다.
- [ ] **Step 3:** 구현한다.
- [ ] **Step 4:** 실행해서 통과하는지 본다. 기존 테스트도 모두 통과해야 한다.
- [ ] **Step 5:** 커밋을 둘로 나눈다.
  - `feat(tools): add remote separation and filtering clients`
  - `feat(tools): fail fast when a remote tool has no url`

### Task 4: CLI `tool-server`

**Files:** Modify `src/audyssey/cli.py`; Test `tests/test_audyssey_cli.py`

**Produces:** `audyssey tool-server --tool separation|filtering --impl stub|real --host 127.0.0.1 --port 8700`
- 토큰은 `Settings().tool_token`(`AUDYSSEY_TOOL_TOKEN`)에서 읽는다.
- host가 loopback이 아니고 토큰이 비어 있으면 아래 메시지를 출력하고 `typer.Exit(1)`로 끝낸다.
  - `"AUDYSSEY_TOOL_TOKEN is required when binding to a non-loopback host"`

- [ ] **Step 1: 실패하는 테스트**
  - `test_tool_server_help`: 도움말에 `--tool`과 `--impl`이 나온다. 비교 전에 `click.unstyle`로 색상 코드를 지운다.
  - `test_tool_server_refuses_public_host_without_token`: `--host 0.0.0.0`이고 토큰이 없으면 exit code 1이고 위 메시지가 나온다. uvicorn은 실제로 띄우지 않는다. 검사 단계에서 끝나야 한다.
- [ ] **Step 2:** 실행해서 실패하는지 본다.
- [ ] **Step 3:** 구현한다.
- [ ] **Step 4:** 실행해서 통과하는지 본다.
- [ ] **Step 5:** 커밋한다: `feat(cli): add tool-server command`

### Task 5: 플래너용 LLM 클라이언트

**Files:**
- Create `src/audyssey/tools/planning/llm.py`
- Modify `pyproject.toml`: `[project.optional-dependencies] llm = ["openai>=2.29.0"]`. 이미 `llm` extra가 있으면 거기에 추가한다. 이후 `uv lock`을 돌린다.
- Modify `.github/workflows/ci-lint.yaml`: `uv sync`에 `--extra llm`을 추가한다.
- Test `tests/test_audyssey_llm.py`

**Produces:** `chat_client(settings: Settings) -> openai.OpenAI`
- `OpenAI(base_url=settings.planner_base_url, api_key=settings.planner_api_key or "unused")`를 `langsmith.wrappers.wrap_openai`로 감싸 돌려준다.
- `planner_base_url`이 비어 있으면 `ValueError`를 낸다.
- 모델 이름은 호출하는 쪽에서 `settings.planner_model`을 쓴다. 이 함수 docstring에 그렇게 적는다.
- 이 파일은 플래닝 담당(도은)이 `real.py`에서 쓰라고 두는 진입점이다. `real.py` 본체는 이번에 쓰지 않는다.

- [ ] **Step 1: 실패하는 테스트**
  - `test_chat_client_uses_settings`: 반환된 client의 `base_url`이 설정값(끝 `/` 포함 비교)과 같다.
  - `test_chat_client_requires_base_url`
- [ ] **Step 2:** 실행해서 실패하는지 본다.
- [ ] **Step 3:** 구현한다.
- [ ] **Step 4:** 실행해서 통과하는지 본다.
- [ ] **Step 5:** 커밋한다: `feat(planning): add openai-compatible chat client for planners`

### Task 6: LangSmith 기록

**Files:** Modify `src/audyssey/tools/__init__.py`, `src/audyssey/graph.py`; Test `tests/test_audyssey_tracing.py`

**Produces:**
- `load_tools`가 돌려주는 네 함수를 `langsmith.traceable(run_type="tool", name=<plan|validate_plan|separate|judge>)`로 감싼다.
  - stub, real, remote 모두에 적용된다.
  - 시그니처와 동작은 그대로다.
- `handle_command`를 `traceable(name="command", process_inputs=...)`로 감싼다.
  - `process_inputs`는 `{"job_id": ..., "command": command.model_dump()}`만 남긴다. Job 전체와 경로 밖의 데이터는 올리지 않는다.
- 그래프(`analyze`)는 LangGraph가 자동으로 기록하므로 손대지 않는다.
- 켜기는 환경변수(`LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`)로만 한다. 코드에 기본값을 넣지 않는다.

- [ ] **Step 1: 실패하는 테스트**
  - `test_tools_and_command_are_traceable`: `langsmith.run_helpers.is_traceable_function`이 `load_tools(Settings())`의 네 함수와 `graph.handle_command`에 대해 참이다.
  - `test_trace_inputs_exclude_job_body`: `langsmith.run_helpers.tracing_context(enabled=True, client=...)`에 가짜 클라이언트를 넘긴다. 가짜 클라이언트는 `unittest.mock.Mock(spec=langsmith.Client)`로 만든다. 그 상태에서 `handle_command`를 한 번 부르고, 기록된 입력에 `job_id`와 `command`만 있는지 확인한다.
    - 이 방식이 langsmith 0.7.x에서 불안정하면 이 테스트는 `process_inputs` 함수를 직접 부르는 단위 테스트로 바꾼다. 바꾼 사실을 Deviations에 적는다.
- [ ] **Step 2:** 실행해서 실패하는지 본다.
- [ ] **Step 3:** 구현한다.
- [ ] **Step 4:** 실행해서 통과하는지 본다. `LANGSMITH_TRACING`이 없는 기본 상태에서 기존 테스트가 모두 통과해야 한다.
- [ ] **Step 5:** 커밋한다: `feat(observability): trace tools and commands with langsmith`

### Task 7: 문서, push, PR

**Files:** Modify `docs/superpowers/specs/2026-11-09-audyssey-skeleton-design.md`, `README.md`

- [ ] **Step 1: 설계 문서 갱신**
  - §1 "목표가 아닌 것"에서 "도구를 별도 서버로 분리하는 원격 호출 구조"를 지운다.
  - §7.1에 remote 모드를 더한다: 도구 서버 엔드포인트, 인증, 환경변수.
  - §10에 도구 서버 실행 방법을 더한다.
  - §12 결정 기록에 다음을 더한다.
    - (a) 작업 상태 변경은 모두 명령으로 한다(버튼·자연어 번역기·에이전트가 같은 명령을 쓰기 위한 필수 설계 원칙).
    - (b) 도구 서버는 HTTP 창구가 없는 모델(분리·필터링)에만 쓴다. 플래닝은 OpenAI 호환 주소·모델·키로 직접 부른다.
    - (c) 관측은 Langfuse 대신 LangSmith로 한다.
    - (d) web/ 삭제(PR #5).
  - §13에 플래너 이식성 규칙을 적는다: 영상 대신 프레임 이미지를 넣고, 오디오는 텍스트로 바꿔 넣어서 Gemini와 vLLM 양쪽에서 같은 코드가 돌게 한다.
  - Langfuse 재사용 언급은 지운다.
- [ ] **Step 2: README 팀 가이드에 세 절을 더한다**
  - "GPU 서버에서 도구 띄우기"
    - 대여 GPU에서 `AUDYSSEY_TOOL_TOKEN=... uv run audyssey tool-server --tool separation --impl real --host 0.0.0.0 --port 8700`
    - main 서버 쪽 환경변수
  - "플래너를 Gemini/vLLM에 연결하기": 주소·모델·키 예시와 이식성 규칙
  - "LangSmith로 보기": 환경변수 3개. 키가 없으면 아무것도 보내지 않는다는 점.
- [ ] **Step 3: 확인**
  - CI 명령 4개가 모두 통과한다.
  - stub 기본값으로 `uv run audyssey serve` E2E가 그대로 돈다. 샘플 업로드 → ready.
  - 도구 서버 두 개(stub)를 띄우고 remote 모드로 같은 E2E를 한 번 돌린다.
- [ ] **Step 4:** 커밋한다.
  - `docs(spec): record remote tools, command principle and langsmith`
  - `docs(readme): add tool server, planner endpoint and langsmith guide`
- [ ] **Step 5:** `git push -u origin feat/remote-tools`, 이어서 `gh pr create --base main`. PR 본문 끝에 `🤖 Generated with [Claude Code](https://claude.com/claude-code)`를 넣는다. 머지하지 않는다.
