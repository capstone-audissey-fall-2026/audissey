# Audyssey 뼈대(Skeleton) 설계 — Phase 1 최종안

- 날짜: 2026-11-09
- 작성: 광웅(gwpark0741), Claude와 논의
- 상태: **구현 승인됨**. 2026-11-09 skeleton 구현 요청으로 승인.
- 출처: "Agentic Audio Studio — 시스템 구현 명세서 (Draft v0.1)", 팀 모듈 역할 초안(§4), 2026-11-09 논의

**표기**: 논의에서 합의한 결정은 표시하지 않는다. Claude가 정한 결정은 **(제안)**으로 표시한다. 바꾸고 싶으면 그 줄만 고치면 된다. 결정 전체 목록은 §12에 있다.

---

## 0. 한눈에 보기

- 영상을 넣으면 **소리를 내는 개체별 트랙**으로 분리한다.
- 사용자는 트랙을 듣고 남기거나 빼고, 음량과 피치를 조절한 뒤 다시 합친다.
- 오케스트레이터는 **고정 흐름(LangGraph)**이다. 플래닝과 필터링에는 각각 **검증 루프**가 있다.
- 모든 도구는 `stub`/`real`로 바꿔 끼울 수 있다. 기본값은 전부 `stub`이어서 GPU 없이 처음부터 끝까지 돈다.
- 팀원은 `contracts.py`(공통 계약)를 지키면서 자기 도구 폴더만 고친다.
- UI 디자인은 아직 정하지 않았다. 뼈대에는 흐름을 확인하는 **개발용 최소 화면**만 넣는다. UI의 경계는 HTTP API(§8)다.

```
업로드 → 전처리 → 플래닝 ⇄ 플래닝 검증 → [소스별] 분리 ⇄ 필터링 검증 → 나머지 계산 → 타임라인
                                                   ↑                            │
                                                   └── 다시 분리 · 소스 추가 ────┘ (해당 소스만)
타임라인 편집(keep/drop/음량/피치) → 합치기 → 편집된 영상 + 멀티트랙 stems
```

## 1. 목적과 범위

애니메이션 클립이나 웹에서 수집한 짧은 영상에서, **섞여 있는 오디오 중 원하는 소리만 조정하는 도구**를 만든다. **분리된 멀티트랙**과 **편집 후 다시 합친 결과**는 똑같이 중요한 결과물이다.

이번 단계의 목표는 품질이 아니라 **처음부터 끝까지 동작하는 뼈대**다. 이 뼈대를 팀원에게 공유하고, 각자 맡은 모듈을 진짜 구현으로 바꿔 끼운다. 일정은 **오늘내일 안에 공유**하는 것이다.

### 성공 기준
1. 레포를 클론하고 README의 명령 몇 줄만 실행하면, GPU와 API 키 없이도 다음 흐름이 끝까지 돈다.
   - `samples/sample.mp4` 업로드
   - 단계별 진행 상태 확인(검증 루프의 회차 포함)
   - 소스 트랙과 나머지 트랙 확인
   - 편집(drop, 음량, 피치)
   - 편집된 mp4와 stems 내보내기
2. 팀원이 자기 도구 폴더만 고치고 설정을 `real`로 바꾸면, 다른 코드는 건드리지 않아도 전체 흐름이 그대로 돈다.
3. 아무것도 바꾸지 않고 합치면 결과 오디오가 원본과 수치 오차 범위에서 같다. 이 성질은 자동 테스트로 확인한다.
4. stub 모드 E2E 테스트에서 두 검증 루프가 각각 한 번 이상 실제로 돈다.

### 목표가 아닌 것 (Phase 1)
- 분리 품질 튜닝, 플래닝 정확도 개선
- 소리 생성(새 오디오 합성). 명세서에 있지만 미룬다.
- 자연어 편집 입력. 단, 이후에 붙일 수 있게 명령 구조는 지금 만든다(§7.2).
- 30초를 넘는 영상과 오디오 청크 처리
- 사용자 인증, 다중 사용자, DB, 작업 큐
- 최종 UI 디자인(§9)

### 실패로 간주할 것 (기술적으로 돌아가도)
- 특정 서버(sv3-a100)의 경로나 계정이 있어야만 돌아가는 구조
- 팀원이 자기 모듈을 테스트하려고 다른 모든 모델을 GPU에 올려야 하는 구조
- 도구를 바꿀 때 오케스트레이터, API, UI 코드를 고쳐야 하는 구조

## 2. 제약

- **이식성이 최우선이다.**
  - 광웅은 sv3-a100(A100 80GB 1~2장)을 쓰고, 팀원은 RunPod 같은 대여 GPU를 쓴다.
  - 서버 고유 경로를 하드코딩하지 않는다. 모델 ID와 경로, API 주소, 키는 전부 환경변수로 받는다.
  - 모델 가중치는 이미지나 레포에 넣지 않는다. `HF_HOME` 캐시를 쓴다.
    - sv3-a100에서는 공용 캐시 `/home/common/huggingface`를 쓰고, 가중치를 복사하지 않는다.
- **SAM Audio 접근**
  - 공식 `facebook/sam-audio-*`는 HF 승인이 필요하다(SAM License).
  - sv3-a100 공용 캐시에는 비공식 미러 `mrfakename/sam-audio-large-tv`(14GB)와 `mrfakename/sam-audio-judge`(7.9GB)가 있다.
  - 모델 ID는 설정값이며, 기본값은 공식 ID다.
- **입력은 10초 안팎의 영상이다.** **(제안)** 30초를 넘으면 업로드 단계에서 명확한 메시지와 함께 거절한다.
- **기술 스택은 명세서를 따른다**: LangGraph, FastAPI, React, PyTorch, SAM Audio. LLM은 vLLM(OpenAI 호환) 또는 Gemini API를 쓴다.
- **협업 규칙**
  - 레포는 `capstone-audissey-fall-2026/audyssey`이고, main 브랜치는 보호돼 있다.
  - main에 반영하려면 PR을 열고 CI(ruff lint/format, ty, unittest, 웹 빌드/테스트)를 통과해야 한다.

## 3. 시스템 구조

### 3.1 모듈과 담당

| 모듈 | 역할 | 입력 → 출력 | 담당 |
|---|---|---|---|
| Orchestrator / API / UI | 흐름 실행, 명령 처리, 상태 전달, 편집 화면 | 명령·도구 결과 → 다음 행동 | 민정·영우·광웅 |
| Source Planning Tool | 영상과 오디오에서 실제로 들리는 소리 개체와 분리 조건 추론, 프레임과 대조해 검증 | 비디오+오디오(+피드백) → 소스 목록 / 검증 문제 목록 | 도은 |
| Source Separation Tool | 조건에 맞는 소스 분리 | 모노 오디오+조건 → 분리 트랙 | 기존 모델(SAM Audio) |
| Filtering Tool | 다른 소리가 섞였는지, 빠진 소리가 있는지 판정, 불합격이면 재분리 조건 제안 | 원본+트랙+조건 → 평가 결과(+수정 제안) | 윤석 |
| Audio Editing / Re-mixing | drop, 음량, 피치 적용과 재합성, 영상에 붙이기 | Job → 믹스·영상·stems | 미정(뼈대는 광웅) |

### 3.2 오케스트레이션 방식
- **MVP는 고정 흐름이다.** LangGraph StateGraph에 노드와 분기를 코드로 고정한다. LLM이 다음 단계를 고르지 않는다.
- 자연어 입력은 나중에 **자연어를 명령 목록으로 번역하는 LLM 노드 하나**로 붙인다(혼합 방식). 오케스트레이터와 도구는 바꾸지 않는다.

### 3.3 사용자 개입: 최소에서 시작
- 업로드하면 타임라인 구성까지 자동으로 진행하고, 중간에 멈추지 않는다.
- 사용자는 결과를 듣고 명령(§7.2)을 보낸다. 다시 분리와 소스 추가는 **해당 소스만** 다시 실행한다.
- 개입 지점(예: 분리 전에 소스 목록을 확인하는 단계)은 필요하면 나중에 추가한다.

## 4. 핵심 개념

### 4.1 소스 = 소리를 내는 개체 하나
- 같은 개체가 여러 번 소리를 내면 **소스 1개에 구간 여러 개**로 다룬다. 예: 칼 소리 5번 = 소스 1개.
- 개체가 다르면 종류가 같아도 다른 소스다. 예: 캐릭터 A의 대사와 캐릭터 B의 대사.
- 같은 개체를 더 잘게 쪼개지 않는다. 플래닝 프롬프트에 이 규칙을 명시하고, 오케스트레이터는 이름이 같은 소스를 합친다.
- 분리 트랙은 소스당 **1개**이고, **원본과 길이가 같은 48kHz 모노**다.

### 4.2 분리 조건
- 텍스트는 필수다(SAM Audio `descriptions`).
- 구간은 선택이다(positive span anchor).
- 시각 힌트는 선택이다.
- 같은 종류의 개체가 여럿이면 텍스트만으로는 구분하기 어렵다. 이때 구간과 시각 힌트가 개체를 구분하는 핵심이 된다.

### 4.3 시각 힌트는 자동으로 만든다
- 플래닝 도구가 화면에 보이는 소스마다 힌트를 만든다. 사용자가 클릭하지 않는다.
  - 힌트 내용: 시점, 박스, 짧은 시각 설명
- 오케스트레이터의 처리 (2026-11-10 구현)
  - 검증 단계에서 힌트마다 grounding의 `locate`를 호출한다. 미검출·예외는
    `visual_hint_not_found` 문제로 기존 재계획 루프에 전달한다.
  - 힌트가 있으면 grounding의 `mask_video`로 전체 영상을 앞뒤로 추적하고,
    `masks/<source_id>.mp4`를 분리 도구의 video 인자로 넘긴다. 마스크 실패는 해당 소스만 실패한다.
  - SAM Audio는 힌트가 있을 때만 `masked_videos`와 `-tv` 모델로 분리한다.
  - 힌트가 없으면 그라운딩 호출 없이 텍스트와 구간만 쓴다.
- 화면 밖 소리(BGM, 내레이터)는 힌트가 없다.
- grounding은 `stub`/`real`/`remote`이며 기본값은 `stub`이다. stub은 고정 박스,
  real은 고정 SAM3.1 commit과 기존 `facebook/sam3.1` 캐시를 사용한다.
  출력은 원본 프레임 수·fps·해상도를 유지하며 대상 밖과 추적 실패 프레임은 검은색이다.
- CPU 어댑터·HTTP 테스트, 공유 gpu/ 의존성 해석, SAM3 import는 확인했다.
  실제 GPU 추론·메모리·시간과 grounding을 포함한 Docker E2E는 운영자 검증 대기다.

### 4.4 나머지 소리 (residual)
- `residual = 원본 − Σ(소스 트랙)`을 "나머지" 트랙으로 둔다. 스테레오 원본이면 모노 소스를 각 채널에서 뺀다.
- 아무것도 바꾸지 않고 합치면 **원본과 같다**. 소스 합에 나머지를 더하면 원본이 되기 때문이다.
- 멀티트랙에는 깨끗한 소스 트랙을 보여 주고, 나머지는 별도 트랙으로 둔다.
- 소스를 다시 분리하거나 추가할 때마다 나머지를 다시 계산한다.
- **(제안)** 나머지 계산에서 빼는 대상은 트랙이 있는 모든 소스다. keep/drop과는 무관하다. 그래서 drop하면 그 소스만큼 정확히 빠진다. 분리에 실패해 트랙이 없는 소스는 빼지 않는다.
- **(제안)** 나머지도 keep/drop과 gain을 가진다(pitch는 없음). 명령에서는 `source_id="residual"`로 가리킨다.
- **알려진 한계**: 같은 소리가 두 소스에 중복으로 분리되면, 한쪽만 뺄 때 소리가 반만 빠지는 등의 아티팩트가 생긴다. 이 경우를 잡아내는 것이 필터링 도구의 역할이다.

## 5. 검증 루프

두 루프 모두 같은 구조다: **검증 → 실패 이유를 다음 시도에 피드백 → 횟수 한도**.

| | 플래닝 검증 루프 (도은) | 필터링 검증 루프 (윤석) |
|---|---|---|
| 검증하는 것 | 계획한 소스를 실제 프레임과 텍스트에 대조한다. 구간 시점에 그 개체가 보이는지, 박스 위치와 설명이 맞는지 | 분리 트랙에 다른 소리가 섞였는지, 빠진 소리가 있는지(real 구현 후보: SAM Audio Judge) |
| 실패 시 피드백 | `PlanIssue` 목록 → `plan(feedback=…)` | `FilterResult.reason` + `suggestion`(수정된 조건) |
| 다시 실행 | 전체 소스 목록을 다시 플래닝 | 해당 소스만 재분리. suggestion이 없으면 같은 조건에 `attempt`를 1 올린다 |
| 한도 **(제안)** | `AUDYSSEY_MAX_PLAN_ROUNDS=2` | `AUDYSSEY_MAX_SEPARATION_RETRIES=2` (소스당) |
| 한도 도달 시 | 마지막 계획으로 진행. 남은 문제는 UI에 경고로 표시 | 마지막 트랙을 쓰고 `filter.passed=False`로 경고 표시. 쓸지 말지는 사용자가 keep/drop으로 정함 |

- **(제안)** 형식 검사는 `validate_plan` 전에 오케스트레이터가 항상 먼저 한다. 구간이 영상 길이 안에 있는지, 이름이 겹치지 않는지, 텍스트가 비어 있지 않은지를 본다. 의미 검사는 `validate_plan`의 `real` 구현이 맡는다.
- **(제안)** `separate`의 `real` 구현은 `attempt`가 올라갈수록 후보 수(`reranking_candidates`)를 늘리는 데 쓸 수 있다.
- 루프 회차는 진행 상태에 드러난다(예: "플래닝 2/2", "소스 B 재분리 1/2"). 데모에서 "에이전트가 스스로 검증하고 고친다"는 모습이 보이는 부분이다.

## 6. 데이터 계약

모든 모듈이 공유하는 계약은 `src/audyssey/contracts.py` 한 파일에 Pydantic 모델로 둔다. 도구 담당자는 이 파일을 바꾸지 않고 구현만 바꾼다. 계약을 바꾸려면 PR로 팀 합의를 거친다.

**공통 규칙**
- 오디오는 파일 경로로 주고받는다. Job의 모든 경로는 job 폴더 기준 상대 경로다.
- 시간은 초 단위 float다.
- **(제안)** 중간 오디오 파일은 float32 wav로 저장한다.

```
Span         { start: float, end: float }
VisualHint   { time: float, box: [x0, y0, x1, y1]  # 0~1 정규화
               description: str }
SourcePlan   { name: str,
               kind: "dialogue" | "sfx" | "music" | "ambience",
               description: str                  # 분리용 텍스트 조건
               spans: list[Span],
               visual_hint: VisualHint | None }
GroundResult { found: bool, box: [x0, y0, x1, y1] | None, score: float | None }
PlanIssue    { source_name: str | None, kind: str, message: str }
FilterResult { passed: bool, scores: dict[str, float], reason: str,
               suggestion: SourcePlan | None }
Source       { id: str, plan: SourcePlan, track_path: str | None,
               filter: FilterResult | None,
               status: "pending" | "running" | "done" | "failed",
               attempts: int,
               keep: bool = True, gain: float = 1.0, pitch_semitones: float = 0.0 }
Residual     { path: str | None, keep: bool = True, gain: float = 1.0 }
StepStatus   { key: str            # "preprocess" | "plan" | "validate_plan" | "separate:<id>" | "judge:<id>" | "residual" | "render"
               status: "pending" | "running" | "done" | "failed" | "warning",
               round: int, max_rounds: int, message: str }
Job          { id: str, video_path: str, duration: float,
               audio_path: str     # 48k, 원본 채널 유지 — residual·render 기준
               mono_path: str      # 48k 모노 — separate·judge 입력
               peaks_path: str     # 파형 표시용
               sources: list[Source], residual: Residual,
               plan_issues: list[PlanIssue],
               steps: list[StepStatus],
               outputs: RenderOutput | None,
               state: "running" | "ready" | "failed", version: int,
               error: str | None }
RenderOutput { mix_wav: str, video_mp4: str, stems: list[str] }
Command      = Keep | Drop | SetGain | SetPitch | Reseparate | AddSource | Render   # §7.2
```

## 7. 도구 인터페이스와 명령

### 7.1 도구 함수
각 도구 함수의 `stub`, `real`, `remote` 구현은 같은 시그니처를 가진다. grounding은 `locate`와 `mask_video` 두 함수를 제공한다.

**real 구현 (2026-11-10)**: separation은 고정 SAM Audio commit `bb4c6999d2677c7402360e426afc01ddfad6dce0`의 텍스트·positive span API를 사용한다. `attempt` 시드를 적용하고 출력 길이를 맞춘다. filtering은 SAM Audio Judge 예시(윤석 교체 가능)이며 네 점수 중 있는 값을 반환하고 overall ≥ 0.5로 판정한다. suggestion은 None이다. 시각 힌트는 오케스트레이터의 grounding 검증·마스크 생성을 거쳐 연결한다. GPU 의존성은 첫 호출에만 import하고 모델은 캐시·잠금으로 직렬화한다. 실측 상태는 §13을 따른다.

**remote 모드**: `audyssey tool-server`가 분리·필터링 구현을 HTTP로 제공한다.
- `GET /health`는 인증 없이 `{"tools": [...], "impl": "..."}`을 반환한다. `--tool`을 반복해 같은 프로세스에서 여러 도구를 제공할 수 있다.
- 그라운딩 서버의 `POST /locate`: multipart `video`, `hint`(VisualHint JSON) → GroundResult JSON. `POST /mask`는 같은 입력 → mp4 바이트다.
- 분리 서버의 `POST /separate`: multipart `mono`, `video`, `plan`(SourcePlan JSON), `attempt`; 응답은 WAV다.
- 필터링 서버의 `POST /judge`: multipart `mono`, `track`, `plan`; 응답은 FilterResult JSON이다.
- `AUDYSSEY_TOOL_TOKEN`이 있으면 `Authorization: Bearer <token>`을 요구하고, 틀리거나 빠지면 401이다. loopback 이외 주소에 바인딩하려면 토큰이 필수다.
- 메인 서버에서 `AUDYSSEY_SEPARATOR=remote`, `AUDYSSEY_SEPARATOR_URL`, `AUDYSSEY_FILTER=remote`, `AUDYSSEY_FILTER_URL`, 같은 `AUDYSSEY_TOOL_TOKEN`을 설정한다. `AUDYSSEY_REMOTE_TIMEOUT_S` 기본값은 600초(양수)다. URL이 빠지면 서버 시작 시 실패한다.
- 업로드 이름은 사용하지 않고 임시 폴더의 고정 이름으로 저장한다. 프로세스당 모델 호출은 잠금 하나로 직렬 실행한다. HTTP 오류와 타임아웃은 해당 소스만 실패시킨다.

```
plan(video_path, audio_path, duration, feedback: list[PlanIssue]) -> list[SourcePlan]
validate_plan(video_path, audio_path, plans: list[SourcePlan]) -> list[PlanIssue]     # 빈 목록 = 통과
separate(mono_path, plan: SourcePlan, video_path, out_path, attempt: int) -> out_path
judge(mono_path, track_path, plan: SourcePlan) -> FilterResult
locate(video: Path, hint: VisualHint) -> GroundResult
mask_video(video: Path, hint: VisualHint, out: Path) -> Path
render(job: Job, job_dir, include_residual_stem: bool) -> RenderOutput
```

**stub 동작 (제안)**: stub도 진짜와 같은 형식의 결과를 돌려준다. 그리고 검증 루프가 실제로 한 번씩 돌게 만든다.
- `plan`: 첫 계획은 ghost_sound 포함 4개, 피드백 후 3개의 고정 소스를 돌려준다. 이름, 구간, 텍스트가 다 채워져 있다.
- `validate_plan`: ghost_로 시작하는 소스를 문제로 보고한다. plan은 피드백에서 지목한 소스를 제거하므로 기본 설정에서 두 번째 회차에 통과한다.
- `separate`: 모노 원본에서 해당 구간만 남기고 나머지를 0으로 채운다. 원본과 길이가 같다.
- `judge`: character_a의 설명이 " (refined)"로 끝나지 않으면 불합격이며 그 접미사를 붙인 suggestion을 돌려준다. 그 밖에는 합격이다.
- `render`: stub이 없다. 처음부터 진짜 DSP로 만든다.

**render 출력 (제안)**
- `mix_wav`와 `video_mp4`: keep된 소스(gain, pitch 적용)와 keep된 나머지(gain 적용)를 합친 결과다.
- 출력은 `job_dir/out/`에 저장한다.
- `stems`: keep된 소스만 편집값을 적용해 각각 wav로 내보낸다. `include_residual_stem=True`면 나머지도 포함한다.
- 편집 전 원본 분리 트랙은 항상 작업 폴더에 남는다.
- pitch 변경은 길이를 유지해야 한다. 타임라인 정렬이 깨지면 안 되기 때문이다.

### 7.2 명령
버튼과, 나중에 붙일 자연어 번역기가 같은 형식의 명령을 만든다.

```
keep(source_id) / drop(source_id)                # "residual" 포함
set_gain(source_id, gain)                         # 선형 0.0~2.0 (제안)
set_pitch(source_id, semitones)                   # -12~+12 (제안), residual 불가
reseparate(source_id, description?, spans?)       → 그 소스만 separate⇄judge 루프 → residual 재계산
add_source(plan: SourcePlan)                      → 형식 검사 → 그 소스만 separate⇄judge 루프 → residual 재계산
render(include_residual_stem)                     → RenderOutput
```
- keep, drop, gain, pitch는 상태 값만 바꾼다. 실제 적용은 render에서 한다.
- 소스 삭제 명령은 없다. 원하지 않는 소스는 drop한다.

## 8. API (UI와의 경계)

UI 디자인은 정해지지 않았지만, UI가 기대는 것은 이 API뿐이다. 그래서 API를 먼저 고정한다. **(제안)**

```
POST /api/jobs                    multipart(video) → Job        # 업로드 + 자동 흐름 시작
GET  /api/jobs                    → list[Job 요약]
GET  /api/jobs/{id}               → Job
GET  /api/jobs/{id}/events        SSE: Job 스냅샷 (상태가 바뀔 때마다)
POST /api/jobs/{id}/commands      Command → Job | 409(실행 중) | 422(형식 오류)
GET  /api/jobs/{id}/files/{path}  작업 폴더의 오디오·영상·파형 파일
```

## 9. UI — 미결정

- 화면 구성, 상태 그래프의 모양, 타임라인 상호작용은 **UI 담당(민정·영우·광웅)이 따로 정한다**.
- 뼈대에는 성공 기준 1을 확인하기 위한 **개발용 최소 화면**만 넣는다.
  - 업로드
  - 단계 상태 목록(회차와 경고 포함)
  - 트랙 목록(재생, keep/drop, 음량, 피치, 텍스트 수정 후 재분리, 소스 추가)
  - 합치기와 다운로드
- 개발용 화면은 바뀌거나 버려질 수 있다. 바뀌지 않는 것은 API(§8)와 계약(§6)이다.
- 개발용 화면은 `src/audyssey/devui/index.html` 한 장이며 vanilla JS를 사용한다. 빌드와 Node가 필요 없다. 기존 `web/`는 PR #5에서 삭제했다. UI 담당이 최종 화면을 새로 만든다.

## 10. 실행과 운영

- **도구 선택**
  - `AUDYSSEY_PLANNER`에는 `stub` 또는 `real`, `AUDYSSEY_SEPARATOR`, `AUDYSSEY_FILTER`, `AUDYSSEY_GROUNDER`에는 `stub`, `real`, `remote`를 넣는다. 기본값은 `stub`이다.
  - `real`로 선택된 모델은 처음 쓰는 시점에 불러온다. 서버 시작 시 미리 불러오지 않는다.
- **도구 서버**: `AUDYSSEY_TOOL_TOKEN=... uv run --project gpu --locked audyssey tool-server --tool separation --impl real --host 0.0.0.0 --port 8700`. 필터링은 `--tool filtering --port 8701`로 별도 실행한다. separation real과 Judge 예시는 구현되었고 GPU 환경이 없는 HTTP 검증에는 루트 환경의 `--impl stub`을 쓴다. 원격 연결은 TLS 프록시나 SSH 터널로 보호한다.
- **플래닝 real**: `AUDYSSEY_PLANNER_BASE_URL`, `AUDYSSEY_PLANNER_MODEL`, `AUDYSSEY_PLANNER_API_KEY`로 Gemini/vLLM의 OpenAI 호환 API를 직접 부른다. `planning/llm.py`의 `chat_client(settings)`를 쓰고 호출 시 `settings.planner_model`을 전달한다. 주소는 필수, 로컬 서버의 키는 생략할 수 있다. real 본체와 모델 선택은 도은 담당이다.
- **관측**: LangSmith의 `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`로 켠다. 기본은 꺼져 있다. LangGraph 그래프, 네 도구, 명령, LLM 사용량을 기록한다. 명령 입력은 job_id와 명령만 남기며 키·토큰과 오디오·영상 바이트는 기록하지 않는다. 파일 경로와 메타데이터만 남긴다.
- **전처리**: 길이를 검사하고, ffmpeg로 48kHz 원본 채널 오디오와 모노 오디오를 추출하고, 파형 피크를 만든다.
- **시스템 요구사항**: stub 모드에는 Python(uv), ffmpeg/ffprobe가 필요하다. 개발용 화면에는 Node가 필요 없다.
- **저장 (제안)**: DB 없이 작업마다 폴더 하나(`work/jobs/<id>/`, `job.json` 포함)를 쓴다. 서버를 재시작해도 작업이 남는다.
- **실패 처리 (제안)**
  - 도구 예외가 나면 그 소스만 `failed`가 되고, 나머지 소스는 계속 진행한다.
  - preprocess나 plan이 예외를 내면 Job 전체가 실패한다.
  - 검증 루프 한도에 도달하는 것은 실패가 아니라 경고(§5)다.
- **동시성 (제안)**
  - Job마다 실행 잠금을 둔다.
  - 실행 중에 reseparate, add_source, render가 오면 409로 거절한다.
  - keep, drop, gain, pitch는 언제든 받는다.
- **의존성 (제안)**
  - 기본 설치에는 torch가 없다. 그래서 stub 모드는 노트북에서도 설치된다.
  - GPU 도구는 별도 `gpu/` uv 프로젝트(Python 3.12, numpy 1.26)로 둔다. 루트는 `[ui]`, `[llm]`을 쓰며 torch를 설치하지 않는다. PyTorch 계열은 공식 cu128 index로 풀고 남의 버전 제약은 덮어쓰지 않는다.
  - 기존 `server/`의 무거운 고정 의존성(flash-attn 휠 등)은 가져오지 않는다.
- **Docker와 RunPod**: `audyssey-app`(Python 3.13)과 `audyssey-tools`(CUDA 12.8, Python 3.12) 이미지 두 개를 쓴다. compose는 main·separation·judge·grounding·vllm 컨테이너를 띄우고, RunPod는 GPU 이미지 기본 CMD로 세 도구를 한 Pod에서 제공한다. main만 띄울 때는 `docker compose up main`을 쓴다. 가중치는 포함하지 않고 HF 캐시를 읽기 전용 볼륨으로 연결한다. GHCR 워크플로는 main push 또는 수동 실행 시 latest·SHA 태그를 올린다. 상세 설정과 템플릿은 README를 따른다. 빌드와 실측 결과는 §13에 있다.

## 11. 코드 구조 (제안)

```
src/audyssey/
  contracts.py        # §6 — 팀 공통 계약
  settings.py         # AUDYSSEY_* (pydantic-settings)
  graph.py            # LangGraph 흐름, 검증 루프, 명령 처리
  store.py            # Job 상태 ↔ work/jobs/<id>/
  api.py              # §8 API + 개발용 화면 정적 제공
  cli.py              # `audyssey serve`, `audyssey tool-server`
  toolserver.py       # 분리·필터링 HTTP 서버
  devui/index.html    # 빌드 없는 개발용 화면 (Node 불필요)
  tools/
    planning/   stub.py real.py     # 도은 (plan, validate_plan)
    separation/ stub.py real.py remote.py     # SAM Audio
    filtering/  stub.py real.py remote.py     # 윤석 (judge)
    grounding/  stub.py real.py remote.py     # SAM3 (locate, mask_video)
    editing/    render.py           # DSP
tests/                # stub E2E(루프 포함), residual 항등, 계약 검증
```

- 같은 레포에 새 패키지로 만든다. 기존 `src/v2a_inspect`, `server/`, `demo/`, `web/` 등은 **PR #5에서 삭제**했다. 삭제 후에도 git 기록과 `legacy` 원격에 남는다.
- **CI**: 기존 CI가 `src/` 전체에 lint, type 검사, unittest를 돌리므로 새 패키지도 자동으로 포함된다. stub E2E와 원격 도구 테스트를 계속 통과해야 한다.
- **재사용**
  - 그대로 쓰거나 일부 고쳐 쓴다: `preprocessing/video.py`(오디오 추출 추가), `audio_generation/mix.py`(render 참고), `ui/`의 SSE와 정적 제공 구조
  - 버린다: SAM3 추적 툴 레이어(시각 힌트 단계에서 다시 검토), Hunyuan, Kokoro, `visualization/`, `demo/`, `.streamlit/`, `CODEX_HANDOFF.md`, 서버 시작 시 모델을 모두 미리 불러오는 구조

## 12. 결정 기록

**논의에서 합의한 것**
1. 오케스트레이터는 고정 흐름으로 하고, 자연어는 나중에 명령 번역 노드로 붙인다.
2. MVP 입력은 버튼과 슬라이더다. 단, 명령 형식은 자연어 번역을 대비해 만든다.
3. 사용자 개입은 최소로 시작해서 필요한 만큼 늘린다. 다시 분리와 소스 추가는 해당 소스만 실행한다.
4. 소스는 소리를 내는 개체 하나다. 같은 개체는 쪼개지 않고, 다른 개체(캐릭터 A와 B)는 따로 둔다.
5. 시각 힌트(영역)는 플래닝 도구나 에이전트가 자동으로 만든다.
6. 합치면 원본과 비슷하게 들려야 하고, 멀티트랙은 깨끗해야 한다. 그래서 나머지 트랙을 둔다.
7. 플래닝과 필터링 모두 검증 루프를 둔다.
8. 입력 영상은 10초 안팎이다.
9. 생성 기능은 미룬다. 멀티트랙 결과와 합친 결과는 똑같이 중요하다.
10. 특정 서버에 의존하지 않는다. 팀원은 RunPod에서 재현한다.
11. sv3-a100에서는 SAM Audio 가중치를 공용 캐시 미러로 읽어 쓴다.
12. UI는 아직 정하지 않았다.
13. 사용자 작업 상태 변경은 모두 `handle_command`의 명령으로 한다. 버튼·자연어 번역기·에이전트가 같은 명령을 쓰기 위한 필수 설계 원칙이다. 초기 분석 흐름은 기존 그래프가 담당한다.
14. 도구 서버는 HTTP 창구가 없는 분리·필터링 모델에만 쓴다. 플래닝은 OpenAI 호환 주소·모델·키로 직접 부른다.
15. 관측은 Langfuse 대신 LangSmith를 쓴다.
16. 기존 `web/`는 PR #5에서 삭제했다. git 기록과 `legacy` 원격에 남아 있다.

**2026-11-10 SAM Audio·GPU 환경·Docker 결정**
- CI는 GPU 의존성 없이 가짜 torch/sam_audio 모듈로 어댑터를 검증한다. 기존 검사와 Python 3.12 + numpy 1.26 테스트를 함께 유지한다.
- laion-clap의 numpy<2 제약 때문에 GPU 환경을 `gpu/`로 분리한다. audyssey의 numpy 최소값만 1.26으로 낮추고 외부 제약은 덮어쓰지 않는다.
- 이미지는 설치 환경 단위로 만든다. 도구별 컨테이너는 compose로, RunPod는 이미지 하나 = Pod 하나다.
- 설치 환경이 다른 도구가 생기면 그때 환경 여러 개를 담는 합친 이미지를 추가한다.
- 도구 서버는 HTTP 창구가 없는 SAM 계열 모델용이다. LLM은 Gemini 또는 vLLM 공식 이미지로 별도 운영한다.

**2026-11-10 플래너·검증기 기본 구현 결정**
- `planning/real.py`에 도은 담당자가 교체할 수 있는 연결 확인용 기본 구현을 둔다. 성능 검증이나 최종 모델 선택을 의미하지 않는다.
- 기존 `chat_client(settings)`로 Gemini OpenAI 호환 API와 vLLM을 동일하게 호출한다. JPEG 프레임(기본 4장, 설정 1~16장), 각 시각, 영상 길이와 `PlanIssue` 피드백을 보낸다.
- 오디오는 이번 기본형에서 보내지 않고 전사도 하지 않는다. 프레임에서 추정한 소스이므로 실제 가청 여부를 보장하지 않는다. 향후 오디오 텍스트 전달 원칙(§13)은 유지한다.
- JSON 형식 오류는 한 번만 재요청한다. 플래너는 계속 오류이면 실패하며, 개별 형식 오류 소스는 버린다. 유효 소스가 없으면 실패한다. 구간·박스는 범위 안으로 자르고 무효 구간·힌트는 버린다.
- 검증기는 의심스러운 소스만 보고한다. 두 차례 JSON 오류이면 경고를 남기고 빈 이슈 목록으로 진행한다. 시각 힌트의 실존 확인은 별도 SAM3 그라운딩 작업 범위다.
- Python 기본값은 stub으로 유지한다. Compose는 real + 내부 vLLM을 기본값으로 하며, `Qwen/Qwen2.5-VL-3B-Instruct`와 공식 `vllm/vllm-openai:v0.32.0` 이미지를 사용한다. 캐시는 읽기 전용·오프라인이며 vLLM 포트는 호스트에 공개하지 않는다.
- Gemini는 base URL과 `gemini-2.5-flash` 모델 설정으로 전환하고 `.env`의 `GOOGLE_API_KEY`를 Compose에서 planner API key로 전달한다. 실제 키 읽기·호출·이미지 pull·GPU 실측은 운영자에게 남긴다.

**Claude가 정한 것 (제안)**: 30초 상한, 루프 한도 2회, 형식 검사의 위치, 나머지 계산 규칙과 나머지의 keep/gain, stub의 루프 유발 동작, render 출력 규칙, 실패 처리, 동시성, 저장 방식, 의존성 분리, API 형식, 코드 구조, 기존 코드 삭제 시점.

## 13. 미결정 / 확인 필요

**플래너 이식성 규칙**: 영상 파일을 API에 직접 넣지 않고 프레임 이미지로 전달한다. 오디오는 텍스트로 바꿔 전달한다. Gemini와 vLLM 양쪽에서 같은 OpenAI 호환 호출 코드가 동작하도록 유지한다.

| 항목 | 누가, 언제 |
|---|---|
| SAM3 자동 그라운딩과 visual_hint 연결 | CPU 구현·테스트 완료, GPU·Docker 실측 대기 |
| UI 디자인 | UI 담당, 뼈대 공유 후 |
| Audio Editing 담당자 | 팀 |
| 플래닝 real 모델(Qwen3-VL / Qwen3-Omni / Gemini)과 프레임 검증 방법 | 도은 |
| 필터링 합격 기준과 suggestion 생성 방법 | 윤석 |
| SAM Audio 실측(GPU 메모리, 10초 처리 시간, torch 버전 호환성) | 광웅, separation real 작업 시 |
| 팀원의 SAM Audio 공식 승인 또는 미러 사용 | 팀 |

### GPU 런타임 호환성 해결 (2026-11-10)

Codex의 정적 리뷰가 찾은 차단 3건을 다음처럼 해결했다(Claude).

- `huggingface-hub` 2.4는 SAM Audio 로더가 쓰는 `proxies`, `resume_download`를 없앴다.
  `gpu/`에 `huggingface-hub<1.0`을 직접 선언했다(우리 환경의 버전 선택이며, 남의 선언 제약을
  덮어쓴 것이 아니다). 그 결과 transformers 4.57.6, hub 0.36.2로 풀린다.
- PyPI의 torchcodec는 CUDA 13용이라 cu128 torch와 함께 로드되지 않는다. torchcodec를 cu128
  인덱스에서 받고, 링크에 필요한 `nvidia-npp-cu12`를 추가한 뒤 이미지에서 `LD_LIBRARY_PATH`로 연다.
- small 모델 58억 파라미터 중 ranker·span predictor가 약 44억이라 fp32로 24GB GPU를 넘는다.
  재순위 후보 1개, 구간 예측 끔(기본값)일 때는 SAM Audio가 이 모듈들을 호출하지 않고 None/부재를
  공식 처리하므로, 어댑터가 GPU로 옮기기 전에 뺀다.

### SAM3 그라운딩 결정 (2026-11-10)

- 그라운딩은 도구 서버의 독립 도구다. 오케스트레이터가 locate·mask를 조합하고 분리 도구는 그라운딩을 직접 호출하지 않는다.
- `AUDYSSEY_GROUNDER` 기본값은 stub, remote 주소는 `AUDYSSEY_GROUNDING_URL`이다.
  모델 ID·장치는 `AUDYSSEY_SAM3_MODEL=facebook/sam3.1`, `AUDYSSEY_SAM3_DEVICE=cuda`다.
- sam3 commit `2814fa619404a722d03e9a012e083e4f293a4e53`은 기존 gpu/에 제약 덮어쓰기 없이 함께 풀렸다.
  이미지와 RunPod CMD는 세 도구를 제공하며 compose GPU는 `AUDYSSEY_GROUNDING_GPU`로 지정한다.
- 로더는 기존 `sam3.1_multiplex.pt` 캐시만 사용한다(`local_files_only=True`). 가중치 다운로드·복사는 하지 않는다.
- 고정 multiplex predictor dispatcher의 잘못된 init_state 인자를 피하려고 model API를 직접 호출한다.
  optional FA3와 compile/warmup은 끄며 state는 호출 후 해제한다. upstream CUDA 전용 런타임은 운영자 실측으로 확인한다.

### SAM Audio 실측 (2026-11-10)

dnn5, RTX A5000 24GB, 드라이버 580.178.04, `facebook/sam-audio-small` + `facebook/sam-audio-judge`,
`samples/sample.mp4`(7.1초), `HF_HUB_OFFLINE=1`.

| 항목 | 결과 |
|---|---|
| 분리 모델 로딩 / 1회 분리 | 95초 / 6.5초 |
| Judge 로딩 / 1회 판정 | 23초 / 3.6초 |
| 최대 GPU 메모리 (분리+Judge 한 프로세스) | 13.4GB |
| 도구별 GPU 메모리 (도구 서버 각각) | 분리 8.9GB, Judge 8.2GB |
| torch / CUDA 런타임 | 2.11.0+cu128 / 12.8 |
| Judge 점수 | overall 약 1.0~1.85, precision 약 4.5 (범위가 1~5로 보임) |
| judge_pass_threshold | 0.5 (미보정, 이 범위에서는 모두 통과) |
| E2E (도구 서버 2개 remote + main, stub 플래너) | ready, 소스 3개 모두 1회에 통과, render 200 |
| main 이미지 | 881MB |
| sam-audio 이미지 | 14.3GB |
| compose E2E (dnn5: main + 분리 GPU0 + Judge GPU1 컨테이너) | ready, 소스 3개 모두 통과, render 200, stems 4개 |

sv3-a100의 `mrfakename/sam-audio-large-tv` 실측은 GPU를 다른 작업이 쓰고 있어 하지 않았다.

이미지 빌드에서 확인한 것:
- 큰 git 의존성(sam-audio, perception_models, pytorchvideo) 중 하나가 새로 받는 도중 자주 끊긴다(sv3·dnn5 모두).
  Dockerfile이 uv 캐시 마운트를 두고 `uv sync`를 최대 5회 재시도한다.
- Judge는 실행 중 triton으로 GPU 커널을 컴파일하므로 이미지에 `gcc`, `libc6-dev`가 필요하다.
- sv3-a100의 기본 Docker 빌드 엔진은 `/home/common/docker/buildkit/...` 경로 문제로 빌드가 실패한다(서버 설정).

### 전체 E2E 실측 (2026-11-10, dnn5 RTX A5000 24GB × 4)

모든 모듈을 real로 두고 서버 사이 API로 연결해 `samples/sample.mp4`(1920×1080 60fps, 7.1초)를 처리했다.
배치: 분리 GPU0(`facebook/sam-audio-small`), Judge GPU1(`facebook/sam-audio-judge`), SAM3 그라운딩 GPU2
(`facebook/sam3.1`, 추적 2fps), vLLM 플래너 GPU3(`Qwen/Qwen2.5-VL-3B-Instruct`, `vllm/vllm-openai:v0.32.0`).

| 실행 | 플래너 | 결과 |
|---|---|---|
| compose 컨테이너 5개 | vLLM | 306초 ready. 플래닝 루프 2라운드(검증 문제 → 재계획 → 한도 도달 경고), SAM3 위치 확인 2회, 분리·Judge·render 정상. GPU 7.9 / 7.4 / 5.1 / 16.5GB |
| 직접 실행(기본) | vLLM | 229초 ready. 위와 같은 흐름 |
| 직접 실행(루프 유도) | Gemini 2.5 Flash | Judge 기준 1.5로 1172초 ready. 시각 힌트 4개 → SAM3 마스크 영상 → SAM Audio 시각 조건 분리. 재분리 루프 최대 3회 후 경고. 앰비언스만 3.17로 통과 |

플래너 단독 실호출: vLLM은 소스 3개·검증 문제 1개(21.8초/3.9초), Gemini는 소스 6개·검증 문제 1개(20.5초/10.5초).

확인된 한계(성능이 아니라 연결 확인 단계의 기록):
- **SAM3 위치 확인이 약하다.** 힌트 박스를 함께 주므로 없는 대상("ghost")도 found가 된다. 검증 기준은 플래닝 담당과 정한다.
- **SAM3가 추적 중 대상을 놓치면 마스크 전체가 실패한다**(`No points are provided`, 아주 작은 박스에서 발생). 해당 소스만 실패하고 작업은 계속된다.
- **Judge 점수:** overall이 대사·효과음에서 약 1.0, 앰비언스에서 약 3.2로 소스 종류에 따라 크게 다르다. 기준값 0.5는 미보정이며 필터링 담당과 정한다.
- **Qwen2.5-VL-3B 플래너**는 2라운드 뒤 소스를 1개만 남기는 등 품질이 낮다. 연결 확인용 기본 구현이며 교체 대상이다.
- **GPU 이미지 빌드:** 큰 git 의존성 다운로드가 자주 끊겨 Dockerfile이 5회 재시도한다. sv3-a100의 기본 Docker 빌드 엔진은 서버 설정 문제로 빌드가 되지 않는다.

## 14. 다음 단계 (승인 후)

1. 구현 계획 작성
2. **뼈대 PR (오늘내일)**: 계약, stub 도구, LangGraph 흐름과 검증 루프, API, 개발용 화면, 테스트, README의 팀원 시작 가이드
3. 각 담당자가 real 구현 착수. 광웅은 SAM Audio separation real과 Judge 연결 예시를 만든다.
4. 기존 코드 삭제 PR, Docker와 RunPod 가이드
5. 시각 힌트(SAM3), 자연어 명령 번역 노드
