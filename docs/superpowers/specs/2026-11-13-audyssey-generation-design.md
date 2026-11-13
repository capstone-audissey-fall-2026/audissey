# Audyssey 생성 경로 통합 설계 (분리 + 생성)

- 날짜: 2026-11-13
- 작성: gwpark0741, Claude와 논의
- 상태: 구현 완료 (PR #23, #24, #25). 이 문서는 현재 구조에 맞춰 정리한 설계 기록이다.

**표기**: 논의에서 합의한 결정은 표시하지 않는다. Claude가 정한 결정은 **(제안)**으로 표시한다.

---

## 0. 한눈에 보기

- 지금의 분리 시스템은 그대로 두고, **생성 경로**를 하나 더 붙인다. 생성 모델은 팀 내부 모델(internal model)이다.
- **유음 영상**은 분리가 메인이고 생성은 보조다.
  - 업로드하면 지금처럼 분리를 끝까지 돌린다.
  - 사용자가 결과를 보고, 마음에 들지 않는 stem이나 빠진 소리를 **누를 때만** 생성한다.
- **무음 영상**은 생성만 한다. 플래닝·검증이 끝나면 대사가 아닌 소스를 모두 자동으로 생성한다.
- 최종 결과물은 "고른 분리 stem + 생성 stem"으로 된 멀티트랙과, 이를 합친 mixture·영상이다.

```
업로드 → 전처리(유음/무음 판정) → 플래닝 ⇄ 검증
  ├ 유음: [소스별] 분리 ⇄ Judge → residual → 타임라인 ─┬→ 생성으로 바꾸기 / 소스 추가(생성)
  │                                                     └→ 다시 분리 / 소스 추가(분리)
  └ 무음: [소스별, 대사 제외] 생성 ⇄ TargetSel → 타임라인 → 다시 생성 / 소스 추가(생성)
타임라인 편집(keep/drop/음량/피치) → 합치기 → 편집된 영상 + 멀티트랙 stems
```

## 1. 왜 하는가

발표 스토리는 다음과 같다.
1. 지난 학기에는 생성 기반 시스템(v2a_inspect)을 만들었다.
2. 생성만으로는 한계가 있었다. 원본 오디오가 있는 영상이라면 원본을 활용하는 편이 더 고품질이다.
3. 그래서 분리를 메인으로 하되, 분리의 한계를 생성으로 보완하는 통합 시스템을 설계했다.

분리는 원본 오디오의 품질을 넘을 수 없다. 생성이 이 한계를 넘는 수단이다. 생성은 세 경우에 쓴다.
- **대체**: 분리 stem이 노이즈가 많거나 품질이 낮을 때, 같은 설명으로 새로 생성해 바꿔 쓴다.
- **추가**: 원본에 없는 소리를 새로 만든다.
- **복구**: 플래닝이나 분리 과정에서 빠진 소리를 채운다.

기존 시스템과는 **개념만 잇는다.** v2a_inspect의 코드(SAM3 추적 에이전트, TTS/ElevenLabs/Hunyuan 라우팅, 이벤트 편집 UI)는 가져오지 않는다.

## 2. 목표, 비목표, 실패 기준

**목표**
- 유음 영상에서 분리 결과를 보고 원하는 소스를 생성으로 대체하거나 새 소스를 생성으로 추가할 수 있다.
- 무음 영상을 올리면 업로드부터 생성, 편집, 합치기까지 끝까지 돈다.
- 생성 도구도 다른 도구처럼 `stub`/`real`/`remote`로 바꿀 수 있다. stub만으로 GPU 없이 전체 흐름이 돈다.
- 팀원이 지금과 같은 방법(`docker compose pull && up -d`, 가중치는 공용 캐시)으로 바로 실행할 수 있다.

**비목표**
- 대사(dialogue) 생성. TTS도 넣지 않는다. 대사는 분리로만 다룬다.
- 유음 영상에서 생성을 자동으로 돌리는 것. 생성은 사용자가 요청할 때만 한다.
- 15초를 넘는 영상의 생성 품질 보장. 돌아가기만 하면 된다.
- 최종 UI 디자인. 개발용 화면(`devui/`)에 필요한 최소 조작만 넣는다.
- 분리 쪽 동작 변경. 플래너 구간을 SAM Audio에 넘기는 문제는 따로 다룬다(§9).

**실패 기준** (돌아가더라도 실패로 본다)
- 생성 stem이 residual 계산에 들어가 residual에 원본에 없는 소리가 섞인다.
- 무음 영상에서 대사 소스를 억지로 생성한다.
- 생성 가중치가 공개 저장소에 올라간다.
- 기존 분리 흐름의 동작이나 시간이 바뀐다.

## 3. 제약

- 생성 모델의 공개 베이스 코드는 라이선스 표시가 없다. 레포에 복사하지 않고 고정 커밋을 의존성으로 설치한다. 가중치 라이선스는 CC-BY-NC-SA-4.0이다.
- 내부 모델 가중치는 외부에 공개하지 않는다. 팀원만 쓸 수 있는 HF 비공개 조직 저장소 `sogang-audyssey/internal-model`에 둔다.
- 서버는 실행 중에 아무것도 내려받지 않는다. `HF_HOME` 캐시를 오프라인으로 읽는다.
- 생성 도구는 다른 GPU 도구와 같은 `gpu/` 환경(torch 2.11, numpy 1.26)과 `audyssey-tools` 이미지에 들어간다. 의존성 충돌이 없어 이미지를 나누지 않는다.

## 4. 유음/무음 판정

- 전처리에서 판정해 `Job.silent`에 저장한다.
- 오디오 스트림이 없거나, 최대 진폭이 −60 dBFS 미만이면 무음이다(스테레오 원본 기준).
- 무음 영상도 영상 길이만큼의 0으로 된 `audio.wav`·`mono.wav`·`peaks.json`을 만든다. 그래서 플래너 계약, 합치기, 파형 표시를 따로 고치지 않는다.
- 무음이면 residual을 만들지 않는다.

## 5. 데이터 계약 (`contracts.py`)

- `Source.origin: "separated" | "generated"`, `Source.replaces: str | None`(분리 stem을 대체한 생성 소스의 원래 id), `SourceStatus`에 `"skipped"`.
- `Job.silent: bool`.
- 명령 `Generate(source_id, description=None)`
  - 분리 소스 대상: 같은 계획으로 생성 소스를 만들고(`<이름> (generated)`, `replaces` 설정) 원래 소스는 `keep=False`로 끈다. 이미 대체 소스가 있으면 그 소스를 다시 생성한다.
  - 생성 소스 대상: seed를 바꿔 다시 생성한다.
  - 대사, residual 대상은 거절(422).
- `AddSource.method: "separate" | "generate"`. 무음 영상의 `separate`, `generate` + 대사는 거절. `Reseparate`는 생성 소스와 무음 영상에서 거절.

## 6. 흐름 (`graph.py`)

- 노드 구성은 그대로다. 소스별 처리가 `job.silent`/`source.origin`에 따라 분리 또는 생성 루프를 고른다.
- 무음 영상: 플래닝 결과를 `origin="generated"`로 만들고, 대사는 `skipped`, 나머지는 생성 루프를 돈다.
- 생성 루프: 생성 → 채점 → 불합격이면 seed를 바꿔 최대 `1 + AUDYSSEY_MAX_GENERATION_RETRIES`회 → margin이 가장 높은 시도만 `tracks/<id>.wav`로 남긴다.
  - seed = `AUDYSSEY_GEN_SEED + attempts − 1`. 다시 생성해도 attempts를 되돌리지 않아 seed가 바뀐다.
  - 단계 키는 `generate:<id>`, `score:<id>`. 시도별 점수는 단계 메시지와 LangSmith에 남긴다.
- residual은 분리 소스만 원본에서 뺀다(`keep`과 무관). 생성 명령 뒤에는 residual을 다시 계산하지 않는다.
- 합치기는 출처와 상관없이 같다. 생성 stem이 원본보다 수백 샘플 짧으면 겹치는 길이만 더한다.

## 7. 생성 도구

**계약**: `generate(video, plan, out_wav, seed) -> Path`(48kHz 모노, 영상 길이), `score_generation(track, plan, siblings) -> FilterResult`. `stub`/`real`/`remote`를 `AUDYSSEY_GENERATOR`로 고른다. 도구 서버 엔드포인트는 `/generate`, `/score`.

**입력**: 설명 텍스트만 쓴다. 플래너 구간과 시각 힌트는 쓰지 않고 결과를 구간으로 자르지도 않는다. 소리가 날 시점은 모델이 영상을 보고 정한다(플래너 타이밍이 부정확하기 때문).

**real (internal model)**
- 공개 베이스 위에 내부 모델 가중치(생성기, 인코더)를 덮어써 쓴다. 설정: bf16, Euler 25회, CFG 4.5, 부정 프롬프트 없음.
- 15초까지는 한 번에 생성하고, 더 길면 8초 창을 1초 이상 겹쳐 잇고 선형 크로스페이드한다(마지막 창은 끝에 맞춤). 44.1kHz 결과를 48kHz로 바꿔 영상 길이에 맞춘다.
- 모델은 첫 호출 때 한 번 올리고 재사용한다. GPU 호출은 한 번에 하나다.

**채점 (텍스트 TargetSel)**
- LAION-CLAP으로 생성음과 자기·형제 소스 설명의 유사도를 구한다. 자기 설명이 가장 가깝고 margin(자기 − 형제 최댓값) ≥ `AUDYSSEY_GENERATION_MARGIN_THRESHOLD`(기본 0.0)이면 합격.
- 10초가 넘는 트랙은 고정 10초 창(1초 겹침, 끝 정렬)의 임베딩을 평균해 매번 같은 점수가 나오게 한다.
- 형제가 없으면 비교할 대상이 없어 CLAP을 부르지 않고 합격 처리한다.

**stub**: 설명으로 정한 주파수의 사인파를 영상 가운데 1초에 넣는다. `sword_clash`만 첫 seed에서 한 번 불합격해 재생성 루프를 보여 준다.

## 8. 배포와 설정

- `audyssey-tools` 이미지 하나가 도구 4개(분리·Judge·그라운딩·생성)를 담는다. 기본 명령은 넷을 한 프로세스로 띄우고(RunPod), Compose는 같은 이미지로 컨테이너를 도구별로 띄운다(`generation` 서비스 포함).
- 이미지는 모두 공개다. 내부 모델 가중치는 이미지에 넣지 않는다.
- 가중치는 `scripts/download-weights.sh`로 받는다. 내부 모델 저장소는 `sogang-audyssey` 조직 멤버만 받을 수 있다.

| 설정 (`AUDYSSEY_` 접두사) | 기본값 |
|---|---|
| `GENERATOR` / `GENERATOR_URL` | `stub` / 빈 값 (Compose 기본 `remote`) |
| `GEN_BASE_MODEL` | 공개 베이스 저장소 |
| `GEN_MODEL`, `GEN_GENERATOR_FILE`, `GEN_ENCODER_FILE` | `sogang-audyssey/internal-model`와 그 안의 파일 |
| `GEN_DEVICE`, `GEN_STEPS`, `GEN_CFG`, `GEN_SEED` | `cuda`, `25`, `4.5`, `42` |
| `GEN_SINGLE_MAX_S`, `GEN_WINDOW_S` | `15`, `8` |
| `MAX_GENERATION_RETRIES`, `GENERATION_MARGIN_THRESHOLD` | `2`, `0.0` |
| `GENERATION_GPU` | `0` (Compose 전용) |

## 9. 남은 일

- 실제 RunPod 계정에서 한 번 돌려 보기.
- 분리 쪽도 플래너 구간을 SAM Audio에 넘긴다. 부정확한 타이밍이 분리 품질에 주는 영향을 확인한다.
- `GENERATION_MARGIN_THRESHOLD` 보정.
- 실측은 `docs/status.md`에 있다.
