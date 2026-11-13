# Audyssey 생성 경로 통합 설계 (분리 + 생성)

- 날짜: 2026-11-13
- 작성: gwpark0741, Claude와 논의
- 상태: 초안. 구현 전 사용자 승인 필요
- 기준 코드: main `37821ae`

**표기**: 논의에서 합의한 결정은 표시하지 않는다. Claude가 정한 결정은 **(제안)**으로 표시한다.

---

## 0. 한눈에 보기

- 지금의 분리 시스템은 그대로 두고, **생성 경로**를 하나 더 붙인다. 생성 모델은 SelVA(FT) + SelSync다.
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
- 10초를 넘는 영상의 생성 품질 보장. 돌아가기만 하면 된다.
- RunPod 구성 갱신. RunPod 검증은 별도로 미뤄져 있다(§9).
- 최종 UI 디자인. 개발용 화면(`devui/`)에 필요한 최소 조작만 넣는다.
- 분리 쪽 동작 변경. 플래너 구간을 SAM Audio에 넘기는 문제는 따로 다룬다(§11).

**실패 기준** (돌아가더라도 실패로 본다)
- 생성 stem이 residual 계산에 들어가 residual에 원본에 없는 소리가 섞인다.
- 무음 영상에서 대사 소스를 억지로 생성한다.
- 생성 가중치가 공개 저장소에 올라간다.
- 기존 분리 흐름의 동작이나 시간이 바뀐다.

## 3. 제약

- 생성 실행 환경은 torch 2.6, numpy 2다. SAM Audio 환경(`gpu/`, numpy 1.x)과 같이 둘 수 없으므로 **별도 uv 프로젝트와 별도 Docker 이미지**로 만든다.
- SelVA 코드 원본(`github.com/jnwnlee/selva`)에는 라이선스 표시가 없다. 우리 레포에 코드를 복사하지 않고, 특정 커밋을 의존성으로 설치한다. 가중치는 CC-BY-NC-SA-4.0이다.
- FT 가중치는 외부에 공개하지 않는다. 팀원만 쓸 수 있는 HF 비공개 조직 저장소에 둔다(§8).
- 서버는 실행 중에 아무것도 내려받지 않는다. 지금처럼 `HF_HOME` 캐시를 오프라인으로 읽는다.
- A100 80GB 한 장에 5+1개 컨테이너가 모두 올라가야 한다. 지금 측정값은 52.9GB이고, SelVA 몫은 10~12GB로 추정하지만 아직 측정하지 않았다.
- 코드 규칙은 기존과 같다(`docs/contributing.md`). 변경은 최소로 하고, 테스트를 먼저 쓰고, 검사를 우회하지 않는다.

## 4. 유음/무음 판정

- 전처리에서 판정해 `Job.silent: bool`에 저장한다.
- 무음으로 보는 경우 (제안):
  - 오디오 스트림이 없을 때
  - 오디오 스트림이 있지만 최대 진폭이 −60 dBFS 미만일 때
- 지금 `media.probe`는 오디오 스트림이 없으면 업로드를 거절한다(`video has no audio stream`). **오디오 없는 영상도 받도록 바꾼다.**
  - 오디오 채널 수는 없으면 0으로 돌려준다.
  - 업로드 검증(길이 상한, 비디오 스트림 존재)은 그대로 둔다.
- 무음 영상도 `audio.wav`, `mono.wav`, `peaks.json`을 영상 길이만큼의 0으로 만든다 (제안). 그러면 플래너 계약, 합치기, 파형 표시를 고치지 않아도 된다.
- 무음이면 residual은 만들지 않는다(`job.residual.path = None`).

## 5. 데이터 계약 변경 (`contracts.py`)

`contracts.py`는 팀 합의 대상이다. PR 설명에 바뀐 부분을 따로 적는다.

**Source**에 필드를 추가한다.
- `origin: Literal["separated", "generated"] = "separated"`: 이 stem이 어디서 왔는지.
- `replaces: str | None = None`: 분리 stem을 대체하려고 만든 생성 소스라면 원래 소스 id.
- `SourceStatus`에 `"skipped"`를 추가한다. 무음 영상의 대사 소스처럼, 처리하지 않기로 한 소스에 쓴다.

**Job**에 필드를 추가한다.
- `silent: bool = False`

**명령**
- 새 명령 `Generate`:
  ```python
  class Generate(Model):
      type: Literal["generate"] = "generate"
      source_id: str
      description: str | None = None
  ```
  - 대상이 분리 소스면 **대체**다. 같은 계획(설명은 바꿀 수 있음)으로 생성 소스를 새로 만든다.
    - 새 소스는 `origin="generated"`, `replaces=<원래 id>`이고, 이름은 `<원래 이름> (generated)`다.
    - 원래 분리 소스는 `keep=False`로 끄되 지우지 않는다. 사용자가 다시 켤 수 있다.
    - 같은 분리 소스를 대체하는 생성 소스가 이미 있으면 새로 만들지 않고 그 소스를 다시 생성한다 (제안).
  - 대상이 생성 소스면 **다시 생성**이다. 그 자리에서 seed를 바꿔 다시 생성한다.
  - 대상이 대사(`kind="dialogue"`)면 거절한다(`ValueError`).
  - 대상이 `residual`이면 거절한다.
- `AddSource`에 `method: Literal["separate", "generate"] = "separate"`를 추가한다.
  - 무음 영상에서 `separate`는 거절한다.
  - `generate`이면서 `kind="dialogue"`면 거절한다.
- `Reseparate`는 생성 소스와 무음 영상에서 거절한다.
- `Keep`, `Drop`, `SetGain`, `SetPitch`, `Render`는 출처와 상관없이 그대로 동작한다.

**생성 채점 결과**는 기존 `FilterResult`를 그대로 쓴다(§7). 새 타입을 만들지 않는다.

## 6. 흐름 변경 (`graph.py`)

**분석 그래프**
- 노드 구성은 바꾸지 않는다. `separate` 노드 안의 소스별 처리(`process_source`)가 `job.silent`와 `source.origin`에 따라 분리 또는 생성 루프를 고른다.
- 무음 영상:
  - 플래닝·검증은 지금과 같다(플래너는 이미 프레임만 본다).
  - 플래닝 결과의 소스는 `origin="generated"`로 만든다.
  - 대사 소스는 `status="skipped"`, `track_path=None`으로 두고 단계 상태에 "대사는 생성하지 않음"을 남긴다.
  - 나머지 소스는 생성 루프를 돈다.
  - `residual` 노드는 무음이면 아무것도 하지 않는다.
- 유음 영상: 지금과 완전히 같다.

**생성 루프** (소스 하나)
```
시도 = 0
반복:
  시도 += 1
  track = generate(video, plan, out, seed=시도에서 정한 값)
  result = score_generation(track, plan, 형제 계획들, 참조 stem)
  기록: 시도별 결과
  result.passed 이거나 시도 >= 최대 시도면 멈춤
최종 track = margin이 가장 높은 시도의 결과
```
- 최대 시도는 `1 + AUDYSSEY_MAX_GENERATION_RETRIES`다. 기본값은 2회 재시도다 (제안: 분리와 같음).
- seed는 `AUDYSSEY_SELVA_SEED + (source.attempts − 1)`이다. 기본 seed는 42다 (제안). 같은 입력이면 같은 결과가 나와야 재현할 수 있다.
  - 생성 소스의 `attempts`는 "다시 생성" 명령에서도 0으로 되돌리지 않고 계속 센다 (제안). 되돌리면 같은 seed가 반복되어 다시 생성해도 결과가 같다.
  - 한 번의 생성 루프에서 쓰는 최대 시도 수는 그 루프 안에서 센다.
- 이전 시도 파일은 지우고, 가장 좋은 시도만 `tracks/<id>.wav`로 남긴다 (제안).
- 시도별 점수는 계약에 새 필드를 만들지 않는다 (제안). 가장 좋은 시도의 결과만 `source.filter`에 넣고, 각 시도의 점수는 `score:<id>` 단계 메시지와 LangSmith 기록에 남긴다.
- 단계 키는 `generate:<id>`, `score:<id>`다. 분리 쪽의 `separate:<id>`, `judge:<id>`와 같은 형식이다.
- 형제 계획은 같은 job에서 대사가 아니고, 자기 자신과 자기가 대체하는 소스를 뺀 소스들의 계획이다.
- 참조 stem은 대체 생성일 때 원래 분리 stem 경로다. 그 밖에는 없다.

**residual**
- `refresh_residual`은 `origin == "separated"`이고 `track_path`가 있는 소스만 원본에서 뺀다.
- 지금처럼 `keep`과 상관없이 뺀다. 그래서 분리 stem을 끄고 생성 stem을 켜도 원본의 그 소리가 residual에 남지 않는다.
- 생성 명령 뒤에는 residual을 다시 계산하지 않는다. 생성은 residual에 영향을 주지 않기 때문이다.

**합치기**: 변경 없음. 생성 stem도 다른 stem처럼 `keep`, `gain`, `pitch`를 적용해 합친다.

## 7. 생성 도구

### 7.1 도구 계약 (`tools/__init__.py`의 `Tools`)

```python
generate: Callable[[Path, SourcePlan, Path, int], Path]
# (video, plan, out_wav, seed) -> out_wav. 48kHz 모노, 영상 길이와 같은 샘플 수.

score_generation: Callable[[Path, SourcePlan, list[SourcePlan], Path | None], FilterResult]
# (track, plan, siblings, reference_stem) -> FilterResult
```

- 설정 `AUDYSSEY_GENERATOR: stub | real | remote` (기본 `stub`)와 `AUDYSSEY_GENERATOR_URL`을 추가한다. 두 함수는 같은 모듈(`tools/generation/{stub,real,remote}.py`)에 둔다.
- 생성에는 플래너 구간(spans)과 시각 힌트를 **쓰지 않는다.** 플래너 타이밍이 부정확해서, 소리가 날 시점은 SelVA가 영상을 보고 정하게 한다. 넘기는 것은 설명 텍스트(`plan.description`)뿐이다.
- 결과를 구간으로 자르지 않는다.

### 7.2 real: SelVA(FT) + SelSync

**모델 구성**
- 공식 SelVA `large_44k`를 바탕으로, 두 가중치를 바꿔 끼운다.
  - 생성기: FT8 `S_O` seed 43의 EMA 가중치(`model_ema_step1000.pth`, 4.1GB)
  - 텍스트-싱크 인코더: 공식 `video_enc_sup_5.pth`를 불러온 뒤, SelSync `S_O/best.pt`의 `state_dict`를 `strict=False`로 덮어쓴다(78MB, 학습한 파라미터만 들어 있음).
- 기준 구현은 SelVA의 `demo.py`와 `selva/utils/eval_utils.py`의 `generate()`다. 원본 기준 경로는 `/home/gwpark0741/NC/MV2A_compare/audio_eval/models/SelVA/` (fork `github.com/gwpark0741/selva` @ `8589976`).
- SelSync 학습과 평가 코드는 `/home/gwpark0741/NC/mt_eval/experiments/selective_encoder/`에 있다. 인코더를 불러오는 방법은 `scr/paper_selva/extract_selva_sync.py:121-123`을 따른다.
- 생성 설정은 평가 때와 같다: `cfg_strength=4.5`, `num_steps=25`(Euler flow matching), bf16, 부정 프롬프트 없음, 영상은 25fps 224×224(싱크).
- 출력은 모노 44.1kHz이고, 48kHz로 리샘플해 영상 길이에 맞춘다(짧으면 0으로 채우고 길면 자름).
- 모델은 첫 요청 때 한 번 올리고 프로세스가 끝날 때까지 재사용한다(다른 도구와 같은 `@cache` 방식).

**길이 처리**
- 영상이 `AUDYSSEY_SELVA_SINGLE_MAX_S`(기본 10초) 이하면 영상 전체 길이로 한 번 생성한다.
- 더 길면 길이 `AUDYSSEY_SELVA_WINDOW_S`(기본 8초) 창을 겹침 1초로 이어 영상 전체를 덮고, 겹친 구간은 선형 크로스페이드로 잇는다 (제안).
- 구현 첫 단계에서 8초와 10초 단일 생성 품질을 비교해, 10초 기본값이 괜찮은지 기록한다(§10).

**채점 (텍스트 TargetSel)**
- LAION-CLAP으로 생성 stem의 오디오 임베딩과, 자기 설명 및 형제 설명들의 텍스트 임베딩 사이 코사인 유사도를 구한다.
- 점수(`FilterResult.scores`):
  - `text_correct`: 자기 설명이 가장 가까우면 1, 아니면 0
  - `text_margin`: 자기 설명과의 유사도 − 형제 설명 가운데 가장 높은 유사도
  - 형제가 없으면 margin은 기록하지 않고 `text_correct=1`로 둔다 (제안).
- 합격: `text_correct == 1`이고 `text_margin >= AUDYSSEY_GENERATION_MARGIN_THRESHOLD`(기본 0.0, 아직 보정 전).
- 참조 stem이 있으면(대체 생성) **분리 stem 기준 TargetSel**도 기록만 한다. 합격 판정에는 쓰지 않는다.
  - 생성 stem과 각 분리 stem의 CLAP 오디오 임베딩 유사도를 구하고, 자기가 대체하는 분리 stem이 가장 가까운지(`ref_correct`)와 margin(`ref_margin`)을 남긴다.
  - 비교 대상은 같은 job의 분리 stem 전체다. 오케스트레이터가 `reference_stem`만 넘기므로, 형제 분리 stem 목록이 필요하면 계약에 인자를 추가해도 된다(빌더 판단).
- CLAP 체크포인트는 분리 쪽 후보 고르기와 같은 `lukewys/laion_clap`의 `630k-best.pt`다(이미 공용 캐시에 있음).
- 불합격 이유(`reason`)는 "closer to sibling: <형제 이름>"처럼 짧게 남긴다.

### 7.3 stub

- `generate`: 영상 길이만큼의 짧은 합성음을 만든다. 예를 들어 설명 문자열에서 정한 주파수의 사인파를 영상 중간 1초에만 넣는다 (제안). GPU와 모델이 필요 없어야 한다.
- `score_generation`: 처음 생성한 소스 하나는 한 번 불합격시켜 재생성 루프가 돌게 하고, 나머지는 합격시킨다. 분리 쪽 stub Judge와 같은 방식이다.

### 7.4 remote와 도구 서버

- `toolserver.py`에 `--tool generation`을 추가한다.
- 엔드포인트:
  - `POST /generate`: 영상 파일, 계획(JSON), seed를 받아 WAV를 돌려준다.
  - `POST /score`: 생성 stem, 계획, 형제 계획(JSON), 참조 stem(선택)을 받아 `FilterResult`를 돌려준다.
- 인증(Bearer 토큰), 업로드 크기 제한(`save_upload`), 시간 제한은 기존 도구와 같다.
- `remote.py`는 기존 `separation/remote.py` 패턴을 따른다.

## 8. 배포

**코드와 환경**
- 새 uv 프로젝트 `gen/`을 만든다. 구성은 `gpu/`와 같다: `package = false`, 루트 패키지를 경로 의존성으로 넣고, 잠금 파일을 둔다.
- 의존성: torch 2.6 계열 CUDA 휠, SelVA(`git+https://github.com/gwpark0741/selva@8589976` 또는 원본의 같은 커밋, 빌더가 원본과 차이를 확인해 고름), LAION-CLAP. 정확한 버전은 기존 평가 환경(`foley-avbench`: torch 2.6.0+cu124, numpy 2.0.2, transformers 4.51.3, open_clip 3.3.0, av 17.0.1)을 기준으로 맞춘다.
- 새 Dockerfile `docker/generation.Dockerfile`과 이미지 `ghcr.io/capstone-audissey-fall-2026/audyssey-generation`을 만든다. `publish-images`의 이미지 목록에 추가한다.
- **이미지는 모두 공개로 한다.** 새 `audyssey-generation`도 기존 `audyssey-app`, `audyssey-tools`처럼 공개다.
  - 이유: 팀원과 RunPod가 로그인 없이 받는다. 공개 패키지는 GHCR 요금 정책과 상관없이 무료다. 저장소가 이미 공개이고, SelVA 코드도 GitHub에 공개되어 있어 새로 드러나는 것이 없다.
  - 숨겨야 하는 FT 가중치는 이미지에 넣지 않고 HF 비공개 저장소에만 둔다.
  - 새 패키지는 처음 올라갈 때 기본이 비공개일 수 있다. 첫 업로드 뒤 공개인지 확인하고, 비공개면 사용자가 GitHub 설정에서 공개로 바꾼다.
- CI: 루트 환경에서는 torch 없이 stub과 가짜 모듈로 검사한다(지금 `tests/sam_audio_fakes.py` 방식).

**Compose**
- `generation` 서비스를 추가한다. 다른 도구 서비스와 같은 형식이고, GPU는 `AUDYSSEY_GENERATION_GPU`(기본 0)로 고른다.
- main 기본값은 `AUDYSSEY_GENERATOR=remote`, `AUDYSSEY_GENERATOR_URL=http://generation:8700`이다.

**가중치**
- 베이스: HF `jnwnlee/SelVA`(공개, 승인 불필요).
  - `weights/generator_large_44khz_sup_5.pth`는 쓰지 않는다. FT 생성기로 대체하기 때문이다.
  - `weights/video_enc_sup_5.pth`, `ext_weights/v1-44.pth`, `ext_weights/synchformer_state_dict.pth`는 필요하다.
  - sv3 공용 캐시에는 이 저장소의 `refs/`만 있고 가중치 파일은 없다. 실제 모델 확인 전에 위 세 파일을 `/home/common/huggingface`에 받아 둔다(공개 저장소라 승인 불필요).
- 텍스트 인코더와 보코더: `apple/DFN5B-CLIP-ViT-H-14-384`, `google/flan-t5-base`, `nvidia/bigvgan_v2_44khz_128band_512x`. 공용 캐시에 이미 있다.
- FT 가중치: 팀 HF **비공개 조직** 저장소에 둔다.
  - 사용자가 조직과 저장소를 만들고 팀원을 초대한 뒤 두 파일을 올린다. 이 작업은 구현 범위 밖이다.
  - 코드는 저장소 ID와 파일 경로를 설정으로만 받는다:
    - `AUDYSSEY_SELVA_FT_MODEL` (저장소 ID, 기본 `sogang-audyssey/selva-ft-selsync`, 비공개)
    - `AUDYSSEY_SELVA_FT_GENERATOR_FILE` (기본 `generator/S_O_s43/model_ema_step1000.pth`)
    - `AUDYSSEY_SELVA_FT_ENCODER_FILE` (기본 `encoder/S_O/best.pt`)
  - 저장소 안의 파일 배치는 위 기본값을 따른다 (제안).
- `docs/running.md`의 가중치 준비 절에 내려받기 명령을 추가한다. 비공개 저장소는 팀원 각자의 토큰으로 받는다.

**설정 요약** (모두 `AUDYSSEY_` 접두사)

| 설정 | 기본값 | 설명 |
|---|---|---|
| `GENERATOR` / `GENERATOR_URL` | `stub` / 빈 값 | 생성 도구 선택 (Compose 기본 `remote`) |
| `SELVA_BASE_MODEL` | `jnwnlee/SelVA` | 베이스 가중치 저장소 |
| `SELVA_FT_MODEL`, `SELVA_FT_GENERATOR_FILE`, `SELVA_FT_ENCODER_FILE` | `sogang-audyssey/selva-ft-selsync`, §8 | FT 가중치 위치 |
| `SELVA_DEVICE` | `cuda` | |
| `SELVA_STEPS`, `SELVA_CFG` | `25`, `4.5` | 평가 설정과 같음 |
| `SELVA_SEED` | `42` | 첫 시도의 seed |
| `SELVA_SINGLE_MAX_S`, `SELVA_WINDOW_S` | `10`, `8` | 길이 처리 |
| `MAX_GENERATION_RETRIES` | `2` | 총 시도 = 1 + 이 값 |
| `GENERATION_MARGIN_THRESHOLD` | `0.0` | 텍스트 TargetSel 합격 기준 (보정 전) |
| `GENERATION_GPU` | `0` | Compose 전용 |

## 9. API와 화면

- HTTP API는 바뀌지 않는다. 새 명령은 기존 `POST /api/jobs/{id}/commands`로 보낸다. 작업 JSON에 `silent`, `origin`, `replaces`가 추가된다.
- 개발용 화면(`devui/index.html`)에 최소 조작만 넣는다:
  - 소스마다 출처 표시(분리/생성)
  - 분리 소스에 "생성으로 바꾸기" 버튼, 생성 소스에 "다시 생성" 버튼(설명 입력란 재사용)
  - 소스 추가 폼에 분리/생성 선택(무음이면 생성만)
  - 무음 영상이면 상단에 "무음 영상: 생성만 지원" 표시
  - 대사 소스가 `skipped`면 "대사는 생성하지 않음" 표시
- RunPod: 지금 tools 이미지는 도구 3개를 한 프로세스로 띄운다. 생성은 이미지가 따로라 RunPod에 별도 Pod 또는 프로세스가 필요하다. 이번에는 `docs/running.md`에 한계로만 적는다.

## 10. 확인 방법

- 단위 테스트 (GPU 없음, stub과 가짜 모듈):
  - 무음 판정: 오디오 스트림 없음, −60 dBFS 미만, 정상 오디오
  - 무음 영상 전체 흐름: 대사 소스가 `skipped`, 나머지가 생성되고, residual이 없고, 합치기가 된다
  - 유음 영상의 `Generate`(대체): 새 소스 생성, 원래 소스 `keep=False`, residual이 바뀌지 않음, 두 번 누르면 같은 생성 소스를 다시 생성
  - `Generate`(다시 생성): seed가 바뀜
  - 거절: 대사 대상, residual 대상, 무음에서 `separate`, 생성 소스에 `Reseparate`
  - `refresh_residual`이 생성 소스를 빼지 않음
  - 생성 루프: 불합격 뒤 재시도, 가장 높은 margin의 결과를 씀
  - 도구 서버 `generation`의 인증과 업로드 제한
  - 기존 테스트가 모두 그대로 통과
- 실제 모델 확인 (A100, 빌린 GPU에서 한 번):
  - SelVA real 단독 실행으로 8초와 10초 영상을 생성하고 시간과 최대 GPU 메모리를 기록
  - Compose 6개 컨테이너를 한 장에 올려 유음 영상(대체 생성 1회)과 무음 영상(자동 생성)을 끝까지 돌리고, 단계별 시간·최대 GPU 메모리·TargetSel 점수를 `docs/status.md`에 기록

## 11. 남은 일과 위험

- FT 가중치용 HF 조직·저장소: `sogang-audyssey/selva-ft-selsync`(비공개) 생성 완료. 두 파일 업로드는 **구현 시작 전에 먼저 한다**.
- 새 `audyssey-generation` 패키지가 공개인지 첫 업로드 뒤 확인 (비공개면 사용자가 공개로 전환).
- SelVA GPU 메모리와 10초 생성 품질은 아직 측정하지 않았다.
- `GENERATION_MARGIN_THRESHOLD`는 보정 전이다.
- 분리 쪽도 플래너 구간을 SAM Audio에 넘긴다. 플래너 타이밍이 부정확하다는 같은 문제가 분리 품질에 영향을 주는지는 따로 확인한다.
- RunPod에서 생성 도구를 띄우는 방법은 RunPod 검증 때 정한다.

## 12. Claude가 정한 결정 (제안) 목록

- 무음 판정 기준 −60 dBFS, 무음 영상도 0으로 된 오디오 파일을 만듦 (§4)
- 같은 분리 소스를 다시 대체하면 기존 생성 소스를 다시 생성함, 생성 소스 이름 형식 (§5)
- 최대 재시도 2회, 기본 seed 42, 가장 좋은 시도만 남김 (§6)
- 생성 소스의 시도 횟수를 명령마다 되돌리지 않고 seed에 씀, 시도별 점수는 단계 메시지와 LangSmith에만 남김 (§6)
- 긴 영상의 8초 창·1초 겹침·선형 크로스페이드 (§7.2)
- 형제가 없을 때 채점 처리 (§7.2)
- stub 생성음과 stub 채점 방식 (§7.3)
- HF 저장소 안의 파일 배치 (§8)
- 채점 임베딩으로 LAION-CLAP 630k-best 사용 (§7.2)
