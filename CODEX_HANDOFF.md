# Codex 구현 인계 문서

작성일: 2026-09-07 (Asia/Seoul)
프로젝트: /home/gwpark0741/v2a_inspect
중단 사유: 사용자 요청에 따라 현재 완료 범위에서 일시 중단

## 1. Git 상태

- 현재 브랜치: feat/sound-timeline-routing
- 분기 기준: e9c024a (feat/audio-artifacts)
- upstream: 아직 설정되지 않음
- origin: https://github.com/sogang-v2a/v2a_inspect
- fork: https://github.com/gwpark0741/v2a_inspect
- push는 수행하지 않음
- 이 문서를 만들기 직전 working tree는 clean 상태였음

재개 후 우선 확인:

    git status --short --branch
    git log --oneline --decorate -8
    git remote -v

최종 확인 후 사용자 저장소에 올릴 명령:

    git push -u fork feat/sound-timeline-routing

## 2. 이번 작업에서 완료한 커밋

### e39d070 chore(repo): ignore local model artifacts

.gitignore에 다음 로컬 산출물을 추가했다.

- .v2a_inspect/
- *.pth
- HunyuanVideo-Foley 로컬 체크아웃/모델 디렉터리
- 중복 Hunyuan YAML 경로
- assets/videos 아래 압축 파일

확인 결과 Hunyuan YAML 네 복사본은 동일한 SHA256이었다. 모델 파일은 약 7.8GB 세트가 두 벌 있어 약 15.6GB가 중복되어 있었다. 파일은 삭제하지 않았고 Git 추적 대상에서만 제외했다.

### 9a83bb0 feat(tts): add Kokoro speech synthesis

로컬 실행 가능한 Kokoro TTS를 inference server와 root client에 추가했다.

주요 변경:

- 서버 의존성: kokoro==0.9.4, soundfile
- Docker 런타임 패키지: espeak-ng
- 설정:
  - kokoro_model_id=hexgrad/Kokoro-82M
  - kokoro_language=a
  - kokoro_voice=af_heart
  - kokoro_device=cpu
- 서버 요청 모델: KokoroGenerateSpeechRequest
- lazy-loading 및 직렬화된 KokoroInferenceClient
- 엔드포인트: POST /infer/kokoro/generate-speech
- root client: KokoroClient.generate_speech
- 서버 단위 테스트 추가

Kokoro 가중치는 아직 다운로드하지 않았다. 첫 실제 TTS 요청 시 Hugging Face에서 lazy download된다. 기본은 Hunyuan GPU 메모리와 충돌하지 않도록 CPU 실행이다. 기본 언어/voice는 영어이며 Kokoro 공식 지원 목록에 한국어는 없다. 한국어는 선택 조건이었으므로 현재 범위에는 포함하지 않았다.

검증 완료:

- uv run ruff check: 통과
- python -m compileall: 통과
- server unittest 2개: 통과
- git diff --check: 통과

## 3. 확정된 제품 요구사항

### SoundTimeline 모델

- 트랙 타입은 speech, sfx, music, ambience를 사용한다.
- 기존 dialogue는 speech로 마이그레이션한다.
- SoundEvent.description은 기존처럼 음향/연출 설명을 유지한다.
- SoundEvent.spoken_text는 speech 이벤트에만 존재하고 필수로 채운다.
- non-speech 이벤트에는 spoken_text를 허용하지 않는다.
- 대사 텍스트 초안은 VLM agent가 생성한다.
- 사용자는 SoundTimeline 편집 단계에서 spoken_text를 수정할 수 있다.

### 생성 모델 라우팅

라우팅은 이벤트별이 아니라 트랙별이다.

저장 가능한 generation model 값:

- t2a
- v2a
- tts

auto, unknown, hybrid, tta, vta는 새 데이터에 저장하지 않는다.

라우팅 기준:

- speech 트랙은 항상 tts
- ambience와 music은 기본 t2a
- sfx는 기본 t2a
- 화면의 정확한 접촉/충돌 시점 등 영상 동기화가 꼭 필요한 SFX만 v2a
- non-speech 트랙은 UI에서 t2a/v2a를 사람이 변경 가능
- 같은 음향 정체성이라도 생성 모델이 다르면 트랙을 분리
- speech에는 초기 버전에서 voice 선택 UI를 만들지 않고 서버 기본 voice를 사용

구 JSON 로드 호환 목표:

- dialogue -> speech
- tta -> t2a
- vta -> v2a
- speech -> tts 강제
- non-speech hybrid/unknown -> t2a
- 구 dialogue description에 따옴표 대사가 있으면 spoken_text로 추출
- 추출할 수 없으면 구 데이터에 한해 description을 spoken_text fallback으로 사용
- description 자체는 변경하지 않음

### UI 흐름

목표 흐름:

1. 영상 업로드
2. agent 추론
3. 사람의 SoundTimeline 편집
4. 라우팅된 실제 사운드 생성
5. 이벤트/트랙/최종 영상 청취

기존 UI에 이미 있는 기능:

- 영상 업로드 및 pipeline 실행
- 트랙/이벤트 추가와 삭제
- 이벤트 drag/resize
- 이벤트 설명 편집
- 오디오 생성 요청
- 이벤트 WAV, 트랙 stem, 최종 합성 영상 재생

추가/수정할 기능:

- speech 트랙 및 spoken_text 편집
- 트랙 lane/header에 generation model 선택기 배치
- 이벤트 편집 modal에서는 description과 speech 전용 spoken_text만 편집
- speech 트랙은 tts로 고정
- 새 non-speech 트랙 기본값은 t2a
- auto 옵션은 만들지 않음

상태 저장은 현재 in-memory store를 유지한다. 서버 재시작 후 복구는 요구하지 않는다.

## 4. 조사한 현재 문제와 원인

### 오디오 라우터

파일: src/v2a_inspect/audio_generation/client.py

현재 문제:

- dialogue 여부와 description 안의 따옴표 유무로 OpenAI TTS/T2A를 혼합 결정한다.
- generation model이 v2a여도 video_id/time이 없으면 T2A 분기로 조용히 떨어질 수 있다.
- OpenAI TTS가 남아 있고 새 KokoroClient는 아직 실제 synthesis router에 연결되지 않았다.
- server_url이 UI upload client에는 전달되지만 Hunyuan 생성 client에는 전달되지 않는다.

권장 수정:

- generation_model을 최우선으로 명시 라우팅
- tts -> KokoroClient와 spoken_text
- v2a -> HunyuanClient; video_id/time 없으면 명시 실패
- t2a -> ElevenLabs SFX/music
- 구 dialogue/tta/vta는 입력 호환만 유지
- router에 server_url 전달
- 라우팅 분기 unittest 한 개 파일 유지

### 중복 AudioPlan 변환

파일:

- src/v2a_inspect/audio_generation/synthesize.py
- src/v2a_inspect/ui/pipeline.py

두 파일이 SoundTimeline을 AudioPlan으로 거의 동일하게 변환한다. 둘 다 모든 생성 전에 영상을 inference server에 업로드한다.

권장 수정:

- 작은 공통 build_audio_plan 함수를 audio_generation 아래에 둔다.
- spoken_text를 AudioPlanItem에 전달한다.
- v2a item이 하나라도 있을 때만 영상 업로드한다.
- UI의 item_context는 event/track id map으로 별도 유지한다.

### 타임라인 스키마

현재 값:

- track_type: dialogue, sfx, music, ambience
- generation_mode: tta, vta, hybrid, unknown
- SoundEvent에 spoken_text 없음

관련 Python 호출 지점:

- models/sound_timeline.py
- tools/sound_timeline/schemas.py
- tools/sound_timeline/editor.py
- tools/sound_timeline/write_tools.py
- tools/sound_timeline/langchain.py
- tools/sound_timeline/read_tools.py
- ui/rows.py
- ui/pipeline.py
- audio_generation/synthesize.py
- models/audio_artifacts.py
- visualization 관련 파일
- sound timeline agent prompt

Pydantic model_validator(mode=before)로 구 JSON을 변환하고, mode=after에서 다음을 검증하는 방향이 적합하다.

- 참조 무결성
- speech track의 spoken_text 필수
- non-speech track의 spoken_text 금지
- speech track의 generation_model은 tts
- non-speech track의 generation_model은 t2a 또는 v2a

### Agent routing prompt

파일: src/v2a_inspect/prompts/files/system/sound_timeline_agent.txt

현재 prompt는 불확실하면 generation_mode=unknown을 쓰게 하며, V2A를 제한하는 명확한 기준이 없다. 다음 기준으로 바꿔야 한다.

- speech에는 spoken_text를 반드시 생성
- SFX 기본 t2a
- 화면 contact timing이 생성 품질에 본질적으로 필요할 때만 v2a
- ambience/music은 t2a
- 불확실할 때 unknown 대신 t2a
- 서로 다른 model이 필요하면 트랙 분리

### Web UI

주요 파일:

- web/src/types.ts
- web/src/components/Timeline.tsx
- web/src/components/VideoEditor.tsx
- web/src/components/Inspector.tsx
- web/src/styles.css

현재 이벤트 상세 modal의 generation_mode 변경이 실제로는 전체 트랙의 값을 바꾼다. 즉 UI 위치와 데이터 소유 단위가 불일치한다. 모델 선택기를 트랙 header/lane으로 옮겨야 한다. 새 트랙 기본값도 현재 vta이므로 t2a로 변경해야 한다.

### 환경변수와 entrypoint

- root script는 v2a = v2a_inspect.cli:main
- server script는 v2a-inspect-server = v2a_inspect_server.runtime:main
- scripts/run.sh는 존재하지 않는 v2a-inspect run을 호출
- .env.example 없음
- audio_generation/synthesize.py가 모듈 import 시 load_dotenv(override=True)를 늦게 호출
- 일부 UI 설정은 os.getenv를 직접 사용
- docker-compose에는 UI 서비스만 있고 inference server가 없음
- 서버 README에는 실제 구현과 맞지 않는 command/endpoint 설명이 있음

## 5. 권장 다음 커밋 순서

1. fix(audio): route generation by explicit track model
   - OpenAI TTS 제거 및 Kokoro 연결
   - V2A의 T2A fallthrough 차단
   - AudioPlanItem.spoken_text
   - server_url 전달
   - 최소 라우팅 테스트

2. feat(timeline): add speech events and explicit routing
   - speech/spoken_text/generation_model 스키마
   - 구 JSON 마이그레이션
   - tools/editor/read output 변경
   - agent prompt 라우팅 규칙
   - Pydantic 검증 테스트

3. feat(web): edit speech and track routing
   - TS 타입 전환
   - spoken_text 편집
   - 트랙별 t2a/v2a selector
   - speech tts 고정
   - npm build 검증

4. refactor(config): clarify environment and entrypoints
   - .env.example
   - 시작 시점 환경 로드
   - scripts/run.sh 수정
   - 실제 CLI entrypoint 문서화

5. fix(server): use one Hunyuan model cache
   - 중복 로컬 모델 경로를 하나의 cache 설정으로 정리
   - 로컬 대용량 파일 자체는 사용자의 확인 없이 삭제하지 않음

6. chore(docker): connect UI and inference services
   - compose에 inference service 추가
   - UI client host 연결
   - 모델 cache volume

7. docs(setup): document integrated workflow
   - 실제 실행 명령
   - 첫 Kokoro model download
   - 편집/생성/청취 흐름
   - server endpoint 정합화

각 단계 후 권장 검증:

- uv run ruff check
- python -m compileall -q src server/src
- 관련 unittest
- web 변경 시 npm run build
- git diff --check
- git status --short

## 6. 중단 직전 시도

오디오 router 패치를 준비했지만 apply_patch 실행이 bwrap loopback 권한 오류로 실패했다. 패치는 적용되지 않았고 working tree에는 남지 않았다.

오류 요약:

    bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted

이 환경에서 apply_patch가 계속 실패하면 정확한 unified diff를 git apply로 적용하는 방식이 필요하다. 재개 시 먼저 git status로 clean 여부를 확인한다.

## 7. 범위 제한과 주의사항

- push하지 말 것: 사용자가 검토 후 fork remote로 올릴 예정
- 로컬 모델/압축 파일을 삭제하지 말 것
- 기존 사용자 변경이 생기면 덮어쓰지 말 것
- 기능 단위 Conventional Commit을 유지할 것
- routing은 event 단위가 아니라 track 단위임
- 저장 데이터에 auto를 추가하지 말 것
- 서버 재시작 복구용 DB/지속성 계층은 만들지 말 것
- Kokoro voice selector, 한국어 별도 모델, job queue는 현재 범위 밖
