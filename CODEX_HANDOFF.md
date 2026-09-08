# Codex 구현 인계 문서

최종 갱신: 2026-09-07 (Asia/Seoul)
프로젝트: /home/gwpark0741/v2a_inspect

## 1. 현재 상태

- 브랜치: feat/sound-timeline-routing
- 기준 커밋: e9c024a (feat/audio-artifacts)
- upstream: 설정되지 않음
- origin: https://github.com/sogang-v2a/v2a_inspect
- fork: https://github.com/gwpark0741/v2a_inspect
- push: 수행하지 않음
- 로컬 영상, 생성 결과, 가중치, 가상환경은 Git ignored 상태
- 사용자 소유 로컬 파일은 삭제하지 않음

사용자 fork로 올릴 때:

    git push -u fork feat/sound-timeline-routing

## 2. 완료된 기능

### 저장소와 실행 구조

- .env.example을 추가하고 실제 .env는 Git에서 제외했다.
- root, inference client, server 설정이 .env를 읽는다.
- 실행 명령을 다음으로 정리했다.
  - uv run v2a run VIDEO
  - uv run v2a ui
  - uv run v2a synthesize
  - uv run --project server v2a-inspect-server serve
- scripts/run.sh의 존재하지 않던 v2a-inspect 호출을 v2a run으로 수정했다.
- README와 server/README의 명령 및 endpoint를 현재 구현과 맞췄다.
- docker-compose.yaml은 UI와 inference 서버를 함께 실행한다.

### Speech와 TTS

- SoundTrack.track_type에 speech를 추가했다.
- SoundEvent.description은 음향/연출 설명으로 유지한다.
- SoundEvent.spoken_text는 speech에서만 필수이며 non-speech에서는 금지한다.
- VLM sound-timeline agent가 spoken_text 초안을 작성한다.
- speech 트랙은 항상 tts로 라우팅한다.
- inference server에 Kokoro 0.9.4 lazy TTS와 endpoint를 추가했다.
  - POST /infer/kokoro/generate-speech
- client에 KokoroClient를 추가하고 오디오 router까지 연결했다.
- 기본 voice는 af_heart, 기본 언어는 영어, 기본 device는 CPU다.

### 트랙 단위 라우팅

저장 가능한 값은 t2a, v2a, tts뿐이다.

- speech: tts 강제
- sfx: t2a 기본
- ambience: t2a 기본
- 화면 접촉·충돌처럼 정확한 영상 동기화가 필요한 효과만 v2a
- 모델 선택 단위는 이벤트가 아니라 트랙
- 모델이 달라야 하는 같은 음향 정체성은 트랙을 분리

구 데이터는 로딩 경계에서만 호환한다.

- dialogue -> speech
- tta -> t2a
- vta -> v2a
- hybrid/unknown -> t2a
- 구 dialogue의 따옴표 대사를 spoken_text로 추출
- description은 변경하지 않음

V2A 요청에 video_id/time이 없을 때 T2A로 조용히 떨어지지 않고 실패한다.
CLI와 UI는 공통 build_audio_plan 함수를 사용하며 V2A 이벤트가 있을 때만
영상을 inference server에 업로드한다.

### 통합 UI

현재 흐름:

1. 영상 업로드 및 agent 추론
2. sound timeline 편집
3. 모델별 오디오 생성
4. 이벤트 WAV, 트랙 stem, 합성 영상 재생 및 다운로드

지원하는 편집:

- 트랙 생성/삭제
- 이벤트 생성/삭제
- 이벤트 drag/resize
- 음향 description 변경
- speech spoken_text 변경
- non-speech 트랙의 T2A/V2A 변경
- speech 트랙 TTS 고정

모델 선택기는 데이터 소유 단위에 맞게 트랙 헤더에만 둔다. 이벤트 dialog는
description과 speech 전용 spoken_text만 편집한다. 서버 상태는 합의대로
메모리에서만 유지하며 재시작 복구는 구현하지 않았다.

### 모델 캐시와 배포

- Hunyuan 가중치와 YAML을 V2A_SERVER_HUNYUAN_MODEL_PATH 한 곳에 받는다.
- 공식 모델 저장소 파일명인 config_xl.yaml/config.yaml을 사용한다.
- Docker 기본 경로는 /data/model-cache/hunyuan이다.
- Kokoro/Hugging Face 캐시는 /data/model-cache/huggingface에 유지한다.
- Compose 서비스 간 inference URL은 http://inference:8080이다.
- UI 컨테이너의 불필요한 GPU 요구는 제거했다.
- 기존 루트/server 하위의 중복 가중치는 삭제하지 않았다.

공식 Hunyuan 파일 목록:
https://huggingface.co/tencent/HunyuanVideo-Foley/tree/main

## 3. 커밋 목록

- e39d070 chore(repo): ignore local model artifacts
- 9a83bb0 feat(tts): add Kokoro speech synthesis
- ea7be66 docs: add implementation handoff
- 43ef130 fix(audio): route generation by explicit model
- 4418f5d feat(timeline): add speech events and explicit routing
- 891dc3e feat(web): edit speech and track routing
- 87fadd8 refactor(audio): share timeline audio planning
- 568c422 refactor(config): clarify environment and entrypoints
- e2d435b refactor(server): centralize Hunyuan model cache
- a3bc675 feat(deploy): run UI with inference server
- de55a07 fix(ui): preserve speech generation context

## 4. 최종 검증 결과

통과:

- uv run ruff check src tests server/src server/tests
- uv run python -m unittest discover -s tests: 8개
- uv run --project server --no-sync python -m unittest discover -s server/tests: 3개
- python -m compileall -q src server/src
- npm run build
- docker compose config --quiet
- docker compose --env-file .env.example config --quiet
- git diff --check
- root/server CLI help smoke test
- .env.example을 root/client/server Settings로 읽는 smoke test

웹 빌드는 성공했지만 로컬 Node.js 18.19.1에 대해 Vite가 Node.js 20.19+
또는 22.12+를 권장한다는 경고가 남는다.

UI 정적 품질 검사에서는 기존 Inter 폰트 사용 경고 1건만 있었고 기능 및
접근성 오류는 없었다.

## 5. 실행하지 않은 검증

다음은 시간·용량·GPU/API 자격 증명이 필요한 실제 외부 실행이므로 수행하지 않았다.

- Docker 이미지 전체 빌드
- 새 Compose volume로 Hunyuan 약 18.6GB 다운로드
- Kokoro 첫 모델 다운로드
- 실제 Gemini agent 호출
- 실제 ElevenLabs T2A 호출
- GPU Hunyuan V2A 생성
- 브라우저 E2E 청취

## 6. 검토 시 주의사항

- T2A는 현재 ElevenLabs를 사용한다. ELEVENLABS_API_KEY가 없거나 호출이
  실패하면 기존 fallback인 짧은 dummy beep가 생성된다.
- Kokoro는 공식 한국어 모델이 아니며 현재 영어 기본값이다.
- 모델 voice 선택 UI는 요구 범위에서 제외했다.
- 서버 재시작 후 timeline 상태 복구는 합의대로 제외했다.
- 대용량 가중치와 기존 생성물은 Git에 포함되지 않으며 자동 삭제되지 않는다.
- 실제 운영 전 Node.js를 20.19+ 또는 22.12+로 올리고 위 외부 통합 테스트를
  수행하는 것이 좋다.

## 7. 내일 확인 순서

    git status --short --branch
    git log --oneline --decorate e9c024a..HEAD
    cp .env.example .env
    docker compose config
    docker compose up --build

로컬 개발은 inference server와 UI를 별도 터미널에서 실행할 수 있다.

    uv run --project server v2a-inspect-server serve
    uv run v2a ui
