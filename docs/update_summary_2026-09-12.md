# 2026-09-12 수정 사항 정리

대상 브랜치: `feat/sound-timeline-routing`

이 문서는 최근 UI, 오디오 생성, 프롬프트, Docker 실행 관련 변경 사항을 리뷰와 시연 준비 관점에서 정리한 것이다.

## UI 검사 패널

- `At Frame` 패널에 `Key Frame` 탭을 추가했다.
- 현재 프레임이 속한 scene의 주요 프레임 이미지를 확인할 수 있다.
- 각 key frame은 scene 번호, frame index, 시간 정보를 함께 보여준다.
- 기존 `Track` 탭 이름은 `Object`로 바꿨다.
- 기존 `Visual` 탭 이름은 `Event`로 바꿨다.

관련 파일:

- `web/src/components/Inspector.tsx`
- `web/src/styles.css`
- `src/v2a_inspect/ui/routes.py`

## Clean workspace

- UI에서 현재 로드된 영상, 분석 결과, 생성 기록을 지울 수 있는 `Clean workspace` 버튼을 추가했다.
- 실행 중인 pipeline이 있을 때는 clean을 막는다.
- clean 시 현재 asset 상태와 work directory의 산출물을 정리한다.

관련 파일:

- `web/src/App.tsx`
- `web/src/components/VideoEditor.tsx`
- `web/src/api.ts`
- `src/v2a_inspect/ui/routes.py`
- `src/v2a_inspect/ui/store.py`

## 오디오 생성과 편집

### 전체 생성

- `Generate Audio`는 sound timeline을 기준으로 이벤트별 WAV를 만들고, track stem과 최종 synthesized video를 다시 만든다.
- speech track은 `tts` 모델을 사용하도록 UI 선택지와 라우팅을 정리했다.
- 기존 sound timeline 편집 내용이 있으면, 생성 전에 draft asset을 넘겨 반영한다.

### 생성 후 개별 오디오 관리

- `Generate Audio` 이후 왼쪽 패널에 `Generated Audio` 목록을 추가했다.
- 목록은 생성된 event WAV를 시간순으로 보여준다.
- 각 항목에서 다음 작업을 바로 할 수 있다.
  - 브라우저 기본 오디오 컨트롤로 개별 WAV 재생
  - `Play with video`로 해당 frame부터 영상과 함께 재생
  - `Delete`로 개별 event WAV 삭제
  - `Edit / Regenerate`로 프롬프트를 수정한 뒤 해당 event만 재생성
- 개별 재생성 시 acoustic description을 직접 수정할 수 있다.
- speech event는 spoken text도 같이 수정할 수 있다.
- 재생성 또는 삭제 후에는 남은 event WAV를 기준으로 track stem과 synthesized video를 다시 만든다.

관련 파일:

- `web/src/components/VideoEditor.tsx`
- `web/src/components/Timeline.tsx`
- `web/src/styles.css`
- `web/src/App.tsx`
- `web/src/api.ts`
- `src/v2a_inspect/ui/routes.py`
- `src/v2a_inspect/ui/pipeline.py`
- `tests/test_ui_audio_events.py`

## 객체 검출 프롬프트 조정

- Gemini 초기 scene 분석 프롬프트를 시연 영상에 맞게 보강했다.
- anime samurai duel 장면에서 다음 label을 우선 사용하도록 했다.
  - `black samurai`
  - `red samurai`
  - `adult man`
  - `child`
  - `sword`
  - `katana sword`
- SAM3 추적에는 label보다 단순한 `tracking_prompt`를 쓰도록 유도했다.
  - 예: label은 `black samurai`, tracking prompt는 `samurai`
- `person1`, `person2`, `object1` 같은 관리용 이름은 쓰지 않도록 했다.
- 비어 있지 않은 scene에서 object seed가 빈 배열로 나오는 경우를 줄이도록, actor나 sword가 보이면 최소 seed를 만들라는 지침을 추가했다.

관련 파일:

- `src/v2a_inspect/prompts/files/system/initial_scene_analysis.txt`

## Docker 실행 반영

- 프론트엔드는 `Dockerfile`의 `frontend` stage에서 `npm run build`로 빌드되고, 결과물이 `/app/web/dist`에 들어간다.
- 따라서 UI 변경을 실행 중인 컨테이너에 반영하려면 이미지를 다시 빌드해야 한다.

```bash
docker compose up --build -d v2a-inspect
```

- inference 서버 이미지까지 바뀐 경우에는 전체 compose를 다시 올린다.

```bash
docker compose up --build -d
```

## 검증 기록

- `npm run build`: 통과
  - 현재 로컬 Node가 `18.19.1`이라 Vite의 Node 권장 버전 경고가 나온다.
  - Docker 빌드는 `node:22-bookworm-slim`을 사용하므로 이 경고의 영향을 받지 않는다.
- `uv run python -m unittest tests.test_ui_audio_events`: 통과
- `npm test`: 로컬 Node 18에서 `crypto.hash is not a function`으로 실패
  - Vite 7 계열은 Node `20.19+` 또는 `22.12+`를 요구한다.
  - 웹 테스트는 Node를 올린 뒤 다시 확인하는 것이 맞다.
- Impeccable detector:
  - 기존 전역 폰트 `Inter` 경고만 표시됨
  - 이번 generated audio UI 추가에서 별도 문제는 감지되지 않음

## 사용 흐름

1. 영상 분석을 실행하거나 기존 `VideoAsset JSON`을 import한다.
2. `At Frame`의 `Key Frame`, `Scene`, `Object`, `Event`, `Sound` 탭에서 분석 결과를 확인한다.
3. 필요하면 timeline의 sound lane에서 track, event, prompt, timestamp를 수정한다.
4. `Generate Audio`를 실행한다.
5. 생성 완료 후 `Generated Audio` 목록에서 event별 WAV를 재생, 삭제, 재생성한다.
6. 개별 재생성 시 prompt를 고친 뒤 `Regenerate Audio`를 누른다.
7. 최종 결과는 `Download Video(MP4)` 또는 `Save Video (MP4)`로 저장한다.
