# Audyssey skeleton

GPU와 API 키 없이 영상 업로드 → 플래닝·필터링 검증 루프 → 소스 트랙과 나머지 → 편집 → 영상과 stems 내보내기를 실행합니다. 기본 도구는 모두 stub이며 분리 품질을 평가하는 구현은 아닙니다.

요구사항: Python 3.11+, [uv](https://docs.astral.sh/uv/), `ffmpeg`/`ffprobe` (PATH에 설치). 개발용 화면에는 Node가 필요 없습니다.

```bash
uv sync --extra ui
uv run audyssey serve
# http://127.0.0.1:8600 에서 samples/sample.mp4 업로드
```

개발용 화면에서 트랙을 재생하고 keep/drop, 음량, 피치를 조절합니다. 설명을 바꾸어 해당 소스만 다시 분리하거나 소스를 추가할 수 있습니다. 합치기는 편집된 MP4, mix WAV, keep된 소스의 stems를 만듭니다. 나머지 stem은 선택 사항입니다. 편집 없이 합친 WAV는 원본 추출 오디오와 수치 오차 범위에서 같습니다. AAC 영상 오디오는 손실 압축됩니다.

```bash
uv sync --dev --extra ui
uv run python -m unittest discover tests
uv run ruff check src/
uv run ruff format --check src/
uv run ty check --error-on-warning src/
```

| 환경변수 | 기본값 | 의미 |
|---|---|---|
| `AUDYSSEY_PLANNER` | `stub` | `stub` 또는 `real` |
| `AUDYSSEY_SEPARATOR` | `stub` | `stub` 또는 `real` |
| `AUDYSSEY_FILTER` | `stub` | `stub` 또는 `real` |
| `AUDYSSEY_MAX_PLAN_ROUNDS` | `2` | 플래닝 최대 회차 (1 이상) |
| `AUDYSSEY_MAX_SEPARATION_RETRIES` | `2` | 재분리 횟수, 총 시도 최대 3회 |
| `AUDYSSEY_MAX_DURATION_S` | `30` | 업로드 영상 길이 상한 |
| `AUDYSSEY_WORK_DIR` | `work/jobs` | 작업 폴더 위치 |
| `AUDYSSEY_SAM_AUDIO_MODEL` | `facebook/sam-audio-large-tv` | 향후 분리 모델 ID |
| `AUDYSSEY_SAM_AUDIO_JUDGE_MODEL` | `facebook/sam-audio-judge` | 향후 필터링 모델 ID |

설정은 환경변수로만 읽으며 `.env`는 읽지 않습니다. 작업은 `<WORK_DIR>/<id>/job.json`과 상대 경로 파일로 보존됩니다. 서버 재시작 후 `GET /api/jobs`와 `GET /api/jobs/{id}`로 다시 조회할 수 있습니다. 단일 서버 프로세스로 실행합니다. 실행 중 재분리·추가·합치기는 409를 반환하며 keep/drop과 음량·피치 변경은 허용합니다.

팀원은 담당 `src/audyssey/tools/<모듈>/real.py`의 동일 시그니처 함수를 구현하고 `AUDYSSEY_PLANNER=real`, `AUDYSSEY_SEPARATOR=real`, `AUDYSSEY_FILTER=real` 중 해당 값을 설정합니다. 현재 real 함수는 담당자가 표시된 `NotImplementedError` placeholder입니다.

- 도은: `planning/real.py`의 `plan`, `validate_plan`
- SAM Audio 담당: `separation/real.py`의 `separate`
- 윤석: `filtering/real.py`의 `judge`
- UI 담당: 최종 UI를 결정합니다. 이전 `web/` React UI는 삭제되었고 git 기록과 `legacy` 원격에 남아 있습니다.

공통 계약 `contracts.py` 변경은 PR로 합의합니다. 이전 `v2a_inspect` 코드는 삭제되었고 git 기록과 `legacy` 원격에 남아 있습니다. 모델은 필요할 때 로드하고 가중치는 `HF_HOME` 캐시를 사용합니다. 뼈대 실행에는 모델 다운로드가 없습니다. 도구 교체 시 API와 오케스트레이터를 수정하지 않습니다.

[설계](docs/superpowers/specs/2026-11-09-audyssey-skeleton-design.md) · [구현 계획](docs/superpowers/plans/2026-11-09-audyssey-skeleton.md)
