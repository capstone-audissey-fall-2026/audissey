from pathlib import Path

from audyssey.contracts import GroundResult, VisualHint
from audyssey.settings import Settings
from audyssey.tools.remote import post


def locate(video: Path, hint: VisualHint) -> GroundResult:
    settings = Settings()
    response = post(
        settings.grounding_url,
        "/locate",
        {"video": ("video.mp4", video.read_bytes(), "video/mp4")},
        {"hint": hint.model_dump_json()},
        settings.tool_token,
        settings.remote_timeout_s,
    )
    return GroundResult.model_validate(response.json())


def mask_video(video: Path, hint: VisualHint, out: Path) -> Path:
    settings = Settings()
    response = post(
        settings.grounding_url,
        "/mask",
        {"video": ("video.mp4", video.read_bytes(), "video/mp4")},
        {"hint": hint.model_dump_json()},
        settings.tool_token,
        settings.remote_timeout_s,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(response.content)
    return out
