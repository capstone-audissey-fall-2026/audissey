from pathlib import Path

from audyssey.contracts import SourcePlan
from audyssey.settings import Settings
from audyssey.tools.remote import post


def separate(
    mono: Path, plan: SourcePlan, video: Path | None, out: Path, attempt: int
) -> Path:
    settings = Settings()
    files = {"mono": ("mono.wav", mono.read_bytes(), "audio/wav")}
    if video is not None:
        files["video"] = ("video.mp4", video.read_bytes(), "video/mp4")
    response = post(
        settings.separator_url,
        "/separate",
        files,
        {"plan": plan.model_dump_json(), "attempt": str(attempt)},
        settings.tool_token,
        settings.remote_timeout_s,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(response.content)
    return out
