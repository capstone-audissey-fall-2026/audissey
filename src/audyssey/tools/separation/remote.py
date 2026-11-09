from pathlib import Path

from audyssey.contracts import SourcePlan
from audyssey.settings import Settings
from audyssey.tools.remote import post


def separate(
    mono: Path, plan: SourcePlan, video: Path, out: Path, attempt: int
) -> Path:
    settings = Settings()
    response = post(
        settings.separator_url,
        "/separate",
        {
            "mono": ("mono.wav", mono.read_bytes(), "audio/wav"),
            "video": ("video.mp4", video.read_bytes(), "video/mp4"),
        },
        {"plan": plan.model_dump_json(), "attempt": str(attempt)},
        settings.tool_token,
        settings.remote_timeout_s,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(response.content)
    return out
