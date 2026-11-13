import json
from pathlib import Path

from audyssey.contracts import FilterResult, SourcePlan
from audyssey.settings import Settings
from audyssey.tools.remote import post


def generate(video: Path, plan: SourcePlan, out: Path, seed: int) -> Path:
    settings = Settings()
    response = post(
        settings.generator_url,
        "/generate",
        {"video": ("video.mp4", video.read_bytes(), "video/mp4")},
        {"plan": plan.model_dump_json(), "seed": str(seed)},
        settings.tool_token,
        settings.remote_timeout_s,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(response.content)
    return out


def score_generation(
    track: Path,
    plan: SourcePlan,
    siblings: list[SourcePlan],
) -> FilterResult:
    settings = Settings()
    response = post(
        settings.generator_url,
        "/score",
        {"track": ("track.wav", track.read_bytes(), "audio/wav")},
        {
            "plan": plan.model_dump_json(),
            "siblings": json.dumps([p.model_dump() for p in siblings]),
        },
        settings.tool_token,
        settings.remote_timeout_s,
    )
    return FilterResult.model_validate(response.json())
