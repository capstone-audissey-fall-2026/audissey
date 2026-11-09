from pathlib import Path

from audyssey.contracts import SourcePlan


def separate(
    mono: Path, plan: SourcePlan, video: Path, out: Path, attempt: int
) -> Path:
    raise NotImplementedError(
        "SAM Audio: see docs/superpowers/specs/2026-11-09-audyssey-skeleton-design.md §7.1"
    )
