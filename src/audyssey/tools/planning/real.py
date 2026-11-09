from pathlib import Path

from audyssey.contracts import PlanIssue, SourcePlan


def plan(
    video: Path, audio: Path, duration: float, feedback: list[PlanIssue]
) -> list[SourcePlan]:
    raise NotImplementedError(
        "도은: see docs/superpowers/specs/2026-11-09-audyssey-skeleton-design.md §7.1"
    )


def validate_plan(video: Path, audio: Path, plans: list[SourcePlan]) -> list[PlanIssue]:
    raise NotImplementedError(
        "도은: see docs/superpowers/specs/2026-11-09-audyssey-skeleton-design.md §7.1"
    )
