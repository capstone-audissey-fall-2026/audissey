from pathlib import Path

from audyssey.contracts import PlanIssue, SourceKind, SourcePlan, Span, VisualHint


def plan(
    video: Path, audio: Path, duration: float, feedback: list[PlanIssue]
) -> list[SourcePlan]:
    rejected = {issue.source_name for issue in feedback}
    entries: list[tuple[str, SourceKind, str, float, float]] = [
        ("character_a", "dialogue", "voice of character a", 0, 0.3),
        ("character_b", "dialogue", "voice of character b", 0.35, 0.6),
        ("sword_clash", "sfx", "sound of sword clash", 0.65, 0.9),
        ("ghost_sound", "sfx", "sound of ghost", 0.1, 0.2),
    ]
    return [
        SourcePlan(
            name=name,
            kind=kind,
            description=description,
            spans=[Span(start=start * duration, end=end * duration)],
            visual_hint=(
                VisualHint(
                    time=1.0,
                    box=(0.0, 0.0, 0.5, 1.0),
                    description="man on the left" if name == "character_a" else "ghost",
                )
                if name in {"character_a", "ghost_sound"}
                else None
            ),
        )
        for name, kind, description, start, end in entries
        if name not in rejected
    ]


def validate_plan(video: Path, audio: Path, plans: list[SourcePlan]) -> list[PlanIssue]:
    return [
        PlanIssue(
            source_name=p.name,
            kind="not_visible_in_frame",
            message="source is not visible in frame",
        )
        for p in plans
        if p.name.startswith("ghost_")
    ]
