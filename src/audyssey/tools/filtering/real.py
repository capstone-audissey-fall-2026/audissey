from pathlib import Path

from audyssey.contracts import FilterResult, SourcePlan


def judge(mono: Path, track: Path, plan: SourcePlan) -> FilterResult:
    raise NotImplementedError(
        "윤석: see docs/superpowers/specs/2026-11-09-audyssey-skeleton-design.md §7.1"
    )
