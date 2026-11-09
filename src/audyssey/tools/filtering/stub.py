from pathlib import Path

from audyssey.contracts import FilterResult, SourcePlan


def judge(mono: Path, track: Path, plan: SourcePlan) -> FilterResult:
    if plan.name == "character_a" and not plan.description.endswith(" (refined)"):
        return FilterResult(
            passed=False,
            reason="leakage from other voices",
            suggestion=plan.model_copy(
                update={"description": plan.description + " (refined)"}
            ),
        )
    return FilterResult(
        passed=True, scores={"overall": 0.9, "recall": 0.9, "precision": 0.9}
    )
