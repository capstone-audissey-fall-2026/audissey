from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from functools import wraps
from pathlib import Path

from langsmith import traceable

from audyssey.contracts import FilterResult, PlanIssue, SourcePlan
from audyssey.settings import Settings


@dataclass(frozen=True)
class Tools:
    plan: Callable[[Path, Path, float, list[PlanIssue]], list[SourcePlan]]
    validate_plan: Callable[[Path, Path, list[SourcePlan]], list[PlanIssue]]
    separate: Callable[[Path, SourcePlan, Path, Path, int], Path]
    judge: Callable[[Path, Path, SourcePlan], FilterResult]


def load_tools(settings: Settings) -> Tools:
    for mode, url, name in [
        (settings.separator, settings.separator_url, "SEPARATOR"),
        (settings.filter, settings.filter_url, "FILTER"),
    ]:
        if mode == "remote" and not url.strip():
            raise ValueError(f"AUDYSSEY_{name}_URL is required for remote mode")
    planning = import_module(f"audyssey.tools.planning.{settings.planner}")
    separation = import_module(f"audyssey.tools.separation.{settings.separator}")
    filtering = import_module(f"audyssey.tools.filtering.{settings.filter}")
    return Tools(
        wraps(planning.plan)(traceable(run_type="tool", name="plan")(planning.plan)),
        wraps(planning.validate_plan)(
            traceable(run_type="tool", name="validate_plan")(planning.validate_plan)
        ),
        wraps(separation.separate)(
            traceable(run_type="tool", name="separate")(separation.separate)
        ),
        wraps(filtering.judge)(
            traceable(run_type="tool", name="judge")(filtering.judge)
        ),
    )
