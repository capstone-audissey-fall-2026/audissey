from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

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
        planning.plan, planning.validate_plan, separation.separate, filtering.judge
    )
