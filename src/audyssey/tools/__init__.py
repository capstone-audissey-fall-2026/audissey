from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from functools import wraps
from pathlib import Path

from langsmith import traceable

from audyssey.contracts import (
    FilterResult,
    GroundResult,
    PlanIssue,
    SourcePlan,
    VisualHint,
)
from audyssey.settings import Settings


@dataclass(frozen=True)
class Tools:
    plan: Callable[[Path, Path, float, list[PlanIssue]], list[SourcePlan]]
    validate_plan: Callable[[Path, Path, list[SourcePlan]], list[PlanIssue]]
    # video: masked video for visual prompting, or None for text-only separation
    separate: Callable[[Path, SourcePlan, Path | None, Path, int], Path]
    judge: Callable[[Path, Path, SourcePlan], FilterResult]
    locate: Callable[[Path, VisualHint], GroundResult]
    mask_video: Callable[[Path, VisualHint, Path], Path]
    generate: Callable[[Path, SourcePlan, Path, int], Path]
    score_generation: Callable[
        [Path, SourcePlan, list[SourcePlan], Path | None, list[Path]], FilterResult
    ]


def load_tools(settings: Settings) -> Tools:
    for mode, url, name in [
        (settings.separator, settings.separator_url, "SEPARATOR"),
        (settings.filter, settings.filter_url, "FILTER"),
        (settings.grounder, settings.grounding_url, "GROUNDING"),
        (settings.generator, settings.generator_url, "GENERATOR"),
    ]:
        if mode == "remote" and not url.strip():
            raise ValueError(f"AUDYSSEY_{name}_URL is required for remote mode")
    planning = import_module(f"audyssey.tools.planning.{settings.planner}")
    separation = import_module(f"audyssey.tools.separation.{settings.separator}")
    filtering = import_module(f"audyssey.tools.filtering.{settings.filter}")
    grounding = import_module(f"audyssey.tools.grounding.{settings.grounder}")
    generation = import_module(f"audyssey.tools.generation.{settings.generator}")
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
        wraps(grounding.locate)(
            traceable(run_type="tool", name="locate")(grounding.locate)
        ),
        wraps(grounding.mask_video)(
            traceable(run_type="tool", name="mask_video")(grounding.mask_video)
        ),
        wraps(generation.generate)(
            traceable(run_type="tool", name="generate")(generation.generate)
        ),
        wraps(generation.score_generation)(
            traceable(run_type="tool", name="score_generation")(
                generation.score_generation
            )
        ),
    )
