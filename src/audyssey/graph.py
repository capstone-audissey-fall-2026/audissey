from dataclasses import dataclass

from audyssey.contracts import Job, PlanIssue, Source, SourcePlan, StepStatus
from audyssey.settings import Settings
from audyssey.store import JobStore
from audyssey.tools import Tools


@dataclass
class Context:
    tools: Tools
    settings: Settings
    store: JobStore


def check_plans(
    plans: list[SourcePlan], duration: float
) -> tuple[list[SourcePlan], list[PlanIssue]]:
    merged: dict[str, SourcePlan] = {}
    for plan in plans:
        if plan.name in merged:
            merged[plan.name].spans.extend(plan.spans)
        else:
            merged[plan.name] = plan.model_copy(deep=True)
    issues = []
    for plan in merged.values():
        if not plan.description.strip():
            issues.append(
                PlanIssue(
                    source_name=plan.name,
                    kind="empty_description",
                    message="description must not be empty",
                )
            )
        if any(span.end > duration for span in plan.spans):
            issues.append(
                PlanIssue(
                    source_name=plan.name,
                    kind="span_out_of_range",
                    message="span exceeds video duration",
                )
            )
    return list(merged.values()), issues


def update_step(job: Job, ctx: Context, step: StepStatus) -> None:
    job.steps = [s for s in job.steps if s.key != step.key] + [step]
    ctx.store.save(job)


def process_source(job: Job, source: Source, ctx: Context) -> None:
    directory = ctx.store.dir(job.id)
    maximum = ctx.settings.max_separation_retries + 1
    source.status = "running"
    try:
        while True:
            source.attempts += 1
            update_step(
                job,
                ctx,
                StepStatus(
                    key=f"separate:{source.id}",
                    status="running",
                    round=source.attempts,
                    max_rounds=maximum,
                ),
            )
            path = ctx.tools.separate(
                directory / job.mono_path,
                source.plan,
                directory / job.video_path,
                directory / f"tracks/{source.id}.wav",
                source.attempts,
            )
            source.track_path = str(path.relative_to(directory))
            update_step(
                job,
                ctx,
                StepStatus(
                    key=f"separate:{source.id}",
                    status="done",
                    round=source.attempts,
                    max_rounds=maximum,
                ),
            )
            update_step(
                job,
                ctx,
                StepStatus(
                    key=f"judge:{source.id}",
                    status="running",
                    round=source.attempts,
                    max_rounds=maximum,
                ),
            )
            source.filter = ctx.tools.judge(
                directory / job.mono_path, path, source.plan
            )
            update_step(
                job,
                ctx,
                StepStatus(
                    key=f"judge:{source.id}",
                    status="done" if source.filter.passed else "warning",
                    round=source.attempts,
                    max_rounds=maximum,
                    message=source.filter.reason,
                ),
            )
            if source.filter.passed or source.attempts >= maximum:
                break
            if source.filter.suggestion is not None:
                source.plan = source.filter.suggestion
        source.status = "done"
    except Exception as exc:
        source.status = "failed"
        source.track_path = None
        key = (
            f"judge:{source.id}"
            if any(
                s.key == f"judge:{source.id}" and s.status == "running"
                for s in job.steps
            )
            else f"separate:{source.id}"
        )
        update_step(
            job,
            ctx,
            StepStatus(
                key=key,
                status="failed",
                round=source.attempts,
                max_rounds=maximum,
                message=str(exc),
            ),
        )
    ctx.store.save(job)
