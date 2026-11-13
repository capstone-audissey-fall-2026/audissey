import json
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from dataclasses import dataclass, field

from langsmith import traceable
from langgraph.graph import END, START, StateGraph

from audyssey.contracts import (
    AddSource,
    Command,
    Drop,
    Generate,
    Job,
    Keep,
    PlanIssue,
    Render,
    Reseparate,
    SetGain,
    SetPitch,
    Source,
    SourcePlan,
    StepStatus,
)
from audyssey.media import compute_peaks, extract_audio, is_silent, probe, write_silence
from audyssey.settings import Settings
from audyssey.store import JobStore
from audyssey.tools import Tools
from audyssey.tools.editing.render import compute_residual, render


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


def siblings(job: Job, source: Source) -> list[SourcePlan]:
    return [
        other.plan
        for other in job.sources
        if other.id not in {source.id, source.replaces}
        and other.plan.kind != "dialogue"
    ]


def generate_source(job: Job, source: Source, ctx: Context) -> None:
    if source.plan.kind == "dialogue":
        source.status, source.track_path = "skipped", None
        update_step(
            job,
            ctx,
            StepStatus(
                key=f"generate:{source.id}",
                status="done",
                message="dialogue is not generated",
            ),
        )
        return
    directory = ctx.store.dir(job.id)
    maximum = 1 + ctx.settings.max_generation_retries
    source.status = "running"
    best_path: Path | None = None
    best_result = None
    attempts: list[Path] = []
    messages: list[str] = []
    key = f"generate:{source.id}"
    reference = job.find_source(source.replaces) if source.replaces else None
    try:
        for round_number in range(1, maximum + 1):
            source.attempts += 1
            key = f"generate:{source.id}"
            update_step(
                job,
                ctx,
                StepStatus(
                    key=key, status="running", round=round_number, max_rounds=maximum
                ),
            )
            out = directory / f"tracks/{source.id}.attempt{source.attempts}.wav"
            attempts.append(out)
            path = ctx.tools.generate(
                directory / job.video_path,
                source.plan,
                out,
                ctx.settings.selva_seed + source.attempts - 1,
            )
            update_step(
                job,
                ctx,
                StepStatus(
                    key=key, status="done", round=round_number, max_rounds=maximum
                ),
            )
            key = f"score:{source.id}"
            update_step(
                job,
                ctx,
                StepStatus(
                    key=key,
                    status="running",
                    round=round_number,
                    max_rounds=maximum,
                    message="; ".join(messages),
                ),
            )
            result = ctx.tools.score_generation(
                path,
                source.plan,
                siblings(job, source),
                directory / reference.track_path
                if reference and reference.track_path
                else None,
            )
            messages.append(
                f"attempt {source.attempts}: correct={result.scores.get('text_correct')} "
                f"margin={result.scores.get('text_margin')} {result.reason}".rstrip()
            )
            if best_result is None or result.scores.get(
                "text_margin", 0
            ) >= best_result.scores.get("text_margin", 0):
                best_path, best_result = path, result
            update_step(
                job,
                ctx,
                StepStatus(
                    key=key,
                    status="done" if result.passed else "warning",
                    round=round_number,
                    max_rounds=maximum,
                    message="; ".join(messages),
                ),
            )
            if result.passed:
                break
        if best_path is None:
            raise RuntimeError("generation produced no track")
        final = directory / f"tracks/{source.id}.wav"
        best_path.replace(final)
        source.track_path = str(final.relative_to(directory))
        source.filter = best_result
        source.status = "done"
    except Exception as exc:
        source.status, source.track_path = "failed", None
        update_step(
            job,
            ctx,
            StepStatus(
                key=key,
                status="failed",
                round=round_number,
                max_rounds=maximum,
                message=str(exc),
            ),
        )
    finally:
        for path in attempts:
            path.unlink(missing_ok=True)
    ctx.store.save(job)


def process_source(job: Job, source: Source, ctx: Context) -> None:
    if job.silent or source.origin == "generated":
        generate_source(job, source, ctx)
        return
    directory = ctx.store.dir(job.id)
    maximum = ctx.settings.max_separation_retries + 1
    source.status = "running"
    masked_for, video = None, None
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
            hint = source.plan.visual_hint if ctx.settings.visual_prompt else None
            if hint is None:
                video = None
            elif hint != masked_for:
                # Masking runs SAM3 over the whole clip; reuse it while the hint holds.
                video = ctx.tools.mask_video(
                    directory / job.video_path,
                    hint,
                    directory / f"masks/{source.id}.mp4",
                )
                masked_for = hint
            path = ctx.tools.separate(
                directory / job.mono_path,
                source.plan,
                video,
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


@dataclass
class AnalysisState:
    job: Job
    issues: list[PlanIssue] = field(default_factory=list)
    round: int = 0


def refresh_residual(job: Job, ctx: Context) -> None:
    if job.silent:
        job.residual.path = None
        return
    directory = ctx.store.dir(job.id)
    update_step(job, ctx, StepStatus(key="residual", status="running"))
    compute_residual(
        directory / job.audio_path,
        [
            directory / s.track_path
            for s in job.sources
            if s.origin == "separated" and s.track_path
        ],
        directory / "residual.wav",
    )
    job.residual.path = "residual.wav"
    update_step(job, ctx, StepStatus(key="residual", status="done"))


def analyze(
    job: Job, ctx: Context, on_finish: Callable[[], None] = lambda: None
) -> Job:
    """Run the analysis graph. on_finish runs before the final state is saved."""
    directory = ctx.store.dir(job.id)
    active_key = "preprocess"

    def preprocess(state: AnalysisState) -> dict:
        update_step(job, ctx, StepStatus(key="preprocess", status="running"))
        duration, channels = probe(directory / job.video_path)
        job.duration = duration
        if channels == 0:
            write_silence(directory / "audio.wav", duration, 2)
            write_silence(directory / "mono.wav", duration, 1)
            job.silent = True
        else:
            extract_audio(directory / job.video_path, directory / "audio.wav", False)
            extract_audio(directory / job.video_path, directory / "mono.wav", True)
            job.silent = is_silent(directory / "mono.wav")
            if job.silent:
                write_silence(directory / "audio.wav", duration, channels)
                write_silence(directory / "mono.wav", duration, 1)
        job.audio_path, job.mono_path, job.peaks_path = (
            "audio.wav",
            "mono.wav",
            "peaks.json",
        )
        (directory / job.peaks_path).write_text(
            json.dumps(compute_peaks(directory / job.audio_path))
        )
        update_step(job, ctx, StepStatus(key="preprocess", status="done"))
        return {}

    def plan(state: AnalysisState) -> dict:
        nonlocal active_key
        active_key = "plan"
        round_number = state.round + 1
        update_step(
            job,
            ctx,
            StepStatus(
                key="plan",
                status="running",
                round=round_number,
                max_rounds=ctx.settings.max_plan_rounds,
            ),
        )
        plans, issues = check_plans(
            ctx.tools.plan(
                directory / job.video_path,
                directory / job.audio_path,
                job.duration,
                state.issues,
            ),
            job.duration,
        )
        previous = {source.plan.name: source for source in job.sources}
        sources = []
        for i, plan_item in enumerate(plans, 1):
            source = previous.get(plan_item.name)
            if source is None:
                source = Source(
                    id=f"s{i}",
                    plan=plan_item,
                    origin="generated" if job.silent else "separated",
                )
            else:
                source.id, source.plan = f"s{i}", plan_item
            sources.append(source)
        job.sources = sources
        update_step(
            job,
            ctx,
            StepStatus(
                key="plan",
                status="done",
                round=round_number,
                max_rounds=ctx.settings.max_plan_rounds,
            ),
        )
        return {"issues": issues, "round": round_number}

    def validate(state: AnalysisState) -> dict:
        nonlocal active_key
        active_key = "validate_plan"
        update_step(
            job,
            ctx,
            StepStatus(
                key=active_key,
                status="running",
                round=state.round,
                max_rounds=ctx.settings.max_plan_rounds,
            ),
        )
        issues = state.issues or ctx.tools.validate_plan(
            directory / job.video_path,
            directory / job.audio_path,
            [s.plan for s in job.sources],
        )
        for source in job.sources:
            hint = source.plan.visual_hint
            if hint is None or not ctx.settings.visual_prompt:
                continue
            try:
                found = ctx.tools.locate(directory / job.video_path, hint).found
                message = "visual hint target was not found"
            except Exception as exc:
                found, message = False, str(exc)
            if not found:
                issues.append(
                    PlanIssue(
                        source_name=source.plan.name,
                        kind="visual_hint_not_found",
                        message=message,
                    )
                )
        job.plan_issues = issues
        update_step(
            job,
            ctx,
            StepStatus(
                key=active_key,
                status="warning" if issues else "done",
                round=state.round,
                max_rounds=ctx.settings.max_plan_rounds,
                message="; ".join(i.message for i in issues),
            ),
        )
        if issues and state.round >= ctx.settings.max_plan_rounds:
            update_step(
                job,
                ctx,
                StepStatus(
                    key="plan",
                    status="warning",
                    round=state.round,
                    max_rounds=ctx.settings.max_plan_rounds,
                    message="; ".join(i.message for i in issues),
                ),
            )
        return {"issues": issues}

    def separate(state: AnalysisState) -> dict:
        nonlocal active_key
        active_key = "separate"
        # ponytail: sequential per-source loop, fan out with Send when real GPU tools need parallelism
        for source in job.sources:
            process_source(job, source, ctx)
        return {}

    def residual(state: AnalysisState) -> dict:
        nonlocal active_key
        active_key = "residual"
        refresh_residual(job, ctx)
        return {}

    builder = StateGraph(AnalysisState)
    for name, node in [
        ("preprocess", preprocess),
        ("plan", plan),
        ("validate", validate),
        ("separate", separate),
        ("residual", residual),
    ]:
        builder.add_node(name, node)
    builder.add_edge(START, "preprocess")
    builder.add_edge("preprocess", "plan")
    builder.add_edge("plan", "validate")
    builder.add_conditional_edges(
        "validate",
        lambda state: (
            "plan"
            if state.issues and state.round < ctx.settings.max_plan_rounds
            else "separate"
        ),
        ["plan", "separate"],
    )
    builder.add_edge("separate", "residual")
    builder.add_edge("residual", END)
    error = None
    try:
        builder.compile().invoke(
            AnalysisState(job=job),
            {"recursion_limit": 2 * ctx.settings.max_plan_rounds + 10},
        )
    except Exception as exc:
        error = str(exc)
        update_step(
            job, ctx, StepStatus(key=active_key, status="failed", message=error)
        )
    finally:
        on_finish()
    # Set the final state only now: the job object is shared, so any save after
    # this line publishes it, and on_finish must already have run.
    if error is None:
        job.state = "ready"
    else:
        job.state, job.error = "failed", error
    return ctx.store.save(job)


def handle_command(job: Job, cmd: Command, ctx: Context) -> Job:
    if isinstance(cmd, Render):
        update_step(job, ctx, StepStatus(key="render", status="running"))
        try:
            job.outputs = render(job, ctx.store.dir(job.id), cmd.include_residual_stem)
        except Exception as exc:
            update_step(
                job, ctx, StepStatus(key="render", status="failed", message=str(exc))
            )
            raise
        update_step(job, ctx, StepStatus(key="render", status="done"))
    elif isinstance(cmd, AddSource):
        if job.silent and cmd.method == "separate":
            raise ValueError("silent videos only support generation")
        if cmd.method == "generate" and cmd.plan.kind == "dialogue":
            raise ValueError("dialogue is not generated")
        plans, issues = check_plans([cmd.plan], job.duration)
        if issues:
            raise ValueError("; ".join(i.message for i in issues))
        if any(s.plan.name == cmd.plan.name for s in job.sources):
            raise ValueError("source name already exists; use reseparate")
        number = max((int(s.id[1:]) for s in job.sources), default=0) + 1
        source = Source(
            id=f"s{number}",
            plan=plans[0],
            origin="generated" if cmd.method == "generate" else "separated",
        )
        job.sources.append(source)
        process_source(job, source, ctx)
        if cmd.method == "separate":
            refresh_residual(job, ctx)
    else:
        source = job.find_source(cmd.source_id)
        target = job.residual if cmd.source_id == "residual" else source
        if target is None:
            raise KeyError(cmd.source_id)
        if isinstance(cmd, Generate):
            if source is None:
                raise ValueError("residual cannot be generated")
            if source.plan.kind == "dialogue":
                raise ValueError("dialogue is not generated")
            target_source = source
            if source.origin == "separated":
                replacement = next(
                    (
                        s
                        for s in job.sources
                        if s.origin == "generated" and s.replaces == source.id
                    ),
                    None,
                )
                target_source = replacement or Source(
                    id=f"s{max(int(s.id[1:]) for s in job.sources) + 1}",
                    plan=source.plan.model_copy(deep=True),
                    origin="generated",
                    replaces=source.id,
                )
            plan = target_source.plan.model_copy(deep=True)
            if source.origin == "separated":
                plan.name = source.plan.name + " (generated)"
            if cmd.description is not None:
                plan.description = cmd.description
            _, issues = check_plans([plan], job.duration)
            if issues:
                raise ValueError("; ".join(i.message for i in issues))
            target_source.plan = plan
            if source.origin == "separated":
                if target_source not in job.sources:
                    job.sources.append(target_source)
                source.keep = False
            generate_source(job, target_source, ctx)
        elif isinstance(cmd, (Keep, Drop)):
            target.keep = isinstance(cmd, Keep)
        elif isinstance(cmd, SetGain):
            target.gain = cmd.gain
        elif isinstance(cmd, SetPitch):
            if source is None:
                raise ValueError("residual does not support pitch")
            source.pitch_semitones = cmd.semitones
        elif isinstance(cmd, Reseparate):
            if source is None:
                raise ValueError("residual cannot be reseparated")
            if job.silent or source.origin == "generated":
                raise ValueError(
                    "generated sources and silent videos cannot be reseparated"
                )
            plan = source.plan.model_copy(deep=True)
            if cmd.description is not None:
                plan.description = cmd.description
            if cmd.spans is not None:
                plan.spans = cmd.spans
            _, issues = check_plans([plan], job.duration)
            if issues:
                raise ValueError("; ".join(i.message for i in issues))
            source.plan, source.attempts = plan, 0
            process_source(job, source, ctx)
            refresh_residual(job, ctx)
    return ctx.store.save(job)


# Preserve the callable signature across LangSmith's extra-keyword protocol.
handle_command: Callable[[Job, Command, Context], Job] = wraps(handle_command)(
    traceable(
        name="command",
        process_inputs=lambda inputs: {
            "job_id": inputs["job"].id,
            "command": inputs["cmd"].model_dump(),
        },
    )(handle_command)
)
