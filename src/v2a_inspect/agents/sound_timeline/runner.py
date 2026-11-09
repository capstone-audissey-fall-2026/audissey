from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from threading import RLock

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain.agents.middleware import ToolRetryMiddleware

from v2a_inspect.config import settings
from v2a_inspect.llm import model_manager
from v2a_inspect.models import VideoAsset
from v2a_inspect.observability import build_langchain_config
from v2a_inspect.prompts.manager import PromptManager
from v2a_inspect.tools.sound_timeline import SoundTimelineEditor

FrameRange = tuple[int, int]
SceneRange = tuple[int, int]


def run_sound_timeline_agent(
    video_asset: VideoAsset,
    *,
    objective: str = "Build a complete SoundTimeline for the full video.",
    model: BaseChatModel | None = None,
    max_iterations: int | None = None,
    on_change: Callable[[str], None] | None = None,
    allowed_frame_range: FrameRange | None = None,
    allowed_scene_range: SceneRange | None = None,
    lock: RLock | None = None,
) -> None:
    """Mutate video_asset.sound_timeline by running the sound timeline agent."""

    editor = SoundTimelineEditor(
        video_asset,
        on_change=on_change,
        allowed_frame_range=allowed_frame_range,
        allowed_scene_range=allowed_scene_range,
        lock=lock,
    )
    run_agent_loop(
        editor, objective=objective, model=model, max_iterations=max_iterations
    )


def run_sound_timeline_agent_parallel(
    video_asset: VideoAsset,
    *,
    segment_seconds: int = 30,
    max_workers: int | None = None,
    objective: str = "Build a complete SoundTimeline for the assigned video segment.",
    model: BaseChatModel | None = None,
    max_iterations: int | None = None,
    on_change: Callable[[str], None] | None = None,
) -> None:
    """Mutate video_asset.sound_timeline with range-constrained parallel agents."""

    batches = _scene_batches(video_asset, segment_seconds=segment_seconds)
    if not batches:
        return
    resolved_workers = (
        len(batches) if max_workers is None or max_workers < 1 else max_workers
    )
    resolved_workers = max(1, min(resolved_workers, len(batches)))
    if resolved_workers == 1:
        run_sound_timeline_agent(
            video_asset,
            objective=objective,
            model=model,
            max_iterations=max_iterations,
            on_change=on_change,
        )
        return
    lock = RLock()
    with ThreadPoolExecutor(max_workers=resolved_workers) as executor:
        futures = [
            executor.submit(
                run_sound_timeline_agent,
                video_asset,
                objective=_range_objective(
                    objective,
                    worker_index=worker_index,
                    worker_count=len(batches),
                    scene_range=scene_range,
                    frame_range=frame_range,
                ),
                model=model,
                max_iterations=max_iterations,
                on_change=on_change,
                allowed_frame_range=frame_range,
                allowed_scene_range=scene_range,
                lock=lock,
            )
            for worker_index, (scene_range, frame_range) in enumerate(batches, start=1)
        ]
        for future in futures:
            future.result()


def run_agent_loop(
    editor: SoundTimelineEditor,
    *,
    objective: str,
    model: BaseChatModel | None,
    max_iterations: int | None,
) -> dict:
    resolved_max_iterations = _resolve_max_iterations(max_iterations)
    system_prompt, user_message = _prompt_messages(objective)
    retry_middleware = ToolRetryMiddleware(
        max_retries=5,
        retry_on=(Exception,),
    )
    agent = create_agent(
        model=model or model_manager.large,
        tools=editor.tools(),
        system_prompt=str(system_prompt.content),
        name="sound_timeline_agent",
        middleware=[retry_middleware],
    )
    return agent.invoke(
        {"messages": [HumanMessage(content=str(user_message.content))]},
        config=_agent_invoke_config(
            editor,
            objective=objective,
            max_iterations=resolved_max_iterations,
        ),
    )


def _resolve_max_iterations(max_iterations: int | None) -> int:
    if max_iterations is None:
        return settings.agent_sound_timeline_recursion_limit
    if max_iterations < 1:
        raise ValueError("max_iterations must be greater than or equal to 1")
    return max_iterations


def _agent_invoke_config(
    editor: SoundTimelineEditor,
    *,
    objective: str,
    max_iterations: int,
) -> RunnableConfig:
    config = RunnableConfig(recursion_limit=max_iterations)
    trace_config = build_langchain_config(
        run_name="sound_timeline_agent",
        tags=["v2a-inspect", "agent", "sound-timeline"],
        metadata=_trace_metadata(
            editor,
            objective=objective,
            max_iterations=max_iterations,
        ),
    )
    if trace_config is not None:
        config.update(trace_config)
    return config


def _trace_metadata(
    editor: SoundTimelineEditor,
    *,
    objective: str,
    max_iterations: int,
) -> dict[str, object]:
    video_asset = editor.video_asset
    timeline = video_asset.sound_timeline
    return {
        "video_id": str(video_asset.video_id),
        "source_path": str(video_asset.source_path),
        "frame_count": video_asset.frame_count,
        "initial_scene_count": len(video_asset.initial_scenes),
        "sound_source_count": 0 if timeline is None else len(timeline.sound_sources),
        "sound_track_count": 0 if timeline is None else len(timeline.sound_tracks),
        "sound_event_count": 0 if timeline is None else len(timeline.sound_events),
        "objective": objective,
        "max_iterations": max_iterations,
        "allowed_frame_range": editor.allowed_frame_range,
        "allowed_scene_range": editor.allowed_scene_range,
    }


def _prompt_messages(objective: str):
    messages = (
        PromptManager()
        .get_prompt("sound_timeline_agent")
        .format_messages(objective=objective)
    )
    if len(messages) != 2:
        raise ValueError(
            "sound_timeline_agent prompt must have system and user messages"
        )
    return messages[0], messages[1]


def _scene_batches(
    video_asset: VideoAsset,
    *,
    segment_seconds: int,
) -> list[tuple[SceneRange, FrameRange]]:
    if segment_seconds < 1:
        raise ValueError("segment_seconds must be greater than or equal to 1")
    if not video_asset.initial_scenes:
        return []
    segment_frames = max(1, round(video_asset.fps * segment_seconds))
    batches: list[tuple[SceneRange, FrameRange]] = []
    current_segment = None
    scene_start = 0
    frame_start = video_asset.initial_scenes[0].start_frame_index
    frame_end = video_asset.initial_scenes[0].end_frame_index
    for scene_index, scene in enumerate(video_asset.initial_scenes):
        segment_index = scene.start_frame_index // segment_frames
        if current_segment is None:
            current_segment = segment_index
        if segment_index != current_segment:
            batches.append(((scene_start, scene_index), (frame_start, frame_end)))
            current_segment = segment_index
            scene_start = scene_index
            frame_start = scene.start_frame_index
        frame_end = scene.end_frame_index
    batches.append(
        ((scene_start, len(video_asset.initial_scenes)), (frame_start, frame_end))
    )
    return batches


def _range_objective(
    objective: str,
    *,
    worker_index: int,
    worker_count: int,
    scene_range: SceneRange,
    frame_range: FrameRange,
) -> str:
    return (
        f"{objective}\n"
        f"Worker {worker_index}/{worker_count}.\n"
        f"Assigned scenes: {scene_range[0]}-{scene_range[1] - 1}.\n"
        f"Assigned frames: {frame_range[0]}-{frame_range[1]}.\n"
        "Only create or edit SoundEvents inside assigned frames. "
        "Other workers may edit the same global SoundTimeline concurrently; "
        "refresh source/track catalogs before creating reusable records."
    )
