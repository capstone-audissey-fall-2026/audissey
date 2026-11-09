from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from v2a_inspect.agents.sound_timeline import run_sound_timeline_agent_parallel
from v2a_inspect.client import SAM3Client, VideoClient
from v2a_inspect.config import settings
from v2a_inspect.models import VideoAsset
from v2a_inspect.preprocessing import (
    analyze_initial_scenes,
    build_visual_identity_layer,
    compute_visual_events,
    detect_initial_scenes,
    extract_keyframes_for_initial_scenes,
    prepare_video,
    sam3_tracking_video_path,
    track_initial_scenes_object_seeds,
)

ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class VideoAssetPipelineOptions:
    scene_threshold: float = 27.0
    max_keyframes_per_scene: int = 20
    server_url: str | None = None


def run_video_asset_pipeline(
    video_path: Path,
    work_dir: Path,
    *,
    options: VideoAssetPipelineOptions | None = None,
    on_stage: ProgressCallback | None = None,
) -> VideoAsset:
    """Run the full VideoAsset pipeline and return the completed asset."""

    resolved_options = options or VideoAssetPipelineOptions()
    work_dir.mkdir(parents=True, exist_ok=True)

    _publish(on_stage, "prepare video")
    video_asset = prepare_video(video_path, work_dir)

    _publish(on_stage, "detect scenes")
    initial_scenes = detect_initial_scenes(
        video_asset,
        threshold=resolved_options.scene_threshold,
    )
    video_asset = video_asset.model_copy(update={"initial_scenes": initial_scenes})

    _publish(on_stage, "extract keyframes")
    initial_scenes = extract_keyframes_for_initial_scenes(
        video_asset,
        work_dir,
        max_keyframes_per_scene=resolved_options.max_keyframes_per_scene,
    )
    video_asset = video_asset.model_copy(update={"initial_scenes": initial_scenes})

    _publish(on_stage, "analyze scenes")
    initial_scenes = analyze_initial_scenes(video_asset.initial_scenes)
    video_asset = video_asset.model_copy(update={"initial_scenes": initial_scenes})

    _publish(on_stage, "track objects")
    video_asset = asyncio.run(
        _track_objects(video_asset, resolved_options.server_url, on_stage)
    )

    _publish(on_stage, "build visual identity")
    video_asset = build_visual_identity_layer(video_asset)

    _publish(on_stage, "compute visual events")
    video_asset = compute_visual_events(video_asset)

    _publish(on_stage, "build sound timeline")
    run_sound_timeline_agent_parallel(
        video_asset,
        segment_seconds=settings.agent_sound_timeline_segment_seconds,
        max_workers=settings.agent_sound_timeline_max_workers,
        on_change=on_stage,
    )

    _publish(on_stage, "complete")
    return video_asset


async def _track_objects(
    video_asset: VideoAsset,
    server_url: str | None,
    on_stage: ProgressCallback | None,
) -> VideoAsset:
    tracking_video_path = sam3_tracking_video_path(video_asset)
    async with VideoClient(base_url=server_url) as video_client:
        try:
            upload_response = await video_client.upload(str(tracking_video_path))
        except Exception as exc:
            raise RuntimeError(f"upload tracking video failed: {exc}") from exc

    async with SAM3Client(base_url=server_url) as sam_client:
        scene_count = len(video_asset.initial_scenes)
        for scene_index in range(scene_count):
            scene_number = scene_index + 1
            try:
                video_asset = await track_initial_scenes_object_seeds(
                    video_asset,
                    video_id=upload_response.video_id,
                    sam_client=sam_client,
                    scene_indexes=[scene_index],
                )
            except Exception as exc:
                raise RuntimeError(
                    f"track objects scene {scene_number}/{scene_count} failed: {exc}"
                ) from exc
            _publish(on_stage, f"tracked objects {scene_number}/{scene_count}")
    return video_asset


def _publish(on_stage: ProgressCallback | None, stage: str) -> None:
    if on_stage is not None:
        on_stage(stage)
