from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

from v2a_inspect.client import SAM3Client
from v2a_inspect.client.endpoints.base import ClientError
from v2a_inspect.client.models import Sam3Track, Sam3TrackPoint
from v2a_inspect.media_utils import resize_coco_rle
from v2a_inspect.models import (
    InitialScene,
    MaskRef,
    ObjectSeed,
    SceneTrack,
    SceneTrackPoint,
    VideoAsset,
)


logger = logging.getLogger(__name__)


async def track_initial_scene_object_seeds(
    initial_scene: InitialScene,
    *,
    video_id: str,
    sam_client: SAM3Client,
    tracking_width: int = 1280,
    tracking_height: int = 720,
    output_width: int = 1280,
    output_height: int = 720,
    seed_frame_index: int | None = None,
    score_threshold: float = 0.35,
    min_points: int = 2,
    high_confidence_threshold: float = 0.45,
    match_threshold: float = 0.45,
    min_track_mean_confidence: float = 0.0,
) -> InitialScene:
    """Track all LLM object seeds for one scene using bounded SAM3 inference."""

    if initial_scene.initial_analysis is None:
        return initial_scene

    scene_tracks: list[SceneTrack] = []

    for object_seed in initial_scene.initial_analysis.object_seeds:
        tracking_prompt = _tracking_prompt(object_seed)
        resolved_seed_frame_index = _seed_frame_index(
            initial_scene,
            object_seed=object_seed,
            seed_frame_index=seed_frame_index,
        )
        seed = SAM3Client.seed_from_prompt(
            tracking_prompt,
            frame_index=resolved_seed_frame_index,
        )
        try:
            response = await sam_client.track_video(
                video_id,
                seeds=[seed],
                start_frame_index=initial_scene.start_frame_index,
                end_frame_index=initial_scene.end_frame_index,
                score_threshold=score_threshold,
                min_points=min_points,
                high_confidence_threshold=high_confidence_threshold,
                match_threshold=match_threshold,
            )
        except ClientError as exc:
            if not _is_seed_tracking_failure(exc):
                raise
            logger.warning(
                "Skipping object seed %s in scene %s: %s",
                object_seed.label,
                initial_scene.initial_scene_id,
                exc,
            )
            continue
        for sam_track in response.tracks:
            scene_track = _scene_track_from_sam_track(
                sam_track,
                object_seed=object_seed,
                tracking_prompt=tracking_prompt,
                tracking_width=tracking_width,
                tracking_height=tracking_height,
                output_width=output_width,
                output_height=output_height,
            )
            if scene_track is None:
                continue
            if scene_track.confidence < min_track_mean_confidence:
                continue
            scene_tracks.append(scene_track)

    return initial_scene.model_copy(update={"scene_tracks": scene_tracks})


async def track_initial_scenes_object_seeds(
    video_asset: VideoAsset,
    *,
    video_id: str,
    sam_client: SAM3Client,
    scene_indexes: Iterable[int],
    score_threshold: float = 0.35,
    min_points: int = 2,
    high_confidence_threshold: float = 0.45,
    match_threshold: float = 0.45,
    min_track_mean_confidence: float = 0.0,
) -> VideoAsset:
    """Track object seeds for explicitly selected scenes and return a new asset.

    The provided ``video_id`` should refer to ``sam3_tracking_video_path(video_asset)``.
    Returned boxes and masks are scaled back to the canonical prepared video size.
    """

    selected_scene_indexes = set(scene_indexes)
    _validate_scene_indexes(selected_scene_indexes, len(video_asset.initial_scenes))
    tracking_width = video_asset.width
    tracking_height = video_asset.height
    if video_asset.sam3_tracking_path is not None:
        tracking_width = video_asset.sam3_tracking_width
        tracking_height = video_asset.sam3_tracking_height

    updated_scenes: list[InitialScene] = []
    for scene_index, initial_scene in enumerate(video_asset.initial_scenes):
        if scene_index not in selected_scene_indexes:
            updated_scenes.append(initial_scene)
            continue

        updated_scene = await track_initial_scene_object_seeds(
            initial_scene,
            video_id=video_id,
            sam_client=sam_client,
            tracking_width=tracking_width,
            tracking_height=tracking_height,
            output_width=video_asset.width,
            output_height=video_asset.height,
            score_threshold=score_threshold,
            min_points=min_points,
            high_confidence_threshold=high_confidence_threshold,
            match_threshold=match_threshold,
            min_track_mean_confidence=min_track_mean_confidence,
        )
        updated_scenes.append(updated_scene)

    return video_asset.model_copy(update={"initial_scenes": updated_scenes})


def sam3_tracking_video_path(video_asset: VideoAsset) -> Path:
    """Return the video path callers should upload for SAM3 tracking."""

    if video_asset.sam3_tracking_path is not None:
        return video_asset.sam3_tracking_path
    return video_asset.source_path


def _validate_scene_indexes(scene_indexes: set[int], scene_count: int) -> None:
    for scene_index in scene_indexes:
        if scene_index < 0 or scene_index >= scene_count:
            raise ValueError(f"scene index {scene_index} is outside available scenes")


def _seed_frame_index(
    initial_scene: InitialScene,
    *,
    object_seed: ObjectSeed | None = None,
    seed_frame_index: int | None,
) -> int:
    if seed_frame_index is not None:
        if _is_frame_in_scene(initial_scene, seed_frame_index):
            return seed_frame_index
        raise ValueError(
            "seed_frame_index must be inside the initial scene frame range"
        )

    if object_seed is not None and object_seed.seed_frame_index is not None:
        if _is_frame_in_scene(initial_scene, object_seed.seed_frame_index):
            return object_seed.seed_frame_index
        logger.warning(
            "Ignoring object seed frame %s outside scene %s range %s-%s",
            object_seed.seed_frame_index,
            initial_scene.initial_scene_id,
            initial_scene.start_frame_index,
            initial_scene.end_frame_index,
        )

    return _midpoint_frame_index(initial_scene)


def _is_frame_in_scene(initial_scene: InitialScene, frame_index: int) -> bool:
    return (
        initial_scene.start_frame_index <= frame_index < initial_scene.end_frame_index
    )


def _midpoint_frame_index(initial_scene: InitialScene) -> int:
    return (initial_scene.start_frame_index + initial_scene.end_frame_index - 1) // 2


def _tracking_prompt(object_seed: ObjectSeed) -> str:
    if object_seed.tracking_prompt:
        return object_seed.tracking_prompt
    return object_seed.label


def _is_seed_tracking_failure(exc: ClientError) -> bool:
    message = str(exc)
    return (
        "No points are provided" in message
        or "HTTP 400" in message
        or "HTTP 422" in message
    )


def _scene_track_from_sam_track(
    sam_track: Sam3Track,
    *,
    object_seed: ObjectSeed,
    tracking_prompt: str,
    tracking_width: int,
    tracking_height: int,
    output_width: int,
    output_height: int,
) -> SceneTrack | None:
    points = _scene_track_points_from_sam_points(
        sam_track.points,
        tracking_width=tracking_width,
        tracking_height=tracking_height,
        output_width=output_width,
        output_height=output_height,
    )
    if not points:
        return None

    confidence = _mean_confidence(points)
    return SceneTrack(
        source_object_seed=object_seed,
        tracking_prompt=tracking_prompt,
        points=points,
        confidence=confidence,
    )


def _scene_track_points_from_sam_points(
    sam_points: list[Sam3TrackPoint],
    *,
    tracking_width: int,
    tracking_height: int,
    output_width: int,
    output_height: int,
) -> list[SceneTrackPoint]:
    points: list[SceneTrackPoint] = []
    for sam_point in sam_points:
        if sam_point.confidence < 0 or sam_point.confidence > 1:
            continue

        mask = None
        if sam_point.mask_rle is not None:
            mask_rle = _scale_mask_rle(
                sam_point.mask_rle,
                tracking_width=tracking_width,
                tracking_height=tracking_height,
                output_width=output_width,
                output_height=output_height,
            )
            mask = MaskRef(rle=mask_rle)

        points.append(
            SceneTrackPoint(
                frame_index=sam_point.frame_index,
                bbox_xyxy=_scale_bbox_xyxy(
                    sam_point.bbox_xyxy,
                    tracking_width=tracking_width,
                    tracking_height=tracking_height,
                    output_width=output_width,
                    output_height=output_height,
                ),
                mask=mask,
                confidence=sam_point.confidence,
            )
        )
    return points


def _scale_bbox_xyxy(
    bbox_xyxy: tuple[float, float, float, float] | None,
    *,
    tracking_width: int,
    tracking_height: int,
    output_width: int,
    output_height: int,
) -> tuple[float, float, float, float] | None:
    if bbox_xyxy is None:
        return None
    if tracking_width == output_width and tracking_height == output_height:
        return bbox_xyxy

    scale_x = output_width / tracking_width
    scale_y = output_height / tracking_height
    x1, y1, x2, y2 = bbox_xyxy
    return (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)


def _scale_mask_rle(
    mask_rle: str,
    *,
    tracking_width: int,
    tracking_height: int,
    output_width: int,
    output_height: int,
) -> str:
    if tracking_width == output_width and tracking_height == output_height:
        return mask_rle
    return resize_coco_rle(mask_rle, width=output_width, height=output_height)


def _mean_confidence(points: list[SceneTrackPoint]) -> float:
    total = 0.0
    for point in points:
        total += point.confidence
    return total / len(points)
