from __future__ import annotations

from collections import Counter
from uuid import UUID

from v2a_inspect.models import (
    SceneTrack,
    SoundEvent,
    SoundSource,
    SoundTrack,
    VideoAsset,
    VisualEvent,
    VisualObject,
)

TableRow = dict[str, object]
_GENERIC_TRACK_LABELS = {
    "",
    "object",
    "thing",
    "item",
    "track",
    "unknown",
    "unknown object",
}


def overview_rows(video_asset: VideoAsset) -> dict[str, int | float | str]:
    track_count = sum(len(scene.scene_tracks) for scene in video_asset.initial_scenes)
    visual_object_count = (
        0
        if video_asset.visual_identity_layer is None
        else len(video_asset.visual_identity_layer.visual_objects)
    )
    visual_event_count = (
        0
        if video_asset.visual_identity_layer is None
        else len(video_asset.visual_identity_layer.visual_events)
    )
    sound_source_count = (
        0
        if video_asset.sound_timeline is None
        else len(video_asset.sound_timeline.sound_sources)
    )
    sound_track_count = (
        0
        if video_asset.sound_timeline is None
        else len(video_asset.sound_timeline.sound_tracks)
    )
    sound_event_count = (
        0
        if video_asset.sound_timeline is None
        else len(video_asset.sound_timeline.sound_events)
    )
    return {
        "video_id": str(video_asset.video_id),
        "frames": video_asset.frame_count,
        "fps": video_asset.fps,
        "duration_sec": round(video_asset.duration_sec, 2),
        "scenes": len(video_asset.initial_scenes),
        "scene_tracks": track_count,
        "visual_objects": visual_object_count,
        "visual_events": visual_event_count,
        "sound_sources": sound_source_count,
        "sound_tracks": sound_track_count,
        "sound_events": sound_event_count,
    }


def scene_rows(video_asset: VideoAsset) -> list[TableRow]:
    rows: list[TableRow] = []
    for scene_index, scene in enumerate(video_asset.initial_scenes):
        rows.append(
            {
                "scene": scene_index,
                "start_frame": scene.start_frame_index,
                "end_frame": scene.end_frame_index,
                "duration_sec": round(scene.frame_count / video_asset.fps, 2),
                "keyframes": len(scene.keyframes),
                "object_seeds": 0
                if scene.initial_analysis is None
                else len(scene.initial_analysis.object_seeds),
                "tracks": len(scene.scene_tracks),
            }
        )
    return rows


def track_rows(video_asset: VideoAsset) -> list[TableRow]:
    rows: list[TableRow] = []
    for scene_index, scene in enumerate(video_asset.initial_scenes):
        for track_index, track in enumerate(scene.scene_tracks):
            rows.append(
                {
                    "scene": scene_index,
                    "track": track_index,
                    "label": _track_label(track),
                    "start_frame": track.start_frame_index,
                    "end_frame": track.end_frame_index,
                    "duration_sec": round(track.frame_count / video_asset.fps, 2),
                    "confidence": round(track.confidence, 3),
                    "points": len(track.points),
                }
            )
    return rows


def visual_event_rows(video_asset: VideoAsset) -> list[TableRow]:
    layer = video_asset.visual_identity_layer
    if layer is None:
        return []
    object_labels = _visual_object_labels(layer.visual_objects)
    return [
        _visual_event_row(event, object_labels, video_asset.fps)
        for event in sorted(
            layer.visual_events,
            key=lambda item: (
                item.start_frame_index,
                item.end_frame_index,
                item.event_type,
            ),
        )
    ]


def sound_event_rows(video_asset: VideoAsset) -> list[TableRow]:
    timeline = video_asset.sound_timeline
    if timeline is None:
        return []
    sources_by_id = {
        source.sound_source_id: source for source in timeline.sound_sources
    }
    track_by_id = {track.sound_track_id: track for track in timeline.sound_tracks}
    rows: list[TableRow] = []
    for event in sorted(
        timeline.sound_events,
        key=lambda item: (
            track_by_id[item.sound_track_id].track_type,
            item.start_frame_index,
            item.end_frame_index,
            item.description,
        ),
    ):
        track = track_by_id[event.sound_track_id]
        source = (
            None
            if track.sound_source_id is None
            else sources_by_id.get(track.sound_source_id)
        )
        rows.append(_sound_event_row(event, track, source, video_asset.fps))
    return rows


def active_scene(video_asset: VideoAsset, frame_index: int) -> TableRow | None:
    for scene_index, scene in enumerate(video_asset.initial_scenes):
        if scene.start_frame_index <= frame_index < scene.end_frame_index:
            return {
                "scene": scene_index,
                "start_frame": scene.start_frame_index,
                "end_frame": scene.end_frame_index,
                "duration_sec": round(scene.frame_count / video_asset.fps, 2),
            }
    return None


def active_track_rows(video_asset: VideoAsset, frame_index: int) -> list[TableRow]:
    rows: list[TableRow] = []
    for scene_index, scene in enumerate(video_asset.initial_scenes):
        for track_index, track in enumerate(scene.scene_tracks):
            point = next(
                (item for item in track.points if item.frame_index == frame_index),
                None,
            )
            if point is None:
                continue
            rows.append(
                {
                    "scene": scene_index,
                    "track": track_index,
                    "label": _track_label(track),
                    "bbox": None
                    if point.bbox_xyxy is None
                    else [round(value, 1) for value in point.bbox_xyxy],
                    "confidence": round(point.confidence, 3),
                    "has_mask": point.mask is not None,
                }
            )
    return rows


def active_visual_event_rows(
    video_asset: VideoAsset, frame_index: int
) -> list[TableRow]:
    return [
        row
        for row in visual_event_rows(video_asset)
        if _frame_value(row, "start_frame")
        <= frame_index
        < _frame_value(row, "end_frame")
    ]


def active_sound_event_rows(
    video_asset: VideoAsset, frame_index: int
) -> list[TableRow]:
    return [
        row
        for row in sound_event_rows(video_asset)
        if _frame_value(row, "start_frame")
        <= frame_index
        < _frame_value(row, "end_frame")
    ]


def timeline_rows(video_asset: VideoAsset) -> list[TableRow]:
    rows: list[TableRow] = []
    for row in scene_rows(video_asset):
        rows.append(
            {
                "lane": "scenes",
                "label": f"scene {row['scene']}",
                "start_frame": row["start_frame"],
                "end_frame": row["end_frame"],
                "kind": "scene",
            }
        )
    for row in track_rows(video_asset):
        rows.append(
            {
                "lane": "tracking",
                "label": row["label"],
                "start_frame": row["start_frame"],
                "end_frame": row["end_frame"],
                "kind": "track",
            }
        )
    for row in visual_event_rows(video_asset):
        rows.append(
            {
                "lane": f"visual: {row['object']}",
                "label": f"{row['event_type']}: {row['description']}",
                "start_frame": row["start_frame"],
                "end_frame": row["end_frame"],
                "kind": row["event_type"],
            }
        )
    for row in sound_event_rows(video_asset):
        rows.append(
            {
                "lane": f"[{row['track_type']}] {row['track_label']}",
                "label": row["description"],
                "start_frame": row["start_frame"],
                "end_frame": row["end_frame"],
                "kind": row["track_type"],
                "sound_event_id": row["sound_event_id"],
                "sound_track_id": row["sound_track_id"],
                "generation_model": row["generation_model"],
            }
        )
    return rows


def current_frame_rows(video_asset: VideoAsset, frame_index: int) -> dict[str, object]:
    return {
        "scene": active_scene(video_asset, frame_index),
        "tracks": active_track_rows(video_asset, frame_index),
        "visual_events": active_visual_event_rows(video_asset, frame_index),
        "sound_events": active_sound_event_rows(video_asset, frame_index),
    }


def tracking_window_rows(
    video_asset: VideoAsset, start_frame: int, end_frame: int
) -> list[TableRow]:
    rows: list[TableRow] = []
    for scene_index, scene in enumerate(video_asset.initial_scenes):
        if scene.end_frame_index < start_frame or scene.start_frame_index > end_frame:
            continue
        for track_index, track in enumerate(scene.scene_tracks):
            if (
                track.end_frame_index < start_frame
                or track.start_frame_index > end_frame
            ):
                continue
            points = [
                {
                    "frame_index": point.frame_index,
                    "bbox_xyxy": point.bbox_xyxy,
                    "confidence": round(point.confidence, 3),
                }
                for point in track.points
                if start_frame <= point.frame_index <= end_frame
                and point.bbox_xyxy is not None
            ]
            if not points:
                continue
            rows.append(
                {
                    "scene": scene_index,
                    "track": track_index,
                    "label": _track_label(track),
                    "start_frame": track.start_frame_index,
                    "end_frame": track.end_frame_index,
                    "confidence": round(track.confidence, 3),
                    "points": points,
                }
            )
    return rows


def _visual_event_row(
    event: VisualEvent,
    object_labels: dict[UUID, str],
    fps: int,
) -> TableRow:
    return {
        "object": object_labels.get(event.visual_object_id, "unknown object"),
        "related": ", ".join(
            object_labels.get(object_id, "unknown object")
            for object_id in event.related_visual_object_ids
        ),
        "event_type": event.event_type,
        "start_frame": event.start_frame_index,
        "end_frame": event.end_frame_index,
        "duration_sec": round(
            (event.end_frame_index - event.start_frame_index) / fps, 2
        ),
        "confidence": round(event.confidence, 3),
        "description": event.description,
        "notes": event.notes,
    }


def _sound_event_row(
    event: SoundEvent,
    track: SoundTrack,
    source: SoundSource | None,
    fps: int,
) -> TableRow:
    return {
        "sound_event_id": str(event.sound_event_id),
        "sound_track_id": str(track.sound_track_id),
        "track_label": track.label,
        "track_type": track.track_type,
        "source": None if source is None else source.label,
        "start_frame": event.start_frame_index,
        "end_frame": event.end_frame_index,
        "duration_sec": round(
            (event.end_frame_index - event.start_frame_index) / fps, 2
        ),
        "generation_model": track.generation_model,
        "description": event.description,
        "spoken_text": event.spoken_text,
        "notes": event.notes,
    }


def track_display_label(track: SceneTrack, fallback_index: int | None = None) -> str:
    if track.source_object_seed is not None:
        seed_label = _useful_label(track.source_object_seed.label)
        if seed_label is not None:
            return seed_label
    prompt_label = _useful_label(track.tracking_prompt)
    if prompt_label is not None:
        return prompt_label
    if fallback_index is not None:
        return f"track {fallback_index + 1}"
    return "track"


def _track_label(track: SceneTrack) -> str:
    return track_display_label(track)


def _useful_label(value: str | None) -> str | None:
    if value is None:
        return None
    label = value.strip()
    if label.lower() in _GENERIC_TRACK_LABELS:
        return None
    return label


def _visual_object_labels(visual_objects: list[VisualObject]) -> dict[UUID, str]:
    labels = [
        visual_object.label or "unknown object" for visual_object in visual_objects
    ]
    counts = Counter(labels)
    seen: Counter[str] = Counter()
    output: dict[UUID, str] = {}
    for visual_object, label in zip(visual_objects, labels, strict=True):
        seen[label] += 1
        output[visual_object.visual_object_id] = (
            f"{label} #{seen[label]}" if counts[label] > 1 else label
        )
    return output


def _frame_value(row: TableRow, key: str) -> int:
    value = row[key]
    if not isinstance(value, int):
        raise TypeError(f"{key} must be an int")
    return value
