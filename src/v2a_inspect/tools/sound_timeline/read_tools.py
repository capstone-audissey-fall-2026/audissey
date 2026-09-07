from __future__ import annotations

import base64
from difflib import SequenceMatcher
import io
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

from v2a_inspect.media_utils.video import extract_frame
from v2a_inspect.visualization.colors import color_for_index
from v2a_inspect.visualization.drawing import draw_bbox, draw_frame_index, draw_label

from .schemas import (
    AnnotatedFrameOutput,
    AnnotatedFrameTrackView,
    FrameResolutionMode,
    ListScenesOutput,
    ListTracksOutput,
    ObjectSeedView,
    SceneListItem,
    SceneSummaryOutput,
    SoundSourceCatalogOutput,
    SoundTimelineViewOutput,
    SoundTrackCatalogOutput,
    TrackSummary,
    VisualEventView,
    VisualEventsOutput,
    disambiguate_labels,
)

if TYPE_CHECKING:
    from .editor import SoundTimelineEditor

LOW_RES_WIDTH = 640
LOW_RES_JPEG_QUALITY = 75
HIGH_RES_JPEG_QUALITY = 90
MAX_SCENE_LIMIT = 5
MAX_EVENT_LIMIT = 50
MAX_CATALOG_LIMIT = 5


class SoundTimelineReadTools:
    def __init__(self, editor: SoundTimelineEditor) -> None:
        self.editor = editor

    def list_scenes(
        self,
        start_scene_index: int = 0,
        limit: int = 3,
    ) -> ListScenesOutput:
        limit = _clamp_limit(limit, MAX_SCENE_LIMIT)
        video_asset = self.editor.video_asset
        scenes = video_asset.initial_scenes
        lower_scene_index, upper_scene_index = self.editor.scene_bounds()
        if start_scene_index < lower_scene_index:
            start_scene_index = lower_scene_index
        if start_scene_index > upper_scene_index:
            raise ValueError(f"start_scene_index out of range: {start_scene_index}")
        selected_scenes = scenes[
            start_scene_index : min(start_scene_index + limit, upper_scene_index)
        ]
        next_scene_index = start_scene_index + len(selected_scenes)
        if next_scene_index >= upper_scene_index:
            next_scene_index = None
        return ListScenesOutput(
            frame_count=video_asset.frame_count,
            fps=video_asset.fps,
            total_scene_count=upper_scene_index - lower_scene_index,
            start_scene_index=start_scene_index,
            returned_scene_count=len(selected_scenes),
            next_scene_index=next_scene_index,
            scenes=[
                SceneListItem(
                    scene_index=start_scene_index + scene_offset,
                    initial_scene_id=scene.initial_scene_id,
                    start_frame_index=scene.start_frame_index,
                    end_frame_index=scene.end_frame_index,
                    frame_count=scene.frame_count,
                    keyframe_indexes=[
                        keyframe.frame_index for keyframe in scene.keyframes
                    ],
                    track_count=len(scene.scene_tracks),
                    object_seed_count=0
                    if scene.initial_analysis is None
                    else len(scene.initial_analysis.object_seeds),
                    object_labels=[]
                    if scene.initial_analysis is None
                    else [seed.label for seed in scene.initial_analysis.object_seeds],
                )
                for scene_offset, scene in enumerate(selected_scenes)
            ],
        )

    def get_scene_summary(self, scene_index: int) -> SceneSummaryOutput:
        self.editor.check_scene_index(scene_index)
        scene = self._scene(scene_index)
        object_seeds: list[ObjectSeedView] = []
        if scene.initial_analysis is not None:
            object_seeds = [
                ObjectSeedView(
                    label=seed.label,
                    tracking_prompt=seed.tracking_prompt,
                    seed_frame_index=seed.seed_frame_index,
                    notes=seed.notes,
                )
                for seed in scene.initial_analysis.object_seeds
            ]
        return SceneSummaryOutput(
            scene_index=scene_index,
            initial_scene_id=scene.initial_scene_id,
            start_frame_index=scene.start_frame_index,
            end_frame_index=scene.end_frame_index,
            frame_count=scene.frame_count,
            keyframe_indexes=[keyframe.frame_index for keyframe in scene.keyframes],
            object_seeds=object_seeds,
            tracks=self.list_tracks(scene_index).tracks,
        )

    def get_annotated_frame(
        self,
        frame_index: int,
        resolution_mode: FrameResolutionMode = "low",
    ) -> AnnotatedFrameOutput:
        self.editor.check_frame_index(frame_index, allow_end=False)
        image = extract_frame(Path(self.editor.video_asset.source_path), frame_index)
        tracks = self._tracks_at_frame(frame_index)
        track_labels = disambiguate_labels([track.display_label for track in tracks])
        for track_index, track_view in enumerate(tracks):
            color = color_for_index(track_index)
            bbox = track_view.bbox_xyxy
            if bbox is not None:
                draw_bbox(image, bbox, color)
                label_position = (int(bbox[0]), max(0, int(bbox[1]) - 16))
            else:
                label_position = (10, 28 + (track_index * 18))
            draw_label(
                image,
                label_position,
                f"{track_labels[track_index]} {track_view.confidence:.2f}",
                color,
            )
        draw_frame_index(image, frame_index)
        image = _render_output_image(image, resolution_mode)
        return AnnotatedFrameOutput(
            frame_index=frame_index,
            resolution_mode=resolution_mode,
            width=image.width,
            height=image.height,
            image=_image_base64(image, quality=_jpeg_quality(resolution_mode)),
            tracks=tracks,
        )

    def list_tracks(self, scene_index: int) -> ListTracksOutput:
        self.editor.check_scene_index(scene_index)
        scene = self._scene(scene_index)
        return ListTracksOutput(
            scene_index=scene_index,
            tracks=[
                TrackSummary(
                    scene_track_id=track.scene_track_id,
                    tracking_prompt=track.tracking_prompt,
                    source_label=None
                    if track.source_object_seed is None
                    else track.source_object_seed.label,
                    start_frame_index=track.start_frame_index,
                    end_frame_index=track.end_frame_index,
                    frame_count=track.frame_count,
                    confidence=track.confidence,
                    notes=track.notes,
                )
                for track in scene.scene_tracks
            ],
        )

    def get_visual_events(
        self,
        start_frame_index: int | None = None,
        end_frame_index: int | None = None,
        limit: int = 20,
    ) -> VisualEventsOutput:
        limit = _clamp_limit(limit, MAX_EVENT_LIMIT)
        if (
            start_frame_index is not None
            and end_frame_index is not None
            and end_frame_index <= start_frame_index
        ):
            raise ValueError("end_frame_index must be greater than start_frame_index")
        start_frame_index, end_frame_index = self.editor.apply_frame_window(
            start_frame_index, end_frame_index
        )
        layer = self.editor.video_asset.visual_identity_layer
        if layer is None:
            return VisualEventsOutput(
                total_matching_event_count=0,
                start_frame_index=start_frame_index,
                end_frame_index=end_frame_index,
                limit=limit,
                returned_event_count=0,
                visual_events=[],
            )
        visual_object_labels = disambiguate_labels(
            [
                visual_object.label or "unknown object"
                for visual_object in layer.visual_objects
            ]
        )
        object_labels = {
            visual_object.visual_object_id: object_label
            for visual_object, object_label in zip(
                layer.visual_objects,
                visual_object_labels,
                strict=True,
            )
        }
        events = []
        for event in layer.visual_events:
            if (
                start_frame_index is not None
                and event.end_frame_index <= start_frame_index
            ):
                continue
            if (
                end_frame_index is not None
                and event.start_frame_index >= end_frame_index
            ):
                continue
            events.append(event)
        events = sorted(
            events,
            key=lambda event: (
                event.start_frame_index,
                event.end_frame_index,
                event.event_type,
                str(event.visual_event_id),
            ),
        )
        paged_events = events[:limit]
        return VisualEventsOutput(
            total_matching_event_count=len(events),
            start_frame_index=start_frame_index,
            end_frame_index=end_frame_index,
            limit=limit,
            returned_event_count=len(paged_events),
            visual_events=[
                VisualEventView(
                    object_label=object_labels.get(
                        event.visual_object_id,
                        "unknown object",
                    ),
                    related_object_labels=[
                        object_labels.get(related_object_id, "unknown object")
                        for related_object_id in event.related_visual_object_ids
                    ],
                    start_frame_index=event.start_frame_index,
                    end_frame_index=event.end_frame_index,
                    event_type=event.event_type,
                    description=event.description,
                    confidence=event.confidence,
                    notes=event.notes,
                )
                for event in paged_events
            ],
        )

    def get_sound_timeline(
        self,
        start_frame_index: int | None = None,
        end_frame_index: int | None = None,
        limit: int = 20,
    ) -> SoundTimelineViewOutput:
        limit = _clamp_limit(limit, MAX_EVENT_LIMIT)
        if (
            start_frame_index is not None
            and end_frame_index is not None
            and end_frame_index <= start_frame_index
        ):
            raise ValueError("end_frame_index must be greater than start_frame_index")
        start_frame_index, end_frame_index = self.editor.apply_frame_window(
            start_frame_index, end_frame_index
        )
        timeline = self.editor.video_asset.sound_timeline
        if timeline is None:
            return SoundTimelineViewOutput(
                sound_sources=[],
                sound_tracks=[],
                sound_events=[],
                total_matching_event_count=0,
                start_frame_index=start_frame_index,
                end_frame_index=end_frame_index,
                limit=limit,
                returned_event_count=0,
            )
        events = []
        for event in timeline.sound_events:
            if (
                start_frame_index is not None
                and event.end_frame_index <= start_frame_index
            ):
                continue
            if (
                end_frame_index is not None
                and event.start_frame_index >= end_frame_index
            ):
                continue
            events.append(event)
        track_by_id = {track.sound_track_id: track for track in timeline.sound_tracks}
        events = sorted(
            events,
            key=lambda event: (
                event.start_frame_index,
                event.end_frame_index,
                track_by_id[event.sound_track_id].track_type,
                str(event.sound_event_id),
            ),
        )
        paged_events = events[:limit]
        selected_track_ids = {event.sound_track_id for event in paged_events}
        selected_tracks = [
            track
            for track in timeline.sound_tracks
            if track.sound_track_id in selected_track_ids
        ]
        selected_source_ids = {
            track.sound_source_id
            for track in selected_tracks
            if track.sound_source_id is not None
        }
        selected_sources = [
            source
            for source in timeline.sound_sources
            if source.sound_source_id in selected_source_ids
        ]
        return SoundTimelineViewOutput(
            sound_sources=selected_sources,
            sound_tracks=selected_tracks,
            sound_events=paged_events,
            total_matching_event_count=len(events),
            start_frame_index=start_frame_index,
            end_frame_index=end_frame_index,
            limit=limit,
            returned_event_count=len(paged_events),
            notes=timeline.notes,
        )

    def list_sound_sources(
        self,
        query: str | None = None,
        limit: int = 3,
    ) -> SoundSourceCatalogOutput:
        limit = _clamp_limit(limit, MAX_CATALOG_LIMIT)
        timeline = self.editor.video_asset.sound_timeline
        sources = [] if timeline is None else timeline.sound_sources
        matching_sources = sorted(
            (
                source
                for source in sources
                if _query_score(query, source.label, source.source_type, source.notes)
                > 0
            ),
            key=lambda source: (
                -_query_score(query, source.label, source.source_type, source.notes),
                source.source_type,
                source.label.lower(),
            ),
        )
        shown_sources = matching_sources[:limit]
        return SoundSourceCatalogOutput(
            total_matching_source_count=len(matching_sources),
            limit=limit,
            returned_source_count=len(shown_sources),
            sound_sources=shown_sources,
            query=query,
        )

    def list_sound_tracks(
        self,
        query: str | None = None,
        limit: int = 3,
    ) -> SoundTrackCatalogOutput:
        limit = _clamp_limit(limit, MAX_CATALOG_LIMIT)
        timeline = self.editor.video_asset.sound_timeline
        if timeline is None:
            return SoundTrackCatalogOutput(
                total_matching_track_count=0,
                limit=limit,
                returned_track_count=0,
                sound_tracks=[],
                sound_sources=[],
                query=query,
            )
        matching_tracks = sorted(
            (
                track
                for track in timeline.sound_tracks
                if _query_score(
                    query,
                    track.label,
                    track.track_type,
                    track.generation_model,
                    track.canonical_key,
                    track.notes,
                )
                > 0
            ),
            key=lambda track: (
                -_query_score(
                    query,
                    track.label,
                    track.track_type,
                    track.generation_model,
                    track.canonical_key,
                    track.notes,
                ),
                track.track_type,
                track.canonical_key or "",
                track.label.lower(),
            ),
        )
        shown_tracks = matching_tracks[:limit]
        source_ids = {
            track.sound_source_id
            for track in shown_tracks
            if track.sound_source_id is not None
        }
        shown_sources = [
            source
            for source in timeline.sound_sources
            if source.sound_source_id in source_ids
        ]
        return SoundTrackCatalogOutput(
            total_matching_track_count=len(matching_tracks),
            limit=limit,
            returned_track_count=len(shown_tracks),
            sound_tracks=shown_tracks,
            sound_sources=shown_sources,
            query=query,
        )

    def _scene(self, scene_index: int):
        scenes = self.editor.video_asset.initial_scenes
        if scene_index < 0 or scene_index >= len(scenes):
            raise ValueError(f"scene_index out of range: {scene_index}")
        return scenes[scene_index]

    def _tracks_at_frame(self, frame_index: int) -> list[AnnotatedFrameTrackView]:
        tracks: list[AnnotatedFrameTrackView] = []
        for scene_index, scene in enumerate(self.editor.video_asset.initial_scenes):
            if not (scene.start_frame_index <= frame_index < scene.end_frame_index):
                continue
            for track in scene.scene_tracks:
                for point in track.points:
                    if point.frame_index != frame_index:
                        continue
                    tracks.append(
                        AnnotatedFrameTrackView(
                            scene_index=scene_index,
                            scene_track_id=track.scene_track_id,
                            tracking_prompt=track.tracking_prompt,
                            source_label=None
                            if track.source_object_seed is None
                            else track.source_object_seed.label,
                            bbox_xyxy=point.bbox_xyxy,
                            confidence=point.confidence,
                        )
                    )
        return tracks


def _render_output_image(
    image: Image.Image,
    resolution_mode: FrameResolutionMode,
) -> Image.Image:
    if resolution_mode == "high":
        return image

    if image.width <= LOW_RES_WIDTH:
        return image

    height = round(image.height * (LOW_RES_WIDTH / image.width))
    return image.resize((LOW_RES_WIDTH, height), Image.Resampling.LANCZOS)


def _query_score(query: str | None, *values: object) -> float:
    if query is None or not query.strip():
        return 1.0
    needle = _normalize_query_text(query)
    if not needle:
        return 1.0
    best = 0.0
    needle_tokens = set(needle.split())
    for value in values:
        if value is None:
            continue
        candidate = _normalize_query_text(str(value))
        if not candidate:
            continue
        if needle in candidate:
            best = max(best, 1.0)
            continue
        candidate_tokens = set(candidate.split())
        overlap = len(needle_tokens & candidate_tokens) / max(1, len(needle_tokens))
        ratio = SequenceMatcher(None, needle, candidate).ratio()
        best = max(best, overlap, ratio)
    return best if best >= 0.45 else 0.0


def _normalize_query_text(value: str) -> str:
    text = "".join(char.lower() if char.isalnum() else " " for char in value)
    tokens = [_stem_token(token) for token in text.split()]
    return " ".join(token for token in tokens if token)


def _stem_token(token: str) -> str:
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def _clamp_limit(limit: int, maximum: int) -> int:
    return max(1, min(limit, maximum))


def _jpeg_quality(resolution_mode: FrameResolutionMode) -> int:
    if resolution_mode == "high":
        return HIGH_RES_JPEG_QUALITY
    return LOW_RES_JPEG_QUALITY


def _image_base64(image: Image.Image, *, quality: int) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("ascii")
