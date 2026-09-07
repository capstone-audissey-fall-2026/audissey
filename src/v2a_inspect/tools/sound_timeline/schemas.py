from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import Field

from v2a_inspect.models import SoundEvent, SoundSource, SoundTrack
from v2a_inspect.models.base import SchemaModel


SoundSourceType = Literal[
    "visual_object",
    "scene_global",
    "offscreen_unknown",
    "non_diegetic",
]
SoundTrackType = Literal["speech", "sfx", "music", "ambience"]
SoundGenerationModel = Literal["t2a", "v2a", "tts"]
FrameResolutionMode = Literal["low", "high"]


class NoArgs(SchemaModel):
    pass


class ListScenesArgs(SchemaModel):
    start_scene_index: int = Field(default=0, ge=0)
    limit: int = Field(default=3, ge=1)


class SceneIndexArgs(SchemaModel):
    scene_index: int = Field(ge=0)


class FrameIndexArgs(SchemaModel):
    frame_index: int = Field(ge=0)
    resolution_mode: FrameResolutionMode = "low"


class VisualEventsArgs(SchemaModel):
    start_frame_index: int | None = Field(default=None, ge=0)
    end_frame_index: int | None = Field(default=None, gt=0)
    limit: int = Field(default=20, ge=1)


class SoundTimelineArgs(SchemaModel):
    start_frame_index: int | None = Field(default=None, ge=0)
    end_frame_index: int | None = Field(default=None, gt=0)
    limit: int = Field(default=20, ge=1)


class CatalogArgs(SchemaModel):
    query: str | None = None
    limit: int = Field(default=3, ge=1)


class UpsertSoundSourceArgs(SchemaModel):
    source_type: SoundSourceType
    label: str = Field(min_length=1)
    sound_source_id: UUID | None = None
    visual_object_id: UUID | None = None
    notes: str | None = None


class DeleteSoundSourceArgs(SchemaModel):
    sound_source_id: UUID


class UpsertSoundTrackArgs(SchemaModel):
    track_type: SoundTrackType
    label: str = Field(min_length=1)
    canonical_key: str | None = Field(default=None, min_length=1)
    sound_track_id: UUID | None = None
    sound_source_id: UUID | None = None
    generation_model: SoundGenerationModel = "t2a"
    notes: str | None = None


class DeleteSoundTrackArgs(SchemaModel):
    sound_track_id: UUID


class UpsertSoundEventArgs(SchemaModel):
    start_frame_index: int = Field(ge=0)
    end_frame_index: int = Field(gt=0)
    description: str = Field(min_length=1)
    sound_track_id: UUID
    spoken_text: str | None = Field(default=None, min_length=1)
    sound_event_id: UUID | None = None
    notes: str | None = None


class DeleteSoundEventArgs(SchemaModel):
    sound_event_id: UUID


class SceneListItem(SchemaModel):
    scene_index: int
    initial_scene_id: UUID
    start_frame_index: int
    end_frame_index: int
    frame_count: int
    keyframe_indexes: list[int]
    track_count: int
    object_seed_count: int
    object_labels: list[str] = Field(default_factory=list)


class ListScenesOutput(SchemaModel):
    frame_count: int
    fps: int
    total_scene_count: int
    start_scene_index: int
    returned_scene_count: int
    next_scene_index: int | None = None
    scenes: list[SceneListItem]

    def to_tool_string(self) -> str:
        lines = [
            (
                f"scenes total={self.total_scene_count} "
                f"range={self.start_scene_index}-"
                f"{self.start_scene_index + self.returned_scene_count}"
            )
        ]
        if self.next_scene_index is not None:
            lines.append(
                "truncated: request list_scenes("
                f"start_scene_index={self.next_scene_index}) for later scenes"
            )
        for scene in self.scenes:
            labels = _compact_list(disambiguate_labels(scene.object_labels))
            lines.append(
                f"- scene {scene.scene_index}: frames "
                f"{scene.start_frame_index}-{scene.end_frame_index}; "
                f"keyframes={scene.keyframe_indexes}; "
                f"tracks={scene.track_count}; object_seeds={scene.object_seed_count}"
                + (f"; objects={labels}" if labels else "")
            )
        return "\n".join(lines)


class ObjectSeedView(SchemaModel):
    label: str
    tracking_prompt: str
    seed_frame_index: int | None = None
    notes: str | None = None


class TrackSummary(SchemaModel):
    scene_track_id: UUID
    tracking_prompt: str
    source_label: str | None = None
    start_frame_index: int
    end_frame_index: int
    frame_count: int
    confidence: float
    notes: str | None = None

    @property
    def display_label(self) -> str:
        return self.source_label or self.tracking_prompt


class ListTracksOutput(SchemaModel):
    scene_index: int
    tracks: list[TrackSummary]

    def to_tool_string(self) -> str:
        lines = [f"scene {self.scene_index} tracks={len(self.tracks)}"]
        for track_number, (label, track) in enumerate(
            zip(
                disambiguate_labels([track.display_label for track in self.tracks]),
                self.tracks,
                strict=True,
            ),
            start=1,
        ):
            lines.append(_track_line(track_number, label, track))
        return "\n".join(lines)


class SceneSummaryOutput(SchemaModel):
    scene_index: int
    initial_scene_id: UUID
    start_frame_index: int
    end_frame_index: int
    frame_count: int
    keyframe_indexes: list[int]
    object_seeds: list[ObjectSeedView]
    tracks: list[TrackSummary]

    def to_tool_string(self) -> str:
        lines = [
            (
                f"scene {self.scene_index}: frames "
                f"{self.start_frame_index}-{self.end_frame_index}; "
                f"keyframes={self.keyframe_indexes}"
            )
        ]
        if self.object_seeds:
            seed_labels = disambiguate_labels(
                [seed.label for seed in self.object_seeds]
            )
            labels = _compact_list(seed_labels)
            lines.append(f"objects={labels}")
            for label, seed in zip(
                seed_labels,
                self.object_seeds,
                strict=True,
            ):
                detail = f"- object {label}: prompt={seed.tracking_prompt}"
                if seed.seed_frame_index is not None:
                    detail += f"; seed_frame={seed.seed_frame_index}"
                if seed.notes:
                    detail += f"; notes={seed.notes}"
                lines.append(detail)
        if self.tracks:
            lines.append(f"tracks={len(self.tracks)}")
            for track_number, (label, track) in enumerate(
                zip(
                    disambiguate_labels([track.display_label for track in self.tracks]),
                    self.tracks,
                    strict=True,
                ),
                start=1,
            ):
                lines.append(_track_line(track_number, label, track))
        return "\n".join(lines)


class AnnotatedFrameTrackView(SchemaModel):
    scene_index: int
    scene_track_id: UUID
    tracking_prompt: str
    source_label: str | None = None
    bbox_xyxy: tuple[float, float, float, float] | None = None
    confidence: float

    @property
    def display_label(self) -> str:
        return self.source_label or self.tracking_prompt


class AnnotatedFrameOutput(SchemaModel):
    frame_index: int
    resolution_mode: FrameResolutionMode
    width: int
    height: int
    image: str
    tracks: list[AnnotatedFrameTrackView]


class VisualEventView(SchemaModel):
    object_label: str
    related_object_labels: list[str] = Field(default_factory=list)
    start_frame_index: int
    end_frame_index: int
    event_type: str
    description: str | None = None
    confidence: float
    notes: str | None = None


class VisualEventsOutput(SchemaModel):
    total_matching_event_count: int
    start_frame_index: int | None = None
    end_frame_index: int | None = None
    limit: int
    returned_event_count: int
    visual_events: list[VisualEventView]

    def to_tool_string(self) -> str:
        frame_range = _frame_range_text(self.start_frame_index, self.end_frame_index)
        lines = [
            (
                f"visual_events frame_range={frame_range} "
                f"matching={self.total_matching_event_count} "
                f"shown={self.returned_event_count}"
            )
        ]
        if self.returned_event_count < self.total_matching_event_count:
            lines.append(
                "truncated: narrow start_frame_index/end_frame_index "
                "for more specific evidence"
            )
        for event in self.visual_events:
            related = _compact_list(event.related_object_labels)
            lines.append(
                f"- {event.start_frame_index}-{event.end_frame_index} "
                f"{event.event_type} object={event.object_label} "
                f"confidence={event.confidence:.2f}"
                + (f" related={related}" if related else "")
                + (f" description={event.description}" if event.description else "")
            )
        return "\n".join(lines)


class SoundTimelineViewOutput(SchemaModel):
    sound_sources: list[SoundSource]
    sound_tracks: list[SoundTrack]
    sound_events: list[SoundEvent]
    total_matching_event_count: int
    start_frame_index: int | None = None
    end_frame_index: int | None = None
    limit: int
    returned_event_count: int
    notes: str | None = None

    def to_tool_string(self) -> str:
        frame_range = _frame_range_text(self.start_frame_index, self.end_frame_index)
        lines = [
            (
                f"sound_timeline frame_range={frame_range} "
                f"sources={len(self.sound_sources)} "
                f"tracks={len(self.sound_tracks)} "
                f"matching_events={self.total_matching_event_count} "
                f"shown_events={self.returned_event_count}"
            )
        ]
        if self.returned_event_count < self.total_matching_event_count:
            lines.append(
                "truncated: narrow start_frame_index/end_frame_index "
                "for local edit state"
            )
        if self.notes:
            lines.append(f"notes: {self.notes}")
        if self.sound_sources:
            lines.append("sources:")
            for source_number, source in enumerate(self.sound_sources, start=1):
                lines.append(
                    f"- source {source_number}: {source.source_type} label={source.label}"
                )
        if self.sound_tracks:
            lines.append("tracks:")
            for track in self.sound_tracks:
                source_label = _sound_source_label(track, self.sound_sources)
                lines.append(
                    f"- {track.sound_track_id} {track.track_type} "
                    f"model={track.generation_model}"
                    + (
                        ""
                        if track.canonical_key is None
                        else f" canonical_key={track.canonical_key}"
                    )
                    + (f" source={source_label}" if source_label else "")
                    + f" label={track.label}"
                )
        if self.sound_events:
            lines.append("events:")
            track_by_id = {track.sound_track_id: track for track in self.sound_tracks}
            for event_number, event in enumerate(self.sound_events, start=1):
                track = track_by_id[event.sound_track_id]
                source_label = _sound_source_label(track, self.sound_sources)
                lines.append(
                    f"- event {event_number}: {event.start_frame_index}-"
                    f"{event.end_frame_index} {track.track_type} "
                    f"model={track.generation_model} "
                    f"track={track.label}"
                    + (f" source={source_label}" if source_label else "")
                    + f" description={event.description}"
                    + (
                        f" spoken_text={event.spoken_text}"
                        if event.spoken_text is not None
                        else ""
                    )
                )
        return "\n".join(lines)


class SoundSourceCatalogOutput(SchemaModel):
    total_matching_source_count: int
    limit: int
    returned_source_count: int
    sound_sources: list[SoundSource]
    query: str | None = None

    def to_tool_string(self) -> str:
        lines = [
            (
                f"sound_sources matching={self.total_matching_source_count} "
                f"shown={self.returned_source_count}"
            )
        ]
        if self.returned_source_count < self.total_matching_source_count:
            lines.append("truncated: use query for narrower catalog")
        for source in self.sound_sources:
            visual = (
                ""
                if source.visual_object_id is None
                else f" visual_object_id={source.visual_object_id}"
            )
            lines.append(
                f"- id={source.sound_source_id} type={source.source_type}"
                f"{visual} label={source.label}"
            )
        return "\n".join(lines)


class SoundTrackCatalogOutput(SchemaModel):
    total_matching_track_count: int
    limit: int
    returned_track_count: int
    sound_tracks: list[SoundTrack]
    sound_sources: list[SoundSource]
    query: str | None = None

    def to_tool_string(self) -> str:
        lines = [
            (
                f"sound_tracks matching={self.total_matching_track_count} "
                f"shown={self.returned_track_count}"
            )
        ]
        if self.returned_track_count < self.total_matching_track_count:
            lines.append("truncated: use query for narrower catalog")
        for track in self.sound_tracks:
            source_label = _sound_source_label(track, self.sound_sources)
            canonical = (
                ""
                if track.canonical_key is None
                else f" canonical_key={track.canonical_key}"
            )
            lines.append(
                f"- id={track.sound_track_id} type={track.track_type}"
                f"{canonical} model={track.generation_model} label={track.label}"
                + (f" source={source_label}" if source_label else "")
            )
        return "\n".join(lines)


class DeleteSoundSourceOutput(SchemaModel):
    deleted_sound_source_id: UUID

    def to_tool_string(self) -> str:
        return "deleted sound source"


class DeleteSoundTrackOutput(SchemaModel):
    deleted_sound_track_id: UUID


class DeleteSoundEventOutput(SchemaModel):
    deleted_sound_event_id: UUID

    def to_tool_string(self) -> str:
        return "deleted sound event"


def _track_line(track_number: int, label: str, track: TrackSummary) -> str:
    line = (
        f"- track {track_number}: {label}; "
        f"frames={track.start_frame_index}-{track.end_frame_index}; "
        f"confidence={track.confidence:.2f}"
    )
    if track.notes:
        line += f"; notes={track.notes}"
    return line


def disambiguate_labels(labels: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    output: list[str] = []
    for label in labels:
        seen[label] = seen.get(label, 0) + 1
        output.append(f"{label} #{seen[label]}" if labels.count(label) > 1 else label)
    return output


def _compact_list(items: list[str]) -> str:
    values = [item for item in items if item]
    return "[" + ", ".join(values) + "]" if values else ""


def _sound_source_label(
    track: SoundTrack,
    sound_sources: list[SoundSource],
) -> str | None:
    if track.sound_source_id is None:
        return None
    for source in sound_sources:
        if source.sound_source_id == track.sound_source_id:
            return source.label
    return None


def _frame_range_text(
    start_frame_index: int | None,
    end_frame_index: int | None,
) -> str:
    start = "*" if start_frame_index is None else str(start_frame_index)
    end = "*" if end_frame_index is None else str(end_frame_index)
    return f"{start}-{end}"
