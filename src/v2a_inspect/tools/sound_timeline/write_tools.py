from __future__ import annotations

import re
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from v2a_inspect.models import SoundEvent, SoundSource, SoundTrack

from .schemas import (
    DeleteSoundEventOutput,
    DeleteSoundSourceOutput,
    DeleteSoundTrackOutput,
    SoundGenerationModel,
    SoundSourceType,
    SoundTrackType,
)

if TYPE_CHECKING:
    from .editor import SoundTimelineEditor


class SoundTimelineWriteTools:
    def __init__(self, editor: SoundTimelineEditor) -> None:
        self.editor = editor

    def upsert_sound_source(
        self,
        source_type: SoundSourceType,
        label: str,
        sound_source_id: UUID | None = None,
        visual_object_id: UUID | None = None,
        notes: str | None = None,
    ) -> SoundSource:
        with self.editor.lock:
            timeline = self.editor.ensure_sound_timeline()
            sources = list(timeline.sound_sources)
            if sound_source_id is None:
                source = _find_matching_source(
                    sources,
                    source_type=source_type,
                    label=label,
                    visual_object_id=visual_object_id,
                )
                if source is None:
                    source = SoundSource(
                        sound_source_id=uuid4(),
                        source_type=source_type,
                        label=label,
                        visual_object_id=visual_object_id,
                        notes=notes,
                    )
                    sources.append(source)
                elif notes is not None and source.notes != notes:
                    source = source.model_copy(update={"notes": notes})
                    sources = [
                        source
                        if existing.sound_source_id == source.sound_source_id
                        else existing
                        for existing in sources
                    ]
            else:
                source = SoundSource(
                    sound_source_id=sound_source_id,
                    source_type=source_type,
                    label=label,
                    visual_object_id=visual_object_id,
                    notes=notes,
                )
                for index, existing in enumerate(sources):
                    if existing.sound_source_id == sound_source_id:
                        sources[index] = source
                        break
                else:
                    raise ValueError(f"Unknown sound_source_id: {sound_source_id}")
            self.editor.video_asset.sound_timeline = timeline.model_copy(
                update={"sound_sources": sources}
            )
        self.editor.notify_changed("updated sound source")
        return source

    def delete_sound_source(self, sound_source_id: UUID) -> DeleteSoundSourceOutput:
        with self.editor.lock:
            timeline = self.editor.ensure_sound_timeline()
            sources = [
                source
                for source in timeline.sound_sources
                if source.sound_source_id != sound_source_id
            ]
            if len(sources) == len(timeline.sound_sources):
                raise ValueError(f"Unknown sound_source_id: {sound_source_id}")
            tracks = [
                track.model_copy(update={"sound_source_id": None})
                if track.sound_source_id == sound_source_id
                else track
                for track in timeline.sound_tracks
            ]
            self.editor.video_asset.sound_timeline = timeline.model_copy(
                update={"sound_sources": sources, "sound_tracks": tracks}
            )
        self.editor.notify_changed("deleted sound source")
        return DeleteSoundSourceOutput(deleted_sound_source_id=sound_source_id)

    def upsert_sound_track(
        self,
        track_type: SoundTrackType,
        label: str,
        canonical_key: str | None = None,
        sound_track_id: UUID | None = None,
        sound_source_id: UUID | None = None,
        generation_model: SoundGenerationModel = "t2a",
        notes: str | None = None,
    ) -> SoundTrack:
        if track_type == "speech":
            generation_model = "tts"
        with self.editor.lock:
            timeline = self.editor.ensure_sound_timeline()
            if sound_source_id is not None and not any(
                source.sound_source_id == sound_source_id
                for source in timeline.sound_sources
            ):
                raise ValueError(f"Unknown sound_source_id: {sound_source_id}")
            tracks = list(timeline.sound_tracks)
            if sound_track_id is None:
                track = _find_matching_track(
                    tracks,
                    track_type=track_type,
                    label=label,
                    canonical_key=canonical_key,
                    sound_source_id=sound_source_id,
                    generation_model=generation_model,
                )
                if track is None:
                    track = SoundTrack(
                        sound_track_id=uuid4(),
                        track_type=track_type,
                        label=label,
                        canonical_key=_normalize_canonical_key(canonical_key),
                        sound_source_id=sound_source_id,
                        generation_model=generation_model,
                        notes=notes,
                    )
                    tracks.append(track)
                elif notes is not None and track.notes != notes:
                    track = track.model_copy(update={"notes": notes})
                    tracks = [
                        track
                        if existing.sound_track_id == track.sound_track_id
                        else existing
                        for existing in tracks
                    ]
            else:
                track = SoundTrack(
                    sound_track_id=sound_track_id,
                    track_type=track_type,
                    label=label,
                    canonical_key=_normalize_canonical_key(canonical_key),
                    sound_source_id=sound_source_id,
                    generation_model=generation_model,
                    notes=notes,
                )
                for index, existing in enumerate(tracks):
                    if existing.sound_track_id == sound_track_id:
                        tracks[index] = track
                        break
                else:
                    raise ValueError(f"Unknown sound_track_id: {sound_track_id}")
            self.editor.video_asset.sound_timeline = timeline.model_copy(
                update={"sound_tracks": tracks}
            )
        self.editor.notify_changed("updated sound track")
        return track

    def delete_sound_track(self, sound_track_id: UUID) -> DeleteSoundTrackOutput:
        with self.editor.lock:
            timeline = self.editor.ensure_sound_timeline()
            if any(
                event.sound_track_id == sound_track_id
                for event in timeline.sound_events
            ):
                raise ValueError(
                    f"Cannot delete sound_track_id with existing events: {sound_track_id}"
                )
            tracks = [
                track
                for track in timeline.sound_tracks
                if track.sound_track_id != sound_track_id
            ]
            if len(tracks) == len(timeline.sound_tracks):
                raise ValueError(f"Unknown sound_track_id: {sound_track_id}")
            self.editor.video_asset.sound_timeline = timeline.model_copy(
                update={"sound_tracks": tracks}
            )
        self.editor.notify_changed("deleted sound track")
        return DeleteSoundTrackOutput(deleted_sound_track_id=sound_track_id)

    def upsert_sound_event(
        self,
        start_frame_index: int,
        end_frame_index: int,
        description: str,
        sound_track_id: UUID,
        spoken_text: str | None = None,
        sound_event_id: UUID | None = None,
        notes: str | None = None,
    ) -> SoundEvent:
        self.editor.check_frame_range(start_frame_index, end_frame_index)
        with self.editor.lock:
            timeline = self.editor.ensure_sound_timeline()
            track = next(
                (
                    track
                    for track in timeline.sound_tracks
                    if track.sound_track_id == sound_track_id
                ),
                None,
            )
            if track is None:
                raise ValueError(f"Unknown sound_track_id: {sound_track_id}")
            if track.track_type == "speech" and not spoken_text:
                raise ValueError("speech events require spoken_text")
            if track.track_type != "speech" and spoken_text is not None:
                raise ValueError("spoken_text is only valid for speech events")
            event = SoundEvent(
                sound_event_id=sound_event_id or uuid4(),
                sound_track_id=sound_track_id,
                start_frame_index=start_frame_index,
                end_frame_index=end_frame_index,
                description=description,
                spoken_text=spoken_text,
                notes=notes,
            )
            events = list(timeline.sound_events)
            if sound_event_id is None:
                events.append(event)
            else:
                for index, existing in enumerate(events):
                    if existing.sound_event_id == sound_event_id:
                        events[index] = event
                        break
                else:
                    raise ValueError(f"Unknown sound_event_id: {sound_event_id}")
            self.editor.video_asset.sound_timeline = timeline.model_copy(
                update={"sound_events": events}
            )
        self.editor.notify_changed("updated sound event")
        return event

    def delete_sound_event(self, sound_event_id: UUID) -> DeleteSoundEventOutput:
        with self.editor.lock:
            timeline = self.editor.ensure_sound_timeline()
            events = [
                event
                for event in timeline.sound_events
                if event.sound_event_id != sound_event_id
            ]
            if len(events) == len(timeline.sound_events):
                raise ValueError(f"Unknown sound_event_id: {sound_event_id}")
            self.editor.video_asset.sound_timeline = timeline.model_copy(
                update={"sound_events": events}
            )
        self.editor.notify_changed("deleted sound event")
        return DeleteSoundEventOutput(deleted_sound_event_id=sound_event_id)


def _find_matching_source(
    sources: list[SoundSource],
    *,
    source_type: SoundSourceType,
    label: str,
    visual_object_id: UUID | None,
) -> SoundSource | None:
    key = (source_type, _normalize_text(label), visual_object_id)
    for source in sources:
        if (
            source.source_type,
            _normalize_text(source.label),
            source.visual_object_id,
        ) == key:
            return source
    return None


def _find_matching_track(
    tracks: list[SoundTrack],
    *,
    track_type: SoundTrackType,
    label: str,
    canonical_key: str | None,
    sound_source_id: UUID | None,
    generation_model: SoundGenerationModel,
) -> SoundTrack | None:
    normalized_key = _normalize_canonical_key(canonical_key)
    fallback_label = _normalize_text(label)
    for track in tracks:
        if (
            track.track_type != track_type
            or track.sound_source_id != sound_source_id
            or track.generation_model != generation_model
        ):
            continue
        if normalized_key is not None:
            if _normalize_canonical_key(track.canonical_key) == normalized_key:
                return track
        elif (
            track.canonical_key is None
            and _normalize_text(track.label) == fallback_label
        ):
            return track
    return None


def _normalize_canonical_key(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    return normalized or None


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())
