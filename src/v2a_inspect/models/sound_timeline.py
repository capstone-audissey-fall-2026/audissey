from __future__ import annotations

import re
from typing import Annotated, Literal
from uuid import UUID, uuid4

from pydantic import Field, StringConstraints, model_validator
from typing_extensions import Self

from .base import SchemaModel

SoundTrackType = Literal["speech", "sfx", "ambience"]
SoundGenerationModel = Literal["t2a", "v2a", "tts"]
SpokenText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class SoundSource(SchemaModel):
    """
    Reusable origin of sound in the editable audio plan.

    A source may correspond to a VisualObject, but it does not have to.
    """

    sound_source_id: UUID = Field(default_factory=uuid4)

    source_type: Literal[
        "visual_object",
        "scene_global",
        "offscreen_unknown",
        "non_diegetic",
    ]
    label: str

    visual_object_id: UUID | None = None

    notes: str | None = None


class SoundTrack(SchemaModel):
    """
    Reusable audible layer in the editable audio plan.

    A track is the timeline lane for one recurring sound identity, such as
    "Red Samurai sword clash" or "battlefield wind ambience".
    """

    sound_track_id: UUID = Field(default_factory=uuid4)

    track_type: SoundTrackType
    label: str
    canonical_key: str | None = None
    sound_source_id: UUID | None = None
    generation_model: SoundGenerationModel = "t2a"

    notes: str | None = None

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_values(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if data.get("track_type") == "dialogue":
            data["track_type"] = "speech"

        legacy_model = data.pop("generation_mode", None)
        generation_model = data.get("generation_model", legacy_model)
        if data.get("track_type") == "speech":
            data["generation_model"] = "tts"
        else:
            data["generation_model"] = {
                None: "t2a",
                "tta": "t2a",
                "vta": "v2a",
                "hybrid": "t2a",
                "unknown": "t2a",
            }.get(generation_model, generation_model)
        return data

    @model_validator(mode="after")
    def check_generation_model(self) -> Self:
        if self.track_type != "speech" and self.generation_model == "tts":
            raise ValueError("Only speech tracks may use tts")
        return self


class SoundEvent(SchemaModel):
    """One occurrence of a SoundTrack over a frame interval."""

    sound_event_id: UUID = Field(default_factory=uuid4)
    sound_track_id: UUID

    start_frame_index: int = Field(ge=0)
    end_frame_index: int = Field(gt=0)

    description: str
    spoken_text: SpokenText | None = None
    notes: str | None = None


class SoundTimeline(SchemaModel):
    """
    Canonical editable audio plan for the video.

    Export formats and multitrack views should be derived from this layer.
    """

    sound_sources: list[SoundSource] = Field(default_factory=list)
    sound_tracks: list[SoundTrack] = Field(default_factory=list)
    sound_events: list[SoundEvent] = Field(default_factory=list)

    notes: str | None = None

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_dialogue(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        legacy_speech_ids = {
            str(track.get("sound_track_id"))
            for track in data.get("sound_tracks", [])
            if isinstance(track, dict) and track.get("track_type") == "dialogue"
        }
        if not legacy_speech_ids:
            return data

        events = []
        for raw_event in data.get("sound_events", []):
            if not isinstance(raw_event, dict):
                events.append(raw_event)
                continue
            event = dict(raw_event)
            if str(event.get("sound_track_id")) in legacy_speech_ids and not event.get(
                "spoken_text"
            ):
                event["spoken_text"] = _extract_legacy_spoken_text(
                    str(event.get("description", ""))
                )
            events.append(event)
        data["sound_events"] = events
        return data

    @model_validator(mode="after")
    def check_references(self) -> Self:
        source_ids = {source.sound_source_id for source in self.sound_sources}
        for track in self.sound_tracks:
            if (
                track.sound_source_id is not None
                and track.sound_source_id not in source_ids
            ):
                raise ValueError(f"Unknown sound_source_id: {track.sound_source_id}")

        track_by_id = {track.sound_track_id: track for track in self.sound_tracks}
        for event in self.sound_events:
            track = track_by_id.get(event.sound_track_id)
            if track is None:
                raise ValueError(f"Unknown sound_track_id: {event.sound_track_id}")
            if track.track_type == "speech" and event.spoken_text is None:
                raise ValueError("speech events require spoken_text")
            if track.track_type != "speech" and event.spoken_text is not None:
                raise ValueError("spoken_text is only valid for speech events")
        return self


def _extract_legacy_spoken_text(description: str) -> str:
    match = re.search(r"""["“]([^"”]+)["”]|['‘]([^'’]+)['’]""", description)
    if match is None:
        return description
    return (match.group(1) or match.group(2)).strip()
