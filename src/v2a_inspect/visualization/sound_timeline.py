from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from PIL import Image, ImageDraw, ImageFont

from v2a_inspect.models import SoundEvent, SoundTrack, VideoAsset

from .colors import Color

_TRACK_COLORS: dict[str, Color] = {
    "speech": (35, 120, 220),
    "sfx": (230, 85, 40),
    "music": (145, 70, 200),
    "ambience": (50, 160, 80),
}
_TRACK_ORDER: dict[str, int] = {
    "speech": 0,
    "sfx": 1,
    "music": 2,
    "ambience": 3,
}
_SCENE_BOUNDARY_COLOR: Color = (225, 225, 225)
_TEXT_COLOR: Color = (25, 25, 25)
_MUTED_TEXT_COLOR: Color = (95, 95, 95)

SoundTimelineSummaryValue = int | float | str | None
TimelineFont = ImageFont.ImageFont | ImageFont.FreeTypeFont


@dataclass
class _SoundTrackGroup:
    track: SoundTrack
    events: list[SoundEvent]


def render_sound_timeline(
    video_asset: VideoAsset,
    *,
    width: int = 1400,
    row_height: int = 30,
) -> Image.Image:
    """Render the final SoundTimeline as SoundTrack rows."""

    timeline = video_asset.sound_timeline
    if timeline is None:
        raise ValueError("video_asset must have a sound_timeline")

    sorted_events = _sorted_events(timeline.sound_events, timeline.sound_tracks)
    track_groups = _group_sound_events(sorted_events, timeline.sound_tracks)

    label_width = 270
    right_padding = 24
    top_padding = 72
    bottom_padding = 42
    timeline_left = label_width
    timeline_right = width - right_padding
    usable_width = timeline_right - timeline_left
    height = max(120, top_padding + (len(track_groups) * row_height) + bottom_padding)

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    draw.text(
        (12, 12),
        (
            f"{len(sorted_events)} sound events, "
            f"{len(timeline.sound_sources)} sources, "
            f"{len(timeline.sound_tracks)} tracks, "
            f"{video_asset.frame_count} frames, "
            f"{video_asset.duration_sec:.1f}s"
        ),
        fill=_TEXT_COLOR,
        font=font,
    )

    total_frames = max(1, video_asset.frame_count)
    _draw_frame_ticks(
        draw, total_frames, timeline_left, usable_width, top_padding, height
    )
    _draw_scene_boundaries(
        draw, video_asset, timeline_left, usable_width, top_padding, height
    )
    _draw_legend(draw, timeline_left, 38, font)

    for row_index, group in enumerate(track_groups):
        row_y = top_padding + (row_index * row_height)
        color = _TRACK_COLORS[group.track.track_type]
        draw.text(
            (12, row_y + 7),
            _short_label(_track_row_label(group.track), max_length=42),
            fill=_TEXT_COLOR,
            font=font,
        )
        draw.line(
            (
                timeline_left,
                row_y + row_height - 3,
                timeline_right,
                row_y + row_height - 3,
            ),
            fill=(240, 240, 240),
        )
        for event in group.events:
            _draw_event_bar(
                draw,
                event,
                total_frames,
                timeline_left,
                usable_width,
                row_y,
                color,
                font,
            )

    if timeline.notes is not None:
        draw.text((12, height - 24), timeline.notes[:180], fill=_MUTED_TEXT_COLOR)

    return image


def summarize_sound_timeline(
    video_asset: VideoAsset,
) -> list[dict[str, SoundTimelineSummaryValue]]:
    timeline = video_asset.sound_timeline
    if timeline is None:
        raise ValueError("video_asset must have a sound_timeline")

    source_by_id = {
        sound_source.sound_source_id: sound_source
        for sound_source in timeline.sound_sources
    }
    track_by_id = {track.sound_track_id: track for track in timeline.sound_tracks}
    rows: list[dict[str, SoundTimelineSummaryValue]] = []
    for event in _sorted_events(timeline.sound_events, timeline.sound_tracks):
        track = track_by_id[event.sound_track_id]
        source = (
            None
            if track.sound_source_id is None
            else source_by_id.get(track.sound_source_id)
        )
        frame_count = event.end_frame_index - event.start_frame_index
        rows.append(
            {
                "sound_event_id": str(event.sound_event_id),
                "sound_track_id": str(event.sound_track_id),
                "track_label": track.label,
                "track_type": track.track_type,
                "start_frame_index": event.start_frame_index,
                "end_frame_index": event.end_frame_index,
                "frame_count": frame_count,
                "duration_sec": frame_count / video_asset.fps,
                "description": event.description,
                "spoken_text": event.spoken_text,
                "sound_source_id": None
                if track.sound_source_id is None
                else str(track.sound_source_id),
                "source_label": None if source is None else source.label,
                "generation_model": track.generation_model,
            }
        )
    return rows


def _group_sound_events(
    events: list[SoundEvent],
    tracks: list[SoundTrack],
) -> list[_SoundTrackGroup]:
    track_by_id = {track.sound_track_id: track for track in tracks}
    events_by_track_id: dict[UUID, list[SoundEvent]] = {
        track.sound_track_id: [] for track in tracks
    }
    for event in events:
        events_by_track_id[event.sound_track_id].append(event)
    groups = [
        _SoundTrackGroup(track=track_by_id[track_id], events=track_events)
        for track_id, track_events in events_by_track_id.items()
        if track_events
    ]
    return sorted(groups, key=_track_group_sort_key)


def _track_group_sort_key(group: _SoundTrackGroup) -> tuple[int, str, int, str]:
    return (
        _TRACK_ORDER[group.track.track_type],
        group.track.label.lower(),
        min(event.start_frame_index for event in group.events),
        str(group.track.sound_track_id),
    )


def _sorted_events(
    events: list[SoundEvent],
    tracks: list[SoundTrack],
) -> list[SoundEvent]:
    track_by_id = {track.sound_track_id: track for track in tracks}
    return sorted(
        events,
        key=lambda event: (
            _TRACK_ORDER[track_by_id[event.sound_track_id].track_type],
            track_by_id[event.sound_track_id].label.lower(),
            event.start_frame_index,
            event.end_frame_index,
            event.description,
        ),
    )


def _track_row_label(track: SoundTrack) -> str:
    return f"{track.track_type}: {track.label}"


def _draw_event_bar(
    draw: ImageDraw.ImageDraw,
    event: SoundEvent,
    total_frames: int,
    timeline_left: int,
    usable_width: int,
    row_y: int,
    color: Color,
    font: TimelineFont,
) -> None:
    x1 = _frame_to_x(event.start_frame_index, total_frames, timeline_left, usable_width)
    x2 = _frame_to_x(event.end_frame_index, total_frames, timeline_left, usable_width)
    draw.rectangle((x1, row_y + 6, max(x1 + 1, x2), row_y + 23), fill=color)
    label = _short_label(event.description)
    text_fill = (255, 255, 255)
    draw.text((x1 + 4, row_y + 9), label, fill=text_fill, font=font)


def _short_label(description: str, *, max_length: int = 64) -> str:
    if len(description) <= max_length:
        return description
    return f"{description[: max_length - 3]}..."


def _draw_frame_ticks(
    draw: ImageDraw.ImageDraw,
    total_frames: int,
    timeline_left: int,
    usable_width: int,
    top_padding: int,
    height: int,
) -> None:
    tick_count = 10
    for tick_index in range(tick_count + 1):
        frame_index = round(total_frames * tick_index / tick_count)
        x = _frame_to_x(frame_index, total_frames, timeline_left, usable_width)
        draw.line((x, top_padding - 8, x, height - 28), fill=(238, 238, 238))
        draw.text((x - 10, height - 24), str(frame_index), fill=_MUTED_TEXT_COLOR)


def _draw_scene_boundaries(
    draw: ImageDraw.ImageDraw,
    video_asset: VideoAsset,
    timeline_left: int,
    usable_width: int,
    top_padding: int,
    height: int,
) -> None:
    total_frames = max(1, video_asset.frame_count)
    for scene_index, scene in enumerate(video_asset.initial_scenes):
        x = _frame_to_x(
            scene.start_frame_index, total_frames, timeline_left, usable_width
        )
        draw.line((x, top_padding - 24, x, height - 28), fill=_SCENE_BOUNDARY_COLOR)
        if scene_index % 2 == 0:
            draw.text(
                (x + 3, top_padding - 36), f"s{scene_index}", fill=_MUTED_TEXT_COLOR
            )


def _draw_legend(
    draw: ImageDraw.ImageDraw,
    left: int,
    top: int,
    font: TimelineFont,
) -> None:
    x = left
    for track_type in _track_types():
        color = _TRACK_COLORS[track_type]
        draw.rectangle((x, top + 2, x + 12, top + 14), fill=color)
        draw.text((x + 16, top), track_type, fill=_TEXT_COLOR, font=font)
        x += 92


def _track_types() -> tuple[Literal["speech", "sfx", "music", "ambience"], ...]:
    return ("speech", "sfx", "music", "ambience")


def _frame_to_x(
    frame_index: int,
    total_frames: int,
    timeline_left: int,
    usable_width: int,
) -> int:
    clamped_frame_index = min(max(0, frame_index), total_frames)
    return timeline_left + round((clamped_frame_index / total_frames) * usable_width)
