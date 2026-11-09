from __future__ import annotations

from v2a_inspect.models import AudioPlan, AudioPlanItem, SoundTimeline


def build_audio_plan(
    timeline: SoundTimeline,
    *,
    fps: float,
    total_duration: float,
) -> AudioPlan:
    """Convert the editable sound timeline into generation-ready audio items."""
    if fps <= 0:
        raise ValueError("fps must be greater than zero")

    tracks = {track.sound_track_id: track for track in timeline.sound_tracks}
    sources = {source.sound_source_id: source for source in timeline.sound_sources}
    items: list[AudioPlanItem] = []

    for event in timeline.sound_events:
        track = tracks.get(event.sound_track_id)
        if track is None:
            continue

        start_time = max(
            0.0,
            min(event.start_frame_index / fps, total_duration - 0.1),
        )
        end_time = max(
            0.0,
            min(event.end_frame_index / fps, total_duration),
        )
        if end_time <= start_time:
            end_time = start_time + 0.1

        source = sources.get(track.sound_source_id)
        source_prefix = (
            f"{source.label}, "
            if source and source.label.lower() not in track.label.lower()
            else ""
        )
        model = track.generation_model
        items.append(
            AudioPlanItem(
                item_id=str(event.sound_event_id),
                type=track.track_type,
                time=(start_time, end_time),
                description=f"{source_prefix}[{track.label}] {event.description}",
                spoken_text=event.spoken_text,
                volume=1.5 if model == "v2a" else 0.8 if model == "t2a" else 1.0,
                track_id=str(track.sound_track_id),
                generation_model=model,
            )
        )

    items.sort(key=lambda item: item.time[0])
    return AudioPlan(items=items, total_duration=total_duration)
