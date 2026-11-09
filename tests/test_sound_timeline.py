from __future__ import annotations

import unittest
from uuid import uuid4

from pydantic import ValidationError

from v2a_inspect.models import SoundEvent, SoundTimeline, SoundTrack
from v2a_inspect.tools.sound_timeline.schemas import UpsertSoundTrackArgs


class SoundTimelineTest(unittest.TestCase):
    def test_music_track_type_is_rejected(self) -> None:
        for model in (SoundTrack, UpsertSoundTrackArgs):
            with self.subTest(model=model.__name__), self.assertRaises(ValidationError):
                model.model_validate({"track_type": "music", "label": "Score"})

    def test_legacy_dialogue_is_migrated(self) -> None:
        speech_id = uuid4()
        sfx_id = uuid4()
        timeline = SoundTimeline.model_validate(
            {
                "sound_tracks": [
                    {
                        "sound_track_id": str(speech_id),
                        "track_type": "dialogue",
                        "label": "Narrator",
                        "generation_mode": "tta",
                    },
                    {
                        "sound_track_id": str(sfx_id),
                        "track_type": "sfx",
                        "label": "Distant dog",
                        "generation_mode": "unknown",
                    },
                ],
                "sound_events": [
                    {
                        "sound_track_id": str(speech_id),
                        "start_frame_index": 0,
                        "end_frame_index": 30,
                        "description": 'A calm voice says "Hello there".',
                    }
                ],
            }
        )

        self.assertEqual(timeline.sound_tracks[0].track_type, "speech")
        self.assertEqual(timeline.sound_tracks[0].generation_model, "tts")
        self.assertEqual(timeline.sound_tracks[1].generation_model, "t2a")
        self.assertEqual(
            timeline.sound_events[0].description,
            'A calm voice says "Hello there".',
        )
        self.assertEqual(timeline.sound_events[0].spoken_text, "Hello there")

    def test_new_speech_requires_spoken_text(self) -> None:
        track = SoundTrack(track_type="speech", label="Narrator")
        event = SoundEvent(
            sound_track_id=track.sound_track_id,
            start_frame_index=0,
            end_frame_index=30,
            description="A calm voice",
        )
        with self.assertRaises(ValidationError):
            SoundTimeline(sound_tracks=[track], sound_events=[event])

    def test_non_speech_rejects_spoken_text(self) -> None:
        track = SoundTrack(track_type="sfx", label="Door")
        event = SoundEvent(
            sound_track_id=track.sound_track_id,
            start_frame_index=0,
            end_frame_index=3,
            description="A wooden door slam",
            spoken_text="slam",
        )
        with self.assertRaises(ValidationError):
            SoundTimeline(sound_tracks=[track], sound_events=[event])


if __name__ == "__main__":
    unittest.main()
