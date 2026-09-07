from __future__ import annotations

import unittest

from v2a_inspect.audio_generation.plan import build_audio_plan
from v2a_inspect.models import SoundEvent, SoundSource, SoundTimeline, SoundTrack


class AudioPlanTest(unittest.TestCase):
    def test_builds_sorted_items_with_track_routing_and_speech(self) -> None:
        source = SoundSource(source_type="visual_object", label="Hero")
        speech = SoundTrack(
            track_type="speech",
            label="Hero voice",
            sound_source_id=source.sound_source_id,
        )
        impact = SoundTrack(
            track_type="sfx",
            label="Sword impact",
            generation_model="v2a",
        )
        timeline = SoundTimeline(
            sound_sources=[source],
            sound_tracks=[speech, impact],
            sound_events=[
                SoundEvent(
                    sound_track_id=impact.sound_track_id,
                    start_frame_index=30,
                    end_frame_index=45,
                    description="A visible sword strike",
                ),
                SoundEvent(
                    sound_track_id=speech.sound_track_id,
                    start_frame_index=0,
                    end_frame_index=30,
                    description="A tense close-mic voice",
                    spoken_text="Stop.",
                ),
            ],
        )

        plan = build_audio_plan(timeline, fps=30, total_duration=2.0)

        self.assertEqual(
            [(item.type, item.generation_model) for item in plan.items],
            [("speech", "tts"), ("sfx", "v2a")],
        )
        self.assertEqual(plan.items[0].spoken_text, "Stop.")
        self.assertEqual(
            plan.items[0].description,
            "[Hero voice] A tense close-mic voice",
        )
        self.assertEqual(plan.items[1].time, (1.0, 1.5))


if __name__ == "__main__":
    unittest.main()
