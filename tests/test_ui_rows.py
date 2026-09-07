from __future__ import annotations

import unittest
from pathlib import Path

from v2a_inspect.models import SoundEvent, SoundTimeline, SoundTrack, VideoAsset
from v2a_inspect.ui.rows import timeline_rows


class TimelineRowsTest(unittest.TestCase):
    def test_speech_row_preserves_text_and_tts_model(self) -> None:
        track = SoundTrack(track_type="speech", label="Narrator")
        event = SoundEvent(
            sound_track_id=track.sound_track_id,
            start_frame_index=0,
            end_frame_index=30,
            description="A calm narration",
            spoken_text="Once upon a time.",
        )
        asset = VideoAsset(
            source_path=Path("video.mp4"),
            frame_count=30,
            sound_timeline=SoundTimeline(
                sound_tracks=[track],
                sound_events=[event],
            ),
        )

        row = timeline_rows(asset)[0]

        self.assertEqual(row["generation_model"], "tts")
        self.assertEqual(row["spoken_text"], "Once upon a time.")


if __name__ == "__main__":
    unittest.main()
