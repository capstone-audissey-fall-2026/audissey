from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from v2a_inspect.models import (
    InitialScene,
    Keyframe,
    SoundEvent,
    SoundTimeline,
    SoundTrack,
    VideoAsset,
)
from v2a_inspect.ui.routes import _delete_asset_work_dir, _keyframe_response_path
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

    def test_keyframe_response_path_stays_inside_asset_work_dir(self) -> None:
        with (
            tempfile.TemporaryDirectory() as root_dir,
            tempfile.TemporaryDirectory() as outside_dir,
        ):
            root = Path(root_dir)
            source = root / "video.mp4"
            source.write_bytes(b"video")
            image = root / "keyframes" / "frame.jpg"
            image.parent.mkdir()
            image.write_bytes(b"image")
            outside = Path(outside_dir) / "frame.jpg"
            outside.write_bytes(b"image")

            keyframe = Keyframe(frame_index=10, image_path=image)
            outside_keyframe = Keyframe(frame_index=20, image_path=outside)
            asset = VideoAsset(
                source_path=source,
                frame_count=30,
                initial_scenes=[
                    InitialScene(
                        start_frame_index=0,
                        end_frame_index=30,
                        keyframes=[keyframe, outside_keyframe],
                    )
                ],
            )

            self.assertEqual(
                _keyframe_response_path(asset, str(keyframe.keyframe_id)),
                image.resolve(),
            )
            self.assertIsNone(
                _keyframe_response_path(asset, str(outside_keyframe.keyframe_id))
            )

    def test_clean_deletes_only_app_work_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "v2a-inspect-ui"
            source = root / "video.mp4"
            source.parent.mkdir()
            source.write_bytes(b"video")
            (root / "record.json").write_text("{}", encoding="utf-8")

            cleaned = _delete_asset_work_dir(
                VideoAsset(source_path=source, frame_count=30)
            )

            self.assertEqual(cleaned, root.resolve())
            self.assertFalse(root.exists())

    def test_clean_ignores_non_app_work_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "video.mp4"
            source.write_bytes(b"video")

            cleaned = _delete_asset_work_dir(
                VideoAsset(source_path=source, frame_count=30)
            )

            self.assertIsNone(cleaned)
            self.assertTrue(source.exists())


if __name__ == "__main__":
    unittest.main()
