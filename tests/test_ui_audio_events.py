from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from v2a_inspect.models import (
    SoundEvent,
    SoundEventAudioArtifact,
    SoundTimeline,
    SoundTrack,
    VideoAsset,
)
from v2a_inspect.ui.pipeline import (
    run_event_audio_delete_pipeline,
    run_event_audio_regeneration_pipeline,
)
from v2a_inspect.ui.store import VideoAssetStore


class UiAudioEventTest(unittest.TestCase):
    def test_delete_event_audio_removes_artifact(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                track = SoundTrack(track_type="sfx", label="Door")
                event = SoundEvent(
                    sound_track_id=track.sound_track_id,
                    start_frame_index=0,
                    end_frame_index=30,
                    description="Door slam",
                )
                wav_path = root / "event.wav"
                wav_path.write_bytes(b"wav")
                artifact = SoundEventAudioArtifact(
                    sound_event_id=event.sound_event_id,
                    sound_track_id=track.sound_track_id,
                    path=wav_path,
                    duration_sec=1.0,
                    generation_model="t2a",
                    description="Door slam",
                )
                asset = VideoAsset(
                    source_path=root / "video.mp4",
                    frame_count=30,
                    sound_timeline=SoundTimeline(
                        sound_tracks=[track],
                        sound_events=[event],
                    ),
                    sound_event_audio_artifacts=[artifact],
                )
                store = VideoAssetStore()
                rebuilt = asset.model_copy(update={"sound_event_audio_artifacts": []})

                with patch(
                    "v2a_inspect.ui.pipeline._rebuild_audio_outputs",
                    new=AsyncMock(return_value=rebuilt),
                ) as rebuild:
                    await run_event_audio_delete_pipeline(
                        asset, store, str(event.sound_event_id)
                    )

                snapshot = await store.snapshot()
                self.assertEqual(snapshot.status, "complete")
                self.assertEqual(snapshot.asset.sound_event_audio_artifacts, [])
                self.assertFalse(wav_path.exists())
                rebuild.assert_awaited_once()

        asyncio.run(run())

    def test_regenerate_event_audio_updates_prompt(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                track = SoundTrack(track_type="speech", label="Narrator")
                event = SoundEvent(
                    sound_track_id=track.sound_track_id,
                    start_frame_index=0,
                    end_frame_index=30,
                    description="Old voice direction",
                    spoken_text="Old line",
                )
                asset = VideoAsset(
                    source_path=root / "video.mp4",
                    frame_count=30,
                    sound_timeline=SoundTimeline(
                        sound_tracks=[track],
                        sound_events=[event],
                    ),
                )
                new_artifact = SoundEventAudioArtifact(
                    sound_event_id=event.sound_event_id,
                    sound_track_id=track.sound_track_id,
                    path=root / "new.wav",
                    duration_sec=1.0,
                    generation_model="tts",
                    description="New voice direction",
                )

                async def rebuild(video_asset, _audio_plan, artifacts, _store):
                    return video_asset.model_copy(
                        update={"sound_event_audio_artifacts": artifacts}
                    )

                store = VideoAssetStore()
                with (
                    patch(
                        "v2a_inspect.ui.pipeline._generate_event_audio_artifact",
                        new=AsyncMock(return_value=new_artifact),
                    ),
                    patch(
                        "v2a_inspect.ui.pipeline._rebuild_audio_outputs",
                        new=AsyncMock(side_effect=rebuild),
                    ),
                ):
                    await run_event_audio_regeneration_pipeline(
                        asset,
                        store,
                        str(event.sound_event_id),
                        "New voice direction",
                        "New line",
                        None,
                    )

                snapshot = await store.snapshot()
                next_event = snapshot.asset.sound_timeline.sound_events[0]
                self.assertEqual(snapshot.status, "complete")
                self.assertEqual(next_event.description, "New voice direction")
                self.assertEqual(next_event.spoken_text, "New line")
                self.assertEqual(
                    snapshot.asset.sound_event_audio_artifacts[0].sound_event_id,
                    event.sound_event_id,
                )

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
