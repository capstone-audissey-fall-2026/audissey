from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from v2a_inspect.audio_generation import mix as mix_module
from v2a_inspect.models.audio_plan import AudioPlan, AudioPlanItem


class _FakeClip:
    duration = 1.0
    fps = 24.0
    audio = None

    def subclipped(self, *_args):
        return self

    def with_duration(self, *_args):
        return self

    def with_start(self, *_args):
        return self

    def with_effects(self, *_args):
        return self

    def with_audio(self, audio):
        self.audio = audio
        return self

    def write_videofile(self, filename, **kwargs):
        self.write_args = (filename, kwargs)

    def close(self):
        return None


class AudioMixTest(unittest.TestCase):
    def test_moviepy_temp_audio_uses_writable_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            wav_path = root / "event.wav"
            wav_path.touch()
            output_path = root / "output" / "preview.mp4"
            video = _FakeClip()
            plan = AudioPlan(
                items=[
                    AudioPlanItem(
                        item_id="event",
                        type="sfx",
                        time=(0.0, 1.0),
                        description="impact",
                    )
                ],
                total_duration=1.0,
            )

            with (
                patch.object(mix_module, "VideoFileClip", return_value=video),
                patch.object(mix_module, "AudioFileClip", return_value=_FakeClip()),
                patch.object(mix_module, "CompositeAudioClip", return_value=_FakeClip()),
            ):
                result = mix_module.mix_audio_into_video(
                    "input.mp4",
                    plan,
                    {"event": str(wav_path)},
                    str(output_path),
                )

            self.assertEqual(result, str(output_path))
            self.assertEqual(
                video.write_args[1]["temp_audiofile_path"],
                str(output_path.parent),
            )


if __name__ == "__main__":
    unittest.main()
