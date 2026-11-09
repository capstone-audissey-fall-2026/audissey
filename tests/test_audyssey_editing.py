import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np
from tests.audyssey_media import make_video
from audyssey.contracts import Job, Source, SourcePlan, Residual
from audyssey.media import extract_audio, read_wav, write_wav
from audyssey.tools.editing import render as editing


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg required")
class EditingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        make_video(self.root / "input.mp4")
        extract_audio(self.root / "input.mp4", self.root / "audio.wav", False)
        self.audio = read_wav(self.root / "audio.wav")
        mono = self.audio.mean(axis=1, keepdims=True)
        self.tracks = []
        for i, (start, end) in enumerate([(0, 48000), (60000, 120000)], 1):
            data = np.zeros_like(mono)
            data[start:end] = mono[start:end]
            self.tracks.append(write_wav(self.root / f"tracks/s{i}.wav", data))
        self.job = Job(
            id="test",
            video_path="input.mp4",
            audio_path="audio.wav",
            sources=[
                Source(
                    id=f"s{i}",
                    plan=SourcePlan(name=f"s{i}", kind="sfx", description="sound"),
                    track_path=f"tracks/s{i}.wav",
                )
                for i in (1, 2)
            ],
            residual=Residual(path="residual.wav"),
        )

    def residual(self):
        return editing.compute_residual(
            self.root / "audio.wav", self.tracks, self.root / "residual.wav"
        )

    def test_residual_ignores_keep_flag(self):
        expected = self.audio - read_wav(self.tracks[0]) - read_wav(self.tracks[1])
        self.job.sources[0].keep = False
        np.testing.assert_allclose(read_wav(self.residual()), expected, atol=1e-7)

    def test_shift_pitch_preserves_length(self):
        for pitch in [5, -7]:
            out = editing.shift_pitch(self.tracks[0], self.root / "pitch.wav", pitch)
            self.assertEqual(read_wav(out).shape, read_wav(self.tracks[0]).shape)

