import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from tests.audyssey_media import make_video
from audyssey.media import (
    MediaError,
    probe,
    check_upload,
    extract_audio,
    read_wav,
    write_wav,
    compute_peaks,
)


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg required")
class MediaTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.video = make_video(self.root / "input.mp4")

    def test_probe_reads_duration_and_channels(self):
        duration, channels = probe(self.video)
        self.assertAlmostEqual(duration, 3, delta=0.1)
        self.assertEqual(channels, 2)

    def test_check_upload_rejects_no_audio(self):
        with self.assertRaisesRegex(MediaError, "audio stream"):
            check_upload(make_video(self.root / "silent.mp4", audio=False), 30)

    def test_check_upload_rejects_too_long(self):
        with self.assertRaisesRegex(MediaError, "maximum is 2s"):
            check_upload(self.video, 2)

    def test_extract_audio_mono_and_original(self):
        original = extract_audio(self.video, self.root / "audio.wav", False)
        mono = extract_audio(self.video, self.root / "mono.wav", True)
        self.assertEqual(read_wav(original).shape[1], 2)
        self.assertEqual(read_wav(mono).shape[1], 1)
        self.assertEqual(len(read_wav(original)), len(read_wav(mono)))
        self.assertEqual(wavfile.read(original)[0], 48000)
        self.assertEqual(wavfile.read(mono)[0], 48000)

    def test_wav_roundtrip_float32(self):
        data = np.random.default_rng(0).uniform(-1, 1, (100, 2)).astype(np.float32)
        result = read_wav(write_wav(self.root / "nested/a.wav", data))
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_equal(result, data)

    def test_peaks_range(self):
        peaks = compute_peaks(extract_audio(self.video, self.root / "audio.wav", False))
        self.assertEqual(len(peaks), 800)
        self.assertTrue(all(0 <= x <= 1 for x in peaks))
