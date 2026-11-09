import json
import subprocess
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

    def test_extract_frames_evenly_with_times(self):
        from audyssey.media import extract_frames

        for count, times in [(1, [1.5]), (4, [0.375, 1.125, 1.875, 2.625])]:
            frames = extract_frames(self.video, count, self.root / f"frames-{count}")
            self.assertEqual([time for time, _ in frames], times)
            for _, path in frames:
                self.assertEqual(path.read_bytes()[:2], b"\xff\xd8")
                info = json.loads(subprocess.check_output([
                    "ffprobe", "-v", "error", "-show_entries", "stream=width,height",
                    "-of", "json", str(path),
                ]))
                self.assertEqual(info["streams"][0]["width"], 160)

    def test_extract_frames_downscales_and_handles_short_video(self):
        from audyssey.media import extract_frames

        wide = self.root / "wide.mp4"
        subprocess.run([
            "ffmpeg", "-v", "error", "-i", str(self.video),
            "-vf", "scale=1024:768", str(wide),
        ], check=True, capture_output=True)
        frame = extract_frames(wide, 1, self.root / "wide-frames")[0][1]
        width = subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries", "stream=width",
            "-of", "csv=p=0", str(frame),
        ], text=True)
        self.assertEqual(int(width), 512)
        short = make_video(self.root / "short.mp4", seconds=0.1)
        frames = extract_frames(short, 16, self.root / "short-frames")
        self.assertEqual(len(frames), 16)
        self.assertTrue(all(path.is_file() for _, path in frames))
        with self.assertRaises(ValueError):
            extract_frames(self.video, 0, self.root / "invalid")
