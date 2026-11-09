import json
import math
import subprocess
from pathlib import Path

import numpy as np
from scipy.io import wavfile


class MediaError(ValueError):
    pass


def probe(video: Path) -> tuple[float, int]:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration:stream=codec_type,channels",
                "-of",
                "json",
                str(video),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        info = json.loads(result.stdout)
        audio = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
        if audio is None:
            raise MediaError("video has no audio stream")
        if not any(s["codec_type"] == "video" for s in info["streams"]):
            raise MediaError("file has no video stream")
        duration = float(info["format"]["duration"])
        if not math.isfinite(duration) or duration <= 0:
            raise MediaError("video duration must be positive and finite")
        return duration, int(audio["channels"])
    except subprocess.CalledProcessError as exc:
        raise MediaError(exc.stderr) from exc
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise MediaError(str(exc)) from exc


def check_upload(video: Path, max_duration_s: float) -> float:
    duration, _ = probe(video)
    if duration > max_duration_s:
        raise MediaError(f"video is {duration:.1f}s; maximum is {max_duration_s:.0f}s")
    return duration


def extract_audio(video: Path, out: Path, mono: bool) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    args = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-i",
        str(video),
        "-vn",
        "-ar",
        "48000",
        "-c:a",
        "pcm_f32le",
    ]
    if mono:
        args += ["-ac", "1"]
    try:
        subprocess.run([*args, str(out)], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise MediaError(exc.stderr) from exc
    return out


def read_wav(path: Path) -> np.ndarray:
    rate, data = wavfile.read(path)
    if rate != 48000:
        raise MediaError("expected 48000Hz wav")
    if np.issubdtype(data.dtype, np.integer):
        if data.dtype == np.uint8:
            data = (data.astype(np.float32) - 128) / 128
        else:
            data = data.astype(np.float32) / (float(np.iinfo(data.dtype).max) + 1)
    return data.astype(np.float32).reshape(len(data), -1)


def write_wav(path: Path, data: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, 48000, data.astype(np.float32))
    return path


def compute_peaks(wav: Path, buckets: int = 800) -> list[float]:
    data = read_wav(wav).mean(axis=1)
    if not len(data) or buckets <= 0:
        return []
    return [
        float(np.clip(np.max(np.abs(chunk)), 0, 1))
        for chunk in np.array_split(data, min(buckets, len(data)))
    ]


def extract_frames(video: Path, count: int, out_dir: Path) -> list[tuple[float, Path]]:
    """Sample one JPEG at each equal interval's midpoint (frame-rate rounded)."""
    if count < 1:
        raise ValueError("frame count must be positive")
    duration, _ = probe(video)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                str(video),
                "-vf",
                f"fps={count / duration},scale='min(512,iw)':-1",
                "-frames:v",
                str(count),
                str(out_dir / "%02d.jpg"),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, OSError) as exc:
        message = (
            exc.stderr if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        )
        raise MediaError(message) from exc
    frames = [
        ((i + 0.5) * duration / count, out_dir / f"{i + 1:02d}.jpg")
        for i in range(count)
    ]
    if any(not path.is_file() for _, path in frames):
        raise MediaError("could not extract all planner frames")
    return frames
