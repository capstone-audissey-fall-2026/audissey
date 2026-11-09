import subprocess
from pathlib import Path

import numpy as np

from audyssey.media import MediaError, read_wav, write_wav


def compute_residual(audio: Path, tracks: list[Path], out: Path) -> Path:
    """Subtract every available track, regardless of the source's keep flag."""
    residual = read_wav(audio).astype(np.float64)
    for track in tracks:
        residual -= read_wav(track)
    return write_wav(out, residual)


def shift_pitch(src: Path, out: Path, semitones: float) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    ratio = 2 ** (semitones / 12)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                str(src),
                "-af",
                f"asetrate=48000*{ratio},aresample=48000,atempo={1 / ratio}",
                "-c:a",
                "pcm_f32le",
                str(out),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise MediaError(exc.stderr) from exc
    original = read_wav(src)
    shifted = read_wav(out)
    result = np.zeros_like(original)
    length = min(len(original), len(shifted))
    result[:length] = shifted[:length]
    return write_wav(out, result)
