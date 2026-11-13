import subprocess
from pathlib import Path

import numpy as np

from audyssey.contracts import Job, RenderOutput
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


def render(job: Job, job_dir: Path, include_residual_stem: bool) -> RenderOutput:
    out = job_dir / "out"
    stems_dir = out / "stems"
    stems_dir.mkdir(parents=True, exist_ok=True)
    mix = np.zeros_like(read_wav(job_dir / job.audio_path), dtype=np.float64)
    stems: list[str] = []
    for source in job.sources:
        if not source.keep or source.track_path is None:
            continue
        path = job_dir / source.track_path
        if source.pitch_semitones:
            path = shift_pitch(
                path, out / f"pitch_{source.id}.wav", source.pitch_semitones
            )
        data = read_wav(path) * source.gain
        stem = write_wav(stems_dir / f"{source.id}.wav", data)
        stems.append(str(stem.relative_to(job_dir)))
        if source.origin == "generated":
            # Original codecs may decode padding beyond the video's sample count.
            length = min(len(mix), len(data))
            mix[:length] += data[:length]
        else:
            mix += data
    if job.residual.keep and job.residual.path:
        data = read_wav(job_dir / job.residual.path) * job.residual.gain
        mix += data
        if include_residual_stem:
            stem = write_wav(stems_dir / "residual.wav", data)
            stems.append(str(stem.relative_to(job_dir)))
    write_wav(out / "mix.wav", mix)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                str(job_dir / job.video_path),
                "-i",
                str(out / "mix.wav"),
                "-map",
                "0:v",
                "-map",
                "1:a",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                str(out / "video.mp4"),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise MediaError(exc.stderr) from exc
    return RenderOutput(mix_wav="out/mix.wav", video_mp4="out/video.mp4", stems=stems)
