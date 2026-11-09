import subprocess
from pathlib import Path


def make_video(
    path: Path, seconds: float = 3.0, channels: int = 2, audio: bool = True
) -> Path:
    args = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=160x120:rate=10",
    ]
    if audio:
        args += [
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000",
            "-ac",
            str(channels),
            "-c:a",
            "aac",
        ]
    args += ["-t", str(seconds), "-c:v", "mpeg4", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(args, check=True, capture_output=True)
    return path
