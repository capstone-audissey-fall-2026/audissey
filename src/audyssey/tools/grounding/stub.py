import subprocess
from pathlib import Path

from audyssey.contracts import GroundResult, VisualHint
from audyssey.media import MediaError


def locate(video: Path, hint: VisualHint) -> GroundResult:
    found = "ghost" not in hint.description
    return GroundResult(found=found, box=hint.box if found else None)


def mask_video(video: Path, hint: VisualHint, out: Path) -> Path:
    # shortcut: fixed box without tracking, use real grounding for moving objects
    x0, y0, x1, y1 = hint.box
    out.parent.mkdir(parents=True, exist_ok=True)
    expression = f"between(X,W*{x0},W*{x1}-1)*between(Y,H*{y0},H*{y1}-1)"
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                str(video),
                "-an",
                "-vf",
                f"format=gbrp,geq=r='r(X,Y)*({expression})':g='g(X,Y)*({expression})':b='b(X,Y)*({expression})'",
                "-c:v",
                "libx264rgb",
                "-crf",
                "0",
                "-vsync",
                "0",
                str(out),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise MediaError(exc.stderr) from exc
    return out
