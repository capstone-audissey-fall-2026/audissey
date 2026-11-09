from pathlib import Path

import numpy as np

from audyssey.contracts import SourcePlan
from audyssey.media import read_wav, write_wav


def separate(
    mono: Path, plan: SourcePlan, video: Path, out: Path, attempt: int
) -> Path:
    audio = read_wav(mono)
    track = np.zeros_like(audio)
    for span in plan.spans:
        start, end = round(span.start * 48000), round(span.end * 48000)
        track[start:end] = audio[start:end]
    return write_wav(out, track)
