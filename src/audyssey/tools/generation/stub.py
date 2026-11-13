from pathlib import Path

import numpy as np

from audyssey.contracts import FilterResult, SourcePlan
from audyssey.media import probe, read_wav, write_wav
from audyssey.settings import Settings


def generate(video: Path, plan: SourcePlan, out: Path, seed: int) -> Path:
    duration, _ = probe(video)
    track = np.zeros(round(duration * 48000), dtype=np.float32)
    length = min(len(track), 48000)
    start = (len(track) - length) // 2
    frequency = 220 + 55 * (sum(map(ord, plan.description)) % 8)
    # Only the stub planner's first non-dialogue source exercises the retry loop.
    amplitude = (
        0.1 if plan.name == "sword_clash" and seed == Settings().gen_seed else 0.2
    )
    track[start : start + length] = amplitude * np.sin(
        2 * np.pi * frequency * np.arange(length) / 48000
    )
    return write_wav(out, track)


def score_generation(
    track: Path,
    plan: SourcePlan,
    siblings: list[SourcePlan],
) -> FilterResult:
    peak = float(np.max(np.abs(read_wav(track)), initial=0))
    passed = peak >= 0.15
    scores = {"text_correct": float(passed)}
    if siblings:
        scores["text_margin"] = peak
    return FilterResult(
        passed=passed, scores=scores, reason="" if passed else "stub: first seed"
    )
