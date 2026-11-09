"""Measure cached SAM Audio and Judge on GPU 0 without downloading weights."""

import argparse
from importlib import import_module
import json
import os
from pathlib import Path
from time import perf_counter

from audyssey.contracts import SourcePlan, Span
from audyssey.media import extract_audio
from audyssey.tools.filtering import real as filtering
from audyssey.tools.separation import real as separation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--description", required=True)
    parser.add_argument("--span", nargs=2, type=float, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if (
        os.environ.get("HF_HUB_OFFLINE") != "1"
        or os.environ.get("CUDA_VISIBLE_DEVICES") != "0"
    ):
        parser.error("set HF_HUB_OFFLINE=1 and CUDA_VISIBLE_DEVICES=0")
    plan = SourcePlan(
        name="smoke",
        kind="sfx",
        description=args.description,
        spans=[Span(start=start, end=end) for start, end in args.span],
    )
    mono = extract_audio(args.video, args.out / "mono.wav", mono=True)
    track = args.out / "track.wav"
    torch = import_module("torch")
    torch.cuda.reset_peak_memory_stats()
    timings = {}
    for name, call in [
        ("separation_load", separation._load),
        ("separation", lambda: separation.separate(mono, plan, args.video, track, 1)),
        ("judge_load", filtering._load),
    ]:
        torch.cuda.synchronize()
        start = perf_counter()
        call()
        torch.cuda.synchronize()
        timings[f"{name}_s"] = perf_counter() - start
    start = perf_counter()
    result = filtering.judge(mono, track, plan)
    torch.cuda.synchronize()
    timings["judge_s"] = perf_counter() - start
    print(
        json.dumps(
            {
                **timings,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
                "judge": result.model_dump(),
                "track": str(track),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
