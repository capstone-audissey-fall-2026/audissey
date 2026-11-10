from functools import cache
from importlib import import_module
from pathlib import Path
from threading import Lock

import numpy as np

from audyssey.contracts import SourcePlan
from audyssey.media import read_wav, write_wav
from audyssey.settings import Settings

# shortcut: one GPU model per process, one call at a time; batch or queue if throughput matters
_model_lock = Lock()


@cache
def _load():
    settings = Settings()
    try:
        torch = import_module("torch")
        sam = import_module("sam_audio")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "SAM Audio is not installed: install the gpu/ environment"
        ) from exc
    # Rankers and the span predictor are ~4.4B of small's 5.8B params and are built in
    # SAMAudio.__init__ (the stock Judge ranker loads facebook/sam-audio-judge). Skip or
    # rebuild them via config overrides; they are not in the checkpoint, so loading
    # stays strict.
    overrides: dict[str, object] = {}
    if settings.sam_audio_reranking_candidates == 1:
        overrides.update(visual_ranker=None, text_ranker=None)
    else:
        # Same ensemble as the released configs, but the Judge ranker uses our
        # configured Judge (official or mirror) instead of the hardcoded official repo.
        clap = settings.sam_audio_clap_checkpoint or None
        imagebind = settings.sam_audio_imagebind_checkpoint
        overrides["text_ranker"] = {
            "kind": "ensemble",
            "rankers": {
                "clap": [{"kind": "clap", "checkpoint": clap}, 5.0],
                "judge": [
                    {
                        "kind": "judge",
                        "checkpoint_or_model_id": settings.sam_audio_judge_model,
                    },
                    1.0,
                ],
            },
        }
        overrides["visual_ranker"] = (
            {"kind": "imagebind", "checkpoint": imagebind} if imagebind else None
        )
    if not settings.sam_audio_predict_spans:
        overrides["span_predictor"] = None
    model = sam.SAMAudio.from_pretrained(settings.sam_audio_model, **overrides)
    model = model.to(settings.sam_audio_device).eval()
    processor = sam.SAMAudioProcessor.from_pretrained(settings.sam_audio_model)
    return torch, model, processor


def separate(
    mono: Path, plan: SourcePlan, video: Path | None, out: Path, attempt: int
) -> Path:
    settings = Settings()
    audio = read_wav(mono)
    anchors = (
        [[["+", span.start, span.end] for span in plan.spans]] if plan.spans else None
    )
    with _model_lock:
        torch, model, processor = _load()
        batch = processor(
            descriptions=[plan.description],
            audios=[torch.from_numpy(audio.T)],
            anchors=anchors,
            **({"masked_videos": [str(video)]} if video is not None else {}),
        ).to(settings.sam_audio_device)
        with torch.inference_mode():
            torch.manual_seed(attempt)
            result = model.separate(
                batch,
                reranking_candidates=settings.sam_audio_reranking_candidates,
                predict_spans=settings.sam_audio_predict_spans,
            )
        target = result.target[0].detach().cpu().float().numpy().reshape(-1, 1)
    track = np.zeros((len(audio), 1), dtype=np.float32)
    length = min(len(audio), len(target))
    track[:length] = target[:length]
    return write_wav(out, track)
