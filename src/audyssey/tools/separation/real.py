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
    model = sam.SAMAudio.from_pretrained(settings.sam_audio_model)
    # Rankers and the span predictor are ~4.4B of small's 5.8B params; SAMAudio
    # skips them when None/absent, so keep them off the GPU unless configured.
    if settings.sam_audio_reranking_candidates == 1:
        model.visual_ranker = model.text_ranker = None
    if not settings.sam_audio_predict_spans and hasattr(model, "span_predictor"):
        del model.span_predictor
    model = model.to(settings.sam_audio_device).eval()
    processor = sam.SAMAudioProcessor.from_pretrained(settings.sam_audio_model)
    return torch, model, processor


def separate(
    mono: Path, plan: SourcePlan, video: Path, out: Path, attempt: int
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
