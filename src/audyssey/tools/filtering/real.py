"""윤석 담당 모듈의 예시 구현. 기준과 구현은 교체 가능."""

from functools import cache
from importlib import import_module
from pathlib import Path
from threading import Lock

from audyssey.contracts import FilterResult, SourcePlan
from audyssey.media import read_wav
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
    model = (
        sam.SAMAudioJudgeModel.from_pretrained(settings.sam_audio_judge_model)
        .to(settings.sam_audio_device)
        .eval()
    )
    processor = sam.SAMAudioJudgeProcessor.from_pretrained(
        settings.sam_audio_judge_model
    )
    return torch, model, processor


def judge(mono: Path, track: Path, plan: SourcePlan) -> FilterResult:
    settings = Settings()
    with _model_lock:
        torch, model, processor = _load()
        batch = processor(
            text=[plan.description],
            input_audio=[torch.from_numpy(read_wav(mono).T)],
            separated_audio=[torch.from_numpy(read_wav(track).T)],
            sampling_rate=48000,
        ).to(settings.sam_audio_device)
        with torch.inference_mode():
            result = model(**batch)
        scores = {}
        for name in ("overall", "recall", "precision", "faithfulness"):
            value = getattr(result, name, None)
            if value is not None:
                scores[name] = float(value.item())
    overall = scores["overall"]
    threshold = settings.judge_pass_threshold
    passed = overall >= threshold
    return FilterResult(
        passed=passed,
        scores=scores,
        reason="" if passed else f"judge overall {overall:.2f} < {threshold}",
    )
