import math
import os
import sys
from copy import deepcopy
from functools import cache
from importlib import import_module
from pathlib import Path
from threading import Lock

import numpy as np
from scipy.signal import resample_poly

from audyssey.contracts import FilterResult, SourcePlan
from audyssey.media import probe, read_wav, write_wav
from audyssey.settings import Settings

# shortcut: one GPU call per process, add a queue if throughput matters
_model_lock = Lock()


def windows(
    duration: float, single_max: float, window: float, overlap: float = 1.0
) -> list[tuple[float, float]]:
    if (
        not all(math.isfinite(x) for x in (duration, single_max, window, overlap))
        or duration <= 0
        or single_max <= 0
        or not 0 <= overlap < window
    ):
        raise ValueError("positive durations and 0 <= overlap < window are required")
    if duration <= single_max or duration <= window:
        return [(0, duration)]
    spans = [(0.0, window)]
    while spans[-1][1] < duration:
        start = spans[-1][0] + window - overlap
        if start >= duration - window:
            spans.append((round(duration - window, 9), duration))
            break
        spans.append((start, start + window))
    return spans


def crossfade_join(
    chunks: list[np.ndarray], spans: list[tuple[float, float]], sample_rate: int
) -> np.ndarray:
    result = np.zeros(round(spans[-1][1] * sample_rate), dtype=np.float32)
    previous_end = 0
    for chunk, (start, end) in zip(chunks, spans, strict=True):
        first, last = round(start * sample_rate), round(end * sample_rate)
        audio = np.zeros(last - first, dtype=np.float32)
        length = min(len(audio), len(chunk))
        audio[:length] = chunk[:length]
        overlap = max(0, previous_end - first)
        fade = np.linspace(0, 1, overlap, dtype=np.float32)
        result[first : first + overlap] *= 1 - fade
        audio[:overlap] *= fade
        result[first:last] += audio
        previous_end = last
    return result


def _require_offline() -> None:
    # SelVA's feature constructors also load CLIP, T5 and BigVGAN internally.
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise RuntimeError(
            "real generation requires HF_HUB_OFFLINE=1 and cached weights"
        )


@cache
def _load():
    _require_offline()
    settings = Settings()
    torch = import_module("torch")
    hub = import_module("huggingface_hub")
    evaluation = import_module("selva.utils.eval_utils")
    encoder_module = import_module("selva.model.networks_video_enc")
    generator_module = import_module("selva.model.networks_generator")
    features_module = import_module("selva.model.utils.features_utils")
    flow_module = import_module("selva.model.flow_matching")

    def checkpoint(repo: str, filename: str) -> str:
        return hub.hf_hub_download(
            repo_id=repo, filename=filename, local_files_only=True
        )

    base = checkpoint(settings.selva_base_model, "weights/video_enc_sup_5.pth")
    vae = checkpoint(settings.selva_base_model, "ext_weights/v1-44.pth")
    synchformer = checkpoint(
        settings.selva_base_model, "ext_weights/synchformer_state_dict.pth"
    )
    generator_path = checkpoint(
        settings.selva_ft_model, settings.selva_ft_generator_file
    )
    encoder_path = checkpoint(settings.selva_ft_model, settings.selva_ft_encoder_file)
    config = evaluation.all_model_cfg["large_44k"]
    encoder = (
        encoder_module.get_my_textsynch("depth1")
        .to(settings.selva_device, torch.bfloat16)
        .eval()
    )
    encoder.load_weights(torch.load(base, map_location="cpu", weights_only=True))
    encoder.load_state_dict(
        torch.load(encoder_path, map_location="cpu", weights_only=True)["state_dict"],
        strict=False,
    )
    generator = (
        generator_module.get_my_mmaudio(config.model_name)
        .to(settings.selva_device, torch.bfloat16)
        .eval()
    )
    generator.load_weights(
        torch.load(generator_path, map_location="cpu", weights_only=True)
    )
    features = (
        features_module.FeaturesUtils(
            tod_vae_ckpt=vae,
            synchformer_ckpt=synchformer,
            enable_conditions=True,
            mode=config.mode,
            bigvgan_vocoder_ckpt=None,
            need_vae_encoder=False,
        )
        .to(settings.selva_device, torch.bfloat16)
        .eval()
    )
    flow = flow_module.FlowMatching(
        min_sigma=0, inference_mode="euler", num_steps=settings.selva_steps
    )
    return (
        torch,
        evaluation,
        deepcopy(config.seq_cfg),
        encoder,
        generator,
        features,
        flow,
    )


def generate(video: Path, plan: SourcePlan, out: Path, seed: int) -> Path:
    settings = Settings()
    duration, _ = probe(video)
    spans = windows(duration, settings.selva_single_max_s, settings.selva_window_s)
    with _model_lock:
        torch, evaluation, seq, encoder, generator, features, flow = _load()
        rng = torch.Generator(device=settings.selva_device)
        rng.manual_seed(seed)
        frames = evaluation.load_video(
            video, duration, load_all_frames=False
        ).sync_frames
        chunks = []
        with torch.inference_mode():
            for start, end in spans:
                sync = frames[round(start * 25) : round(end * 25)]
                if not sync.shape[0]:
                    raise ValueError(
                        "SelVA decoded no frames for the generation window"
                    )
                if sync.shape[0] < 16:
                    sync = torch.cat(
                        [sync, sync[-1:].expand(16 - sync.shape[0], *sync.shape[1:])]
                    )
                # Avoid losing one frame when SequenceConfig multiplies duration by 25.
                seq.duration = math.nextafter(sync.shape[0] / 25, math.inf)
                generator.update_seq_lengths(
                    seq.latent_seq_len, seq.clip_seq_len, seq.sync_seq_len
                )
                encoder.update_seq_lengths(video_seq_len=seq.sync_seq_len)
                audio = evaluation.generate(
                    None,
                    sync.unsqueeze(0),
                    [plan.description],
                    feature_utils=features,
                    net_video_enc=encoder,
                    net_generator=generator,
                    fm=flow,
                    rng=rng,
                    cfg_strength=settings.selva_cfg,
                )
                mono = audio.float().cpu().numpy()[0].reshape(-1)
                chunks.append(resample_poly(mono, 160, 147))
    return write_wav(out, crossfade_join(chunks, spans, 48000))


@cache
def _load_clap():
    _require_offline()
    torch = import_module("torch")
    hub = import_module("huggingface_hub")
    # CLAP 1.1.5 parses training flags at import; Audyssey parsed its CLI already.
    argv = sys.argv
    try:
        sys.argv = argv[:1]
        clap_module = import_module("laion_clap")
    finally:
        sys.argv = argv
    checkpoint = hub.hf_hub_download(
        repo_id="lukewys/laion_clap", filename="630k-best.pt", local_files_only=True
    )
    clap = clap_module.CLAP_Module(enable_fusion=False, device=Settings().selva_device)
    clap.load_ckpt(checkpoint)
    clap.eval()
    return torch, clap


def score_generation(
    track: Path,
    plan: SourcePlan,
    siblings: list[SourcePlan],
) -> FilterResult:
    # Nothing to tell apart; also avoids CLAP's tokenizer, which breaks on one text.
    if not siblings:
        return FilterResult(passed=True, scores={"text_correct": 1})
    threshold = Settings().generation_margin_threshold
    mono = read_wav(track).mean(axis=1)
    # CLAP randomly crops inputs over 10 s; embed fixed 10 s windows and average.
    spans = windows(len(mono) / 48000, 10, 10)
    clips = np.stack(
        [mono]
        if len(spans) == 1
        else [
            mono[min(round(start * 48000), len(mono) - 480000) :][:480000]
            for start, _ in spans
        ]
    )
    with _model_lock:
        torch, clap = _load_clap()
        with torch.inference_mode():
            # CLAP returns L2-normalized embeddings.
            audio = clap.get_audio_embedding_from_data(x=clips).mean(axis=0)
            texts = clap.get_text_embedding(
                [plan.description, *[s.description for s in siblings]]
            )
    similarities = texts @ (audio / np.linalg.norm(audio))
    best = int(np.argmax(similarities[1:]))
    margin = float(similarities[0] - similarities[best + 1])
    correct = float(margin > 0)
    passed = correct == 1 and margin >= threshold
    reason = ""
    if not passed:
        reason = (
            f"closer to sibling: {siblings[best].name}"
            if not correct
            else f"text margin {margin:.3f} below threshold {threshold}"
        )
    return FilterResult(
        passed=passed,
        scores={"text_correct": correct, "text_margin": margin},
        reason=reason,
    )
