from __future__ import annotations

import logging
import threading
import uuid

import numpy as np

from ..models.speech import KokoroGenerateSpeechRequest
from ..settings import settings

logger = logging.getLogger("uvicorn.error")

try:
    import soundfile as sf
    from kokoro import KPipeline

    KOKORO_AVAILABLE = True
except ImportError:
    sf = None
    KPipeline = None
    KOKORO_AVAILABLE = False


class KokoroInferenceClient:
    """Lazy, serialized Kokoro speech inference."""

    def __init__(self) -> None:
        self._pipeline = None
        self._lock = threading.Lock()

    def close(self) -> None:
        self._pipeline = None

    def generate_speech(self, request: KokoroGenerateSpeechRequest) -> str:
        if not KOKORO_AVAILABLE:
            raise RuntimeError("Kokoro is not installed.")

        with self._lock:
            pipeline = self._get_pipeline()
            chunks = [
                _to_numpy(audio)
                for _graphemes, _phonemes, audio in pipeline(
                    request.text,
                    voice=settings.kokoro_voice,
                    speed=_speech_speed(
                        request.text,
                        request.target_duration_sec,
                    ),
                )
            ]

        if not chunks:
            raise RuntimeError("Kokoro returned no audio.")

        output_path = settings.upload_dir / f"speech_{uuid.uuid4().hex}.wav"
        sf.write(output_path, np.concatenate(chunks), 24_000)
        return str(output_path)

    def _get_pipeline(self):
        if self._pipeline is None:
            logger.info(
                "Loading Kokoro model %s on %s.",
                settings.kokoro_model_id,
                settings.kokoro_device,
            )
            self._pipeline = KPipeline(
                lang_code=settings.kokoro_language,
                repo_id=settings.kokoro_model_id,
                device=settings.kokoro_device,
            )
        return self._pipeline


def _speech_speed(text: str, target_duration_sec: float | None) -> float:
    if target_duration_sec is None:
        return 1.0
    estimated_duration = max(len(text) / 12.0, 0.5)
    return min(2.0, max(0.5, estimated_duration / target_duration_sec))


def _to_numpy(audio: object) -> np.ndarray:
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    return np.asarray(audio, dtype=np.float32)
