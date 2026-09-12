"""
오디오 생성 클라이언트.

- Kokoro: Speech 오디오 생성
- HunyuanVideo-Foley: 영상 동기화가 필요한 V2A 생성
- ElevenLabs: SFX / Ambience T2A 생성
- Dummy: 생성 실패 시 fallback

환경 변수:
  ELEVENLABS_API_KEY — ElevenLabs 호출에 필요
"""

from __future__ import annotations

import asyncio
import logging
import shutil

import numpy as np
import scipy.io.wavfile as wavfile
from elevenlabs.client import ElevenLabs

from v2a_inspect.client import HunyuanClient, KokoroClient
from v2a_inspect.config.settings import get_settings

logger = logging.getLogger(__name__)


# ── Kokoro TTS ────────────────────────────────────────────────────────────────


def generate_speech_kokoro(
    text: str,
    out_path: str,
    duration: float | None = None,
    server_url: str | None = None,
) -> str:
    """Generate speech through the local Kokoro inference endpoint."""
    try:

        async def _generate() -> str:
            async with KokoroClient(base_url=server_url) as client:
                return await client.generate_speech(
                    text,
                    target_duration_sec=duration,
                )

        shutil.copy(asyncio.run(_generate()), out_path)
        return out_path
    except Exception as exc:
        logger.error("Failed to call Kokoro API: %s", exc)
        return generate_dummy_audio(duration or 1.0, out_path)


# ── ElevenLabs SFX ────────────────────────────────────────────────────────────


def generate_sfx_elevenlabs(
    text: str, out_path: str, duration: float | None = None
) -> str:
    """Generate sound effects using ElevenLabs API."""
    api_key = get_settings().elevenlabs_api_key
    if api_key is None:
        logger.warning("ELEVENLABS_API_KEY not found. Falling back to dummy audio.")
        return generate_dummy_audio(duration or 1.0, out_path)

    try:
        client = ElevenLabs(api_key=api_key.get_secret_value())
        dur_seconds = min(max(duration, 0.5), 30.0) if duration else None

        audio_generator = client.text_to_sound_effects.convert(
            text=text,
            duration_seconds=dur_seconds,
        )

        with open(out_path, "wb") as f:
            for chunk in audio_generator:
                f.write(chunk)
        return out_path
    except Exception as e:
        logger.error("ElevenLabs SFX generation failed: %s", e)
        return generate_dummy_audio(duration or 1.0, out_path)


# ── Dummy (fallback) ──────────────────────────────────────────────────────────


def generate_dummy_audio(
    duration_sec: float, out_path: str, sample_rate: int = 44100
) -> str:
    """Generate a placeholder silent/beep audio file (fallback)."""
    if duration_sec <= 0:
        duration_sec = 0.1

    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    audio = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    fade_len = int(0.1 * sample_rate)
    if audio.size > fade_len:
        audio[fade_len:] = 0

    wavfile.write(out_path, sample_rate, (audio * 32767).astype(np.int16))
    return out_path


# ── V2A Hunyuan API ───────────────────────────────────────────────────────────


def generate_v2a_hunyuan(
    video_id: str,
    fps: float,
    time: tuple[float, float],
    text: str,
    out_path: str,
    duration: float | None = None,
    server_url: str | None = None,
) -> str:
    """Generate audio using HunyuanVideo-Foley via unified client."""
    try:
        logger.info("Calling Hunyuan V2A API via unified client...")
        start_idx = int(time[0] * fps)
        end_idx = int(time[1] * fps)

        async def _generate():
            async with HunyuanClient(base_url=server_url) as client:
                return await client.generate_v2a(
                    video_id=video_id,
                    start_frame_index=start_idx,
                    end_frame_index=end_idx,
                    prompt=text,
                    guidance_scale=4.5,
                    num_inference_steps=50,
                )

        tmp_audio_path = asyncio.run(_generate())

        shutil.copy(tmp_audio_path, out_path)
        return out_path
    except Exception as e:
        logger.error("Failed to call Hunyuan API: %s", e)
        return generate_dummy_audio(duration or 1.0, out_path)


# ── Router ────────────────────────────────────────────────────────────────────


def generate_audio_for_item(
    kind: str,
    description: str,
    out_path: str,
    duration: float,
    time: tuple[float, float] | None = None,
    generation_model: str = "t2a",
    video_id: str | None = None,
    fps: float = 30.0,
    spoken_text: str | None = None,
    server_url: str | None = None,
) -> str | None:
    """Route one audio item to its explicitly selected generation model."""
    try:
        generation_model = {
            "tta": "t2a",
            "vta": "v2a",
            "hybrid": "t2a",
            "unknown": "t2a",
        }.get(generation_model, generation_model)

        if kind in ("dialogue", "speech"):
            generation_model = "tts"

        if generation_model == "tts":
            if kind not in ("dialogue", "speech"):
                raise ValueError("TTS generation requires speech kind")
            text = (spoken_text or description).strip()
            if not text:
                raise ValueError("TTS generation requires spoken_text")
            return generate_speech_kokoro(text, out_path, duration, server_url)

        if generation_model == "v2a":
            if not video_id or not time:
                raise ValueError("V2A generation requires video_id and time")
            return generate_v2a_hunyuan(
                video_id,
                fps,
                time,
                description,
                out_path,
                duration,
                server_url,
            )

        if generation_model != "t2a":
            raise ValueError(f"Unsupported generation model: {generation_model}")
        if kind in ("sfx", "ambience"):
            return generate_sfx_elevenlabs(description, out_path, duration=duration)
        raise ValueError(f"Unsupported sound type for T2A: {kind}")
    except Exception as exc:
        logger.error("Audio generation failed for '%s': %s", kind, exc)
        return None
