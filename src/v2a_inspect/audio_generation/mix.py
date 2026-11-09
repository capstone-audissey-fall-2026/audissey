"""
오디오 믹싱 및 영상 합성 모듈.

AudioPlan 기반으로 생성된 오디오 파일들을 원본 비디오에 합성합니다.

주요 기능:
  - Volume 적용: 각 AudioPlanItem의 volume 값 반영
  - Pan 적용: 스테레오 패닝 (-1.0 좌 ~ 1.0 우)
  - 시간 정렬: 각 오디오를 정확한 타임라인 위치에 배치
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

from moviepy import AudioFileClip, CompositeAudioClip, VideoFileClip
from moviepy.audio.fx import MultiplyVolume, AudioFadeOut

from v2a_inspect.models.audio_plan import AudioPlan, AudioPlanItem


logger = logging.getLogger(__name__)


def mix_audio_into_video(
    video_path: str,
    audio_plan: AudioPlan,
    generated_audio: dict[str, str],
    output_path: str,
    *,
    keep_original_audio: bool = False,
) -> str | None:
    """
    생성된 오디오 트랙들을 원본 비디오에 합성합니다.

    Parameters
    ----------
    video_path : str
        원본 비디오 파일 경로.
    audio_plan : AudioPlan
        오디오 플랜 (타이밍, 볼륨, 팬 정보).
    generated_audio : dict[str, str]
        item_id → 생성된 wav 파일 경로 매핑.
    output_path : str
        출력 비디오 파일 경로.
    keep_original_audio : bool
        True이면 원본 비디오의 오디오를 유지합니다.

    Returns
    -------
    str | None
        성공 시 출력 파일 경로, 실패 시 None.
    """
    if not generated_audio:
        logger.error("No generated audio to mix.")
        return None

    try:
        video = VideoFileClip(video_path)
        audio_clips: list = []

        # 원본 오디오 유지 옵션
        if keep_original_audio and video.audio is not None:
            audio_clips.append(video.audio)

        # 각 AudioPlanItem에 대해 오디오 클립 생성 및 배치
        n_mixed = 0
        for item in audio_plan.items:
            if item.type == "silence":
                continue

            wav_path = generated_audio.get(item.item_id)
            if not wav_path or not Path(wav_path).exists():
                continue

            duration = item.time[1] - item.time[0]
            clip = AudioFileClip(wav_path)
            clip_dur = clip.duration or duration
            target_duration = max(0.01, min(clip_dur, duration))
            clip = clip.subclipped(0, target_duration).with_duration(target_duration)
            clip = clip.with_start(item.time[0])

            effects = []
            # 1. Volume 적용
            if item.volume != 1.0:
                effects.append(MultiplyVolume(item.volume))

            # 자연스러운 페이드아웃 추가. Very short clips can trip MoviePy's
            # bounds checks during composite rendering, so leave them untouched.
            fade_duration = min(0.15, target_duration / 2.0)
            if target_duration >= 0.3 and fade_duration > 0:
                effects.append(AudioFadeOut(fade_duration))

            if effects:
                clip = clip.with_effects(effects)

            # 2. Stereo Pan 적용
            if item.pan != 0.0:
                clip = _apply_pan(clip, item.pan)

            audio_clips.append(clip)
            n_mixed += 1

        if not audio_clips:
            logger.warning("No audio clips to mix into video.")
            video.close()
            return None

        # 오디오 합성
        final_audio = CompositeAudioClip(audio_clips)
        if (
            final_audio.duration is not None
            and video.duration is not None
            and final_audio.duration > video.duration
        ):
            final_audio = final_audio.subclipped(0, video.duration)

        video = video.with_audio(final_audio)

        # 비디오 출력
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        logger.info("Writing mixed video: %s (%d tracks)", output_path, n_mixed)
        video.write_videofile(
            output_path,
            audio_codec="aac",
            fps=video.fps,
            temp_audiofile_path=str(Path(output_path).parent),
            logger=None,
        )

        # 리소스 정리
        video.close()
        for clip in audio_clips:
            if hasattr(clip, "close"):
                clip.close()

        return output_path

    except Exception as e:
        logger.error("Failed to mix video: %s", e)
        return None


def mix_audio_items_to_wav(
    items: Iterable[AudioPlanItem],
    generated_audio: dict[str, str],
    output_path: str,
    *,
    total_duration: float,
    fps: int = 44100,
) -> str | None:
    """Mix selected AudioPlan items into one full-duration WAV stem."""

    audio_clips: list = []
    try:
        n_mixed = 0
        for item in items:
            if item.type == "silence":
                continue

            wav_path = generated_audio.get(item.item_id)
            if not wav_path or not Path(wav_path).exists():
                continue

            duration = item.time[1] - item.time[0]
            clip = AudioFileClip(wav_path)
            clip_dur = clip.duration or duration
            target_duration = max(0.01, min(clip_dur, duration))
            clip = clip.subclipped(0, target_duration).with_duration(target_duration)
            clip = clip.with_start(item.time[0])

            effects = []
            if item.volume != 1.0:
                effects.append(MultiplyVolume(item.volume))

            fade_duration = min(0.15, target_duration / 2.0)
            if target_duration >= 0.3 and fade_duration > 0:
                effects.append(AudioFadeOut(fade_duration))

            if effects:
                clip = clip.with_effects(effects)

            if item.pan != 0.0:
                clip = _apply_pan(clip, item.pan)

            audio_clips.append(clip)
            n_mixed += 1

        if not audio_clips:
            logger.warning("No audio clips to mix into stem: %s", output_path)
            return None

        final_audio = CompositeAudioClip(audio_clips).with_duration(total_duration)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        logger.info("Writing audio stem: %s (%d clips)", output_path, n_mixed)
        final_audio.write_audiofile(
            output_path,
            fps=fps,
            codec="pcm_s16le",
            logger=None,
        )
        final_audio.close()
        return output_path
    except Exception as e:
        logger.error("Failed to mix audio stem: %s", e)
        return None
    finally:
        for clip in audio_clips:
            if hasattr(clip, "close"):
                clip.close()


# ── Pan 효과 ──────────────────────────────────────────────────────────────────


def _apply_pan(clip, pan: float):
    """
    Simple stereo pan: -1.0 = left only, 0.0 = center, 1.0 = right only.
    Uses channel volume weighting.
    """
    try:
        left_vol = max(0.0, 1.0 - pan)
        right_vol = max(0.0, 1.0 + pan)
        peak = max(left_vol, right_vol, 1e-6)
        left_vol /= peak
        right_vol /= peak

        def _pan_audio(get_frame, t):
            import numpy as np

            frame = get_frame(t)
            if frame.ndim == 1:
                frame = np.c_[frame, frame]  # mono → stereo
            frame = frame.copy().astype(float)
            frame[:, 0] *= left_vol
            frame[:, 1] *= right_vol
            return frame

        return clip.transform(_pan_audio, apply_to="audio")
    except Exception:
        return clip
