"""
Audio generation and composition module.
"""

from .client import generate_audio_for_item
from .mix import mix_audio_into_video

__all__ = ["generate_audio_for_item", "mix_audio_into_video"]
