from __future__ import annotations

from .endpoints.video import VideoClient
from .endpoints.sam3 import SAM3Client
from .endpoints.embed import EmbeddingClient
from .endpoints.score import ScoringClient
from .endpoints.hunyuan import HunyuanClient
from .endpoints.speech import KokoroClient
from .config import settings

__all__ = [
    "VideoClient",
    "SAM3Client",
    "EmbeddingClient",
    "ScoringClient",
    "HunyuanClient",
    "KokoroClient",
    "settings",
]
