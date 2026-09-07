from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

from .base import BaseClient


class KokoroClient(BaseClient):
    """Client for Kokoro speech generation."""

    async def generate_speech(
        self,
        text: str,
        *,
        target_duration_sec: float | None = None,
    ) -> str:
        request = {
            "text": text,
            "target_duration_sec": target_duration_sec,
        }
        response = await self._request(
            "POST",
            "/infer/kokoro/generate-speech",
            json=request,
        )

        output_path = Path(tempfile.gettempdir()) / f"kokoro_{uuid.uuid4().hex}.wav"
        output_path.write_bytes(response.content)
        return str(output_path)
