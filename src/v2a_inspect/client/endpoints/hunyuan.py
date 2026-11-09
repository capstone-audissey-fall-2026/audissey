from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

from .base import BaseClient


class HunyuanClient(BaseClient):
    """Client for HunyuanVideo-Foley V2A generation."""

    async def generate_v2a(
        self,
        video_id: str,
        start_frame_index: int,
        end_frame_index: int,
        prompt: str,
        guidance_scale: float = 4.5,
        num_inference_steps: int = 50,
        negative_prompt: str | None = None,
    ) -> str:
        """
        Generate audio using HunyuanVideo-Foley and return the saved local audio path.
        """
        request = {
            "video_id": video_id,
            "start_frame_index": start_frame_index,
            "end_frame_index": end_frame_index,
            "prompt": prompt,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "negative_prompt": negative_prompt,
        }

        # BaseClient._request assumes returning httpx.Response
        response = await self._request(
            "POST", "/infer/hunyuan/generate-v2a", json=request
        )

        # Save the audio stream to a temporary file
        out_path = str(Path(tempfile.gettempdir()) / f"hunyuan_{uuid.uuid4().hex}.wav")
        with open(out_path, "wb") as f:
            f.write(response.content)

        return out_path
