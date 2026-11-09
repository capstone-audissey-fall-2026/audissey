from __future__ import annotations

from pydantic import BaseModel, Field


class HunyuanGenerateV2ARequest(BaseModel):
    video_id: str = Field(..., description="ID of the uploaded video")
    start_frame_index: int = Field(
        ..., description="Start frame index for audio generation"
    )
    end_frame_index: int = Field(
        ..., description="End frame index for audio generation"
    )
    prompt: str = Field(..., description="Text prompt for Foley sound generation")
    guidance_scale: float = Field(4.5, description="Guidance scale for generation")
    num_inference_steps: int = Field(50, description="Number of inference steps")
    negative_prompt: str | None = Field(None, description="Negative prompt")
