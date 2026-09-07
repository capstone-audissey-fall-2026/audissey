from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings


class ServerSettings(BaseSettings):
    upload_dir: Path = Field(
        default=Path("/tmp/v2a_uploads"),
        validation_alias=AliasChoices("V2A_SERVER_UPLOAD_DIR", "V2A_UPLOAD_DIR"),
    )
    host: str = "0.0.0.0"
    port: int = 8080
    sam31_max_num_objects: int = 48
    sam31_use_fa3: bool = False
    sam31_use_rope_real: bool = True
    sam31_compile: bool = False
    sam31_warm_up: bool = False
    opencv_video_backend: str = "ffmpeg"
    opencv_hw_acceleration: bool = True
    opencv_hw_device: int | None = None
    opencv_ffmpeg_capture_options: str | None = "hw_decoders_any;cuda"
    embedding_model_id: str = "facebook/dinov2-base"
    label_model_id: str = "google/siglip2-base-patch16-224"
    enable_nvenc: bool = True
    hunyuan_model_id: str = "tencent/HunyuanVideo-Foley"
    hunyuan_model_size: str = "xl"
    hunyuan_enable_offload: bool = False
    kokoro_model_id: str = "hexgrad/Kokoro-82M"
    kokoro_language: str = "a"
    kokoro_voice: str = "af_heart"
    kokoro_device: Literal["cpu", "cuda"] = "cpu"
    pytorch_cuda_alloc_conf: str = "expandable_segments:True"

    class Config:
        env_prefix = "V2A_SERVER_"


settings = ServerSettings()

# Ensure the upload directory exists
settings.upload_dir.mkdir(parents=True, exist_ok=True)
