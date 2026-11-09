from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


DEFAULT_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts" / "files"
ThinkingLevel = Literal["minimal", "low", "medium", "high"]


def _is_nvenc_available() -> bool:
    import subprocess
    import shutil

    if not shutil.which("ffmpeg"):
        return False
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-f",
                "lavfi",
                "-i",
                "color=black:s=64x64:r=1:d=0.1",
                "-c:v",
                "h264_nvenc",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


class Settings(BaseSettings):
    prompts_dir: Path = DEFAULT_PROMPTS_DIR
    prompt_file_suffix: str = ".txt"

    llm_base_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "V2A_INSPECT_LLM_BASE_URL",
            "V2A_LLM_BASE_URL",
        ),
    )
    llm_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "V2A_INSPECT_LLM_API_KEY",
            "V2A_LLM_API_KEY",
            "GEMINI_API_KEY",
            "API_KEY",
        ),
    )
    llm_small_model: str = Field(
        default="gemini-3.1-flash-lite",
        validation_alias=AliasChoices(
            "V2A_INSPECT_LLM_SMALL_MODEL",
            "V2A_LLM_SMALL_MODEL",
        ),
    )
    llm_medium_model: str = Field(
        default="gemini-3.5-flash",
        validation_alias=AliasChoices(
            "V2A_INSPECT_LLM_MEDIUM_MODEL",
            "V2A_LLM_MEDIUM_MODEL",
            "V2A_INSPECT_LLM_MODEL",
            "V2A_LLM_MODEL",
        ),
    )
    llm_large_model: str = Field(
        default="gemini-3.5-flash",
        validation_alias=AliasChoices(
            "V2A_INSPECT_LLM_LARGE_MODEL",
            "V2A_LLM_LARGE_MODEL",
        ),
    )
    llm_thinking_level: ThinkingLevel | None = Field(
        default="low",
        validation_alias=AliasChoices(
            "V2A_INSPECT_LLM_THINKING_LEVEL",
            "V2A_LLM_THINKING_LEVEL",
        ),
    )
    llm_small_thinking_level: ThinkingLevel | None = Field(
        default="medium",
        validation_alias=AliasChoices(
            "V2A_INSPECT_LLM_SMALL_THINKING_LEVEL",
            "V2A_LLM_SMALL_THINKING_LEVEL",
        ),
    )
    llm_medium_thinking_level: ThinkingLevel | None = Field(
        default="low",
        validation_alias=AliasChoices(
            "V2A_INSPECT_LLM_MEDIUM_THINKING_LEVEL",
            "V2A_LLM_MEDIUM_THINKING_LEVEL",
        ),
    )
    llm_large_thinking_level: ThinkingLevel | None = Field(
        default="low",
        validation_alias=AliasChoices(
            "V2A_INSPECT_LLM_LARGE_THINKING_LEVEL",
            "V2A_LLM_LARGE_THINKING_LEVEL",
        ),
    )
    llm_thinking_budget: int | None = Field(
        default=None,
        ge=0,
        validation_alias=AliasChoices(
            "V2A_INSPECT_LLM_THINKING_BUDGET",
            "V2A_LLM_THINKING_BUDGET",
        ),
    )
    llm_small_thinking_budget: int | None = Field(
        default=None,
        ge=0,
        validation_alias=AliasChoices(
            "V2A_INSPECT_LLM_SMALL_THINKING_BUDGET",
            "V2A_LLM_SMALL_THINKING_BUDGET",
        ),
    )
    llm_medium_thinking_budget: int | None = Field(
        default=None,
        ge=0,
        validation_alias=AliasChoices(
            "V2A_INSPECT_LLM_MEDIUM_THINKING_BUDGET",
            "V2A_LLM_MEDIUM_THINKING_BUDGET",
        ),
    )
    llm_large_thinking_budget: int | None = Field(
        default=None,
        ge=0,
        validation_alias=AliasChoices(
            "V2A_INSPECT_LLM_LARGE_THINKING_BUDGET",
            "V2A_LLM_LARGE_THINKING_BUDGET",
        ),
    )
    llm_temperature: float = Field(
        default=1.0,
        validation_alias=AliasChoices(
            "V2A_INSPECT_LLM_TEMPERATURE",
            "V2A_LLM_TEMPERATURE",
        ),
    )
    llm_timeout_seconds: float | None = Field(
        default=600.0,
        validation_alias=AliasChoices(
            "V2A_INSPECT_LLM_TIMEOUT_SECONDS",
            "V2A_LLM_TIMEOUT_SECONDS",
        ),
    )
    llm_max_retries: int = Field(
        default=5,
        ge=1,
        validation_alias=AliasChoices(
            "V2A_INSPECT_LLM_MAX_RETRIES",
            "V2A_LLM_MAX_RETRIES",
        ),
    )
    llm_initial_scene_analysis_batch_size: int = Field(
        default=32,
        ge=1,
        validation_alias=AliasChoices(
            "V2A_INSPECT_LLM_INITIAL_SCENE_ANALYSIS_BATCH_SIZE",
            "V2A_LLM_INITIAL_SCENE_ANALYSIS_BATCH_SIZE",
        ),
    )
    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_API_KEY"),
    )
    elevenlabs_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("ELEVENLABS_API_KEY"),
    )
    agent_sound_timeline_recursion_limit: int = Field(
        default=500,
        ge=1,
        validation_alias=AliasChoices(
            "V2A_INSPECT_AGENT_SOUND_TIMELINE_RECURSION_LIMIT",
        ),
    )
    agent_sound_timeline_max_workers: int = Field(
        default=3,
        ge=0,
        validation_alias=AliasChoices(
            "V2A_INSPECT_AGENT_SOUND_TIMELINE_MAX_WORKERS",
        ),
    )
    agent_sound_timeline_segment_seconds: int = Field(
        default=30,
        ge=1,
        validation_alias=AliasChoices(
            "V2A_INSPECT_AGENT_SOUND_TIMELINE_SEGMENT_SECONDS",
        ),
    )
    langfuse_public_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "V2A_INSPECT_LANGFUSE_PUBLIC_KEY",
            "LANGFUSE_PUBLIC_KEY",
        ),
    )
    langfuse_secret_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "V2A_INSPECT_LANGFUSE_SECRET_KEY",
            "LANGFUSE_SECRET_KEY",
        ),
    )
    langfuse_base_url: str | None = Field(
        default="https://langfuse.riverfog7.com",
        validation_alias=AliasChoices(
            "V2A_INSPECT_LANGFUSE_BASE_URL",
            "LANGFUSE_BASE_URL",
        ),
    )
    langfuse_environment: str = Field(
        default="prod",
        validation_alias=AliasChoices(
            "V2A_INSPECT_LANGFUSE_ENVIRONMENT",
            "LANGFUSE_ENVIRONMENT",
        ),
    )
    langfuse_release: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "V2A_INSPECT_LANGFUSE_RELEASE",
            "LANGFUSE_RELEASE",
        ),
    )
    video_encode_use_nvenc: bool = Field(
        default_factory=_is_nvenc_available,
        validation_alias=AliasChoices(
            "V2A_INSPECT_VIDEO_ENCODE_USE_NVENC",
            "V2A_VIDEO_ENCODE_USE_NVENC",
        ),
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="V2A_INSPECT_",
        extra="ignore",
        secrets_dir="/run/secrets" if os.path.exists("/run/secrets") else None,
    )

    @field_validator(
        "llm_small_thinking_level",
        "llm_medium_thinking_level",
        "llm_large_thinking_level",
        "llm_thinking_level",
        "llm_thinking_budget",
        "llm_small_thinking_budget",
        "llm_medium_thinking_budget",
        "llm_large_thinking_budget",
        mode="before",
    )
    @classmethod
    def empty_thinking_config_to_none(cls, value: object) -> object:
        if value == "":
            return None
        return value

    @model_validator(mode="after")
    def validate_langfuse_keys(self) -> Settings:
        if (self.langfuse_public_key is None) != (self.langfuse_secret_key is None):
            raise ValueError(
                "Langfuse public and secret keys must either both be set or both be omitted."
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
