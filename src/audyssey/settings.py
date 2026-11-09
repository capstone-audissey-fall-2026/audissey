from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AUDYSSEY_")

    planner: Literal["stub", "real"] = "stub"
    separator: Literal["stub", "real"] = "stub"
    filter: Literal["stub", "real"] = "stub"
    max_plan_rounds: int = Field(default=2, ge=1)
    max_separation_retries: int = Field(default=2, ge=0)
    max_duration_s: float = Field(default=30, gt=0)
    work_dir: Path = Path("work/jobs")
    sam_audio_model: str = "facebook/sam-audio-large-tv"
    sam_audio_judge_model: str = "facebook/sam-audio-judge"
