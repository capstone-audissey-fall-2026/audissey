from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AUDYSSEY_")

    planner: Literal["stub", "real"] = "stub"
    separator: Literal["stub", "real", "remote"] = "stub"
    filter: Literal["stub", "real", "remote"] = "stub"
    grounder: Literal["stub", "real", "remote"] = "stub"
    grounding_url: str = ""
    generator: Literal["stub", "real", "remote"] = "stub"
    generator_url: str = ""
    selva_base_model: str = "jnwnlee/SelVA"
    selva_ft_model: str = "sogang-audyssey/selva-ft-selsync"
    selva_ft_generator_file: str = "generator/S_O_s43/model_ema_step1000.pth"
    selva_ft_encoder_file: str = "encoder/S_O/best.pt"
    selva_device: str = "cuda"
    selva_steps: int = 25
    selva_cfg: float = 4.5
    selva_seed: int = 42
    selva_single_max_s: float = 10
    selva_window_s: float = 8
    max_generation_retries: int = Field(default=2, ge=0)
    generation_margin_threshold: float = 0.0
    # Visual prompting (SAM3 locate + masked video for SAM Audio) is off by default:
    # with a text description the mask barely changed separation in our tests.
    visual_prompt: bool = False
    sam3_model: str = "facebook/sam3.1"
    sam3_device: str = "cuda"
    # SAM3 keeps per-frame tracking state on the GPU; 1080p60 at 4 fps already needs >24 GB.
    # 0 tracks at the source frame rate.
    sam3_track_fps: float = Field(default=2, ge=0)
    separator_url: str = ""
    filter_url: str = ""
    tool_token: str = Field(default="", repr=False)
    remote_timeout_s: float = Field(default=600, gt=0)
    planner_base_url: str = ""
    planner_frames: int = Field(default=4, ge=1, le=16)
    planner_model: str = ""
    planner_api_key: str = Field(default="", repr=False)
    max_plan_rounds: int = Field(default=2, ge=1)
    max_separation_retries: int = Field(default=2, ge=0)
    max_duration_s: float = Field(default=30, gt=0)
    # Per uploaded file, on the app and the tool server; checked while streaming.
    max_upload_mb: float = Field(default=200, gt=0)
    work_dir: Path = Path("work/jobs")
    sam_audio_model: str = "facebook/sam-audio-large-tv"
    sam_audio_judge_model: str = "facebook/sam-audio-judge"
    sam_audio_device: str = "cuda"
    sam_audio_reranking_candidates: int = Field(default=1, ge=1)
    sam_audio_predict_spans: bool = False
    # Reranking (candidates > 1) only: CLAP and ImageBind checkpoint files. Empty CLAP
    # uses the cached lukewys/laion_clap 630k-best.pt; empty ImageBind disables the
    # visual ranker, so masked videos fall back to the text rankers (CLAP + Judge).
    sam_audio_clap_checkpoint: str = ""
    sam_audio_imagebind_checkpoint: str = ""
    judge_pass_threshold: float = 0.5
