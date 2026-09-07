from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints


class KokoroGenerateSpeechRequest(BaseModel):
    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    target_duration_sec: float | None = Field(
        default=None,
        gt=0,
        le=300,
    )
