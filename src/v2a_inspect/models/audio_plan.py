from pydantic import Field
from .base import SchemaModel


class AudioPlanItem(SchemaModel):
    item_id: str
    type: str
    time: tuple[float, float]
    description: str
    spoken_text: str | None = None
    volume: float = 0.8
    intensity: float = 0.5
    pan: float = 0.0
    confidence: float = 1.0
    track_id: str | None = None
    generation_model: str = "t2a"


class AudioPlan(SchemaModel):
    items: list[AudioPlanItem] = Field(default_factory=list)
    total_duration: float = 0.0
