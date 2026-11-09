from uuid import UUID, uuid4

from pydantic import Field

from .base import SchemaModel


class ObjectSeed(SchemaModel):
    """
    Initial object candidate extracted from an InitialScene by the LLM.

    This is only a seed for later SAM3 tracking.
    It is not yet a confirmed object timeline.
    """

    object_seed_id: UUID = Field(default_factory=uuid4)

    label: str
    tracking_prompt: str | None = None
    seed_frame_index: int | None = Field(default=None, ge=0)

    notes: str | None = None


class InitialSceneAnalysis(SchemaModel):
    """
    Rough LLM analysis for one InitialScene.

    The rough_description can be noisy.
    The object_seeds are the important output because they drive SAM3 tracking.
    """

    analysis_id: UUID = Field(default_factory=uuid4)

    rough_description: str
    object_seeds: list[ObjectSeed] = Field(default_factory=list)
