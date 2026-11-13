from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Span(Model):
    start: float = Field(ge=0)
    end: float

    @model_validator(mode="after")
    def check_end(self) -> Self:
        if self.end <= self.start:
            raise ValueError("end must be greater than start")
        return self


class VisualHint(Model):
    time: float
    box: tuple[float, float, float, float]
    description: str

    @model_validator(mode="after")
    def check_box(self) -> Self:
        if any(x < 0 or x > 1 for x in self.box):
            raise ValueError("box must be normalized to 0..1")
        return self


class GroundResult(Model):
    found: bool
    box: tuple[float, float, float, float] | None = None
    score: float | None = None

    @model_validator(mode="after")
    def check_box(self) -> Self:
        if self.box is not None:
            x0, y0, x1, y1 = self.box
            if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
                raise ValueError("box must be ordered and normalized to 0..1")
        return self


SourceKind = Literal["dialogue", "sfx", "music", "ambience"]


class SourcePlan(Model):
    name: str = Field(min_length=1)
    kind: SourceKind
    description: str
    spans: list[Span] = Field(default_factory=list)
    visual_hint: VisualHint | None = None


class PlanIssue(Model):
    source_name: str | None
    kind: str
    message: str


class FilterResult(Model):
    passed: bool
    scores: dict[str, float] = Field(default_factory=dict)
    reason: str = ""
    suggestion: SourcePlan | None = None


SourceStatus = Literal["pending", "running", "done", "failed", "skipped"]


class Source(Model):
    id: str
    plan: SourcePlan
    origin: Literal["separated", "generated"] = "separated"
    replaces: str | None = None
    track_path: str | None = None
    filter: FilterResult | None = None
    status: SourceStatus = "pending"
    attempts: int = 0
    keep: bool = True
    gain: float = Field(default=1, ge=0, le=2)
    pitch_semitones: float = Field(default=0, ge=-12, le=12)


class Residual(Model):
    path: str | None = None
    keep: bool = True
    gain: float = Field(default=1, ge=0, le=2)


class StepStatus(Model):
    key: str
    status: Literal["pending", "running", "done", "failed", "warning"]
    round: int = 0
    max_rounds: int = 0
    message: str = ""


class RenderOutput(Model):
    mix_wav: str
    video_mp4: str
    stems: list[str]


class Job(Model):
    id: str
    video_path: str
    silent: bool = False
    duration: float = 0
    audio_path: str = ""
    mono_path: str = ""
    peaks_path: str = ""
    sources: list[Source] = Field(default_factory=list)
    residual: Residual = Field(default_factory=Residual)
    plan_issues: list[PlanIssue] = Field(default_factory=list)
    steps: list[StepStatus] = Field(default_factory=list)
    outputs: RenderOutput | None = None
    state: Literal["running", "ready", "failed"] = "running"
    version: int = 0
    error: str | None = None

    def find_source(self, source_id: str) -> Source | None:
        return next((s for s in self.sources if s.id == source_id), None)


class Keep(Model):
    type: Literal["keep"] = "keep"
    source_id: str


class Drop(Model):
    type: Literal["drop"] = "drop"
    source_id: str


class SetGain(Model):
    type: Literal["set_gain"] = "set_gain"
    source_id: str
    gain: float = Field(ge=0, le=2)


class SetPitch(Model):
    type: Literal["set_pitch"] = "set_pitch"
    source_id: str
    semitones: float = Field(ge=-12, le=12)


class Reseparate(Model):
    type: Literal["reseparate"] = "reseparate"
    source_id: str
    description: str | None = None
    spans: list[Span] | None = None


class Generate(Model):
    type: Literal["generate"] = "generate"
    source_id: str
    description: str | None = None


class AddSource(Model):
    type: Literal["add_source"] = "add_source"
    plan: SourcePlan
    method: Literal["separate", "generate"] = "separate"


class Render(Model):
    type: Literal["render"] = "render"
    include_residual_stem: bool = False


Command = Annotated[
    Keep | Drop | SetGain | SetPitch | Reseparate | Generate | AddSource | Render,
    Field(discriminator="type"),
]


def parse_command(data: dict) -> Command:
    return TypeAdapter(Command).validate_python(data)
