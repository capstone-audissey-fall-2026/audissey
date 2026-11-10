"""플래닝 담당 담당 모듈의 기본 구현. 성능이 아니라 연결 확인용. 교체 가능."""

import base64
import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypeVar

from langsmith import trace
from langsmith.run_helpers import tracing_context
from openai.types.chat import ChatCompletionContentPartParam, ChatCompletionMessageParam
from pydantic import TypeAdapter, ValidationError

from audyssey import media
from audyssey.contracts import Model, PlanIssue, SourcePlan, Span, VisualHint
from audyssey.settings import Settings
from audyssey.tools.planning import llm


class _Sources(Model):
    sources: list[object]


class _Issues(Model):
    issues: list[PlanIssue]


class _Span(Model):
    start: float
    end: float


class _Hint(Model):
    time: float
    box: tuple[float, float, float, float]
    description: str


_Response = TypeVar("_Response", bound=Model)


def _messages(
    video: Path, settings: Settings, system: str, text: str
) -> list[ChatCompletionMessageParam]:
    content: list[ChatCompletionContentPartParam] = [{"type": "text", "text": text}]
    with TemporaryDirectory() as tmp:
        for time, path in media.extract_frames(
            video, settings.planner_frames, Path(tmp)
        ):
            content.append({"type": "text", "text": f"Frame at {time:g} seconds"})
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                }
            )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": content},
    ]


def _request(
    settings: Settings,
    messages: list[ChatCompletionMessageParam],
    response_type: type[_Response],
) -> _Response:
    with llm.chat_client(settings) as client:
        for attempt in range(2):
            # Record model/usage only: the SDK wrapper would also trace JPEG bytes.
            with trace(
                "planner_completion",
                run_type="llm",
                inputs={"model": settings.planner_model},
                metadata={
                    "ls_model_name": settings.planner_model,
                    "ls_model_type": "chat",
                },
            ) as run:
                with tracing_context(enabled=False, parent=False):
                    response = client.chat.completions.create(
                        model=settings.planner_model,
                        messages=messages,
                        response_format={"type": "json_object"},
                    )
                usage = response.usage
                run.end(
                    outputs={
                        "usage_metadata": {
                            "input_tokens": usage.prompt_tokens,
                            "output_tokens": usage.completion_tokens,
                            "total_tokens": usage.total_tokens,
                        }
                    }
                    if usage
                    else {}
                )
            raw = response.choices[0].message.content if response.choices else None
            try:
                data = json.loads(raw or "")
                if isinstance(data, list):
                    # Gemini sometimes drops the {"sources": [...]} wrapper.
                    (field,) = response_type.model_fields
                    data = {field: data}
                return response_type.model_validate(data)
            except (json.JSONDecodeError, ValidationError):
                if attempt == 0:
                    messages.append(
                        {
                            "role": "user",
                            "content": "Return valid JSON matching the schema.",
                        }
                    )
    excerpt = raw or ""
    if settings.planner_api_key:
        excerpt = excerpt.replace(settings.planner_api_key, "***")
    raise RuntimeError(f"planner returned invalid JSON: {excerpt[:500]}") from None


def _source(item: object, duration: float) -> SourcePlan:
    data = TypeAdapter(dict[str, object]).validate_python(item)
    if "spans" in data:
        if not isinstance(data["spans"], list):
            raise ValueError("spans must be a list")
        spans = []
        for value in data["spans"]:
            span = _Span.model_validate(value)
            start, end = (
                max(0, min(duration, span.start)),
                max(0, min(duration, span.end)),
            )
            if start < end:
                spans.append(Span(start=start, end=end))
        data["spans"] = spans
    if data.get("visual_hint") is not None:
        hint = _Hint.model_validate(data["visual_hint"])
        box = tuple(max(0, min(1, x)) for x in hint.box)
        data["visual_hint"] = (
            VisualHint(
                time=max(0, min(duration, hint.time)),
                box=(box[0], box[1], box[2], box[3]),
                description=hint.description,
            )
            if box[0] < box[2] and box[1] < box[3]
            else None
        )
    return SourcePlan.model_validate(data)


def plan(
    video: Path, audio: Path, duration: float, feedback: list[PlanIssue]
) -> list[SourcePlan]:
    settings = Settings()
    messages = _messages(
        video,
        settings,
        "You are an audio source planner. A source is one sound-producing entity. "
        "Do not split the same entity; use multiple spans for repeated sounds. "
        "Kinds: dialogue, sfx, music, ambience. Only frames are available, no audio; "
        "infer likely sources without claiming to hear them. Spans and hint times "
        "are seconds; boxes are [x0,y0,x1,y1] normalized to 0..1. "
        "Offscreen sources have visual_hint=null. Return JSON with a sources list. "
        "Each source follows this schema: "
        + json.dumps(SourcePlan.model_json_schema()),
        f"Video duration: {duration:g} seconds. Feedback: "
        + json.dumps([issue.model_dump() for issue in feedback]),
    )
    response = _request(settings, messages, _Sources)
    sources = []
    for item in response.sources:
        try:
            sources.append(_source(item, duration))
        except ValueError:
            continue
    if not sources:
        raise RuntimeError("planner returned no valid sources")
    return sources


def validate_plan(video: Path, audio: Path, plans: list[SourcePlan]) -> list[PlanIssue]:
    settings = Settings()
    messages = _messages(
        video,
        settings,
        "Check the source list against the frames. Report only suspicious sources; "
        "do not report unsupported guesses about unheard audio. "
        "Return JSON with an issues list (empty when none). Each issue follows: "
        + json.dumps(PlanIssue.model_json_schema()),
        "Sources: " + json.dumps([source.model_dump() for source in plans]),
    )
    try:
        return _request(settings, messages, _Issues).issues
    except RuntimeError:
        logging.getLogger(__name__).warning(
            "plan validator returned invalid JSON twice; proceeding without issues"
        )
        return []
