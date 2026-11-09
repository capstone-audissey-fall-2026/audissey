from __future__ import annotations

import base64
import mimetypes
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from pydantic import Field

from v2a_inspect.config import settings
from v2a_inspect.llm import model_manager
from v2a_inspect.models import (
    InitialScene,
    InitialSceneAnalysis,
    Keyframe,
    ObjectSeed,
    SchemaModel,
)
from v2a_inspect.observability import build_langchain_config
from v2a_inspect.prompts.manager import PromptManager


def analyze_initial_scene(
    initial_scene: InitialScene,
    model: BaseChatModel | None = None,
) -> InitialScene:
    prompt = PromptManager().get_multimodal_prompt(
        "initial_scene_analysis",
        content_blocks=_image_blocks(initial_scene.keyframes),
    )
    chat_model = model or model_manager.small
    chain = prompt | chat_model.with_structured_output(_InitialSceneAnalysisOutput)
    result = chain.invoke(
        _scene_inputs(initial_scene),
        config=_trace_config(initial_scene),
    )
    analysis = _to_initial_scene_analysis(
        _InitialSceneAnalysisOutput.model_validate(result)
    )
    return initial_scene.model_copy(update={"initial_analysis": analysis})


class _ObjectSeedOutput(SchemaModel):
    label: str
    tracking_prompt: str | None = None
    seed_frame_index: int | None = Field(default=None, ge=0)
    notes: str | None = None


class _InitialSceneAnalysisOutput(SchemaModel):
    rough_description: str
    object_seeds: list[_ObjectSeedOutput] = Field(default_factory=list)


def _to_initial_scene_analysis(
    output: _InitialSceneAnalysisOutput,
) -> InitialSceneAnalysis:
    return InitialSceneAnalysis(
        rough_description=output.rough_description,
        object_seeds=[
            ObjectSeed(
                label=object_seed.label,
                tracking_prompt=object_seed.tracking_prompt,
                seed_frame_index=object_seed.seed_frame_index,
                notes=object_seed.notes,
            )
            for object_seed in output.object_seeds
        ],
    )


def analyze_initial_scenes(
    initial_scenes: list[InitialScene],
    model: BaseChatModel | None = None,
) -> list[InitialScene]:
    if not initial_scenes:
        return []

    chat_model = model or model_manager.small
    analyzed_scenes = [analyze_initial_scene(initial_scenes[0], model=chat_model)]
    batch_size = settings.llm_initial_scene_analysis_batch_size
    remaining_scenes = initial_scenes[1:]

    def analyze_batch_scene(initial_scene: InitialScene) -> InitialScene:
        return analyze_initial_scene(initial_scene, model=chat_model)

    for batch_start in range(0, len(remaining_scenes), batch_size):
        batch = remaining_scenes[batch_start : batch_start + batch_size]
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            analyzed_scenes.extend(executor.map(analyze_batch_scene, batch))
    return analyzed_scenes


def _keyframe_indexes_text(keyframes: list[Keyframe]) -> str:
    frame_indexes: list[str] = []
    for frame_index in _keyframe_indexes(keyframes):
        frame_indexes.append(str(frame_index))
    return ", ".join(frame_indexes)


def _keyframe_indexes(keyframes: list[Keyframe]) -> list[int]:
    frame_indexes: list[int] = []
    for keyframe in keyframes:
        frame_indexes.append(keyframe.frame_index)
    return frame_indexes


def _scene_inputs(initial_scene: InitialScene) -> dict[str, Any]:
    return {
        "initial_scene_id": initial_scene.initial_scene_id,
        "start_frame_index": initial_scene.start_frame_index,
        "end_frame_index": initial_scene.end_frame_index,
        "frame_count": initial_scene.frame_count,
        "keyframe_indexes": _keyframe_indexes_text(initial_scene.keyframes),
    }


def _trace_config(initial_scene: InitialScene) -> RunnableConfig | None:
    return build_langchain_config(
        run_name="initial_scene_analysis",
        tags=["v2a-inspect", "preprocessing", "initial-scene-analysis"],
        metadata={
            "initial_scene_id": str(initial_scene.initial_scene_id),
            "start_frame_index": str(initial_scene.start_frame_index),
            "end_frame_index": str(initial_scene.end_frame_index),
            "frame_count": str(initial_scene.frame_count),
            "keyframe_count": str(len(initial_scene.keyframes)),
            "keyframe_indexes": _keyframe_indexes_text(initial_scene.keyframes),
        },
    )


def _image_blocks(keyframes: list[Keyframe]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for keyframe in keyframes:
        blocks.append(_image_block(keyframe.image_path))
    return blocks


def _image_block(image_path: Path) -> dict[str, Any]:
    mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
    }
