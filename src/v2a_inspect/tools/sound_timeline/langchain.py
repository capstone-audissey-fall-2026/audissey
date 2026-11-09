from __future__ import annotations

from collections.abc import Callable
import json
from typing import TYPE_CHECKING
from uuid import UUID

from langchain_core.tools import StructuredTool

from v2a_inspect.models import SchemaModel, SoundEvent, SoundSource, SoundTrack

from .schemas import (
    AnnotatedFrameOutput,
    CatalogArgs,
    DeleteSoundEventArgs,
    DeleteSoundSourceArgs,
    DeleteSoundTrackArgs,
    FrameIndexArgs,
    FrameResolutionMode,
    ListScenesArgs,
    SceneIndexArgs,
    SoundGenerationModel,
    SoundSourceType,
    SoundTimelineArgs,
    SoundTrackType,
    UpsertSoundEventArgs,
    UpsertSoundSourceArgs,
    UpsertSoundTrackArgs,
    VisualEventsArgs,
    disambiguate_labels,
)

if TYPE_CHECKING:
    from .editor import SoundTimelineEditor

ToolMessageContentBlock = str | dict[str, object]
MAX_TOOL_STRING_CHARS = 6000


def build_sound_timeline_tools(editor: SoundTimelineEditor) -> list[StructuredTool]:
    return [
        StructuredTool.from_function(
            _make_list_scenes_tool(editor),
            args_schema=ListScenesArgs,
            description=(
                "List a page of initial scenes, frame ranges, keyframes, and "
                "counts. Use start_scene_index and limit for the next small "
                "batch only; do not page ahead before writing timeline edits."
            ),
        ),
        StructuredTool.from_function(
            _make_get_scene_summary_tool(editor),
            args_schema=SceneIndexArgs,
            description=(
                "Primary per-scene inspection tool. Get frame range, object seeds, "
                "keyframes, and tracks before using frame images. Use only for "
                "the current scene batch; do not call repeatedly without writing events."
            ),
        ),
        StructuredTool.from_function(
            _make_annotated_frame_tool(editor),
            args_schema=FrameIndexArgs,
            description=(
                "Targeted visual confirmation only. Returns LangChain message "
                "blocks: text, image, and JSON track metadata. Use sparingly "
                "after scene summary and visual events. Use resolution_mode='low' "
                "by default; use 'high' only when low-res lacks needed detail."
            ),
        ),
        StructuredTool.from_function(
            _make_list_tracks_tool(editor),
            args_schema=SceneIndexArgs,
            description="List tracked visual objects inside one scene.",
        ),
        StructuredTool.from_function(
            _make_get_visual_events_tool(editor),
            args_schema=VisualEventsArgs,
            description=(
                "Primary visual event timeline. List bounded computed visual "
                "events for the current scene or batch frame window. If truncated, "
                "narrow the frame interval."
            ),
        ),
        StructuredTool.from_function(
            _make_get_sound_timeline_tool(editor),
            args_schema=SoundTimelineArgs,
            description=(
                "Read a bounded view of the current SoundTimeline. Use frame "
                "windows after editing each batch before inspecting later scenes."
            ),
        ),
        StructuredTool.from_function(
            _make_list_sound_sources_tool(editor),
            args_schema=CatalogArgs,
            description=(
                "Fuzzy-search compact global SoundSource catalog for reuse checks. "
                "Default limit=3. Use query before creating a source."
            ),
        ),
        StructuredTool.from_function(
            _make_list_sound_tracks_tool(editor),
            args_schema=CatalogArgs,
            description=(
                "Fuzzy-search compact global SoundTrack catalog for reuse checks. "
                "Default limit=3. Use query before creating a track."
            ),
        ),
        StructuredTool.from_function(
            _make_upsert_sound_source_tool(editor),
            args_schema=UpsertSoundSourceArgs,
            description=(
                "Create or update a SoundSource in the SoundTimeline. Use as soon "
                "as a recurring sound emitter is known. Returns id=<uuid> for "
                "later track/source updates."
            ),
        ),
        StructuredTool.from_function(
            _make_delete_sound_source_tool(editor),
            args_schema=DeleteSoundSourceArgs,
            description="Delete a SoundSource and detach tracks that referenced it.",
        ),
        StructuredTool.from_function(
            _make_upsert_sound_track_tool(editor),
            args_schema=UpsertSoundTrackArgs,
            description=(
                "Create or update a SoundTrack, the reusable audible layer / "
                "timeline lane. Reuse an existing track for the same sound identity. "
                "Provide canonical_key as stable snake_case acoustic identity. "
                "Returns id=<uuid> for later SoundEvent calls."
            ),
        ),
        StructuredTool.from_function(
            _make_delete_sound_track_tool(editor),
            args_schema=DeleteSoundTrackArgs,
            description=(
                "Delete a SoundTrack. Fails while SoundEvents still reference it."
            ),
        ),
        StructuredTool.from_function(
            _make_upsert_sound_event_tool(editor),
            args_schema=UpsertSoundEventArgs,
            description=(
                "Create or update one audible occurrence on an existing "
                "SoundTrack. For discrete contacts or transients, write one "
                "tight SoundEvent per beat instead of one broad repeated-action "
                "summary. Use as soon as a sound interval is known and before "
                "moving to later scenes."
            ),
        ),
        StructuredTool.from_function(
            _make_delete_sound_event_tool(editor),
            args_schema=DeleteSoundEventArgs,
            description="Delete a SoundEvent from the SoundTimeline.",
        ),
    ]


def _tool_string(value: object) -> str:
    if isinstance(value, SoundSource):
        visual_source = (
            ""
            if value.visual_object_id is None
            else f" visual_object_id={value.visual_object_id}"
        )
        return _cap_tool_string(
            f"sound_source id={value.sound_source_id} "
            f"type={value.source_type}{visual_source} label={value.label}"
        )
    if isinstance(value, SoundEvent):
        return _cap_tool_string(_sound_event_tool_string(value))
    if isinstance(value, SoundTrack):
        source = (
            ""
            if value.sound_source_id is None
            else f" source_id={value.sound_source_id}"
        )
        return _cap_tool_string(
            f"sound_track id={value.sound_track_id} "
            f"type={value.track_type} model={value.generation_model}"
            + (
                ""
                if value.canonical_key is None
                else f" canonical_key={value.canonical_key}"
            )
            + f"{source} "
            f"label={value.label}"
        )
    if isinstance(value, SchemaModel):
        return _cap_tool_string(value.to_tool_string())
    return _cap_tool_string(json.dumps(value, indent=2, default=str))


def _sound_event_tool_string(event: SoundEvent) -> str:
    return (
        f"sound_event id={event.sound_event_id} track_id={event.sound_track_id} "
        f"frames={event.start_frame_index}-{event.end_frame_index} "
        f"description={event.description}"
        + (f" spoken_text={event.spoken_text}" if event.spoken_text is not None else "")
    )


def _cap_tool_string(value: str) -> str:
    if len(value) <= MAX_TOOL_STRING_CHARS:
        return value
    return (
        value[:MAX_TOOL_STRING_CHARS]
        + "\n[truncated: narrow query/time range; tool output cap reached]"
    )


def _make_list_scenes_tool(editor: SoundTimelineEditor) -> Callable[[int, int], str]:
    def list_scenes(start_scene_index: int = 0, limit: int = 3) -> str:
        return _tool_string(editor.list_scenes(start_scene_index, limit))

    return list_scenes


def _make_get_scene_summary_tool(editor: SoundTimelineEditor) -> Callable[[int], str]:
    def get_scene_summary(scene_index: int) -> str:
        return _tool_string(editor.get_scene_summary(scene_index))

    return get_scene_summary


def _make_list_tracks_tool(editor: SoundTimelineEditor) -> Callable[[int], str]:
    def list_tracks(scene_index: int) -> str:
        return _tool_string(editor.list_tracks(scene_index))

    return list_tracks


def _make_get_visual_events_tool(
    editor: SoundTimelineEditor,
) -> Callable[[int | None, int | None, int], str]:
    def get_visual_events(
        start_frame_index: int | None = None,
        end_frame_index: int | None = None,
        limit: int = 20,
    ) -> str:
        return _tool_string(
            editor.get_visual_events(
                start_frame_index,
                end_frame_index,
                limit,
            )
        )

    return get_visual_events


def _make_get_sound_timeline_tool(
    editor: SoundTimelineEditor,
) -> Callable[[int | None, int | None, int], str]:
    def get_sound_timeline(
        start_frame_index: int | None = None,
        end_frame_index: int | None = None,
        limit: int = 20,
    ) -> str:
        return _tool_string(
            editor.get_sound_timeline(
                start_frame_index,
                end_frame_index,
                limit,
            )
        )

    return get_sound_timeline


def _make_list_sound_sources_tool(
    editor: SoundTimelineEditor,
) -> Callable[[str | None, int], str]:
    def list_sound_sources(query: str | None = None, limit: int = 3) -> str:
        return _tool_string(editor.list_sound_sources(query, limit))

    return list_sound_sources


def _make_list_sound_tracks_tool(
    editor: SoundTimelineEditor,
) -> Callable[[str | None, int], str]:
    def list_sound_tracks(query: str | None = None, limit: int = 3) -> str:
        return _tool_string(editor.list_sound_tracks(query, limit))

    return list_sound_tracks


def _make_upsert_sound_source_tool(
    editor: SoundTimelineEditor,
) -> Callable[
    [SoundSourceType, str, UUID | None, UUID | None, str | None],
    str,
]:
    def upsert_sound_source(
        source_type: SoundSourceType,
        label: str,
        sound_source_id: UUID | None = None,
        visual_object_id: UUID | None = None,
        notes: str | None = None,
    ) -> str:
        return _tool_string(
            editor.upsert_sound_source(
                source_type=source_type,
                label=label,
                sound_source_id=sound_source_id,
                visual_object_id=visual_object_id,
                notes=notes,
            )
        )

    return upsert_sound_source


def _make_delete_sound_source_tool(
    editor: SoundTimelineEditor,
) -> Callable[[UUID], str]:
    def delete_sound_source(sound_source_id: UUID) -> str:
        return _tool_string(editor.delete_sound_source(sound_source_id))

    return delete_sound_source


def _make_upsert_sound_track_tool(
    editor: SoundTimelineEditor,
) -> Callable[
    [
        SoundTrackType,
        str,
        str | None,
        UUID | None,
        UUID | None,
        SoundGenerationModel,
        str | None,
    ],
    str,
]:
    def upsert_sound_track(
        track_type: SoundTrackType,
        label: str,
        canonical_key: str | None = None,
        sound_track_id: UUID | None = None,
        sound_source_id: UUID | None = None,
        generation_model: SoundGenerationModel = "t2a",
        notes: str | None = None,
    ) -> str:
        return _tool_string(
            editor.upsert_sound_track(
                track_type=track_type,
                label=label,
                canonical_key=canonical_key,
                sound_track_id=sound_track_id,
                sound_source_id=sound_source_id,
                generation_model=generation_model,
                notes=notes,
            )
        )

    return upsert_sound_track


def _make_delete_sound_track_tool(editor: SoundTimelineEditor) -> Callable[[UUID], str]:
    def delete_sound_track(sound_track_id: UUID) -> str:
        return _tool_string(editor.delete_sound_track(sound_track_id))

    return delete_sound_track


def _make_upsert_sound_event_tool(
    editor: SoundTimelineEditor,
) -> Callable[[int, int, str, UUID, str | None, UUID | None, str | None], str]:
    def upsert_sound_event(
        start_frame_index: int,
        end_frame_index: int,
        description: str,
        sound_track_id: UUID,
        spoken_text: str | None = None,
        sound_event_id: UUID | None = None,
        notes: str | None = None,
    ) -> str:
        return _tool_string(
            editor.upsert_sound_event(
                start_frame_index=start_frame_index,
                end_frame_index=end_frame_index,
                description=description,
                sound_track_id=sound_track_id,
                spoken_text=spoken_text,
                sound_event_id=sound_event_id,
                notes=notes,
            )
        )

    return upsert_sound_event


def _make_delete_sound_event_tool(editor: SoundTimelineEditor) -> Callable[[UUID], str]:
    def delete_sound_event(sound_event_id: UUID) -> str:
        return _tool_string(editor.delete_sound_event(sound_event_id))

    return delete_sound_event


def _make_annotated_frame_tool(
    editor: SoundTimelineEditor,
) -> Callable[[int, FrameResolutionMode], list[ToolMessageContentBlock]]:
    def get_annotated_frame_message(
        frame_index: int,
        resolution_mode: FrameResolutionMode = "low",
    ) -> list[ToolMessageContentBlock]:
        output = editor.get_annotated_frame(frame_index, resolution_mode)
        return _to_annotated_frame_message_blocks(output)

    return get_annotated_frame_message


def _to_annotated_frame_message_blocks(
    output: AnnotatedFrameOutput,
) -> list[ToolMessageContentBlock]:
    track_count = len(output.tracks)
    track_labels = disambiguate_labels([track.display_label for track in output.tracks])
    return [
        {
            "type": "text",
            "text": (
                f"Annotated frame {output.frame_index} with "
                f"{track_count} visible tracks."
            ),
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{output.image}"},
        },
        {
            "type": "json",
            "json": {
                "frame_index": output.frame_index,
                "resolution_mode": output.resolution_mode,
                "width": output.width,
                "height": output.height,
                "tracks": [
                    {
                        "scene_index": track.scene_index,
                        "label": label,
                        "bbox_xyxy": _rounded_bbox(track.bbox_xyxy),
                        "confidence": round(track.confidence, 2),
                    }
                    for label, track in zip(
                        track_labels,
                        output.tracks,
                        strict=True,
                    )
                ],
            },
        },
    ]


def _rounded_bbox(
    bbox_xyxy: tuple[float, float, float, float] | None,
) -> tuple[int, int, int, int] | None:
    if bbox_xyxy is None:
        return None
    return (
        round(bbox_xyxy[0]),
        round(bbox_xyxy[1]),
        round(bbox_xyxy[2]),
        round(bbox_xyxy[3]),
    )
