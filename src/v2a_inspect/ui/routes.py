from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import ValidationError

from v2a_inspect.media_utils import probe_prepared_video
from v2a_inspect.models import VideoAsset
from v2a_inspect.preprocessing import prepare_video

from .overlays import render_tracking_overlay
from .pipeline import (
    PipelineOptions,
    run_sound_timeline_pipeline,
    run_uploaded_video_pipeline,
    run_audio_generation_pipeline,
)
from .rows import current_frame_rows, timeline_rows, tracking_window_rows
from .store import VideoAssetSnapshot, VideoAssetStore


DEFAULT_WORK_DIR = os.getenv("V2A_INSPECT_UI_WORK_DIR")
DEFAULT_SERVER_URL = os.getenv("V2A_INSPECT_UI_SERVER_URL")
DEFAULT_SCENE_THRESHOLD = float(os.getenv("V2A_INSPECT_UI_SCENE_THRESHOLD", "27.0"))
DEFAULT_MAX_KEYFRAMES_PER_SCENE = int(
    os.getenv("V2A_INSPECT_UI_MAX_KEYFRAMES_PER_SCENE", "20")
)


def create_router(store: VideoAssetStore) -> APIRouter:
    router = APIRouter()

    @router.get("/api/asset")
    async def get_asset() -> dict[str, object]:
        snapshot = await store.snapshot()
        return _full_snapshot_payload(snapshot)

    @router.get("/api/asset/export")
    async def export_asset() -> Response:
        snapshot = await store.snapshot()
        if snapshot.asset is None:
            raise HTTPException(status_code=404, detail="No video asset loaded")
        return Response(
            snapshot.asset.model_dump_json(
                indent=2,
                exclude_computed_fields=True,
            ),
            media_type="application/json",
            headers={
                "Content-Disposition": 'attachment; filename="video-asset.json"',
            },
        )

    @router.get("/api/asset-summary")
    async def get_asset_summary() -> dict[str, object]:
        snapshot = await store.snapshot()
        return _summary_payload(snapshot)

    @router.post("/api/asset/clean")
    async def clean_asset() -> dict[str, object]:
        snapshot = await store.snapshot()
        if snapshot.status == "running":
            raise HTTPException(status_code=409, detail="Pipeline is running")
        cleaned = (
            None
            if snapshot.asset is None
            else await asyncio.to_thread(_delete_asset_work_dir, snapshot.asset)
        )
        await store.clear()
        return {
            "status": "cleaned",
            "deleted": None if cleaned is None else str(cleaned),
        }

    @router.get("/api/video")
    async def get_video() -> FileResponse:
        snapshot = await store.snapshot()
        if snapshot.asset is None:
            raise HTTPException(status_code=404, detail="No video asset loaded")
        return FileResponse(snapshot.asset.source_path)

    @router.get("/api/synthesized-video")
    async def get_synthesized_video() -> FileResponse:
        snapshot = await store.snapshot()
        if (
            snapshot.asset is None
            or not snapshot.asset.synthesized_video_path
            or not snapshot.asset.synthesized_video_path.exists()
        ):
            raise HTTPException(
                status_code=404, detail="No synthesized video available"
            )
        return FileResponse(
            snapshot.asset.synthesized_video_path,
            media_type="video/mp4",
            headers={"Accept-Ranges": "bytes"},
        )

    @router.get("/api/audio/tracks/{sound_track_id}")
    async def get_audio_track(sound_track_id: str) -> FileResponse:
        snapshot = await store.snapshot()
        if snapshot.asset is None:
            raise HTTPException(status_code=404, detail="No video asset loaded")
        artifact = next(
            (
                item
                for item in snapshot.asset.sound_track_audio_artifacts
                if str(item.sound_track_id) == sound_track_id
            ),
            None,
        )
        if artifact is None:
            raise HTTPException(status_code=404, detail="No track audio available")
        return _audio_file_response(artifact.path, "No track audio file available")

    @router.get("/api/audio/events/{sound_event_id}")
    async def get_audio_event(sound_event_id: str) -> FileResponse:
        snapshot = await store.snapshot()
        if snapshot.asset is None:
            raise HTTPException(status_code=404, detail="No video asset loaded")
        artifact = next(
            (
                item
                for item in snapshot.asset.sound_event_audio_artifacts
                if str(item.sound_event_id) == sound_event_id
            ),
            None,
        )
        if artifact is None:
            raise HTTPException(status_code=404, detail="No event audio available")
        return _audio_file_response(artifact.path, "No event audio file available")

    @router.get("/api/rows/timeline")
    async def get_timeline_rows() -> dict[str, object]:
        snapshot = await store.snapshot()
        rows = [] if snapshot.asset is None else timeline_rows(snapshot.asset)
        return {"version": snapshot.version, "rows": rows}

    @router.get("/api/rows/current-frame")
    async def get_current_frame(frame: int) -> dict[str, object]:
        snapshot = await store.snapshot()
        if snapshot.asset is None:
            return {"version": snapshot.version, "frame": frame, "rows": None}
        return {
            "version": snapshot.version,
            "frame": frame,
            "rows": current_frame_rows(snapshot.asset, frame),
        }

    @router.get("/api/tracks/window")
    async def get_tracking_window(
        start_frame: int, end_frame: int
    ) -> dict[str, object]:
        snapshot = await store.snapshot()
        if snapshot.asset is None:
            return {
                "version": snapshot.version,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "tracks": [],
            }
        start = max(0, start_frame)
        end = min(snapshot.asset.frame_count - 1, end_frame)
        if end < start:
            raise HTTPException(status_code=400, detail="Invalid frame window")
        return {
            "version": snapshot.version,
            "start_frame": start,
            "end_frame": end,
            "tracks": tracking_window_rows(snapshot.asset, start, end),
        }

    @router.get("/api/frames/tracking-overlay")
    async def get_tracking_overlay(frame: int) -> Response:
        snapshot = await store.snapshot()
        if snapshot.asset is None:
            raise HTTPException(status_code=404, detail="No video asset loaded")
        if frame < 0 or frame >= snapshot.asset.frame_count:
            raise HTTPException(status_code=400, detail="Frame out of range")
        return Response(
            render_tracking_overlay(snapshot.asset, frame),
            media_type="image/png",
            headers={"Cache-Control": "no-store"},
        )

    @router.get("/api/keyframes/{keyframe_id}")
    async def get_keyframe(keyframe_id: str) -> FileResponse:
        snapshot = await store.snapshot()
        if snapshot.asset is None:
            raise HTTPException(status_code=404, detail="No video asset loaded")
        path = _keyframe_response_path(snapshot.asset, keyframe_id)
        if path is None:
            raise HTTPException(status_code=404, detail="No keyframe image available")
        return FileResponse(
            path,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    @router.post("/api/runs")
    async def start_run(
        background_tasks: BackgroundTasks,
        video: Annotated[UploadFile, File()],
        work_dir: Annotated[str | None, Form()] = None,
        server_url: Annotated[str | None, Form()] = None,
        scene_threshold: Annotated[float, Form()] = DEFAULT_SCENE_THRESHOLD,
        max_keyframes_per_scene: Annotated[
            int, Form()
        ] = DEFAULT_MAX_KEYFRAMES_PER_SCENE,
    ) -> dict[str, object]:
        if not video.filename:
            raise HTTPException(status_code=400, detail="No video filename provided")

        root_dir = _root_dir(work_dir)
        upload_path = _write_upload(root_dir, video.filename, await video.read())

        options = PipelineOptions(
            scene_threshold=scene_threshold,
            max_keyframes_per_scene=max_keyframes_per_scene,
            server_url=server_url or DEFAULT_SERVER_URL or None,
        )
        background_tasks.add_task(
            run_uploaded_video_pipeline,
            upload_path,
            root_dir,
            store,
            options,
        )
        await store.set_running(stage="queued")
        return {"status": "queued"}

    @router.post("/api/asset/import")
    async def import_asset(
        video: Annotated[UploadFile, File()],
        asset: Annotated[UploadFile, File()],
        work_dir: Annotated[str | None, Form()] = None,
    ) -> dict[str, object]:
        snapshot = await store.snapshot()
        if snapshot.status == "running":
            return {"status": "ignored", "detail": "Pipeline is already running"}
        if not video.filename:
            raise HTTPException(status_code=400, detail="No video filename provided")
        if not asset.filename:
            raise HTTPException(status_code=400, detail="No asset filename provided")

        try:
            imported_asset = VideoAsset.model_validate_json(await asset.read())
        except ValidationError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid VideoAsset JSON: {exc}",
            ) from exc

        root_dir = _root_dir(work_dir)
        upload_path = _write_upload(
            root_dir,
            video.filename,
            await video.read(),
        )
        try:
            resolved_video_asset = await _resolve_import_video(upload_path, root_dir)
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid video: {exc}",
            ) from exc
        if resolved_video_asset.frame_count != imported_asset.frame_count:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Imported video frame count does not match VideoAsset: "
                    f"{resolved_video_asset.frame_count} != "
                    f"{imported_asset.frame_count}"
                ),
            )

        sam3_tracking_path = resolved_video_asset.sam3_tracking_path
        if sam3_tracking_path is not None and not sam3_tracking_path.exists():
            sam3_tracking_path = None
        imported_asset = imported_asset.model_copy(
            update={
                "source_path": resolved_video_asset.source_path,
                "sam3_tracking_path": sam3_tracking_path,
            }
        )
        await store.set_complete_asset(imported_asset, stage="imported asset")
        return {"status": "imported"}

    @router.post("/api/sound-timeline/reset-run")
    async def reset_sound_timeline(
        background_tasks: BackgroundTasks,
    ) -> dict[str, object]:
        snapshot = await store.snapshot()
        if snapshot.status == "running":
            return {"status": "ignored", "detail": "Pipeline is already running"}
        if snapshot.asset is None:
            raise HTTPException(status_code=404, detail="No video asset loaded")
        video_asset = snapshot.asset.model_copy(
            update={
                "sound_timeline": None,
                "synthesized_video_path": None,
                "sound_event_audio_artifacts": [],
                "sound_track_audio_artifacts": [],
            }
        )
        await store.publish_asset_mutation(
            video_asset,
            stage="reset sound timeline",
        )
        background_tasks.add_task(run_sound_timeline_pipeline, video_asset, store)
        return {"status": "queued"}

    @router.post("/api/audio/generate")
    async def generate_audio(
        background_tasks: BackgroundTasks,
        server_url: Annotated[str | None, Form()] = None,
        asset: Annotated[UploadFile | None, File()] = None,
    ) -> dict[str, object]:
        snapshot = await store.snapshot()
        if snapshot.status == "running":
            return {"status": "ignored", "detail": "Pipeline is already running"}

        if asset is not None:
            try:
                import_asset = VideoAsset.model_validate_json(await asset.read())
                await store.set_complete_asset(
                    import_asset, stage="applied timeline edits"
                )
                video_asset = import_asset
            except ValidationError as exc:
                raise HTTPException(
                    status_code=400, detail=f"Invalid VideoAsset JSON: {exc}"
                ) from exc
        else:
            if snapshot.asset is None:
                raise HTTPException(status_code=404, detail="No video asset loaded")
            video_asset = snapshot.asset

        await store.publish_asset_mutation(
            video_asset,
            stage="queued audio generation",
        )
        background_tasks.add_task(
            run_audio_generation_pipeline,
            video_asset,
            store,
            server_url or DEFAULT_SERVER_URL or None,
        )
        return {"status": "queued"}

    @router.get("/events")
    async def events() -> StreamingResponse:
        async def event_stream():
            snapshot = await store.snapshot()
            yield _sse(snapshot)
            version = snapshot.version
            while True:
                snapshot = await store.wait_for_change(version)
                version = snapshot.version
                yield _sse(snapshot)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return router


def _root_dir(work_dir: str | None) -> Path:
    root_dir = Path(work_dir or DEFAULT_WORK_DIR or tempfile.gettempdir())
    if root_dir.name != "v2a-inspect-ui":
        root_dir /= "v2a-inspect-ui"
    return root_dir


def _write_upload(root_dir: Path, filename: str, data: bytes) -> Path:
    upload_dir = root_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    upload_path = upload_dir / f"{uuid.uuid4()}{Path(filename).suffix}"
    upload_path.write_bytes(data)
    return upload_path


async def _resolve_import_video(upload_path: Path, root_dir: Path) -> VideoAsset:
    try:
        probe = probe_prepared_video(upload_path)
    except Exception:
        return await asyncio.to_thread(prepare_video, upload_path, root_dir)
    return VideoAsset(
        source_path=probe.path,
        frame_count=probe.frame_count,
    )


def _summary_payload(snapshot: VideoAssetSnapshot) -> dict[str, object]:
    asset = snapshot.asset
    return {
        "status": snapshot.status,
        "stage": snapshot.current_stage,
        "error": snapshot.error,
        "version": snapshot.version,
        "asset_version": snapshot.asset_version,
        "updated_at": snapshot.updated_at.isoformat(),
        "video": None
        if asset is None
        else {
            "video_id": str(asset.video_id),
            "frame_count": asset.frame_count,
            "fps": asset.fps,
            "duration_sec": asset.duration_sec,
            "width": asset.width,
            "height": asset.height,
        },
        "timeline_rows": [] if asset is None else timeline_rows(asset),
        "audio_tracks": []
        if asset is None
        else [
            item.model_dump(mode="json")
            for item in asset.sound_track_audio_artifacts
        ],
        "audio_events": []
        if asset is None
        else [
            item.model_dump(mode="json")
            for item in asset.sound_event_audio_artifacts
        ],
    }


def _audio_file_response(path: Path, missing_detail: str) -> FileResponse:
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail=missing_detail)
    return FileResponse(
        path,
        media_type="audio/wav",
        headers={"Accept-Ranges": "bytes"},
    )


def _keyframe_response_path(asset: VideoAsset, keyframe_id: str) -> Path | None:
    for scene in asset.initial_scenes:
        for keyframe in scene.keyframes:
            if str(keyframe.keyframe_id) != keyframe_id:
                continue
            try:
                path = keyframe.image_path.resolve(strict=True)
                root = asset.source_path.resolve(strict=True).parent
            except OSError:
                return None
            if path.suffix.lower() not in {".jpg", ".jpeg"}:
                return None
            if not path.is_file() or not path.is_relative_to(root):
                return None
            return path
    return None


def _delete_asset_work_dir(asset: VideoAsset) -> Path | None:
    root = _asset_work_root(asset)
    if root is None or not root.exists():
        return None
    shutil.rmtree(root)
    return root


def _asset_work_root(asset: VideoAsset) -> Path | None:
    root = asset.source_path.parent
    if root.name == "uploads":
        root = root.parent
    root = root.resolve()
    if root.name != "v2a-inspect-ui":
        return None
    return root


def _full_snapshot_payload(snapshot: VideoAssetSnapshot) -> dict[str, object]:
    payload = _summary_payload(snapshot)
    payload["asset"] = (
        None
        if snapshot.asset is None
        else snapshot.asset.model_dump(mode="json", exclude_computed_fields=True)
    )
    return payload


def _sse(snapshot: VideoAssetSnapshot) -> str:
    data = {
        "version": snapshot.version,
        "asset_version": snapshot.asset_version,
        "status": snapshot.status,
        "stage": snapshot.current_stage,
        "error": snapshot.error,
    }
    return f"event: asset_update\ndata: {json.dumps(data)}\n\n"
