from __future__ import annotations

import asyncio
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from moviepy import AudioFileClip
from scipy.io import wavfile

from v2a_inspect.agents.sound_timeline import run_sound_timeline_agent_parallel
from v2a_inspect.client import SAM3Client, VideoClient
from v2a_inspect.config import settings
from v2a_inspect.models import (
    AudioPlan,
    AudioPlanItem,
    InitialScene,
    SoundEventAudioArtifact,
    SoundTrackAudioArtifact,
    VideoAsset,
)
from v2a_inspect.audio_generation.client import generate_audio_for_item
from v2a_inspect.audio_generation.mix import mix_audio_into_video, mix_audio_items_to_wav
from v2a_inspect.preprocessing import (
    analyze_initial_scene,
    build_visual_identity_layer,
    compute_visual_events,
    detect_initial_scenes,
    extract_keyframes_for_initial_scene,
    prepare_video,
    sam3_tracking_video_path,
    track_initial_scenes_object_seeds,
)

from .store import VideoAssetStore


@dataclass(frozen=True)
class PipelineOptions:
    scene_threshold: float = 27.0
    max_keyframes_per_scene: int = 20
    server_url: str | None = None


async def run_uploaded_video_pipeline(
    upload_path: Path,
    work_dir: Path,
    store: VideoAssetStore,
    options: PipelineOptions,
) -> None:
    """Run the full pipeline and publish each completed VideoAsset stage."""

    try:
        loop = asyncio.get_running_loop()
        await store.set_running(stage="prepare video")
        video_asset = await asyncio.to_thread(prepare_video, upload_path, work_dir)
        await store.set_asset(video_asset, stage="prepared video")

        await store.touch(stage="detect scenes")
        scenes = await asyncio.to_thread(
            detect_initial_scenes,
            video_asset,
            threshold=options.scene_threshold,
        )
        video_asset = video_asset.model_copy(update={"initial_scenes": scenes})
        await store.publish_asset_mutation(video_asset, stage="detected scenes")

        await store.touch(stage="extract keyframes")
        video_asset = await _extract_keyframes_incrementally(
            video_asset,
            work_dir,
            store,
            max_keyframes_per_scene=options.max_keyframes_per_scene,
        )
        await store.publish_asset_mutation(video_asset, stage="extracted keyframes")

        await store.touch(stage="analyze scenes")
        video_asset = await _analyze_scenes_incrementally(
            video_asset,
            store,
        )
        await store.publish_asset_mutation(video_asset, stage="analyzed scenes")

        await store.touch(stage="track objects")
        video_asset = await _track_objects(video_asset, options.server_url, store)
        await store.publish_asset_mutation(video_asset, stage="tracked objects")

        await store.touch(stage="build visual identity")
        video_asset = await asyncio.to_thread(build_visual_identity_layer, video_asset)
        await store.publish_asset_mutation(video_asset, stage="built visual identity")

        await store.touch(stage="compute visual events")
        video_asset = await asyncio.to_thread(compute_visual_events, video_asset)
        await store.publish_asset_mutation(video_asset, stage="computed visual events")

        await store.touch(stage="build sound timeline")
        on_sound_change = _threadsafe_publish_callback(loop, store, video_asset)
        await asyncio.to_thread(
            run_sound_timeline_agent_parallel,
            video_asset,
            segment_seconds=settings.agent_sound_timeline_segment_seconds,
            max_workers=settings.agent_sound_timeline_max_workers,
            on_change=on_sound_change,
        )
        await store.publish_asset_mutation(video_asset, stage="built sound timeline")

        await store.set_complete()
    except Exception as exc:  # noqa: BLE001 - UI needs stage-specific failure text.
        await store.set_error(str(exc))


async def run_sound_timeline_pipeline(
    video_asset: VideoAsset,
    store: VideoAssetStore,
) -> None:
    """Rebuild only the final SoundTimeline stage for an existing VideoAsset."""

    try:
        loop = asyncio.get_running_loop()
        await store.touch(stage="build sound timeline")
        on_sound_change = _threadsafe_publish_callback(loop, store, video_asset)
        await asyncio.to_thread(
            run_sound_timeline_agent_parallel,
            video_asset,
            segment_seconds=settings.agent_sound_timeline_segment_seconds,
            max_workers=settings.agent_sound_timeline_max_workers,
            on_change=on_sound_change,
        )
        await store.publish_asset_mutation(video_asset, stage="built sound timeline")
        await store.set_complete()
    except Exception as exc:  # noqa: BLE001 - UI needs stage-specific failure text.
        await store.set_error(str(exc))


async def _extract_keyframes_incrementally(
    video_asset: VideoAsset,
    work_dir: Path,
    store: VideoAssetStore,
    *,
    max_keyframes_per_scene: int,
) -> VideoAsset:
    updated_scenes = list(video_asset.initial_scenes)
    for scene_index, scene in enumerate(video_asset.initial_scenes):
        keyframes = await asyncio.to_thread(
            extract_keyframes_for_initial_scene,
            video_asset,
            scene,
            work_dir,
            max_keyframes_per_scene,
        )
        updated_scenes[scene_index] = scene.model_copy(update={"keyframes": keyframes})
        video_asset = video_asset.model_copy(update={"initial_scenes": updated_scenes})
        await store.publish_asset_mutation(
            video_asset,
            stage=f"extracted keyframes {scene_index + 1}/{len(updated_scenes)}",
        )
    return video_asset


async def _analyze_scenes_incrementally(
    video_asset: VideoAsset,
    store: VideoAssetStore,
) -> VideoAsset:
    updated_scenes = list(video_asset.initial_scenes)
    completed = 0
    batch_size = settings.llm_initial_scene_analysis_batch_size
    for batch_start in range(0, len(updated_scenes), batch_size):
        batch = updated_scenes[batch_start : batch_start + batch_size]
        tasks = [
            asyncio.create_task(_analyze_scene(batch_start + offset, scene))
            for offset, scene in enumerate(batch)
        ]
        for task in asyncio.as_completed(tasks):
            scene_index, analyzed_scene = await task
            updated_scenes[scene_index] = analyzed_scene
            completed += 1
            video_asset = video_asset.model_copy(
                update={"initial_scenes": updated_scenes}
            )
            await store.publish_asset_mutation(
                video_asset,
                stage=f"analyzed scenes {completed}/{len(updated_scenes)}",
            )
    return video_asset


async def _analyze_scene(
    scene_index: int,
    scene: InitialScene,
) -> tuple[int, InitialScene]:
    analyzed_scene = await asyncio.to_thread(analyze_initial_scene, scene)
    return scene_index, analyzed_scene


async def _track_objects(
    video_asset: VideoAsset,
    server_url: str | None,
    store: VideoAssetStore,
) -> VideoAsset:
    tracking_video_path = sam3_tracking_video_path(video_asset)
    async with VideoClient(base_url=server_url) as video_client:
        try:
            upload_response = await video_client.upload(str(tracking_video_path))
        except Exception as exc:
            raise RuntimeError(f"upload tracking video failed: {exc}") from exc
    async with SAM3Client(base_url=server_url) as sam_client:
        for scene_index in range(len(video_asset.initial_scenes)):
            scene_number = scene_index + 1
            scene_count = len(video_asset.initial_scenes)
            try:
                video_asset = await track_initial_scenes_object_seeds(
                    video_asset,
                    video_id=upload_response.video_id,
                    sam_client=sam_client,
                    scene_indexes=[scene_index],
                )
            except Exception as exc:
                raise RuntimeError(
                    f"track objects scene {scene_number}/{scene_count} failed: {exc}"
                ) from exc
            video_asset = await _rebuild_visual_layers(video_asset)
            await store.publish_asset_mutation(
                video_asset,
                stage=f"tracked objects {scene_number}/{scene_count}",
            )
    return video_asset


async def _rebuild_visual_layers(video_asset: VideoAsset) -> VideoAsset:
    video_asset = await asyncio.to_thread(build_visual_identity_layer, video_asset)
    return await asyncio.to_thread(compute_visual_events, video_asset)


def _threadsafe_publish_callback(
    loop: asyncio.AbstractEventLoop,
    store: VideoAssetStore,
    video_asset: VideoAsset,
) -> Callable[[str], None]:
    def publish(stage: str) -> None:
        future = asyncio.run_coroutine_threadsafe(
            store.publish_asset_mutation(video_asset, stage=stage),
            loop,
        )
        future.result()

    return publish



async def run_audio_generation_pipeline(
    video_asset: VideoAsset,
    store: VideoAssetStore,
    server_url: str | None,
) -> None:
    try:
        timeline = video_asset.sound_timeline
        if not timeline:
            raise ValueError("No sound timeline found in video asset.")
        if not timeline.sound_events:
            raise ValueError("No audio items to generate.")

        await store.set_running(stage="preparing audio plan")
        video_asset = video_asset.model_copy(
            update={
                "synthesized_video_path": None,
                "sound_event_audio_artifacts": [],
                "sound_track_audio_artifacts": [],
            }
        )
        await store.publish_asset_mutation(video_asset, stage="preparing audio plan")

        video_duration = video_asset.duration_sec
        fps = video_asset.fps
        work_root = _audio_work_root(video_asset)
        event_dir = work_root / "audio" / "events"
        track_dir = work_root / "audio" / "tracks"
        event_dir.mkdir(parents=True, exist_ok=True)
        track_dir.mkdir(parents=True, exist_ok=True)

        audio_plan = AudioPlan(total_duration=video_duration)
        track_map = {track.sound_track_id: track for track in timeline.sound_tracks}
        item_context: dict[str, tuple[object, object]] = {}

        await store.touch(stage="uploading video to inference server")
        async with VideoClient(base_url=server_url) as video_client:
            try:
                res = await video_client.upload(str(video_asset.source_path))
                video_id = res.video_id
            except Exception:
                video_id = "dummy"

        for event in timeline.sound_events:
            track = track_map.get(event.sound_track_id)
            if not track:
                continue

            start_time = event.start_frame_index / fps
            end_time = event.end_frame_index / fps
            start_time = max(0.0, min(start_time, video_duration - 0.1))
            end_time = max(0.0, min(end_time, video_duration))
            if end_time <= start_time:
                end_time = start_time + 0.1

            source_label = ""
            if track.sound_source_id:
                for source in timeline.sound_sources:
                    if source.sound_source_id == track.sound_source_id:
                        source_label = source.label
                        break

            if source_label and source_label.lower() not in track.label.lower():
                desc = f"{source_label}, [{track.label}] {event.description}"
            else:
                desc = f"[{track.label}] {event.description}"

            gen_mode = track.generation_mode
            vol = 1.0
            if gen_mode == "vta":
                gen_model = "v2a"
                vol = 1.5
            elif gen_mode == "tta":
                gen_model = "t2a"
                vol = 0.8
            else:
                gen_model = gen_mode

            item = AudioPlanItem(
                item_id=str(event.sound_event_id),
                type=track.track_type,
                time=(start_time, end_time),
                description=desc,
                spoken_text=getattr(event, "spoken_text", None),
                volume=vol,
                track_id=str(track.sound_track_id),
                generation_model=gen_model,
            )
            audio_plan.items.append(item)
            item_context[item.item_id] = (event, track)

        audio_plan.items.sort(key=lambda x: x.time[0])
        n_items = len(audio_plan.items)

        generated_audio: dict[str, str] = {}
        event_artifacts: list[SoundEventAudioArtifact] = []

        for i, item in enumerate(audio_plan.items, 1):
            context = item_context.get(item.item_id)
            if context is None:
                continue
            event, track = context
            await store.touch(stage=f"generating audio {i}/{n_items}")
            duration = item.time[1] - item.time[0]
            event_path = _event_audio_path(event_dir, item.item_id, track.label)
            raw_path = event_path.with_name(f".{event_path.stem}.generated.wav")

            audio_file = await asyncio.to_thread(
                generate_audio_for_item,
                kind=item.type,
                description=item.description,
                out_path=str(raw_path),
                duration=duration,
                video_id=video_id,
                fps=fps,
                time=item.time,
                generation_model=item.generation_model,
                spoken_text=item.spoken_text,
                server_url=server_url,
            )
            if not audio_file:
                continue

            normalized = await asyncio.to_thread(
                _convert_audio_to_wav,
                Path(audio_file),
                event_path,
            )
            if raw_path.exists() and raw_path != event_path:
                raw_path.unlink(missing_ok=True)
            if normalized is None:
                continue

            generated_audio[item.item_id] = str(normalized)
            event_artifacts.append(
                SoundEventAudioArtifact(
                    sound_event_id=event.sound_event_id,
                    sound_track_id=track.sound_track_id,
                    path=normalized,
                    duration_sec=round(duration, 3),
                    generation_model=item.generation_model,
                    description=item.description,
                )
            )

        if not generated_audio:
            raise ValueError("No audio files were generated.")

        await store.touch(stage="mixing track stems")
        track_items: dict[str, list[AudioPlanItem]] = defaultdict(list)
        for item in audio_plan.items:
            if item.track_id and item.item_id in generated_audio:
                track_items[item.track_id].append(item)

        track_artifacts: list[SoundTrackAudioArtifact] = []
        for track in timeline.sound_tracks:
            items = track_items.get(str(track.sound_track_id), [])
            if not items:
                continue
            track_path = _track_audio_path(track_dir, str(track.sound_track_id), track.label)
            stem_path = await asyncio.to_thread(
                mix_audio_items_to_wav,
                items,
                generated_audio,
                str(track_path),
                total_duration=video_duration,
            )
            if not stem_path or not Path(stem_path).exists():
                continue
            peaks = await asyncio.to_thread(_waveform_peaks, Path(stem_path))
            track_artifacts.append(
                SoundTrackAudioArtifact(
                    sound_track_id=track.sound_track_id,
                    track_label=track.label,
                    track_type=track.track_type,
                    path=Path(stem_path),
                    duration_sec=round(video_duration, 3),
                    event_count=len(items),
                    waveform_peaks=peaks,
                )
            )

        if not track_artifacts:
            raise RuntimeError("No track stems were generated.")

        await store.touch(stage="mixing audio")
        output_path = str(work_root / "preview.mp4")
        preview_plan = AudioPlan(total_duration=video_duration)
        preview_audio: dict[str, str] = {}
        for artifact in track_artifacts:
            item_id = str(artifact.sound_track_id)
            preview_plan.items.append(
                AudioPlanItem(
                    item_id=item_id,
                    type=artifact.track_type,
                    time=(0.0, video_duration),
                    description=artifact.track_label,
                    volume=1.0,
                    track_id=item_id,
                    generation_model="stem",
                )
            )
            preview_audio[item_id] = str(artifact.path)

        result = await asyncio.to_thread(
            mix_audio_into_video,
            video_path=str(video_asset.source_path),
            audio_plan=preview_plan,
            generated_audio=preview_audio,
            output_path=output_path,
            keep_original_audio=False,
        )

        if not result or not Path(result).exists():
            raise RuntimeError("Audio synthesis and mixing failed.")

        video_asset = video_asset.model_copy(
            update={
                "synthesized_video_path": Path(result),
                "sound_event_audio_artifacts": event_artifacts,
                "sound_track_audio_artifacts": track_artifacts,
            }
        )
        await store.set_complete_asset(
            video_asset, stage="Audio generation complete. Ready for preview."
        )
    except Exception as exc:
        await store.set_error(str(exc))


def _audio_work_root(video_asset: VideoAsset) -> Path:
    root = video_asset.source_path.parent
    if root.name == "uploads":
        return root.parent
    return root


def _event_audio_path(audio_dir: Path, event_id: str, track_label: str) -> Path:
    return audio_dir / f"event_{_short_id(event_id)}_{_slug(track_label)}.wav"


def _track_audio_path(audio_dir: Path, track_id: str, track_label: str) -> Path:
    return audio_dir / f"track_{_short_id(track_id)}_{_slug(track_label)}.wav"


def _short_id(value: object) -> str:
    return str(value).replace("-", "")[:12] or "unknown"


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-_.").lower()
    return (slug[:48] or "track")


def _convert_audio_to_wav(input_path: Path, output_path: Path) -> Path | None:
    if not input_path.exists():
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        clip = AudioFileClip(str(input_path))
        try:
            clip.write_audiofile(
                str(output_path),
                fps=44100,
                codec="pcm_s16le",
                logger=None,
            )
        finally:
            clip.close()
        return output_path
    except Exception:
        return None


def _waveform_peaks(path: Path, *, bucket_count: int = 768) -> list[float]:
    try:
        _sample_rate, data = wavfile.read(path)
    except Exception:
        return []
    if data.size == 0:
        return []

    samples = data.astype(np.float32)
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        samples = samples / max(abs(info.min), info.max)
    if samples.ndim > 1:
        samples = np.max(np.abs(samples), axis=1)
    else:
        samples = np.abs(samples)

    buckets = max(1, min(bucket_count, samples.size))
    peaks = [float(chunk.max()) for chunk in np.array_split(samples, buckets)]
    max_peak = max(peaks) if peaks else 0.0
    if max_peak > 0:
        peaks = [peak / max_peak for peak in peaks]
    return [round(min(max(peak, 0.0), 1.0), 4) for peak in peaks]
