"""
V2A Inspect → V2A Audio Synthesis and Video Mix CLI.

Takes a SoundTimeline (timeline.json) and a video file, generates audio
using LLM TTS or specialized models, and mixes it into an output video.

Usage:
  uv run v2a synthesize --timeline timeline.json --video videos/my_video.mp4 --output output.mp4
"""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv
from moviepy import VideoFileClip

from v2a_inspect.audio_generation.client import generate_audio_for_item
from v2a_inspect.audio_generation.mix import mix_audio_into_video
from v2a_inspect.audio_generation.plan import build_audio_plan
from v2a_inspect.models import SoundTimeline, VideoAsset

load_dotenv(override=True)

logger = logging.getLogger(__name__)


def synthesize(
    video: Annotated[
        str,
        typer.Option("--video", help="Path to the original video file."),
    ],
    timeline: Annotated[
        str | None,
        typer.Option(
            "--timeline",
            help="Path to the timeline.json file containing the SoundTimeline.",
        ),
    ] = None,
    asset: Annotated[
        str | None,
        typer.Option(
            "--asset",
            help="Path to the video-asset.json file containing the VideoAsset (overrides --timeline).",
        ),
    ] = None,
    output: Annotated[
        str | None,
        typer.Option(
            "--output",
            "-o",
            help="Output video path (default: same directory as video).",
        ),
    ] = None,
    keep_original_audio: Annotated[
        bool,
        typer.Option(
            "--keep-original-audio",
            help="Keep the original audio of the video (default: replace).",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Detailed logging."),
    ] = False,
) -> None:
    """Generate audio from a SoundTimeline and mix it into a video."""

    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="[%(levelname)s] %(name)s: %(message)s",
    )

    if not timeline and not asset:
        typer.echo("Error: Must provide either --timeline or --asset", err=True)
        raise typer.Exit(code=1)

    sound_timeline: SoundTimeline
    if asset:
        asset_path = Path(asset)
        if not asset_path.exists():
            typer.echo(f"Error: asset json not found: {asset_path}", err=True)
            raise typer.Exit(code=1)
        typer.echo(f"[1/4] Loading VideoAsset: {asset_path}", err=True)

        asset_data = json.loads(asset_path.read_text(encoding="utf-8"))
        video_asset = VideoAsset(**asset_data)
        if video_asset.sound_timeline is None:
            typer.echo("Error: VideoAsset does not contain a sound_timeline", err=True)
            raise typer.Exit(code=1)
        sound_timeline = video_asset.sound_timeline
    else:
        if timeline is None:
            typer.echo("Error: Must provide either --timeline or --asset", err=True)
            raise typer.Exit(code=1)
        timeline_path = Path(timeline)
        if not timeline_path.exists():
            typer.echo(f"Error: timeline.json not found: {timeline_path}", err=True)
            raise typer.Exit(code=1)

        typer.echo(f"[1/4] Loading timeline: {timeline_path}", err=True)
        timeline_data = json.loads(timeline_path.read_text(encoding="utf-8"))
        sound_timeline = SoundTimeline(**timeline_data)

    video_path = video
    if not Path(video_path).exists():
        typer.echo(f"Error: Video not found: {video_path}", err=True)
        raise typer.Exit(code=1)

    typer.echo("[2/4] Converting SoundTimeline to AudioPlan...", err=True)
    try:
        video_clip = VideoFileClip(video_path)
        fps = video_clip.fps
        video_duration = video_clip.duration or 0.0
        video_clip.close()
    except Exception as e:
        typer.echo(f"Error: Could not read video file: {e}", err=True)
        raise typer.Exit(code=1) from e

    if not fps or fps <= 0:
        fps = 30.0

    audio_plan = build_audio_plan(
        sound_timeline,
        fps=fps,
        total_duration=video_duration,
    )
    n_items = len(audio_plan.items)
    typer.echo(f"  → {n_items} audio events scheduled.", err=True)

    if n_items == 0:
        typer.echo("Warning: No audio items to generate.", err=True)
        return

    video_id: str | None = None
    if any(item.generation_model == "v2a" for item in audio_plan.items):
        typer.echo("[3/4] Uploading video for V2A generation...", err=True)
        from v2a_inspect.client import VideoClient

        async def _upload_video(vp: str) -> str:
            async with VideoClient() as client:
                res = await client.upload(vp)
                return res.video_id

        try:
            video_id = asyncio.run(_upload_video(video_path))
        except Exception as e:
            typer.echo(f"Warning: Could not upload video for V2A. ({e})", err=True)
    else:
        typer.echo("[3/4] No V2A events; skipping video upload.", err=True)

    typer.echo(f"[4/4] Generating {n_items} audio tracks...", err=True)
    audio_dir = Path(tempfile.mkdtemp(prefix="v2a_synth_audio_"))
    generated_audio: dict[str, str] = {}

    for i, item in enumerate(audio_plan.items, 1):
        duration = item.time[1] - item.time[0]
        out_path = str(audio_dir / f"{item.item_id}.wav")

        typer.echo(
            f"  [{i}/{n_items}] {item.item_id} ({item.type}, {item.time[0]}s-{item.time[1]}s): {item.description}",
        )
        audio_file = generate_audio_for_item(
            kind=item.type,
            description=item.description,
            out_path=out_path,
            duration=duration,
            video_id=video_id,
            fps=fps,
            time=item.time,
            generation_model=item.generation_model,
            spoken_text=item.spoken_text,
        )
        if audio_file:
            generated_audio[item.item_id] = audio_file

    n_generated = len(generated_audio)
    typer.echo(f"  → Generated {n_generated}/{n_items} audio files.", err=True)

    if n_generated == 0:
        typer.echo("Error: No audio files were generated.", err=True)
        raise typer.Exit(code=1)

    output_path = output
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(Path(video_path).parent / f"synthesized_{timestamp}.mp4")

    typer.echo(f"[4/4] Mixing audio into video → {output_path}", err=True)
    result = mix_audio_into_video(
        video_path=video_path,
        audio_plan=audio_plan,
        generated_audio=generated_audio,
        output_path=output_path,
        keep_original_audio=keep_original_audio,
    )

    if result:
        typer.echo(f"\n✅ Synthesis complete: {Path(result).resolve()}", err=True)
        return

    typer.echo("\n❌ Synthesis failed.", err=True)
    raise typer.Exit(code=1)


def main() -> None:
    typer.run(synthesize)


if __name__ == "__main__":
    main()
