from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated
from uuid import uuid4

import typer

from v2a_inspect.models import VideoAsset
from v2a_inspect.pipeline import VideoAssetPipelineOptions, run_video_asset_pipeline
from v2a_inspect.audio_generation.synthesize import synthesize
from v2a_inspect.ui.cli import ui

app = typer.Typer(no_args_is_help=True, help="Run v2a-inspect.")


@app.callback()
def callback() -> None:
    """Run v2a-inspect."""


@app.command()
def run(
    video: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Input video file.",
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            dir_okay=False,
            writable=True,
            resolve_path=True,
            help="Write completed VideoAsset JSON to this file.",
        ),
    ] = None,
    work_dir: Annotated[
        Path | None,
        typer.Option(
            "--work-dir",
            file_okay=False,
            resolve_path=True,
            help="Directory for normalized video, keyframes, and tracking artifacts.",
        ),
    ] = None,
    server_url: Annotated[
        str | None,
        typer.Option("--server-url", help="SAM3 inference server URL."),
    ] = None,
    scene_threshold: Annotated[
        float,
        typer.Option("--scene-threshold", help="PySceneDetect threshold."),
    ] = 27.0,
    max_keyframes_per_scene: Annotated[
        int,
        typer.Option(
            "--max-keyframes-per-scene",
            min=1,
            help="Maximum keyframes extracted from each detected scene.",
        ),
    ] = 20,
    indent: Annotated[
        int,
        typer.Option(
            "--indent",
            min=0,
            help="JSON indentation. Use 0 for compact JSON.",
        ),
    ] = 2,
) -> None:
    """Run the full pipeline and export a completed VideoAsset JSON."""

    resolved_work_dir = work_dir or _default_work_dir(video)
    options = VideoAssetPipelineOptions(
        scene_threshold=scene_threshold,
        max_keyframes_per_scene=max_keyframes_per_scene,
        server_url=server_url,
    )

    try:
        video_asset = run_video_asset_pipeline(
            video,
            resolved_work_dir,
            options=options,
            on_stage=_print_stage,
        )
    except Exception as exc:  # noqa: BLE001 - CLI should surface any pipeline failure.
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    payload = _video_asset_json(video_asset, indent=indent)
    if output is None:
        typer.echo(payload)
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload + "\n", encoding="utf-8")
    typer.echo(f"wrote {output}", err=True)


def main() -> None:
    app()


def _default_work_dir(video: Path) -> Path:
    stem = _slug(video.stem) or "video"
    return Path.cwd() / ".v2a_inspect" / "work" / f"{stem}-{uuid4().hex[:8]}"


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-_.")


def _print_stage(stage: str) -> None:
    typer.echo(f"[v2a-inspect] {stage}", err=True)


def _video_asset_json(video_asset: VideoAsset, *, indent: int) -> str:
    return video_asset.model_dump_json(
        indent=None if indent < 1 else indent,
        exclude_computed_fields=True,
    )


app.command(name="synthesize")(synthesize)
app.command(name="ui")(ui)


if __name__ == "__main__":
    main()
