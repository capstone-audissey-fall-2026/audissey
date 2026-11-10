import json
from bisect import bisect_right
import subprocess
from contextlib import contextmanager
from fractions import Fraction
from functools import cache
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock

import numpy as np

from audyssey.contracts import GroundResult, VisualHint
from audyssey.settings import Settings

# shortcut: one model and one video at a time; add a queue if throughput matters
_model_lock = Lock()


@cache
def _load():
    settings = Settings()
    try:
        torch = import_module("torch")
        sam3 = import_module("sam3.model_builder")
        hub = import_module("huggingface_hub")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "SAM3 is not installed: install the gpu/ environment"
        ) from exc
    checkpoint = hub.hf_hub_download(
        repo_id=settings.sam3_model,
        filename="sam3.1_multiplex.pt",
        local_files_only=True,
    )
    # The pinned builder initializes CUDA internally; select its device before building.
    with (
        torch.cuda.device(settings.sam3_device),
        torch.autocast(device_type="cuda", enabled=False),
    ):
        torch.autocast_increment_nesting()
        depth = torch.autocast_decrement_nesting()
        try:
            predictor = sam3.build_sam3_multiplex_video_predictor(
                checkpoint_path=checkpoint, use_fa3=False, compile=False, warm_up=False
            )
        finally:
            # The pinned builder enters persistent autocast scopes, even on partial failure.
            torch.autocast_increment_nesting()
            while torch.autocast_decrement_nesting() > depth:
                pass
    return torch, predictor.model


def _video_info(video: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate:frame=best_effort_timestamp_time",
            "-of",
            "json",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    info = json.loads(result.stdout)
    stream = info["streams"][0]
    timestamps = [
        Fraction(frame["best_effort_timestamp_time"]) for frame in info["frames"]
    ]
    stream["timestamps"] = [timestamp - timestamps[0] for timestamp in timestamps]
    return stream


@contextmanager
def _session(resource: Path):
    torch, model = _load()
    with (
        torch.cuda.device(Settings().sam3_device),
        torch.inference_mode(),
        torch.autocast(device_type="cuda", dtype=torch.bfloat16),
    ):
        # Avoid the pinned predictor dispatcher: it passes an unsupported offload_state_to_cpu.
        state = model.init_state(
            resource_path=str(resource),
            offload_video_to_cpu=True,
            async_loading_frames=False,
        )
        try:
            yield model, state
        finally:
            state.clear()


def _add_prompt(model, state: dict, index: int, hint: VisualHint) -> dict:
    x0, y0, x1, y1 = hint.box
    _, outputs = model.add_prompt(
        inference_state=state,
        frame_idx=index,
        text_str=hint.description,
        boxes_xywh=[[x0, y0, x1 - x0, y1 - y0]],
        box_labels=[1],
    )
    return outputs


def _best(outputs: dict) -> int | None:
    return int(np.argmax(outputs["out_probs"])) if len(outputs["out_obj_ids"]) else None


def _hint_index(info: dict, hint: VisualHint) -> int:
    timestamps = info["timestamps"]
    time = Fraction(str(hint.time))
    if time < 0 or time >= timestamps[-1] + 1 / Fraction(info["r_frame_rate"]):
        raise ValueError("visual hint time is outside the video")
    return bisect_right(timestamps, time) - 1


def locate(video: Path, hint: VisualHint) -> GroundResult:
    # Only the hint frame matters: SAM3 takes an image as a one-frame video, so the
    # whole clip is never decoded onto the GPU.
    index = _hint_index(_video_info(video), hint)
    with _model_lock, TemporaryDirectory() as tmp:
        frame = Path(tmp) / "frame.png"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(video), "-an"]
            + ["-vf", f"select=eq(n\\,{index})", "-frames:v", "1", str(frame)],
            check=True,
            capture_output=True,
        )
        with _session(frame) as (model, state):
            outputs = _add_prompt(model, state, 0, hint)
            best = _best(outputs)
            if best is None:
                return GroundResult(found=False)
            x, y, width, height = outputs["out_boxes_xywh"][best]
            return GroundResult(
                found=True,
                box=(x, y, min(1, x + width), min(1, y + height)),
                score=float(outputs["out_probs"][best]),
            )


def mask_video(video: Path, hint: VisualHint, out: Path) -> Path:
    track_fps = Settings().sam3_track_fps
    if track_fps:
        with TemporaryDirectory() as tmp:
            low = Path(tmp) / "track.mp4"
            subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-i", str(video), "-an"]
                + [
                    "-vf",
                    f"fps={track_fps}",
                    "-c:v",
                    "libx264rgb",
                    "-crf",
                    "0",
                    str(low),
                ],
                check=True,
                capture_output=True,
            )
            return _mask_video(low, hint, out)
    return _mask_video(video, hint, out)


def _mask_video(video: Path, hint: VisualHint, out: Path) -> Path:
    info = _video_info(video)
    fps = Fraction(info["r_frame_rate"])
    width, height = info["width"], info["height"]
    timestamps = info["timestamps"]
    index = _hint_index(info, hint)
    with _model_lock, TemporaryDirectory() as tmp:
        root = Path(tmp)
        with _session(video) as (model, state):
            if len(timestamps) != state["num_frames"]:
                raise ValueError("SAM3 and ffmpeg decoded different frame counts")
            outputs = _add_prompt(model, state, index, hint)
            best = _best(outputs)
            if best is not None:
                target = outputs["out_obj_ids"][best]
                np.save(root / f"{index}.npy", outputs["out_binary_masks"][best])
                for reverse in (False, True):
                    for frame_index, result in model.propagate_in_video(
                        inference_state=state,
                        start_frame_idx=index,
                        reverse=reverse,
                        is_last_batch=True,
                    ):
                        matches = np.flatnonzero(result["out_obj_ids"] == target)
                        if len(matches):
                            np.save(
                                root / f"{frame_index}.npy",
                                result["out_binary_masks"][matches[0]],
                            )
        raw = root / "input.rgb"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                str(video),
                "-an",
                "-vsync",
                "0",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                str(raw),
            ],
            check=True,
            capture_output=True,
        )
        data = np.memmap(raw, dtype=np.uint8, mode="r").reshape(-1, height, width, 3)
        if len(data) != len(timestamps):
            raise ValueError("ffmpeg decoded a different frame count from ffprobe")
        manifest = ["ffconcat version 1.0"]
        for i, frame in enumerate(data):
            path = root / f"{i}.npy"
            mask = (
                np.load(path)
                if path.exists()
                else np.zeros((height, width), dtype=bool)
            )
            with (root / f"{i}.ppm").open("wb") as stream:
                stream.write(f"P6\n{width} {height}\n255\n".encode())
                # Keep the target and black out the rest (the paper's V*M). The pinned
                # SAMAudioProcessor.mask_videos does the opposite (frames * (mask == 0));
                # on sam-audio-bench speaker pairs only this direction picked the
                # masked speaker, so do not "fix" it to match mask_videos.
                stream.write((frame * mask[:, :, None]).tobytes())
            duration = (
                timestamps[i + 1] - timestamps[i] if i + 1 < len(data) else 1 / fps
            )
            manifest.extend(
                [
                    f"file '{i}.ppm'",
                    f"option framerate {fps}",
                    f"duration {float(duration):.9f}",
                ]
            )
        del data
        concat = root / "frames.txt"
        concat.write_text("\n".join(manifest) + "\n")
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-safe",
                "0",
                "-i",
                str(concat),
                "-vsync",
                "vfr",
                "-an",
                "-c:v",
                "libx264rgb",
                "-crf",
                "0",
                str(out),
            ],
            check=True,
            capture_output=True,
        )
    return out
