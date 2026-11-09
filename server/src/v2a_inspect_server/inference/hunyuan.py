from __future__ import annotations

import logging
import random
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

from ..settings import settings
from ..models.hunyuan import HunyuanGenerateV2ARequest

logger = logging.getLogger("uvicorn.error")

try:
    import torch
    from hunyuanvideo_foley.utils.model_utils import load_model, denoise_process
    from hunyuanvideo_foley.utils.feature_utils import feature_process

    HUNYUAN_AVAILABLE = True
except ImportError:
    HUNYUAN_AVAILABLE = False


def _is_cuda_oom(exc: Exception) -> bool:
    message = str(exc).lower()
    return "cuda" in message and "out of memory" in message


class HunyuanInferenceClient:
    def __init__(self) -> None:
        if not HUNYUAN_AVAILABLE:
            logger.warning(
                "HunyuanVideo-Foley is not installed. V2A generation will fail."
            )
            self.model_dict = None
            self.cfg = None
            return

        logger.info("Initializing HunyuanVideo-Foley model...")
        model_path = settings.hunyuan_model_path
        model_size = settings.hunyuan_model_size

        config_filename = "config_xl.yaml" if model_size == "xl" else "config.yaml"
        required_files = [
            f"hunyuanvideo_foley_{model_size}.pth",
            "vae_128d_48k.pth",
            "synchformer_state_dict.pth",
            config_filename,
        ]
        for filename in required_files:
            if (model_path / filename).exists():
                continue
            logger.info(
                "Downloading %s from %s to %s.",
                filename,
                settings.hunyuan_model_id,
                model_path,
            )
            try:
                import huggingface_hub

                huggingface_hub.hf_hub_download(
                    repo_id=settings.hunyuan_model_id,
                    filename=filename,
                    local_dir=model_path,
                )
            except Exception as exc:
                logger.error("Failed to download %s: %s", filename, exc)

        enable_offload = settings.hunyuan_enable_offload
        logger.info(
            "Hunyuan offload %s.",
            "enabled" if enable_offload else "disabled",
        )
        config_path = (model_path / config_filename).resolve()

        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
        device_idx = 0

        try:
            with torch.cuda.device(device_idx):
                self.model_dict, self.cfg = load_model(
                    model_path=str(model_path.resolve()),
                    config_path=str(config_path),
                    device=torch.device(device_name),
                    enable_offload=enable_offload,
                    model_size=model_size,
                )
            logger.info("HunyuanVideo-Foley model loaded successfully.")
        except Exception as e:
            logger.error("Failed to load HunyuanVideo-Foley model: %s", e)
            self.model_dict = None
            self.cfg = None

    def close(self) -> None:
        if self.model_dict is not None:
            # Free memory
            del self.model_dict
            self.model_dict = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def generate_v2a(self, request: HunyuanGenerateV2ARequest) -> str:
        if not HUNYUAN_AVAILABLE or self.model_dict is None:
            raise RuntimeError("HunyuanVideo-Foley model is not loaded.")

        start_time = time.perf_counter()
        video_path = self._find_video_path(request.video_id)

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 24.0
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        video_duration = total_frames / fps if total_frames > 0 else 0
        cap.release()

        start_s = request.start_frame_index / fps
        end_s = request.end_frame_index / fps

        # Clamp to avoid ffmpeg seeking past EOF
        if video_duration > 0:
            start_s = max(0.0, min(start_s, video_duration - 0.1))
            end_s = max(0.0, min(end_s, video_duration))

        # Re-encoding을 피하기 위해 tpad 필터를 사용할 수 없으므로,
        # 최소 1.5초가 필요하다면 원본 비디오 내에서 구간(window) 자체를 1.5초로 확장합니다.
        actual_duration = end_s - start_s
        if actual_duration < 1.5:
            end_s = start_s + 1.5
            if video_duration > 0 and end_s > video_duration:
                end_s = video_duration
                start_s = max(0.0, end_s - 1.5)

        final_duration = end_s - start_s
        if final_duration <= 0:
            final_duration = 0.1

        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_dir = Path(temp_dir)
            audio_tensor, sample_rate = self._infer_from_video_window(
                ffmpeg_exe=ffmpeg_exe,
                video_path=video_path,
                temp_dir=tmp_dir,
                start_s=start_s,
                duration=final_duration,
                prompt=request.prompt,
                guidance_scale=request.guidance_scale,
                num_inference_steps=request.num_inference_steps,
                neg_prompt=request.negative_prompt,
            )

            # Save audio to a temp file using soundfile to bypass torchcodec bugs
            out_audio_path = str(Path(temp_dir) / "output.wav")
            import soundfile as sf

            # audio_tensor is (channels, frames)
            audio_np = audio_tensor.cpu().numpy().T
            sf.write(out_audio_path, audio_np, sample_rate)

            logger.info(
                f"Hunyuan generation took {time.perf_counter() - start_time:.2f}s"
            )

            final_audio_path = settings.upload_dir / f"audio_{uuid.uuid4().hex}.wav"

            shutil.copy(out_audio_path, final_audio_path)

        return str(final_audio_path)

    def _infer_from_video_window(
        self,
        *,
        ffmpeg_exe: str,
        video_path: Path,
        temp_dir: Path,
        start_s: float,
        duration: float,
        prompt: str,
        guidance_scale: float,
        num_inference_steps: int,
        neg_prompt: str | None,
    ) -> tuple[torch.Tensor, int]:
        if duration < 1.5:
            reencoded = temp_dir / "cropped_reencoded.mp4"
            self._reencode_crop_video(
                ffmpeg_exe=ffmpeg_exe,
                video_path=video_path,
                output_path=reencoded,
                start_s=start_s,
                duration=1.5,
                pad_duration=1.5,
            )
            return self._infer(
                str(reencoded),
                prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                neg_prompt=neg_prompt,
            )

        copied = temp_dir / f"cropped_copy{video_path.suffix}"
        try:
            self._copy_crop_video(
                ffmpeg_exe=ffmpeg_exe,
                video_path=video_path,
                output_path=copied,
                start_s=start_s,
                duration=duration,
            )
            return self._infer(
                str(copied),
                prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                neg_prompt=neg_prompt,
            )
        except Exception as exc:
            if _is_cuda_oom(exc):
                raise
            logger.warning(
                "Hunyuan copy-cropped clip failed; retrying with re-encode.",
                exc_info=True,
            )

        reencoded = temp_dir / "cropped_reencoded.mp4"
        self._reencode_crop_video(
            ffmpeg_exe=ffmpeg_exe,
            video_path=video_path,
            output_path=reencoded,
            start_s=start_s,
            duration=duration,
            pad_duration=None,
        )
        return self._infer(
            str(reencoded),
            prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            neg_prompt=neg_prompt,
        )

    def _copy_crop_video(
        self,
        *,
        ffmpeg_exe: str,
        video_path: Path,
        output_path: Path,
        start_s: float,
        duration: float,
    ) -> None:
        self._run_ffmpeg(
            [
                ffmpeg_exe,
                "-y",
                "-ss",
                str(start_s),
                "-t",
                str(duration),
                "-i",
                str(video_path),
                "-map",
                "0:v:0",
                "-c:v",
                "copy",
                "-an",
                str(output_path),
            ],
            "Hunyuan copy crop",
        )
        self._validate_video_clip(output_path)

    def _reencode_crop_video(
        self,
        *,
        ffmpeg_exe: str,
        video_path: Path,
        output_path: Path,
        start_s: float,
        duration: float,
        pad_duration: float | None,
    ) -> None:
        codecs = ["h264_nvenc", "libx264"] if settings.enable_nvenc else ["libx264"]
        last_error: RuntimeError | None = None

        for codec in codecs:
            cmd = [
                ffmpeg_exe,
                "-y",
                "-ss",
                str(start_s),
                "-t",
                str(duration),
                "-i",
                str(video_path),
                "-map",
                "0:v:0",
            ]
            if pad_duration is not None:
                cmd.extend(["-vf", f"tpad=stop_mode=clone:stop_duration={pad_duration}"])
            cmd.extend(
                [
                    "-c:v",
                    codec,
                    "-pix_fmt",
                    "yuv420p",
                    "-an",
                    str(output_path),
                ]
            )

            try:
                self._run_ffmpeg(cmd, f"Hunyuan re-encode crop ({codec})")
                self._validate_video_clip(output_path)
                return
            except RuntimeError as exc:
                last_error = exc
                if codec == "h264_nvenc":
                    logger.warning(
                        "Hunyuan NVENC crop failed; retrying with libx264: %s",
                        exc,
                    )

        if last_error is not None:
            raise last_error
        raise RuntimeError("Hunyuan re-encode crop failed before ffmpeg was invoked.")

    def _run_ffmpeg(self, cmd: list[str], label: str) -> None:
        result = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0:
            return

        stderr = (result.stderr or "").strip()
        if len(stderr) > 4000:
            stderr = stderr[-4000:]
        raise RuntimeError(f"{label} failed with exit {result.returncode}: {stderr}")

    def _validate_video_clip(self, path: Path) -> None:
        if not path.exists():
            raise RuntimeError(f"Hunyuan crop output does not exist: {path}")
        if path.stat().st_size <= 0:
            raise RuntimeError(f"Hunyuan crop output is empty: {path}")

        cap = cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():
                raise RuntimeError(f"Hunyuan crop output cannot be opened: {path}")
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(
                    f"Hunyuan crop output has no decodable first frame: {path}"
                )
        finally:
            cap.release()

    def _infer(
        self,
        video_path: str,
        prompt: str,
        guidance_scale: float = 4.5,
        num_inference_steps: int = 50,
        neg_prompt: str | None = None,
    ) -> tuple[torch.Tensor, int]:
        self._set_manual_seed(42)

        with torch.cuda.device(0):
            visual_feats, text_feats, audio_len_in_s = feature_process(
                video_path, prompt, self.model_dict, self.cfg, neg_prompt=neg_prompt
            )
            audio, sample_rate = denoise_process(
                visual_feats,
                text_feats,
                audio_len_in_s,
                self.model_dict,
                self.cfg,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
            )
        return audio[0], sample_rate

    def _set_manual_seed(self, global_seed: int) -> None:
        random.seed(global_seed)
        np.random.seed(global_seed)
        if torch.cuda.is_available():
            torch.manual_seed(global_seed)

    def _find_video_path(self, video_id: str) -> Path:
        for ext in [".mp4", ".mov", ".avi", ".mkv"]:
            path = settings.upload_dir / f"{video_id}{ext}"
            if path.exists():
                return path
        raise FileNotFoundError(f"Video {video_id} not found")
