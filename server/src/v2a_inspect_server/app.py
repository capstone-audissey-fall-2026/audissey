from __future__ import annotations
import asyncio
import shutil
import uuid
import multiprocessing
import logging
from pathlib import Path

# Fix for PyTorch CUDA DataLoader deadlock in FastAPI/Uvicorn
if multiprocessing.get_start_method(allow_none=True) != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from v2a_inspect_server.models import (
    Sam3SegmentImageRequest,
    Sam3TrackVideoRequest,
    KokoroGenerateSpeechRequest,
)
from v2a_inspect_server.inference.sam3 import Sam3InferenceClient
from v2a_inspect_server.inference.hunyuan import HunyuanInferenceClient
from v2a_inspect_server.inference.speech import KokoroInferenceClient
from v2a_inspect_server.settings import settings
from v2a_inspect_server.models.hunyuan import HunyuanGenerateV2ARequest
from fastapi.responses import FileResponse

logger = logging.getLogger("uvicorn.error")

sam3_client: Sam3InferenceClient | None = None
hunyuan_client: HunyuanInferenceClient | None = None
speech_client: KokoroInferenceClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global sam3_client, hunyuan_client, speech_client
    # Initialize the clients on startup
    sam3_client = Sam3InferenceClient()
    hunyuan_client = HunyuanInferenceClient()
    speech_client = KokoroInferenceClient()
    yield
    # Cleanup on shutdown
    if sam3_client is not None:
        sam3_client.close()
    if hunyuan_client is not None:
        hunyuan_client.close()
    if speech_client is not None:
        speech_client.close()
    sam3_client = hunyuan_client = speech_client = None


app = FastAPI(title="v2a-inspect-server", lifespan=lifespan)


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/videos/upload")
async def upload_video(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Only allow basic video extensions
    ext = Path(file.filename).suffix.lower()
    if ext not in [".mp4", ".mov", ".avi", ".mkv"]:
        raise HTTPException(status_code=400, detail="Invalid video format")

    video_id = str(uuid.uuid4())
    save_path = settings.upload_dir / f"{video_id}{ext}"

    try:
        with save_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save file: {str(e)}")

    return {"video_id": video_id}


@app.post("/infer/sam3/track-video")
async def track_video_sam3(request: Sam3TrackVideoRequest):
    if sam3_client is None:
        raise HTTPException(status_code=503, detail="SAM3 client not initialized")
    try:
        return sam3_client.track_video(request)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/infer/sam3/segment-image")
async def segment_image_sam3(request: Sam3SegmentImageRequest):
    if sam3_client is None:
        raise HTTPException(status_code=503, detail="SAM3 client not initialized")
    try:
        return sam3_client.segment_image(request)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/infer/hunyuan/generate-v2a")
async def generate_v2a_hunyuan(request: HunyuanGenerateV2ARequest):
    if hunyuan_client is None:
        raise HTTPException(status_code=503, detail="Hunyuan client not initialized")
    try:
        audio_path = hunyuan_client.generate_v2a(request)
        return FileResponse(audio_path, media_type="audio/wav")
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        if "model is not loaded" in str(e):
            raise HTTPException(status_code=503, detail=str(e))
        logger.exception("Hunyuan V2A generation failed.")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Hunyuan V2A generation failed.")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/infer/kokoro/generate-speech")
async def generate_speech_kokoro(request: KokoroGenerateSpeechRequest):
    if speech_client is None:
        raise HTTPException(status_code=503, detail="Kokoro client not initialized")
    try:
        audio_path = await asyncio.to_thread(speech_client.generate_speech, request)
        return FileResponse(audio_path, media_type="audio/wav")
    except RuntimeError as exc:
        status_code = 503 if "not installed" in str(exc) else 500
        logger.exception("Kokoro speech generation failed.")
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Kokoro speech generation failed.")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
