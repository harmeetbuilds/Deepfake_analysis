"""
Deepfake Detector API.

Run from the project root (the folder that contains detector.py):
    uvicorn backend.main:app --port 8000
Interactive docs / test UI:  http://localhost:8000/docs
"""
import logging
import os
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.schemas import HealthResponse, ModelInfo, PredictionResponse
from detector import DeepfakeDetector

log = logging.getLogger("deepfake-api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading model from %s ...", settings.model_path)
    app.state.detector = DeepfakeDetector(settings.model_path)   # loaded ONCE
    app.state.lock = threading.Lock()                            # one inference at a time
    log.info("Model ready: %s", app.state.detector.info)
    yield


app = FastAPI(title="Deepfake Detector API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=settings.allowed_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------- helpers
async def read_limited(file: UploadFile, max_mb: int) -> bytes:
    limit, size, chunks = max_mb * 1024 * 1024, 0, []
    while chunk := await file.read(1024 * 1024):
        size += len(chunk)
        if size > limit:
            raise HTTPException(413, f"File too large (max {max_mb} MB)")
        chunks.append(chunk)
    if size == 0:
        raise HTTPException(400, "Empty file")
    return b"".join(chunks)


def _ext(file: UploadFile) -> str:
    return Path(file.filename or "").suffix.lower()


def _to_response(raw: dict, filename: str, started: float) -> PredictionResponse:
    margin = settings.uncertain_margin
    uncertain = abs(raw["fake_probability"] - raw["threshold"]) < margin
    warning = None
    if not raw["face_found"]:
        warning = "No face detected - the whole frame was analysed, so this result is unreliable."
    elif uncertain:
        warning = "The score is close to the decision threshold - treat this result as inconclusive."
    return PredictionResponse(filename=filename, uncertain=uncertain, warning=warning,
                              inference_ms=int((time.time() - started) * 1000), **raw)


def _infer_image(request: Request, data: bytes) -> dict:
    with request.app.state.lock:
        return request.app.state.detector.predict_image(data)


def _infer_video(request: Request, data: bytes, suffix: str, frames: int) -> dict:
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        with request.app.state.lock:
            return request.app.state.detector.predict_video(path, frames)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


async def _run(fn, *args):
    try:
        return await run_in_threadpool(fn, *args)
    except ValueError as e:                      # unreadable / corrupt media
        raise HTTPException(422, str(e))
    except HTTPException:
        raise
    except Exception:
        log.exception("Inference failed")
        raise HTTPException(500, "Internal error while analysing the file")


async def _handle_image(request: Request, file: UploadFile) -> PredictionResponse:
    if _ext(file) not in IMAGE_EXT:
        raise HTTPException(415, f"Unsupported image type. Allowed: {sorted(IMAGE_EXT)}")
    data = await read_limited(file, settings.max_image_mb)
    t = time.time()
    raw = await _run(_infer_image, request, data)
    return _to_response(raw, file.filename, t)


async def _handle_video(request: Request, file: UploadFile, frames: int) -> PredictionResponse:
    if _ext(file) not in VIDEO_EXT:
        raise HTTPException(415, f"Unsupported video type. Allowed: {sorted(VIDEO_EXT)}")
    data = await read_limited(file, settings.max_video_mb)
    t = time.time()
    raw = await _run(_infer_video, request, data, _ext(file), frames)
    return _to_response(raw, file.filename, t)


# ---------------------------------------------------------------- routes
@app.get("/", include_in_schema=False)
def root():
    return {"service": "Deepfake Detector API", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse)
def health(request: Request):
    return HealthResponse(status="ok", model_loaded=hasattr(request.app.state, "detector"))


@app.get("/api/model-info", response_model=ModelInfo)
def model_info(request: Request):
    return request.app.state.detector.info


@app.post("/api/predict/image", response_model=PredictionResponse)
async def predict_image(request: Request, file: UploadFile = File(...)):
    return await _handle_image(request, file)


@app.post("/api/predict/video", response_model=PredictionResponse)
async def predict_video(request: Request, file: UploadFile = File(...),
                        frames: int = Query(settings.video_frames, ge=4, le=32,
                                            description="Number of frames sampled from the video")):
    return await _handle_video(request, file, frames)


@app.post("/api/predict", response_model=PredictionResponse)
async def predict_auto(request: Request, file: UploadFile = File(...),
                       frames: int = Query(settings.video_frames, ge=4, le=32)):
    """Single endpoint for the frontend: detects image vs video from the file."""
    ext, ctype = _ext(file), (file.content_type or "")
    if ext in IMAGE_EXT or (not ext and ctype.startswith("image/")):
        return await _handle_image(request, file)
    if ext in VIDEO_EXT or (not ext and ctype.startswith("video/")):
        return await _handle_video(request, file, frames)
    raise HTTPException(415, "Unsupported file type. Upload an image or a video.")
