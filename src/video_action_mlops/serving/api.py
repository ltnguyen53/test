"""serving/api.py — FastAPI, /predict + /healthz.

Logic suy luận (decode/resize/preprocess/2 lần onnxruntime) nằm ở
inference/pipeline.py (phiên 10.2, dùng chung với demo/gradio_app.py) —
file này chỉ còn phần đặc thù HTTP: lifespan load model, wiring request/
response, wiring logging.

Model + config load 1 LẦN lúc STARTUP (FastAPI lifespan), không load lại
mỗi request — tạo lại InferenceSession mỗi request sẽ chậm và lãng phí.
"""

from __future__ import annotations

import os
import time
import uuid
from contextlib import asynccontextmanager

import onnxruntime as ort
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile

from video_action_mlops.config.loader import load_config
from video_action_mlops.inference.pipeline import run_two_stage_inference
from video_action_mlops.monitoring.logging_sink import PredictionLogEntry, log_prediction
from video_action_mlops.serving.schemas import HealthResponse, PredictionResponse

CONFIG_PATH = "configs/base.yaml"
SPATIAL_ONNX_PATH = "models/spatial.onnx"
TEMPORAL_ONNX_PATH = "models/temporal.onnx"

# Nợ kỹ thuật, ghi rõ: CHƯA nối với MLflow Model Registry version thật
# (mục 3.2/4.3) — đặt qua env var, mặc định "unknown" nếu không set. Việc
# tự động lấy version từ registry lúc serving nằm ngoài phạm vi phiên này.
MODEL_VERSION = os.environ.get("MODEL_VERSION", "unknown")

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Không bọc try/except ở đây — nếu load lỗi (thiếu file .onnx, config
    # sai), process PHẢI chết ngay lúc khởi động, không được lên "healthy"
    # rồi mới lỗi ở request đầu tiên (fail-fast, nhất quán triết lý xuyên
    # suốt project).
    cfg = load_config(CONFIG_PATH)
    _state["cfg"] = cfg
    _state["spatial_session"] = ort.InferenceSession(
        SPATIAL_ONNX_PATH, providers=["CPUExecutionProvider"]
    )
    _state["temporal_session"] = ort.InferenceSession(
        TEMPORAL_ONNX_PATH, providers=["CPUExecutionProvider"]
    )
    yield
    _state.clear()


app = FastAPI(title="video-action-mlops", lifespan=lifespan)


def _run_inference(video_bytes: bytes) -> tuple[PredictionResponse, dict]:
    cfg = _state["cfg"]
    try:
        result = run_two_stage_inference(
            video_bytes, cfg, _state["spatial_session"], _state["temporal_session"]
        )
    except NotImplementedError as exc:
        # use_motion_channel chưa hỗ trợ (pipeline.py raise NotImplementedError,
        # framework-agnostic) — ở tầng HTTP mới bọc thành HTTPException.
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    response = PredictionResponse(
        predicted_class=result["predicted_class"], probabilities=result["probabilities"]
    )
    stats = {
        "frame_count": result["frame_count"],
        "mean_motion_score": result["mean_motion_score"],
        "confidence": result["confidence"],
    }
    return response, stats


@app.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    """Không tự kiểm tra lại model ở đây — nếu lifespan startup lỗi,
    process không lên nổi, /healthz sẽ không bao giờ được gọi tới. Trả 200
    nghĩa là process sống VÀ model đã load xong (2 điều kiện gộp làm 1)."""
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictionResponse)
async def predict(background_tasks: BackgroundTasks, file: UploadFile = File(...)) -> PredictionResponse:
    if not file.filename or not file.filename.lower().endswith((".mp4", ".mov", ".avi")):
        raise HTTPException(status_code=400, detail="Chỉ nhận .mp4/.mov/.avi")

    video_bytes = await file.read()
    start = time.perf_counter()
    try:
        response, stats = _run_inference(video_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    latency_ms = (time.perf_counter() - start) * 1000

    # BackgroundTasks: chạy SAU KHI response đã trả cho user — ghi log
    # KHÔNG cộng thêm latency vào response chính (mục 4.8). log_prediction
    # là hàm sync (psycopg2) — Starlette tự chạy nó qua threadpool, không
    # chặn event loop.
    background_tasks.add_task(
        log_prediction,
        PredictionLogEntry(
            request_id=uuid.uuid4(),
            input_frame_count=stats["frame_count"],
            input_mean_motion_score=stats["mean_motion_score"],
            predicted_class=response.predicted_class,
            confidence=stats["confidence"],
            latency_ms=latency_ms,
            model_version=MODEL_VERSION,
        ),
    )

    return response
