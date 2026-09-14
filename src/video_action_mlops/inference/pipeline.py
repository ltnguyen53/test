"""inference/pipeline.py — logic suy luận 2 giai đoạn DÙNG CHUNG giữa
serving/api.py (phiên 8.1, FastAPI, deploy Render) và demo/gradio_app.py
(phiên 10.2, Gradio, deploy HF Spaces) — cả 2 nơi cần ĐÚNG 1 pipeline suy
luận (decode -> resize -> preprocess -> spatial.onnx -> temporal.onnx),
tách ra đây thay vì copy-paste giữa 2 file (2 nơi lệch logic theo thời
gian là lỗi rất khó phát hiện).

Đây cũng chính là module `inference/` mà Dockerfile mẫu ở roadmap (mục
4.5) giả định tồn tại từ đầu ("COPY src/video_action_mlops/inference
./inference") — phiên 8.1 gộp thẳng logic vào serving/api.py vì lúc đó
chỉ có 1 nơi dùng (chưa cần trừu tượng hoá sớm); giờ có nơi thứ 2
(gradio_app.py) mới thật sự cần tách.

CỐ Ý KHÔNG import torch — chỉ onnxruntime + numpy + opencv, khớp footprint
CPU-only đã giữ nhất quán xuyên suốt (serving mục 4.5, demo mục 4.7 chọn
CPU Basic thay vì ZeroGPU vì đã có ONNX, không cần GPU).
"""

from __future__ import annotations

import tempfile

import cv2
import numpy as np
import onnxruntime as ort

from video_action_mlops.config.schema import AppConfig
from video_action_mlops.data.preprocessing import compute_motion_scores, preprocess_frames


def decode_video_bytes(video_bytes: bytes) -> np.ndarray:
    """(bytes) -> (T, H, W, C) uint8 RGB.

    cv2.VideoCapture cần đọc từ FILE, không đọc thẳng bytes trong RAM cho
    video container — ghi ra file tạm là hạn chế thật của opencv. opencv
    đọc BGR, phải tự đổi RGB cho khớp torchvision (mặc định RGB, dùng ở
    run_preprocess.py phiên 4.2) — thiếu bước này là lỗi "chạy được, dự
    đoán sai" âm thầm, không crash."""
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        tmp.flush()
        cap = cv2.VideoCapture(tmp.name)
        if not cap.isOpened():
            raise ValueError("Không đọc được video — file hỏng hoặc sai định dạng")

        frames = []
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        cap.release()

    if not frames:
        raise ValueError("Video không có frame nào")
    return np.stack(frames, axis=0)


def resize_frames_np(frames: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    h, w = size
    resized = [cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR) for f in frames]
    return np.stack(resized, axis=0)


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def run_two_stage_inference(
    video_bytes: bytes,
    cfg: AppConfig,
    spatial_session: ort.InferenceSession,
    temporal_session: ort.InferenceSession,
) -> dict:
    """Trả về dict thuần (không phải PredictionResponse — pydantic model
    đó thuộc serving/schemas.py, module này KHÔNG biết gì về FastAPI hay
    Gradio, chỉ trả data thô để caller tự quyết định bọc thế nào):
    {"predicted_class": int, "probabilities": list[float],
     "frame_count": int, "mean_motion_score": float, "confidence": float}
    """
    if cfg.data.use_motion_channel:
        # Gap đã ghi rõ ở features/extract_embeddings.py (phiên 5.1):
        # SpatialBackbone chưa hỗ trợ 4 channel.
        raise NotImplementedError(
            "cfg.data.use_motion_channel=True chưa được hỗ trợ (xem gap phiên 5.1)"
        )

    frames = decode_video_bytes(video_bytes)
    frames = resize_frames_np(frames, cfg.data.frame_size)

    # Tính TRƯỚC khi sample, trên CÙNG input mà preprocess_frames dùng nội
    # bộ để quyết định sample (phiên 2.1) — chỉ dùng để log/hiển thị,
    # không ảnh hưởng kết quả dự đoán.
    motion_scores = compute_motion_scores(frames)

    sampled = preprocess_frames(
        frames,
        frames_per_video=cfg.data.frames_per_video,
        use_motion_channel=False,
        apply_hflip=False,  # suy luận không augment
    )

    frames_chw = np.transpose(sampled, (0, 3, 1, 2)).astype(np.float32)
    embeddings = spatial_session.run(None, {"frame": frames_chw})[0]
    embedding_sequence = embeddings[None, ...].astype(np.float32)
    logits = temporal_session.run(None, {"embedding_sequence": embedding_sequence})[0]

    probs = softmax(logits[0])
    return {
        "predicted_class": int(np.argmax(probs)),
        "probabilities": probs.tolist(),
        "frame_count": int(frames.shape[0]),
        "mean_motion_score": float(motion_scores.mean()),
        "confidence": float(probs.max()),
    }
