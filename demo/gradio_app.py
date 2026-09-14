"""demo/gradio_app.py — demo public trên Hugging Face Spaces (mục 4.7).

Chọn SDK Gradio, hardware CPU Basic (miễn phí, không giới hạn) — dùng
onnxruntime, KHÔNG cần GPU/ZeroGPU vì đã export ONNX từ phiên 7.1. Roadmap
có nhắc ZeroGPU như lựa chọn khác (test PyTorch chưa export), nhưng project
này đã có pipeline CPU-only hoàn chỉnh — dùng ZeroGPU sẽ là công cụ thừa,
không phải thiếu sót khi không dùng.

Dùng LẠI đúng inference/pipeline.py (phiên 10.2) — CÙNG pipeline suy luận
với serving/api.py (deploy Render, phiên 10.1), khác nơi deploy (HF Spaces
thay vì Render) nhưng không lệch logic.

Không cài package qua pip trên HF Spaces (tránh kéo theo torch/torchvision
— cùng lý do đã giải quyết ở colab/bootstrap.ipynb phiên 6.3 và
docker/serve.Dockerfile phiên 8.2, đây là lần thứ 3 cùng 1 mẫu hình xuyên
suốt project). Thay vào đó, thêm thẳng src/ vào sys.path.
"""

from __future__ import annotations

import sys
from pathlib import Path

# HF Spaces chạy app.py trực tiếp từ repo của Space — src/ phải là thư mục
# anh em (sibling) với demo/ trong CÙNG repo được push lên Space (xem
# hướng dẫn deploy đi kèm). Không dùng `pip install -e .` để tránh kéo
# theo torch/torchvision qua [project.dependencies] (mục 3.1) — CPU Basic
# của HF Spaces không cần và không nên cài chúng.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import gradio as gr  # noqa: E402 — import sau khi chỉnh sys.path, cố ý
import onnxruntime as ort  # noqa: E402

from video_action_mlops.config.loader import load_config  # noqa: E402
from video_action_mlops.inference.pipeline import run_two_stage_inference  # noqa: E402

CONFIG_PATH = "configs/base.yaml"
SPATIAL_ONNX_PATH = "models/spatial.onnx"
TEMPORAL_ONNX_PATH = "models/temporal.onnx"

# Load 1 LẦN ở module scope — Gradio không có hook khởi động riêng như
# FastAPI lifespan (serving/api.py, phiên 8.1), nên load ngay khi script
# chạy đóng vai trò tương đương.
_cfg = load_config(CONFIG_PATH)
_spatial_session = ort.InferenceSession(SPATIAL_ONNX_PATH, providers=["CPUExecutionProvider"])
_temporal_session = ort.InferenceSession(TEMPORAL_ONNX_PATH, providers=["CPUExecutionProvider"])


def predict_video(video_path: str | None) -> dict[str, float]:
    """gr.Video trả về ĐƯỜNG DẪN FILE local (khác FastAPI UploadFile ở
    serving/api.py, vốn nhận bytes qua HTTP multipart) — đọc file trực
    tiếp từ đĩa rồi tái dùng ĐÚNG pipeline suy luận chung.

    return: dict {label: probability} — gr.Label hiển thị dạng thanh
    ngang xếp hạng, không cần tự vẽ biểu đồ.
    """
    if video_path is None:
        raise gr.Error("Chưa chọn video nào.")

    with open(video_path, "rb") as f:
        video_bytes = f.read()

    try:
        result = run_two_stage_inference(video_bytes, _cfg, _spatial_session, _temporal_session)
    except (ValueError, NotImplementedError) as exc:
        raise gr.Error(str(exc)) from exc

    # Project chưa có bảng ánh xạ class index -> tên hành động (cùng nợ kỹ
    # thuật đã ghi ở monitoring/schema.sql, phiên 9.2) — hiển thị "class {i}"
    # thay vì bịa tên hành động giả.
    return {f"class {i}": float(p) for i, p in enumerate(result["probabilities"])}


demo = gr.Interface(
    fn=predict_video,
    inputs=gr.Video(label="Video đầu vào"),
    outputs=gr.Label(label="Dự đoán", num_top_classes=5),
    title="video-action-mlops — demo",
    description=(
        "Upload 1 video ngắn để nhận diện hành động. Chạy CPU-only qua "
        "ONNX Runtime (không cần GPU) — cùng pipeline suy luận với API "
        "production trên Render (phiên 10.1)."
    ),
)

if __name__ == "__main__":
    demo.launch()
