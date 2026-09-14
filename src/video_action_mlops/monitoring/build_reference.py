"""monitoring/build_reference.py — sinh "cửa sổ tham chiếu" cho drift
report (mục 4.10, phiên 11.1): chạy pipeline suy luận THẬT
(inference/pipeline.py, phiên 10.2 — lần thứ 4 module này được tái dùng,
sau serving, demo, giờ tới monitoring) trên các video THUỘC TẬP VAL (chưa
từng train — is_val_sample, data/labels.py phiên 7.2), ghi lại ĐÚNG 3
thống kê mà production cũng ghi (monitoring/logging_sink.py phiên 9.2):
input_mean_motion_score, confidence, predicted_class.

Vì sao cần module riêng này: roadmap muốn so sánh "cửa sổ tham chiếu (lúc
train/validate)" với "cửa sổ gần đây (production)" — nhưng
evaluation/evaluate.py (phiên 7.2) chỉ đọc EMBEDDING đã trích xuất
(data/processed/), không có quyền truy cập frame gốc nên KHÔNG tính được
input_mean_motion_score. Script này chạy lại từ VIDEO GỐC (data/raw/) qua
đúng pipeline production để có đủ 3 cột thống nhất với bảng predictions.

Chạy 1 LẦN thủ công (hoặc mỗi khi train lại model mới, tay hoặc thêm vào
CI sau này) — KHÔNG phải stage trong dvc.yaml, vì so sánh chỉ có ý nghĩa
khi ĐÃ có traffic production để so, không nằm trong pipeline train/eval.
"""

from __future__ import annotations

import csv
from pathlib import Path

import onnxruntime as ort

from video_action_mlops.config.loader import load_config
from video_action_mlops.data.labels import is_val_sample
from video_action_mlops.inference.pipeline import run_two_stage_inference

_FIELDNAMES = ["input_mean_motion_score", "confidence", "predicted_class"]


def build_reference(
    config_path: str,
    raw_dir: str,
    spatial_onnx_path: str,
    temporal_onnx_path: str,
    output_csv: str,
) -> Path:
    cfg = load_config(config_path)
    spatial_session = ort.InferenceSession(spatial_onnx_path, providers=["CPUExecutionProvider"])
    temporal_session = ort.InferenceSession(temporal_onnx_path, providers=["CPUExecutionProvider"])

    video_paths = [p for p in sorted(Path(raw_dir).glob("*.mp4")) if is_val_sample(p.stem)]
    if not video_paths:
        raise ValueError(
            f"Không có video val nào trong {raw_dir} (is_val_sample lọc hết — "
            f"cần đủ video mẫu, xem gap đã ghi ở phiên 7.2)."
        )

    rows: list[dict] = []
    for video_path in video_paths:
        with open(video_path, "rb") as f:
            video_bytes = f.read()
        result = run_two_stage_inference(video_bytes, cfg, spatial_session, temporal_session)
        rows.append(
            {
                "input_mean_motion_score": result["mean_motion_score"],
                "confidence": result["confidence"],
                "predicted_class": result["predicted_class"],
            }
        )
        print(f"[build_reference] {video_path.name}: {rows[-1]}")

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[build_reference] {len(rows)} video val -> {output_path}")
    return output_path
