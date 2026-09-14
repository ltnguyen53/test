"""evaluation/evaluate.py — đánh giá model TRÊN ĐÚNG NHỮNG GÌ SẼ DEPLOY
(models/temporal.onnx qua onnxruntime), KHÔNG phải checkpoint PyTorch —
đo hiệu năng của cái sẽ THẬT SỰ chạy production (mục 4.5: serving không
cài torch), không phải cái đã train.

CỐ Ý KHÔNG import torch (module này chỉ cần numpy + onnxruntime) — khớp
đúng footprint của serving image thật (tuần 8), và là lý do
Phase2SequenceDataset (phụ thuộc torch, data/datasets.py) KHÔNG được dùng
ở đây — evaluate.py tự đọc .npy + nhãn qua data/labels.py (torch-free,
phiên 7.2).

Dùng tập VAL (split="val", data/labels.py) — chưa từng được model nhìn
thấy lúc train_phase2 (phiên 5.2, đã lọc split="train" từ phiên 7.2).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

from video_action_mlops.data.labels import is_val_sample, load_label_by_stem


def evaluate_temporal_onnx(
    onnx_path: str | Path, processed_dir: str | Path, labels_csv: str | Path
) -> dict:
    """Chạy toàn bộ video ở tập val qua models/temporal.onnx, trả về
    {"val_accuracy": ..., "num_val_samples": ...}."""
    processed_dir = Path(processed_dir)
    label_by_stem = load_label_by_stem(labels_csv)

    npy_paths = sorted(processed_dir.glob("*.npy"))
    if not npy_paths:
        raise FileNotFoundError(
            f"Không tìm thấy .npy nào trong {processed_dir} — chạy stage "
            f"'extract_features' (phiên 5.1) trước."
        )

    val_paths_labels: list[tuple[Path, int]] = []
    for npy_path in npy_paths:
        stem = npy_path.stem
        if not is_val_sample(stem):
            continue
        if stem not in label_by_stem:
            raise ValueError(
                f"{npy_path.name} không có nhãn tương ứng trong {labels_csv} "
                f"(thiếu dòng cho '{stem}')"
            )
        val_paths_labels.append((npy_path, label_by_stem[stem]))

    if not val_paths_labels:
        raise ValueError(
            f"Tập val rỗng (0 video sau khi lọc is_val_sample) — dataset quá "
            f"nhỏ để hash-split 80/20 có ý nghĩa (cần nhiều video mẫu hơn "
            f"trong data/raw/labels.csv), hoặc processed_dir sai đường dẫn."
        )

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    correct = 0
    for npy_path, label in val_paths_labels:
        seq = np.load(npy_path).astype(np.float32)  # (T, embed_dim_out)
        seq_batch = seq[None, ...]  # (1, T, embed_dim_out) — graph ONNX cần chiều batch
        logits = session.run(None, {"embedding_sequence": seq_batch})[0]
        pred = int(np.argmax(logits[0]))
        correct += int(pred == label)

    accuracy = correct / len(val_paths_labels)
    return {"val_accuracy": accuracy, "num_val_samples": len(val_paths_labels)}
