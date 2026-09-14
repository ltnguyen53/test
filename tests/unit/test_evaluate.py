"""tests/unit/test_evaluate.py — test evaluation/evaluate.py (phiên 14.4).

Cần torch + onnxruntime thật để export 1 model temporal siêu nhỏ rồi
evaluate trên đó — end-to-end thật, không mock onnxruntime.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest
import torch

from video_action_mlops.config.schema import AppConfig
from video_action_mlops.data.labels import is_val_sample
from video_action_mlops.evaluation.evaluate import evaluate_temporal_onnx
from video_action_mlops.export.to_onnx import export_temporal_onnx
from video_action_mlops.models.temporal import TemporalAggregatorTrainable


def _write_labels_csv(path, rows: dict[str, int]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_filename", "label"])
        for stem, label in rows.items():
            writer.writerow([f"{stem}.mp4", label])


def _make_tiny_cfg() -> AppConfig:
    return AppConfig.model_validate(
        {
            "project": {"name": "test", "seed": 42},
            "data": {
                "frames_per_video": 2,
                "frame_size": [16, 16],
                "use_motion_channel": False,
                "num_classes": 3,
            },
            "phase1": {"embed_dim_out": 8, "epochs": 1, "learning_rate": 0.001},
            "phase2": {"input_dim": 8, "num_heads": 2, "epochs": 1, "learning_rate": 0.001},
        }
    )


def _export_tiny_temporal_onnx(tmp_path) -> tuple:
    cfg = _make_tiny_cfg()
    trainable = TemporalAggregatorTrainable(
        input_dim=cfg.phase2.input_dim,
        num_heads=cfg.phase2.num_heads,
        num_classes=cfg.data.num_classes,
    )
    checkpoint_path = tmp_path / "phase2.pt"
    torch.save(trainable.state_dict(), checkpoint_path)
    onnx_path = tmp_path / "temporal.onnx"
    export_temporal_onnx(cfg, checkpoint_path, onnx_path)
    return cfg, onnx_path


def test_evaluate_temporal_onnx_end_to_end(tmp_path):
    cfg, onnx_path = _export_tiny_temporal_onnx(tmp_path)

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    # 30 video giả — đủ nhiều để hash-split (is_val_sample, phiên 7.2)
    # gần như chắc chắn có ít nhất vài video rơi vào "val".
    stems = [f"video_{i:04d}" for i in range(30)]
    labels = {}
    for i, stem in enumerate(stems):
        seq = np.random.default_rng(i).normal(size=(2, cfg.phase2.input_dim)).astype(np.float32)
        np.save(processed_dir / f"{stem}.npy", seq)
        labels[stem] = i % cfg.data.num_classes

    labels_csv = tmp_path / "labels.csv"
    _write_labels_csv(labels_csv, labels)

    result = evaluate_temporal_onnx(onnx_path, processed_dir, labels_csv)

    assert set(result.keys()) == {"val_accuracy", "num_val_samples"}
    assert 0.0 <= result["val_accuracy"] <= 1.0
    assert result["num_val_samples"] > 0
    assert result["num_val_samples"] < len(stems)  # phải nhỏ hơn tổng (đã lọc theo split)


def test_evaluate_temporal_onnx_empty_val_raises(tmp_path):
    """Chọn CHÍNH XÁC những stem hash vào 'train' (không phải val, dùng
    thẳng is_val_sample thật — phiên 7.2) để tạo tập val RỖNG một cách
    TẤT ĐỊNH, không phụ thuộc may rủi của hash như random sample."""
    _, onnx_path = _export_tiny_temporal_onnx(tmp_path)

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    train_only_stems = [
        f"video_{i:04d}" for i in range(300) if not is_val_sample(f"video_{i:04d}")
    ][:5]
    assert len(train_only_stems) == 5  # sanity-check chính giả định của test

    labels = {}
    for stem in train_only_stems:
        np.save(processed_dir / f"{stem}.npy", np.zeros((2, 8), dtype=np.float32))
        labels[stem] = 0
    labels_csv = tmp_path / "labels.csv"
    _write_labels_csv(labels_csv, labels)

    with pytest.raises(ValueError, match="[Rr]ỗng|rong"):
        evaluate_temporal_onnx(onnx_path, processed_dir, labels_csv)


def test_evaluate_temporal_onnx_missing_processed_dir_raises(tmp_path):
    _, onnx_path = _export_tiny_temporal_onnx(tmp_path)

    empty_dir = tmp_path / "processed_empty"
    empty_dir.mkdir()
    labels_csv = tmp_path / "labels.csv"
    _write_labels_csv(labels_csv, {})

    with pytest.raises(FileNotFoundError):
        evaluate_temporal_onnx(onnx_path, empty_dir, labels_csv)


def test_evaluate_temporal_onnx_missing_label_raises(tmp_path):
    _, onnx_path = _export_tiny_temporal_onnx(tmp_path)

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    # Cần video CHẮC CHẮN rơi vào val để nhánh "thiếu nhãn" thật sự được
    # kiểm tra (nếu vô tình rơi vào train, video bị lọc trước khi tới bước
    # kiểm tra nhãn, test sẽ pass sai lý do).
    val_stem = next(f"video_{i:04d}" for i in range(300) if is_val_sample(f"video_{i:04d}"))
    np.save(processed_dir / f"{val_stem}.npy", np.zeros((2, 8), dtype=np.float32))

    labels_csv = tmp_path / "labels.csv"
    _write_labels_csv(labels_csv, {"video_KHONG_KHOP": 0})  # cố tình không khớp val_stem

    with pytest.raises(ValueError, match=val_stem):
        evaluate_temporal_onnx(onnx_path, processed_dir, labels_csv)


def test_evaluate_temporal_onnx_accuracy_formula_is_exact(tmp_path):
    """4 test trên phủ đủ MỌI nhánh raise + happy-path "trong khoảng
    [0,1]" -- nhưng chưa test CÔNG THỨC accuracy đúng con số, chỉ đúng
    khoảng giá trị hợp lệ (model random, không kiểm soát được dự đoán cụ
    thể). Test này KHÓA weight của head (Linear) về 0 + bias one-hot lệch
    hẳn về 1 class -- output logits KHÔNG PHỤ THUỘC input nữa (pooled @ 0
    = 0, chỉ còn bias), model LUÔN dự đoán đúng 1 class biết trước, cho
    phép tính tay accuracy kỳ vọng chính xác thay vì chỉ đoán khoảng."""
    cfg = _make_tiny_cfg()
    trainable = TemporalAggregatorTrainable(
        input_dim=cfg.phase2.input_dim,
        num_heads=cfg.phase2.num_heads,
        num_classes=cfg.data.num_classes,
    )
    with torch.no_grad():
        trainable.head.weight.zero_()
        trainable.head.bias.copy_(torch.tensor([-10.0, -10.0, 10.0]))  # LUON du doan class 2

    checkpoint_path = tmp_path / "phase2.pt"
    torch.save(trainable.state_dict(), checkpoint_path)
    onnx_path = tmp_path / "temporal.onnx"
    export_temporal_onnx(cfg, checkpoint_path, onnx_path)

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    stems = [f"video_{i:04d}" for i in range(300) if is_val_sample(f"video_{i:04d}")][:10]
    assert len(stems) == 10  # sanity-check gia dinh cua test

    labels = {}
    for i, stem in enumerate(stems):
        seq = np.random.default_rng(i).normal(size=(2, cfg.phase2.input_dim)).astype(np.float32)
        np.save(processed_dir / f"{stem}.npy", seq)
        # 6/10 video gan nhan DUNG (class 2, khop du doan luon co dinh cua
        # model) -- 4/10 con lai gan nhan SAI (class 0) -- ky vong CHINH
        # XAC accuracy = 0.6, khong phai "mot gia tri nao do trong [0,1]".
        labels[stem] = 2 if i < 6 else 0

    labels_csv = tmp_path / "labels.csv"
    _write_labels_csv(labels_csv, labels)

    result = evaluate_temporal_onnx(onnx_path, processed_dir, labels_csv)

    assert result["num_val_samples"] == 10
    assert result["val_accuracy"] == pytest.approx(0.6)
