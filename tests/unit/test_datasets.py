"""tests/unit/test_datasets.py — test data/datasets.py (phiên 14.2).

Cần torch thật (Dataset base class, tensor operations trong __getitem__)
— khác test_labels.py (thuần Python). Trên máy có `pip install -e ".[dev]"`
sẽ chạy bình thường.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from video_action_mlops.data.datasets import Phase1FrameDataset, Phase2SequenceDataset
from video_action_mlops.data.labels import is_val_sample


def _write_labels_csv(path: Path, rows: dict[str, int]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_filename", "label"])
        for stem, label in rows.items():
            writer.writerow([f"{stem}.mp4", label])


# ---------- Phase1FrameDataset ----------


def test_phase1_frame_dataset_flattens_frames(tmp_path):
    interim_dir = tmp_path / "interim"
    interim_dir.mkdir()
    # 2 video, mỗi video 3 frame -> tổng 6 sample sau khi PHẲNG HOÁ theo
    # frame (khác Phase2, xem docstring datasets.py phiên 4.3)
    np.save(interim_dir / "video_a.npy", np.zeros((3, 4, 4, 3), dtype=np.float32))
    np.save(interim_dir / "video_b.npy", np.zeros((3, 4, 4, 3), dtype=np.float32))

    labels_csv = tmp_path / "labels.csv"
    _write_labels_csv(labels_csv, {"video_a": 0, "video_b": 1})

    dataset = Phase1FrameDataset(interim_dir=interim_dir, labels_csv=labels_csv, split="all")

    assert len(dataset) == 6  # 2 video x 3 frame

    frame_tensor, label = dataset[0]
    assert tuple(frame_tensor.shape) == (3, 4, 4)  # (C, H, W) sau permute
    assert label in (0, 1)


def test_phase1_frame_dataset_missing_label_raises(tmp_path):
    interim_dir = tmp_path / "interim"
    interim_dir.mkdir()
    np.save(interim_dir / "video_a.npy", np.zeros((2, 4, 4, 3), dtype=np.float32))

    labels_csv = tmp_path / "labels.csv"
    _write_labels_csv(labels_csv, {"video_OTHER": 0})  # không khớp video_a

    with pytest.raises(ValueError, match="video_a"):
        Phase1FrameDataset(interim_dir=interim_dir, labels_csv=labels_csv, split="all")


def test_phase1_frame_dataset_empty_interim_raises(tmp_path):
    interim_dir = tmp_path / "interim"
    interim_dir.mkdir()
    labels_csv = tmp_path / "labels.csv"
    _write_labels_csv(labels_csv, {})

    with pytest.raises(FileNotFoundError):
        Phase1FrameDataset(interim_dir=interim_dir, labels_csv=labels_csv, split="all")


def test_phase1_frame_dataset_train_val_split_partitions_all(tmp_path):
    interim_dir = tmp_path / "interim"
    interim_dir.mkdir()
    stems = [f"video_{i:04d}" for i in range(30)]  # đủ nhiều để cả 2 phía non-empty
    for stem in stems:
        np.save(interim_dir / f"{stem}.npy", np.zeros((2, 4, 4, 3), dtype=np.float32))

    labels_csv = tmp_path / "labels.csv"
    _write_labels_csv(labels_csv, dict.fromkeys(stems, 0))

    train_ds = Phase1FrameDataset(interim_dir=interim_dir, labels_csv=labels_csv, split="train")
    val_ds = Phase1FrameDataset(interim_dir=interim_dir, labels_csv=labels_csv, split="val")
    all_ds = Phase1FrameDataset(interim_dir=interim_dir, labels_csv=labels_csv, split="all")

    assert len(train_ds) + len(val_ds) == len(all_ds)


def test_phase1_frame_dataset_split_filters_out_everything_raises(tmp_path):
    """KHÁC test_phase1_frame_dataset_empty_interim_raises (FileNotFoundError
    -- không có .npy nào cả): ở đây .npy TỒN TẠI thật, nhưng split filter
    (is_val_sample) loại bỏ HẾT -- nhánh ValueError riêng, phát hiện qua
    coverage report (chưa từng được test dù có message riêng, khác hẳn
    message của FileNotFoundError)."""
    interim_dir = tmp_path / "interim"
    interim_dir.mkdir()
    val_stem = next(f"video_{i:04d}" for i in range(300) if is_val_sample(f"video_{i:04d}"))
    np.save(interim_dir / f"{val_stem}.npy", np.zeros((2, 4, 4, 3), dtype=np.float32))

    labels_csv = tmp_path / "labels.csv"
    _write_labels_csv(labels_csv, {val_stem: 0})

    # Chi co 1 video VAL duy nhat -> xin split="train" se loc sach, khong con gi.
    with pytest.raises(ValueError, match="không còn video nào"):
        Phase1FrameDataset(interim_dir=interim_dir, labels_csv=labels_csv, split="train")


# ---------- Phase2SequenceDataset ----------


def test_phase2_sequence_dataset_one_sample_per_video(tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    np.save(processed_dir / "video_a.npy", np.zeros((16, 512), dtype=np.float32))
    np.save(processed_dir / "video_b.npy", np.zeros((16, 512), dtype=np.float32))

    labels_csv = tmp_path / "labels.csv"
    _write_labels_csv(labels_csv, {"video_a": 0, "video_b": 1})

    dataset = Phase2SequenceDataset(processed_dir=processed_dir, labels_csv=labels_csv, split="all")

    # 1 sample / VIDEO (không phẳng theo frame như Phase1) — khác biệt cốt
    # lõi giữa 2 Dataset, xem docstring datasets.py.
    assert len(dataset) == 2

    seq_tensor, label = dataset[0]
    assert tuple(seq_tensor.shape) == (16, 512)


def test_phase2_sequence_dataset_missing_processed_dir_raises(tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    labels_csv = tmp_path / "labels.csv"
    _write_labels_csv(labels_csv, {})

    with pytest.raises(FileNotFoundError):
        Phase2SequenceDataset(processed_dir=processed_dir, labels_csv=labels_csv, split="all")


def test_phase2_sequence_dataset_missing_label_raises(tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    np.save(processed_dir / "video_a.npy", np.zeros((16, 512), dtype=np.float32))

    labels_csv = tmp_path / "labels.csv"
    _write_labels_csv(labels_csv, {"video_OTHER": 0})

    with pytest.raises(ValueError, match="video_a"):
        Phase2SequenceDataset(processed_dir=processed_dir, labels_csv=labels_csv, split="all")


def test_phase2_sequence_dataset_split_filters_out_everything_raises(tmp_path):
    """Tương ứng test_phase1_frame_dataset_split_filters_out_everything_raises
    ở trên, cho Phase2SequenceDataset (code riêng, không kế thừa chung)."""
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    val_stem = next(f"video_{i:04d}" for i in range(300) if is_val_sample(f"video_{i:04d}"))
    np.save(processed_dir / f"{val_stem}.npy", np.zeros((16, 512), dtype=np.float32))

    labels_csv = tmp_path / "labels.csv"
    _write_labels_csv(labels_csv, {val_stem: 0})

    with pytest.raises(ValueError, match="không còn video nào"):
        Phase2SequenceDataset(processed_dir=processed_dir, labels_csv=labels_csv, split="train")
