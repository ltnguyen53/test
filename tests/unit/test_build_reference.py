"""tests/unit/test_build_reference.py — test monitoring/build_reference.py
(phiên 14.7).

`ort.InferenceSession` (constructor, KHÔNG phải object session như
test_pipeline.py — build_reference() tự dựng session bên trong, không
nhận session làm tham số) và `is_val_sample` đều bị MONKEYPATCH — không
video/model ONNX thật nào bị chạm tới. Video test dùng cùng kỹ thuật màu
đặc đã xác nhận đúng ở test_pipeline.py (phiên 14.1): gradient tinh vi bị
codec mp4v nén mất tín hiệu.
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from video_action_mlops.monitoring import build_reference as br_module
from video_action_mlops.monitoring.build_reference import build_reference


def _make_test_video_file(
    directory: Path, name: str, frame_values: list[int], size=(64, 48)
) -> Path:
    path = directory / name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, size)
    for v in frame_values:
        frame_bgr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        frame_bgr[:, :, 2] = v
        writer.write(frame_bgr)
    writer.release()
    return path


def _make_test_config(tmp_path, frames_per_video=3, num_classes=4):
    config_content = f"""
project:
  name: test-project
  seed: 42
data:
  frames_per_video: {frames_per_video}
  frame_size: [32, 32]
  use_motion_channel: false
  num_classes: {num_classes}
phase1:
  embed_dim_out: 16
  epochs: 1
  learning_rate: 0.001
phase2:
  input_dim: 16
  num_heads: 2
  epochs: 1
  learning_rate: 0.001
"""
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(config_content)
    return config_path


class _FakeSpatialSession:
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def run(self, output_names, input_feed):
        frames = input_feed["frame"]
        t = frames.shape[0]
        rng = np.random.default_rng(0)
        return [rng.normal(size=(t, self.embed_dim)).astype(np.float32)]


class _FakeTemporalSession:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def run(self, output_names, input_feed):
        seq = input_feed["embedding_sequence"]
        batch = seq.shape[0]
        rng = np.random.default_rng(1)
        return [rng.normal(size=(batch, self.num_classes)).astype(np.float32)]


def _patch_fake_onnx(monkeypatch, num_classes=4, embed_dim=16):
    """build_reference() tự gọi ort.InferenceSession(path, providers=...) 2
    lần (spatial rồi temporal) -- đếm số lần gọi để trả đúng loại fake
    session theo thứ tự, không phụ thuộc path thật nào tồn tại trên đĩa."""
    calls = {"n": 0}

    def _fake_inference_session(path, providers=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeSpatialSession(embed_dim)
        return _FakeTemporalSession(num_classes)

    monkeypatch.setattr(br_module.ort, "InferenceSession", _fake_inference_session)
    return calls


# ---------- lọc đúng video val, bỏ qua video train ----------


def test_build_reference_filters_to_val_samples_only(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _make_test_video_file(raw_dir, "video_val_a.mp4", [0, 80, 160])
    _make_test_video_file(raw_dir, "video_val_b.mp4", [10, 90, 170])
    _make_test_video_file(raw_dir, "video_train_c.mp4", [20, 100, 180])

    monkeypatch.setattr(
        br_module,
        "is_val_sample",
        lambda stem: stem in {"video_val_a", "video_val_b"},
    )
    _patch_fake_onnx(monkeypatch)

    config_path = _make_test_config(tmp_path)
    output_csv = tmp_path / "reports" / "reference_stats.csv"

    build_reference(
        config_path=str(config_path),
        raw_dir=str(raw_dir),
        spatial_onnx_path="fake_spatial.onnx",
        temporal_onnx_path="fake_temporal.onnx",
        output_csv=str(output_csv),
    )

    with output_csv.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2  # CHỈ 2 video val, video_train_c bị loại


def test_build_reference_ignores_non_mp4_files(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _make_test_video_file(raw_dir, "video_val_a.mp4", [0, 80, 160])
    (raw_dir / "labels.csv").write_text("video_id,label\n")
    (raw_dir / "notes.txt").write_text("khong phai video")

    monkeypatch.setattr(br_module, "is_val_sample", lambda stem: True)
    _patch_fake_onnx(monkeypatch)

    output_csv = tmp_path / "out.csv"
    build_reference(
        config_path=str(_make_test_config(tmp_path)),
        raw_dir=str(raw_dir),
        spatial_onnx_path="x.onnx",
        temporal_onnx_path="y.onnx",
        output_csv=str(output_csv),
    )

    with output_csv.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1  # chi .mp4, labels.csv/notes.txt khong duoc doc nhu video


# ---------- raise khi không có video val ----------


def test_build_reference_raises_when_no_val_videos(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _make_test_video_file(raw_dir, "video_train_only.mp4", [0, 80, 160])

    monkeypatch.setattr(br_module, "is_val_sample", lambda stem: False)
    _patch_fake_onnx(monkeypatch)

    with pytest.raises(ValueError, match="video val"):
        build_reference(
            config_path=str(_make_test_config(tmp_path)),
            raw_dir=str(raw_dir),
            spatial_onnx_path="x.onnx",
            temporal_onnx_path="y.onnx",
            output_csv=str(tmp_path / "out.csv"),
        )


# ---------- CSV output đúng cột, đúng nội dung ----------


def test_build_reference_csv_has_exact_expected_columns(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _make_test_video_file(raw_dir, "v1.mp4", [0, 80, 160, 240])

    monkeypatch.setattr(br_module, "is_val_sample", lambda stem: True)
    _patch_fake_onnx(monkeypatch, num_classes=4, embed_dim=16)

    output_csv = tmp_path / "out.csv"
    build_reference(
        config_path=str(_make_test_config(tmp_path, num_classes=4)),
        raw_dir=str(raw_dir),
        spatial_onnx_path="x.onnx",
        temporal_onnx_path="y.onnx",
        output_csv=str(output_csv),
    )

    with output_csv.open() as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ["input_mean_motion_score", "confidence", "predicted_class"]
        row = next(reader)

    assert 0 <= int(row["predicted_class"]) < 4
    assert 0.0 <= float(row["confidence"]) <= 1.0
    assert float(row["input_mean_motion_score"]) >= 0.0


def test_build_reference_creates_output_parent_dir(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _make_test_video_file(raw_dir, "v1.mp4", [0, 80, 160])
    monkeypatch.setattr(br_module, "is_val_sample", lambda stem: True)
    _patch_fake_onnx(monkeypatch)

    output_csv = tmp_path / "sub" / "dir" / "out.csv"
    assert not output_csv.parent.exists()

    result_path = build_reference(
        config_path=str(_make_test_config(tmp_path)),
        raw_dir=str(raw_dir),
        spatial_onnx_path="x.onnx",
        temporal_onnx_path="y.onnx",
        output_csv=str(output_csv),
    )

    assert output_csv.exists()
    assert result_path == output_csv


def test_build_reference_processes_videos_in_sorted_order(tmp_path, monkeypatch):
    """sorted(Path(raw_dir).glob(...)) trong build_reference() -- thu tu xu
    ly phai on dinh (khong phu thuoc thu tu OS tra ve tu filesystem), quan
    trong de output CSV lap lai giong nhau qua nhieu lan chay."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _make_test_video_file(raw_dir, "z_video.mp4", [5, 5, 5])
    _make_test_video_file(raw_dir, "a_video.mp4", [9, 9, 9])

    monkeypatch.setattr(br_module, "is_val_sample", lambda stem: True)
    _patch_fake_onnx(monkeypatch)

    printed_lines: list[str] = []
    monkeypatch.setattr(
        br_module,
        "print",
        lambda *a: printed_lines.append(" ".join(str(x) for x in a)),
        raising=False,
    )

    build_reference(
        config_path=str(_make_test_config(tmp_path)),
        raw_dir=str(raw_dir),
        spatial_onnx_path="x.onnx",
        temporal_onnx_path="y.onnx",
        output_csv=str(tmp_path / "out.csv"),
    )

    per_video_lines = [
        line for line in printed_lines if "a_video.mp4" in line or "z_video.mp4" in line
    ]
    assert "a_video.mp4" in per_video_lines[0]
    assert "z_video.mp4" in per_video_lines[1]
