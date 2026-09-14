"""tests/unit/test_pipeline.py — test inference/pipeline.py (phiên 14.1).

Module này được dùng lại ở serving/api.py (8.1), demo/gradio_app.py
(10.2), monitoring/build_reference.py (11.1) — ưu tiên viết test đầu tiên
trong Tuần 14 vì hỏng ở đây ảnh hưởng rộng nhất trong cả project.

Video test dùng MÀU ĐẶC (không phải gradient tinh vi) — kỹ thuật đã xác
nhận đúng ở phiên 8.1/10.2: gradient nhỏ bị codec mp4v (lossy) nén mất
gần hết tín hiệu, từng gây 1 lỗi test giả ở phiên 8.1.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from video_action_mlops.config.loader import load_config
from video_action_mlops.inference import pipeline as pipeline_module
from video_action_mlops.inference.pipeline import (
    decode_video_bytes,
    resize_frames_np,
    run_two_stage_inference,
    softmax,
)


def _make_test_video_bytes(frame_values: list[int], size: tuple[int, int] = (64, 48)) -> bytes:
    """size = (width, height) theo quy ước cv2.VideoWriter. Mỗi frame là 1
    màu đặc (kênh đỏ = frame_values[i]) — dễ phân biệt qua nén lossy."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, 10.0, size)
    for v in frame_values:
        frame_bgr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        frame_bgr[:, :, 2] = v  # opencv ghi BGR, kênh đỏ là index 2
        writer.write(frame_bgr)
    writer.release()

    video_bytes = Path(tmp_path).read_bytes()
    Path(tmp_path).unlink()
    return video_bytes


def _make_test_config(tmp_path, frames_per_video=3, num_classes=4, use_motion_channel=False):
    """Sinh 1 AppConfig THẬT (qua load_config, không mock) — nhỏ gọn, đủ
    field bắt buộc theo schema.py (phiên 1.2)."""
    config_content = f"""
project:
  name: test-project
  seed: 42
data:
  frames_per_video: {frames_per_video}
  frame_size: [32, 32]
  use_motion_channel: {str(use_motion_channel).lower()}
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
    return load_config(config_path)


class _FakeSpatialSession:
    """Giả lập onnxruntime.InferenceSession cho spatial.onnx — trả
    embedding NGẪU NHIÊN có ĐÚNG SHAPE (test này kiểm tra logic glue của
    pipeline.py, không kiểm tra model thật có chính xác không — đó là
    việc của test_export.py, phiên 14.4)."""

    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def run(self, output_names, input_feed):
        frames = input_feed["frame"]  # (T, C, H, W)
        t = frames.shape[0]
        rng = np.random.default_rng(0)
        return [rng.normal(size=(t, self.embed_dim)).astype(np.float32)]


class _FakeTemporalSession:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def run(self, output_names, input_feed):
        seq = input_feed["embedding_sequence"]  # (1, T, D)
        batch = seq.shape[0]
        rng = np.random.default_rng(1)
        return [rng.normal(size=(batch, self.num_classes)).astype(np.float32)]


# ---------- decode_video_bytes ----------


def test_decode_video_bytes_shape_dtype_and_order():
    frame_values = [0, 60, 120, 180, 240]
    video_bytes = _make_test_video_bytes(frame_values)

    frames = decode_video_bytes(video_bytes)

    assert frames.shape == (5, 48, 64, 3)
    assert frames.dtype == np.uint8

    actual_r = [int(frames[i, :, :, 0].mean()) for i in range(5)]
    for actual, expected in zip(actual_r, frame_values, strict=True):
        assert abs(actual - expected) < 20  # dung sai nén lossy, không so bằng tuyệt đối
    assert actual_r == sorted(actual_r), "thứ tự frame không được đảo lộn"


def test_decode_video_bytes_invalid_input_raises():
    # Hành vi cụ thể (loại lỗi nào) tuỳ backend opencv của máy chạy test —
    # không match message cụ thể, chỉ đảm bảo KHÔNG crash âm thầm mà raise
    # đúng loại lỗi đã khai (ValueError).
    with pytest.raises(ValueError):
        decode_video_bytes(b"day khong phai video that")


def test_decode_video_bytes_opens_but_zero_frames_raises(monkeypatch):
    """Nhánh KHÁC với test ở trên: container MỞ ĐƯỢC (isOpened() -> True)
    nhưng không đọc được frame nào (read() -> False ngay từ đầu) -- 1
    video "mở được nhưng rỗng" thật khó tái tạo bằng cv2.VideoWriter thật
    (thử tay: ghi 0 frame -> file không mở được, rơi vào nhánh isOpened()
    == False, KHÁC nhánh cần test). Fake VideoCapture ở đúng ranh giới cv2
    để cô lập nhánh "Video không có frame nào" này."""

    class _FakeCapture:
        def isOpened(self):
            return True

        def read(self):
            return False, None

        def release(self):
            pass

    monkeypatch.setattr(pipeline_module.cv2, "VideoCapture", lambda path: _FakeCapture())

    with pytest.raises(ValueError, match="không có frame"):
        decode_video_bytes(b"gia lap: bytes bat ky, VideoCapture da bi fake")


# ---------- resize_frames_np ----------


def test_resize_frames_np_changes_shape_keeps_dtype():
    frames = np.zeros((3, 48, 64, 3), dtype=np.uint8)
    resized = resize_frames_np(frames, (32, 32))
    assert resized.shape == (3, 32, 32, 3)
    assert resized.dtype == np.uint8


# ---------- softmax ----------


def test_softmax_sums_to_one_and_picks_max():
    x = np.array([1.0, 2.0, 3.0])
    probs = softmax(x)
    assert abs(probs.sum() - 1.0) < 1e-6
    assert int(probs.argmax()) == 2


def test_softmax_numerically_stable_on_large_values():
    # x - x.max() trước exp (xem pipeline.py) -> không overflow dù input lớn
    x = np.array([1000.0, 1001.0, 1002.0])
    probs = softmax(x)
    assert not np.isnan(probs).any()
    assert abs(probs.sum() - 1.0) < 1e-6


# ---------- run_two_stage_inference (end-to-end, model giả) ----------


def test_run_two_stage_inference_end_to_end(tmp_path):
    cfg = _make_test_config(tmp_path, frames_per_video=3, num_classes=4)
    video_bytes = _make_test_video_bytes([0, 80, 160, 240, 255])  # 5 frame, đủ để sample còn 3

    result = run_two_stage_inference(
        video_bytes, cfg, _FakeSpatialSession(embed_dim=16), _FakeTemporalSession(num_classes=4)
    )

    assert set(result.keys()) == {
        "predicted_class",
        "probabilities",
        "frame_count",
        "mean_motion_score",
        "confidence",
    }
    assert 0 <= result["predicted_class"] < 4
    assert len(result["probabilities"]) == 4
    assert abs(sum(result["probabilities"]) - 1.0) < 1e-4
    assert result["frame_count"] == 5  # đếm TRƯỚC khi sample xuống 3
    assert result["mean_motion_score"] >= 0.0
    assert 0.0 <= result["confidence"] <= 1.0


def test_run_two_stage_inference_raises_when_motion_channel_enabled(tmp_path):
    cfg = _make_test_config(tmp_path, use_motion_channel=True)
    video_bytes = _make_test_video_bytes([0, 100])

    with pytest.raises(NotImplementedError, match="use_motion_channel"):
        run_two_stage_inference(
            video_bytes, cfg, _FakeSpatialSession(16), _FakeTemporalSession(4)
        )
