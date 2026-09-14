"""tests/unit/test_serving.py — test serving/api.py + serving/schemas.py
(phiên 14.3). Lần đầu project có test cho tầng HTTP.

CỐ Ý KHÔNG dùng `with TestClient(app) as client:` — cú pháp đó kích hoạt
`lifespan` THẬT của FastAPI, cố `load_config()`/`InferenceSession()` từ
đường dẫn hard-code trong api.py (configs/base.yaml, models/*.onnx) —
không có trong môi trường test. Thay vào đó, tự điền `_state` (dict
module-level, phiên 8.1) qua `monkeypatch.setitem`, tương đương "đã qua
bước khởi động" mà không cần model thật trên đĩa.

Cần package `httpx` (xem pyproject.toml phiên 14.3) — Starlette gần đây
deprecate httpx cho TestClient (chuyển sang "httpx2"), có thể thấy
DeprecationWarning khi chạy — KHÔNG phải lỗi, chỉ là cảnh báo.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from video_action_mlops.serving import api
from video_action_mlops.serving.schemas import HealthResponse, PredictionResponse


def _make_test_video_bytes(num_frames: int = 5, size: tuple[int, int] = (64, 48)) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, 10.0, size)
    for i in range(num_frames):
        value = (i * 40) % 255
        frame = np.full((size[1], size[0], 3), fill_value=value, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    data = Path(tmp_path).read_bytes()
    Path(tmp_path).unlink()
    return data


class _FakeSpatialSession:
    def run(self, output_names, input_feed):
        t = input_feed["frame"].shape[0]
        return [np.random.default_rng(0).normal(size=(t, 16)).astype(np.float32)]


class _FakeTemporalSession:
    def run(self, output_names, input_feed):
        batch = input_feed["embedding_sequence"].shape[0]
        return [np.random.default_rng(1).normal(size=(batch, 4)).astype(np.float32)]


class _FakeDataConfig:
    """Dùng __init__ (KHÔNG phải class attribute trực tiếp) — tránh lỗi
    kinh điển: nếu _FakeCfg.data = _FakeDataConfig() là thuộc tính CẤP
    CLASS, mọi instance _FakeCfg() sẽ CHIA SẺ CHUNG 1 object data, sửa ở
    test này rò rỉ sang test khác chạy sau."""

    def __init__(self, frame_size=(32, 32), frames_per_video=3, use_motion_channel=False):
        self.frame_size = frame_size
        self.frames_per_video = frames_per_video
        self.use_motion_channel = use_motion_channel


class _FakeCfg:
    def __init__(self, **data_kwargs):
        self.data = _FakeDataConfig(**data_kwargs)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setitem(api._state, "cfg", _FakeCfg())
    monkeypatch.setitem(api._state, "spatial_session", _FakeSpatialSession())
    monkeypatch.setitem(api._state, "temporal_session", _FakeTemporalSession())
    return TestClient(api.app)


# ---------- /healthz ----------


def test_healthz_returns_ok(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ---------- /predict ----------


def test_predict_success(client):
    video_bytes = _make_test_video_bytes(num_frames=5)
    response = client.post("/predict", files={"file": ("test.mp4", video_bytes, "video/mp4")})

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"predicted_class", "probabilities"}
    assert 0 <= body["predicted_class"] < 4
    assert len(body["probabilities"]) == 4
    assert abs(sum(body["probabilities"]) - 1.0) < 1e-4


def test_predict_rejects_wrong_extension(client):
    response = client.post(
        "/predict", files={"file": ("test.txt", b"khong phai video", "text/plain")}
    )
    assert response.status_code == 400


def test_predict_rejects_missing_filename(client):
    # UploadFile khong co filename (truong hop hiem nhung hop le ve mat HTTP)
    response = client.post("/predict", files={"file": ("", b"data", "application/octet-stream")})
    assert response.status_code == 400


def test_predict_returns_500_when_motion_channel_enabled(client, monkeypatch):
    monkeypatch.setitem(api._state, "cfg", _FakeCfg(use_motion_channel=True))

    video_bytes = _make_test_video_bytes(num_frames=3)
    response = client.post("/predict", files={"file": ("test.mp4", video_bytes, "video/mp4")})

    assert response.status_code == 500


def test_predict_rejects_corrupt_video(client):
    response = client.post(
        "/predict", files={"file": ("test.mp4", b"day khong phai video that", "video/mp4")}
    )
    assert response.status_code == 400


# ---------- schemas.py ----------


def test_health_response_schema():
    r = HealthResponse(status="ok")
    assert r.status == "ok"


def test_prediction_response_rejects_negative_class():
    with pytest.raises(ValidationError):
        PredictionResponse(predicted_class=-1, probabilities=[0.5, 0.5])


def test_prediction_response_accepts_valid_data():
    r = PredictionResponse(predicted_class=2, probabilities=[0.1, 0.2, 0.7])
    assert r.predicted_class == 2
    assert r.probabilities == [0.1, 0.2, 0.7]


# ---------- lifespan() thật — phát hiện qua coverage report (api.py 88%,
# thiếu đúng khối lifespan) sau khi bị hỏi "còn gì chưa implement". Fixture
# `client` ở trên CỐ TÌNH né lifespan thật (đọc docstring đầu file) — hợp
# lý cho test /predict, nhưng để lại toàn bộ cơ chế STARTUP (load_config +
# 2 lần InferenceSession + dọn _state lúc shutdown) chưa từng được verify.
# Test dưới đây làm điều ngược lại: trỏ CONFIG_PATH/*_ONNX_PATH vào file
# giả trong tmp_path + fake InferenceSession, rồi để lifespan THẬT chạy
# qua `with TestClient(app) as client:`. ----------


def test_lifespan_loads_config_and_sessions_into_state(tmp_path, monkeypatch):
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(
        """
project:
  name: test
  seed: 0
data:
  frames_per_video: 4
  frame_size: [32, 32]
  use_motion_channel: false
  num_classes: 3
phase1:
  embed_dim_out: 8
  epochs: 1
  learning_rate: 0.001
phase2:
  input_dim: 8
  num_heads: 2
  epochs: 1
  learning_rate: 0.001
"""
    )
    monkeypatch.setattr(api, "CONFIG_PATH", str(config_path))
    monkeypatch.setattr(api, "SPATIAL_ONNX_PATH", "fake_spatial.onnx")
    monkeypatch.setattr(api, "TEMPORAL_ONNX_PATH", "fake_temporal.onnx")
    monkeypatch.setattr(
        api.ort, "InferenceSession", lambda path, providers=None: _FakeSpatialSession()
    )

    assert api._state == {}  # sanity-check: chua khoi dong, _state phai rong

    with TestClient(api.app) as test_client:
        # Trong khoi "with" -- lifespan da chay xong phan truoc "yield".
        assert api._state["cfg"].data.num_classes == 3
        assert api._state["spatial_session"] is not None
        assert api._state["temporal_session"] is not None
        response = test_client.get("/healthz")
        assert response.status_code == 200

    # Ra khoi "with" -- lifespan chay tiep phan SAU "yield" (_state.clear()).
    assert api._state == {}
