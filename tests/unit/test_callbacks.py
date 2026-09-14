"""tests/unit/test_callbacks.py — test training/callbacks.py (phiên 14.5).

mlflow được MONKEYPATCH — import mlflow thật cần thiết (callbacks.py
import ở module level, nhẹ, không tự kết nối gì khi chỉ import), nhưng
mọi hàm GỌI MẠNG (start_run, register_model, MlflowClient...) đều bị thay
bằng Mock. Không có test nào trong file này chạm tới MLflow server thật.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

from video_action_mlops.training import callbacks

# ---------- configure_mlflow ----------


def test_configure_mlflow_raises_when_env_vars_missing(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("DAGSHUB_USERNAME", raising=False)
    monkeypatch.delenv("DAGSHUB_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="Thiếu biến môi trường"):
        callbacks.configure_mlflow()


def test_configure_mlflow_bridges_env_vars_and_sets_tracking_uri(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://dagshub.com/u/r.mlflow")
    monkeypatch.setenv("DAGSHUB_USERNAME", "myuser")
    monkeypatch.setenv("DAGSHUB_TOKEN", "mytoken")
    monkeypatch.delenv("MLFLOW_TRACKING_USERNAME", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_PASSWORD", raising=False)

    fake_set_tracking_uri = MagicMock()
    monkeypatch.setattr(callbacks.mlflow, "set_tracking_uri", fake_set_tracking_uri)

    callbacks.configure_mlflow()

    # Bridge DAGSHUB_* -> MLFLOW_TRACKING_* (phiên 5.3) — tên biến mlflow
    # thật sự cần, khác tên đã đặt ở .env.example (phiên 1.3).
    assert os.environ["MLFLOW_TRACKING_USERNAME"] == "myuser"
    assert os.environ["MLFLOW_TRACKING_PASSWORD"] == "mytoken"
    fake_set_tracking_uri.assert_called_once_with("https://dagshub.com/u/r.mlflow")


# ---------- get_or_create_parent_run ----------


def test_get_or_create_parent_run_creates_once_then_reuses(tmp_path, monkeypatch):
    run_id_file = tmp_path / "parent_run_id.txt"

    fake_run = MagicMock()
    fake_run.info.run_id = "fake-run-123"
    fake_context = MagicMock()
    fake_context.__enter__ = MagicMock(return_value=fake_run)
    fake_context.__exit__ = MagicMock(return_value=False)
    fake_start_run = MagicMock(return_value=fake_context)
    monkeypatch.setattr(callbacks.mlflow, "start_run", fake_start_run)
    monkeypatch.setattr(callbacks.mlflow, "log_dict", MagicMock())

    fake_cfg = MagicMock()
    fake_cfg.model_dump.return_value = {"project": {"name": "test"}}

    # Lần 1: file run_id CHƯA tồn tại -> phải tạo run cha mới
    run_id_1 = callbacks.get_or_create_parent_run(fake_cfg, parent_run_id_file=run_id_file)
    assert run_id_1 == "fake-run-123"
    assert fake_start_run.call_count == 1
    assert run_id_file.read_text().strip() == "fake-run-123"

    # Lần 2: file ĐÃ tồn tại -> PHẢI đọc lại, TUYỆT ĐỐI không gọi start_run
    # thêm lần nào — đây là phần quan trọng nhất mục 3.2 (cặp checkpoint
    # dưới đúng 1 run cha).
    run_id_2 = callbacks.get_or_create_parent_run(fake_cfg, parent_run_id_file=run_id_file)
    assert run_id_2 == "fake-run-123"
    assert fake_start_run.call_count == 1  # KHÔNG tăng so với lần 1


# ---------- log_phase_result ----------


def test_log_phase_result_logs_params_metrics_artifact_and_filters_non_numeric(monkeypatch):
    fake_nested_cm = MagicMock()
    fake_nested_cm.__enter__ = MagicMock(return_value=MagicMock())
    fake_nested_cm.__exit__ = MagicMock(return_value=False)
    fake_parent_cm = MagicMock()
    fake_parent_cm.__enter__ = MagicMock(return_value=MagicMock())
    fake_parent_cm.__exit__ = MagicMock(return_value=False)

    call_log = []

    def fake_start_run(run_id=None, run_name=None, nested=False):
        call_log.append({"run_id": run_id, "run_name": run_name, "nested": nested})
        return fake_nested_cm if nested else fake_parent_cm

    monkeypatch.setattr(callbacks.mlflow, "start_run", fake_start_run)
    fake_log_params = MagicMock()
    fake_log_metrics = MagicMock()
    fake_log_artifact = MagicMock()
    monkeypatch.setattr(callbacks.mlflow, "log_params", fake_log_params)
    monkeypatch.setattr(callbacks.mlflow, "log_metrics", fake_log_metrics)
    monkeypatch.setattr(callbacks.mlflow, "log_artifact", fake_log_artifact)

    history = [
        {"epoch": 0, "train_loss": 1.0, "train_acc": 0.5, "unfrozen_up_to": None},
        {"epoch": 1, "train_loss": 0.8, "train_acc": 0.6, "unfrozen_up_to": "layer4"},
    ]

    callbacks.log_phase_result(
        phase_name="phase1",
        parent_run_id="parent-123",
        params={"epochs": 2},
        history=history,
        checkpoint_path="fake_checkpoint.pt",
    )

    # start_run PHẢI được gọi đúng 2 lần: 1 lần mở lại run cha (run_id=...,
    # nested=False), 1 lần mở nested run con (nested=True) — đúng kỹ thuật
    # "multi-worker" đã ghi trong docstring callbacks.py (phiên 5.3).
    assert call_log[0] == {"run_id": "parent-123", "run_name": None, "nested": False}
    assert call_log[1] == {"run_id": None, "run_name": "phase1", "nested": True}

    fake_log_params.assert_called_once_with({"epochs": 2})
    assert fake_log_metrics.call_count == 2  # 1 lần / epoch

    # "unfrozen_up_to" là string (không phải int/float) — PHẢI bị loại
    # khỏi log_metrics (chỉ nhận số), xem thiết kế đã ghi ở phiên 5.3.
    first_call_metrics = fake_log_metrics.call_args_list[0][0][0]
    assert "unfrozen_up_to" not in first_call_metrics
    assert "train_loss" in first_call_metrics and "train_acc" in first_call_metrics

    fake_log_artifact.assert_called_once_with("fake_checkpoint.pt")


# ---------- register_model_at_parent ----------


def test_register_model_at_parent_calls_mlflow_register_with_correct_uri(monkeypatch):
    """Verify đúng luồng MLflow 3.x (vá gap thật phát hiện khi chạy dvc
    repro thật đầu-đến-cuối lần đầu — xem docstring register_model_at_parent):
    tạo LoggedModel qua create_external_model() TRƯỚC, dùng model_uri của
    NÓ (không phải runs:/.../config_snapshot.json cũ, đã lỗi thật với
    MLflow 3.x) để register_model()."""
    fake_logged_model = MagicMock()
    fake_logged_model.model_uri = "models:/m-abc123"
    fake_create_external_model = MagicMock(return_value=fake_logged_model)
    monkeypatch.setattr(callbacks.mlflow, "create_external_model", fake_create_external_model)

    fake_result = MagicMock()
    fake_result.version = "3"
    fake_register_model = MagicMock(return_value=fake_result)
    monkeypatch.setattr(callbacks.mlflow, "register_model", fake_register_model)

    version = callbacks.register_model_at_parent("parent-123", model_name="my-model")

    assert version == "3"
    fake_create_external_model.assert_called_once_with(name="my-model", source_run_id="parent-123")
    fake_register_model.assert_called_once_with(model_uri="models:/m-abc123", name="my-model")


# ---------- promote_to_staging ----------


def test_promote_to_staging_calls_client_transition(monkeypatch):
    fake_client_instance = MagicMock()
    fake_client_class = MagicMock(return_value=fake_client_instance)
    monkeypatch.setattr(callbacks.mlflow.tracking, "MlflowClient", fake_client_class)

    callbacks.promote_to_staging("my-model", "3")

    fake_client_instance.transition_model_version_stage.assert_called_once_with(
        name="my-model", version="3", stage="Staging"
    )
