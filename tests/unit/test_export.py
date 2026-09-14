"""tests/unit/test_export.py — test export/to_onnx.py + export/verify_parity.py
(phiên 14.4).

Test THẬT (KHÔNG mock torch.onnx.export/onnxruntime) — dùng model KÍCH
THƯỚC SIÊU NHỎ (embed_dim_out=8, frames_per_video=2, frame_size=16x16) để
export/verify chạy trong vài giây thay vì vài phút, nhưng vẫn là hành vi
thật, không phải giả lập bằng mock. Cần torch + torchvision + onnxruntime
cài đủ (`pip install -e ".[export]"` — torch/torchvision đã là dependency
chính, không cần thêm).
"""

from __future__ import annotations

import pytest
import torch

from video_action_mlops.config.schema import AppConfig
from video_action_mlops.export.to_onnx import export_spatial_onnx, export_temporal_onnx
from video_action_mlops.export.verify_parity import (
    ParityError,
    verify_spatial_parity,
    verify_temporal_parity,
)
from video_action_mlops.models.registry import build_model
from video_action_mlops.models.temporal import TemporalAggregatorTrainable


def _make_tiny_config() -> AppConfig:
    """Kích thước siêu nhỏ — chỉ để export/verify chạy nhanh, không phản
    ánh config thật (configs/base.yaml)."""
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


# ---------- Spatial ----------


def test_export_and_verify_spatial_onnx_end_to_end(tmp_path):
    cfg = _make_tiny_config()
    model = build_model(cfg, stage="phase1", pretrained=False)
    checkpoint_path = tmp_path / "phase1.pt"
    torch.save(model.state_dict(), checkpoint_path)

    onnx_path = tmp_path / "spatial.onnx"
    result_path = export_spatial_onnx(cfg, checkpoint_path, onnx_path)

    assert result_path == onnx_path
    assert onnx_path.exists()
    assert onnx_path.stat().st_size > 0

    max_diff = verify_spatial_parity(cfg, checkpoint_path, onnx_path)
    assert max_diff < 1e-4


def test_verify_spatial_parity_detects_real_mismatch(tmp_path):
    """PHẢN CHỨNG (cùng tinh thần các test 'phải phát hiện lỗi' từ phiên
    2.2): export ONNX cho model A, verify bằng checkpoint model B (trọng
    số random khác hẳn) — PHẢI báo ParityError, không được im lặng pass.
    Đây là test quan trọng nhất file này — verify_parity() vô dụng nếu
    không tự phát hiện được trường hợp sai thật."""
    cfg = _make_tiny_config()

    model_a = build_model(cfg, stage="phase1", pretrained=False)
    checkpoint_a = tmp_path / "a.pt"
    torch.save(model_a.state_dict(), checkpoint_a)
    onnx_path = tmp_path / "a.onnx"
    export_spatial_onnx(cfg, checkpoint_a, onnx_path)

    model_b = build_model(cfg, stage="phase1", pretrained=False)  # trọng số random KHÁC model_a
    checkpoint_b = tmp_path / "b.pt"
    torch.save(model_b.state_dict(), checkpoint_b)

    with pytest.raises(ParityError):
        verify_spatial_parity(cfg, checkpoint_b, onnx_path)


# ---------- Temporal ----------


def test_export_and_verify_temporal_onnx_end_to_end(tmp_path):
    cfg = _make_tiny_config()
    trainable = TemporalAggregatorTrainable(
        input_dim=cfg.phase2.input_dim,
        num_heads=cfg.phase2.num_heads,
        num_classes=cfg.data.num_classes,
    )
    checkpoint_path = tmp_path / "phase2.pt"
    torch.save(trainable.state_dict(), checkpoint_path)

    onnx_path = tmp_path / "temporal.onnx"
    export_temporal_onnx(cfg, checkpoint_path, onnx_path)

    assert onnx_path.exists()

    # Đây chính là test quan trọng nhất mục 3.3 (roadmap) ở tầng ONNX:
    # convert_trainable_to_export() (phiên 3.2) + export qua ONNX (phiên
    # 7.1) phải CÙNG cho kết quả khớp PyTorch gốc.
    max_diff = verify_temporal_parity(cfg, checkpoint_path, onnx_path)
    assert max_diff < 1e-4


def test_verify_temporal_parity_detects_real_mismatch(tmp_path):
    cfg = _make_tiny_config()

    trainable_a = TemporalAggregatorTrainable(input_dim=8, num_heads=2, num_classes=3)
    checkpoint_a = tmp_path / "a.pt"
    torch.save(trainable_a.state_dict(), checkpoint_a)
    onnx_path = tmp_path / "a.onnx"
    export_temporal_onnx(cfg, checkpoint_a, onnx_path)

    trainable_b = TemporalAggregatorTrainable(input_dim=8, num_heads=2, num_classes=3)
    checkpoint_b = tmp_path / "b.pt"
    torch.save(trainable_b.state_dict(), checkpoint_b)

    with pytest.raises(ParityError):
        verify_temporal_parity(cfg, checkpoint_b, onnx_path)
