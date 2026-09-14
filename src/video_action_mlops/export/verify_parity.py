"""export/verify_parity.py — xác nhận PyTorch vs ONNX Runtime ra CÙNG SỐ
trong sai số cho phép, cho CẢ 2 model export ở to_onnx.py.

Đây là bước KHÔNG ĐƯỢC BỎ QUA: 1 file .onnx sinh ra không lỗi KHÔNG có
nghĩa nó tính đúng (cùng tinh thần với
test_convert_is_lossless_within_tolerance ở phiên 3.2 — export cũng là 1
phép "viết lại cùng phép tính", cần verify riêng, không suy luận từ việc
convert_trainable_to_export đã đúng).

Cố ý dùng batch=2 khi verify (khác batch=1 lúc export ở to_onnx.py) — để
xác nhận `dynamic_axes` THẬT SỰ hoạt động, không chỉ đúng ngẫu nhiên ở
đúng batch size lúc export.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from video_action_mlops.config.schema import AppConfig
from video_action_mlops.models.registry import build_model
from video_action_mlops.models.temporal import (
    TemporalAggregatorTrainable,
    convert_trainable_to_export,
)


class ParityError(RuntimeError):
    """Raised khi PyTorch và ONNX Runtime lệch nhau quá atol — file .onnx
    KHÔNG được dùng tiếp trong trường hợp này."""


def verify_spatial_parity(
    cfg: AppConfig, checkpoint_path: str | Path, onnx_path: str | Path, atol: float = 1e-4
) -> float:
    model = build_model(cfg, stage="phase1", pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    torch.manual_seed(0)
    x = torch.randn(2, 3, *cfg.data.frame_size)  # batch=2, CỐ Ý khác batch=1 lúc export

    with torch.no_grad():
        out_torch = model(x).numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out_onnx = session.run(None, {"frame": x.numpy()})[0]

    max_diff = float(np.abs(out_torch - out_onnx).max())
    if max_diff > atol:
        raise ParityError(
            f"Spatial parity FAIL: lệch {max_diff} > atol={atol} — ONNX export "
            f"KHÔNG khớp PyTorch, KHÔNG dùng file {onnx_path}."
        )
    return max_diff


def verify_temporal_parity(
    cfg: AppConfig, checkpoint_path: str | Path, onnx_path: str | Path, atol: float = 1e-4
) -> float:
    trainable = TemporalAggregatorTrainable(
        input_dim=cfg.phase2.input_dim,
        num_heads=cfg.phase2.num_heads,
        num_classes=cfg.data.num_classes,
    )
    trainable.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    trainable.eval()
    export_model = convert_trainable_to_export(trainable)
    export_model.eval()

    torch.manual_seed(0)
    # batch=2 (khác batch=1 lúc export), T PHẢI đúng frames_per_video —
    # graph ONNX cố định T (xem docstring to_onnx.py), truyền T khác sẽ lỗi
    # ngay ở onnxruntime, không phải lỗi parity — đúng ý đồ thiết kế.
    x = torch.randn(2, cfg.data.frames_per_video, cfg.phase2.input_dim)

    with torch.no_grad():
        out_torch = export_model(x).numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out_onnx = session.run(None, {"embedding_sequence": x.numpy()})[0]

    max_diff = float(np.abs(out_torch - out_onnx).max())
    if max_diff > atol:
        raise ParityError(
            f"Temporal parity FAIL: lệch {max_diff} > atol={atol} — ONNX export "
            f"KHÔNG khớp PyTorch, KHÔNG dùng file {onnx_path}."
        )
    return max_diff
