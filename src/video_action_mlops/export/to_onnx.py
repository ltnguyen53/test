"""export/to_onnx.py — xuất ONNX cho CẢ 2 giai đoạn, tách riêng 2 file
(spatial.onnx, temporal.onnx) — đúng ranh giới 2-stage đã giữ xuyên suốt
project (mục 3.1: contract giữa 2 giai đoạn là dữ liệu có version, mục 3.3:
export-friendly là class riêng).

RÀNG BUỘC BẮT BUỘC (mục 3.3): giai đoạn temporal PHẢI export từ
TemporalAggregatorExport (phiên 3.2), KHÔNG PHẢI TemporalAggregatorTrainable
— nn.MultiheadAttention (trong bản Trainable) không trace ổn định qua
torch.onnx.export ở nhiều phiên bản/backend. Checkpoint train ra
(train_phase2, phiên 5.2) là của bản Trainable — phải
convert_trainable_to_export() (phiên 3.2) TRƯỚC khi export, không được
export thẳng bản Trainable.

QUYẾT ĐỊNH KỸ THUẬT (cần biết): pin `dynamo=False` tường minh ở mọi lời gọi
torch.onnx.export(). Từ PyTorch 2.9, `dynamo=True` (exporter mới, dựa
torch.export, dùng `dynamic_shapes`) trở thành mặc định, thay cho API cũ
`dynamic_axes`. KHÔNG để mặc định tự chọn theo version torch cài đặt —
hành vi export phải NHẤT QUÁN bất kể máy nào chạy (nguyên tắc
reproducibility), không phải vì exporter mới kém hơn. Cân nhắc thử
`dynamo=True` sau khi có điều kiện tự kiểm chứng trên máy thật.
"""

from __future__ import annotations

from pathlib import Path

import torch

from video_action_mlops.config.schema import AppConfig
from video_action_mlops.models.registry import build_model
from video_action_mlops.models.temporal import (
    TemporalAggregatorTrainable,
    convert_trainable_to_export,
)


def export_spatial_onnx(cfg: AppConfig, checkpoint_path: str | Path, out_path: str | Path) -> Path:
    """Export SpatialBackbone (phase 1) sang ONNX.

    Backbone là CNN chuẩn (torchvision resnet18) — mục 3.3 chỉ cảnh báo về
    attention (nn.MultiheadAttention), không về CNN thường; không cần biến
    thể "export-friendly" riêng cho spatial như temporal.
    """
    model = build_model(cfg, stage="phase1", pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    dummy_input = torch.randn(1, 3, *cfg.data.frame_size)  # (B=1, C=3, H, W)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        str(out_path),
        input_names=["frame"],
        output_names=["embedding"],
        dynamic_axes={"frame": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17,
        dynamo=False,  # xem docstring module — pin tường minh, không theo mặc định version torch
    )
    return out_path


def export_temporal_onnx(cfg: AppConfig, checkpoint_path: str | Path, out_path: str | Path) -> Path:
    """Export temporal aggregator sang ONNX — BẮT BUỘC qua
    TemporalAggregatorExport (convert từ checkpoint Trainable, phiên 3.2),
    xem cảnh báo ở docstring module.

    Chỉ khai báo batch động (`dynamic_axes`), KHÔNG khai báo chiều thời
    gian (T) động — pipeline luôn sample đúng cfg.data.frames_per_video
    frame cho MỌI video (hybrid_frame_sample, phiên 2.1 luôn trả về đúng
    số lượng yêu cầu) nên T CỐ ĐỊNH là đúng theo đúng hợp đồng dữ liệu của
    project, không phải giới hạn kỹ thuật bị bỏ sót.
    """
    trainable = TemporalAggregatorTrainable(
        input_dim=cfg.phase2.input_dim,
        num_heads=cfg.phase2.num_heads,
        num_classes=cfg.data.num_classes,
    )
    trainable.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    trainable.eval()

    export_model = convert_trainable_to_export(trainable)
    export_model.eval()

    dummy_input = torch.randn(1, cfg.data.frames_per_video, cfg.phase2.input_dim)  # (B=1, T cố định, D)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        export_model,
        dummy_input,
        str(out_path),
        input_names=["embedding_sequence"],
        output_names=["logits"],
        dynamic_axes={"embedding_sequence": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    return out_path
