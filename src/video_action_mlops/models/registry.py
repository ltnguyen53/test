"""Model registry — 1 điểm build_model() DUY NHẤT để tạo model từ config.

Đây là nơi duy nhất biết cách map từ AppConfig (config/schema.py, phiên
1.2) sang instance model cụ thể (models/spatial.py, models/temporal.py).
Không script/module nào khác (training/, serving/, ...) được tự gọi thẳng
`SpatialBackbone(...)` hay `TemporalAggregatorTrainable(...)` — luôn qua
build_model() để đổi kiến trúc (vd đổi kích thước, đổi variant) chỉ cần
sửa CONFIG, không phải sửa rải rác nhiều nơi gọi model (mục 2, bảng
"Model build rải rác nhiều nơi" trong roadmap — bug điển hình thực tế).

LƯU Ý: đây KHÔNG phải MLflow Model Registry (stage None/Staging/
Production, mục 3.2 roadmap, sẽ làm ở tuần 5 khi có training/callbacks.py)
— 2 khái niệm "registry" khác nhau, trùng tên do quy ước ngành, không phải
lỗi đặt tên trong project này.
"""

from __future__ import annotations

from typing import Literal

from torch import nn

from video_action_mlops.config.schema import AppConfig
from video_action_mlops.models.spatial import SpatialBackbone
from video_action_mlops.models.temporal import (
    TemporalAggregatorExport,
    TemporalAggregatorTrainable,
)

ModelStage = Literal["phase1", "phase2"]
ModelVariant = Literal["trainable", "export"]


def build_model(
    cfg: AppConfig,
    stage: ModelStage,
    *,
    pretrained: bool = False,
    variant: ModelVariant = "trainable",
) -> nn.Module:
    """Build đúng model cho đúng giai đoạn, MỌI kích thước đọc từ cfg — không hard-code.

    stage="phase1" -> SpatialBackbone(embed_dim_out=cfg.phase1.embed_dim_out).
    stage="phase2" -> TemporalAggregatorTrainable hoặc TemporalAggregatorExport
                       tuỳ `variant`, đọc cfg.phase2.input_dim/num_heads và
                       cfg.data.num_classes (num_classes nằm ở DataConfig vì
                       nó phụ thuộc DATASET, không phải kiến trúc model).

    Raise ValueError ngay nếu stage/variant không hợp lệ (fail-fast, cùng
    triết lý mọi hàm khác trong project — gõ nhầm string không được âm
    thầm trả về None hay model sai).
    """
    if stage == "phase1":
        return SpatialBackbone(embed_dim_out=cfg.phase1.embed_dim_out, pretrained=pretrained)

    if stage == "phase2":
        if variant == "trainable":
            return TemporalAggregatorTrainable(
                input_dim=cfg.phase2.input_dim,
                num_heads=cfg.phase2.num_heads,
                num_classes=cfg.data.num_classes,
            )
        if variant == "export":
            return TemporalAggregatorExport(
                input_dim=cfg.phase2.input_dim,
                num_heads=cfg.phase2.num_heads,
                num_classes=cfg.data.num_classes,
            )
        raise ValueError(f"variant '{variant}' không hợp lệ, phải là 'trainable' hoặc 'export'")

    raise ValueError(f"stage '{stage}' không hợp lệ, phải là 'phase1' hoặc 'phase2'")
