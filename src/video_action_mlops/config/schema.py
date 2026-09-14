"""Typed, fail-fast configuration schema cho toàn bộ project.

Đây là NƠI DUY NHẤT định nghĩa "config hợp lệ" nghĩa là gì. Không module
nào khác (training/, serving/, ...) được định nghĩa Config class riêng —
mọi nơi chỉ import từ đây. Đây là hệ quả trực tiếp của nguyên tắc bất biến
#1 (src/ không phụ thuộc hạ tầng): schema không biết DVC, MLflow, FastAPI
là gì, nó chỉ biết "giá trị nào hợp lệ".
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class ProjectConfig(BaseModel):
    """Metadata không ảnh hưởng logic huấn luyện, chỉ dùng để log/track."""

    name: str
    seed: int = 42


class DataConfig(BaseModel):
    frames_per_video: int = Field(gt=0)
    frame_size: tuple[int, int]
    use_motion_channel: bool = False
    num_classes: int = Field(gt=0)

    @model_validator(mode="after")
    def check_frame_size_positive(self) -> "DataConfig":
        h, w = self.frame_size
        if h <= 0 or w <= 0:
            raise ValueError(f"frame_size phải > 0 ở cả 2 chiều, nhận ({h}, {w})")
        return self


class Phase1Config(BaseModel):
    """Config train spatial backbone (giai đoạn 1).

    embed_dim_out chính là "hợp đồng" (contract) với Phase2Config.input_dim
    — xem mục 3.1 roadmap. check_dim_contract ở AppConfig bên dưới phải
    chặn lớp lỗi này ngay lúc load config, trước khi tốn 1 giây GPU nào.
    """

    embed_dim_out: int = Field(gt=0)
    epochs: int = Field(gt=0)
    learning_rate: float = Field(gt=0)


class Phase2Config(BaseModel):
    input_dim: int = Field(gt=0)
    num_heads: int = Field(gt=0)
    epochs: int = Field(gt=0)
    learning_rate: float = Field(gt=0)

    @model_validator(mode="after")
    def check_divisible(self) -> "Phase2Config":
        if self.input_dim % self.num_heads != 0:
            raise ValueError(
                f"input_dim ({self.input_dim}) phải chia hết cho "
                f"num_heads ({self.num_heads})"
            )
        return self


class AppConfig(BaseModel):
    project: ProjectConfig
    data: DataConfig
    phase1: Phase1Config
    phase2: Phase2Config

    @model_validator(mode="after")
    def check_dim_contract(self) -> "AppConfig":
        if self.phase1.embed_dim_out != self.phase2.input_dim:
            raise ValueError(
                f"Contract vi phạm: embed_dim_out của phase1 "
                f"({self.phase1.embed_dim_out}) != input_dim của phase2 "
                f"({self.phase2.input_dim}). Sửa 1 trong 2 giá trị trong config."
            )
        return self
