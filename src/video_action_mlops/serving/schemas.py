"""serving/schemas.py — Pydantic model cho request/response qua HTTP.

Tách riêng khỏi config/schema.py (phiên 1.2) — 2 loại "schema" khác mục
đích hoàn toàn dù cùng dùng pydantic: config/schema.py định nghĩa "config
hợp lệ", ở đây định nghĩa "request/response hợp lệ qua HTTP". Gộp chung sẽ
làm rối ranh giới.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str


class PredictionResponse(BaseModel):
    predicted_class: int = Field(ge=0)
    probabilities: list[float]
