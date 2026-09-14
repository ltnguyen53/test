"""monitoring/logging_sink.py — ghi lại mỗi prediction request thành công
vào Neon Postgres (mục 4.8). Đây là nguồn dữ liệu DUY NHẤT để mục 4.10
(drift monitoring, tuần 11) có gì để phân tích — không log, không có gì
giám sát.

CỐ Ý KHÔNG dùng ORM (SQLAlchemy) — chỉ 1 bảng, 1 câu INSERT duy nhất,
thêm ORM là over-engineering cho quy mô này. Dùng psycopg2 (driver thuần,
đã khai trong pyproject.toml nhóm "monitoring" từ phiên 1.1) trực tiếp.

Kết nối cấu hình qua biến môi trường DATABASE_URL (đã có sẵn trong
.env.example từ phiên 1.3, tên biến giữ nguyên, giờ mới thật sự dùng tới).

LỖI GHI LOG KHÔNG ĐƯỢC LÀM HỎNG REQUEST CHÍNH: nếu Neon down hoặc mạng
lỗi, /predict vẫn phải trả kết quả cho user — logging là "best-effort",
không phải bước bắt buộc của luồng suy luận. Nuốt lỗi + log warning, KHÔNG
raise lên caller — 1 trong SỐ ÍT chỗ trong project cố tình vi phạm
fail-fast, cùng lý do với kernels/loader.py (phiên 6.1): 1 tính năng phụ
trợ không được phép làm sập tính năng chính.

PHẠM VI HIỆN TẠI: chỉ log request THÀNH CÔNG (có predicted_class). Request
lỗi (video hỏng, validation fail — xem serving/api.py) CHƯA được log —
muốn giám sát cả tỷ lệ lỗi cần mở rộng bảng thêm cột status, nằm ngoài
phạm vi phiên này.
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass

import psycopg2

logger = logging.getLogger(__name__)

_INSERT_SQL = """
    INSERT INTO predictions (
        request_id, input_frame_count, input_mean_motion_score,
        predicted_class, confidence, latency_ms, model_version
    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
"""


@dataclass
class PredictionLogEntry:
    request_id: uuid.UUID
    input_frame_count: int
    input_mean_motion_score: float
    predicted_class: int
    confidence: float
    latency_ms: float
    model_version: str


def log_prediction(entry: PredictionLogEntry) -> None:
    """Best-effort — xem docstring module. Gọi từ FastAPI BackgroundTasks
    (serving/api.py), SAU KHI response đã trả cho user, không chặn latency
    request chính."""
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        logger.warning("monitoring.logging_sink: thiếu DATABASE_URL, bỏ qua ghi log (best-effort).")
        return

    try:
        conn = psycopg2.connect(database_url, connect_timeout=3)
    except Exception as exc:  # noqa: BLE001 — cố ý bắt rộng, xem docstring module
        logger.warning(f"monitoring.logging_sink: không kết nối được Neon ({exc}).")
        return

    try:
        # `with conn:` ở psycopg2 chỉ tự commit/rollback theo transaction,
        # KHÔNG tự đóng connection (khác nhiều driver khác) — phải tự
        # conn.close() ở finally, thiếu bước này sẽ rò rỉ connection dần
        # qua nhiều request.
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    _INSERT_SQL,
                    (
                        str(entry.request_id),
                        entry.input_frame_count,
                        entry.input_mean_motion_score,
                        entry.predicted_class,
                        entry.confidence,
                        entry.latency_ms,
                        entry.model_version,
                    ),
                )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"monitoring.logging_sink: ghi log thất bại ({exc}) — request chính không bị ảnh hưởng.")
    finally:
        conn.close()
