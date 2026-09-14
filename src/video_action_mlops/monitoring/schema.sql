-- monitoring/schema.sql — chạy 1 LẦN trên Neon (SQL Editor trên Neon
-- console, hoặc `psql $DATABASE_URL -f monitoring/schema.sql`) để tạo
-- bảng trước khi logging_sink.py có chỗ ghi. Không phải migration tool
-- (Alembic...) — quy mô project này 1 bảng, chạy tay 1 lần là đủ.
--
-- KHÁC bản mẫu roadmap mục 4.8 ở 2 chỗ, có chủ đích:
-- 1. predicted_class là INT (không phải TEXT) — khớp đúng
--    PredictionResponse.predicted_class: int trong serving/schemas.py
--    (phiên 8.1). Project chưa có bảng ánh xạ class index -> tên hành
--    động — để INT thay vì TEXT giả tên, tránh bịa dữ liệu.
-- 2. Thêm index theo created_at — mục 4.10 (drift monitoring, tuần 11)
--    sẽ query theo khoảng thời gian (vd "7 ngày gần nhất"), không có
--    index này full table scan mỗi lần chạy report.

CREATE TABLE IF NOT EXISTS predictions (
    id BIGSERIAL PRIMARY KEY,
    request_id UUID NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now(),
    input_frame_count INT,
    input_mean_motion_score FLOAT,   -- thống kê input, dùng phát hiện drift (mục 4.10, tuần 11)
    predicted_class INT,
    confidence FLOAT,
    latency_ms FLOAT,
    model_version TEXT               -- placeholder qua env MODEL_VERSION — CHƯA nối với MLflow Model Registry version thật (nợ kỹ thuật, ghi rõ ở logging_sink.py)
);

CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions (created_at);
