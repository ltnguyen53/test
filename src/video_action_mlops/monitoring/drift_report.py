"""monitoring/drift_report.py — sinh HTML report so sánh phân phối
input_mean_motion_score, confidence, predicted_class giữa "cửa sổ tham
chiếu" (val set, monitoring/build_reference.py phiên 11.1) và "cửa sổ gần
đây" (query từ bảng predictions trên Neon, phiên 9.2) — mục 4.10.

QUYẾT ĐỊNH KỸ THUẬT (đã tra cứu ngay trước khi viết, KHÔNG phải nhớ từ
training data): Evidently trải qua redesign API LỚN ở bản 0.6 (01/2025) và
0.7 (04/2025, thành mặc định). API CŨ vẫn xuất hiện nhiều trên mạng
(`from evidently.report import Report`,
`from evidently.metric_preset import DataDriftPreset`) nhưng KHÔNG còn là
API mặc định của bản mới nhất. pyproject.toml (phiên 1.1) ghi
"evidently>=0.4.30" — cận dưới cũ, pip sẽ cài bản MỚI NHẤT theo mặc định
(hiện ~0.7.21) — tức là bản dùng API MỚI. Code dưới đây dùng API MỚI
(`from evidently import Report`, `from evidently.presets import
DataDriftPreset`) cho khớp bản sẽ thật sự được cài.

ĐÃ VÁ 1 BUG THẬT (phiên 14.7, verify bằng evidently 0.7.21 cài thật): bản
đầu tiên truyền THẲNG `pandas.DataFrame` cho `report.run(current_data=...,
reference_data=...)` — chạy thật thì Evidently tự SUY ĐOÁN kiểu cột, và
đoán NHẦM `predicted_class` (chuỗi "0"/"1"/"2") thành cột TEXT thay vì
CATEGORICAL, dẫn tới lỗi `ValueError: empty vocabulary` sâu bên trong 1
stattest so khớp văn bản (TF-IDF) không phù hợp cho vài ký tự số. Sửa
bằng cách khai báo tường minh qua `Dataset.from_pandas(df,
data_definition=DataDefinition(...))` (API `Dataset`/`DataDefinition` của
bản 0.7) thay vì đưa thẳng DataFrame — không còn chỗ nào để Evidently
đoán sai kiểu cột nữa.

Khác monitoring/logging_sink.py (phiên 9.2, best-effort, nuốt lỗi): ở đây
lỗi kết nối/build report PHẢI raise — 1 report drift SAI (hoặc thiếu dữ
liệu) mà vẫn "chạy xong" còn nguy hiểm hơn không chạy gì (im lặng báo
"không có drift" trong khi thực ra report rỗng).
"""

from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import psycopg2
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

_NUMERIC_COLUMNS = ["input_mean_motion_score", "confidence"]
_CATEGORICAL_COLUMNS = ["predicted_class"]
_DATA_DEFINITION = DataDefinition(
    numerical_columns=_NUMERIC_COLUMNS,
    categorical_columns=_CATEGORICAL_COLUMNS,
)


def _load_reference(reference_csv: str | Path) -> list[dict]:
    with open(reference_csv, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _query_recent_predictions(database_url: str, limit: int = 500) -> list[dict]:
    """Query N row GẦN NHẤT từ bảng predictions (Neon, phiên 9.2) làm "cửa
    sổ gần đây". KHÔNG best-effort — lỗi kết nối PHẢI raise (xem docstring
    module)."""
    conn = psycopg2.connect(database_url, connect_timeout=5)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT input_mean_motion_score, confidence, predicted_class
                FROM predictions
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    return [
        {"input_mean_motion_score": r[0], "confidence": r[1], "predicted_class": r[2]} for r in rows
    ]


def _to_dataframe(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in _NUMERIC_COLUMNS:
        df[col] = df[col].astype(float)
    for col in _CATEGORICAL_COLUMNS:
        # predicted_class là CHỈ SỐ LỚP (categorical), không phải thang đo
        # liên tục — ép string tường minh để Evidently không hiểu nhầm
        # 0,1,2,... thành numeric ordinal.
        df[col] = df[col].astype(str)
    return df


def build_drift_report(
    reference_csv: str | Path,
    database_url: str,
    output_dir: str | Path = "reports",
    recent_limit: int = 500,
) -> Path:
    reference_rows = _load_reference(reference_csv)
    if not reference_rows:
        raise ValueError(
            f"{reference_csv} rỗng — chạy monitoring/build_reference.py (phiên 11.1) trước."
        )

    recent_rows = _query_recent_predictions(database_url, limit=recent_limit)
    if not recent_rows:
        raise ValueError(
            "Bảng predictions trên Neon chưa có row nào — cần ít nhất vài request "
            "/predict thật (phiên 8.1/10.1) trước khi có gì để so sánh."
        )

    reference_df = _to_dataframe(reference_rows)
    current_df = _to_dataframe(recent_rows)

    # Dataset.from_pandas(..., data_definition=...) — KHÔNG đưa thẳng
    # DataFrame cho report.run() (xem docstring module: đưa thẳng khiến
    # Evidently tự đoán predicted_class là cột TEXT thay vì CATEGORICAL,
    # lỗi thật đã verify + vá ở phiên 14.7).
    reference_dataset = Dataset.from_pandas(reference_df, data_definition=_DATA_DEFINITION)
    current_dataset = Dataset.from_pandas(current_df, data_definition=_DATA_DEFINITION)

    report = Report([DataDriftPreset()])
    result = report.run(current_data=current_dataset, reference_data=reference_dataset)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = output_dir / f"drift_{timestamp}.html"
    result.save_html(str(out_path))
    return out_path
