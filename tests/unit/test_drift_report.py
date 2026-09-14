"""tests/unit/test_drift_report.py — test monitoring/drift_report.py
(phiên 14.7).

CHỈ `psycopg2.connect` bị monkeypatch (không Neon thật nào bị chạm tới) —
`evidently.Report`/`Dataset`/`DataDefinition` chạy THẬT, không mock. Đây
là module DUY NHẤT trong dự án mà bản thân thư viện ngoài (Evidently) mới
đổi API lớn nên "import được" không có nghĩa "chạy được" — test
end-to-end ở đây chính là bằng chứng thay cho lời cảnh báo "CHƯA TỪNG
chạy" từng ghi trong docstring module (nay đã vá 1 bug thật tìm được khi
chạy lần đầu: xem drift_report.py để biết chi tiết predicted_class bị
đoán nhầm thành cột text).
"""

from __future__ import annotations

import csv

import pandas as pd
import pytest

from video_action_mlops.monitoring import drift_report as dr_module
from video_action_mlops.monitoring.drift_report import (
    _load_reference,
    _to_dataframe,
    build_drift_report,
)


def _write_reference_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["input_mean_motion_score", "confidence", "predicted_class"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class _FakeCursor:
    def __init__(self, rows, raise_on_execute: Exception | None = None):
        self.rows = rows
        self.raise_on_execute = raise_on_execute
        self.executed_params = None

    def execute(self, sql, params):
        if self.raise_on_execute is not None:
            raise self.raise_on_execute
        self.executed_params = params

    def fetchall(self):
        return self.rows

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class _FakeConnection:
    def __init__(self, rows=(), raise_on_execute: Exception | None = None):
        self.cursor_obj = _FakeCursor(rows, raise_on_execute=raise_on_execute)
        self.closed = False

    def cursor(self):
        return self.cursor_obj

    def close(self):
        self.closed = True


def _sample_reference_rows(n=8):
    return [
        {
            "input_mean_motion_score": str(0.1 * i),
            "confidence": str(0.5 + 0.05 * i),
            "predicted_class": str(i % 3),
        }
        for i in range(n)
    ]


def _sample_recent_rows(n=8):
    # psycopg2 fetchall() trả tuple, khác dict cua CSV -- xem
    # _query_recent_predictions() build dict tu index 0/1/2.
    return [(0.2 * i, 0.4 + 0.05 * i, i % 3) for i in range(n)]


# ---------- _load_reference ----------


def test_load_reference_reads_csv_rows(tmp_path):
    csv_path = tmp_path / "ref.csv"
    _write_reference_csv(csv_path, _sample_reference_rows(3))

    rows = _load_reference(csv_path)

    assert len(rows) == 3
    assert set(rows[0].keys()) == {"input_mean_motion_score", "confidence", "predicted_class"}


# ---------- _to_dataframe ----------


def test_to_dataframe_casts_numeric_and_categorical_columns():
    rows = [{"input_mean_motion_score": "0.5", "confidence": "0.9", "predicted_class": "2"}]
    df = _to_dataframe(rows)

    assert df["input_mean_motion_score"].dtype == float
    assert df["confidence"].dtype == float
    # predicted_class ep string (khong phai numeric) -- kiem tra kieu du
    # lieu THUC TE cua gia tri thay vi so sanh 1 pandas dtype cu the (bien
    # dong theo phien ban pandas: 3.x mac dinh StringDtype thay vi object
    # cho .astype(str), phat hien khi chay that trong sandbox nay).
    assert not pd.api.types.is_numeric_dtype(df["predicted_class"])
    assert str(df["predicted_class"].iloc[0]) == "2"


# ---------- build_drift_report: các nhánh raise ----------


def test_build_drift_report_raises_when_reference_csv_empty(tmp_path, monkeypatch):
    csv_path = tmp_path / "ref.csv"
    _write_reference_csv(csv_path, [])  # chỉ header, không row nào

    def _should_not_connect(*a, **k):
        raise AssertionError("khong duoc goi psycopg2.connect khi reference rong")

    monkeypatch.setattr(dr_module.psycopg2, "connect", _should_not_connect)

    with pytest.raises(ValueError, match="rỗng"):
        build_drift_report(reference_csv=str(csv_path), database_url="postgresql://fake")


def test_build_drift_report_raises_when_no_recent_predictions(tmp_path, monkeypatch):
    csv_path = tmp_path / "ref.csv"
    _write_reference_csv(csv_path, _sample_reference_rows(5))

    monkeypatch.setattr(dr_module.psycopg2, "connect", lambda *a, **k: _FakeConnection(rows=[]))

    with pytest.raises(ValueError, match="predictions"):
        build_drift_report(reference_csv=str(csv_path), database_url="postgresql://fake")


def test_build_drift_report_connect_failure_propagates_not_swallowed(tmp_path, monkeypatch):
    """KHÁC logging_sink.py (best-effort, nuốt lỗi) -- ở đây lỗi PHẢI
    raise lên caller, không được âm thầm bỏ qua (xem docstring module)."""
    csv_path = tmp_path / "ref.csv"
    _write_reference_csv(csv_path, _sample_reference_rows(5))

    def _raise_connect(*a, **k):
        raise OSError("khong ket noi duoc Neon (gia lap)")

    monkeypatch.setattr(dr_module.psycopg2, "connect", _raise_connect)

    with pytest.raises(OSError, match="khong ket noi"):
        build_drift_report(reference_csv=str(csv_path), database_url="postgresql://fake")


def test_query_recent_predictions_closes_connection_even_when_execute_fails(tmp_path, monkeypatch):
    csv_path = tmp_path / "ref.csv"
    _write_reference_csv(csv_path, _sample_reference_rows(5))

    fake_conn = _FakeConnection(
        raise_on_execute=RuntimeError("bang predictions khong ton tai (gia lap)")
    )
    monkeypatch.setattr(dr_module.psycopg2, "connect", lambda *a, **k: fake_conn)

    with pytest.raises(RuntimeError, match="khong ton tai"):
        build_drift_report(reference_csv=str(csv_path), database_url="postgresql://fake")

    # finally: conn.close() trong _query_recent_predictions() van phai chay
    # du execute() loi -- khong duoc ro ri connection ngay ca khi raise.
    assert fake_conn.closed is True


# ---------- end-to-end: Evidently THẬT, không mock ----------


def test_build_drift_report_end_to_end_produces_real_html(tmp_path, monkeypatch):
    csv_path = tmp_path / "ref.csv"
    _write_reference_csv(csv_path, _sample_reference_rows(10))

    fake_conn = _FakeConnection(rows=_sample_recent_rows(10))
    monkeypatch.setattr(dr_module.psycopg2, "connect", lambda *a, **k: fake_conn)

    output_dir = tmp_path / "reports"
    out_path = build_drift_report(
        reference_csv=str(csv_path), database_url="postgresql://fake", output_dir=str(output_dir)
    )

    assert out_path.exists()
    assert out_path.suffix == ".html"
    assert out_path.parent == output_dir
    content = out_path.read_text(encoding="utf-8")
    assert len(content) > 1000  # report Evidently thật, không phải file rỗng/placeholder
    assert "<html" in content.lower()
    assert fake_conn.closed is True


def test_build_drift_report_filename_contains_utc_timestamp(tmp_path, monkeypatch):
    csv_path = tmp_path / "ref.csv"
    _write_reference_csv(csv_path, _sample_reference_rows(6))
    monkeypatch.setattr(
        dr_module.psycopg2, "connect", lambda *a, **k: _FakeConnection(rows=_sample_recent_rows(6))
    )

    out_path = build_drift_report(
        reference_csv=str(csv_path), database_url="postgresql://fake", output_dir=str(tmp_path)
    )

    assert out_path.name.startswith("drift_")
    assert out_path.name.endswith("Z.html")  # xem timestamp format trong build_drift_report()


def test_build_drift_report_respects_recent_limit_param(tmp_path, monkeypatch):
    csv_path = tmp_path / "ref.csv"
    _write_reference_csv(csv_path, _sample_reference_rows(6))
    fake_conn = _FakeConnection(rows=_sample_recent_rows(6))
    monkeypatch.setattr(dr_module.psycopg2, "connect", lambda *a, **k: fake_conn)

    build_drift_report(
        reference_csv=str(csv_path),
        database_url="postgresql://fake",
        output_dir=str(tmp_path),
        recent_limit=42,
    )

    assert fake_conn.cursor_obj.executed_params == (42,)
