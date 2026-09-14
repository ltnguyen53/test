"""tests/unit/test_logging_sink.py — test monitoring/logging_sink.py (phiên
14.7).

psycopg2.connect() được MONKEYPATCH bằng connection/cursor GIẢ tự viết
(_FakeConnection/_FakeCursor) — không có Postgres/Neon thật nào bị chạm
tới. Đây là ranh giới hợp lý để fake (giống build_model ở
test_phase1_trainer.py, phiên 14.6): log_prediction() không phải thứ ta
muốn kiểm tra hành vi của driver psycopg2, mà là hành vi CỦA CHÍNH TA
xung quanh nó — có gọi đúng SQL/tham số không, có nuốt lỗi đúng cách
không, có luôn đóng connection không (đọc docstring module: "LỖI GHI LOG
KHÔNG ĐƯỢC LÀM HỎNG REQUEST CHÍNH" là bất biến quan trọng nhất cần test).
"""

from __future__ import annotations

import uuid

from video_action_mlops.monitoring import logging_sink
from video_action_mlops.monitoring.logging_sink import PredictionLogEntry, log_prediction


def _make_entry(**overrides) -> PredictionLogEntry:
    defaults = dict(
        request_id=uuid.UUID("12345678-1234-5678-1234-567812345678"),
        input_frame_count=16,
        input_mean_motion_score=0.42,
        predicted_class=3,
        confidence=0.91,
        latency_ms=123.4,
        model_version="v1.2.3",
    )
    defaults.update(overrides)
    return PredictionLogEntry(**defaults)


class _FakeCursor:
    def __init__(self, raise_on_execute: Exception | None = None):
        self.raise_on_execute = raise_on_execute
        self.executed: list[tuple] = []

    def execute(self, sql, params):
        if self.raise_on_execute is not None:
            raise self.raise_on_execute
        self.executed.append((sql, params))

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class _FakeConnection:
    """Mô phỏng đúng 2 điểm psycopg2 thật có mà driver khác thường không:
    `with conn:` chỉ commit/rollback transaction (KHÔNG tự đóng connection),
    và cursor cũng là context manager riêng — xem docstring log_prediction()
    giải thích vì sao code thật phải tự conn.close() ở finally."""

    def __init__(self, raise_on_execute: Exception | None = None):
        self.cursor_obj = _FakeCursor(raise_on_execute=raise_on_execute)
        self.closed = False
        self.committed = False
        self.rolled_back = False

    def cursor(self):
        return self.cursor_obj

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.committed = True
        else:
            self.rolled_back = True
        return False  # không nuốt exception ở tầng `with conn:` -- giống psycopg2 thật


# ---------- thiếu DATABASE_URL ----------


def test_log_prediction_missing_database_url_skips_silently(monkeypatch, caplog):
    monkeypatch.delenv("DATABASE_URL", raising=False)

    def _should_not_be_called(*args, **kwargs):
        raise AssertionError("psycopg2.connect KHONG duoc goi khi thieu DATABASE_URL")

    monkeypatch.setattr(logging_sink.psycopg2, "connect", _should_not_be_called)

    with caplog.at_level("WARNING"):
        log_prediction(_make_entry())  # không được raise

    assert "DATABASE_URL" in caplog.text


# ---------- đường thành công ----------


def test_log_prediction_success_calls_execute_with_correct_params(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://fake/db")
    fake_conn = _FakeConnection()
    connect_calls = []

    def _fake_connect(database_url, connect_timeout):
        connect_calls.append((database_url, connect_timeout))
        return fake_conn

    monkeypatch.setattr(logging_sink.psycopg2, "connect", _fake_connect)

    entry = _make_entry()
    log_prediction(entry)

    assert connect_calls == [("postgresql://fake/db", 3)]
    assert len(fake_conn.cursor_obj.executed) == 1
    sql, params = fake_conn.cursor_obj.executed[0]
    assert "INSERT INTO predictions" in sql
    # request_id phải là str() (không phải uuid.UUID thô) -- xem
    # log_prediction(): str(entry.request_id) truyền cho psycopg2.
    assert params == (
        str(entry.request_id),
        entry.input_frame_count,
        entry.input_mean_motion_score,
        entry.predicted_class,
        entry.confidence,
        entry.latency_ms,
        entry.model_version,
    )
    assert isinstance(params[0], str)


def test_log_prediction_success_closes_connection(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://fake/db")
    fake_conn = _FakeConnection()
    monkeypatch.setattr(logging_sink.psycopg2, "connect", lambda *a, **k: fake_conn)

    log_prediction(_make_entry())

    assert fake_conn.closed is True
    assert fake_conn.committed is True


# ---------- best-effort: lỗi không được làm sập request chính ----------


def test_log_prediction_connect_failure_is_caught_not_raised(monkeypatch, caplog):
    monkeypatch.setenv("DATABASE_URL", "postgresql://fake/db")

    def _raise_connect(*a, **k):
        raise OSError("khong ket noi duoc Neon (gia lap)")

    monkeypatch.setattr(logging_sink.psycopg2, "connect", _raise_connect)

    with caplog.at_level("WARNING"):
        log_prediction(_make_entry())  # KHONG duoc raise -- day la diem quan trong nhat

    assert "không kết nối được" in caplog.text.lower() or "khong ket noi" in caplog.text.lower()


def test_log_prediction_execute_failure_is_caught_and_connection_still_closed(monkeypatch, caplog):
    monkeypatch.setenv("DATABASE_URL", "postgresql://fake/db")
    fake_conn = _FakeConnection(
        raise_on_execute=RuntimeError("bang predictions khong ton tai (gia lap)")
    )
    monkeypatch.setattr(logging_sink.psycopg2, "connect", lambda *a, **k: fake_conn)

    with caplog.at_level("WARNING"):
        log_prediction(_make_entry())  # KHONG duoc raise

    # Bat buoc nhat cua bai test nay: du execute() loi, finally: conn.close()
    # o logging_sink.py van phai chay -- thieu dong nay se ro ri connection
    # dan qua nhieu request that (xem comment trong log_prediction()).
    assert fake_conn.closed is True
    assert "ghi log thất bại" in caplog.text.lower() or "that bai" in caplog.text.lower()


def test_log_prediction_never_raises_regardless_of_entry_content(monkeypatch):
    """Dù entry có giá trị "lạ" (confidence âm, frame_count=0...),
    log_prediction() vẫn không được raise -- best-effort nghĩa là KHÔNG có
    input nào của luồng chính được phép làm sập qua đường logging phụ."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://fake/db")
    fake_conn = _FakeConnection()
    monkeypatch.setattr(logging_sink.psycopg2, "connect", lambda *a, **k: fake_conn)

    weird_entry = _make_entry(
        confidence=-1.0, input_frame_count=0, input_mean_motion_score=float("nan")
    )
    log_prediction(weird_entry)  # khong duoc raise

    assert len(fake_conn.cursor_obj.executed) == 1
