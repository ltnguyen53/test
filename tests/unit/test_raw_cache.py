"""tests/unit/test_raw_cache.py — test data/raw_cache.py (logic chuyển ra
từ colab/bootstrap.ipynb, phát hiện khi bị hỏi thẳng "vẫn còn function
trong notebook là sao?" — notebook giờ chỉ gọi các hàm này, không tự định
nghĩa logic nữa).

`.rar` được test THẬT bằng file .rar THẬT (tự tạo bằng lệnh hệ thống
`rar`, verify trong sandbox có unrar/rarfile cài thật) — không mock
rarfile, vì chính bug thật (thiếu unrar, sai định dạng cấu trúc) chỉ lộ
ra khi chạy thật qua thư viện thật.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tarfile
import zipfile

import pytest

from video_action_mlops.data.raw_cache import (
    extract_archive,
    has_video_files,
    restore_raw_from_cache,
    sync_raw_to_cache,
)

_HAS_RAR_CLI = shutil.which("rar") is not None
_HAS_UNRAR_CLI = shutil.which("unrar") is not None


def _make_source_tree(base):
    (base / "ApplyEyeMakeup").mkdir(parents=True)
    (base / "ApplyLipstick").mkdir(parents=True)
    (base / "ApplyEyeMakeup" / "v1.avi").write_text("fake video 1")
    (base / "ApplyLipstick" / "v2.avi").write_text("fake video 2")


# ---------- extract_archive: .zip ----------


def test_extract_archive_zip(tmp_path):
    src = tmp_path / "src"
    _make_source_tree(src)
    archive = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for p in src.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(src))

    dest = tmp_path / "extracted"
    result = extract_archive(archive, dest)

    assert result == dest
    assert (dest / "ApplyEyeMakeup" / "v1.avi").read_text() == "fake video 1"
    assert (dest / "ApplyLipstick" / "v2.avi").read_text() == "fake video 2"


# ---------- extract_archive: .tar.gz ----------


def test_extract_archive_tar_gz(tmp_path):
    src = tmp_path / "src"
    _make_source_tree(src)
    archive = tmp_path / "dataset.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        for p in src.rglob("*"):
            if p.is_file():
                tf.add(p, arcname=p.relative_to(src))

    dest = tmp_path / "extracted"
    extract_archive(archive, dest)

    assert (dest / "ApplyEyeMakeup" / "v1.avi").read_text() == "fake video 1"


# ---------- extract_archive: .rar — THẬT, không mock ----------


@pytest.mark.skipif(
    not (_HAS_RAR_CLI and _HAS_UNRAR_CLI),
    reason="cần cả 'rar' (tạo, để test) và 'unrar' (đọc) cài trong hệ thống",
)
def test_extract_archive_rar_real_file(tmp_path):
    """UCF101/UCF11 bản chính thức là .rar (không phải .zip) -- test bằng
    file .rar THẬT (tự tạo qua lệnh hệ thống `rar`), giải nén lại bằng
    chính extract_archive(), xác nhận đúng cấu trúc thư mục lồng theo
    class -- không mock rarfile/unrar, vì đây chính là ranh giới có rủi
    ro thật (thiếu unrar là lỗi thường gặp nhất khi ingest UCF)."""
    src = tmp_path / "src"
    _make_source_tree(src)
    archive = tmp_path / "dataset.rar"
    subprocess.run(
        ["rar", "a", "-r", "-inul", str(archive), "ApplyEyeMakeup", "ApplyLipstick"],
        cwd=src,
        check=True,
    )

    dest = tmp_path / "extracted"
    extract_archive(archive, dest)

    assert (dest / "ApplyEyeMakeup" / "v1.avi").read_text() == "fake video 1"
    assert (dest / "ApplyLipstick" / "v2.avi").read_text() == "fake video 2"


def test_extract_archive_rejects_unsupported_extension(tmp_path):
    archive = tmp_path / "dataset.7z"
    archive.write_bytes(b"khong quan trong noi dung, chi test phan mo rong")
    with pytest.raises(ValueError, match="7z"):
        extract_archive(archive, tmp_path / "dest")


def test_extract_archive_rar_missing_rarfile_package_raises_clear_error(tmp_path, monkeypatch):
    """import rarfile (deferred, bên trong extract_archive) that bai neu
    package chua cai -- mo phong bang cach dat sys.modules['rarfile'] =
    None (hanh vi chuan cua Python: import 1 module dat None trong
    sys.modules se raise ImportError), khong can go cai that."""
    monkeypatch.setitem(sys.modules, "rarfile", None)
    archive = tmp_path / "dataset.rar"
    archive.write_bytes(b"khong quan trong, se raise truoc khi doc noi dung")

    with pytest.raises(RuntimeError, match="rarfile"):
        extract_archive(archive, tmp_path / "dest")


def test_extract_archive_rar_missing_unrar_binary_raises_clear_error(tmp_path, monkeypatch):
    """rarfile import duoc nhung khong tim thay cong cu he thong 'unrar'
    -- mo phong bang fake RarFile raise dung loai loi rarfile that su
    raise trong tinh huong nay (RarCannotExec)."""
    import rarfile

    class _FakeRarFile:
        def __init__(self, *a, **k):
            raise rarfile.RarCannotExec("gia lap: khong tim thay unrar")

    monkeypatch.setattr(rarfile, "RarFile", _FakeRarFile)
    archive = tmp_path / "dataset.rar"
    archive.write_bytes(b"khong quan trong")

    with pytest.raises(RuntimeError, match="unrar"):
        extract_archive(archive, tmp_path / "dest")


# ---------- has_video_files ----------


def test_has_video_files_true_when_video_present(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "clip.mp4").write_bytes(b"x")
    assert has_video_files(tmp_path) is True


def test_has_video_files_false_when_empty(tmp_path):
    assert has_video_files(tmp_path) is False


def test_has_video_files_false_when_dir_missing(tmp_path):
    assert has_video_files(tmp_path / "khong_ton_tai") is False


def test_has_video_files_ignores_non_video_extensions(tmp_path):
    (tmp_path / "labels.csv").write_text("video_filename,label\n")
    (tmp_path / "notes.txt").write_text("khong phai video")
    assert has_video_files(tmp_path) is False


# ---------- restore_raw_from_cache / sync_raw_to_cache ----------


def test_restore_raw_from_cache_copies_when_cache_has_videos(tmp_path):
    cache = tmp_path / "cache"
    _make_source_tree(cache)
    local_raw = tmp_path / "data_raw"

    result = restore_raw_from_cache(cache, local_raw)

    assert result is True
    assert (local_raw / "ApplyEyeMakeup" / "v1.avi").read_text() == "fake video 1"


def test_restore_raw_from_cache_returns_false_when_cache_empty(tmp_path):
    cache = tmp_path / "empty_cache"
    cache.mkdir()
    local_raw = tmp_path / "data_raw"

    result = restore_raw_from_cache(cache, local_raw)

    assert result is False


def test_sync_raw_to_cache_copies_when_local_has_videos(tmp_path):
    local_raw = tmp_path / "data_raw"
    _make_source_tree(local_raw)
    cache = tmp_path / "cache"

    result = sync_raw_to_cache(local_raw, cache)

    assert result is True
    assert (cache / "ApplyEyeMakeup" / "v1.avi").read_text() == "fake video 1"


def test_sync_raw_to_cache_returns_false_when_local_empty(tmp_path):
    local_raw = tmp_path / "data_raw"
    local_raw.mkdir()
    cache = tmp_path / "cache"

    result = sync_raw_to_cache(local_raw, cache)

    assert result is False
