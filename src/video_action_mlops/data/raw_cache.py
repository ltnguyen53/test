"""data/raw_cache.py — logic THẬT đứng sau 2 việc trước đây bị viết TRỰC
TIẾP trong colab/bootstrap.ipynb (vi phạm quy ước "script/notebook chỉ
orchestrate, logic nằm ở src/" đã áp dụng xuyên suốt project — phát hiện
khi người dùng hỏi thẳng "bootstrap.ipynb vẫn còn function là sao?"):

1. `extract_archive()` — giải nén file dataset thô (`.zip`/`.tar*`/`.rar`)
   vào `data/raw/`. UCF101/UCF11 bản CHÍNH THỨC tải từ crcv.ucf.edu là
   `.rar` (KHÔNG phải `.zip` — nhầm lẫn phổ biến) — cần thêm package
   `rarfile` (pip) + công cụ hệ thống `unrar` (`apt install unrar` trên
   Ubuntu/Debian/Colab, đã cài sẵn trên Colab theo mặc định). Đã verify
   THẬT bằng cách tự tạo 1 file .rar thật (lệnh `rar a`), giải nén lại
   bằng chính hàm này, xác nhận đúng cấu trúc thư mục lồng theo class.

2. `has_video_files()`/`restore_raw_from_cache()`/`sync_raw_to_cache()` —
   đồng bộ `data/raw/` với cache trên Google Drive (phiên 13.3), để không
   phải tải lại dataset mỗi phiên Colab mới.
"""

from __future__ import annotations

import shutil
import tarfile
import zipfile
from pathlib import Path

_VIDEO_EXTENSIONS = ("*.avi", "*.mp4", "*.mov")


def extract_archive(archive_path: str | Path, dest_dir: str | Path) -> Path:
    """Giải nén `archive_path` (.zip, .tar/.tar.gz/.tar.bz2, hoặc .rar)
    vào `dest_dir`. Nhận diện định dạng qua ĐUÔI FILE (không đoán qua
    magic bytes — dataset tải về luôn giữ đúng đuôi gốc).

    Raise RuntimeError với hướng dẫn cài đặt rõ ràng nếu thiếu
    `rarfile`/`unrar` cho file `.rar` — đây là 2 phụ thuộc NGOÀI
    pyproject.toml (không thêm vào base dependencies vì chỉ cần lúc
    ingest dữ liệu 1 lần, không cần cho training/serving/monitoring).
    """
    archive_path = Path(archive_path)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    name_lower = archive_path.name.lower()

    if name_lower.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(dest_dir)
    elif name_lower.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2")):
        with tarfile.open(archive_path) as tf:
            # filter="data" (PEP 706, Python >=3.11.4/3.12) — chặn path
            # traversal (member ghi ra ngoài dest_dir) từ archive không
            # tin cậy tuyệt đối (dataset tải từ internet).
            tf.extractall(dest_dir, filter="data")
    elif name_lower.endswith(".rar"):
        try:
            import rarfile
        except ImportError as e:
            raise RuntimeError(
                "Thiếu package 'rarfile' — cần để giải nén UCF101/UCF11 bản CHÍNH "
                "THỨC (.rar từ crcv.ucf.edu). Cài: pip install rarfile."
            ) from e
        try:
            with rarfile.RarFile(archive_path) as rf:
                rf.extractall(dest_dir)
        except rarfile.RarCannotExec as e:
            raise RuntimeError(
                "rarfile không tìm thấy công cụ hệ thống 'unrar' — cài bằng "
                "'apt install unrar' (Ubuntu/Debian; Colab thường có sẵn) rồi thử lại."
            ) from e
    else:
        raise ValueError(
            f"Định dạng archive không hỗ trợ: '{archive_path.suffix}' "
            f"(chỉ hỗ trợ .zip/.tar/.tar.gz/.tar.bz2/.rar)"
        )

    return dest_dir


def has_video_files(directory: str | Path) -> bool:
    """True nếu `directory` (đệ quy) có ít nhất 1 file .avi/.mp4/.mov.

    BUG THẬT đã có ở đây (phát hiện qua chính unit test viết cho hàm
    này — bài học trực tiếp cho lý do KHÔNG được để logic sống trong
    notebook không test): bản đầu viết
    `any(directory.rglob(ext) for ext in _VIDEO_EXTENSIONS)` — mỗi
    `directory.rglob(ext)` là 1 GENERATOR OBJECT, và một generator OBJECT
    luôn truthy (Python không gọi `__bool__`/`__len__` để xem nó có yield
    gì không) — nghĩa là `any([gen1, gen2, gen3])` LUÔN True dù cả 3
    generator không yield gì cả. Phải ép kiểm tra PHẦN TỬ yield ra, không
    kiểm tra chính generator object.
    """
    directory = Path(directory)
    if not directory.exists():
        return False
    return any(True for ext in _VIDEO_EXTENSIONS for _ in directory.rglob(ext))


def restore_raw_from_cache(cache_dir: str | Path, local_raw_dir: str | Path) -> bool:
    """Copy `cache_dir` (vd Google Drive) -> `local_raw_dir` NẾU cache có
    video thật. Trả về True nếu đã khôi phục, False nếu cache rỗng (lần
    chạy đầu tiên, chưa có gì để khôi phục — KHÔNG phải lỗi)."""
    cache_dir = Path(cache_dir)
    local_raw_dir = Path(local_raw_dir)
    if not has_video_files(cache_dir):
        return False
    local_raw_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(cache_dir, local_raw_dir, dirs_exist_ok=True)
    return True


def sync_raw_to_cache(local_raw_dir: str | Path, cache_dir: str | Path) -> bool:
    """Copy `local_raw_dir` -> `cache_dir` (vd Google Drive) NẾU
    local_raw_dir có video thật. Trả về True nếu đã đồng bộ, False nếu
    không có gì để cache."""
    local_raw_dir = Path(local_raw_dir)
    cache_dir = Path(cache_dir)
    if not has_video_files(local_raw_dir):
        return False
    cache_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(local_raw_dir, cache_dir, dirs_exist_ok=True)
    return True
