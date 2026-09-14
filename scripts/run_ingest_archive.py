"""CLI wrapper mỏng cho data/raw_cache.py::extract_archive(). Không chứa
logic — logic nằm ở src/ (cùng quy ước mọi script khác trong thư mục
này).

Bổ sung cùng đợt với data/raw_cache.py (phát hiện qua câu hỏi thực tế
"unzip UCF101 ở đâu?") — trước đó KHÔNG có cách nào giải nén dataset tải
về ngoài tự viết Python tay.
"""

from __future__ import annotations

import argparse

from video_action_mlops.data.raw_cache import extract_archive


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Giải nén dataset thô (.zip/.tar*/.rar) vào data/raw/. "
            "UCF101/UCF11 bản chính thức (crcv.ucf.edu) là .rar — cần "
            "'pip install -e \".[ingest]\"' + 'apt install unrar' trước."
        )
    )
    parser.add_argument("archive", help="Đường dẫn file .zip/.tar*/.rar đã tải về")
    parser.add_argument("--dest", default="data/raw", help="Thư mục đích (mặc định data/raw)")
    args = parser.parse_args()

    out_dir = extract_archive(args.archive, args.dest)
    print(f"[ingest_archive] {args.archive} -> {out_dir}")


if __name__ == "__main__":
    main()
