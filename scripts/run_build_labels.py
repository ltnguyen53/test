"""CLI wrapper mỏng cho data/labels.py::write_labels_csv_from_ucf(). Không
chứa logic — logic nằm ở src/ (cùng quy ước mọi script khác trong thư mục
này, phiên 4.2/12.2).

THIẾU từ phiên 13.1 (phát hiện khi trả lời câu hỏi thực tế "unzip UCF vào
data/raw/ rồi làm sao ra labels.csv?"): write_labels_csv_from_ucf() có
sẵn + có test (tests/unit/test_labels.py) nhưng KHÔNG có script nào gọi
nó — chỉ dùng được nếu tự viết Python. Bổ sung script này để đúng quy ước
"mọi hàm src/ có thể gọi qua 1 script scripts/" như phần còn lại của dự
án (xem run_build_reference.py, run_export.py...).

CHỈ dùng cho UCF101 (hoặc dataset khác đi kèm classInd.txt +
trainlistXX.txt/testlistXX.txt ĐÚNG format UCF101 chuẩn tải từ
crcv.ucf.edu) — UCF11 (YouTube Action) KHÔNG đi kèm 2 file này (cấu trúc
gốc là <ClassName>/<group>/*.mpg, không có annotation file rời), nên với
UCF11 vẫn phải tự điền data/raw/labels.csv bằng tay (copy từ
labels.csv.template, xem docs/runbook.md mục 1) — script này không giúp
được cho UCF11.
"""

from __future__ import annotations

import argparse

from video_action_mlops.data.labels import write_labels_csv_from_ucf


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sinh data/raw/labels.csv từ classInd.txt + trainlistXX.txt/testlistXX.txt "
            "(format UCF101 chuẩn) — KHÔNG dùng được cho UCF11, xem docstring module."
        )
    )
    parser.add_argument("--class-ind", required=True, help="Đường dẫn classInd.txt")
    parser.add_argument(
        "--split-list", required=True, help="Đường dẫn trainlistXX.txt hoặc testlistXX.txt"
    )
    parser.add_argument(
        "--video-ext", default=".avi", help="Đuôi file video thật trong data/raw/ (mặc định .avi)"
    )
    parser.add_argument("--output", default="data/raw/labels.csv")
    args = parser.parse_args()

    out_path = write_labels_csv_from_ucf(
        args.class_ind, args.split_list, args.video_ext, args.output
    )
    print(f"[build_labels] -> {out_path}")


if __name__ == "__main__":
    main()
