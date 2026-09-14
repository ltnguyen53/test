"""data/labels.py — đọc nhãn CSV + chia train/val, KHÔNG phụ thuộc
torch/numpy (chỉ dùng thư viện chuẩn: csv, hashlib).

Tách riêng khỏi data/datasets.py (phiên 4.3, phụ thuộc torch) để
evaluation/evaluate.py (phiên 7.2) đọc nhãn được mà KHÔNG cần cài torch —
đúng tinh thần "đánh giá đúng cái sẽ deploy" (evaluate chỉ cần
onnxruntime), khớp với serving image thật (mục 4.5 roadmap) vốn cũng
không cài torch.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


def load_label_by_stem(labels_csv: str | Path) -> dict[str, int]:
    """Đọc CSV (video_filename,label) -> {stem: label}. Dùng CHUNG giữa
    Phase1FrameDataset, Phase2SequenceDataset (datasets.py) và
    evaluate_temporal_onnx (evaluation/evaluate.py)."""
    label_by_stem: dict[str, int] = {}
    with open(labels_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label_by_stem[Path(row["video_filename"]).stem] = int(row["label"])
    return label_by_stem


def is_val_sample(stem: str, val_ratio: float = 0.2) -> bool:
    """Chia train/val DETERMINISTIC theo hash tên video — không cần thêm
    cột 'split' vào labels.csv (giữ nguyên format đã có từ phiên 4.3,
    không phá dữ liệu bạn đã tự điền tay).

    Cùng 1 video LUÔN rơi vào cùng 1 phía (train hoặc val) qua MỌI lần
    chạy, mọi lần train lại — quan trọng để không rò rỉ (leak) dữ liệu val
    vào train giữa các lần train khác nhau, điều mà 1 phép random không
    seed sẽ vi phạm.
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio phải trong (0, 1), nhận {val_ratio}")
    digest = hashlib.sha256(stem.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF  # số thực xác định trong [0, 1]
    return bucket < val_ratio


def load_ucf_class_index(class_ind_path: str | Path) -> dict[str, int]:
    """Đọc classInd.txt (UCF101/UCF11, thư mục ucfTrainTestlist/, phiên
    13.1) -> {tên_class: nhãn 0-indexed}.

    File gốc UCF 1-INDEXED (dòng đầu "1 ApplyEyeMakeup") — trừ 1 để khớp
    quy ước 0-indexed dùng xuyên suốt project (CrossEntropyLoss,
    DataConfig.num_classes, phiên 1.2/3.3).
    """
    class_by_name: dict[str, int] = {}
    with open(class_ind_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx_str, name = line.split(maxsplit=1)
            class_by_name[name] = int(idx_str) - 1
    return class_by_name


def load_ucf_split(class_ind_path: str | Path, split_list_path: str | Path) -> dict[str, int]:
    """Đọc trainlistXX.txt HOẶC testlistXX.txt (CÙNG 1 hàm, 2 định dạng
    khác nhau — xem bên dưới) -> {video_stem: nhãn 0-indexed}.

    - trainlistXX.txt: mỗi dòng "ClassName/video.avi <id 1-indexed>".
    - testlistXX.txt: mỗi dòng "ClassName/video.avi" (KHÔNG có id).

    CỐ Ý bỏ qua id có sẵn trong trainlistXX.txt (nếu có) — luôn tự suy
    nhãn từ TÊN THƯ MỤC trong đường dẫn, tra qua load_ucf_class_index().
    Chỉ tin DUY NHẤT 1 nguồn (classInd.txt) cho việc ánh xạ tên -> số,
    tránh lệch 1-indexed/0-indexed nếu 2 nguồn vô tình không khớp quy ước.
    """
    class_by_name = load_ucf_class_index(class_ind_path)

    label_by_stem: dict[str, int] = {}
    with open(split_list_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            relative_path = line.split()[0]  # bỏ phần id nếu có (trainlist), chỉ lấy path
            class_name = relative_path.split("/")[0]
            if class_name not in class_by_name:
                raise ValueError(
                    f"Class '{class_name}' trong {split_list_path} không có trong "
                    f"{class_ind_path} — 2 file không khớp nhau, kiểm tra lại nguồn tải UCF."
                )
            stem = Path(relative_path).stem
            label_by_stem[stem] = class_by_name[class_name]
    return label_by_stem


def write_labels_csv_from_ucf(
    class_ind_path: str | Path,
    split_list_path: str | Path,
    video_ext: str,
    output_csv: str | Path,
) -> Path:
    """Chuyển classInd.txt + (trainlistXX.txt HOẶC testlistXX.txt) thành
    labels.csv ĐÚNG FORMAT pipeline hiện tại đã dùng (video_filename,label,
    phiên 4.3) — để KHÔNG phải sửa Phase1FrameDataset/Phase2SequenceDataset/
    evaluate.py, chỉ thêm 1 nguồn nhãn mới song song với việc tự gõ tay.

    GIỚI HẠN CẦN BIẾT (phạm vi phiên 13.1, chưa giải quyết triệt để): file
    labels.csv sinh ra ở đây chỉ chứa video của 1 phía (train HOẶC test
    theo split_list_path bạn truyền) — vòng lặp train/val 80/20 hiện tại
    của project (is_val_sample, hash tất định, phiên 7.2) sẽ CHIA LẠI theo
    hash trên đúng tập bạn đưa vào, KHÔNG dùng nguyên vẹn 3-split chuẩn
    của UCF101 (trainlist01/02/03 dùng để so sánh benchmark công bố).
    Nghĩa là: ĐỌC ĐÚNG dữ liệu/nhãn, nhưng KHÔNG đúng protocol đánh giá
    chuẩn UCF101. Muốn khớp protocol chuẩn, cần sửa thêm datasets.py để
    nhận thẳng 2 dict train/test riêng thay vì 1 file rồi hash lại — nằm
    ngoài phạm vi phiên này.
    """
    label_by_stem = load_ucf_split(class_ind_path, split_list_path)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_filename", "label"])
        for stem, label in sorted(label_by_stem.items()):
            writer.writerow([f"{stem}{video_ext}", label])

    return output_path
