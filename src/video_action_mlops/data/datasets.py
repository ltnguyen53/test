"""Dataset cho phase 1 + phase 2: đọc frame cache / embedding cache + nhãn.

Nhãn lấy từ CSV (mặc định data/raw/labels.csv, cột video_filename,label) —
file này PHẢI do người dùng tự cung cấp tương ứng với video trong
data/raw/. Đây là giới hạn thật của mọi pipeline video action recognition:
nhãn luôn phải đến từ bên ngoài, không thể tự suy ra từ chính video.
Fail-fast nếu thiếu nhãn cho video nào — không âm thầm bỏ qua.

Cả 2 class hỗ trợ tham số `split` ("train"/"val"/"all", phiên 7.2) — chia
tất định theo hash tên video (data/labels.py), KHÔNG cần cột "split" riêng
trong labels.csv.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from video_action_mlops.data.labels import is_val_sample, load_label_by_stem

Split = Literal["train", "val", "all"]


def _keep_by_split(stem: str, split: Split) -> bool:
    if split == "all":
        return True
    is_val = is_val_sample(stem)
    return is_val if split == "val" else not is_val


class Phase1FrameDataset(Dataset):
    """Mỗi sample = 1 FRAME (không phải 1 video) — mỗi frame kế thừa nhãn
    của video chứa nó. Dùng để fine-tune SpatialBackbone bằng proxy task
    phân loại per-frame (training/phase1_trainer.py) trước khi cache
    embedding cho phase 2 ở extract_embeddings.py (tuần 5).
    """

    def __init__(
        self, interim_dir: str | Path, labels_csv: str | Path, split: Split = "all"
    ) -> None:
        self.interim_dir = Path(interim_dir)
        label_by_stem = load_label_by_stem(labels_csv)

        npy_paths = sorted(self.interim_dir.glob("*.npy"))
        if not npy_paths:
            raise FileNotFoundError(
                f"Không tìm thấy .npy nào trong {self.interim_dir} — chạy stage "
                f"'preprocess' (phiên 4.2) trước."
            )

        # Index phẳng: (đường dẫn .npy, frame_idx trong video, label). Phẳng
        # hoá ra list thay vì giữ nguyên theo video để DataLoader shuffle
        # được ở CẤP FRAME (đúng ý đồ proxy task per-frame).
        self._index: list[tuple[Path, int, int]] = []
        for npy_path in npy_paths:
            stem = npy_path.stem
            if not _keep_by_split(stem, split):
                continue
            if stem not in label_by_stem:
                raise ValueError(
                    f"{npy_path.name} không có nhãn tương ứng trong {labels_csv} "
                    f"(thiếu dòng cho '{stem}')"
                )
            label = label_by_stem[stem]
            num_frames = np.load(npy_path, mmap_mode="r").shape[0]  # mmap: không load hết array chỉ để lấy shape
            for frame_idx in range(num_frames):
                self._index.append((npy_path, frame_idx, label))

        if not self._index:
            raise ValueError(
                f"split='{split}' không còn video nào sau khi lọc — dataset quá "
                f"nhỏ để chia train/val có ý nghĩa, hoặc tất cả video rơi vào "
                f"phía kia. Thử split='all' để debug."
            )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        npy_path, frame_idx, label = self._index[idx]
        frames = np.load(npy_path)  # (T, H, W, C) float32 [0,1] — xem preprocessing.py phiên 2.1
        frame = frames[frame_idx]  # (H, W, C)
        frame_chw = torch.from_numpy(frame).permute(2, 0, 1).contiguous()  # (C, H, W)
        return frame_chw, label


class Phase2SequenceDataset(Dataset):
    """Mỗi sample = 1 VIDEO (nguyên chuỗi embedding), khác Phase1FrameDataset
    ở chỗ KHÔNG phẳng hoá theo frame — TemporalAggregatorTrainable (phiên
    3.2) cần cả chuỗi (T, embed_dim_out) để tự làm attention theo thời gian,
    không train per-frame như phase 1.

    Đọc từ thư mục cache ĐÃ ĐƯỢC RESOLVE bằng
    features.extract_embeddings.resolve_cache_dir() — không tự đoán đường
    dẫn, để tránh lệch với nơi extract_embeddings.py thật sự ghi (phiên 5.1).
    """

    def __init__(
        self, processed_dir: str | Path, labels_csv: str | Path, split: Split = "all"
    ) -> None:
        self.processed_dir = Path(processed_dir)
        label_by_stem = load_label_by_stem(labels_csv)

        npy_paths = sorted(self.processed_dir.glob("*.npy"))
        if not npy_paths:
            raise FileNotFoundError(
                f"Không tìm thấy .npy nào trong {self.processed_dir} — chạy stage "
                f"'extract_features' (phiên 5.1) trước, hoặc processed_dir "
                f"chưa đúng (kiểm tra resolve_cache_dir)."
            )

        self._paths_labels: list[tuple[Path, int]] = []
        for npy_path in npy_paths:
            stem = npy_path.stem
            if not _keep_by_split(stem, split):
                continue
            if stem not in label_by_stem:
                raise ValueError(
                    f"{npy_path.name} không có nhãn tương ứng trong {labels_csv} "
                    f"(thiếu dòng cho '{stem}')"
                )
            self._paths_labels.append((npy_path, label_by_stem[stem]))

        if not self._paths_labels:
            raise ValueError(
                f"split='{split}' không còn video nào sau khi lọc — dataset quá "
                f"nhỏ để chia train/val có ý nghĩa, hoặc tất cả video rơi vào "
                f"phía kia. Thử split='all' để debug."
            )

    def __len__(self) -> int:
        return len(self._paths_labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        npy_path, label = self._paths_labels[idx]
        seq = np.load(npy_path)  # (T, embed_dim_out) float32 — xem extract_embeddings.py phiên 5.1
        return torch.from_numpy(seq).float(), label
