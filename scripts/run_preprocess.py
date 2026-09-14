"""CLI wrapper mỏng: đọc video thô từ data/raw/, decode, resize, gọi
preprocess_frames() (hàm thuần, phiên 2.1-2.2), ghi cache frame vào
data/interim/ — đây là TẦNG CACHE THỨ NHẤT trong 2 tầng cache (mục 3.5
roadmap, phiên 2.1).

SỬA LẠI (lỗi nghiêm trọng phát hiện qua rà soát dependency, KHÔNG phải
cải tiến tuỳ chọn): bản gốc dùng torchvision.io.read_video — API này bị
torchvision XOÁ HẲN từ bản 0.28 (deprecate từ 0.22, đã tra cứu changelog
chính thức pytorch/vision, không phải suy đoán), trong khi pyproject.toml
khai "torchvision>=0.17" không giới hạn trên — nghĩa là `pip install -e .`
hôm nay cài bản mới nhất và CRASH NGAY ở dòng import torchvision.io. Đã
chuyển sang dùng CHUNG decode_video_bytes()/resize_frames_np()
(inference/pipeline.py, phiên 10.2 — vốn đã dùng opencv cho serving) —
nhân tiện xoá luôn "bất đối xứng" giữa decode lúc train (từng là
torchvision) và decode lúc serving (opencv) từng ghi chú ở nhiều phiên
trước: hoá ra dùng chung opencv cho cả 2 nơi đơn giản và an toàn hơn hẳn.

Script này KHÔNG chứa logic xử lý frame — toàn bộ logic đó nằm ở
src/video_action_mlops/data/preprocessing.py và inference/pipeline.py.
Sau khi sửa, script này KHÔNG còn cần torch/torchvision — chỉ opencv +
numpy (giống inference/pipeline.py).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from video_action_mlops.config.loader import load_config
from video_action_mlops.data.preprocessing import preprocess_frames
from video_action_mlops.inference.pipeline import decode_video_bytes, resize_frames_np


def run(config_path: str, raw_dir: str, interim_dir: str) -> None:
    cfg = load_config(config_path)
    raw = Path(raw_dir)
    out_dir = Path(interim_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Đệ quy (rglob) — UCF101/UCF11 (phiên 13.1) đặt video trong thư mục
    # con theo tên class, khác video tự quay phẳng 1 cấp (tuần 1-12).
    video_paths = sorted(set(raw.rglob("*.mp4")) | set(raw.rglob("*.avi")))
    if not video_paths:
        raise FileNotFoundError(
            f"Không tìm thấy video .mp4/.avi nào trong {raw} (kể cả thư mục con) — "
            f"cần vài video mẫu trước khi chạy 'dvc repro' (xem đầu ra kiểm chứng "
            f"phiên 4.2, cấu trúc UCF xem phiên 13.1)."
        )

    for video_path in video_paths:
        frames = decode_video_bytes(video_path.read_bytes())  # (T, H, W, C) uint8 RGB (opencv)
        frames = resize_frames_np(frames, cfg.data.frame_size)

        processed = preprocess_frames(
            frames,
            frames_per_video=cfg.data.frames_per_video,
            use_motion_channel=cfg.data.use_motion_channel,
            apply_hflip=False,  # cache phải deterministic — xem docstring module
        )

        out_path = out_dir / f"{video_path.stem}.npy"
        np.save(out_path, processed)
        print(f"[preprocess] {video_path.name} -> {out_path} shape={processed.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess video thô -> frame cache (data/interim/)"
    )
    parser.add_argument("--config", required=True, help="Đường dẫn config YAML")
    parser.add_argument("--raw-dir", default="data/raw", help="Thư mục video thô")
    parser.add_argument("--interim-dir", default="data/interim", help="Thư mục ghi cache frame")
    args = parser.parse_args()
    run(args.config, args.raw_dir, args.interim_dir)


if __name__ == "__main__":
    main()
