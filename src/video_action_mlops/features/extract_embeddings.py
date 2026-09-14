"""Extract embedding: TẦNG CACHE THỨ HAI trong 2 tầng cache (mục 3.5, phiên
2.1). Input: frame cache đã sample (data/interim/*.npy, phiên 4.2) +
checkpoint phase1 đã train (models/phase1_checkpoint.pt, phiên 4.3).
Output: sequence embedding (T, embed_dim_out) mỗi video.

Mục 3.1 roadmap: cache PHẢI khoá theo content-hash của (checkpoint phase1,
embedding_dim, frames_per_video, sampling strategy) — KHÔNG theo tên file
gốc. Thiếu điều này: đổi checkpoint phase1 mà quên xoá cache cũ sẽ cho ra
accuracy VÔ NGHĨA ở phase 2 mà KHÔNG có lỗi crash nào báo trước — lớp lỗi
silent nguy hiểm nhất của toàn bộ 2-stage pipeline.

GIỚI HẠN CHƯA XỬ LÝ (nợ kỹ thuật thật, không phải phạm vi phiên này):
SpatialBackbone (phiên 3.1) build từ resnet18 pretrained, conv1 cố định 3
input channel. Nếu cfg.data.use_motion_channel=True, data/interim/*.npy có
4 channel — sẽ fail-fast rõ ràng ở đây thay vì để PyTorch crash với lỗi
shape-mismatch khó hiểu. Cần vá spatial.py (mở rộng conv1) mới dùng được.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from video_action_mlops.config.schema import AppConfig
from video_action_mlops.models.registry import build_model
from video_action_mlops.models.spatial import SpatialBackbone


def _hash_file(path: str | Path, length: int = 8) -> str:
    """sha256 của NỘI DUNG file checkpoint. Đổi 1 byte trong checkpoint
    (vd train lại dù cùng config, weight init khác) cũng ra hash khác —
    đúng ý đồ "con người nhìn tên thư mục biết ngay cache khớp checkpoint
    nào" (mục 3.1)."""
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:length]


def _hash_sampling_config(cfg: AppConfig, length: int = 8) -> str:
    """sha256 của các field ẢNH HƯỞNG TRỰC TIẾP tới ý nghĩa của embedding.

    Cố ý KHÔNG hash toàn bộ AppConfig — vd đổi project.seed hay
    phase2.num_heads không ảnh hưởng gì tới nội dung embedding, không nên
    làm cache của TẦNG NÀY invalidate (dù dvc.yaml params vẫn đúng phạm vi
    riêng cho từng stage, xem phiên 4.2/4.3).
    """
    canonical = {
        "embed_dim_out": cfg.phase1.embed_dim_out,
        "frames_per_video": cfg.data.frames_per_video,
        "frame_size": list(cfg.data.frame_size),
        "use_motion_channel": cfg.data.use_motion_channel,
    }
    payload = json.dumps(canonical, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def resolve_cache_dir(cfg: AppConfig, checkpoint_path: str | Path, processed_root: str | Path) -> Path:
    """Ví dụ output: data/processed/video-action-mlops/3f9a1b2c_8e21fa0d/

    PUBLIC vì cả 2 phía đều cần gọi đúng 1 công thức này: extract_embeddings()
    (WRITER, ghi cache) và data/datasets.py + scripts/run_train_phase2.py
    (READER, đọc lại cache ở phiên 5.2) — nếu mỗi bên tự tính hash riêng,
    2 công thức dễ lệch nhau theo thời gian (bug kinh điển). Chỉ 1 nơi định
    nghĩa công thức, cả 2 phía import dùng chung.
    """
    checkpoint_hash = _hash_file(checkpoint_path)
    sampling_hash = _hash_sampling_config(cfg)
    return Path(processed_root) / cfg.project.name / f"{checkpoint_hash}_{sampling_hash}"


@torch.no_grad()
def extract_embeddings(
    cfg: AppConfig,
    checkpoint_path: str | Path,
    interim_dir: str | Path,
    processed_root: str | Path,
    device: str = "cpu",
) -> Path:
    """Chạy backbone đã train lên từng frame cache, trả về thư mục cache
    VỪA GHI (đã bao gồm hash — mục 3.1)."""
    if cfg.data.use_motion_channel:
        raise NotImplementedError(
            "cfg.data.use_motion_channel=True chưa được SpatialBackbone hỗ trợ "
            "(conv1 cố định 3 channel) — xem docstring đầu file. Đặt False hoặc "
            "vá spatial.py trước."
        )

    model: SpatialBackbone = build_model(cfg, stage="phase1", pretrained=False)  # type: ignore[assignment]
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    out_dir = resolve_cache_dir(cfg, checkpoint_path, processed_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    interim_paths = sorted(Path(interim_dir).glob("*.npy"))
    if not interim_paths:
        raise FileNotFoundError(
            f"Không tìm thấy .npy nào trong {interim_dir} — chạy stage "
            f"'preprocess' (phiên 4.2) trước."
        )

    for npy_path in interim_paths:
        frames = np.load(npy_path)  # (T, H, W, C) float32 [0,1] — xem preprocessing.py phiên 2.1
        frames_chw = torch.from_numpy(frames).permute(0, 3, 1, 2).float().to(device)  # (T, C, H, W)

        embeddings = model(frames_chw)  # (T, embed_dim_out) — mỗi frame 1 sample độc lập qua backbone

        out_path = out_dir / npy_path.name
        np.save(out_path, embeddings.cpu().numpy())
        print(f"[extract_features] {npy_path.name} -> {out_path} shape={tuple(embeddings.shape)}")

    return out_dir
