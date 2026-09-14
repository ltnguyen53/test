"""tests/unit/test_extract_embeddings.py — test features/extract_embeddings.py
(phiên 14.5).

_hash_file/_hash_sampling_config/resolve_cache_dir là hàm THUẦN (chỉ
hashlib/json/pathlib, không cần torch) — chạy được ngay cả khi torch chưa
cài, khác extract_embeddings() chính (cần build model thật, phiên 3.3).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from video_action_mlops.config.schema import AppConfig
from video_action_mlops.features.extract_embeddings import (
    _hash_file,
    _hash_sampling_config,
    extract_embeddings,
    resolve_cache_dir,
)
from video_action_mlops.models.registry import build_model


def _make_tiny_cfg_dict() -> dict:
    return {
        "project": {"name": "test-project", "seed": 42},
        "data": {
            "frames_per_video": 2,
            "frame_size": [16, 16],
            "use_motion_channel": False,
            "num_classes": 3,
        },
        "phase1": {"embed_dim_out": 8, "epochs": 1, "learning_rate": 0.001},
        "phase2": {"input_dim": 8, "num_heads": 2, "epochs": 1, "learning_rate": 0.001},
    }


def _make_tiny_cfg() -> AppConfig:
    return AppConfig.model_validate(_make_tiny_cfg_dict())


# ---------- _hash_file ----------


def test_hash_file_same_content_same_hash(tmp_path):
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_bytes(b"noi dung giong het")
    f2.write_bytes(b"noi dung giong het")
    assert _hash_file(f1) == _hash_file(f2)


def test_hash_file_different_content_different_hash(tmp_path):
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_bytes(b"noi dung 1")
    f2.write_bytes(b"noi dung KHAC hang")
    assert _hash_file(f1) != _hash_file(f2)


# ---------- _hash_sampling_config ----------


def test_hash_sampling_config_changes_with_frames_per_video():
    cfg_a = _make_tiny_cfg()
    cfg_b_dict = _make_tiny_cfg_dict()
    cfg_b_dict["data"]["frames_per_video"] = 99
    cfg_b = AppConfig.model_validate(cfg_b_dict)
    assert _hash_sampling_config(cfg_a) != _hash_sampling_config(cfg_b)


def test_hash_sampling_config_unaffected_by_phase1_hyperparams():
    """epochs/learning_rate KHÔNG ảnh hưởng ý nghĩa embedding — không nên
    làm cache invalidate (xem docstring extract_embeddings.py, phiên 5.1)."""
    cfg_a = _make_tiny_cfg()
    cfg_b_dict = _make_tiny_cfg_dict()
    cfg_b_dict["phase1"]["epochs"] = 999
    cfg_b_dict["phase1"]["learning_rate"] = 0.5
    cfg_b = AppConfig.model_validate(cfg_b_dict)
    assert _hash_sampling_config(cfg_a) == _hash_sampling_config(cfg_b)


# ---------- resolve_cache_dir ----------


def test_resolve_cache_dir_consistent_across_calls(tmp_path):
    cfg = _make_tiny_cfg()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"fake checkpoint bytes")

    dir1 = resolve_cache_dir(cfg, checkpoint, tmp_path / "processed")
    dir2 = resolve_cache_dir(cfg, checkpoint, tmp_path / "processed")
    # WRITER (extract_embeddings) va READER (run_train_phase2.py, phien
    # 5.2) PHAI luon ra CUNG 1 duong dan tu cung input.
    assert dir1 == dir2


def test_resolve_cache_dir_changes_when_checkpoint_changes(tmp_path):
    cfg = _make_tiny_cfg()
    checkpoint_a = tmp_path / "a.pt"
    checkpoint_b = tmp_path / "b.pt"
    checkpoint_a.write_bytes(b"checkpoint A")
    checkpoint_b.write_bytes(b"checkpoint B khac han")

    dir_a = resolve_cache_dir(cfg, checkpoint_a, tmp_path / "processed")
    dir_b = resolve_cache_dir(cfg, checkpoint_b, tmp_path / "processed")
    assert dir_a != dir_b


# ---------- extract_embeddings (end-to-end, model thật siêu nhỏ) ----------


def test_extract_embeddings_end_to_end(tmp_path):
    cfg = _make_tiny_cfg()
    model = build_model(cfg, stage="phase1", pretrained=False)
    checkpoint_path = tmp_path / "phase1.pt"
    torch.save(model.state_dict(), checkpoint_path)

    interim_dir = tmp_path / "interim"
    interim_dir.mkdir()
    rng = np.random.default_rng(0)
    np.save(interim_dir / "video_a.npy", rng.random((2, 16, 16, 3)).astype(np.float32))
    np.save(interim_dir / "video_b.npy", rng.random((2, 16, 16, 3)).astype(np.float32))

    out_dir = extract_embeddings(cfg, checkpoint_path, interim_dir, tmp_path / "processed")

    assert out_dir.exists()
    assert (out_dir / "video_a.npy").exists()
    assert (out_dir / "video_b.npy").exists()

    emb_a = np.load(out_dir / "video_a.npy")
    assert emb_a.shape == (2, 8)  # (T, embed_dim_out)


def test_extract_embeddings_raises_on_motion_channel(tmp_path):
    cfg_dict = _make_tiny_cfg_dict()
    cfg_dict["data"]["use_motion_channel"] = True
    cfg = AppConfig.model_validate(cfg_dict)

    # Check use_motion_channel raise NGAY ĐẦU hàm, TRƯỚC khi chạm tới
    # checkpoint — không cần file checkpoint thật cho test này (fail-fast
    # trước khi tốn công load model).
    with pytest.raises(NotImplementedError, match="use_motion_channel"):
        extract_embeddings(cfg, tmp_path / "khong_ton_tai.pt", tmp_path, tmp_path / "processed")


def test_extract_embeddings_empty_interim_raises(tmp_path):
    cfg = _make_tiny_cfg()
    model = build_model(cfg, stage="phase1", pretrained=False)
    checkpoint_path = tmp_path / "phase1.pt"
    torch.save(model.state_dict(), checkpoint_path)

    interim_dir = tmp_path / "interim"
    interim_dir.mkdir()  # rỗng, cố ý

    with pytest.raises(FileNotFoundError):
        extract_embeddings(cfg, checkpoint_path, interim_dir, tmp_path / "processed")
