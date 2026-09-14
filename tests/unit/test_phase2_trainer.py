"""tests/unit/test_phase2_trainer.py — test training/phase2_trainer.py
(phiên 14.6).

KHÁC test_phase1_trainer.py: build_model(cfg, stage="phase2",
variant="trainable") trả về TemporalAggregatorTrainable — model train TỪ
ĐẦU (không có "pretrained=True" tải mạng như SpatialBackbone). Vì vậy
KHÔNG cần mock build_model — có thể chạy thật với kích thước siêu nhỏ,
nhanh (không phải giả lập).
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import Dataset

from video_action_mlops.config.schema import AppConfig
from video_action_mlops.training import phase2_trainer


class _TinySequenceDataset(Dataset):
    def __init__(self, n=8, seq_len=2, input_dim=8, num_classes=3, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.data = [torch.randn(seq_len, input_dim, generator=g) for _ in range(n)]
        self.labels = [i % num_classes for i in range(n)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def _make_tiny_cfg(epochs: int = 3, learning_rate: float = 0.01) -> AppConfig:
    return AppConfig.model_validate(
        {
            "project": {"name": "test", "seed": 42},
            "data": {
                "frames_per_video": 2,
                "frame_size": [4, 4],
                "use_motion_channel": False,
                "num_classes": 3,
            },
            "phase1": {"embed_dim_out": 8, "epochs": 1, "learning_rate": 0.01},
            "phase2": {
                "input_dim": 8,
                "num_heads": 2,
                "epochs": epochs,
                "learning_rate": learning_rate,
            },
        }
    )


def test_train_phase2_runs_full_loop_no_mock_needed():
    cfg = _make_tiny_cfg(epochs=3)
    dataset = _TinySequenceDataset(
        n=8,
        seq_len=cfg.data.frames_per_video,
        input_dim=cfg.phase2.input_dim,
        num_classes=cfg.data.num_classes,
    )

    result = phase2_trainer.train_phase2(cfg, train_dataset=dataset, batch_size=4)

    assert len(result["history"]) == cfg.phase2.epochs
    assert result["final"] == result["history"][-1]
    for epoch_metrics in result["history"]:
        assert "train_loss" in epoch_metrics
        assert "train_acc" in epoch_metrics
        assert 0.0 <= epoch_metrics["train_acc"] <= 1.0


def test_train_phase2_with_val_dataset_includes_val_metrics():
    cfg = _make_tiny_cfg(epochs=2)
    train_ds = _TinySequenceDataset(
        n=8,
        seq_len=cfg.data.frames_per_video,
        input_dim=cfg.phase2.input_dim,
        num_classes=cfg.data.num_classes,
        seed=0,
    )
    val_ds = _TinySequenceDataset(
        n=4,
        seq_len=cfg.data.frames_per_video,
        input_dim=cfg.phase2.input_dim,
        num_classes=cfg.data.num_classes,
        seed=1,
    )

    result = phase2_trainer.train_phase2(
        cfg, train_dataset=train_ds, val_dataset=val_ds, batch_size=4
    )

    for epoch_metrics in result["history"]:
        assert "val_loss" in epoch_metrics
        assert "val_acc" in epoch_metrics


def test_train_phase2_saves_checkpoint_with_expected_keys(tmp_path):
    cfg = _make_tiny_cfg(epochs=1)
    dataset = _TinySequenceDataset(
        n=4,
        seq_len=cfg.data.frames_per_video,
        input_dim=cfg.phase2.input_dim,
        num_classes=cfg.data.num_classes,
    )
    checkpoint_path = tmp_path / "phase2.pt"

    phase2_trainer.train_phase2(
        cfg, train_dataset=dataset, batch_size=4, checkpoint_path=checkpoint_path
    )

    assert checkpoint_path.exists()
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    # "in_proj_weight" là tham số ĐẶC TRƯNG của nn.MultiheadAttention (bên
    # trong TemporalAggregatorTrainable, phiên 3.2) — không lẫn với
    # TemporalAggregatorExport (q_proj/k_proj/v_proj riêng).
    assert "attn.in_proj_weight" in state_dict


def test_train_phase2_loss_decreases_over_many_epochs():
    """Kiểm tra HỌC ĐƯỢC THẬT, không chỉ 'chạy không lỗi cú pháp' — với đủ
    epoch trên dataset nhỏ (dễ overfit), loss trung bình epoch CUỐI phải
    THẤP HƠN epoch ĐẦU. Có torch.manual_seed để giảm nhiễu ngẫu nhiên,
    nhưng không tuyệt đối loại trừ flaky — nếu fail, thử chạy lại trước
    khi nghi ngờ code sai."""
    torch.manual_seed(0)
    cfg = _make_tiny_cfg(epochs=20, learning_rate=0.05)
    dataset = _TinySequenceDataset(
        n=8,
        seq_len=cfg.data.frames_per_video,
        input_dim=cfg.phase2.input_dim,
        num_classes=cfg.data.num_classes,
    )

    result = phase2_trainer.train_phase2(cfg, train_dataset=dataset, batch_size=8)

    first_loss = result["history"][0]["train_loss"]
    last_loss = result["history"][-1]["train_loss"]
    assert last_loss < first_loss


# ---------- resume state (phiên 13.2 — cùng thiết kế test_phase1_trainer.py,
# xem docstring bên đó cho lý do đầy đủ; ở đây đơn giản hơn vì
# train_phase2 không cần mock build_model) ----------


class _CrashingSequenceDataset(Dataset):
    """Giống _TinySequenceDataset, nhưng __getitem__ raise RuntimeError
    đúng lúc bắt đầu epoch thứ `crash_at_epoch` (0-indexed) — crash THẬT,
    không mock vòng lặp training (xem _CrashingFrameDataset trong
    test_phase1_trainer.py cho giải thích đầy đủ cơ chế đếm call)."""

    def __init__(
        self, n_per_epoch: int, crash_at_epoch: int, seq_len: int, input_dim: int,
        num_classes: int = 3, seed: int = 0,
    ):
        g = torch.Generator().manual_seed(seed)
        self.data = [torch.randn(seq_len, input_dim, generator=g) for _ in range(n_per_epoch)]
        self.labels = [i % num_classes for i in range(n_per_epoch)]
        self.n_per_epoch = n_per_epoch
        self.crash_at_call = n_per_epoch * crash_at_epoch
        self._calls = 0

    def __len__(self):
        return self.n_per_epoch

    def __getitem__(self, idx):
        if self._calls == self.crash_at_call:
            self._calls += 1
            raise RuntimeError("simulated crash: mat dien / het quota GPU giua chung")
        self._calls += 1
        return self.data[idx], self.labels[idx]


def test_resume_state_save_load_roundtrip(tmp_path):
    cfg = _make_tiny_cfg(epochs=3)
    model = phase2_trainer.build_model(cfg, stage="phase2", variant="trainable")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    path = tmp_path / "resume.pt"

    phase2_trainer.save_resume_state(
        model, optimizer, epoch=1, history=[{"epoch": 0}], path=path
    )

    assert path.exists()
    assert not (tmp_path / "resume.pt.tmp").exists()

    loaded = phase2_trainer.load_resume_state(path)
    assert loaded["epoch"] == 1
    assert loaded["history"] == [{"epoch": 0}]
    assert set(loaded["model_state"].keys()) == set(model.state_dict().keys())
    assert set(loaded["optimizer_state"].keys()) == {"state", "param_groups"}


def test_resume_state_load_returns_none_when_missing(tmp_path):
    assert phase2_trainer.load_resume_state(tmp_path / "khong_ton_tai.pt") is None


def test_resume_state_save_failure_cleans_up_tmp_and_keeps_old_file(monkeypatch, tmp_path):
    cfg = _make_tiny_cfg(epochs=3)
    model = phase2_trainer.build_model(cfg, stage="phase2", variant="trainable")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    path = tmp_path / "resume.pt"

    phase2_trainer.save_resume_state(model, optimizer, 1, [{"epoch": 0}], path)
    old_bytes = path.read_bytes()

    monkeypatch.setattr(
        phase2_trainer.os,
        "replace",
        lambda *a, **k: (_ for _ in ()).throw(OSError("disk day")),
    )

    with pytest.raises(OSError, match="disk day"):
        phase2_trainer.save_resume_state(model, optimizer, 2, [{"epoch": 0}, {"epoch": 1}], path)

    assert path.read_bytes() == old_bytes
    assert not (tmp_path / "resume.pt.tmp").exists()


def test_train_phase2_without_resume_path_never_writes_resume_file(tmp_path):
    cfg = _make_tiny_cfg(epochs=2)
    dataset = _TinySequenceDataset(
        n=8, seq_len=cfg.data.frames_per_video, input_dim=cfg.phase2.input_dim,
        num_classes=cfg.data.num_classes,
    )

    phase2_trainer.train_phase2(cfg, train_dataset=dataset, batch_size=4)

    assert list(tmp_path.iterdir()) == []


def test_train_phase2_genuine_crash_then_resume_completes_remaining_epochs_only(tmp_path):
    """Cùng kịch bản test_train_phase1_genuine_crash_then_resume... —
    crash THẬT ở epoch 2/4, resume, verify history liền mạch 0..3, file
    resume bị xoá sau khi xong."""
    cfg = _make_tiny_cfg(epochs=4)
    resume_path = tmp_path / "phase2_resume.pt"

    crashing_dataset = _CrashingSequenceDataset(
        n_per_epoch=8, crash_at_epoch=2, seq_len=cfg.data.frames_per_video,
        input_dim=cfg.phase2.input_dim, num_classes=cfg.data.num_classes,
    )
    with pytest.raises(RuntimeError, match="simulated crash"):
        phase2_trainer.train_phase2(
            cfg, train_dataset=crashing_dataset, batch_size=4, resume_state_path=resume_path
        )

    assert resume_path.exists()
    partial_state = phase2_trainer.load_resume_state(resume_path)
    assert partial_state["epoch"] == 2
    assert len(partial_state["history"]) == 2

    safe_dataset = _TinySequenceDataset(
        n=8, seq_len=cfg.data.frames_per_video, input_dim=cfg.phase2.input_dim,
        num_classes=cfg.data.num_classes,
    )
    result = phase2_trainer.train_phase2(
        cfg, train_dataset=safe_dataset, batch_size=4, resume_state_path=resume_path
    )

    assert [m["epoch"] for m in result["history"]] == [0, 1, 2, 3]
    assert not resume_path.exists()
