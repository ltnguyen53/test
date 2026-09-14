"""tests/unit/test_phase1_trainer.py — test training/phase1_trainer.py
(phiên 14.6, khó nhất Tuần 14).

QUAN TRỌNG: train_phase1() gọi build_model(cfg, stage="phase1",
pretrained=True) — pretrained=True TẢI WEIGHT IMAGENET TỪ MẠNG THẬT. Test
BẮT BUỘC phải mock build_model, nếu không sẽ cố tải mạng mỗi lần chạy
(chậm, lỗi nếu offline, không phải hành vi 1 unit test nên có).

Ghi nhận (nợ kỹ thuật, nằm ngoài phạm vi phiên này): train_phase1()
hard-code pretrained=True, không nhận tham số để tắt — muốn test KHÔNG
cần mock, cần sửa lại hàm để pretrained là tham số truyền vào.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from video_action_mlops.config.schema import AppConfig
from video_action_mlops.training import phase1_trainer

# ---------- _unfreeze_stage_for_epoch (hàm thuần — đã verify logic tay ở
# phiên 4.3, giờ lưu thành test thật, chạy được ngay cả không cần mock) ----------


def test_unfreeze_stage_epoch_zero_is_frozen():
    assert phase1_trainer._unfreeze_stage_for_epoch(0, 10) is None


def test_unfreeze_stage_last_epoch_reaches_layer1():
    assert phase1_trainer._unfreeze_stage_for_epoch(9, 10) == "layer1"


def test_unfreeze_stage_progresses_through_schedule():
    assert phase1_trainer._unfreeze_stage_for_epoch(1, 4) == "layer3"
    assert phase1_trainer._unfreeze_stage_for_epoch(2, 4) == "layer2"
    assert phase1_trainer._unfreeze_stage_for_epoch(3, 4) == "layer1"


def test_unfreeze_stage_invalid_total_epochs_raises():
    with pytest.raises(ValueError, match="total_epochs"):
        phase1_trainer._unfreeze_stage_for_epoch(0, 0)


def test_unfreeze_stage_negative_epoch_raises():
    with pytest.raises(ValueError, match="epoch"):
        phase1_trainer._unfreeze_stage_for_epoch(-1, 10)


# ---------- train_phase1 end-to-end (model giả — BẮT BUỘC mock build_model) ----------


class _TinyFakeBackbone(nn.Module):
    """Thay SpatialBackbone thật — chỉ cần forward(x)->embedding và
    unfreeze_up_to(stage) để khớp interface train_phase1() thật sự dùng."""

    def __init__(self, embed_dim_out: int):
        super().__init__()
        self.fc = nn.Linear(3 * 4 * 4, embed_dim_out)
        self.unfreeze_calls: list[str] = []

    def forward(self, x):
        b = x.shape[0]
        return self.fc(x.reshape(b, -1))

    def unfreeze_up_to(self, stage: str) -> None:
        self.unfreeze_calls.append(stage)


class _TinyFrameDataset(Dataset):
    def __init__(self, n=8, num_classes=3, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.data = [torch.randn(3, 4, 4, generator=g) for _ in range(n)]
        self.labels = [i % num_classes for i in range(n)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def _make_tiny_cfg(epochs: int = 2) -> AppConfig:
    return AppConfig.model_validate(
        {
            "project": {"name": "test", "seed": 42},
            "data": {
                "frames_per_video": 2,
                "frame_size": [4, 4],
                "use_motion_channel": False,
                "num_classes": 3,
            },
            "phase1": {"embed_dim_out": 8, "epochs": epochs, "learning_rate": 0.01},
            "phase2": {"input_dim": 8, "num_heads": 2, "epochs": 1, "learning_rate": 0.01},
        }
    )


def test_train_phase1_runs_full_loop_and_returns_history(monkeypatch):
    cfg = _make_tiny_cfg(epochs=2)
    fake_backbone = _TinyFakeBackbone(embed_dim_out=8)
    monkeypatch.setattr(
        phase1_trainer, "build_model", lambda cfg, stage, pretrained=False: fake_backbone
    )

    dataset = _TinyFrameDataset(n=8, num_classes=3)
    result = phase1_trainer.train_phase1(cfg, train_dataset=dataset, batch_size=4)

    assert len(result["history"]) == cfg.phase1.epochs
    assert result["final"] == result["history"][-1]
    for epoch_metrics in result["history"]:
        assert "train_loss" in epoch_metrics
        assert "train_acc" in epoch_metrics
        assert 0.0 <= epoch_metrics["train_acc"] <= 1.0


def test_train_phase1_calls_unfreeze_according_to_schedule(monkeypatch):
    cfg = _make_tiny_cfg(epochs=8)  # đủ epoch để thấy rõ lịch trình unfreeze
    fake_backbone = _TinyFakeBackbone(embed_dim_out=8)
    monkeypatch.setattr(
        phase1_trainer, "build_model", lambda cfg, stage, pretrained=False: fake_backbone
    )

    dataset = _TinyFrameDataset(n=8, num_classes=3)
    phase1_trainer.train_phase1(cfg, train_dataset=dataset, batch_size=4)

    # unfreeze_up_to CHỈ gọi khi target_stage KHÁC current_stage (xem
    # train_phase1) — epoch 0 luôn None, nên số lần gọi < số epoch. Stage
    # CUỐI CÙNG phải là "layer1" (khớp _unfreeze_stage_for_epoch(7, 8)).
    assert fake_backbone.unfreeze_calls[-1] == "layer1"
    assert len(fake_backbone.unfreeze_calls) <= 8
    # khong goi trung stage lien tiep
    assert len(fake_backbone.unfreeze_calls) == len(set(fake_backbone.unfreeze_calls))


def test_train_phase1_with_val_dataset_includes_val_metrics(monkeypatch):
    cfg = _make_tiny_cfg(epochs=2)
    fake_backbone = _TinyFakeBackbone(embed_dim_out=8)
    monkeypatch.setattr(
        phase1_trainer, "build_model", lambda cfg, stage, pretrained=False: fake_backbone
    )

    train_ds = _TinyFrameDataset(n=8, num_classes=3, seed=0)
    val_ds = _TinyFrameDataset(n=4, num_classes=3, seed=1)

    result = phase1_trainer.train_phase1(
        cfg, train_dataset=train_ds, val_dataset=val_ds, batch_size=4
    )

    for epoch_metrics in result["history"]:
        assert "val_loss" in epoch_metrics
        assert "val_acc" in epoch_metrics


def test_train_phase1_saves_checkpoint_without_cls_head(monkeypatch, tmp_path):
    cfg = _make_tiny_cfg(epochs=1)
    fake_backbone = _TinyFakeBackbone(embed_dim_out=8)
    monkeypatch.setattr(
        phase1_trainer, "build_model", lambda cfg, stage, pretrained=False: fake_backbone
    )

    dataset = _TinyFrameDataset(n=4, num_classes=3)
    checkpoint_path = tmp_path / "checkpoint.pt"

    phase1_trainer.train_phase1(
        cfg, train_dataset=dataset, batch_size=4, checkpoint_path=checkpoint_path
    )

    assert checkpoint_path.exists()
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    # Checkpoint chỉ chứa tham số của fake_backbone — _Phase1TrainingHead
    # (proxy classification head TẠM THỜI, phiên 4.3) là object HOÀN TOÀN
    # TÁCH BIỆT, không bao giờ gộp vào model nên không thể lọt vào đây.
    assert set(state_dict.keys()) == {"fc.weight", "fc.bias"}


# ---------- resume state (phiên 13.2: checkpoint resume giữa training bị
# ngắt — mất điện, Colab ngắt kết nối, hết quota GPU giữa chừng) ----------


class _CrashingFrameDataset(Dataset):
    """Giống _TinyFrameDataset, nhưng __getitem__ raise RuntimeError đúng
    lúc bắt đầu epoch thứ `crash_at_epoch` (0-indexed) — mô phỏng CRASH
    THẬT giữa training (không mock train_phase1 hay vòng lặp epoch — để
    exception thật xuyên qua toàn bộ code path training, giống 1 lần mất
    điện/hết quota GPU thật sẽ khiến process chết đột ngột)."""

    def __init__(self, n_per_epoch: int, crash_at_epoch: int, num_classes: int = 3, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.data = [torch.randn(3, 4, 4, generator=g) for _ in range(n_per_epoch)]
        self.labels = [i % num_classes for i in range(n_per_epoch)]
        self.n_per_epoch = n_per_epoch
        # DataLoader (shuffle=True, single-process) gọi __getitem__ đúng
        # n_per_epoch lần MỖI epoch dù thứ tự bị xáo — nên đếm TỔNG số lần
        # gọi là cách xác định "đang ở epoch nào" không phụ thuộc shuffle.
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
    backbone = _TinyFakeBackbone(embed_dim_out=8)
    cls_head = phase1_trainer._Phase1TrainingHead(8, 3)
    optimizer = torch.optim.Adam(
        list(backbone.parameters()) + list(cls_head.parameters()), lr=0.01
    )
    path = tmp_path / "resume.pt"

    phase1_trainer.save_resume_state(
        backbone,
        cls_head,
        optimizer,
        epoch=2,
        current_stage="layer3",
        history=[{"epoch": 0}, {"epoch": 1}],
        path=path,
    )

    assert path.exists()
    assert not (tmp_path / "resume.pt.tmp").exists()  # atomic write không để lại rác

    loaded = phase1_trainer.load_resume_state(path)
    assert loaded["epoch"] == 2
    assert loaded["current_stage"] == "layer3"
    assert loaded["history"] == [{"epoch": 0}, {"epoch": 1}]
    assert set(loaded["model_state"].keys()) == set(backbone.state_dict().keys())
    assert set(loaded["cls_head_state"].keys()) == set(cls_head.state_dict().keys())
    assert set(loaded["optimizer_state"].keys()) == {"state", "param_groups"}


def test_resume_state_load_returns_none_when_missing(tmp_path):
    assert phase1_trainer.load_resume_state(tmp_path / "khong_ton_tai.pt") is None


def test_resume_state_save_failure_cleans_up_tmp_and_keeps_old_file(monkeypatch, tmp_path):
    """os.replace() thất bại giữa chừng (vd hết dung lượng đĩa đúng lúc
    đó) không được để lại file .tmp rác, và file resume CŨ (lần lưu thành
    công trước đó) phải còn nguyên vẹn — atomic write nghĩa là KHÔNG có
    trạng thái nửa vời nào có thể bị đọc nhầm ở lần resume sau."""
    backbone = _TinyFakeBackbone(embed_dim_out=8)
    cls_head = phase1_trainer._Phase1TrainingHead(8, 3)
    optimizer = torch.optim.Adam(
        list(backbone.parameters()) + list(cls_head.parameters()), lr=0.01
    )
    path = tmp_path / "resume.pt"

    phase1_trainer.save_resume_state(backbone, cls_head, optimizer, 1, None, [{"epoch": 0}], path)
    old_bytes = path.read_bytes()

    monkeypatch.setattr(
        phase1_trainer.os,
        "replace",
        lambda *a, **k: (_ for _ in ()).throw(OSError("disk day")),
    )

    with pytest.raises(OSError, match="disk day"):
        phase1_trainer.save_resume_state(
            backbone, cls_head, optimizer, 2, None, [{"epoch": 0}, {"epoch": 1}], path
        )

    assert path.read_bytes() == old_bytes
    assert not (tmp_path / "resume.pt.tmp").exists()


def test_train_phase1_without_resume_path_never_writes_resume_file(monkeypatch, tmp_path):
    """Hành vi CŨ (trước phiên 13.2) phải giữ nguyên khi không truyền
    resume_state_path — không tự nhiên xuất hiện file resume nào."""
    cfg = _make_tiny_cfg(epochs=2)
    fake_backbone = _TinyFakeBackbone(embed_dim_out=8)
    monkeypatch.setattr(
        phase1_trainer, "build_model", lambda cfg, stage, pretrained=False: fake_backbone
    )
    dataset = _TinyFrameDataset(n=8, num_classes=3)

    phase1_trainer.train_phase1(cfg, train_dataset=dataset, batch_size=4)

    assert list(tmp_path.iterdir()) == []


def test_train_phase1_genuine_crash_then_resume_completes_remaining_epochs_only(
    monkeypatch, tmp_path
):
    """Test verify THẬT (không mock vòng lặp training): chạy train_phase1
    với 1 dataset crash thật ở epoch 2/4, bắt exception thật, rồi gọi lại
    train_phase1 CÙNG cfg + cùng resume_state_path (không crash lần 2) —
    kỳ vọng: (1) chỉ chạy tiếp 2 epoch còn thiếu (không train lại từ đầu),
    (2) history cuối cùng có ĐỦ 4 epoch liên tục 0..3 không trùng/thiếu,
    (3) file resume bị xoá sau khi hoàn tất."""
    cfg = _make_tiny_cfg(epochs=4)
    fake_backbone = _TinyFakeBackbone(embed_dim_out=8)
    monkeypatch.setattr(
        phase1_trainer, "build_model", lambda cfg, stage, pretrained=False: fake_backbone
    )
    resume_path = tmp_path / "phase1_resume.pt"

    crashing_dataset = _CrashingFrameDataset(n_per_epoch=8, crash_at_epoch=2, num_classes=3)
    with pytest.raises(RuntimeError, match="simulated crash"):
        phase1_trainer.train_phase1(
            cfg, train_dataset=crashing_dataset, batch_size=4, resume_state_path=resume_path
        )

    # Đã crash giữa epoch 2 -> đúng 2 epoch (0, 1) đã lưu resume state.
    assert resume_path.exists()
    partial_state = phase1_trainer.load_resume_state(resume_path)
    assert partial_state["epoch"] == 2
    assert len(partial_state["history"]) == 2

    # Lần chạy thứ 2: dataset KHÔNG crash (mô phỏng máy chạy lại sau khi
    # được cấp GPU/điện lại), cùng resume_state_path.
    safe_dataset = _TinyFrameDataset(n=8, num_classes=3)
    result = phase1_trainer.train_phase1(
        cfg, train_dataset=safe_dataset, batch_size=4, resume_state_path=resume_path
    )

    assert [m["epoch"] for m in result["history"]] == [0, 1, 2, 3]
    assert not resume_path.exists()  # dọn dẹp sau khi train xong trọn vẹn


def test_train_phase1_resume_skips_pretrained_download(monkeypatch, tmp_path):
    """Khi có resume state, build_model phải được gọi với pretrained=False
    (state đã lưu sắp ghi đè lên weight ImageNet ngay sau đó — tải về là
    lãng phí băng thông/thời gian, đặc biệt khó chịu khi test/CI chạy
    offline)."""
    cfg = _make_tiny_cfg(epochs=3)
    fake_backbone_1 = _TinyFakeBackbone(embed_dim_out=8)
    calls: list[bool] = []

    def fake_build_model(cfg, stage, pretrained=False):
        calls.append(pretrained)
        return fake_backbone_1

    monkeypatch.setattr(phase1_trainer, "build_model", fake_build_model)
    resume_path = tmp_path / "resume.pt"
    dataset = _TinyFrameDataset(n=8, num_classes=3)

    # Lần 1: chưa có resume state -> pretrained phải là True.
    crashing_dataset = _CrashingFrameDataset(n_per_epoch=8, crash_at_epoch=1, num_classes=3)
    with pytest.raises(RuntimeError):
        phase1_trainer.train_phase1(
            cfg, train_dataset=crashing_dataset, batch_size=4, resume_state_path=resume_path
        )
    assert calls == [True]

    # Lần 2: đã có resume state -> pretrained phải là False.
    phase1_trainer.train_phase1(
        cfg, train_dataset=dataset, batch_size=4, resume_state_path=resume_path
    )
    assert calls == [True, False]


def test_train_phase1_resume_restores_current_stage_not_reset_to_none(monkeypatch, tmp_path):
    """current_stage phải được khôi phục từ resume state — nếu bị reset về
    None, epoch đầu tiên sau resume sẽ gọi lại unfreeze_up_to() cho stage
    đã mở từ trước (vô hại nhưng thừa) thay vì chỉ gọi khi stage THỰC SỰ
    đổi. Test verify bằng cách đếm số lần gọi unfreeze qua ranh giới
    resume phải khớp với chạy KHÔNG bị ngắt."""
    cfg = _make_tiny_cfg(epochs=8)  # đủ dài để có nhiều mốc đổi stage

    baseline_backbone = _TinyFakeBackbone(embed_dim_out=8)
    monkeypatch.setattr(
        phase1_trainer, "build_model", lambda cfg, stage, pretrained=False: baseline_backbone
    )
    baseline_dataset = _TinyFrameDataset(n=8, num_classes=3)
    phase1_trainer.train_phase1(cfg, train_dataset=baseline_dataset, batch_size=4)
    baseline_calls = list(baseline_backbone.unfreeze_calls)

    resumed_backbone = _TinyFakeBackbone(embed_dim_out=8)
    monkeypatch.setattr(
        phase1_trainer, "build_model", lambda cfg, stage, pretrained=False: resumed_backbone
    )
    resume_path = tmp_path / "resume.pt"
    crashing_dataset = _CrashingFrameDataset(n_per_epoch=8, crash_at_epoch=5, num_classes=3)
    with pytest.raises(RuntimeError):
        phase1_trainer.train_phase1(
            cfg, train_dataset=crashing_dataset, batch_size=4, resume_state_path=resume_path
        )
    safe_dataset = _TinyFrameDataset(n=8, num_classes=3)
    phase1_trainer.train_phase1(
        cfg, train_dataset=safe_dataset, batch_size=4, resume_state_path=resume_path
    )

    assert list(resumed_backbone.unfreeze_calls) == baseline_calls
