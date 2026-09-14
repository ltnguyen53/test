"""Training loop giai đoạn 2: TemporalAggregatorTrainable (phiên 3.2) học
phân loại hành động từ sequence embedding (data/processed/, phiên 5.1).

Khác phase 1 (phiên 4.3): KHÔNG cần proxy task/head tạm —
TemporalAggregatorTrainable đã có sẵn head thật (Linear -> num_classes)
trong forward(), train trực tiếp bằng cross-entropy trên nhãn VIDEO-LEVEL
(không phải per-frame như phase 1).

KHÔNG có curriculum unfreeze ở đây — TemporalAggregatorTrainable train từ
đầu (không phải backbone pretrained như SpatialBackbone), không có gì để
"đóng băng dần".

Dùng cfg.phase2.epochs / cfg.phase2.learning_rate riêng (KHÔNG dùng chung
với phase1) — 2 giai đoạn train độc lập, hyperparameter độc lập.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from video_action_mlops.config.schema import AppConfig
from video_action_mlops.models.registry import build_model
from video_action_mlops.models.temporal import TemporalAggregatorTrainable


def save_checkpoint(model: TemporalAggregatorTrainable, path: str | Path) -> None:
    """Lưu CHỈ state_dict — không kèm optimizer (cùng lý do save_checkpoint
    ở phase1_trainer.py, phiên 4.3: checkpoint này dùng để suy luận/export
    (tuần 7), không dùng để resume training)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def save_resume_state(
    model: TemporalAggregatorTrainable,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: list[dict],
    path: str | Path,
) -> None:
    """Lưu đủ trạng thái để resume training GIỮA CHỪNG (phiên 13.2) — cùng
    thiết kế với phase1_trainer.save_resume_state() (đọc docstring ở đó
    cho lý do đầy đủ): tách biệt save_checkpoint() ở trên, có cả optimizer
    state, ghi ATOMIC qua tmp file + os.replace(), bị xoá khi train xong
    trọn vẹn. Đơn giản hơn bản phase1 vì KHÔNG có cls_head tạm / lịch
    unfreeze — TemporalAggregatorTrainable train từ đầu, không có gì để
    "đóng băng dần" (xem docstring module)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    state = {
        "epoch": epoch,
        "history": history,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    try:
        torch.save(state, tmp_path)
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def load_resume_state(path: str | Path) -> dict | None:
    """Đọc resume state đã lưu bởi save_resume_state(), hoặc None nếu chưa
    có file (lần chạy đầu tiên)."""
    path = Path(path)
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu")


@torch.no_grad()
def _evaluate(
    model: TemporalAggregatorTrainable, loader: DataLoader, criterion: nn.Module, device: str
) -> dict:
    model.eval()
    running_loss, running_correct, running_total = 0.0, 0, 0
    for seqs, labels in loader:
        seqs, labels = seqs.to(device), labels.to(device)
        logits = model(seqs)
        loss = criterion(logits, labels)
        running_loss += loss.item() * seqs.size(0)
        running_correct += (logits.argmax(dim=1) == labels).sum().item()
        running_total += seqs.size(0)
    return {"val_loss": running_loss / running_total, "val_acc": running_correct / running_total}


def train_phase2(
    cfg: AppConfig,
    train_dataset: Dataset,
    val_dataset: Dataset | None = None,
    device: str = "cpu",
    checkpoint_path: str | Path | None = None,
    batch_size: int = 32,
    resume_state_path: str | Path | None = None,
) -> dict:
    """Trả về {"history": [...], "final": {...}}.

    Nhận train_dataset/val_dataset làm THAM SỐ — cùng lý do tách logic
    khỏi data như train_phase1() (phiên 4.3): test được bằng dataset giả
    nhỏ (vd sequence embedding random), không cần data/processed/ thật.

    `resume_state_path` (phiên 13.2): xem docstring
    phase1_trainer.train_phase1() cho thiết kế đầy đủ — cùng cơ chế, đơn
    giản hơn vì không có pretrained/cls_head/lịch unfreeze cần khôi phục.
    """
    resume_state = load_resume_state(resume_state_path) if resume_state_path is not None else None

    model: TemporalAggregatorTrainable = build_model(
        cfg, stage="phase2", variant="trainable"
    )  # type: ignore[assignment]
    model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = (
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.phase2.learning_rate)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    history: list[dict] = []

    if resume_state is not None:
        model.load_state_dict(resume_state["model_state"])
        optimizer.load_state_dict(resume_state["optimizer_state"])
        start_epoch = resume_state["epoch"]
        history = list(resume_state["history"])
        print(
            f"[train_phase2] resume tu epoch {start_epoch}/{cfg.phase2.epochs} "
            f"(doc resume state tu {resume_state_path})"
        )

    for epoch in range(start_epoch, cfg.phase2.epochs):
        model.train()
        running_loss, running_correct, running_total = 0.0, 0, 0
        for seqs, labels in train_loader:
            seqs, labels = seqs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(seqs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * seqs.size(0)
            running_correct += (logits.argmax(dim=1) == labels).sum().item()
            running_total += seqs.size(0)

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": running_loss / running_total,
            "train_acc": running_correct / running_total,
        }
        if val_loader is not None:
            epoch_metrics.update(_evaluate(model, val_loader, criterion, device))
        history.append(epoch_metrics)

        if resume_state_path is not None:
            save_resume_state(model, optimizer, epoch + 1, history, resume_state_path)

    if resume_state_path is not None:
        Path(resume_state_path).unlink(missing_ok=True)

    if checkpoint_path is not None:
        save_checkpoint(model, checkpoint_path)

    return {"history": history, "final": history[-1] if history else {}}
