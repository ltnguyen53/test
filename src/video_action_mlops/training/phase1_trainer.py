"""Training loop giai đoạn 1: fine-tune SpatialBackbone bằng proxy task
phân loại per-frame, áp dụng curriculum unfreeze (mở dần backbone) theo
epoch (mục 1.2 roadmap). Output: models/phase1_checkpoint.pt (CHỈ
SpatialBackbone.state_dict()) + reports/phase1_metrics.json.

Vì sao có "proxy task phân loại": SpatialBackbone.forward() trả về
embedding (embed_dim_out chiều), không phải logits — không tính
cross-entropy trực tiếp được. _Phase1TrainingHead là head TẠM THỜI chỉ
tồn tại trong lúc train, ánh xạ embedding -> logits để có gradient fine-
tune backbone. Head này KHÔNG được lưu vào checkpoint, KHÔNG dùng ở phase 2
(extract_embeddings.py, tuần 5, chỉ gọi SpatialBackbone.forward()).

Mốc unfreeze theo epoch: chia cfg.phase1.epochs thành 4 phần bằng nhau,
mở dần layer4 -> layer3 -> layer2 -> layer1 (KHÔNG tự mở "stem" — mở stem
chỉ nên làm thủ công khi thật sự cần, tránh phá pretrained feature tổng
quát nhất một cách vô ý).
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from video_action_mlops.config.schema import AppConfig
from video_action_mlops.models.registry import build_model
from video_action_mlops.models.spatial import SpatialBackbone

_CURRICULUM_SCHEDULE: tuple[str, ...] = ("layer4", "layer3", "layer2", "layer1")


class _Phase1TrainingHead(nn.Module):
    """Head phân loại TẠM THỜI, chỉ dùng để tính loss lúc train phase 1."""

    def __init__(self, embed_dim_out: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(embed_dim_out, num_classes)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.fc(embedding)


def _unfreeze_stage_for_epoch(epoch: int, total_epochs: int) -> str | None:
    """Trả về stage cuối cùng cần mở (dùng với unfreeze_up_to), hoặc None
    nếu vẫn ở epoch 0 (đóng băng hết, chỉ train head — warm-up).

    Hàm THUẦN (không phụ thuộc torch/model) — cố ý tách riêng để test được
    logic lịch curriculum mà không cần dựng model thật.
    """
    if total_epochs <= 0:
        raise ValueError(f"total_epochs phải > 0, nhận {total_epochs}")
    if epoch < 0:
        raise ValueError(f"epoch phải >= 0, nhận {epoch}")
    if epoch == 0:
        return None
    stage_length = total_epochs / len(_CURRICULUM_SCHEDULE)
    stage_index = min(int(epoch / stage_length), len(_CURRICULUM_SCHEDULE) - 1)
    return _CURRICULUM_SCHEDULE[stage_index]


def save_checkpoint(model: SpatialBackbone, path: str | Path) -> None:
    """Lưu CHỈ SpatialBackbone.state_dict() — không kèm cls_head tạm, không
    kèm optimizer state (checkpoint dùng để EXTRACT EMBEDDING ở tuần 5,
    không dùng để resume training giai đoạn 1)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def save_resume_state(
    model: SpatialBackbone,
    cls_head: _Phase1TrainingHead,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    current_stage: str | None,
    history: list[dict],
    path: str | Path,
) -> None:
    """Lưu ĐỦ trạng thái để resume training GIỮA CHỪNG sau khi bị ngắt
    (phiên 13.2, gap ghi ở README "Checkpoint không lưu optimizer state").

    KHÁC save_checkpoint() ở trên: file này có cả cls_head tạm +
    optimizer state + epoch đã hoàn thành + lịch sử unfreeze — đủ để tiếp
    tục ĐÚNG chỗ đã dừng (kể cả momentum của optimizer, không chỉ trọng
    số). CHỈ tồn tại trong lúc train — bị xoá khi train xong trọn vẹn
    (xem cuối train_phase1()), không nên dùng file này để suy luận.

    `epoch` lưu là EPOCH TIẾP THEO cần chạy (đã hoàn thành xong `epoch`
    epoch, 0-indexed) — để load_resume_state() dùng thẳng làm start_epoch
    của `range()`, không cần +1 ở chỗ gọi.

    Ghi ATOMIC (tmp file cùng thư mục + os.replace): nếu process bị kill
    (mất điện, Colab ngắt kết nối, hết quota GPU) ĐÚNG LÚC đang ghi, file
    resume CŨ (lần lưu trước, còn nguyên vẹn) không bị hỏng — os.replace
    là 1 syscall duy nhất trên cùng filesystem, không có trạng thái ghi dở
    dang nào lộ ra ngoài. Nếu ghi tmp file thất bại giữa chừng, dọn tmp
    file rồi raise lại (không nuốt lỗi — fail-fast, cùng tinh thần
    ConfigError ở config/loader.py).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    state = {
        "epoch": epoch,
        "current_stage": current_stage,
        "history": history,
        "model_state": model.state_dict(),
        "cls_head_state": cls_head.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    try:
        torch.save(state, tmp_path)
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def load_resume_state(path: str | Path) -> dict | None:
    """Đọc resume state đã lưu bởi save_resume_state(), hoặc trả về None
    nếu file chưa tồn tại (lần chạy ĐẦU TIÊN, không phải resume — hành vi
    mặc định phải giữ nguyên như trước phiên 13.2 khi không có gì để
    resume)."""
    path = Path(path)
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu")


@torch.no_grad()
def _evaluate(
    model: SpatialBackbone,
    cls_head: _Phase1TrainingHead,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> dict:
    model.eval()
    cls_head.eval()
    running_loss, running_correct, running_total = 0.0, 0, 0
    for frames, labels in loader:
        frames, labels = frames.to(device), labels.to(device)
        logits = cls_head(model(frames))
        loss = criterion(logits, labels)
        running_loss += loss.item() * frames.size(0)
        running_correct += (logits.argmax(dim=1) == labels).sum().item()
        running_total += frames.size(0)
    return {"val_loss": running_loss / running_total, "val_acc": running_correct / running_total}


def train_phase1(
    cfg: AppConfig,
    train_dataset: Dataset,
    val_dataset: Dataset | None = None,
    device: str = "cpu",
    checkpoint_path: str | Path | None = None,
    batch_size: int = 32,
    resume_state_path: str | Path | None = None,
) -> dict:
    """Chạy training loop đầy đủ, trả về {"history": [...], "final": {...}}.

    Nhận `train_dataset`/`val_dataset` làm THAM SỐ thay vì tự dựng bên
    trong — tách logic training khỏi cách data được load, để test được
    bằng dataset giả nhỏ (vd TensorDataset), không cần data/interim/ thật.

    `resume_state_path` (phiên 13.2, mặc định None = hành vi CŨ, không
    đổi): nếu truyền vào và file đã tồn tại (từ 1 lần chạy trước bị ngắt
    giữa chừng), tự động load lại model/cls_head/optimizer/epoch/lịch
    unfreeze/history rồi tiếp tục từ epoch còn thiếu — KHÔNG train lại từ
    đầu. Nếu file resume TỒN TẠI, `pretrained=False` khi build model
    (bỏ qua tải ImageNet weight — sắp bị ghi đè bởi state đã lưu ngay sau
    đó, tải về chỉ tốn thời gian/băng thông vô ích). File resume bị XOÁ
    khi train xong TRỌN VẸN (mọi epoch cfg.phase1.epochs đã chạy) — không
    để lại state cũ có thể vô tình dùng nhầm cho lần train tiếp theo.
    """
    resume_state = load_resume_state(resume_state_path) if resume_state_path is not None else None

    model: SpatialBackbone = build_model(
        cfg, stage="phase1", pretrained=resume_state is None
    )  # type: ignore[assignment]
    model.to(device)
    cls_head = _Phase1TrainingHead(cfg.phase1.embed_dim_out, cfg.data.num_classes).to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = (
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None
    )

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(cls_head.parameters()), lr=cfg.phase1.learning_rate
    )
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    current_stage: str | None = None
    history: list[dict] = []

    if resume_state is not None:
        model.load_state_dict(resume_state["model_state"])
        cls_head.load_state_dict(resume_state["cls_head_state"])
        optimizer.load_state_dict(resume_state["optimizer_state"])
        start_epoch = resume_state["epoch"]
        current_stage = resume_state["current_stage"]
        history = list(resume_state["history"])
        print(
            f"[train_phase1] resume tu epoch {start_epoch}/{cfg.phase1.epochs} "
            f"(doc resume state tu {resume_state_path})"
        )

    for epoch in range(start_epoch, cfg.phase1.epochs):
        target_stage = _unfreeze_stage_for_epoch(epoch, cfg.phase1.epochs)
        if target_stage is not None and target_stage != current_stage:
            model.unfreeze_up_to(target_stage)
            current_stage = target_stage

        model.train()
        cls_head.train()
        running_loss, running_correct, running_total = 0.0, 0, 0
        for frames, labels in train_loader:
            frames, labels = frames.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = cls_head(model(frames))
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * frames.size(0)
            running_correct += (logits.argmax(dim=1) == labels).sum().item()
            running_total += frames.size(0)

        epoch_metrics = {
            "epoch": epoch,
            "unfrozen_up_to": current_stage,
            "train_loss": running_loss / running_total,
            "train_acc": running_correct / running_total,
        }
        if val_loader is not None:
            epoch_metrics.update(_evaluate(model, cls_head, val_loader, criterion, device))

        history.append(epoch_metrics)

        if resume_state_path is not None:
            save_resume_state(
                model, cls_head, optimizer, epoch + 1, current_stage, history, resume_state_path
            )

    if resume_state_path is not None:
        Path(resume_state_path).unlink(missing_ok=True)

    if checkpoint_path is not None:
        save_checkpoint(model, checkpoint_path)

    return {"history": history, "final": history[-1] if history else {}}
