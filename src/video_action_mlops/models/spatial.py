"""Spatial backbone (giai đoạn 1): trích embedding per-frame + curriculum freeze/unfreeze.

Model này chạy trên TỪNG frame độc lập (không biết gì về chiều thời gian —
đó là việc của models/temporal.py, phiên sau). Output là 1 vector
embed_dim_out chiều mỗi frame. embed_dim_out chính là "hợp đồng" khớp với
Phase1Config.embed_dim_out / Phase2Config.input_dim ở config/schema.py
(phiên 1.2) — xem AppConfig.check_dim_contract.

Curriculum unfreeze (mục 1.2 roadmap — pipeline ghi "train_phase1: spatial
backbone — curriculum unfreeze"): backbone khởi tạo ĐÓNG BĂNG hoàn toàn,
chỉ head được train ở epoch đầu. Sau đó mở dần từ layer4 (gần output, đặc
trưng cụ thể cho bài toán) về layer1/stem (gần input, đặc trưng tổng quát
như cạnh/màu) — mở từ cuối lên đầu để giữ pretrained feature tổng quát ổn
định lâu nhất, tránh gradient lớn phá hỏng chúng ngay từ epoch 1.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class SpatialBackbone(nn.Module):
    """ResNet18 + linear head, có API freeze/unfreeze theo curriculum stage.

    pretrained=False mặc định: build nhanh, offline, deterministic — dùng
    cho test/CI (không cần tải weight ImageNet, không cần mạng). Script
    train thật (training/phase1_trainer.py, tuần 5) sẽ tự truyền
    pretrained=True.
    """

    # Thứ tự TỪ OUTPUT VỀ INPUT — curriculum unfreeze mở dần đúng thứ tự này.
    _UNFREEZE_STAGES: tuple[str, ...] = ("layer4", "layer3", "layer2", "layer1", "stem")

    def __init__(self, embed_dim_out: int, pretrained: bool = False) -> None:
        super().__init__()
        if embed_dim_out <= 0:
            raise ValueError(f"embed_dim_out phải > 0, nhận {embed_dim_out}")

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)

        # Gom conv1+bn1+relu+maxpool thành 1 khối "stem" để freeze/unfreeze
        # như 1 stage duy nhất, khớp với _UNFREEZE_STAGES.
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

        in_features = backbone.fc.in_features  # 512 với resnet18
        self.head = nn.Linear(in_features, embed_dim_out)

        self.freeze_backbone()  # mặc định đóng băng hết — curriculum tự unfreeze dần sau

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) — 1 frame, KHÔNG có chiều thời gian. return: (B, embed_dim_out)."""
        if x.ndim != 4:
            raise ValueError(
                f"SpatialBackbone.forward nhận (B, C, H, W), nhận shape {tuple(x.shape)}"
            )
        h = self.stem(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        h = self.avgpool(h)
        h = torch.flatten(h, 1)
        return self.head(h)

    def freeze_backbone(self) -> None:
        """Đóng băng toàn bộ backbone (stem + layer1..4). head luôn train được, không bị đụng tới."""
        for stage_name in self._UNFREEZE_STAGES:
            self._set_stage_trainable(stage_name, trainable=False)

    def unfreeze_up_to(self, stage: str) -> None:
        """Mở băng từ layer4 xuống đến (bao gồm) `stage`, đúng thứ tự curriculum.

        Vd unfreeze_up_to("layer3") => layer4, layer3 trainable; layer2/1/stem vẫn đóng.
        Raise ValueError nếu `stage` không hợp lệ — fail-fast, tránh gõ nhầm
        tên rồi curriculum âm thầm không unfreeze gì mà không ai biết.
        """
        if stage not in self._UNFREEZE_STAGES:
            raise ValueError(
                f"stage '{stage}' không hợp lệ, phải là 1 trong {self._UNFREEZE_STAGES}"
            )
        cutoff = self._UNFREEZE_STAGES.index(stage)
        for stage_name in self._UNFREEZE_STAGES[: cutoff + 1]:
            self._set_stage_trainable(stage_name, trainable=True)

    def unfreeze_all(self) -> None:
        for stage_name in self._UNFREEZE_STAGES:
            self._set_stage_trainable(stage_name, trainable=True)

    def _set_stage_trainable(self, stage_name: str, trainable: bool) -> None:
        module: nn.Module = getattr(self, stage_name)
        for param in module.parameters():
            param.requires_grad = trainable

    def trainable_stage_summary(self) -> dict[str, bool]:
        """Tiện cho logging curriculum mỗi epoch: stage nào đang train, stage nào đóng."""
        return {
            stage_name: next(getattr(self, stage_name).parameters()).requires_grad
            for stage_name in self._UNFREEZE_STAGES
        }
