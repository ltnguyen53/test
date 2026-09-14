"""tests/unit/test_spatial.py — test models/spatial.py (phiên 3.1, viết
muộn: file này bị BỎ SÓT hoàn toàn cho tới khi người dùng hỏi thẳng "13.6
và 3.1 đâu?" — checklist gốc ghi rõ "viết test_spatial.py tự tay" nhưng
chưa từng được tạo. Verify THẬT bằng torch/torchvision cài thật trong
sandbox (điều mà checklist gốc ghi "tôi không tự chạy được, sandbox thiếu
torch" — nay đã khác, từ phiên 13.2).

pretrained=False (mặc định) — không tải ImageNet weight, chạy nhanh/offline,
khớp đúng lý do đã ghi trong docstring SpatialBackbone.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from video_action_mlops.models.spatial import SpatialBackbone

# ---------- __init__ ----------


def test_init_rejects_non_positive_embed_dim():
    with pytest.raises(ValueError, match="embed_dim_out"):
        SpatialBackbone(embed_dim_out=0)
    with pytest.raises(ValueError, match="embed_dim_out"):
        SpatialBackbone(embed_dim_out=-8)


def test_default_pretrained_is_false_and_builds_offline():
    """Không truyền `pretrained` -> phải dùng weights=None (không tải
    mạng) -- nếu default vô tình là True, test này sẽ treo/lỗi mạng thay
    vì chạy nhanh như mong đợi."""
    model = SpatialBackbone(embed_dim_out=16)
    assert isinstance(model, SpatialBackbone)


def test_embed_dim_out_reflected_in_head_output_features():
    model = SpatialBackbone(embed_dim_out=37)
    assert model.head.out_features == 37


def test_unfreeze_stages_order_is_output_to_input():
    """Thứ tự PHẢI từ layer4 (gần output) về stem (gần input) -- đảo
    ngược thứ tự này sẽ phá đúng lý do curriculum ghi trong docstring
    module (mở dần từ đặc trưng cụ thể tới đặc trưng tổng quát)."""
    assert SpatialBackbone._UNFREEZE_STAGES == ("layer4", "layer3", "layer2", "layer1", "stem")


# ---------- forward ----------


def test_forward_output_shape():
    model = SpatialBackbone(embed_dim_out=16)
    x = torch.randn(3, 3, 64, 64)  # (B, C, H, W)
    out = model(x)
    assert out.shape == (3, 16)


def test_forward_rejects_missing_batch_dimension():
    """Input thiếu chiều batch (chỉ (C, H, W), 3 chiều thay vì 4) phải bị
    chặn NGAY với thông báo rõ ràng, không lỗi mù mờ sâu bên trong
    resnet18 (vd lỗi shape mismatch ở 1 conv layer nào đó)."""
    model = SpatialBackbone(embed_dim_out=16)
    x = torch.randn(3, 64, 64)  # thiếu chieu batch
    with pytest.raises(ValueError, match="B, C, H, W"):
        model(x)


def test_forward_batch_size_one_works():
    model = SpatialBackbone(embed_dim_out=8)
    out = model(torch.randn(1, 3, 64, 64))
    assert out.shape == (1, 8)


# ---------- freeze_backbone() gọi trong __init__ ----------


def test_all_stages_frozen_by_default_head_stays_trainable():
    model = SpatialBackbone(embed_dim_out=16)
    summary = model.trainable_stage_summary()
    assert all(trainable is False for trainable in summary.values())
    assert all(p.requires_grad is True for p in model.head.parameters())


# ---------- unfreeze_up_to ----------


def test_unfreeze_up_to_layer4_opens_only_one_stage():
    model = SpatialBackbone(embed_dim_out=16)
    model.unfreeze_up_to("layer4")
    summary = model.trainable_stage_summary()
    assert summary == {
        "layer4": True,
        "layer3": False,
        "layer2": False,
        "layer1": False,
        "stem": False,
    }


def test_unfreeze_up_to_layer2_opens_three_stages_output_side():
    model = SpatialBackbone(embed_dim_out=16)
    model.unfreeze_up_to("layer2")
    summary = model.trainable_stage_summary()
    assert summary == {
        "layer4": True,
        "layer3": True,
        "layer2": True,
        "layer1": False,
        "stem": False,
    }


def test_unfreeze_up_to_stem_opens_everything():
    model = SpatialBackbone(embed_dim_out=16)
    model.unfreeze_up_to("stem")
    summary = model.trainable_stage_summary()
    assert all(trainable is True for trainable in summary.values())


def test_unfreeze_up_to_invalid_stage_raises():
    model = SpatialBackbone(embed_dim_out=16)
    with pytest.raises(ValueError, match="không hợp lệ"):
        model.unfreeze_up_to("layer5")
    with pytest.raises(ValueError, match="không hợp lệ"):
        model.unfreeze_up_to("head")  # head KHONG nam trong _UNFREEZE_STAGES


def test_unfreeze_up_to_is_idempotent_and_does_not_reset_already_open_stage():
    model = SpatialBackbone(embed_dim_out=16)
    model.unfreeze_up_to("layer3")
    model.unfreeze_up_to("layer4")  # cutoff "nho hon" -- KHONG duoc dong lai layer3 da mo
    summary = model.trainable_stage_summary()
    assert summary["layer4"] is True
    assert summary["layer3"] is True  # van con mo, khong bi unfreeze_up_to sau "dong lai nham"


def test_unfreeze_all_opens_every_stage():
    model = SpatialBackbone(embed_dim_out=16)
    model.unfreeze_all()
    assert all(model.trainable_stage_summary().values())


def test_freeze_backbone_can_be_called_again_to_close_everything():
    model = SpatialBackbone(embed_dim_out=16)
    model.unfreeze_all()
    model.freeze_backbone()
    assert not any(model.trainable_stage_summary().values())
    assert all(p.requires_grad is True for p in model.head.parameters())  # head khong bi dung toi


# ---------- head không nằm trong _UNFREEZE_STAGES ----------


def test_head_requires_grad_unaffected_by_freeze_or_unfreeze():
    model = SpatialBackbone(embed_dim_out=16)
    states = []
    for p in model.head.parameters():
        states.append(p.requires_grad)
    assert all(states)  # ngay tu dau, truoc ca khi goi freeze/unfreeze tay

    model.freeze_backbone()
    assert all(p.requires_grad for p in model.head.parameters())
    model.unfreeze_up_to("layer2")
    assert all(p.requires_grad for p in model.head.parameters())


# ---------- trainable_stage_summary với stage KHÔNG có parameter ----------


def test_trainable_stage_summary_raises_stop_iteration_on_empty_stage(monkeypatch):
    """Câu hỏi kiểm tra gốc: `trainable_stage_summary` với stage rỗng
    param thì sao? Trả lời bằng verify THẬT thay vì suy đoán:
    `next(...parameters())` không có giá trị mặc định -- 1 stage rỗng
    tham số (nn.Sequential() rỗng, về lý thuyết không xảy ra với resnet18
    thật nhưng KHÔNG có gì trong code chặn việc này) sẽ làm
    trainable_stage_summary() raise StopIteration thay vì trả về dict
    thiếu key đó hay giá trị mặc định nào đó -- đây là hành vi HIỆN TẠI,
    không phải hành vi "đúng" hay "sai", chỉ là điều cần biết trước khi
    dựa vào hàm này ở nơi khác."""
    model = SpatialBackbone(embed_dim_out=16)
    monkeypatch.setattr(model, "layer4", nn.Sequential())  # rong, khong param nao

    with pytest.raises(StopIteration):
        model.trainable_stage_summary()


# ---------- verify THẬT bằng backward pass — điều checklist gốc ghi "tôi
# không tự chạy được, sandbox thiếu torch" — nay verify được thật ----------


def test_frozen_stages_receive_no_gradient_after_backward():
    """Bài test QUAN TRỌNG NHẤT của file này: build model, unfreeze 1
    phần, chạy forward+backward THẬT, rồi kiểm tra .grad -- không chỉ tin
    `requires_grad` (cấu hình) mà xác nhận GRADIENT THẬT SỰ có chảy đúng
    hay không sau 1 backward pass thật."""
    model = SpatialBackbone(embed_dim_out=8)
    model.unfreeze_up_to("layer3")  # layer4, layer3 trainable; layer2/1/stem dong

    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    loss = out.sum()
    loss.backward()

    for p in model.layer4.parameters():
        assert p.grad is not None
    for p in model.layer3.parameters():
        assert p.grad is not None
    for p in model.head.parameters():
        assert p.grad is not None

    for p in model.layer2.parameters():
        assert p.grad is None
    for p in model.layer1.parameters():
        assert p.grad is None
    for p in model.stem.parameters():
        assert p.grad is None


def test_fully_frozen_backbone_only_head_receives_gradient():
    model = SpatialBackbone(embed_dim_out=8)  # mac dinh: dong bang het

    x = torch.randn(2, 3, 64, 64)
    loss = model(x).sum()
    loss.backward()

    assert all(p.grad is not None for p in model.head.parameters())
    for stage_name in model._UNFREEZE_STAGES:
        stage = getattr(model, stage_name)
        assert all(p.grad is None for p in stage.parameters())


def test_forward_output_differs_for_different_inputs_real_computation():
    """Kiểm tra tối thiểu rằng forward() thực sự tính toán dựa trên input
    (không phải hằng số/hardcode) -- 2 input khác nhau phải cho output
    khác nhau."""
    torch.manual_seed(0)
    model = SpatialBackbone(embed_dim_out=8)
    model.eval()
    with torch.no_grad():
        out_a = model(torch.zeros(1, 3, 64, 64))
        out_b = model(torch.ones(1, 3, 64, 64))
    assert not torch.allclose(out_a, out_b)
