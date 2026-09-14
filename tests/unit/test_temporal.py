import pytest
import torch

from video_action_mlops.models.temporal import (
    TemporalAggregatorExport,
    TemporalAggregatorTrainable,
    convert_trainable_to_export,
)

# ---------- TemporalAggregatorTrainable ----------

def test_trainable_forward_shape():
    model = TemporalAggregatorTrainable(input_dim=8, num_heads=2, num_classes=5)
    x = torch.randn(3, 6, 8)  # (B, T, input_dim)
    out = model(x)
    assert out.shape == (3, 5)


def test_trainable_rejects_indivisible_heads():
    with pytest.raises(ValueError, match="chia hết"):
        TemporalAggregatorTrainable(input_dim=7, num_heads=2, num_classes=5)


def test_trainable_rejects_wrong_input_shape():
    model = TemporalAggregatorTrainable(input_dim=8, num_heads=2, num_classes=5)
    with pytest.raises(ValueError, match="forward nhận"):
        model(torch.randn(3, 6, 16))  # sai input_dim ở chiều cuối


# ---------- TemporalAggregatorExport ----------

def test_export_forward_shape():
    model = TemporalAggregatorExport(input_dim=8, num_heads=2, num_classes=5)
    x = torch.randn(3, 6, 8)
    out = model(x)
    assert out.shape == (3, 5)


def test_export_rejects_indivisible_heads():
    with pytest.raises(ValueError, match="chia hết"):
        TemporalAggregatorExport(input_dim=7, num_heads=2, num_classes=5)


def test_export_rejects_wrong_input_shape():
    """Cùng logic validate với TemporalAggregatorTrainable
    (test_trainable_rejects_wrong_input_shape ở trên) nhưng
    TemporalAggregatorExport có forward() RIÊNG (không kế thừa) — 2 khối
    code TRÙNG NHAU, phải test RIÊNG (phát hiện qua coverage report:
    thiếu đúng nhánh raise này, dòng validate ở Trainable đã covered từ
    trước nhưng ở Export thì chưa)."""
    model = TemporalAggregatorExport(input_dim=8, num_heads=2, num_classes=5)
    with pytest.raises(ValueError, match="forward nhận"):
        model(torch.randn(3, 6, 16))  # sai input_dim o chieu cuoi


# ---------- convert_trainable_to_export: PHẦN QUAN TRỌNG NHẤT (mục 3.3) ----------

def test_convert_is_lossless_within_tolerance():
    """Đây là test quan trọng nhất project tính đến giờ (roadmap mục 3.3):
    2 class kiến trúc khác nhau ở CÁCH VIẾT nhưng phải cho CÙNG SỐ trên
    cùng input. Sai ở đây là lỗi silent nguy hiểm nhất khi export production.
    """
    torch.manual_seed(0)
    trainable = TemporalAggregatorTrainable(input_dim=16, num_heads=4, num_classes=10)
    trainable.eval()  # bắt buộc: tắt dropout, so sánh phải ở cùng chế độ suy luận

    export = convert_trainable_to_export(trainable)
    export.eval()

    x = torch.randn(4, 12, 16)
    with torch.no_grad():
        out_trainable = trainable(x)
        out_export = export(x)

    assert out_trainable.shape == out_export.shape == (4, 10)
    max_diff = (out_trainable - out_export).abs().max().item()
    assert max_diff < 1e-4, f"Lệch {max_diff} — vượt ngưỡng atol=1e-4, convert KHÔNG lossless"


def test_convert_preserves_num_classes_and_dims():
    trainable = TemporalAggregatorTrainable(input_dim=12, num_heads=3, num_classes=7)
    export = convert_trainable_to_export(trainable)
    assert export.input_dim == 12
    assert export.num_heads == 3
    assert export.head.out_features == 7


def test_convert_is_deterministic_not_a_fresh_random_init():
    """Convert phải TRANSPLANT trọng số, không phải khởi tạo lại ngẫu nhiên
    — 2 lần convert từ CÙNG 1 trainable phải cho export có trọng số y hệt.
    """
    torch.manual_seed(1)
    trainable = TemporalAggregatorTrainable(input_dim=8, num_heads=2, num_classes=4)
    export_a = convert_trainable_to_export(trainable)
    export_b = convert_trainable_to_export(trainable)
    assert torch.equal(export_a.q_proj.weight, export_b.q_proj.weight)
    assert torch.equal(export_a.out_proj.weight, export_b.out_proj.weight)
