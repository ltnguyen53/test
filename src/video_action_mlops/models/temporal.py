"""Temporal aggregator (giai đoạn 2): chuỗi embedding per-frame -> logits.

Roadmap mục 3.3: kiến trúc export-friendly phải là 1 CLASS RIÊNG, không phải
flag runtime. "Toán tử export kém" ở đây chính là nn.MultiheadAttention mà
roadmap liệt kê thẳng — module này dùng đường tính fused nội bộ của
PyTorch (packed in_proj, nhánh xử lý is_batched/is_causal phụ thuộc dữ
liệu), trace qua torch.onnx.export không ổn định qua mọi phiên bản/backend
(đặc biệt TensorRT thường không tiêu thụ được op MultiheadAttention của
ONNX trực tiếp).

- TemporalAggregatorTrainable: dùng nn.MultiheadAttention — ít code, tối ưu
  tốc độ, dùng để TRAIN (training/phase2_trainer.py, tuần 5).
- TemporalAggregatorExport: TƯƠNG ĐƯƠNG TOÁN HỌC, tự viết attention bằng
  matmul/softmax tường minh (không có op nào PyTorch "đóng gói" bên trong)
  — dùng để EXPORT (export/to_onnx.py, tuần 7), KHÔNG dùng để train.

convert_trainable_to_export(): lossless — tách in_proj_weight/in_proj_bias
(QKV packed của nn.MultiheadAttention, shape (3*D, D) và (3*D,)) thành
q_proj/k_proj/v_proj riêng theo đúng thứ tự [Q, K, V] mà PyTorch pack bên
trong. Test đi kèm PHẢI xác nhận output 2 class khớp atol=1e-4 trên CÙNG
input, cùng eval() mode (tắt dropout) — đây là bước quan trọng nhất cả
project: thiếu nó là nguồn lỗi silent phổ biến nhất khi đưa model sequence
ra production (roadmap mục 3.3, đoạn cuối).
"""

from __future__ import annotations

import torch
from torch import nn


class TemporalAggregatorTrainable(nn.Module):
    """Self-attention (nn.MultiheadAttention) + residual + mean-pool + linear head. Dùng để TRAIN."""

    def __init__(
        self, input_dim: int, num_heads: int, num_classes: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        if input_dim % num_heads != 0:
            raise ValueError(
                f"input_dim ({input_dim}) phải chia hết cho num_heads ({num_heads})"
            )
        self.input_dim = input_dim
        self.num_heads = num_heads

        self.attn = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(input_dim)
        self.head = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, input_dim) — chuỗi embedding từ phase1. return: (B, num_classes)."""
        if x.ndim != 3 or x.shape[-1] != self.input_dim:
            raise ValueError(
                f"forward nhận (B, T, {self.input_dim}), nhận shape {tuple(x.shape)}"
            )
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        h = self.norm(x + attn_out)  # residual, giống 1 khối Transformer encoder chuẩn
        pooled = h.mean(dim=1)  # (B, input_dim) — mean-pool theo chiều thời gian
        return self.head(pooled)


class TemporalAggregatorExport(nn.Module):
    """Tương đương toán học với TemporalAggregatorTrainable, tự viết attention
    bằng matmul/softmax tường minh (KHÔNG dùng nn.MultiheadAttention) để
    trace ổn định khi export. Dùng để EXPORT, KHÔNG dùng để train (không
    tối ưu tốc độ bằng bản trainable — thiếu fused kernel)."""

    def __init__(self, input_dim: int, num_heads: int, num_classes: int) -> None:
        super().__init__()
        if input_dim % num_heads != 0:
            raise ValueError(
                f"input_dim ({input_dim}) phải chia hết cho num_heads ({num_heads})"
            )
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        self.q_proj = nn.Linear(input_dim, input_dim)
        self.k_proj = nn.Linear(input_dim, input_dim)
        self.v_proj = nn.Linear(input_dim, input_dim)
        self.out_proj = nn.Linear(input_dim, input_dim)

        self.norm = nn.LayerNorm(input_dim)
        self.head = nn.Linear(input_dim, num_classes)

    def _split_heads(self, x: torch.Tensor, batch: int, seq_len: int) -> torch.Tensor:
        # (B, T, input_dim) -> (B, num_heads, T, head_dim)
        return x.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.input_dim:
            raise ValueError(
                f"forward nhận (B, T, {self.input_dim}), nhận shape {tuple(x.shape)}"
            )
        b, t, _ = x.shape

        q = self._split_heads(self.q_proj(x), b, t)  # (B, H, T, D_h)
        k = self._split_heads(self.k_proj(x), b, t)
        v = self._split_heads(self.v_proj(x), b, t)

        scale = self.head_dim**-0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, T, T)
        weights = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(weights, v)  # (B, H, T, D_h)

        attn_out = attn_out.transpose(1, 2).contiguous().view(b, t, self.input_dim)
        attn_out = self.out_proj(attn_out)

        h = self.norm(x + attn_out)
        pooled = h.mean(dim=1)
        return self.head(pooled)


@torch.no_grad()
def convert_trainable_to_export(trainable: TemporalAggregatorTrainable) -> TemporalAggregatorExport:
    """Chuyển LOSSLESS trainable -> export. Không train lại, không xấp xỉ.

    nn.MultiheadAttention pack Q,K,V thành 1 ma trận in_proj_weight (3*D, D)
    theo đúng thứ tự [Q; K; V] (xem torch/nn/modules/activation.py nội bộ
    PyTorch) — tách theo đúng thứ tự đó là điều kiện bắt buộc để lossless.
    """
    export = TemporalAggregatorExport(
        input_dim=trainable.input_dim,
        num_heads=trainable.num_heads,
        num_classes=trainable.head.out_features,
    )

    d = trainable.input_dim
    in_proj_weight = trainable.attn.in_proj_weight  # (3*d, d), thứ tự [Q; K; V]
    in_proj_bias = trainable.attn.in_proj_bias  # (3*d,)

    export.q_proj.weight.copy_(in_proj_weight[0:d])
    export.k_proj.weight.copy_(in_proj_weight[d : 2 * d])
    export.v_proj.weight.copy_(in_proj_weight[2 * d : 3 * d])
    export.q_proj.bias.copy_(in_proj_bias[0:d])
    export.k_proj.bias.copy_(in_proj_bias[d : 2 * d])
    export.v_proj.bias.copy_(in_proj_bias[2 * d : 3 * d])

    export.out_proj.weight.copy_(trainable.attn.out_proj.weight)
    export.out_proj.bias.copy_(trainable.attn.out_proj.bias)

    export.norm.load_state_dict(trainable.norm.state_dict())
    export.head.load_state_dict(trainable.head.state_dict())

    return export
