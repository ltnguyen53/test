"""kernels/loader.py — JIT compile CUDA kernel 1 lần, cache kết quả, fallback
êm về PyTorch thuần nếu compile lỗi (mục 3.4 roadmap).

3 RÀNG BUỘC BẮT BUỘC (mục 3.4), cả 3 đều được ENFORCE bằng code chứ không
chỉ ghi trong docstring:

1. Compile lỗi (thiếu nvcc, sai version CUDA/PyTorch, không có GPU) -> log
   warning, KHÔNG crash, rơi về `F.layer_norm`. Xem `_try_compile()`: bắt
   `Exception` rộng có chủ đích.
2. Kernel này KHÔNG có custom backward (`torch.autograd.Function`) — chỉ
   an toàn khi `torch.is_grad_enabled()` là False (inference /
   `torch.no_grad()`). Nếu đang cần gradient, `fused_layer_norm()` TỰ ĐỘNG
   fallback về `F.layer_norm` dù compile thành công — không tin caller tự
   nhớ gọi đúng lúc, vì dùng nhầm sẽ cho gradient sai một cách IM LẶNG
   (đúng cảnh báo của roadmap).
3. Serving image (`docker/serve.Dockerfile`, tuần 8) KHÔNG cài toolchain
   CUDA build — kernel này chỉ có ý nghĩa lúc training (phase2, phiên
   5.2), KHÔNG được gọi trong đường suy luận sau khi export ONNX (tuần 7,
   graph ONNX đã "đóng băng", không gọi lại được Python/CUDA kernel này).
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_CSRC_DIR = Path(__file__).parent / "csrc"

# Cache ở cấp MODULE (không phải per-call): None = chưa thử compile,
# False = đã thử và lỗi (hoặc không có CUDA), object = compile thành công.
_EXTENSION: object | bool | None = None
_COMPILE_ATTEMPTED = False


def _try_compile() -> None:
    """Chỉ thử compile ĐÚNG 1 LẦN cho cả process — JIT compile tốn vài
    giây tới vài phút, tuyệt đối không thử lại mỗi lần gọi fused_layer_norm()."""
    global _EXTENSION, _COMPILE_ATTEMPTED
    if _COMPILE_ATTEMPTED:
        return
    _COMPILE_ATTEMPTED = True

    if not torch.cuda.is_available():
        logger.warning(
            "kernels.loader: không có CUDA device — bỏ qua compile, dùng "
            "F.layer_norm thuần PyTorch."
        )
        _EXTENSION = False
        return

    try:
        from torch.utils.cpp_extension import load

        _EXTENSION = load(
            name="fused_layer_norm_ext",
            sources=[
                str(_CSRC_DIR / "fused_layer_norm_binding.cpp"),
                str(_CSRC_DIR / "fused_layer_norm.cu"),
            ],
            verbose=False,
        )
        logger.info("kernels.loader: compile fused_layer_norm CUDA kernel thành công.")
    except Exception as exc:  # noqa: BLE001 — CỐ Ý bắt rộng: bất kỳ lỗi gì
        # (thiếu nvcc, sai version, lỗi cú pháp .cu, thiếu quyền ghi cache
        # dir...) đều phải rơi về fallback, không được làm crash cả
        # pipeline chỉ vì 1 optimization TUỲ CHỌN thất bại — đúng triết lý
        # mục 3.4: "luôn là optimization, không bao giờ dependency cứng".
        logger.warning(
            f"kernels.loader: compile CUDA kernel thất bại ({exc}) — dùng "
            f"F.layer_norm thuần PyTorch (chậm hơn nhưng luôn đúng)."
        )
        _EXTENSION = False


def fused_layer_norm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """Drop-in thay thế `F.layer_norm(x, [x.shape[-1]], weight, bias, eps)`
    khi có thể — tự động fallback trong 3 trường hợp: chưa compile được,
    input không phải CUDA float32 tensor, hoặc đang cần gradient (ràng
    buộc #2 ở docstring module)."""
    _try_compile()

    can_use_kernel = (
        _EXTENSION is not False
        and _EXTENSION is not None
        and x.is_cuda
        and x.dtype == torch.float32
        and not torch.is_grad_enabled()
    )
    if can_use_kernel:
        return _EXTENSION.fused_layer_norm_cuda(x, weight, bias, eps)  # type: ignore[attr-defined]

    return F.layer_norm(x, [x.shape[-1]], weight, bias, eps)


def is_kernel_active() -> bool:
    """Tiện cho logging/benchmark (phiên 6.2): kernel CUDA có thật sự sẵn
    sàng hay đang fallback hoàn toàn."""
    _try_compile()
    return _EXTENSION is not False and _EXTENSION is not None
