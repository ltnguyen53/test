"""So sánh fused_layer_norm() (kernels/loader.py, phiên 6.1) với
F.layer_norm chuẩn của PyTorch — cả về ĐỘ CHÍNH XÁC (parity) lẫn TỐC ĐỘ.

Chạy: python -m video_action_mlops.kernels.benchmarks.benchmark_layer_norm

QUAN TRỌNG: chỉ có ý nghĩa trên máy CÓ GPU + nvcc. Trên máy không có CUDA,
script in ra 1 dòng giải thích rồi THOÁT SỚM (không phải lỗi — đúng triết
lý mục 3.4: thiếu GPU không phải điều kiện lỗi, chỉ là không có gì để đo).

NGUYÊN TẮC: LUÔN kiểm tra ĐÚNG NGHĨA (parity) trước khi đo TỐC ĐỘ. Benchmark
tốc độ của 1 kernel tính sai là vô nghĩa — tệ hơn là không benchmark gì,
vì nó tạo cảm giác an toàn giả.
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from video_action_mlops.kernels.loader import fused_layer_norm, is_kernel_active


def _check_parity(batch: int, seq_len: int, dim: int, device: str) -> float:
    """Chạy CẢ 2 đường trên CÙNG input, trả về sai số tuyệt đối lớn nhất."""
    torch.manual_seed(0)
    x = torch.randn(batch, seq_len, dim, device=device)
    weight = torch.randn(dim, device=device)
    bias = torch.randn(dim, device=device)

    with torch.no_grad():  # bắt buộc: fused_layer_norm() tự fallback nếu grad_enabled (ràng buộc #2, phiên 6.1)
        out_kernel = fused_layer_norm(x, weight, bias)
        out_reference = F.layer_norm(x, [dim], weight, bias)

    return (out_kernel - out_reference).abs().max().item()


def _time_fn(fn, iters: int, device: str) -> float:
    """Thời gian trung bình mỗi lần gọi (ms). torch.cuda.synchronize() bắt
    buộc trước/sau — CUDA call không đồng bộ (async), thiếu synchronize sẽ
    đo thời gian QUEUE lệnh chứ không phải thời gian CHẠY thật, ra con số
    sai lệch (thường là thấy tốc độ nhanh giả tạo)."""
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return (elapsed / iters) * 1000.0


def run_benchmark(batch: int = 32, seq_len: int = 16, dim: int = 512, iters: int = 100) -> None:
    if not torch.cuda.is_available():
        print(
            "[benchmark] Không có CUDA device — benchmark chỉ có ý nghĩa trên "
            "GPU thật. Bỏ qua (không phải lỗi, xem mục 3.4 roadmap)."
        )
        return

    device = "cuda"
    print(f"[benchmark] shape=({batch},{seq_len},{dim}), iters={iters}, device={device}")

    max_diff = _check_parity(batch, seq_len, dim, device)
    print(f"[benchmark] Sai số tối đa kernel vs F.layer_norm: {max_diff:.2e}")
    if max_diff > 1e-4:
        print(
            "[benchmark] CẢNH BÁO: sai số vượt 1e-4 — DỪNG, không benchmark tốc "
            "độ của kernel có khả năng SAI. Quay lại sửa kernel (phiên 6.1) trước."
        )
        return

    if not is_kernel_active():
        print(
            "[benchmark] Kernel không compile được (xem log WARNING ở trên) — "
            "chỉ có F.layer_norm chuẩn, không có gì để so sánh tốc độ."
        )
        return

    torch.manual_seed(0)
    x = torch.randn(batch, seq_len, dim, device=device)
    weight = torch.randn(dim, device=device)
    bias = torch.randn(dim, device=device)

    with torch.no_grad():
        # Warm-up: lần gọi CUDA đầu tiên thường chậm hơn hẳn (context init,
        # cache JIT) — không tính vào thời gian đo, đúng chuẩn mọi
        # benchmark GPU (không riêng gì kernel tự viết).
        fused_layer_norm(x, weight, bias)
        F.layer_norm(x, [dim], weight, bias)

        time_kernel = _time_fn(lambda: fused_layer_norm(x, weight, bias), iters, device)
        time_reference = _time_fn(lambda: F.layer_norm(x, [dim], weight, bias), iters, device)

    speedup = time_reference / time_kernel if time_kernel > 0 else float("nan")
    print(f"[benchmark] fused_layer_norm (kernel):    {time_kernel:.4f} ms/call")
    print(f"[benchmark] F.layer_norm (PyTorch chuẩn): {time_reference:.4f} ms/call")
    print(f"[benchmark] speedup: {speedup:.2f}x")
    if speedup < 1.0:
        print(
            "[benchmark] LƯU Ý: speedup < 1x nghĩa là kernel tự viết CHẬM HƠN "
            "PyTorch chuẩn — hoàn toàn có thể xảy ra (kernel ở phiên 6.1 dùng "
            "shared-memory reduction cơ bản, chưa tối ưu warp-shuffle; PyTorch "
            "nội bộ dùng cuDNN đã tối ưu rất sâu). Trong trường hợp này, giữ "
            "F.layer_norm, không có lý do dùng kernel tự viết."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark fused_layer_norm vs F.layer_norm")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()
    run_benchmark(args.batch, args.seq_len, args.dim, args.iters)


if __name__ == "__main__":
    main()
