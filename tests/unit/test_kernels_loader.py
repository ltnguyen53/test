"""tests/unit/test_kernels_loader.py — test kernels/loader.py (phiên 14.8).

Sandbox này KHÔNG có GPU thật (`torch.cuda.is_available() is False`) —
mọi test KHÔNG đánh dấu `@pytest.mark.gpu` chỉ verify được nhánh FALLBACK
(chưa từng thử verify được nhánh compile CUDA thật thành công). Test đánh
dấu `@pytest.mark.gpu` viết SẴN cho người chạy trên máy có GPU thật (vd
Colab T4, xem colab/bootstrap.ipynb) — bị loại bởi `-m "not gpu"` như
checklist yêu cầu, không chạy được ở đây (`torch.cuda.is_available()` giả
lập bằng monkeypatch KHÔNG đủ — `torch.utils.cpp_extension.load()` thật
sự cần nvcc + GPU thật để compile file .cu, không thể giả lập bằng Python
mock).

`_EXTENSION`/`_COMPILE_ATTEMPTED` là cache CẤP MODULE (chỉ thử compile 1
lần/process, xem docstring loader.py) — PHẢI reset trước MỖI test qua
fixture autouse dưới đây, nếu không test chạy sau sẽ ăn cache của test
chạy trước (thứ tự test khác nhau sẽ ra kết quả khác nhau — lỗi kinh điển
khi test đụng vào cache toàn cục).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from video_action_mlops.kernels import loader


@pytest.fixture(autouse=True)
def _reset_module_level_cache(monkeypatch):
    """Đảm bảo MỌI test trong file này bắt đầu từ trạng thái "chưa từng
    thử compile" — không phụ thuộc thứ tự test chạy trước."""
    monkeypatch.setattr(loader, "_EXTENSION", None)
    monkeypatch.setattr(loader, "_COMPILE_ATTEMPTED", False)
    yield


# ---------- _try_compile() trên máy không có CUDA ----------


def test_try_compile_without_cuda_sets_extension_false_and_warns(monkeypatch, caplog):
    monkeypatch.setattr(loader.torch.cuda, "is_available", lambda: False)

    with caplog.at_level("WARNING"):
        loader._try_compile()

    assert loader._EXTENSION is False
    assert loader._COMPILE_ATTEMPTED is True
    assert "không có cuda" in caplog.text.lower() or "CUDA" in caplog.text


def test_try_compile_only_attempts_once_per_process(monkeypatch):
    """JIT compile tốn vài giây tới vài phút (docstring loader.py) --
    _try_compile() phải short-circuit ngay từ lần gọi thứ 2 trở đi, KHÔNG
    được gọi lại torch.cuda.is_available() (hay nặng hơn: thử compile lại)
    mỗi lần fused_layer_norm() được gọi."""
    call_count = {"n": 0}

    def _counting_is_available():
        call_count["n"] += 1
        return False

    monkeypatch.setattr(loader.torch.cuda, "is_available", _counting_is_available)

    loader._try_compile()
    loader._try_compile()
    loader._try_compile()

    assert call_count["n"] == 1  # chi lan dau tien thuc su chay logic ben trong


def test_try_compile_swallows_compile_exception_and_falls_back(monkeypatch, caplog):
    """Ràng buộc #1 (docstring module): compile lỗi vì bất kỳ lý do gì
    (thiếu nvcc, sai version...) -- KHÔNG được crash, phải rơi về
    fallback. Giả lập bằng cách để torch.cuda.is_available() trả True (đi
    vào nhánh thử compile) nhưng torch.utils.cpp_extension.load bị patch
    để raise."""
    monkeypatch.setattr(loader.torch.cuda, "is_available", lambda: True)

    import torch.utils.cpp_extension as cpp_extension

    def _raise_load(*a, **k):
        raise RuntimeError("gia lap: khong tim thay nvcc")

    monkeypatch.setattr(cpp_extension, "load", _raise_load)

    with caplog.at_level("WARNING"):
        loader._try_compile()  # KHONG duoc raise

    assert loader._EXTENSION is False
    assert "compile" in caplog.text.lower()


# ---------- fused_layer_norm(): fallback trên CPU ----------


def test_fused_layer_norm_cpu_tensor_matches_pytorch_reference():
    torch.manual_seed(0)
    x = torch.randn(4, 8)
    weight = torch.ones(8)
    bias = torch.zeros(8)

    out = loader.fused_layer_norm(x, weight, bias)
    expected = F.layer_norm(x, [8], weight, bias, 1e-5)

    assert torch.allclose(out, expected, atol=1e-6)


def test_fused_layer_norm_cpu_tensor_with_grad_enabled_matches_reference():
    """Trên CPU, luôn fallback bất kể grad state -- nhưng vẫn phải cho
    kết quả ĐÚNG (không chỉ "không crash")."""
    torch.manual_seed(1)
    x = torch.randn(2, 6, requires_grad=True)
    weight = torch.ones(6, requires_grad=True)
    bias = torch.zeros(6, requires_grad=True)

    out = loader.fused_layer_norm(x, weight, bias)
    expected = F.layer_norm(x, [6], weight, bias, 1e-5)

    assert torch.allclose(out, expected, atol=1e-6)
    assert out.requires_grad is True  # fallback F.layer_norm phai giu duoc autograd graph


def test_fused_layer_norm_respects_custom_eps():
    torch.manual_seed(2)
    x = torch.randn(3, 5)
    weight = torch.ones(5)
    bias = torch.zeros(5)

    out = loader.fused_layer_norm(x, weight, bias, eps=1e-2)
    expected = F.layer_norm(x, [5], weight, bias, 1e-2)

    assert torch.allclose(out, expected, atol=1e-6)
    # eps lon hon ro ret phai cho ket qua KHAC eps mac dinh (kiem tra tham
    # so thuc su duoc dung, khong bi bo qua/hard-code trong loader.py)
    default_out = F.layer_norm(x, [5], weight, bias, 1e-5)
    assert not torch.allclose(out, default_out, atol=1e-8)


# ---------- is_kernel_active() ----------


def test_is_kernel_active_false_without_cuda(monkeypatch):
    monkeypatch.setattr(loader.torch.cuda, "is_available", lambda: False)
    assert loader.is_kernel_active() is False


def test_is_kernel_active_triggers_compile_attempt_if_not_yet_tried(monkeypatch):
    calls = {"n": 0}

    def _counting_is_available():
        calls["n"] += 1
        return False

    monkeypatch.setattr(loader.torch.cuda, "is_available", _counting_is_available)
    assert loader._COMPILE_ATTEMPTED is False  # dam bao fixture da reset dung

    loader.is_kernel_active()

    assert calls["n"] == 1  # da tu goi _try_compile() ben trong


# ---------- Đánh dấu gpu: cần GPU thật, KHÔNG chạy được ở sandbox này ----------
# (bị loại bởi `-m "not gpu"` -- viết sẵn cho người chạy trên Colab T4)


@pytest.mark.gpu
def test_fused_layer_norm_real_cuda_kernel_matches_reference():
    assert torch.cuda.is_available(), "Test này cần GPU thật (chạy Colab, không phải sandbox này)"
    torch.manual_seed(0)
    x = torch.randn(4, 8, device="cuda")
    weight = torch.ones(8, device="cuda")
    bias = torch.zeros(8, device="cuda")

    with torch.no_grad():
        out = loader.fused_layer_norm(x, weight, bias)
    expected = F.layer_norm(x, [8], weight, bias, 1e-5)

    assert loader.is_kernel_active() is True
    assert torch.allclose(out, expected, atol=1e-4)


@pytest.mark.gpu
def test_fused_layer_norm_falls_back_when_grad_enabled_despite_real_cuda():
    """Ràng buộc #2 (docstring module): dù compile CUDA thành công, đang
    cần gradient thì VẪN phải fallback -- không tin caller tự nhớ gọi
    torch.no_grad()."""
    assert torch.cuda.is_available(), "Test này cần GPU thật"
    torch.manual_seed(0)
    x = torch.randn(4, 8, device="cuda", requires_grad=True)
    weight = torch.ones(8, device="cuda", requires_grad=True)
    bias = torch.zeros(8, device="cuda", requires_grad=True)

    out = loader.fused_layer_norm(x, weight, bias)  # KHONG boc torch.no_grad()

    # chi F.layer_norm (fallback) moi giu duoc autograd graph nguyen ven
    assert out.requires_grad is True
