// fused_layer_norm_binding.cpp — pybind11 bindings. Tách riêng khỏi file
// .cu (pattern chuẩn của torch.utils.cpp_extension.load: 1 file .cpp chứa
// PYBIND11_MODULE, 1 file .cu chỉ chứa kernel + hàm host — tránh vấn đề
// tương thích khi nvcc compile macro pybind trực tiếp).

#include <torch/extension.h>

// Khai báo hàm định nghĩa trong fused_layer_norm.cu — compile riêng (nvcc
// cho .cu, g++ cho .cpp này), link chung bởi torch.utils.cpp_extension.load
// khi truyền cả 2 file vào tham số sources=[...].
torch::Tensor fused_layer_norm_cuda(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "fused_layer_norm_cuda",
        &fused_layer_norm_cuda,
        "Fused LayerNorm forward (CUDA, KHÔNG có backward — chỉ dùng khi không cần gradient)"
    );
}
