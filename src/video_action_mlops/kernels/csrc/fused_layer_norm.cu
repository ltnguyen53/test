// fused_layer_norm.cu — Fused LayerNorm forward, CHỈ inference (không có
// backward — xem ràng buộc #2 trong kernels/loader.py). Mục tiêu tối ưu:
// self.norm = nn.LayerNorm(input_dim) trong TemporalAggregatorTrainable
// (models/temporal.py, phiên 3.2), gọi mỗi forward pass lúc train phase2
// (phiên 5.2).
//
// Mỗi CUDA block xử lý 1 "row" (1 vector cần normalize — vd 1 (batch,
// timestep) trong chuỗi (B, T, D)). Dùng shared memory reduction cho
// mean/variance — pattern chuẩn, KHÔNG phải kernel tối ưu nhất có thể có
// (chưa dùng warp-shuffle, chưa fuse residual-add), phù hợp quy mô học
// tập của project này (roadmap mục 3.4: "optimization", không phải yêu
// cầu bắt buộc để đúng).

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fused_layer_norm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int64_t num_rows,
    int64_t row_size,
    float eps
) {
    int64_t row = blockIdx.x;
    if (row >= num_rows) return;

    extern __shared__ float shared[];
    float* shared_sum = shared;                 // [blockDim.x]
    float* shared_sqsum = shared + blockDim.x;   // [blockDim.x]

    const float* row_in = x + row * row_size;
    float* row_out = out + row * row_size;

    float local_sum = 0.0f;
    float local_sqsum = 0.0f;
    for (int64_t i = threadIdx.x; i < row_size; i += blockDim.x) {
        float v = row_in[i];
        local_sum += v;
        local_sqsum += v * v;
    }
    shared_sum[threadIdx.x] = local_sum;
    shared_sqsum[threadIdx.x] = local_sqsum;
    __syncthreads();

    // Reduction cây nhị phân chuẩn trong shared memory.
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
            shared_sqsum[threadIdx.x] += shared_sqsum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float mean = shared_sum[0] / static_cast<float>(row_size);
    float var = shared_sqsum[0] / static_cast<float>(row_size) - mean * mean;
    float inv_std = rsqrtf(var + eps);

    for (int64_t i = threadIdx.x; i < row_size; i += blockDim.x) {
        float normed = (row_in[i] - mean) * inv_std;
        row_out[i] = normed * weight[i] + bias[i];
    }
}

torch::Tensor fused_layer_norm_cuda(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps
) {
    TORCH_CHECK(x.is_cuda(), "x phải là CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "chỉ hỗ trợ float32 (bản tối giản, chưa hỗ trợ fp16/bf16)");
    TORCH_CHECK(x.size(-1) == weight.size(0), "chiều cuối của x phải khớp weight");
    TORCH_CHECK(weight.size(0) == bias.size(0), "weight và bias phải cùng kích thước");

    auto x_contig = x.contiguous();
    int64_t row_size = x_contig.size(-1);
    int64_t num_rows = x_contig.numel() / row_size;

    auto out = torch::empty_like(x_contig);

    const int threads = 256;
    const int shared_mem = 2 * threads * sizeof(float);

    fused_layer_norm_kernel<<<num_rows, threads, shared_mem>>>(
        x_contig.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        num_rows,
        row_size,
        static_cast<float>(eps)
    );

    return out;
}
