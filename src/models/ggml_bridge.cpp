#include "models/ggml_bridge.h"

#include "ggml-cpu.h"
#include "ggml.h"

#include <cstring>
#include <stdexcept>

namespace vlacpp {

void ggml_linear(
    const Tensor & weight,
    const Tensor & bias,
    const std::vector<float> & input,
    std::vector<float> & output,
    int n_threads) {
    ggml_linear_batch(weight, bias, input, 1, output, n_threads);
}

void ggml_linear_batch(
    const Tensor & weight,
    const Tensor & bias,
    const std::vector<float> & input,
    int batch,
    std::vector<float> & output,
    int n_threads) {
    if (weight.shape.size() != 2 || bias.shape.size() != 1) {
        throw std::invalid_argument("linear expects rank-2 weight and rank-1 bias");
    }
    const int64_t out_dim = weight.shape[0];
    const int64_t in_dim = weight.shape[1];
    if (out_dim <= 0 || in_dim <= 0 || batch <= 0 ||
        bias.shape[0] != out_dim ||
        input.size() != static_cast<size_t>(batch) * static_cast<size_t>(in_dim) ||
        weight.data.size() != static_cast<size_t>(out_dim * in_dim) ||
        bias.data.size() != static_cast<size_t>(out_dim)) {
        throw std::invalid_argument("linear tensor shapes are inconsistent");
    }

    const size_t tensor_bytes =
        weight.data.size() * sizeof(float) +
        bias.data.size() * sizeof(float) +
        input.size() * sizeof(float) +
        static_cast<size_t>(batch) * static_cast<size_t>(out_dim) * sizeof(float);
    const size_t context_size = std::max<size_t>(64 * 1024 * 1024, tensor_bytes * 4 + 1024 * 1024);
    ggml_init_params params{};
    params.mem_size = context_size;
    params.mem_buffer = nullptr;
    params.no_alloc = false;
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        throw std::runtime_error("failed to initialize ggml context");
    }

    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_dim, out_dim);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_dim, batch);
    ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_dim);
    std::memcpy(ggml_get_data_f32(w), weight.data.data(), weight.data.size() * sizeof(float));
    std::memcpy(ggml_get_data_f32(x), input.data(), input.size() * sizeof(float));
    std::memcpy(ggml_get_data_f32(b), bias.data.data(), bias.data.size() * sizeof(float));

    ggml_tensor * y = ggml_add(ctx, ggml_mul_mat(ctx, w, x), b);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    const int threads = std::max(1, n_threads);
    const ggml_status status = ggml_graph_compute_with_ctx(ctx, graph, threads);
    if (status != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        throw std::runtime_error("ggml linear graph compute failed");
    }

    output.assign(
        ggml_get_data_f32(y),
        ggml_get_data_f32(y) + static_cast<size_t>(batch) * static_cast<size_t>(out_dim));
    ggml_free(ctx);
}

} // namespace vlacpp
