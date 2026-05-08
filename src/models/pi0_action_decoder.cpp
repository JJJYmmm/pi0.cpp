#include "models/pi0_action_decoder.h"

#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace vlacpp {
namespace {

float swish(float x) {
    return x / (1.0f + std::exp(-x));
}

void run_mul_mat_add_batch(
    const Tensor & weight,
    const Tensor & bias,
    const std::vector<float> & input,
    int batch,
    std::vector<float> & output,
    int n_threads) {
    if (weight.shape.size() != 2 || bias.shape.size() != 1) {
        throw std::invalid_argument("linear expects rank-2 weight and rank-1 bias");
    }
    const int64_t ne0 = weight.shape[0];
    const int64_t ne1 = weight.shape[1];
    if (ne0 <= 0 || ne1 <= 0 || batch <= 0 ||
        bias.shape[0] != ne1 ||
        input.size() != static_cast<size_t>(batch) * static_cast<size_t>(ne0) ||
        weight.data.size() != static_cast<size_t>(ne0 * ne1) ||
        bias.data.size() != static_cast<size_t>(ne1)) {
        throw std::invalid_argument("linear tensor shapes are inconsistent");
    }

    const size_t tensor_bytes =
        weight.data.size() * sizeof(float) +
        bias.data.size() * sizeof(float) +
        input.size() * sizeof(float) +
        static_cast<size_t>(batch) * static_cast<size_t>(ne1) * sizeof(float);
    const size_t context_size = std::max<size_t>(64 * 1024 * 1024, tensor_bytes * 4 + 1024 * 1024);
    ggml_init_params params{};
    params.mem_size = context_size;
    params.mem_buffer = nullptr;
    params.no_alloc = false;
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        throw std::runtime_error("failed to initialize ggml context");
    }

    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, batch);
    ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne1);
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
        ggml_get_data_f32(y) + static_cast<size_t>(batch) * static_cast<size_t>(ne1));
    ggml_free(ctx);
}

void run_mul_mat_add(
    const Tensor & weight,
    const Tensor & bias,
    const std::vector<float> & input,
    std::vector<float> & output,
    int n_threads) {
    run_mul_mat_add_batch(weight, bias, input, 1, output, n_threads);
}

std::vector<float> posemb_sincos(float time, size_t width) {
    std::vector<float> result(width, 0.0f);
    const size_t half = width / 2;
    if (half == 0) {
        return result;
    }
    constexpr float min_period = 4.0e-3f;
    constexpr float max_period = 4.0f;
    constexpr float pi = 3.14159265358979323846f;
    for (size_t i = 0; i < half; ++i) {
        const float fraction = half == 1 ? 0.0f : static_cast<float>(i) / static_cast<float>(half - 1);
        const float period = min_period * std::pow(max_period / min_period, fraction);
        const float angle = time / period * 2.0f * pi;
        result[i] = std::sin(angle);
        result[half + i] = std::cos(angle);
    }
    return result;
}

} // namespace

Pi0ActionDecoder::Pi0ActionDecoder(
    const ModelConfig & config,
    const BackendConfig & backend,
    const TensorMap & tensors)
    : config_(config), backend_(backend), tensors_(tensors) {}

bool Pi0ActionDecoder::has_pi05_action_head() const {
    return find_tensor("vlacpp.openpi.pi05.time_mlp_in.weight") != nullptr;
}

bool Pi0ActionDecoder::has_pi0_action_head() const {
    return find_tensor("vlacpp.openpi.action_in_proj.weight") != nullptr && !has_pi05_action_head();
}

void Pi0ActionDecoder::state_context(const std::vector<float> & state, std::vector<float> & out) const {
    const Tensor * state_w = find_tensor("vlacpp.openpi.state_proj.weight");
    const Tensor * state_b = find_tensor("vlacpp.openpi.state_proj.bias");
    if (state_w == nullptr || state_b == nullptr) {
        out.clear();
        return;
    }
    run_mul_mat_add(*state_w, *state_b, state, out, backend_.n_threads);
}

void Pi0ActionDecoder::velocity_batch(
    float time,
    const std::vector<float> & actions,
    const std::vector<float> & state_context,
    std::vector<float> & out) const {
    const Tensor & in_w = *find_tensor("vlacpp.openpi.action_in_proj.weight");
    const Tensor & in_b = *find_tensor("vlacpp.openpi.action_in_proj.bias");
    const Tensor & out_w = *find_tensor("vlacpp.openpi.action_out_proj.weight");
    const Tensor & out_b = *find_tensor("vlacpp.openpi.action_out_proj.bias");

    const int batch = config_.action_horizon;
    const size_t width = static_cast<size_t>(in_w.shape[1]);
    std::vector<float> action_tokens;
    run_mul_mat_add_batch(in_w, in_b, actions, batch, action_tokens, backend_.n_threads);

    if (has_pi05_action_head()) {
        const Tensor & time_in_w = *find_tensor("vlacpp.openpi.pi05.time_mlp_in.weight");
        const Tensor & time_in_b = *find_tensor("vlacpp.openpi.pi05.time_mlp_in.bias");
        const Tensor & time_out_w = *find_tensor("vlacpp.openpi.pi05.time_mlp_out.weight");
        const Tensor & time_out_b = *find_tensor("vlacpp.openpi.pi05.time_mlp_out.bias");
        std::vector<float> hidden;
        run_mul_mat_add(time_in_w, time_in_b, posemb_sincos(time, width), hidden, backend_.n_threads);
        for (float & value : hidden) {
            value = swish(value);
        }
        std::vector<float> time_context;
        run_mul_mat_add(time_out_w, time_out_b, hidden, time_context, backend_.n_threads);
        for (float & value : time_context) {
            value = swish(value);
        }
        for (int row = 0; row < batch; ++row) {
            const size_t offset = static_cast<size_t>(row) * width;
            for (size_t i = 0; i < width; ++i) {
                action_tokens[offset + i] += time_context[i];
            }
        }
        run_mul_mat_add_batch(out_w, out_b, action_tokens, batch, out, backend_.n_threads);
        return;
    }

    if (!state_context.empty()) {
        for (int row = 0; row < batch; ++row) {
            const size_t offset = static_cast<size_t>(row) * width;
            for (size_t i = 0; i < width; ++i) {
                action_tokens[offset + i] += state_context[i];
            }
        }
    }
    const Tensor & time_in_w = *find_tensor("vlacpp.openpi.action_time_mlp_in.weight");
    const Tensor & time_in_b = *find_tensor("vlacpp.openpi.action_time_mlp_in.bias");
    const Tensor & time_out_w = *find_tensor("vlacpp.openpi.action_time_mlp_out.weight");
    const Tensor & time_out_b = *find_tensor("vlacpp.openpi.action_time_mlp_out.bias");
    const std::vector<float> time_emb = posemb_sincos(time, width);
    std::vector<float> action_time(static_cast<size_t>(batch) * width * 2, 0.0f);
    for (int row = 0; row < batch; ++row) {
        const size_t src = static_cast<size_t>(row) * width;
        const size_t dst = static_cast<size_t>(row) * width * 2;
        std::copy(
            action_tokens.begin() + static_cast<std::ptrdiff_t>(src),
            action_tokens.begin() + static_cast<std::ptrdiff_t>(src + width),
            action_time.begin() + static_cast<std::ptrdiff_t>(dst));
        std::copy(
            time_emb.begin(),
            time_emb.end(),
            action_time.begin() + static_cast<std::ptrdiff_t>(dst + width));
    }
    std::vector<float> hidden;
    run_mul_mat_add_batch(time_in_w, time_in_b, action_time, batch, hidden, backend_.n_threads);
    for (float & value : hidden) {
        value = swish(value);
    }
    std::vector<float> mixed;
    run_mul_mat_add_batch(time_out_w, time_out_b, hidden, batch, mixed, backend_.n_threads);
    for (float & value : mixed) {
        value = swish(value);
    }
    run_mul_mat_add_batch(out_w, out_b, mixed, batch, out, backend_.n_threads);
}

const Tensor * Pi0ActionDecoder::find_tensor(const std::string & name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        return nullptr;
    }
    return &it->second;
}

} // namespace vlacpp
