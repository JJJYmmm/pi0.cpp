#include "models/pi0_action_expert.h"

#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace vlacpp {
namespace {

void copy_tensor_data(ggml_tensor * dst, const Tensor & src) {
    std::memcpy(ggml_get_data_f32(dst), src.data.data(), src.data.size() * sizeof(float));
}

void require_weight_shape(const Tensor & tensor, int64_t ne0, int64_t ne1, const char * name) {
    if (tensor.shape.size() != 2 ||
        tensor.shape[0] != ne0 ||
        tensor.shape[1] != ne1 ||
        tensor.data.size() != static_cast<size_t>(ne0 * ne1)) {
        throw std::invalid_argument(std::string(name) + " has incompatible ggml shape");
    }
}

} // namespace

Pi0ActionExpert::Pi0ActionExpert(
    const ModelConfig & config,
    const BackendConfig & backend,
    const TensorMap & tensors)
    : config_(config), backend_(backend), tensors_(tensors) {}

bool Pi0ActionExpert::has_layer(int layer) const {
    return find_tensor(layer_prefix(layer) + "mlp.gate_proj.weight") != nullptr &&
        find_tensor(layer_prefix(layer) + "mlp.up_proj.weight") != nullptr &&
        find_tensor(layer_prefix(layer) + "mlp.down_proj.weight") != nullptr;
}

void Pi0ActionExpert::mlp_batch(
    int layer,
    const std::vector<float> & tokens,
    int batch,
    std::vector<float> & out) const {
    if (batch <= 0 ||
        config_.openpi_action_expert_width <= 0 ||
        config_.openpi_action_expert_mlp_width <= 0) {
        throw std::invalid_argument("invalid pi0 action expert MLP dimensions");
    }
    const std::string prefix = layer_prefix(layer);
    const Tensor * gate_w = find_tensor(prefix + "mlp.gate_proj.weight");
    const Tensor * up_w = find_tensor(prefix + "mlp.up_proj.weight");
    const Tensor * down_w = find_tensor(prefix + "mlp.down_proj.weight");
    if (gate_w == nullptr || up_w == nullptr || down_w == nullptr) {
        throw std::invalid_argument("missing pi0 action expert MLP tensors");
    }
    const int64_t width = config_.openpi_action_expert_width;
    const int64_t hidden = config_.openpi_action_expert_mlp_width;
    require_weight_shape(*gate_w, width, hidden, "action expert gate_proj");
    require_weight_shape(*up_w, width, hidden, "action expert up_proj");
    require_weight_shape(*down_w, hidden, width, "action expert down_proj");
    if (tokens.size() != static_cast<size_t>(batch) * static_cast<size_t>(width)) {
        throw std::invalid_argument("action expert token batch has incompatible shape");
    }

    const size_t tensor_bytes =
        (gate_w->data.size() + up_w->data.size() + down_w->data.size() + tokens.size()) * sizeof(float) +
        static_cast<size_t>(batch) * static_cast<size_t>(hidden + width) * sizeof(float);
    const size_t context_size = std::max<size_t>(64 * 1024 * 1024, tensor_bytes * 4 + 1024 * 1024);
    ggml_init_params params{};
    params.mem_size = context_size;
    params.mem_buffer = nullptr;
    params.no_alloc = false;
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        throw std::runtime_error("failed to initialize ggml context");
    }

    ggml_tensor * gate = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, hidden);
    ggml_tensor * up = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, hidden);
    ggml_tensor * down = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, width);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, batch);
    copy_tensor_data(gate, *gate_w);
    copy_tensor_data(up, *up_w);
    copy_tensor_data(down, *down_w);
    std::memcpy(ggml_get_data_f32(x), tokens.data(), tokens.size() * sizeof(float));

    ggml_tensor * gate_out = ggml_gelu(ctx, ggml_mul_mat(ctx, gate, x));
    ggml_tensor * up_out = ggml_mul_mat(ctx, up, x);
    ggml_tensor * hidden_out = ggml_mul(ctx, gate_out, up_out);
    ggml_tensor * y = ggml_mul_mat(ctx, down, hidden_out);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    const ggml_status status = ggml_graph_compute_with_ctx(ctx, graph, std::max(1, backend_.n_threads));
    if (status != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        throw std::runtime_error("ggml action expert MLP graph compute failed");
    }

    out.assign(
        ggml_get_data_f32(y),
        ggml_get_data_f32(y) + static_cast<size_t>(batch) * static_cast<size_t>(width));
    ggml_free(ctx);
}

const Tensor * Pi0ActionExpert::find_tensor(const std::string & name) const {
    auto it = tensors_.find(name);
    if (it != tensors_.end()) {
        return &it->second;
    }
    it = tensors_.find("model." + name);
    if (it != tensors_.end()) {
        return &it->second;
    }
    return nullptr;
}

std::string Pi0ActionExpert::layer_prefix(int layer) const {
    if (layer < 0 || layer >= config_.openpi_action_expert_layers) {
        throw std::invalid_argument("action expert layer index is out of range");
    }
    return "paligemma_with_expert.gemma_expert.model.layers." + std::to_string(layer) + ".";
}

} // namespace vlacpp
