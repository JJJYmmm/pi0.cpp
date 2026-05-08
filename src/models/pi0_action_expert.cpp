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

void require_vector_shape(const Tensor & tensor, int64_t ne0, const char * name) {
    if (tensor.shape.size() != 1 ||
        tensor.shape[0] != ne0 ||
        tensor.data.size() != static_cast<size_t>(ne0)) {
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

void Pi0ActionExpert::input_norm_batch(
    int layer,
    const std::vector<float> & tokens,
    int batch,
    std::vector<float> & out) const {
    norm_batch(layer, "input_layernorm.weight", tokens, batch, out);
}

void Pi0ActionExpert::post_attention_norm_batch(
    int layer,
    const std::vector<float> & tokens,
    int batch,
    std::vector<float> & out) const {
    norm_batch(layer, "post_attention_layernorm.weight", tokens, batch, out);
}

void Pi0ActionExpert::norm_batch(
    int layer,
    const char * weight_name,
    const std::vector<float> & tokens,
    int batch,
    std::vector<float> & out) const {
    if (batch <= 0 || config_.openpi_action_expert_width <= 0) {
        throw std::invalid_argument("invalid pi0 action expert norm dimensions");
    }
    const Tensor * norm_w = find_tensor(layer_prefix(layer) + weight_name);
    if (norm_w == nullptr) {
        throw std::invalid_argument("missing pi0 action expert norm tensor");
    }
    const int64_t width = config_.openpi_action_expert_width;
    require_vector_shape(*norm_w, width, "action expert norm");
    if (tokens.size() != static_cast<size_t>(batch) * static_cast<size_t>(width)) {
        throw std::invalid_argument("action expert norm input has incompatible shape");
    }

    const size_t tensor_bytes =
        (norm_w->data.size() + tokens.size() + static_cast<size_t>(width) * static_cast<size_t>(batch)) *
        sizeof(float);
    const size_t context_size = std::max<size_t>(64 * 1024 * 1024, tensor_bytes * 4 + 1024 * 1024);
    ggml_init_params params{};
    params.mem_size = context_size;
    params.mem_buffer = nullptr;
    params.no_alloc = false;
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        throw std::runtime_error("failed to initialize ggml context");
    }

    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, batch);
    ggml_tensor * scale = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, width);
    std::memcpy(ggml_get_data_f32(x), tokens.data(), tokens.size() * sizeof(float));
    float * scale_data = ggml_get_data_f32(scale);
    for (int64_t i = 0; i < width; ++i) {
        scale_data[i] = 1.0f + norm_w->data[static_cast<size_t>(i)];
    }

    ggml_tensor * y = ggml_mul(ctx, ggml_rms_norm(ctx, x, 1.0e-6f), scale);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    const ggml_status status = ggml_graph_compute_with_ctx(ctx, graph, std::max(1, backend_.n_threads));
    if (status != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        throw std::runtime_error("ggml action expert norm graph compute failed");
    }

    out.assign(
        ggml_get_data_f32(y),
        ggml_get_data_f32(y) + static_cast<size_t>(batch) * static_cast<size_t>(width));
    ggml_free(ctx);
}

void Pi0ActionExpert::qkv_batch(
    int layer,
    const std::vector<float> & tokens,
    int batch,
    std::vector<float> & q,
    std::vector<float> & k,
    std::vector<float> & v) const {
    if (batch <= 0 ||
        config_.openpi_action_expert_width <= 0 ||
        config_.openpi_action_expert_q_out <= 0 ||
        config_.openpi_action_expert_kv_out <= 0) {
        throw std::invalid_argument("invalid pi0 action expert QKV dimensions");
    }
    const std::string prefix = layer_prefix(layer) + "self_attn.";
    const Tensor * q_w = find_tensor(prefix + "q_proj.weight");
    const Tensor * k_w = find_tensor(prefix + "k_proj.weight");
    const Tensor * v_w = find_tensor(prefix + "v_proj.weight");
    if (q_w == nullptr || k_w == nullptr || v_w == nullptr) {
        throw std::invalid_argument("missing pi0 action expert QKV tensors");
    }
    const int64_t width = config_.openpi_action_expert_width;
    const int64_t q_out = config_.openpi_action_expert_q_out;
    const int64_t kv_out = config_.openpi_action_expert_kv_out;
    require_weight_shape(*q_w, width, q_out, "action expert q_proj");
    require_weight_shape(*k_w, width, kv_out, "action expert k_proj");
    require_weight_shape(*v_w, width, kv_out, "action expert v_proj");
    if (tokens.size() != static_cast<size_t>(batch) * static_cast<size_t>(width)) {
        throw std::invalid_argument("action expert QKV input has incompatible shape");
    }

    const size_t tensor_bytes =
        (q_w->data.size() + k_w->data.size() + v_w->data.size() + tokens.size()) * sizeof(float) +
        static_cast<size_t>(batch) * static_cast<size_t>(q_out + 2 * kv_out) * sizeof(float);
    const size_t context_size = std::max<size_t>(64 * 1024 * 1024, tensor_bytes * 4 + 1024 * 1024);
    ggml_init_params params{};
    params.mem_size = context_size;
    params.mem_buffer = nullptr;
    params.no_alloc = false;
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        throw std::runtime_error("failed to initialize ggml context");
    }

    ggml_tensor * qw = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, q_out);
    ggml_tensor * kw = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, kv_out);
    ggml_tensor * vw = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, kv_out);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, batch);
    copy_tensor_data(qw, *q_w);
    copy_tensor_data(kw, *k_w);
    copy_tensor_data(vw, *v_w);
    std::memcpy(ggml_get_data_f32(x), tokens.data(), tokens.size() * sizeof(float));

    ggml_tensor * q_out_tensor = ggml_mul_mat(ctx, qw, x);
    ggml_tensor * k_out_tensor = ggml_mul_mat(ctx, kw, x);
    ggml_tensor * v_out_tensor = ggml_mul_mat(ctx, vw, x);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, q_out_tensor);
    ggml_build_forward_expand(graph, k_out_tensor);
    ggml_build_forward_expand(graph, v_out_tensor);
    const ggml_status status = ggml_graph_compute_with_ctx(ctx, graph, std::max(1, backend_.n_threads));
    if (status != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        throw std::runtime_error("ggml action expert QKV graph compute failed");
    }

    q.assign(
        ggml_get_data_f32(q_out_tensor),
        ggml_get_data_f32(q_out_tensor) + static_cast<size_t>(batch) * static_cast<size_t>(q_out));
    k.assign(
        ggml_get_data_f32(k_out_tensor),
        ggml_get_data_f32(k_out_tensor) + static_cast<size_t>(batch) * static_cast<size_t>(kv_out));
    v.assign(
        ggml_get_data_f32(v_out_tensor),
        ggml_get_data_f32(v_out_tensor) + static_cast<size_t>(batch) * static_cast<size_t>(kv_out));
    ggml_free(ctx);
}

void Pi0ActionExpert::attention_out_batch(
    int layer,
    const std::vector<float> & values,
    int batch,
    std::vector<float> & out) const {
    if (batch <= 0 ||
        config_.openpi_action_expert_width <= 0 ||
        config_.openpi_action_expert_q_out <= 0) {
        throw std::invalid_argument("invalid pi0 action expert attention output dimensions");
    }
    const Tensor * out_w = find_tensor(layer_prefix(layer) + "self_attn.o_proj.weight");
    if (out_w == nullptr) {
        throw std::invalid_argument("missing pi0 action expert attention output tensor");
    }
    const int64_t width = config_.openpi_action_expert_width;
    const int64_t q_out = config_.openpi_action_expert_q_out;
    require_weight_shape(*out_w, q_out, width, "action expert o_proj");
    if (values.size() != static_cast<size_t>(batch) * static_cast<size_t>(q_out)) {
        throw std::invalid_argument("action expert attention output input has incompatible shape");
    }

    const size_t tensor_bytes =
        (out_w->data.size() + values.size()) * sizeof(float) +
        static_cast<size_t>(batch) * static_cast<size_t>(width) * sizeof(float);
    const size_t context_size = std::max<size_t>(64 * 1024 * 1024, tensor_bytes * 4 + 1024 * 1024);
    ggml_init_params params{};
    params.mem_size = context_size;
    params.mem_buffer = nullptr;
    params.no_alloc = false;
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        throw std::runtime_error("failed to initialize ggml context");
    }

    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, q_out, width);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, q_out, batch);
    copy_tensor_data(w, *out_w);
    std::memcpy(ggml_get_data_f32(x), values.data(), values.size() * sizeof(float));

    ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    const ggml_status status = ggml_graph_compute_with_ctx(ctx, graph, std::max(1, backend_.n_threads));
    if (status != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        throw std::runtime_error("ggml action expert attention output graph compute failed");
    }

    out.assign(
        ggml_get_data_f32(y),
        ggml_get_data_f32(y) + static_cast<size_t>(batch) * static_cast<size_t>(width));
    ggml_free(ctx);
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
