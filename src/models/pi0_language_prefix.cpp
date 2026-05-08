#include "models/pi0_language_prefix.h"

#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
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

void run_norm(
    const Tensor & norm_w,
    const std::vector<float> & tokens,
    int batch,
    int64_t width,
    int n_threads,
    std::vector<float> & out) {
    require_vector_shape(norm_w, width, "language prefix norm");
    if (batch <= 0 || tokens.size() != static_cast<size_t>(batch) * static_cast<size_t>(width)) {
        throw std::invalid_argument("language prefix norm input has incompatible shape");
    }

    const size_t tensor_bytes =
        (norm_w.data.size() + tokens.size() + static_cast<size_t>(width) * static_cast<size_t>(batch)) *
        sizeof(float);
    const size_t context_size = std::max<size_t>(64 * 1024 * 1024, tensor_bytes * 4 + 1024 * 1024);
    ggml_init_params params{};
    params.mem_size = context_size;
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        throw std::runtime_error("failed to initialize ggml context");
    }

    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, batch);
    ggml_tensor * scale = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, width);
    std::memcpy(ggml_get_data_f32(x), tokens.data(), tokens.size() * sizeof(float));
    float * scale_data = ggml_get_data_f32(scale);
    for (int64_t i = 0; i < width; ++i) {
        scale_data[i] = 1.0f + norm_w.data[static_cast<size_t>(i)];
    }

    ggml_tensor * y = ggml_mul(ctx, ggml_rms_norm(ctx, x, 1.0e-6f), scale);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    const ggml_status status = ggml_graph_compute_with_ctx(ctx, graph, std::max(1, n_threads));
    if (status != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        throw std::runtime_error("ggml language prefix norm graph compute failed");
    }

    out.assign(
        ggml_get_data_f32(y),
        ggml_get_data_f32(y) + static_cast<size_t>(batch) * static_cast<size_t>(width));
    ggml_free(ctx);
}

} // namespace

Pi0LanguagePrefix::Pi0LanguagePrefix(
    const ModelConfig & config,
    const BackendConfig & backend,
    const TensorMap & tensors)
    : config_(config), backend_(backend), tensors_(tensors) {}

bool Pi0LanguagePrefix::has_layer(int layer) const {
    return find_tensor(layer_prefix(layer) + "mlp.gate_proj.weight") != nullptr &&
        find_tensor(layer_prefix(layer) + "self_attn.q_proj.weight") != nullptr;
}

void Pi0LanguagePrefix::prefill_batch(
    const std::vector<float> & tokens,
    const std::vector<int> & positions,
    int batch,
    int heads,
    int kv_heads,
    int head_dim,
    std::vector<PrefixLayerKv> & cache,
    std::vector<float> & out) const {
    const size_t width = static_cast<size_t>(config_.openpi_language_width);
    if (batch <= 0 || width == 0 || tokens.size() != static_cast<size_t>(batch) * width ||
        positions.size() != static_cast<size_t>(batch)) {
        throw std::invalid_argument("language prefix input has incompatible shape");
    }

    std::vector<float> hidden = tokens;
    cache.clear();
    cache.reserve(static_cast<size_t>(config_.openpi_language_layers));
    for (int layer = 0; layer < config_.openpi_language_layers; ++layer) {
        std::vector<float> normed;
        norm_batch(layer, "input_layernorm.weight", hidden, batch, normed);

        std::vector<float> q;
        std::vector<float> k;
        std::vector<float> v;
        qkv_batch(layer, normed, batch, q, k, v);

        PrefixLayerKv layer_cache;
        rope_batch(q, positions, batch, heads, head_dim, q);
        rope_batch(k, positions, batch, kv_heads, head_dim, layer_cache.k);
        layer_cache.v = v;
        cache.push_back(layer_cache);

        std::vector<float> attn_values;
        self_attention_batch(q, cache.back().k, cache.back().v, batch, heads, kv_heads, head_dim, attn_values);

        std::vector<float> attn_out;
        attention_out_batch(layer, attn_values, batch, attn_out);

        std::vector<float> first_residual(hidden.size(), 0.0f);
        for (size_t i = 0; i < first_residual.size(); ++i) {
            first_residual[i] = hidden[i] + attn_out[i];
        }

        std::vector<float> post_norm;
        norm_batch(layer, "post_attention_layernorm.weight", first_residual, batch, post_norm);

        std::vector<float> mlp_out;
        mlp_batch(layer, post_norm, batch, mlp_out);

        hidden.resize(first_residual.size());
        for (size_t i = 0; i < hidden.size(); ++i) {
            hidden[i] = first_residual[i] + mlp_out[i];
        }
    }
    final_norm_batch(hidden, batch, out);
}

void Pi0LanguagePrefix::norm_batch(
    int layer,
    const char * weight_name,
    const std::vector<float> & tokens,
    int batch,
    std::vector<float> & out) const {
    const Tensor * norm_w = find_tensor(layer_prefix(layer) + weight_name);
    if (norm_w == nullptr) {
        throw std::invalid_argument("missing pi0 language prefix norm tensor");
    }
    run_norm(*norm_w, tokens, batch, config_.openpi_language_width, backend_.n_threads, out);
}

void Pi0LanguagePrefix::final_norm_batch(const std::vector<float> & tokens, int batch, std::vector<float> & out) const {
    const Tensor * norm_w = find_tensor("paligemma_with_expert.paligemma.model.language_model.norm.weight");
    if (norm_w == nullptr) {
        throw std::invalid_argument("missing pi0 language prefix final norm tensor");
    }
    run_norm(*norm_w, tokens, batch, config_.openpi_language_width, backend_.n_threads, out);
}

void Pi0LanguagePrefix::qkv_batch(
    int layer,
    const std::vector<float> & tokens,
    int batch,
    std::vector<float> & q,
    std::vector<float> & k,
    std::vector<float> & v) const {
    const std::string prefix = layer_prefix(layer) + "self_attn.";
    const Tensor * q_w = find_tensor(prefix + "q_proj.weight");
    const Tensor * k_w = find_tensor(prefix + "k_proj.weight");
    const Tensor * v_w = find_tensor(prefix + "v_proj.weight");
    if (q_w == nullptr || k_w == nullptr || v_w == nullptr) {
        throw std::invalid_argument("missing pi0 language prefix QKV tensors");
    }
    const int64_t width = config_.openpi_language_width;
    const int64_t q_out = config_.openpi_language_q_out;
    const int64_t kv_out = config_.openpi_language_kv_out;
    require_weight_shape(*q_w, width, q_out, "language prefix q_proj");
    require_weight_shape(*k_w, width, kv_out, "language prefix k_proj");
    require_weight_shape(*v_w, width, kv_out, "language prefix v_proj");
    if (batch <= 0 || tokens.size() != static_cast<size_t>(batch) * static_cast<size_t>(width)) {
        throw std::invalid_argument("language prefix QKV input has incompatible shape");
    }

    const size_t tensor_bytes =
        (q_w->data.size() + k_w->data.size() + v_w->data.size() + tokens.size()) * sizeof(float) +
        static_cast<size_t>(batch) * static_cast<size_t>(q_out + 2 * kv_out) * sizeof(float);
    const size_t context_size = std::max<size_t>(64 * 1024 * 1024, tensor_bytes * 4 + 1024 * 1024);
    ggml_init_params params{};
    params.mem_size = context_size;
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
        throw std::runtime_error("ggml language prefix QKV graph compute failed");
    }

    q.assign(ggml_get_data_f32(q_out_tensor), ggml_get_data_f32(q_out_tensor) + static_cast<size_t>(batch) * q_out);
    k.assign(ggml_get_data_f32(k_out_tensor), ggml_get_data_f32(k_out_tensor) + static_cast<size_t>(batch) * kv_out);
    v.assign(ggml_get_data_f32(v_out_tensor), ggml_get_data_f32(v_out_tensor) + static_cast<size_t>(batch) * kv_out);
    ggml_free(ctx);
}

void Pi0LanguagePrefix::rope_batch(
    const std::vector<float> & values,
    const std::vector<int> & positions,
    int tokens,
    int heads,
    int head_dim,
    std::vector<float> & out) const {
    if (tokens <= 0 || heads <= 0 || head_dim <= 0 || head_dim % 2 != 0 ||
        positions.size() != static_cast<size_t>(tokens)) {
        throw std::invalid_argument("invalid pi0 language prefix RoPE dimensions");
    }
    const size_t value_size =
        static_cast<size_t>(tokens) * static_cast<size_t>(heads) * static_cast<size_t>(head_dim);
    if (values.size() != value_size) {
        throw std::invalid_argument("language prefix RoPE input has incompatible shape");
    }

    const size_t tensor_bytes = (values.size() + positions.size()) * sizeof(float);
    const size_t context_size = std::max<size_t>(64 * 1024 * 1024, tensor_bytes * 4 + 1024 * 1024);
    ggml_init_params params{};
    params.mem_size = context_size;
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        throw std::runtime_error("failed to initialize ggml context");
    }

    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, heads, tokens);
    ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, tokens);
    std::memcpy(ggml_get_data_f32(x), values.data(), values.size() * sizeof(float));
    int32_t * pos_data = static_cast<int32_t *>(pos->data);
    for (int i = 0; i < tokens; ++i) {
        pos_data[i] = static_cast<int32_t>(positions[static_cast<size_t>(i)]);
    }

    ggml_tensor * y = ggml_rope_ext(
        ctx, x, pos, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, 8192, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    const ggml_status status = ggml_graph_compute_with_ctx(ctx, graph, std::max(1, backend_.n_threads));
    if (status != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        throw std::runtime_error("ggml language prefix RoPE graph compute failed");
    }

    out.assign(ggml_get_data_f32(y), ggml_get_data_f32(y) + value_size);
    ggml_free(ctx);
}

void Pi0LanguagePrefix::self_attention_batch(
    const std::vector<float> & q,
    const std::vector<float> & k,
    const std::vector<float> & v,
    int tokens,
    int heads,
    int kv_heads,
    int head_dim,
    std::vector<float> & out) const {
    if (tokens <= 0 || heads <= 0 || kv_heads <= 0 || head_dim <= 0 || heads % kv_heads != 0) {
        throw std::invalid_argument("invalid pi0 language prefix attention dimensions");
    }
    const size_t q_size = static_cast<size_t>(tokens) * static_cast<size_t>(heads) * static_cast<size_t>(head_dim);
    const size_t kv_size = static_cast<size_t>(tokens) * static_cast<size_t>(kv_heads) * static_cast<size_t>(head_dim);
    if (q.size() != q_size || k.size() != kv_size || v.size() != kv_size) {
        throw std::invalid_argument("language prefix attention inputs must have matching Q/K/V shape");
    }

    const size_t tensor_bytes = (q_size * 2 + kv_size * 2) * sizeof(float);
    const size_t context_size = std::max<size_t>(64 * 1024 * 1024, tensor_bytes * 8 + 1024 * 1024);
    ggml_init_params params{};
    params.mem_size = context_size;
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        throw std::runtime_error("failed to initialize ggml context");
    }

    ggml_tensor * q_cur = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, heads, tokens);
    ggml_tensor * k_cur = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, kv_heads, tokens);
    ggml_tensor * v_cur = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, kv_heads, tokens);
    std::memcpy(ggml_get_data_f32(q_cur), q.data(), q.size() * sizeof(float));
    std::memcpy(ggml_get_data_f32(k_cur), k.data(), k.size() * sizeof(float));
    std::memcpy(ggml_get_data_f32(v_cur), v.data(), v.size() * sizeof(float));
    if (kv_heads != heads) {
        ggml_tensor * kv_repeat_shape = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, heads, tokens);
        k_cur = ggml_repeat(ctx, k_cur, kv_repeat_shape);
        v_cur = ggml_repeat(ctx, v_cur, kv_repeat_shape);
    }

    ggml_tensor * q_perm = ggml_permute(ctx, q_cur, 0, 2, 1, 3);
    ggml_tensor * k_perm = ggml_permute(ctx, k_cur, 0, 2, 1, 3);
    ggml_tensor * v_perm = ggml_cont(ctx, ggml_permute(ctx, v_cur, 1, 2, 0, 3));
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    ggml_tensor * scores = ggml_mul_mat(ctx, k_perm, q_perm);
    scores = ggml_soft_max_ext(ctx, scores, nullptr, scale, 0.0f);
    ggml_tensor * values = ggml_mul_mat(ctx, v_perm, scores);
    ggml_tensor * y = ggml_permute(ctx, values, 0, 2, 1, 3);
    y = ggml_cont_2d(ctx, y, static_cast<int64_t>(head_dim) * heads, tokens);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    const ggml_status status = ggml_graph_compute_with_ctx(ctx, graph, std::max(1, backend_.n_threads));
    if (status != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        throw std::runtime_error("ggml language prefix attention graph compute failed");
    }

    out.assign(ggml_get_data_f32(y), ggml_get_data_f32(y) + q_size);
    ggml_free(ctx);
}

void Pi0LanguagePrefix::attention_out_batch(
    int layer,
    const std::vector<float> & values,
    int batch,
    std::vector<float> & out) const {
    const Tensor * out_w = find_tensor(layer_prefix(layer) + "self_attn.o_proj.weight");
    if (out_w == nullptr) {
        throw std::invalid_argument("missing pi0 language prefix attention output tensor");
    }
    const int64_t width = config_.openpi_language_width;
    const int64_t q_out = config_.openpi_language_q_out;
    require_weight_shape(*out_w, q_out, width, "language prefix o_proj");
    if (batch <= 0 || values.size() != static_cast<size_t>(batch) * static_cast<size_t>(q_out)) {
        throw std::invalid_argument("language prefix attention output input has incompatible shape");
    }

    const size_t tensor_bytes =
        (out_w->data.size() + values.size()) * sizeof(float) +
        static_cast<size_t>(batch) * static_cast<size_t>(width) * sizeof(float);
    const size_t context_size = std::max<size_t>(64 * 1024 * 1024, tensor_bytes * 4 + 1024 * 1024);
    ggml_init_params params{};
    params.mem_size = context_size;
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
        throw std::runtime_error("ggml language prefix attention output graph compute failed");
    }

    out.assign(ggml_get_data_f32(y), ggml_get_data_f32(y) + static_cast<size_t>(batch) * static_cast<size_t>(width));
    ggml_free(ctx);
}

void Pi0LanguagePrefix::mlp_batch(
    int layer,
    const std::vector<float> & tokens,
    int batch,
    std::vector<float> & out) const {
    const std::string prefix = layer_prefix(layer);
    const Tensor * gate_w = find_tensor(prefix + "mlp.gate_proj.weight");
    const Tensor * up_w = find_tensor(prefix + "mlp.up_proj.weight");
    const Tensor * down_w = find_tensor(prefix + "mlp.down_proj.weight");
    if (gate_w == nullptr || up_w == nullptr || down_w == nullptr) {
        throw std::invalid_argument("missing pi0 language prefix MLP tensors");
    }
    const int64_t width = config_.openpi_language_width;
    const int64_t hidden = config_.openpi_language_mlp_width;
    require_weight_shape(*gate_w, width, hidden, "language prefix gate_proj");
    require_weight_shape(*up_w, width, hidden, "language prefix up_proj");
    require_weight_shape(*down_w, hidden, width, "language prefix down_proj");
    if (batch <= 0 || tokens.size() != static_cast<size_t>(batch) * static_cast<size_t>(width)) {
        throw std::invalid_argument("language prefix token batch has incompatible shape");
    }

    const size_t tensor_bytes =
        (gate_w->data.size() + up_w->data.size() + down_w->data.size() + tokens.size()) * sizeof(float) +
        static_cast<size_t>(batch) * static_cast<size_t>(hidden + width) * sizeof(float);
    const size_t context_size = std::max<size_t>(64 * 1024 * 1024, tensor_bytes * 4 + 1024 * 1024);
    ggml_init_params params{};
    params.mem_size = context_size;
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
        throw std::runtime_error("ggml language prefix MLP graph compute failed");
    }

    out.assign(ggml_get_data_f32(y), ggml_get_data_f32(y) + static_cast<size_t>(batch) * static_cast<size_t>(width));
    ggml_free(ctx);
}

const Tensor * Pi0LanguagePrefix::find_tensor(const std::string & name) const {
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

std::string Pi0LanguagePrefix::layer_prefix(int layer) const {
    if (layer < 0 || layer >= config_.openpi_language_layers) {
        throw std::invalid_argument("language prefix layer index is out of range");
    }
    return "paligemma_with_expert.paligemma.model.language_model.layers." + std::to_string(layer) + ".";
}

} // namespace vlacpp
