#include "models/pi0_action_expert.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

float gelu(float x) {
    constexpr float sqrt_2_over_pi = 0.79788456080286535587989211986876f;
    constexpr float coef = 0.044715f;
    return 0.5f * x * (1.0f + std::tanh(sqrt_2_over_pi * x * (1.0f + coef * x * x)));
}

std::vector<float> linear(
    const std::vector<float> & weight,
    const std::vector<float> & input,
    int batch,
    int in,
    int out) {
    std::vector<float> result(static_cast<size_t>(batch) * static_cast<size_t>(out), 0.0f);
    for (int row = 0; row < batch; ++row) {
        for (int col = 0; col < out; ++col) {
            float value = 0.0f;
            for (int i = 0; i < in; ++i) {
                value += weight[static_cast<size_t>(col) * static_cast<size_t>(in) + static_cast<size_t>(i)] *
                    input[static_cast<size_t>(row) * static_cast<size_t>(in) + static_cast<size_t>(i)];
            }
            result[static_cast<size_t>(row) * static_cast<size_t>(out) + static_cast<size_t>(col)] = value;
        }
    }
    return result;
}

std::vector<float> rms_norm(
    const std::vector<float> & weight,
    const std::vector<float> & input,
    int batch,
    int width) {
    std::vector<float> result(input.size(), 0.0f);
    for (int row = 0; row < batch; ++row) {
        float sum = 0.0f;
        for (int col = 0; col < width; ++col) {
            const float value = input[static_cast<size_t>(row) * static_cast<size_t>(width) + static_cast<size_t>(col)];
            sum += value * value;
        }
        const float inv = 1.0f / std::sqrt(sum / static_cast<float>(width) + 1.0e-6f);
        for (int col = 0; col < width; ++col) {
            const size_t index = static_cast<size_t>(row) * static_cast<size_t>(width) + static_cast<size_t>(col);
            result[index] = input[index] * inv * (1.0f + weight[static_cast<size_t>(col)]);
        }
    }
    return result;
}

std::vector<float> self_attention(
    const std::vector<float> & q,
    const std::vector<float> & k,
    const std::vector<float> & v,
    int tokens,
    int heads,
    int head_dim) {
    std::vector<float> result(q.size(), 0.0f);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    for (int head = 0; head < heads; ++head) {
        for (int tq = 0; tq < tokens; ++tq) {
            std::vector<float> scores(static_cast<size_t>(tokens), 0.0f);
            float max_score = -INFINITY;
            for (int tk = 0; tk < tokens; ++tk) {
                float score = 0.0f;
                for (int dim = 0; dim < head_dim; ++dim) {
                    const size_t q_index =
                        (static_cast<size_t>(tq) * static_cast<size_t>(heads) + static_cast<size_t>(head)) *
                            static_cast<size_t>(head_dim) +
                        static_cast<size_t>(dim);
                    const size_t k_index =
                        (static_cast<size_t>(tk) * static_cast<size_t>(heads) + static_cast<size_t>(head)) *
                            static_cast<size_t>(head_dim) +
                        static_cast<size_t>(dim);
                    score += q[q_index] * k[k_index];
                }
                scores[static_cast<size_t>(tk)] = score * scale;
                max_score = std::max(max_score, scores[static_cast<size_t>(tk)]);
            }
            float denom = 0.0f;
            for (float & score : scores) {
                score = std::exp(score - max_score);
                denom += score;
            }
            for (int dim = 0; dim < head_dim; ++dim) {
                float value = 0.0f;
                for (int tk = 0; tk < tokens; ++tk) {
                    const size_t v_index =
                        (static_cast<size_t>(tk) * static_cast<size_t>(heads) + static_cast<size_t>(head)) *
                            static_cast<size_t>(head_dim) +
                        static_cast<size_t>(dim);
                    value += scores[static_cast<size_t>(tk)] / denom * v[v_index];
                }
                const size_t out_index =
                    (static_cast<size_t>(tq) * static_cast<size_t>(heads) + static_cast<size_t>(head)) *
                        static_cast<size_t>(head_dim) +
                    static_cast<size_t>(dim);
                result[out_index] = value;
            }
        }
    }
    return result;
}

void require_close(const std::vector<float> & actual, const std::vector<float> & expected) {
    if (actual.size() != expected.size()) {
        std::cerr << "size mismatch\n";
        std::exit(1);
    }
    float max_abs = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        max_abs = std::max(max_abs, std::fabs(actual[i] - expected[i]));
    }
    if (max_abs > 3.0e-5f) {
        std::cerr << "action expert MLP mismatch: max_abs=" << max_abs << "\n";
        std::exit(1);
    }
}

vlacpp::Tensor tensor(std::vector<int64_t> shape, std::vector<float> data) {
    vlacpp::Tensor result;
    result.shape = std::move(shape);
    result.data = std::move(data);
    return result;
}

} // namespace

int main() {
    vlacpp::ModelConfig config;
    config.openpi_action_expert_width = 2;
    config.openpi_action_expert_q_out = 4;
    config.openpi_action_expert_kv_out = 2;
    config.openpi_action_expert_mlp_width = 3;
    config.openpi_action_expert_layers = 1;
    vlacpp::BackendConfig backend;
    backend.n_threads = 1;
    vlacpp::TensorMap tensors;
    const std::string layer_prefix = "model.paligemma_with_expert.gemma_expert.model.layers.0.";
    const std::string mlp_prefix = layer_prefix + "mlp.";
    tensors[layer_prefix + "input_layernorm.weight"] = tensor({2}, {-0.1f, 0.25f});
    tensors[layer_prefix + "post_attention_layernorm.weight"] = tensor({2}, {0.05f, -0.2f});
    tensors[layer_prefix + "self_attn.q_proj.weight"] =
        tensor({2, 4}, {0.1f, -0.2f, 0.3f, 0.4f, -0.5f, 0.6f, 0.7f, -0.8f});
    tensors[layer_prefix + "self_attn.k_proj.weight"] = tensor({2, 2}, {0.2f, 0.1f, -0.3f, 0.5f});
    tensors[layer_prefix + "self_attn.v_proj.weight"] = tensor({2, 2}, {-0.4f, 0.6f, 0.8f, -0.2f});
    tensors[layer_prefix + "self_attn.o_proj.weight"] =
        tensor({4, 2}, {0.2f, -0.1f, 0.3f, 0.4f, -0.2f, 0.5f, 0.6f, -0.7f});
    tensors[mlp_prefix + "gate_proj.weight"] = tensor({2, 3}, {0.2f, -0.1f, -0.3f, 0.4f, 0.1f, 0.5f});
    tensors[mlp_prefix + "up_proj.weight"] = tensor({2, 3}, {0.6f, 0.2f, -0.2f, 0.3f, 0.4f, -0.5f});
    tensors[mlp_prefix + "down_proj.weight"] = tensor({3, 2}, {0.3f, -0.2f, 0.1f, -0.4f, 0.2f, 0.5f});

    const std::vector<float> input = {1.0f, -2.0f, 0.5f, 0.25f};
    vlacpp::Pi0ActionExpert expert(config, backend, tensors);
    if (!expert.has_layer(0)) {
        std::cerr << "expected action expert layer\n";
        return 1;
    }
    std::vector<float> actual;
    expert.mlp_batch(0, input, 2, actual);

    std::vector<float> norm_actual;
    expert.input_norm_batch(0, input, 2, norm_actual);
    require_close(norm_actual, rms_norm(tensors[layer_prefix + "input_layernorm.weight"].data, input, 2, 2));

    expert.post_attention_norm_batch(0, input, 2, norm_actual);
    require_close(norm_actual, rms_norm(tensors[layer_prefix + "post_attention_layernorm.weight"].data, input, 2, 2));

    std::vector<float> q;
    std::vector<float> k;
    std::vector<float> v;
    expert.qkv_batch(0, input, 2, q, k, v);
    require_close(q, linear(tensors[layer_prefix + "self_attn.q_proj.weight"].data, input, 2, 2, 4));
    require_close(k, linear(tensors[layer_prefix + "self_attn.k_proj.weight"].data, input, 2, 2, 2));
    require_close(v, linear(tensors[layer_prefix + "self_attn.v_proj.weight"].data, input, 2, 2, 2));

    const std::vector<float> attention_q = {
        0.2f, -0.1f, 0.5f, 0.3f,
        -0.4f, 0.6f, 0.1f, -0.2f,
        0.7f, 0.0f, -0.3f, 0.4f,
    };
    const std::vector<float> attention_k = {
        -0.2f, 0.4f, 0.3f, -0.5f,
        0.6f, -0.1f, 0.2f, 0.7f,
        -0.3f, 0.8f, -0.4f, 0.1f,
    };
    const std::vector<float> attention_v = {
        0.9f, -0.6f, 0.2f, 0.0f,
        -0.5f, 0.3f, 0.8f, -0.7f,
        0.1f, 0.4f, -0.2f, 0.5f,
    };
    std::vector<float> attention_core;
    expert.self_attention_batch(attention_q, attention_k, attention_v, 3, 2, 2, attention_core);
    require_close(attention_core, self_attention(attention_q, attention_k, attention_v, 3, 2, 2));

    std::vector<float> attention_out;
    const std::vector<float> attention_values = {0.25f, -0.5f, 0.75f, 1.0f, -1.0f, 0.5f, 0.0f, 0.2f};
    expert.attention_out_batch(0, attention_values, 2, attention_out);
    require_close(
        attention_out,
        linear(tensors[layer_prefix + "self_attn.o_proj.weight"].data, attention_values, 2, 4, 2));

    std::vector<float> gate = linear(tensors[mlp_prefix + "gate_proj.weight"].data, input, 2, 2, 3);
    std::vector<float> up = linear(tensors[mlp_prefix + "up_proj.weight"].data, input, 2, 2, 3);
    for (size_t i = 0; i < gate.size(); ++i) {
        gate[i] = gelu(gate[i]) * up[i];
    }
    const std::vector<float> expected = linear(tensors[mlp_prefix + "down_proj.weight"].data, gate, 2, 3, 2);
    require_close(actual, expected);
    return 0;
}
