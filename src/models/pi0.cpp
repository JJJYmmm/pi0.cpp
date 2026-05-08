#include "models/model.h"

#include "core/error.h"
#include "sampling/flow.h"

#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace vlacpp {
namespace {

float mean_or_zero(const std::vector<float> & values) {
    if (values.empty()) {
        return 0.0f;
    }
    return std::accumulate(values.begin(), values.end(), 0.0f) / static_cast<float>(values.size());
}

float swish(float x) {
    return x / (1.0f + std::exp(-x));
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
    const int64_t in_dim = weight.shape[0];
    const int64_t out_dim = weight.shape[1];
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

void ggml_linear(
    const Tensor & weight,
    const Tensor & bias,
    const std::vector<float> & input,
    std::vector<float> & output,
    int n_threads) {
    ggml_linear_batch(weight, bias, input, 1, output, n_threads);
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

class Pi0Model final : public Model {
public:
    Pi0Model(ModelConfig config, BackendConfig backend, TensorMap tensors)
        : config_(std::move(config)), backend_(backend), tensors_(std::move(tensors)) {}

    const ModelConfig & config() const override {
        return config_;
    }

    const char * capability() const override {
        if (find_tensor("vlacpp.openpi.pi05.time_mlp_in.weight") != nullptr) {
            return "restricted-pi05-action-head";
        }
        if (find_tensor("vlacpp.openpi.action_in_proj.weight") != nullptr) {
            return "restricted-pi0-state-action-head";
        }
        if (find_tensor("pi0.velocity.weight") != nullptr) {
            return "tiny-velocity";
        }
        return "mock-pi0";
    }

    vlacpp_status reset_cache(KvCache & cache) override {
        cache.reset();
        return VLACPP_STATUS_OK;
    }

    vlacpp_status infer(
        KvCache & cache,
        const RuntimeConfig & runtime,
        const ObservationData & observation,
        std::vector<float> & out_actions) override {
        if (backend_.backend == VLACPP_BACKEND_CUDA) {
#if !defined(VLACPP_HAVE_GGML)
            return fail(VLACPP_STATUS_UNSUPPORTED, "CUDA backend requires a llama.cpp/ggml build");
#endif
        }

        prefill_prefix(cache, observation);

        const float state_mean = mean_or_zero(observation.state);
        float image_mean = 0.0f;
        size_t image_values = 0;
        for (const auto & image : observation.images) {
            image_mean += std::accumulate(image.data.begin(), image.data.end(), 0.0f);
            image_values += image.data.size();
        }
        if (image_values > 0) {
            image_mean /= static_cast<float>(image_values);
        }
        const float prompt_signal = static_cast<float>(observation.prompt.size() % 97) / 97.0f;
        const Tensor * velocity_weight = find_tensor("pi0.velocity.weight");
        const Tensor * time_weight = find_tensor("pi0.velocity.time_weight");
        const bool has_pi05_action_head = find_tensor("vlacpp.openpi.pi05.time_mlp_in.weight") != nullptr;
        const bool has_action_head =
            find_tensor("vlacpp.openpi.action_in_proj.weight") != nullptr && !has_pi05_action_head;
        std::vector<float> state_context;
        if (find_tensor("vlacpp.openpi.state_proj.weight") != nullptr) {
            action_head_state_context(observation.state, state_context);
        }

        std::vector<float> features;
        features.reserve(observation.state.size() + 3);
        features.push_back(1.0f);
        features.insert(features.end(), observation.state.begin(), observation.state.end());
        features.push_back(image_mean);
        features.push_back(prompt_signal);
        const float legacy_features[4] = {1.0f, state_mean, image_mean, prompt_signal};
        const float target_base = 0.5f * state_mean + 0.25f * image_mean + 0.25f * prompt_signal;

        sample_flow_euler(
            runtime.flow_steps,
            runtime.seed,
            config_.action_horizon,
            config_.action_dim,
            [&](float time, const std::vector<float> & x, std::vector<float> & v) {
                if (has_action_head || has_pi05_action_head) {
                    std::vector<float> action_velocity;
                    if (has_pi05_action_head) {
                        pi05_action_head_velocity_batch(time, x, action_velocity);
                    } else {
                        action_head_velocity_batch(time, x, state_context, action_velocity);
                    }
                    std::copy(action_velocity.begin(), action_velocity.end(), v.begin());
                    return;
                }
                for (size_t i = 0; i < x.size(); ++i) {
                    const int action_col = static_cast<int>(i % static_cast<size_t>(config_.action_dim));
                    float target = 0.0f;
                    float time_scale = 0.001f;
                    if (velocity_weight != nullptr) {
                        const size_t feature_dim = static_cast<size_t>(velocity_weight->shape[0]);
                        const size_t row = static_cast<size_t>(action_col) * feature_dim;
                        const float * active_features =
                            feature_dim == features.size() ? features.data() : legacy_features;
                        for (size_t feature = 0; feature < feature_dim; ++feature) {
                            target += velocity_weight->data[row + feature] * active_features[feature];
                        }
                        if (time_weight != nullptr) {
                            time_scale = time_weight->data[static_cast<size_t>(action_col)];
                        }
                    } else {
                        const float phase = static_cast<float>(action_col + 1) / static_cast<float>(config_.action_dim);
                        target = target_base + phase * 0.01f;
                    }
                    v[i] = x[i] - target + time * time_scale;
                }
            },
            out_actions);

        apply_action_denorm(out_actions);
        return VLACPP_STATUS_OK;
    }

private:
    void prefill_prefix(KvCache & cache, const ObservationData & observation) const {
        if (cache.prefix_valid) {
            return;
        }
        size_t image_tokens = 0;
        for (const auto & image : observation.images) {
            image_tokens += static_cast<size_t>((image.width / 14) * (image.height / 14));
        }
        const size_t prompt_tokens = std::min(
            static_cast<size_t>(config_.max_token_len),
            observation.prompt.empty() ? size_t{1} : observation.prompt.size() / 4 + 1);
        cache.token_count = image_tokens + prompt_tokens;
        cache.prefix_valid = true;
    }

    void apply_action_denorm(std::vector<float> & actions) const {
        if (config_.action_mean.size() != static_cast<size_t>(config_.action_dim) ||
            config_.action_std.size() != static_cast<size_t>(config_.action_dim)) {
            return;
        }
        for (size_t i = 0; i < actions.size(); ++i) {
            const size_t col = i % static_cast<size_t>(config_.action_dim);
            actions[i] = actions[i] * config_.action_std[col] + config_.action_mean[col];
        }
    }

    const Tensor * find_tensor(const std::string & name) const {
        auto it = tensors_.find(name);
        if (it == tensors_.end()) {
            return nullptr;
        }
        return &it->second;
    }

    void action_head_state_context(const std::vector<float> & state, std::vector<float> & out) const {
        const Tensor & state_w = *find_tensor("vlacpp.openpi.state_proj.weight");
        const Tensor & state_b = *find_tensor("vlacpp.openpi.state_proj.bias");
        ggml_linear(state_w, state_b, state, out, backend_.n_threads);
    }

    void action_head_velocity(
        float time,
        const std::vector<float> & action,
        const std::vector<float> & state_context,
        std::vector<float> & out) const {
        const Tensor & in_w = *find_tensor("vlacpp.openpi.action_in_proj.weight");
        const Tensor & in_b = *find_tensor("vlacpp.openpi.action_in_proj.bias");
        const Tensor & time_in_w = *find_tensor("vlacpp.openpi.action_time_mlp_in.weight");
        const Tensor & time_in_b = *find_tensor("vlacpp.openpi.action_time_mlp_in.bias");
        const Tensor & time_out_w = *find_tensor("vlacpp.openpi.action_time_mlp_out.weight");
        const Tensor & time_out_b = *find_tensor("vlacpp.openpi.action_time_mlp_out.bias");
        const Tensor & out_w = *find_tensor("vlacpp.openpi.action_out_proj.weight");
        const Tensor & out_b = *find_tensor("vlacpp.openpi.action_out_proj.bias");

        std::vector<float> action_token;
        ggml_linear(in_w, in_b, action, action_token, backend_.n_threads);
        if (!state_context.empty()) {
            for (size_t i = 0; i < action_token.size(); ++i) {
                action_token[i] += state_context[i];
            }
        }
        std::vector<float> time_emb = posemb_sincos(time, action_token.size());
        std::vector<float> action_time;
        action_time.reserve(action_token.size() + time_emb.size());
        action_time.insert(action_time.end(), action_token.begin(), action_token.end());
        action_time.insert(action_time.end(), time_emb.begin(), time_emb.end());

        std::vector<float> hidden;
        ggml_linear(time_in_w, time_in_b, action_time, hidden, backend_.n_threads);
        for (float & value : hidden) {
            value = swish(value);
        }
        std::vector<float> mixed;
        ggml_linear(time_out_w, time_out_b, hidden, mixed, backend_.n_threads);
        for (float & value : mixed) {
            value = swish(value);
        }
        ggml_linear(out_w, out_b, mixed, out, backend_.n_threads);
    }

    void action_head_velocity_batch(
        float time,
        const std::vector<float> & actions,
        const std::vector<float> & state_context,
        std::vector<float> & out) const {
        const Tensor & in_w = *find_tensor("vlacpp.openpi.action_in_proj.weight");
        const Tensor & in_b = *find_tensor("vlacpp.openpi.action_in_proj.bias");
        const Tensor & time_in_w = *find_tensor("vlacpp.openpi.action_time_mlp_in.weight");
        const Tensor & time_in_b = *find_tensor("vlacpp.openpi.action_time_mlp_in.bias");
        const Tensor & time_out_w = *find_tensor("vlacpp.openpi.action_time_mlp_out.weight");
        const Tensor & time_out_b = *find_tensor("vlacpp.openpi.action_time_mlp_out.bias");
        const Tensor & out_w = *find_tensor("vlacpp.openpi.action_out_proj.weight");
        const Tensor & out_b = *find_tensor("vlacpp.openpi.action_out_proj.bias");

        const int batch = config_.action_horizon;
        const size_t width = static_cast<size_t>(in_w.shape[1]);
        std::vector<float> action_tokens;
        ggml_linear_batch(in_w, in_b, actions, batch, action_tokens, backend_.n_threads);
        if (!state_context.empty()) {
            for (int row = 0; row < batch; ++row) {
                const size_t offset = static_cast<size_t>(row) * width;
                for (size_t i = 0; i < width; ++i) {
                    action_tokens[offset + i] += state_context[i];
                }
            }
        }

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
        ggml_linear_batch(time_in_w, time_in_b, action_time, batch, hidden, backend_.n_threads);
        for (float & value : hidden) {
            value = swish(value);
        }
        std::vector<float> mixed;
        ggml_linear_batch(time_out_w, time_out_b, hidden, batch, mixed, backend_.n_threads);
        for (float & value : mixed) {
            value = swish(value);
        }
        ggml_linear_batch(out_w, out_b, mixed, batch, out, backend_.n_threads);
    }

    void pi05_action_head_velocity(float time, const std::vector<float> & action, std::vector<float> & out) const {
        const Tensor & in_w = *find_tensor("vlacpp.openpi.action_in_proj.weight");
        const Tensor & in_b = *find_tensor("vlacpp.openpi.action_in_proj.bias");
        const Tensor & time_in_w = *find_tensor("vlacpp.openpi.pi05.time_mlp_in.weight");
        const Tensor & time_in_b = *find_tensor("vlacpp.openpi.pi05.time_mlp_in.bias");
        const Tensor & time_out_w = *find_tensor("vlacpp.openpi.pi05.time_mlp_out.weight");
        const Tensor & time_out_b = *find_tensor("vlacpp.openpi.pi05.time_mlp_out.bias");
        const Tensor & out_w = *find_tensor("vlacpp.openpi.action_out_proj.weight");
        const Tensor & out_b = *find_tensor("vlacpp.openpi.action_out_proj.bias");

        std::vector<float> action_token;
        ggml_linear(in_w, in_b, action, action_token, backend_.n_threads);
        std::vector<float> time_emb = posemb_sincos(time, action_token.size());
        std::vector<float> hidden;
        ggml_linear(time_in_w, time_in_b, time_emb, hidden, backend_.n_threads);
        for (float & value : hidden) {
            value = swish(value);
        }
        std::vector<float> time_context;
        ggml_linear(time_out_w, time_out_b, hidden, time_context, backend_.n_threads);
        for (float & value : time_context) {
            value = swish(value);
        }
        for (size_t i = 0; i < action_token.size(); ++i) {
            action_token[i] += time_context[i];
        }
        ggml_linear(out_w, out_b, action_token, out, backend_.n_threads);
    }

    void pi05_action_head_velocity_batch(float time, const std::vector<float> & actions, std::vector<float> & out) const {
        const Tensor & in_w = *find_tensor("vlacpp.openpi.action_in_proj.weight");
        const Tensor & in_b = *find_tensor("vlacpp.openpi.action_in_proj.bias");
        const Tensor & time_in_w = *find_tensor("vlacpp.openpi.pi05.time_mlp_in.weight");
        const Tensor & time_in_b = *find_tensor("vlacpp.openpi.pi05.time_mlp_in.bias");
        const Tensor & time_out_w = *find_tensor("vlacpp.openpi.pi05.time_mlp_out.weight");
        const Tensor & time_out_b = *find_tensor("vlacpp.openpi.pi05.time_mlp_out.bias");
        const Tensor & out_w = *find_tensor("vlacpp.openpi.action_out_proj.weight");
        const Tensor & out_b = *find_tensor("vlacpp.openpi.action_out_proj.bias");

        const int batch = config_.action_horizon;
        const size_t width = static_cast<size_t>(in_w.shape[1]);
        std::vector<float> action_tokens;
        ggml_linear_batch(in_w, in_b, actions, batch, action_tokens, backend_.n_threads);
        std::vector<float> time_emb = posemb_sincos(time, width);
        std::vector<float> hidden;
        ggml_linear(time_in_w, time_in_b, time_emb, hidden, backend_.n_threads);
        for (float & value : hidden) {
            value = swish(value);
        }
        std::vector<float> time_context;
        ggml_linear(time_out_w, time_out_b, hidden, time_context, backend_.n_threads);
        for (float & value : time_context) {
            value = swish(value);
        }
        for (int row = 0; row < batch; ++row) {
            const size_t offset = static_cast<size_t>(row) * width;
            for (size_t i = 0; i < width; ++i) {
                action_tokens[offset + i] += time_context[i];
            }
        }
        ggml_linear_batch(out_w, out_b, action_tokens, batch, out, backend_.n_threads);
    }

    ModelConfig config_;
    BackendConfig backend_;
    TensorMap tensors_;
};

} // namespace

std::unique_ptr<Model> make_pi0_model(ModelConfig config, BackendConfig backend, TensorMap tensors) {
    return std::unique_ptr<Model>(new Pi0Model(std::move(config), backend, std::move(tensors)));
}

} // namespace vlacpp
