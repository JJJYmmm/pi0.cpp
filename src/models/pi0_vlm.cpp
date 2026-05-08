#include "models/pi0_vlm.h"

#include "models/ggml_runtime.h"

#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace vlacpp {
namespace {

float mean_or_zero(const std::vector<float> & values) {
    if (values.empty()) {
        return 0.0f;
    }
    return std::accumulate(values.begin(), values.end(), 0.0f) / static_cast<float>(values.size());
}

} // namespace

Pi0Vlm::Pi0Vlm(const ModelConfig & config, const BackendConfig & backend, const TensorMap & tensors)
    : config_(config),
      backend_(backend),
      tensors_(tensors),
      language_prefix_(config, backend, tensors),
      vision_mtmd_(config, backend) {}

bool Pi0Vlm::has_vision_projector() const {
    return find_tensor("vlacpp.openpi.vision_projector.weight") != nullptr &&
        find_tensor("vlacpp.openpi.vision_projector.bias") != nullptr;
}

bool Pi0Vlm::has_mtmd_vision_encoder() const {
    return vision_mtmd_.available() && vision_mtmd_.output_width() == config_.openpi_language_width;
}

bool Pi0Vlm::has_language_prefix() const {
    return config_.openpi_language_layers > 0 &&
        config_.openpi_language_width > 0 &&
        config_.openpi_language_q_out > 0 &&
        config_.openpi_language_kv_out > 0 &&
        language_prefix_.has_layer(0);
}

void Pi0Vlm::project_vision_tokens(
    const std::vector<float> & vision_tokens,
    int token_count,
    std::vector<float> & out) const {
    const Tensor * weight = find_tensor("vlacpp.openpi.vision_projector.weight");
    const Tensor * bias = find_tensor("vlacpp.openpi.vision_projector.bias");
    if (weight == nullptr || bias == nullptr) {
        throw std::invalid_argument("missing pi0 vision projector tensors");
    }
    const int64_t vision_width = config_.openpi_vision_width;
    const int64_t language_width = config_.openpi_language_width;
    if (token_count <= 0 ||
        vision_width <= 0 ||
        language_width <= 0 ||
        weight->shape.size() != 2 ||
        weight->shape[0] != vision_width ||
        weight->shape[1] != language_width ||
        bias->shape.size() != 1 ||
        bias->shape[0] != language_width ||
        vision_tokens.size() != static_cast<size_t>(token_count) * static_cast<size_t>(vision_width) ||
        weight->data.size() != static_cast<size_t>(vision_width) * static_cast<size_t>(language_width) ||
        bias->data.size() != static_cast<size_t>(language_width)) {
        throw std::invalid_argument("pi0 vision projector tensors have incompatible shape");
    }

    const size_t tensor_bytes =
        (weight->data.size() + bias->data.size() + vision_tokens.size()) * sizeof(float) +
        static_cast<size_t>(token_count) * static_cast<size_t>(language_width) * sizeof(float);
    const size_t context_size = std::max<size_t>(64 * 1024 * 1024, tensor_bytes * 4 + 1024 * 1024);
    GgmlRunner runner(backend_);
    ggml_init_params params = runner.init_params(context_size);
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        throw std::runtime_error("failed to initialize ggml context");
    }

    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, vision_width, language_width);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, vision_width, token_count);
    ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, language_width);
    std::vector<GgmlInput> inputs;
    runner.set_input(inputs, w, weight->data.data(), weight->data.size() * sizeof(float));
    runner.set_input(inputs, x, vision_tokens.data(), vision_tokens.size() * sizeof(float));
    runner.set_input(inputs, b, bias->data.data(), bias->data.size() * sizeof(float));

    ggml_tensor * y = ggml_add(ctx, ggml_mul_mat(ctx, w, x), b);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    runner.compute(ctx, graph, inputs, "ggml pi0 vision projector graph compute failed");

    out.resize(static_cast<size_t>(token_count) * static_cast<size_t>(language_width));
    runner.get_output(y, out.data(), out.size() * sizeof(float));
    ggml_free(ctx);
}

void Pi0Vlm::prefill_prefix_from_embeddings(
    KvCache & cache,
    const std::vector<float> & embeddings,
    int token_count) const {
    if (token_count <= 0) {
        cache.reset();
        cache.prefix_valid = true;
        return;
    }
    const int width = config_.openpi_language_width;
    if (!has_language_prefix() ||
        width <= 0 ||
        config_.openpi_language_kv_out <= 0 ||
        config_.openpi_language_q_out % config_.openpi_language_kv_out != 0 ||
        embeddings.size() != static_cast<size_t>(token_count) * static_cast<size_t>(width)) {
        throw std::invalid_argument("pi0 prefix embeddings have incompatible shape");
    }

    const int head_dim = config_.openpi_language_kv_out;
    const int heads = config_.openpi_language_q_out / head_dim;
    const int kv_heads = config_.openpi_language_kv_out / head_dim;
    std::vector<int> positions(static_cast<size_t>(token_count), 0);
    for (int i = 0; i < token_count; ++i) {
        positions[static_cast<size_t>(i)] = i;
    }

    std::vector<float> hidden;
    language_prefix_.prefill_batch(
        embeddings,
        positions,
        token_count,
        heads,
        kv_heads,
        head_dim,
        cache.prefix_layers,
        hidden);
    cache.token_count = static_cast<size_t>(token_count);
    cache.prefix_valid = true;
}

void Pi0Vlm::prefill_prefix(KvCache & cache, const ObservationData & observation) const {
    if (cache.prefix_valid) {
        return;
    }
    if (has_mtmd_vision_encoder() && has_language_prefix() && !observation.images.empty()) {
        std::vector<float> embeddings;
        int token_count = 0;
        if (!vision_mtmd_.encode(observation.images, backend_.n_threads, embeddings, token_count)) {
            throw std::runtime_error("mtmd pi0 vision encode failed");
        }
        prefill_prefix_from_embeddings(cache, embeddings, token_count);
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

Pi0VlmSignals Pi0Vlm::encode(const ObservationData & observation) const {
    Pi0VlmSignals signals;
    signals.state_mean = mean_or_zero(observation.state);
    size_t image_values = 0;
    for (const auto & image : observation.images) {
        signals.image_mean += std::accumulate(image.data.begin(), image.data.end(), 0.0f);
        image_values += image.data.size();
    }
    if (image_values > 0) {
        signals.image_mean /= static_cast<float>(image_values);
    }
    signals.prompt_signal = static_cast<float>(observation.prompt.size() % 97) / 97.0f;
    signals.features.reserve(observation.state.size() + 3);
    signals.features.push_back(1.0f);
    signals.features.insert(signals.features.end(), observation.state.begin(), observation.state.end());
    signals.features.push_back(signals.image_mean);
    signals.features.push_back(signals.prompt_signal);
    signals.legacy_features[0] = 1.0f;
    signals.legacy_features[1] = signals.state_mean;
    signals.legacy_features[2] = signals.image_mean;
    signals.legacy_features[3] = signals.prompt_signal;
    signals.target_base = 0.5f * signals.state_mean + 0.25f * signals.image_mean + 0.25f * signals.prompt_signal;
    return signals;
}

const Tensor * Pi0Vlm::find_tensor(const std::string & name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        return nullptr;
    }
    return &it->second;
}

} // namespace vlacpp
