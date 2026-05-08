#pragma once

#include "models/pi0_language_prefix.h"
#include "models/model.h"

namespace vlacpp {

struct Pi0VlmSignals {
    float state_mean = 0.0f;
    float image_mean = 0.0f;
    float prompt_signal = 0.0f;
    std::vector<float> features;
    float legacy_features[4] = {};
    float target_base = 0.0f;
};

class Pi0Vlm {
public:
    Pi0Vlm(const ModelConfig & config, const BackendConfig & backend, const TensorMap & tensors);

    bool has_vision_projector() const;
    bool has_language_prefix() const;
    void project_vision_tokens(
        const std::vector<float> & vision_tokens,
        int token_count,
        std::vector<float> & out) const;
    void prefill_prefix_from_embeddings(
        KvCache & cache,
        const std::vector<float> & embeddings,
        int token_count) const;
    void prefill_prefix(KvCache & cache, const ObservationData & observation) const;
    Pi0VlmSignals encode(const ObservationData & observation) const;

private:
    const Tensor * find_tensor(const std::string & name) const;

    const ModelConfig & config_;
    const BackendConfig & backend_;
    const TensorMap & tensors_;
    Pi0LanguagePrefix language_prefix_;
};

} // namespace vlacpp
