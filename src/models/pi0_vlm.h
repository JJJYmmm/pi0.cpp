#pragma once

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
    explicit Pi0Vlm(const ModelConfig & config);

    void prefill_prefix(KvCache & cache, const ObservationData & observation) const;
    Pi0VlmSignals encode(const ObservationData & observation) const;

private:
    const ModelConfig & config_;
};

} // namespace vlacpp
