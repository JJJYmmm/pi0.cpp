#pragma once

#include "models/pi0_action_expert.h"
#include "models/model.h"

namespace vlacpp {

class Pi0ActionDecoder {
public:
    Pi0ActionDecoder(const ModelConfig & config, const BackendConfig & backend, const TensorMap & tensors);

    bool has_pi0_action_head() const;
    bool has_pi05_action_head() const;
    bool has_pi0_action_expert() const;
    void state_context(const std::vector<float> & state, std::vector<float> & out) const;
    void suffix_embeddings(
        float time,
        const std::vector<float> & actions,
        const std::vector<float> & state_context,
        std::vector<float> & out) const;
    void velocity_batch(
        float time,
        const std::vector<float> & actions,
        const std::vector<float> & state_context,
        std::vector<float> & out) const;

private:
    const Tensor * find_tensor(const std::string & name) const;

    const ModelConfig & config_;
    const BackendConfig & backend_;
    const TensorMap & tensors_;
    Pi0ActionExpert action_expert_;
};

} // namespace vlacpp
