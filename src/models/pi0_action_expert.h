#pragma once

#include "models/model.h"

namespace vlacpp {

class Pi0ActionExpert {
public:
    Pi0ActionExpert(const ModelConfig & config, const BackendConfig & backend, const TensorMap & tensors);

    bool has_layer(int layer) const;
    void mlp_batch(int layer, const std::vector<float> & tokens, int batch, std::vector<float> & out) const;

private:
    const Tensor * find_tensor(const std::string & name) const;
    std::string layer_prefix(int layer) const;

    const ModelConfig & config_;
    const BackendConfig & backend_;
    const TensorMap & tensors_;
};

} // namespace vlacpp
