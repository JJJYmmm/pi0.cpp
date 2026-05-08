#pragma once

#include "models/model.h"

namespace vlacpp {

class Pi0ActionExpert {
public:
    Pi0ActionExpert(const ModelConfig & config, const BackendConfig & backend, const TensorMap & tensors);

    bool has_layer(int layer) const;
    void input_norm_batch(int layer, const std::vector<float> & tokens, int batch, std::vector<float> & out) const;
    void post_attention_norm_batch(
        int layer,
        const std::vector<float> & tokens,
        int batch,
        std::vector<float> & out) const;
    void qkv_batch(
        int layer,
        const std::vector<float> & tokens,
        int batch,
        std::vector<float> & q,
        std::vector<float> & k,
        std::vector<float> & v) const;
    void self_attention_batch(
        const std::vector<float> & q,
        const std::vector<float> & k,
        const std::vector<float> & v,
        int tokens,
        int heads,
        int kv_heads,
        int head_dim,
        std::vector<float> & out) const;
    void attention_out_batch(int layer, const std::vector<float> & values, int batch, std::vector<float> & out) const;
    void mlp_batch(int layer, const std::vector<float> & tokens, int batch, std::vector<float> & out) const;

private:
    void norm_batch(
        int layer,
        const char * weight_name,
        const std::vector<float> & tokens,
        int batch,
        std::vector<float> & out) const;
    const Tensor * find_tensor(const std::string & name) const;
    std::string layer_prefix(int layer) const;

    const ModelConfig & config_;
    const BackendConfig & backend_;
    const TensorMap & tensors_;
};

} // namespace vlacpp
