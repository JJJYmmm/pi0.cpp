#include "models/model.h"

#include "core/error.h"
#include "models/pi0_action_decoder.h"
#include "models/pi0_vlm.h"
#include "sampling/flow.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace vlacpp {
namespace {

class Pi0Model final : public Model {
public:
    Pi0Model(ModelConfig config, BackendConfig backend, TensorMap tensors)
        : config_(std::move(config)),
          backend_(backend),
          tensors_(std::move(tensors)),
          vlm_(config_, tensors_),
          action_decoder_(config_, backend_, tensors_) {}

    const ModelConfig & config() const override {
        return config_;
    }

    const char * capability() const override {
        if (action_decoder_.has_pi05_action_head()) {
            return "restricted-pi05-action-head";
        }
        if (action_decoder_.has_pi0_action_head()) {
            if (vlm_.has_vision_projector()) {
                return "restricted-pi0-action-projector";
            }
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

        vlm_.prefill_prefix(cache, observation);
        const Pi0VlmSignals signals = vlm_.encode(observation);
        const Tensor * velocity_weight = find_tensor("pi0.velocity.weight");
        const Tensor * time_weight = find_tensor("pi0.velocity.time_weight");
        const bool has_action_decoder =
            action_decoder_.has_pi0_action_head() || action_decoder_.has_pi05_action_head();
        std::vector<float> state_context;
        if (has_action_decoder) {
            action_decoder_.state_context(observation.state, state_context);
        }

        sample_flow_euler(
            runtime.flow_steps,
            runtime.seed,
            config_.action_horizon,
            config_.action_dim,
            [&](float time, const std::vector<float> & x, std::vector<float> & v) {
                if (has_action_decoder) {
                    std::vector<float> action_velocity;
                    action_decoder_.velocity_batch(
                        time,
                        x,
                        state_context,
                        cache.prefix_layers,
                        cache.token_count,
                        action_velocity);
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
                            feature_dim == signals.features.size() ? signals.features.data() : signals.legacy_features;
                        for (size_t feature = 0; feature < feature_dim; ++feature) {
                            target += velocity_weight->data[row + feature] * active_features[feature];
                        }
                        if (time_weight != nullptr) {
                            time_scale = time_weight->data[static_cast<size_t>(action_col)];
                        }
                    } else {
                        const float phase = static_cast<float>(action_col + 1) / static_cast<float>(config_.action_dim);
                        target = signals.target_base + phase * 0.01f;
                    }
                    v[i] = x[i] - target + time * time_scale;
                }
            },
            out_actions);

        apply_action_denorm(out_actions);
        return VLACPP_STATUS_OK;
    }

private:
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

    ModelConfig config_;
    BackendConfig backend_;
    TensorMap tensors_;
    Pi0Vlm vlm_;
    Pi0ActionDecoder action_decoder_;
};

} // namespace

std::unique_ptr<Model> make_pi0_model(ModelConfig config, BackendConfig backend, TensorMap tensors) {
    return std::unique_ptr<Model>(new Pi0Model(std::move(config), backend, std::move(tensors)));
}

} // namespace vlacpp
