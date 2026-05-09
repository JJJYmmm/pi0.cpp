#include "models/model.h"

#include "core/error.h"
#include "models/pi0/pi0_action_decoder.h"
#include "models/pi0/pi0_vlm.h"
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
          vlm_(config_, backend_, tensors_),
          action_decoder_(config_, backend_, tensors_) {}

    const ModelConfig & config() const override {
        return config_;
    }

    const char * capability() const override {
        if (action_decoder_.has_pi0_action_head()) {
            if (vlm_.has_mtmd_vision_encoder() && vlm_.has_language_prefix()) {
                return "restricted-pi0-mtmd-vlm-action-decoder";
            }
            if (vlm_.has_vision_projector()) {
                return "restricted-pi0-action-projector";
            }
            return "restricted-pi0-state-action-head";
        }
        return "unsupported-pi0";
    }

    vlacpp_status reset_cache(KvCache & cache) override {
        cache.reset();
        return VLACPP_STATUS_OK;
    }

    vlacpp_status infer(
        KvCache & cache,
        RuntimeConfig & runtime,
        const ObservationData & observation,
        std::vector<float> & out_actions) override {
        if (backend_.backend == VLACPP_BACKEND_CUDA) {
#if !defined(VLACPP_HAVE_GGML)
            return fail(VLACPP_STATUS_UNSUPPORTED, "CUDA backend requires a llama.cpp/ggml build");
#endif
        }

        vlm_.prefill_prefix(cache, observation);
        if (!action_decoder_.has_pi0_action_head()) {
            return fail(VLACPP_STATUS_UNSUPPORTED, "pi0 inference requires mapped OpenPI action-head tensors");
        }
        std::vector<float> state_context;
        action_decoder_.state_context(observation.state, state_context);

        sample_flow_euler(
            runtime.flow_steps,
            runtime.rng,
            config_.action_horizon,
            config_.action_dim,
            observation.noise.empty() ? nullptr : &observation.noise,
            [&](float time, const std::vector<float> & x, std::vector<float> & v) {
                std::vector<float> action_velocity;
                action_decoder_.velocity_batch(
                    time,
                    x,
                    state_context,
                    cache.prefix_layers,
                    cache.token_count,
                    action_velocity);
                std::copy(action_velocity.begin(), action_velocity.end(), v.begin());
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
