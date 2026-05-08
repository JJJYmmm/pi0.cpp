#include "models/model.h"

#include "core/error.h"
#include "core/gguf.h"
#include "core/json.h"

#include <fstream>
#include <memory>
#include <string>

namespace vlacpp {
namespace {

bool has_tiny_velocity_tensors(const ModelConfig & config, const TensorMap & tensors) {
    auto weight = tensors.find("pi0.velocity.weight");
    auto time_weight = tensors.find("pi0.velocity.time_weight");
    if (weight == tensors.end() || time_weight == tensors.end()) {
        return false;
    }
    const int64_t feature_dim = static_cast<int64_t>(config.state_dim) + 3;
    return weight->second.shape.size() == 2 &&
        weight->second.shape[1] == static_cast<int64_t>(config.action_dim) &&
        (weight->second.shape[0] == 4 || weight->second.shape[0] == feature_dim) &&
        weight->second.data.size() ==
            static_cast<size_t>(config.action_dim) * static_cast<size_t>(weight->second.shape[0]) &&
        time_weight->second.shape.size() == 1 &&
        time_weight->second.shape[0] == static_cast<int64_t>(config.action_dim) &&
        time_weight->second.data.size() == static_cast<size_t>(config.action_dim);
}

bool has_action_head_tensors(const ModelConfig & config, const TensorMap & tensors) {
    const char * required[] = {
        "vlacpp.openpi.action_in_proj.weight",
        "vlacpp.openpi.action_in_proj.bias",
        "vlacpp.openpi.action_time_mlp_in.weight",
        "vlacpp.openpi.action_time_mlp_in.bias",
        "vlacpp.openpi.action_time_mlp_out.weight",
        "vlacpp.openpi.action_time_mlp_out.bias",
        "vlacpp.openpi.action_out_proj.weight",
        "vlacpp.openpi.action_out_proj.bias",
    };
    for (const char * name : required) {
        if (tensors.find(name) == tensors.end()) {
            return false;
        }
    }

    const Tensor & in_w = tensors.at("vlacpp.openpi.action_in_proj.weight");
    const Tensor & in_b = tensors.at("vlacpp.openpi.action_in_proj.bias");
    const Tensor & time_in_w = tensors.at("vlacpp.openpi.action_time_mlp_in.weight");
    const Tensor & time_in_b = tensors.at("vlacpp.openpi.action_time_mlp_in.bias");
    const Tensor & time_out_w = tensors.at("vlacpp.openpi.action_time_mlp_out.weight");
    const Tensor & time_out_b = tensors.at("vlacpp.openpi.action_time_mlp_out.bias");
    const Tensor & out_w = tensors.at("vlacpp.openpi.action_out_proj.weight");
    const Tensor & out_b = tensors.at("vlacpp.openpi.action_out_proj.bias");

    if (in_w.shape.size() != 2 || in_w.shape[0] != config.action_dim || in_b.shape.size() != 1) {
        return false;
    }
    const int64_t width = in_w.shape[1];
    return in_b.shape[0] == width &&
        time_in_w.shape.size() == 2 && time_in_w.shape[0] == 2 * width && time_in_w.shape[1] == width &&
        time_in_b.shape.size() == 1 && time_in_b.shape[0] == width &&
        time_out_w.shape.size() == 2 && time_out_w.shape[0] == width && time_out_w.shape[1] == width &&
        time_out_b.shape.size() == 1 && time_out_b.shape[0] == width &&
        out_w.shape.size() == 2 && out_w.shape[0] == width && out_w.shape[1] == config.action_dim &&
        out_b.shape.size() == 1 && out_b.shape[0] == config.action_dim;
}

bool has_pi05_action_head_tensors(const ModelConfig & config, const TensorMap & tensors) {
    const char * required[] = {
        "vlacpp.openpi.action_in_proj.weight",
        "vlacpp.openpi.action_in_proj.bias",
        "vlacpp.openpi.pi05.time_mlp_in.weight",
        "vlacpp.openpi.pi05.time_mlp_in.bias",
        "vlacpp.openpi.pi05.time_mlp_out.weight",
        "vlacpp.openpi.pi05.time_mlp_out.bias",
        "vlacpp.openpi.action_out_proj.weight",
        "vlacpp.openpi.action_out_proj.bias",
    };
    for (const char * name : required) {
        if (tensors.find(name) == tensors.end()) {
            return false;
        }
    }
    const Tensor & in_w = tensors.at("vlacpp.openpi.action_in_proj.weight");
    const Tensor & in_b = tensors.at("vlacpp.openpi.action_in_proj.bias");
    const Tensor & time_in_w = tensors.at("vlacpp.openpi.pi05.time_mlp_in.weight");
    const Tensor & time_in_b = tensors.at("vlacpp.openpi.pi05.time_mlp_in.bias");
    const Tensor & time_out_w = tensors.at("vlacpp.openpi.pi05.time_mlp_out.weight");
    const Tensor & time_out_b = tensors.at("vlacpp.openpi.pi05.time_mlp_out.bias");
    const Tensor & out_w = tensors.at("vlacpp.openpi.action_out_proj.weight");
    const Tensor & out_b = tensors.at("vlacpp.openpi.action_out_proj.bias");

    if (in_w.shape.size() != 2 || in_w.shape[0] != config.action_dim || in_b.shape.size() != 1) {
        return false;
    }
    const int64_t width = in_w.shape[1];
    return in_b.shape[0] == width &&
        time_in_w.shape.size() == 2 && time_in_w.shape[0] == width && time_in_w.shape[1] == width &&
        time_in_b.shape.size() == 1 && time_in_b.shape[0] == width &&
        time_out_w.shape.size() == 2 && time_out_w.shape[0] == width && time_out_w.shape[1] == width &&
        time_out_b.shape.size() == 1 && time_out_b.shape[0] == width &&
        out_w.shape.size() == 2 && out_w.shape[0] == width && out_w.shape[1] == config.action_dim &&
        out_b.shape.size() == 1 && out_b.shape[0] == config.action_dim;
}

vlacpp_status validate_pi0_tensors(const ModelConfig & config, const TensorMap & tensors) {
    if (config.model_type == "mock-pi0") {
        return VLACPP_STATUS_OK;
    }

    if (has_tiny_velocity_tensors(config, tensors) ||
        has_action_head_tensors(config, tensors) ||
        has_pi05_action_head_tensors(config, tensors)) {
        return VLACPP_STATUS_OK;
    }
    return fail(
        VLACPP_STATUS_PARSE_ERROR,
        "pi0/pi05 model requires tiny velocity tensors or mapped OpenPI action-head tensors");
}

} // namespace

std::unique_ptr<Model> make_pi0_model(ModelConfig config, BackendConfig backend, TensorMap tensors);

vlacpp_status load_model_from_path(
    const std::string & path,
    const BackendConfig & backend,
    std::unique_ptr<Model> & out) {
    ModelConfig config;
    TensorMap tensors;
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return fail(VLACPP_STATUS_IO_ERROR, "failed to open model file: " + path);
    }
    char magic[4] = {};
    file.read(magic, sizeof(magic));
    const bool is_gguf = file && magic[0] == 'G' && magic[1] == 'G' && magic[2] == 'U' && magic[3] == 'F';

    vlacpp_status status = is_gguf ? load_gguf_model_file(path, config, tensors) : load_config_file(path, config);
    if (status != VLACPP_STATUS_OK) {
        return status;
    }

    if (config.model_type == "mock-pi0" || config.model_type == "pi0" || config.model_type == "pi05") {
        status = validate_pi0_tensors(config, tensors);
        if (status != VLACPP_STATUS_OK) {
            return status;
        }
        out = make_pi0_model(std::move(config), backend, std::move(tensors));
        return VLACPP_STATUS_OK;
    }

    return fail(VLACPP_STATUS_UNSUPPORTED, "unsupported model_type: " + config.model_type);
}

} // namespace vlacpp
