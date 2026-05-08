#include "vlacpp.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

void usage(const char * argv0) {
    std::cerr << "usage: " << argv0
              << " --model model.gguf [--prompt text] [--state v0,v1,...] [--steps n] [--seed n] [--info] [--require-full]\n";
}

std::vector<float> parse_state(const std::string & text) {
    std::vector<float> result;
    size_t begin = 0;
    while (begin < text.size()) {
        size_t end = text.find(',', begin);
        std::string item = text.substr(begin, end == std::string::npos ? end : end - begin);
        result.push_back(std::strtof(item.c_str(), nullptr));
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1;
    }
    return result;
}

} // namespace

int main(int argc, char ** argv) {
    std::string model_path;
    std::string prompt = "pick up the fork";
    std::string state_text;
    int steps = 10;
    uint32_t seed = 1;
    bool info = false;
    bool require_full = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (std::strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (std::strcmp(argv[i], "--state") == 0 && i + 1 < argc) {
            state_text = argv[++i];
        } else if (std::strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            steps = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "--info") == 0) {
            info = true;
        } else if (std::strcmp(argv[i], "--require-full") == 0) {
            require_full = true;
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    if (model_path.empty()) {
        usage(argv[0]);
        return 2;
    }

    vlacpp_model_params model_params = vlacpp_default_model_params();
    vlacpp_model * model = nullptr;
    vlacpp_status status = vlacpp_load_model(model_path.c_str(), &model_params, &model);
    if (status != VLACPP_STATUS_OK) {
        std::cerr << "load failed: " << vlacpp_last_error() << "\n";
        return 1;
    }

    if (info) {
        std::cout << "{\n  \"capability\": \"" << vlacpp_model_capability(model) << "\"\n}\n";
        vlacpp_free_model(model);
        return 0;
    }
    if (require_full && std::strcmp(vlacpp_model_capability(model), "full-openpi") != 0) {
        std::cerr << "model capability is " << vlacpp_model_capability(model)
                  << ", not full-openpi\n";
        vlacpp_free_model(model);
        return 1;
    }

    vlacpp_context_params context_params = vlacpp_default_context_params();
    context_params.flow_steps = steps;
    context_params.seed = seed;
    vlacpp_context * context = nullptr;
    status = vlacpp_create_context(model, &context_params, &context);
    if (status != VLACPP_STATUS_OK) {
        std::cerr << "context failed: " << vlacpp_last_error() << "\n";
        vlacpp_free_model(model);
        return 1;
    }

    std::vector<uint8_t> image(static_cast<size_t>(224) * 224 * 3, 127);
    vlacpp_image_view view;
    view.name = "base_0_rgb";
    view.data = image.data();
    view.width = 224;
    view.height = 224;
    view.channels = 3;
    view.stride_bytes = 224 * 3;

    std::vector<float> state = parse_state(state_text);
    vlacpp_observation obs;
    obs.images = &view;
    obs.image_count = 1;
    obs.state = state.empty() ? nullptr : state.data();
    obs.state_count = state.size();
    obs.prompt = prompt.c_str();

    vlacpp_action_chunk actions{};
    status = vlacpp_infer_actions(context, &obs, &actions);
    if (status != VLACPP_STATUS_OK) {
        std::cerr << "infer failed: " << vlacpp_last_error() << "\n";
        vlacpp_free_context(context);
        vlacpp_free_model(model);
        return 1;
    }

    std::cout << "{\n  \"horizon\": " << actions.horizon
              << ",\n  \"action_dim\": " << actions.action_dim
              << ",\n  \"actions\": [";
    const int n = actions.horizon * actions.action_dim;
    for (int i = 0; i < n; ++i) {
        if (i != 0) {
            std::cout << ", ";
        }
        std::cout << actions.data[i];
    }
    std::cout << "]\n}\n";

    vlacpp_free_action_chunk(&actions);
    vlacpp_free_context(context);
    vlacpp_free_model(model);
    return 0;
}
