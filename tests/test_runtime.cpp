#include "vlacpp.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

namespace {

void require_status(vlacpp_status status, const char * what) {
    if (status != VLACPP_STATUS_OK) {
        std::cerr << what << ": " << vlacpp_last_error() << "\n";
        std::exit(1);
    }
}

void require_not_status(vlacpp_status status, const char * what) {
    if (status == VLACPP_STATUS_OK) {
        std::cerr << what << ": expected failure\n";
        std::exit(1);
    }
}

} // namespace

int main() {
    const char * path = "vlacpp-test-model.json";
    {
        std::ofstream file(path);
        file << "{"
             << "\"model_type\":\"mock-pi0\","
             << "\"image_width\":16,"
             << "\"image_height\":16,"
             << "\"state_dim\":3,"
             << "\"action_dim\":2,"
             << "\"action_horizon\":4,"
             << "\"image_keys\":[\"base_0_rgb\"]"
             << "}";
    }

    vlacpp_model * model = nullptr;
    vlacpp_model_params model_params = vlacpp_default_model_params();
    require_status(vlacpp_load_model(path, &model_params, &model), "load model");
    vlacpp_openpi_graph_info graph{};
    require_status(vlacpp_model_openpi_graph_info(model, &graph), "graph info");
    if (graph.action_width != 0 || graph.vision_layers != 0 || graph.action_expert_layers != 0) {
        std::cerr << "mock model should not report openpi graph dimensions\n";
        return 1;
    }

    vlacpp_context * context = nullptr;
    vlacpp_context_params context_params = vlacpp_default_context_params();
    context_params.seed = 7;
    context_params.flow_steps = 4;
    require_status(vlacpp_create_context(model, &context_params, &context), "create context");

    std::vector<unsigned char> image(8 * 8 * 3, 128);
    vlacpp_image_view view{};
    view.name = "base_0_rgb";
    view.data = image.data();
    view.width = 8;
    view.height = 8;
    view.channels = 3;
    view.stride_bytes = 8 * 3;
    float state[3] = {1.0f, 2.0f, 3.0f};

    vlacpp_observation obs{};
    obs.images = &view;
    obs.image_count = 1;
    obs.state = state;
    obs.state_count = 3;
    obs.prompt = "open drawer";

    vlacpp_action_chunk actions{};
    require_status(vlacpp_infer_actions(context, &obs, &actions), "infer actions");

    if (actions.horizon != 4 || actions.action_dim != 2 || actions.data == nullptr) {
        std::cerr << "unexpected action shape\n";
        return 1;
    }
    for (int i = 0; i < actions.horizon * actions.action_dim; ++i) {
        if (actions.data[i] != actions.data[i]) {
            std::cerr << "nan action\n";
            return 1;
        }
    }

    vlacpp_free_action_chunk(&actions);
    require_status(vlacpp_reset_cache(context), "reset cache");
    int32_t invalid_prompt_token = -1;
    obs.prompt_tokens = &invalid_prompt_token;
    obs.prompt_token_count = 1;
    require_not_status(vlacpp_infer_actions(context, &obs, &actions), "infer with negative prompt token");
    vlacpp_free_context(context);
    vlacpp_free_model(model);
    std::remove(path);

    const char * pi0_without_tensors_path = "vlacpp-test-pi0-without-tensors.json";
    {
        std::ofstream file(pi0_without_tensors_path);
        file << "{"
             << "\"model_type\":\"pi0\","
             << "\"image_width\":16,"
             << "\"image_height\":16,"
             << "\"state_dim\":3,"
             << "\"action_dim\":2,"
             << "\"action_horizon\":4,"
             << "\"image_keys\":[\"base_0_rgb\"]"
             << "}";
    }
    vlacpp_model * invalid_model = nullptr;
    require_not_status(
        vlacpp_load_model(pi0_without_tensors_path, &model_params, &invalid_model),
        "load pi0 without tensors");
    if (invalid_model != nullptr) {
        std::cerr << "invalid model should not be returned\n";
        return 1;
    }
    std::remove(pi0_without_tensors_path);
    return 0;
}
