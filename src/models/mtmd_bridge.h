#pragma once

#include <string>

namespace vlacpp {

struct MtmdBridgeInfo {
    std::string media_marker;
    bool use_gpu = false;
    bool print_timings = false;
    int n_threads = 0;
    bool warmup = false;
    int image_min_tokens = 0;
    int image_max_tokens = 0;
};

bool mtmd_bridge_available();
MtmdBridgeInfo mtmd_bridge_default_info();

} // namespace vlacpp
