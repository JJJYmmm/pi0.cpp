#include "models/mtmd_bridge.h"

#include "mtmd.h"

namespace vlacpp {

bool mtmd_bridge_available() {
    const char * marker = mtmd_default_marker();
    return marker != nullptr && marker[0] != '\0';
}

MtmdBridgeInfo mtmd_bridge_default_info() {
    const mtmd_context_params params = mtmd_context_params_default();
    MtmdBridgeInfo info;
    info.media_marker = params.media_marker == nullptr ? "" : params.media_marker;
    info.use_gpu = params.use_gpu;
    info.print_timings = params.print_timings;
    info.n_threads = params.n_threads;
    info.warmup = params.warmup;
    info.image_min_tokens = params.image_min_tokens;
    info.image_max_tokens = params.image_max_tokens;
    return info;
}

} // namespace vlacpp
