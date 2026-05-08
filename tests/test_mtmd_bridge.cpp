#include "models/mtmd_bridge.h"

#include <iostream>

int main() {
    if (!vlacpp::mtmd_bridge_available()) {
        std::cerr << "mtmd bridge should be available\n";
        return 1;
    }

    const vlacpp::MtmdBridgeInfo info = vlacpp::mtmd_bridge_default_info();
    if (info.media_marker != "<__media__>") {
        std::cerr << "unexpected mtmd media marker: " << info.media_marker << "\n";
        return 1;
    }
    if (info.n_threads <= 0) {
        std::cerr << "mtmd default n_threads should be positive\n";
        return 1;
    }
    if (info.image_min_tokens != -1 || info.image_max_tokens != -1) {
        std::cerr << "mtmd dynamic image token defaults changed\n";
        return 1;
    }
    return 0;
}
