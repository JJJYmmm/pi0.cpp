#include "mtmd.h"

#include <iostream>
#include <string>

int main() {
    const char * marker = mtmd_default_marker();
    if (marker == nullptr || std::string(marker) != "<__media__>") {
        std::cerr << "unexpected mtmd media marker\n";
        return 1;
    }

    const mtmd_context_params params = mtmd_context_params_default();
    if (params.media_marker == nullptr || std::string(params.media_marker) != "<__media__>") {
        std::cerr << "unexpected mtmd context media marker\n";
        return 1;
    }
    if (params.n_threads <= 0) {
        std::cerr << "mtmd default n_threads should be positive\n";
        return 1;
    }
    if (params.image_min_tokens != -1 || params.image_max_tokens != -1) {
        std::cerr << "mtmd dynamic image token defaults changed\n";
        return 1;
    }
    return 0;
}
