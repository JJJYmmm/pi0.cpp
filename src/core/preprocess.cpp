#include "core/preprocess.h"

#include "core/error.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <set>

namespace vlacpp {
namespace {

float sample_u8_rgb(const vlacpp_image_view & image, int x, int y, int c) {
    const int channel = std::min(c, image.channels - 1);
    const uint8_t * row = image.data + static_cast<size_t>(y) * image.stride_bytes;
    return static_cast<float>(row[x * image.channels + channel]) / 255.0f * 2.0f - 1.0f;
}

} // namespace

vlacpp_status validate_and_preprocess(
    const ModelConfig & config,
    const vlacpp_observation & raw,
    ObservationData & out) {
    out = {};

    if (config.state_dim > 0) {
        if (raw.state == nullptr) {
            return fail(VLACPP_STATUS_INVALID_ARGUMENT, "observation state is required");
        }
        if (raw.state_count != static_cast<size_t>(config.state_dim)) {
            return fail(VLACPP_STATUS_INVALID_ARGUMENT, "observation state dimension mismatch");
        }
        out.state.assign(raw.state, raw.state + raw.state_count);
        if (config.state_mean.size() == static_cast<size_t>(config.state_dim) ||
            config.state_std.size() == static_cast<size_t>(config.state_dim)) {
            if (config.state_mean.size() != static_cast<size_t>(config.state_dim) ||
                config.state_std.size() != static_cast<size_t>(config.state_dim)) {
                return fail(VLACPP_STATUS_PARSE_ERROR, "state_mean and state_std must both match state_dim");
            }
            for (int i = 0; i < config.state_dim; ++i) {
                const float std = config.state_std[static_cast<size_t>(i)];
                if (std == 0.0f) {
                    return fail(VLACPP_STATUS_PARSE_ERROR, "state_std contains zero");
                }
                out.state[static_cast<size_t>(i)] =
                    (out.state[static_cast<size_t>(i)] - config.state_mean[static_cast<size_t>(i)]) / std;
            }
        }
    }

    if (raw.prompt != nullptr) {
        out.prompt = raw.prompt;
    }

    std::set<std::string> required(config.image_keys.begin(), config.image_keys.end());
    for (size_t i = 0; i < raw.image_count; ++i) {
        const vlacpp_image_view & image = raw.images[i];
        if (image.name == nullptr || image.data == nullptr) {
            return fail(VLACPP_STATUS_INVALID_ARGUMENT, "image view has null name or data");
        }
        if (image.width <= 0 || image.height <= 0 || image.channels <= 0 || image.channels > 4) {
            return fail(VLACPP_STATUS_INVALID_ARGUMENT, "image view has invalid dimensions");
        }
        if (image.stride_bytes < image.width * image.channels) {
            return fail(VLACPP_STATUS_INVALID_ARGUMENT, "image stride is too small");
        }

        required.erase(image.name);

        ImageTensor tensor;
        tensor.name = image.name;
        tensor.width = config.image_width;
        tensor.height = config.image_height;
        tensor.channels = 3;
        tensor.data.resize(static_cast<size_t>(tensor.width) * tensor.height * tensor.channels);

        for (int y = 0; y < tensor.height; ++y) {
            const int src_y = std::min(image.height - 1, static_cast<int>(
                std::floor((static_cast<float>(y) + 0.5f) * image.height / tensor.height)));
            for (int x = 0; x < tensor.width; ++x) {
                const int src_x = std::min(image.width - 1, static_cast<int>(
                    std::floor((static_cast<float>(x) + 0.5f) * image.width / tensor.width)));
                for (int c = 0; c < 3; ++c) {
                    tensor.data[(static_cast<size_t>(y) * tensor.width + x) * 3 + c] =
                        sample_u8_rgb(image, src_x, src_y, c);
                }
            }
        }
        out.images.push_back(std::move(tensor));
    }

    if (!required.empty()) {
        return fail(VLACPP_STATUS_INVALID_ARGUMENT, "observation missing required image view: " + *required.begin());
    }

    return VLACPP_STATUS_OK;
}

} // namespace vlacpp
