#pragma once

#include <cstdint>
#include <functional>
#include <vector>

namespace vlacpp {

using VelocityFn = std::function<void(float time, const std::vector<float> & x, std::vector<float> & v)>;

void sample_flow_euler(
    int steps,
    uint32_t seed,
    int horizon,
    int action_dim,
    const VelocityFn & velocity,
    std::vector<float> & out);

} // namespace vlacpp
