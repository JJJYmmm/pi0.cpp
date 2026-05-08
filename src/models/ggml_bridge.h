#pragma once

#include "core/types.h"

#include <vector>

namespace vlacpp {

void ggml_linear(
    const Tensor & weight,
    const Tensor & bias,
    const std::vector<float> & input,
    std::vector<float> & output,
    int n_threads);

} // namespace vlacpp
