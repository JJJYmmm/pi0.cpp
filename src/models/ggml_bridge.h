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

void ggml_linear_batch(
    const Tensor & weight,
    const Tensor & bias,
    const std::vector<float> & input,
    int batch,
    std::vector<float> & output,
    int n_threads);

} // namespace vlacpp
