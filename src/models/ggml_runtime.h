#pragma once

#include "core/types.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <cstddef>
#include <string>
#include <vector>

namespace vlacpp {

struct GgmlInput {
    ggml_tensor * tensor = nullptr;
    const void * data = nullptr;
    size_t size = 0;
};

class GgmlRunner {
public:
    explicit GgmlRunner(const BackendConfig & backend);
    ~GgmlRunner();

    GgmlRunner(const GgmlRunner &) = delete;
    GgmlRunner & operator=(const GgmlRunner &) = delete;

    bool uses_backend() const;
    ggml_init_params init_params(size_t mem_size) const;
    ggml_tensor * new_weight_1d(ggml_context * ctx, const Tensor & tensor) const;
    ggml_tensor * new_weight_2d(ggml_context * ctx, const Tensor & tensor) const;

    void set_input(
        std::vector<GgmlInput> & inputs,
        ggml_tensor * tensor,
        const void * data,
        size_t size) const;

    void compute(
        ggml_context * ctx,
        ggml_cgraph * graph,
        const std::vector<GgmlInput> & inputs,
        const char * error_message) const;

    void get_output(const ggml_tensor * tensor, void * data, size_t size) const;

private:
    const BackendConfig & backend_config_;
    ggml_backend_t gpu_backend_ = nullptr;
    ggml_backend_t cpu_backend_ = nullptr;
    ggml_backend_sched_t sched_ = nullptr;
    mutable std::string profile_label_;
};

} // namespace vlacpp
