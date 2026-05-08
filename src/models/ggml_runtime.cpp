#include "models/ggml_runtime.h"

#include "ggml-cpu.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>

namespace vlacpp {
namespace {

struct BackendState {
    ggml_backend_t gpu = nullptr;
    ggml_backend_t cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;

    ~BackendState() {
        if (sched != nullptr) {
            ggml_backend_sched_free(sched);
        }
        if (cpu != nullptr) {
            ggml_backend_free(cpu);
        }
        if (gpu != nullptr) {
            ggml_backend_free(gpu);
        }
    }
};

thread_local std::unique_ptr<BackendState> cuda_state;

void set_cpu_threads(ggml_backend_t backend, int n_threads) {
    ggml_backend_reg_t cpu_reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    auto set_threads = reinterpret_cast<ggml_backend_set_n_threads_t>(
        ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_set_n_threads"));
    if (set_threads != nullptr) {
        set_threads(backend, std::max(1, n_threads));
    }
}

BackendState & get_cuda_state(int n_threads) {
    if (!cuda_state) {
        cuda_state = std::make_unique<BackendState>();
        cuda_state->gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
        if (cuda_state->gpu == nullptr) {
            cuda_state.reset();
            throw std::runtime_error("CUDA backend requested but no ggml GPU backend is available");
        }
        cuda_state->cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (cuda_state->cpu == nullptr) {
            cuda_state.reset();
            throw std::runtime_error("failed to initialize ggml CPU fallback backend");
        }

        ggml_backend_t backends[] = {cuda_state->gpu, cuda_state->cpu};
        cuda_state->sched = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
        if (cuda_state->sched == nullptr) {
            cuda_state.reset();
            throw std::runtime_error("failed to initialize ggml backend scheduler");
        }
    }
    set_cpu_threads(cuda_state->cpu, n_threads);
    return *cuda_state;
}

} // namespace

GgmlRunner::GgmlRunner(const BackendConfig & backend)
    : backend_config_(backend) {
    if (backend_config_.backend != VLACPP_BACKEND_CUDA) {
        return;
    }

    BackendState & state = get_cuda_state(backend_config_.n_threads);
    gpu_backend_ = state.gpu;
    cpu_backend_ = state.cpu;
    sched_ = state.sched;
}

GgmlRunner::~GgmlRunner() = default;

bool GgmlRunner::uses_backend() const {
    return sched_ != nullptr;
}

ggml_init_params GgmlRunner::init_params(size_t mem_size) const {
    ggml_init_params params{};
    params.mem_size = mem_size;
    params.mem_buffer = nullptr;
    params.no_alloc = uses_backend();
    return params;
}

void GgmlRunner::set_input(
    std::vector<GgmlInput> & inputs,
    ggml_tensor * tensor,
    const void * data,
    size_t size) const {
    if (uses_backend()) {
        ggml_set_input(tensor);
        inputs.push_back({tensor, data, size});
        return;
    }
    std::memcpy(tensor->data, data, size);
}

void GgmlRunner::compute(
    ggml_context * ctx,
    ggml_cgraph * graph,
    const std::vector<GgmlInput> & inputs,
    const char * error_message) const {
    ggml_status status = GGML_STATUS_SUCCESS;
    if (!uses_backend()) {
        status = ggml_graph_compute_with_ctx(ctx, graph, std::max(1, backend_config_.n_threads));
    } else {
        ggml_backend_sched_reset(sched_);
        if (!ggml_backend_sched_alloc_graph(sched_, graph)) {
            throw std::runtime_error("ggml backend graph allocation failed");
        }
        for (const GgmlInput & input : inputs) {
            ggml_backend_tensor_set(input.tensor, input.data, 0, input.size);
        }
        status = ggml_backend_sched_graph_compute(sched_, graph);
    }
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error(error_message);
    }
}

void GgmlRunner::get_output(const ggml_tensor * tensor, void * data, size_t size) const {
    if (uses_backend()) {
        ggml_backend_tensor_get(tensor, data, 0, size);
        return;
    }
    std::memcpy(data, tensor->data, size);
}

} // namespace vlacpp
