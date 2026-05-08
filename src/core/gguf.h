#pragma once

#include "core/types.h"
#include "vlacpp.h"

#include <string>

namespace vlacpp {

vlacpp_status load_gguf_model_file(
    const std::string & path,
    ModelConfig & out_config,
    TensorMap & out_tensors);

} // namespace vlacpp
