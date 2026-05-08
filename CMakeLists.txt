cmake_minimum_required(VERSION 3.16)
project(vlacpp VERSION 0.1.0 LANGUAGES C CXX)

option(VLACPP_BUILD_EXAMPLES "Build vlacpp examples" ON)
option(VLACPP_BUILD_TESTS "Build vlacpp tests" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(vlacpp
    src/core/context.cpp
    src/core/error.cpp
    src/core/gguf.cpp
    src/core/json.cpp
    src/core/preprocess.cpp
    src/models/model.cpp
    src/models/pi0_action_expert.cpp
    src/models/pi0_action_decoder.cpp
    src/models/pi0_language_prefix.cpp
    src/models/pi0_vlm.cpp
    src/models/pi0.cpp
    src/sampling/flow.cpp
)

target_include_directories(vlacpp
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_features(vlacpp PUBLIC cxx_std_17)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/CMakeLists.txt")
    message(FATAL_ERROR "third_party/llama.cpp is required; run `git submodule update --init --recursive`")
endif()
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/llama.cpp EXCLUDE_FROM_ALL)
if(NOT TARGET ggml OR NOT TARGET llama OR NOT TARGET mtmd)
    message(FATAL_ERROR "llama.cpp must provide ggml, llama, and mtmd targets")
endif()
target_link_libraries(vlacpp PRIVATE ggml llama mtmd)
target_compile_definitions(vlacpp PRIVATE VLACPP_HAVE_GGML=1 VLACPP_HAVE_MTMD=1)

if(VLACPP_BUILD_EXAMPLES)
    add_executable(vlacpp-pi0 examples/pi0-cli/main.cpp)
    target_link_libraries(vlacpp-pi0 PRIVATE vlacpp)
endif()

if(VLACPP_BUILD_TESTS)
    enable_testing()
    add_executable(vlacpp-tests tests/test_runtime.cpp)
    target_link_libraries(vlacpp-tests PRIVATE vlacpp)
    add_test(NAME vlacpp-runtime COMMAND vlacpp-tests)
    add_executable(vlacpp-mtmd-tests tests/test_mtmd_api.cpp)
    target_link_libraries(vlacpp-mtmd-tests PRIVATE vlacpp mtmd)
    add_test(NAME vlacpp-mtmd-api COMMAND vlacpp-mtmd-tests)
    add_executable(vlacpp-pi0-action-expert-tests tests/test_pi0_action_expert.cpp)
    target_include_directories(vlacpp-pi0-action-expert-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(vlacpp-pi0-action-expert-tests PRIVATE vlacpp)
    add_test(NAME vlacpp-pi0-action-expert COMMAND vlacpp-pi0-action-expert-tests)

    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        set(VLACPP_TINY_CHECKPOINT "${CMAKE_CURRENT_SOURCE_DIR}/tests/tiny-openpi-checkpoint.json")
        set(VLACPP_TINY_GGUF "${CMAKE_CURRENT_BINARY_DIR}/tiny-openpi.gguf")
        set(VLACPP_ACTION_HEAD_CHECKPOINT "${CMAKE_CURRENT_SOURCE_DIR}/tests/action-head-checkpoint.json")
        set(VLACPP_ACTION_HEAD_GGUF "${CMAKE_CURRENT_BINARY_DIR}/action-head.gguf")
        set(VLACPP_PROJECTOR_METADATA_CHECKPOINT "${CMAKE_CURRENT_SOURCE_DIR}/tests/projector-metadata-checkpoint.json")
        set(VLACPP_PROJECTOR_METADATA_GGUF "${CMAKE_CURRENT_BINARY_DIR}/projector-metadata.gguf")
        set(VLACPP_ACTION_HEAD_SAFETENSORS "${CMAKE_CURRENT_BINARY_DIR}/action-head-openpi-names.safetensors")
        set(VLACPP_ACTION_HEAD_PREFIXED_SAFETENSORS "${CMAKE_CURRENT_BINARY_DIR}/action-head-model-prefix-openpi-names.safetensors")
        set(VLACPP_ACTION_HEAD_MANIFEST "${CMAKE_CURRENT_BINARY_DIR}/action-head-openpi-map.json")
        set(VLACPP_ACTION_HEAD_PREFIXED_MANIFEST "${CMAKE_CURRENT_BINARY_DIR}/action-head-model-prefix-openpi-map.json")
        set(VLACPP_ACTION_HEAD_PREFIXED_GGUF "${CMAKE_CURRENT_BINARY_DIR}/action-head-model-prefix-openpi.gguf")
        set(VLACPP_ACTION_HEAD_INVENTORY_MANIFEST "${CMAKE_CURRENT_BINARY_DIR}/action-head-openpi-inventory-map.json")
        set(VLACPP_ACTION_HEAD_ALL_MANIFEST "${CMAKE_CURRENT_BINARY_DIR}/action-head-openpi-all-map.json")
        set(VLACPP_ACTION_HEAD_ALL_GGUF "${CMAKE_CURRENT_BINARY_DIR}/action-head-openpi-all.gguf")
        set(VLACPP_PI0_FULL_MANIFEST "${CMAKE_CURRENT_BINARY_DIR}/action-head-openpi-full-map.json")
        set(VLACPP_PI0_FULL_GGUF "${CMAKE_CURRENT_BINARY_DIR}/action-head-openpi-full.gguf")
        set(VLACPP_ACTION_HEAD_MAPPED_GGUF "${CMAKE_CURRENT_BINARY_DIR}/action-head-openpi-mapped.gguf")
        set(VLACPP_ACTION_HEAD_NORM_STATS "${CMAKE_CURRENT_SOURCE_DIR}/tests/action-head-norm-stats.json")
        set(VLACPP_ACTION_HEAD_MAPPED_NORM_GGUF "${CMAKE_CURRENT_BINARY_DIR}/action-head-openpi-mapped-norm.gguf")
        set(VLACPP_PI05_ACTION_HEAD_CHECKPOINT "${CMAKE_CURRENT_SOURCE_DIR}/tests/pi05-action-head-checkpoint.json")
        set(VLACPP_PI05_ACTION_HEAD_GGUF "${CMAKE_CURRENT_BINARY_DIR}/pi05-action-head.gguf")
        set(VLACPP_PI05_ACTION_HEAD_SAFETENSORS "${CMAKE_CURRENT_BINARY_DIR}/pi05-action-head-openpi-names.safetensors")
        set(VLACPP_PI05_ACTION_HEAD_MANIFEST "${CMAKE_CURRENT_BINARY_DIR}/pi05-action-head-openpi-map.json")
        set(VLACPP_PI05_ACTION_HEAD_INVENTORY_MANIFEST "${CMAKE_CURRENT_BINARY_DIR}/pi05-action-head-openpi-inventory-map.json")
        set(VLACPP_PI05_FULL_MANIFEST "${CMAKE_CURRENT_BINARY_DIR}/pi05-action-head-openpi-full-map.json")
        set(VLACPP_PI05_FULL_GGUF "${CMAKE_CURRENT_BINARY_DIR}/pi05-action-head-openpi-full.gguf")
        set(VLACPP_PI05_ACTION_HEAD_MAPPED_GGUF "${CMAKE_CURRENT_BINARY_DIR}/pi05-action-head-openpi-mapped.gguf")
        set(VLACPP_TINY_SAFETENSORS "${CMAKE_CURRENT_BINARY_DIR}/tiny-openpi.safetensors")
        set(VLACPP_TINY_SAFETENSORS_GGUF "${CMAKE_CURRENT_BINARY_DIR}/tiny-openpi-safetensors.gguf")
        set(VLACPP_PI05_F16_SAFETENSORS "${CMAKE_CURRENT_BINARY_DIR}/pi05-action-head-f16.safetensors")
        set(VLACPP_PI05_F16_GGUF "${CMAKE_CURRENT_BINARY_DIR}/pi05-action-head-f16.gguf")
        set(VLACPP_OPENPI_GRAPH_MANIFEST "${CMAKE_CURRENT_SOURCE_DIR}/tests/openpi-graph-manifest.json")
        set(VLACPP_OPENPI_GRAPH_CHECKPOINT "${CMAKE_CURRENT_SOURCE_DIR}/tests/openpi-graph-checkpoint.json")
        set(VLACPP_OPENPI_GRAPH_DEFAULT_METADATA "${CMAKE_CURRENT_BINARY_DIR}/openpi-graph-default-metadata.json")
        set(VLACPP_OPENPI_GRAPH_PI0_FULL_MANIFEST "${CMAKE_CURRENT_BINARY_DIR}/openpi-graph-pi0-full-map.json")
        set(VLACPP_PI0_VISION_MTMD_MANIFEST "${CMAKE_CURRENT_BINARY_DIR}/pi0-vision-mtmd-map.json")
        set(VLACPP_PI0_VISION_PROJECTOR_MANIFEST "${CMAKE_CURRENT_BINARY_DIR}/pi0-vision-projector-map.json")
        add_test(
            NAME vlacpp-openpi-graph-summary
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/summarize-openpi-graph.py"
                "${VLACPP_OPENPI_GRAPH_MANIFEST}"
                --expect-action-dim 32
                --expect-action-width 1024
                --expect-vision-layers 2
                --expect-language-layers 2
                --expect-action-expert-layers 2
        )
        add_test(
            NAME vlacpp-openpi-graph-default-metadata-convert
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/convert-openpi-to-gguf.py"
                --checkpoint "${VLACPP_OPENPI_GRAPH_CHECKPOINT}"
                --output "${VLACPP_OPENPI_GRAPH_DEFAULT_METADATA}"
                --output-format json
        )
        add_test(
            NAME vlacpp-openpi-graph-default-metadata-check
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_json_checkpoint_metadata.py"
                "${VLACPP_OPENPI_GRAPH_DEFAULT_METADATA}"
                --expect "model_type=pi0"
                --expect "action_horizon=50"
                --expect "action_dim=32"
                --expect "openpi_action_expert_layers=1"
        )
        set_tests_properties(vlacpp-openpi-graph-default-metadata-check PROPERTIES DEPENDS vlacpp-openpi-graph-default-metadata-convert)
        add_test(
            NAME vlacpp-llama-components
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/check-llama-components.py"
                --repo "${CMAKE_CURRENT_SOURCE_DIR}"
                --build-dir "${CMAKE_CURRENT_BINARY_DIR}"
                --require-ggml-targets
                --require-mtmd-target
        )
        add_test(
            NAME vlacpp-openpi-llama-reuse
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/check-openpi-llama-reuse.py"
                "${VLACPP_OPENPI_GRAPH_MANIFEST}"
                --repo "${CMAKE_CURRENT_SOURCE_DIR}"
                --expect-vision-decision reuse
                --expect-vision-projector-decision custom-direct-ggml
                --expect-action-expert-decision custom-ggml-graph
                --expect-action-head-decision custom-direct-ggml
        )
        add_test(
            NAME vlacpp-pi0-vision-mtmd-map
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/map-openpi-tensors.py"
                "${VLACPP_OPENPI_GRAPH_MANIFEST}"
                --family pi0-vision-mtmd
                --require-complete
                --include-inventory
                --output "${VLACPP_PI0_VISION_MTMD_MANIFEST}"
        )
        add_test(
            NAME vlacpp-openpi-graph-pi0-full-map
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/map-openpi-tensors.py"
                "${VLACPP_OPENPI_GRAPH_MANIFEST}"
                --family pi0-full
                --require-complete
                --include-inventory
                --output "${VLACPP_OPENPI_GRAPH_PI0_FULL_MANIFEST}"
        )
        add_test(
            NAME vlacpp-openpi-graph-pi0-full-summary
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/summarize-tensor-map.py"
                "${VLACPP_OPENPI_GRAPH_PI0_FULL_MANIFEST}"
                --expect-family pi0-full
                --expect-expected 16
                --expect-mapped 16
                --expect-total 16
                --expect-unmapped 0
        )
        set_tests_properties(vlacpp-openpi-graph-pi0-full-summary PROPERTIES DEPENDS vlacpp-openpi-graph-pi0-full-map)
        add_test(
            NAME vlacpp-pi0-vision-projector-map
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/map-openpi-tensors.py"
                "${VLACPP_OPENPI_GRAPH_MANIFEST}"
                --family pi0-vision-projector
                --require-complete
                --include-inventory
                --output "${VLACPP_PI0_VISION_PROJECTOR_MANIFEST}"
        )
        add_test(
            NAME vlacpp-pi0-vision-projector-summary
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/summarize-tensor-map.py"
                "${VLACPP_PI0_VISION_PROJECTOR_MANIFEST}"
                --expect-family pi0-vision-projector
                --expect-expected 2
                --expect-mapped 2
                --expect-total 16
                --expect-unmapped 14
        )
        set_tests_properties(vlacpp-pi0-vision-projector-summary PROPERTIES DEPENDS vlacpp-pi0-vision-projector-map)
        add_test(
            NAME vlacpp-pi0-vision-mtmd-summary
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/summarize-tensor-map.py"
                "${VLACPP_PI0_VISION_MTMD_MANIFEST}"
                --expect-family pi0-vision-mtmd
                --expect-expected 3
                --expect-mapped 3
                --expect-total 16
                --expect-unmapped 13
        )
        set_tests_properties(vlacpp-pi0-vision-mtmd-summary PROPERTIES DEPENDS vlacpp-pi0-vision-mtmd-map)
        add_test(
            NAME vlacpp-tiny-gguf-convert
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/convert-openpi-to-gguf.py"
                --checkpoint "${VLACPP_TINY_CHECKPOINT}"
                --output "${VLACPP_TINY_GGUF}"
        )
        add_test(
            NAME vlacpp-tiny-openpi-compare
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/compare-openpi-reference.py"
                --checkpoint "${VLACPP_TINY_CHECKPOINT}"
                --model "${VLACPP_TINY_GGUF}"
                --binary "$<TARGET_FILE:vlacpp-pi0>"
                --state "1,2,3"
                --prompt "pick up the fork"
                --steps 4
                --seed 1
        )
        add_test(
            NAME vlacpp-tiny-capability-info
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_cli_actions.py"
                --binary "$<TARGET_FILE:vlacpp-pi0>"
                --model "${VLACPP_TINY_GGUF}"
                --state "1,2,3"
                --prompt "pick up the fork"
                --steps 1
                --seed 1
                --horizon 4
                --action-dim 2
                --capability "tiny-velocity"
        )
        add_test(
            NAME vlacpp-require-full-rejects-restricted
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_require_full.py"
                --binary "$<TARGET_FILE:vlacpp-pi0>"
                --model "${VLACPP_TINY_GGUF}"
        )
        set_tests_properties(vlacpp-tiny-openpi-compare PROPERTIES DEPENDS vlacpp-tiny-gguf-convert)
        set_tests_properties(vlacpp-tiny-capability-info PROPERTIES DEPENDS vlacpp-tiny-gguf-convert)
        set_tests_properties(vlacpp-require-full-rejects-restricted PROPERTIES DEPENDS vlacpp-tiny-gguf-convert)
        add_test(
            NAME vlacpp-fake-openpi-policy-compare
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_fake_openpi_policy_compare.py"
                "${CMAKE_CURRENT_SOURCE_DIR}"
                "$<TARGET_FILE:vlacpp-pi0>"
                "${VLACPP_TINY_CHECKPOINT}"
                "${VLACPP_TINY_GGUF}"
        )
        set_tests_properties(vlacpp-fake-openpi-policy-compare PROPERTIES DEPENDS vlacpp-tiny-gguf-convert)
        add_test(
            NAME vlacpp-official-openpi-compare-skip
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/compare-openpi-policy.py"
                --openpi-config "pi05_libero"
                --openpi-checkpoint "/tmp/not-used"
                --vlacpp-model "/tmp/not-used.gguf"
                --allow-missing-openpi
        )
        add_test(
            NAME vlacpp-official-openpi-pytorch-smoke-skip
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/run-openpi-pytorch-smoke.py"
                --checkpoint "/tmp/not-used.safetensors"
                --allow-missing-openpi
        )
        add_test(
            NAME vlacpp-download-lock
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_download_lock.py"
                --repo "${CMAKE_CURRENT_SOURCE_DIR}"
                --work-dir "${CMAKE_CURRENT_BINARY_DIR}/download-lock"
        )
        add_test(
            NAME vlacpp-pi0-goal-audit-incomplete
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/audit-pi0-goal.py"
                --repo "${CMAKE_CURRENT_SOURCE_DIR}"
                --binary "$<TARGET_FILE:vlacpp-pi0>"
                --checkpoint "${CMAKE_CURRENT_SOURCE_DIR}/ckpts/lerobot-pi0/model.safetensors"
                --vlacpp-model "${VLACPP_TINY_GGUF}"
                --allow-incomplete
        )
        set_tests_properties(vlacpp-pi0-goal-audit-incomplete PROPERTIES DEPENDS vlacpp-tiny-gguf-convert)

        add_test(
            NAME vlacpp-action-head-gguf-convert
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/convert-openpi-to-gguf.py"
                --checkpoint "${VLACPP_ACTION_HEAD_CHECKPOINT}"
                --output "${VLACPP_ACTION_HEAD_GGUF}"
        )
        add_test(
            NAME vlacpp-action-head-cli
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_cli_actions.py"
                --binary "$<TARGET_FILE:vlacpp-pi0>"
                --model "${VLACPP_ACTION_HEAD_GGUF}"
                --state "1,-2"
                --prompt "pick up the fork"
                --steps 2
                --seed 1
                --horizon 3
                --action-dim 2
                --capability "restricted-pi0-state-action-head"
                --graph-action-width 4
        )
        add_test(
            NAME vlacpp-action-head-gguf-inspect
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_gguf_tensors.py"
                --repo "${CMAKE_CURRENT_SOURCE_DIR}"
                --model "${VLACPP_ACTION_HEAD_GGUF}"
        )
        add_test(
            NAME vlacpp-action-head-reference-compare
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/compare_action_head_reference.py"
                --checkpoint "${VLACPP_ACTION_HEAD_CHECKPOINT}"
                --model "${VLACPP_ACTION_HEAD_GGUF}"
                --binary "$<TARGET_FILE:vlacpp-pi0>"
                --state "1,-2"
                --prompt "pick up the fork"
                --steps 2
                --seed 1
        )
        set_tests_properties(vlacpp-action-head-cli PROPERTIES DEPENDS vlacpp-action-head-gguf-convert)
        set_tests_properties(vlacpp-action-head-gguf-inspect PROPERTIES DEPENDS vlacpp-action-head-gguf-convert)
        set_tests_properties(vlacpp-action-head-reference-compare PROPERTIES DEPENDS vlacpp-action-head-gguf-convert)

        add_test(
            NAME vlacpp-projector-metadata-convert
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/convert-openpi-to-gguf.py"
                --checkpoint "${VLACPP_PROJECTOR_METADATA_CHECKPOINT}"
                --output "${VLACPP_PROJECTOR_METADATA_GGUF}"
        )
        add_test(
            NAME vlacpp-projector-metadata-check
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_projector_metadata.py"
                --repo "${CMAKE_CURRENT_SOURCE_DIR}"
                --model "${VLACPP_PROJECTOR_METADATA_GGUF}"
        )
        set_tests_properties(vlacpp-projector-metadata-check PROPERTIES DEPENDS vlacpp-projector-metadata-convert)
        add_test(
            NAME vlacpp-truncated-manifest-convert
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_truncated_manifest_convert.py"
                --repo "${CMAKE_CURRENT_SOURCE_DIR}"
                --work-dir "${CMAKE_CURRENT_BINARY_DIR}/truncated-manifest-convert"
        )
        set_tests_properties(vlacpp-truncated-manifest-convert PROPERTIES SKIP_RETURN_CODE 77)

        add_test(
            NAME vlacpp-action-head-safetensors-create
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/make_action_head_safetensors.py"
                "${VLACPP_ACTION_HEAD_CHECKPOINT}"
                "${VLACPP_ACTION_HEAD_SAFETENSORS}"
        )
        set_tests_properties(vlacpp-action-head-safetensors-create PROPERTIES SKIP_RETURN_CODE 77)
        add_test(
            NAME vlacpp-action-head-prefixed-safetensors-create
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/make_action_head_safetensors.py"
                "${VLACPP_ACTION_HEAD_CHECKPOINT}"
                "${VLACPP_ACTION_HEAD_PREFIXED_SAFETENSORS}"
                --prefix "model."
        )
        set_tests_properties(vlacpp-action-head-prefixed-safetensors-create PROPERTIES SKIP_RETURN_CODE 77)
        add_test(
            NAME vlacpp-action-head-safetensors-map
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/map-openpi-tensors.py"
                "${VLACPP_ACTION_HEAD_SAFETENSORS}"
                --family action-expert
                --require-complete
                --output "${VLACPP_ACTION_HEAD_MANIFEST}"
        )
        add_test(
            NAME vlacpp-action-head-prefixed-safetensors-map
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/map-openpi-tensors.py"
                "${VLACPP_ACTION_HEAD_PREFIXED_SAFETENSORS}"
                --family action-expert
                --require-complete
                --output "${VLACPP_ACTION_HEAD_PREFIXED_MANIFEST}"
        )
        add_test(
            NAME vlacpp-action-head-prefixed-safetensors-convert
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/convert-openpi-to-gguf.py"
                --tensor-map-manifest "${VLACPP_ACTION_HEAD_PREFIXED_MANIFEST}"
                --output "${VLACPP_ACTION_HEAD_PREFIXED_GGUF}"
        )
        add_test(
            NAME vlacpp-action-head-prefixed-safetensors-reference-compare
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/compare_action_head_reference.py"
                --checkpoint "${VLACPP_ACTION_HEAD_CHECKPOINT}"
                --model "${VLACPP_ACTION_HEAD_PREFIXED_GGUF}"
                --binary "$<TARGET_FILE:vlacpp-pi0>"
                --state "1,-2"
                --prompt "pick up the fork"
                --steps 2
                --seed 1
        )
        add_test(
            NAME vlacpp-action-head-safetensors-inventory
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/map-openpi-tensors.py"
                "${VLACPP_ACTION_HEAD_SAFETENSORS}"
                --family action-expert
                --require-complete
                --include-inventory
                --output "${VLACPP_ACTION_HEAD_INVENTORY_MANIFEST}"
        )
        add_test(
            NAME vlacpp-action-head-safetensors-inventory-check
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_tensor_inventory.py"
                --manifest "${VLACPP_ACTION_HEAD_INVENTORY_MANIFEST}"
        )
        add_test(
            NAME vlacpp-action-head-safetensors-summary
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/summarize-tensor-map.py"
                "${VLACPP_ACTION_HEAD_INVENTORY_MANIFEST}"
                --expect-family action-expert
                --expect-expected 10
                --expect-mapped 10
                --expect-total 10
                --expect-unmapped 0
        )
        add_test(
            NAME vlacpp-action-head-safetensors-all-map
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/map-openpi-tensors.py"
                "${VLACPP_ACTION_HEAD_SAFETENSORS}"
                --family all
                --require-complete
                --include-inventory
                --output "${VLACPP_ACTION_HEAD_ALL_MANIFEST}"
        )
        add_test(
            NAME vlacpp-action-head-safetensors-all-summary
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/summarize-tensor-map.py"
                "${VLACPP_ACTION_HEAD_ALL_MANIFEST}"
                --expect-family all
                --expect-expected 10
                --expect-mapped 10
                --expect-total 10
                --expect-unmapped 0
        )
        add_test(
            NAME vlacpp-action-head-safetensors-all-convert
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/convert-openpi-to-gguf.py"
                --tensor-map-manifest "${VLACPP_ACTION_HEAD_ALL_MANIFEST}"
                --output "${VLACPP_ACTION_HEAD_ALL_GGUF}"
        )
        add_test(
            NAME vlacpp-action-head-safetensors-pi0-full-map
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/map-openpi-tensors.py"
                "${VLACPP_ACTION_HEAD_SAFETENSORS}"
                --family pi0-full
                --require-complete
                --include-inventory
                --output "${VLACPP_PI0_FULL_MANIFEST}"
        )
        add_test(
            NAME vlacpp-action-head-safetensors-pi0-full-summary
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/summarize-tensor-map.py"
                "${VLACPP_PI0_FULL_MANIFEST}"
                --expect-family pi0-full
                --expect-expected 10
                --expect-mapped 10
                --expect-total 10
                --expect-unmapped 0
        )
        add_test(
            NAME vlacpp-action-head-safetensors-pi0-full-convert
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/convert-openpi-to-gguf.py"
                --tensor-map-manifest "${VLACPP_PI0_FULL_MANIFEST}"
                --output "${VLACPP_PI0_FULL_GGUF}"
        )
        add_test(
            NAME vlacpp-action-head-safetensors-pi0-full-reference-compare
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/compare_action_head_reference.py"
                --checkpoint "${VLACPP_ACTION_HEAD_CHECKPOINT}"
                --model "${VLACPP_PI0_FULL_GGUF}"
                --binary "$<TARGET_FILE:vlacpp-pi0>"
                --state "1,-2"
                --prompt "pick up the fork"
                --steps 2
                --seed 1
        )
        add_test(
            NAME vlacpp-action-head-mapped-safetensors-convert
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/convert-openpi-to-gguf.py"
                --tensor-map-manifest "${VLACPP_ACTION_HEAD_MANIFEST}"
                --output "${VLACPP_ACTION_HEAD_MAPPED_GGUF}"
        )
        add_test(
            NAME vlacpp-action-head-mapped-safetensors-reference-compare
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/compare_action_head_reference.py"
                --checkpoint "${VLACPP_ACTION_HEAD_CHECKPOINT}"
                --model "${VLACPP_ACTION_HEAD_MAPPED_GGUF}"
                --binary "$<TARGET_FILE:vlacpp-pi0>"
                --state "1,-2"
                --prompt "pick up the fork"
                --steps 2
                --seed 1
        )
        add_test(
            NAME vlacpp-action-head-local-smoke
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/run-action-head-smoke.py"
                --repo "${CMAKE_CURRENT_SOURCE_DIR}"
                --source "${VLACPP_ACTION_HEAD_SAFETENSORS}"
                --binary "$<TARGET_FILE:vlacpp-pi0>"
                --work-dir "${CMAKE_CURRENT_BINARY_DIR}/action-head-local-smoke"
                --steps 2
                --seed 1
        )
        add_test(
            NAME vlacpp-action-head-mapped-safetensors-norm-convert
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/convert-openpi-to-gguf.py"
                --tensor-map-manifest "${VLACPP_ACTION_HEAD_MANIFEST}"
                --norm-stats "${VLACPP_ACTION_HEAD_NORM_STATS}"
                --output "${VLACPP_ACTION_HEAD_MAPPED_NORM_GGUF}"
        )
        add_test(
            NAME vlacpp-action-head-mapped-safetensors-metadata
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_gguf_metadata.py"
                --repo "${CMAKE_CURRENT_SOURCE_DIR}"
                --model "${VLACPP_ACTION_HEAD_MAPPED_NORM_GGUF}"
        )
        add_test(
            NAME vlacpp-action-head-mapped-safetensors-norm-reference-compare
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/compare_action_head_reference.py"
                --checkpoint "${VLACPP_ACTION_HEAD_CHECKPOINT}"
                --norm-stats "${VLACPP_ACTION_HEAD_NORM_STATS}"
                --model "${VLACPP_ACTION_HEAD_MAPPED_NORM_GGUF}"
                --binary "$<TARGET_FILE:vlacpp-pi0>"
                --state "1,-2"
                --prompt "pick up the fork"
                --steps 2
                --seed 1
        )
        set_tests_properties(vlacpp-action-head-safetensors-map PROPERTIES DEPENDS vlacpp-action-head-safetensors-create)
        set_tests_properties(vlacpp-action-head-prefixed-safetensors-map PROPERTIES DEPENDS vlacpp-action-head-prefixed-safetensors-create)
        set_tests_properties(vlacpp-action-head-prefixed-safetensors-convert PROPERTIES DEPENDS vlacpp-action-head-prefixed-safetensors-map)
        set_tests_properties(vlacpp-action-head-prefixed-safetensors-reference-compare PROPERTIES DEPENDS vlacpp-action-head-prefixed-safetensors-convert)
        set_tests_properties(vlacpp-action-head-safetensors-inventory PROPERTIES DEPENDS vlacpp-action-head-safetensors-create)
        set_tests_properties(vlacpp-action-head-safetensors-inventory-check PROPERTIES DEPENDS vlacpp-action-head-safetensors-inventory)
        set_tests_properties(vlacpp-action-head-safetensors-summary PROPERTIES DEPENDS vlacpp-action-head-safetensors-inventory)
        set_tests_properties(vlacpp-action-head-safetensors-all-map PROPERTIES DEPENDS vlacpp-action-head-safetensors-create)
        set_tests_properties(vlacpp-action-head-safetensors-all-summary PROPERTIES DEPENDS vlacpp-action-head-safetensors-all-map)
        set_tests_properties(vlacpp-action-head-safetensors-all-convert PROPERTIES DEPENDS vlacpp-action-head-safetensors-all-map)
        set_tests_properties(vlacpp-action-head-safetensors-pi0-full-map PROPERTIES DEPENDS vlacpp-action-head-safetensors-create)
        set_tests_properties(vlacpp-action-head-safetensors-pi0-full-summary PROPERTIES DEPENDS vlacpp-action-head-safetensors-pi0-full-map)
        set_tests_properties(vlacpp-action-head-safetensors-pi0-full-convert PROPERTIES DEPENDS vlacpp-action-head-safetensors-pi0-full-map)
        set_tests_properties(vlacpp-action-head-safetensors-pi0-full-reference-compare PROPERTIES DEPENDS vlacpp-action-head-safetensors-pi0-full-convert)
        set_tests_properties(vlacpp-action-head-mapped-safetensors-convert PROPERTIES DEPENDS vlacpp-action-head-safetensors-map)
        set_tests_properties(vlacpp-action-head-mapped-safetensors-reference-compare PROPERTIES DEPENDS vlacpp-action-head-mapped-safetensors-convert)
        set_tests_properties(vlacpp-action-head-local-smoke PROPERTIES DEPENDS vlacpp-action-head-safetensors-create)
        set_tests_properties(vlacpp-action-head-mapped-safetensors-norm-convert PROPERTIES DEPENDS vlacpp-action-head-safetensors-map)
        set_tests_properties(vlacpp-action-head-mapped-safetensors-metadata PROPERTIES DEPENDS vlacpp-action-head-mapped-safetensors-norm-convert)
        set_tests_properties(vlacpp-action-head-mapped-safetensors-norm-reference-compare PROPERTIES DEPENDS vlacpp-action-head-mapped-safetensors-norm-convert)

        add_test(
            NAME vlacpp-pi05-action-head-gguf-convert
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/convert-openpi-to-gguf.py"
                --checkpoint "${VLACPP_PI05_ACTION_HEAD_CHECKPOINT}"
                --output "${VLACPP_PI05_ACTION_HEAD_GGUF}"
        )
        add_test(
            NAME vlacpp-pi05-action-head-cli
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_cli_actions.py"
                --binary "$<TARGET_FILE:vlacpp-pi0>"
                --model "${VLACPP_PI05_ACTION_HEAD_GGUF}"
                --state "1,-2"
                --prompt "pick up the fork"
                --steps 2
                --seed 1
                --horizon 3
                --action-dim 2
                --capability "restricted-pi05-action-head"
        )
        add_test(
            NAME vlacpp-pi05-action-head-reference-compare
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/compare_pi05_action_head_reference.py"
                --checkpoint "${VLACPP_PI05_ACTION_HEAD_CHECKPOINT}"
                --model "${VLACPP_PI05_ACTION_HEAD_GGUF}"
                --binary "$<TARGET_FILE:vlacpp-pi0>"
                --state "1,-2"
                --prompt "pick up the fork"
                --steps 2
                --seed 1
        )
        set_tests_properties(vlacpp-pi05-action-head-cli PROPERTIES DEPENDS vlacpp-pi05-action-head-gguf-convert)
        set_tests_properties(vlacpp-pi05-action-head-reference-compare PROPERTIES DEPENDS vlacpp-pi05-action-head-gguf-convert)

        add_test(
            NAME vlacpp-pi05-action-head-safetensors-create
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/make_pi05_action_head_safetensors.py"
                "${VLACPP_PI05_ACTION_HEAD_CHECKPOINT}"
                "${VLACPP_PI05_ACTION_HEAD_SAFETENSORS}"
        )
        set_tests_properties(vlacpp-pi05-action-head-safetensors-create PROPERTIES SKIP_RETURN_CODE 77)
        add_test(
            NAME vlacpp-pi05-action-head-safetensors-map
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/map-openpi-tensors.py"
                "${VLACPP_PI05_ACTION_HEAD_SAFETENSORS}"
                --family pi05-action-expert
                --require-complete
                --output "${VLACPP_PI05_ACTION_HEAD_MANIFEST}"
        )
        add_test(
            NAME vlacpp-pi05-action-head-safetensors-inventory
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/map-openpi-tensors.py"
                "${VLACPP_PI05_ACTION_HEAD_SAFETENSORS}"
                --family pi05-action-expert
                --require-complete
                --include-inventory
                --output "${VLACPP_PI05_ACTION_HEAD_INVENTORY_MANIFEST}"
        )
        add_test(
            NAME vlacpp-pi05-action-head-safetensors-inventory-check
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_tensor_inventory.py"
                --manifest "${VLACPP_PI05_ACTION_HEAD_INVENTORY_MANIFEST}"
                --family pi05-action-expert
                --total 8
                --mapped 8
                --group action_in_proj
                --group action_out_proj
                --group time_mlp_in
                --group time_mlp_out
        )
        add_test(
            NAME vlacpp-pi05-action-head-safetensors-summary
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/summarize-tensor-map.py"
                "${VLACPP_PI05_ACTION_HEAD_INVENTORY_MANIFEST}"
                --expect-family pi05-action-expert
                --expect-expected 8
                --expect-mapped 8
                --expect-total 8
                --expect-unmapped 0
        )
        add_test(
            NAME vlacpp-pi05-action-head-safetensors-pi05-full-map
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/map-openpi-tensors.py"
                "${VLACPP_PI05_ACTION_HEAD_SAFETENSORS}"
                --family pi05-full
                --require-complete
                --include-inventory
                --output "${VLACPP_PI05_FULL_MANIFEST}"
        )
        add_test(
            NAME vlacpp-pi05-action-head-safetensors-pi05-full-summary
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/summarize-tensor-map.py"
                "${VLACPP_PI05_FULL_MANIFEST}"
                --expect-family pi05-full
                --expect-expected 8
                --expect-mapped 8
                --expect-total 8
                --expect-unmapped 0
        )
        add_test(
            NAME vlacpp-pi05-action-head-safetensors-pi05-full-convert
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/convert-openpi-to-gguf.py"
                --tensor-map-manifest "${VLACPP_PI05_FULL_MANIFEST}"
                --output "${VLACPP_PI05_FULL_GGUF}"
        )
        add_test(
            NAME vlacpp-pi05-action-head-safetensors-pi05-full-reference-compare
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/compare_pi05_action_head_reference.py"
                --checkpoint "${VLACPP_PI05_ACTION_HEAD_CHECKPOINT}"
                --model "${VLACPP_PI05_FULL_GGUF}"
                --binary "$<TARGET_FILE:vlacpp-pi0>"
                --state "1,-2"
                --prompt "pick up the fork"
                --steps 2
                --seed 1
        )
        add_test(
            NAME vlacpp-pi05-action-head-mapped-safetensors-convert
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/convert-openpi-to-gguf.py"
                --tensor-map-manifest "${VLACPP_PI05_ACTION_HEAD_MANIFEST}"
                --output "${VLACPP_PI05_ACTION_HEAD_MAPPED_GGUF}"
        )
        add_test(
            NAME vlacpp-pi05-action-head-mapped-safetensors-reference-compare
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/compare_pi05_action_head_reference.py"
                --checkpoint "${VLACPP_PI05_ACTION_HEAD_CHECKPOINT}"
                --model "${VLACPP_PI05_ACTION_HEAD_MAPPED_GGUF}"
                --binary "$<TARGET_FILE:vlacpp-pi0>"
                --state "1,-2"
                --prompt "pick up the fork"
                --steps 2
                --seed 1
        )
        set_tests_properties(vlacpp-pi05-action-head-safetensors-map PROPERTIES DEPENDS vlacpp-pi05-action-head-safetensors-create)
        set_tests_properties(vlacpp-pi05-action-head-safetensors-inventory PROPERTIES DEPENDS vlacpp-pi05-action-head-safetensors-create)
        set_tests_properties(vlacpp-pi05-action-head-safetensors-inventory-check PROPERTIES DEPENDS vlacpp-pi05-action-head-safetensors-inventory)
        set_tests_properties(vlacpp-pi05-action-head-safetensors-summary PROPERTIES DEPENDS vlacpp-pi05-action-head-safetensors-inventory)
        set_tests_properties(vlacpp-pi05-action-head-safetensors-pi05-full-map PROPERTIES DEPENDS vlacpp-pi05-action-head-safetensors-create)
        set_tests_properties(vlacpp-pi05-action-head-safetensors-pi05-full-summary PROPERTIES DEPENDS vlacpp-pi05-action-head-safetensors-pi05-full-map)
        set_tests_properties(vlacpp-pi05-action-head-safetensors-pi05-full-convert PROPERTIES DEPENDS vlacpp-pi05-action-head-safetensors-pi05-full-map)
        set_tests_properties(vlacpp-pi05-action-head-safetensors-pi05-full-reference-compare PROPERTIES DEPENDS vlacpp-pi05-action-head-safetensors-pi05-full-convert)
        set_tests_properties(vlacpp-pi05-action-head-mapped-safetensors-convert PROPERTIES DEPENDS vlacpp-pi05-action-head-safetensors-map)
        set_tests_properties(vlacpp-pi05-action-head-mapped-safetensors-reference-compare PROPERTIES DEPENDS vlacpp-pi05-action-head-mapped-safetensors-convert)

        add_test(
            NAME vlacpp-pi05-f16-safetensors-create
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/make_pi05_f16_safetensors.py"
                "${VLACPP_PI05_F16_SAFETENSORS}"
        )
        set_tests_properties(vlacpp-pi05-f16-safetensors-create PROPERTIES SKIP_RETURN_CODE 77)
        add_test(
            NAME vlacpp-pi05-f16-safetensors-convert
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/convert-openpi-to-gguf.py"
                --checkpoint "${VLACPP_PI05_F16_SAFETENSORS}"
                --output "${VLACPP_PI05_F16_GGUF}"
        )
        add_test(
            NAME vlacpp-pi05-f16-safetensors-cli
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_cli_actions.py"
                --binary "$<TARGET_FILE:vlacpp-pi0>"
                --model "${VLACPP_PI05_F16_GGUF}"
                --state "1,-2"
                --prompt "pick up the fork"
                --steps 2
                --seed 1
                --horizon 3
                --action-dim 2
                --capability "restricted-pi05-action-head"
        )
        set_tests_properties(vlacpp-pi05-f16-safetensors-convert PROPERTIES DEPENDS vlacpp-pi05-f16-safetensors-create)
        set_tests_properties(vlacpp-pi05-f16-safetensors-cli PROPERTIES DEPENDS vlacpp-pi05-f16-safetensors-convert)

        add_test(
            NAME vlacpp-tiny-safetensors-create
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/make_tiny_safetensors.py"
                "${VLACPP_TINY_SAFETENSORS}"
        )
        set_tests_properties(vlacpp-tiny-safetensors-create PROPERTIES SKIP_RETURN_CODE 77)
        add_test(
            NAME vlacpp-tiny-safetensors-convert
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/convert-openpi-to-gguf.py"
                --checkpoint "${VLACPP_TINY_SAFETENSORS}"
                --output "${VLACPP_TINY_SAFETENSORS_GGUF}"
        )
        add_test(
            NAME vlacpp-tiny-safetensors-inspect
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/inspect-safetensors.py"
                "${VLACPP_TINY_SAFETENSORS}"
                --contains "pi0.velocity"
                --limit 10
        )
        add_test(
            NAME vlacpp-tiny-safetensors-map
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/map-openpi-tensors.py"
                "${VLACPP_TINY_SAFETENSORS}"
                --family tiny-velocity
                --require-complete
        )
        add_test(
            NAME vlacpp-tiny-safetensors-compare
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/compare-openpi-reference.py"
                --checkpoint "${VLACPP_TINY_SAFETENSORS}"
                --model "${VLACPP_TINY_SAFETENSORS_GGUF}"
                --binary "$<TARGET_FILE:vlacpp-pi0>"
                --state "1,2,3"
                --prompt "pick up the fork"
                --steps 4
                --seed 1
        )
        set_tests_properties(vlacpp-tiny-safetensors-convert PROPERTIES DEPENDS vlacpp-tiny-safetensors-create)
        set_tests_properties(vlacpp-tiny-safetensors-inspect PROPERTIES DEPENDS vlacpp-tiny-safetensors-create)
        set_tests_properties(vlacpp-tiny-safetensors-map PROPERTIES DEPENDS vlacpp-tiny-safetensors-create)
        set_tests_properties(vlacpp-tiny-safetensors-compare PROPERTIES DEPENDS vlacpp-tiny-safetensors-convert)
    endif()
endif()
