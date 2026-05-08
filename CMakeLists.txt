cmake_minimum_required(VERSION 3.16)
project(vlacpp VERSION 0.1.0 LANGUAGES C CXX)

option(VLACPP_BUILD_EXAMPLES "Build vlacpp examples" ON)
option(VLACPP_BUILD_TESTS "Build vlacpp tests" ON)
option(VLACPP_USE_LLAMA_CPP "Use vendored llama.cpp/ggml submodule when present" ON)

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

if(VLACPP_USE_LLAMA_CPP AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/CMakeLists.txt")
    set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    add_subdirectory(third_party/llama.cpp EXCLUDE_FROM_ALL)
    if(TARGET ggml)
        target_link_libraries(vlacpp PRIVATE ggml)
        target_compile_definitions(vlacpp PRIVATE VLACPP_HAVE_GGML=1)
    endif()
endif()

if(VLACPP_BUILD_EXAMPLES)
    add_executable(vlacpp-pi0 examples/pi0-cli/main.cpp)
    target_link_libraries(vlacpp-pi0 PRIVATE vlacpp)
endif()

if(VLACPP_BUILD_TESTS)
    enable_testing()
    add_executable(vlacpp-tests tests/test_runtime.cpp)
    target_link_libraries(vlacpp-tests PRIVATE vlacpp)
    add_test(NAME vlacpp-runtime COMMAND vlacpp-tests)

    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        set(VLACPP_TINY_CHECKPOINT "${CMAKE_CURRENT_SOURCE_DIR}/tests/tiny-openpi-checkpoint.json")
        set(VLACPP_TINY_GGUF "${CMAKE_CURRENT_BINARY_DIR}/tiny-openpi.gguf")
        set(VLACPP_ACTION_HEAD_CHECKPOINT "${CMAKE_CURRENT_SOURCE_DIR}/tests/action-head-checkpoint.json")
        set(VLACPP_ACTION_HEAD_GGUF "${CMAKE_CURRENT_BINARY_DIR}/action-head.gguf")
        set(VLACPP_ACTION_HEAD_SAFETENSORS "${CMAKE_CURRENT_BINARY_DIR}/action-head-openpi-names.safetensors")
        set(VLACPP_ACTION_HEAD_MANIFEST "${CMAKE_CURRENT_BINARY_DIR}/action-head-openpi-map.json")
        set(VLACPP_ACTION_HEAD_INVENTORY_MANIFEST "${CMAKE_CURRENT_BINARY_DIR}/action-head-openpi-inventory-map.json")
        set(VLACPP_ACTION_HEAD_MAPPED_GGUF "${CMAKE_CURRENT_BINARY_DIR}/action-head-openpi-mapped.gguf")
        set(VLACPP_ACTION_HEAD_NORM_STATS "${CMAKE_CURRENT_SOURCE_DIR}/tests/action-head-norm-stats.json")
        set(VLACPP_ACTION_HEAD_MAPPED_NORM_GGUF "${CMAKE_CURRENT_BINARY_DIR}/action-head-openpi-mapped-norm.gguf")
        set(VLACPP_PI05_ACTION_HEAD_CHECKPOINT "${CMAKE_CURRENT_SOURCE_DIR}/tests/pi05-action-head-checkpoint.json")
        set(VLACPP_PI05_ACTION_HEAD_GGUF "${CMAKE_CURRENT_BINARY_DIR}/pi05-action-head.gguf")
        set(VLACPP_PI05_ACTION_HEAD_SAFETENSORS "${CMAKE_CURRENT_BINARY_DIR}/pi05-action-head-openpi-names.safetensors")
        set(VLACPP_PI05_ACTION_HEAD_MANIFEST "${CMAKE_CURRENT_BINARY_DIR}/pi05-action-head-openpi-map.json")
        set(VLACPP_PI05_ACTION_HEAD_INVENTORY_MANIFEST "${CMAKE_CURRENT_BINARY_DIR}/pi05-action-head-openpi-inventory-map.json")
        set(VLACPP_PI05_ACTION_HEAD_MAPPED_GGUF "${CMAKE_CURRENT_BINARY_DIR}/pi05-action-head-openpi-mapped.gguf")
        set(VLACPP_TINY_SAFETENSORS "${CMAKE_CURRENT_BINARY_DIR}/tiny-openpi.safetensors")
        set(VLACPP_TINY_SAFETENSORS_GGUF "${CMAKE_CURRENT_BINARY_DIR}/tiny-openpi-safetensors.gguf")
        set(VLACPP_PI05_F16_SAFETENSORS "${CMAKE_CURRENT_BINARY_DIR}/pi05-action-head-f16.safetensors")
        set(VLACPP_PI05_F16_GGUF "${CMAKE_CURRENT_BINARY_DIR}/pi05-action-head-f16.gguf")
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
            NAME vlacpp-action-head-safetensors-create
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/make_action_head_safetensors.py"
                "${VLACPP_ACTION_HEAD_CHECKPOINT}"
                "${VLACPP_ACTION_HEAD_SAFETENSORS}"
        )
        set_tests_properties(vlacpp-action-head-safetensors-create PROPERTIES SKIP_RETURN_CODE 77)
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
        set_tests_properties(vlacpp-action-head-safetensors-inventory PROPERTIES DEPENDS vlacpp-action-head-safetensors-create)
        set_tests_properties(vlacpp-action-head-safetensors-inventory-check PROPERTIES DEPENDS vlacpp-action-head-safetensors-inventory)
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
