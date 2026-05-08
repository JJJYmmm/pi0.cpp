# pi0 Inference Progress Audit

This audit tracks the user goal: initialize a branch and continue pi0 inference
end to end from a small HF checkpoint through GGUF conversion, model forward,
sampling, and OpenPI comparison.

## Success Criteria

| Requirement | Current Evidence | Status |
| --- | --- | --- |
| Git branch initialized | `git branch --show-current` reports `pi0-infer`; baseline commit is `cd35c54` (`pi0: add restricted gguf inference path`). | Done |
| HF/ModelScope checkpoint discovery/download path | `tools/inspect-safetensors.py` and `tools/run-action-head-smoke.py` consume `hf://maxqualia/openpi-pi0-corkinbox100-1882950e/model.safetensors` with range reads; `ms://lerobot/pi0/model.safetensors` is supported for ModelScope mirrors and real pi0 action-head range-read conversion; pi0.5 action-head smoke was verified against `hf://Tacoin/openpi-pi0.5-libero-onnx/checkpoints/pi05_libero_pytorch/model.safetensors`; `convert-openpi-to-gguf.py --norm-stats` consumes HF pi0.5 Libero config/norm stats. | Partial |
| GGUF conversion | `tools/convert-openpi-to-gguf.py` writes GGUF through `tools/gguf_writer.py` for tiny velocity tensors, tiny safetensors, local/remote mapped OpenPI action-head tensors, BF16 remote tensors, and F16 local tensors. Tensor-map manifests preserve `vlacpp.metadata` from safetensors headers when present, so mapped local fixtures infer dimensions without command-line overrides. Metadata-less mapped OpenPI action-head shards infer `action_dim` from action projection tensors, `state_dim` from `state_proj.weight` when available, and visible OpenPI graph dimensions into `vlacpp.openpi.*` metadata keys. `tools/inspect-gguf.py` verifies emitted tensor names/shapes. Real pi0 HF inventory currently maps 10/777 tensors for the action-head subset, while `--family all` can generate a 777/777 identity manifest from the same header. `--family pi0-full` and `--family pi05-full` preserve every tensor while aliasing currently supported head tensors to runtime names. Real pi0.5 HF action-head inventory maps 8/812 tensors. Runtime support remains partial. | Partial |
| Model forward | `src/models/pi0.cpp` implements mock/tiny velocity forward, restricted pi0 state/action-head forward, and restricted pi0.5 action/time-head forward. Full SigLIP/PaliGemma/Gemma backbone is not implemented. | Partial |
| Flow sampling | `src/sampling/flow.cpp` Euler flow sampler is wired into mock, tiny velocity, and action-head paths. | Done for implemented paths |
| OpenPI comparison | `tools/compare-openpi-reference.py` compares tiny OpenPI-style math; `tools/compare-openpi-policy.py` can call official OpenPI policy API when installed and requires `full-openpi` capability by default; `tests/run_fake_openpi_policy_compare.py` validates both the restricted-model rejection and explicit subset-test override. Real official checkpoint parity has not been executed. | Partial |
| llama.cpp/ggml reuse | `third_party/llama.cpp` is now a required submodule gitlink; default CMake configures and links `ggml`, `llama`, and `mtmd` for `vlacpp`; full OpenPI graph wiring still remains. | Partial |

## Default Test Coverage

Run:

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Expected CTest coverage:

- `vlacpp-runtime`: public C API lifecycle and mock inference.
- `vlacpp-tiny-gguf-convert`: tiny JSON checkpoint to GGUF.
- `vlacpp-tiny-openpi-compare`: tiny GGUF runtime vs Python reference.
- `vlacpp-tiny-capability-info`: CLI capability reporting for tiny velocity models.
- `vlacpp-require-full-rejects-restricted`: `--require-full` rejects restricted
  runtime subsets instead of silently treating them as full OpenPI parity.
- `vlacpp-fake-openpi-policy-compare`: official OpenPI policy API shape through
  a fake package, including default rejection of restricted vlacpp models and
  the explicit `--allow-restricted-vlacpp` subset-test override.
- `vlacpp-official-openpi-compare-skip`: explicit skipped result when official OpenPI is not installed.
- `vlacpp-openpi-graph-summary`: `summarize-openpi-graph.py` extracts action
  width/dimension and backbone layer counts from a full OpenPI-style manifest,
  including ModelScope-style `model.` prefixes.
- `vlacpp-llama-components`: when the llama.cpp submodule is present, verifies
  the expected ggml/llama CMake targets and SigLIP/mtmd source files are
  available for the next graph integration step.
- `vlacpp-action-head-gguf-convert`: small mapped action-head JSON checkpoint to GGUF.
- `vlacpp-action-head-cli`: action-head GGUF runtime shape/finite smoke.
- `vlacpp-action-head-gguf-inspect`: GGUF tensor name/shape inspection.
- `vlacpp-action-head-reference-compare`: action-head runtime vs independent math reference.
- `vlacpp-action-head-safetensors-create`: generated OpenPI-named action-head
  safetensors fixture.
- `vlacpp-action-head-safetensors-map`: local OpenPI-named safetensors header to
  vlacpp tensor-map manifest, including embedded `vlacpp.metadata`.
- `vlacpp-action-head-prefixed-safetensors-map`: local OpenPI-named safetensors
  with the ModelScope-style `model.` prefix maps to the same vlacpp runtime
  names.
- `vlacpp-action-head-prefixed-safetensors-reference-compare`: ModelScope-style
  prefixed safetensors GGUF runtime vs independent math reference.
- `vlacpp-action-head-safetensors-inventory`: local tensor inventory with mapped
  coverage emitted by `map-openpi-tensors.py --include-inventory`, including
  top-level `family`, `expected_count`, `mapped_count`, and `coverage`.
- `vlacpp-action-head-safetensors-inventory-check`: verifies local inventory
  mapped/unmapped counts and groups.
- `vlacpp-action-head-safetensors-summary`: `summarize-tensor-map.py` asserts
  pi0 action-head manifest family and mapped/total/unmapped coverage.
- `vlacpp-action-head-safetensors-all-map`: local OpenPI-named safetensors
  identity manifest for all tensors in the fixture.
- `vlacpp-action-head-safetensors-all-summary`: `summarize-tensor-map.py`
  asserts the all-family identity manifest coverage.
- `vlacpp-action-head-safetensors-all-convert`: all-family identity manifest to
  GGUF, validating full-manifest conversion mechanics on a tiny fixture.
- `vlacpp-action-head-safetensors-pi0-full-map`: full pi0 manifest family that
  preserves every source tensor and aliases currently supported head tensors.
- `vlacpp-action-head-safetensors-pi0-full-summary`: `summarize-tensor-map.py`
  asserts the pi0-full family coverage.
- `vlacpp-action-head-safetensors-pi0-full-convert`: pi0-full manifest to GGUF.
- `vlacpp-action-head-safetensors-pi0-full-reference-compare`: pi0-full GGUF
  runtime vs independent action-head math reference.
- `vlacpp-action-head-mapped-safetensors-convert`: mapped safetensors payloads to
  GGUF using `--tensor-map-manifest`; this test intentionally relies on
  manifest metadata instead of passing model dimensions by command line.
- `vlacpp-action-head-mapped-safetensors-reference-compare`: mapped safetensors
  GGUF runtime vs independent math reference.
- `vlacpp-action-head-local-smoke`: local `run-action-head-smoke.py` chain using
  OpenPI-named safetensors, with output shape read from converted GGUF metadata.
- `vlacpp-action-head-mapped-safetensors-norm-convert`: mapped safetensors plus
  local norm_stats to GGUF.
- `vlacpp-action-head-mapped-safetensors-metadata`: `inspect-gguf.py` verifies
  model dimensions and state/action normalization metadata in the mapped GGUF.
- `vlacpp-action-head-mapped-safetensors-norm-reference-compare`: norm-stats
  mapped GGUF runtime vs independent math reference, covering state
  normalization and action denormalization in the restricted action-head path.
- `vlacpp-pi05-action-head-gguf-convert`: small pi0.5 action-head JSON checkpoint to GGUF.
- `vlacpp-pi05-action-head-cli`: pi0.5 action-head GGUF runtime shape/finite smoke.
- `vlacpp-pi05-action-head-reference-compare`: pi0.5 action-head runtime vs independent math reference.
- `vlacpp-pi05-action-head-safetensors-create`: generated OpenPI-named pi0.5
  action-head safetensors fixture.
- `vlacpp-pi05-action-head-safetensors-map`: local OpenPI-named pi0.5
  safetensors header to vlacpp tensor-map manifest.
- `vlacpp-pi05-action-head-safetensors-inventory`: local pi0.5 tensor inventory
  with mapped coverage emitted by `map-openpi-tensors.py --include-inventory`.
- `vlacpp-pi05-action-head-safetensors-inventory-check`: verifies local pi0.5
  inventory mapped/unmapped counts and groups.
- `vlacpp-pi05-action-head-safetensors-summary`: `summarize-tensor-map.py`
  asserts pi0.5 action-head manifest family and mapped/total/unmapped coverage.
- `vlacpp-pi05-action-head-safetensors-pi05-full-map`: full pi0.5 manifest
  family that preserves every source tensor and aliases currently supported
  action/time-head tensors.
- `vlacpp-pi05-action-head-safetensors-pi05-full-summary`:
  `summarize-tensor-map.py` asserts the pi05-full family coverage.
- `vlacpp-pi05-action-head-safetensors-pi05-full-convert`: pi05-full manifest to
  GGUF.
- `vlacpp-pi05-action-head-safetensors-pi05-full-reference-compare`: pi05-full
  GGUF runtime vs independent pi0.5 action-head math reference.
- `vlacpp-pi05-action-head-mapped-safetensors-convert`: mapped OpenPI-named
  pi0.5 safetensors payloads to GGUF.
- `vlacpp-pi05-action-head-mapped-safetensors-reference-compare`: mapped pi0.5
  GGUF runtime vs independent math reference.
- `vlacpp-pi05-f16-safetensors-create`: generated F16 pi0.5 safetensors fixture.
- `vlacpp-pi05-f16-safetensors-convert`: F16 pi0.5 safetensors to F32 GGUF.
- `vlacpp-pi05-f16-safetensors-cli`: F16-converted pi0.5 GGUF runtime shape/finite smoke.
- CLI `--info` and `vlacpp_model_capability()` report which restricted runtime
  subset is active, so subset smoke tests are not mistaken for full-model parity.
  CLI `--require-full` fails for these restricted subsets.
- `vlacpp_model_openpi_graph_info()` and CLI `--info` expose parsed
  `vlacpp.openpi.*` graph metadata from loaded GGUF models; the action-head CLI
  test verifies `action_width`.
- `vlacpp-tiny-safetensors-create`: generated safetensors tiny fixture.
- `vlacpp-tiny-safetensors-convert`: tiny safetensors to GGUF.
- `vlacpp-tiny-safetensors-inspect`: safetensors header inspection.
- `vlacpp-tiny-safetensors-map`: tiny safetensors mapping manifest.
- `vlacpp-tiny-safetensors-compare`: tiny safetensors GGUF runtime vs Python reference.

## Manual HF Action-Head Smoke

Run:

```sh
python3 tools/run-action-head-smoke.py \
  --binary build/vlacpp-pi0 \
  --work-dir /tmp/vlacpp-action-head-smoke
```

For the pi0.5 action/time-head subset:

```sh
python3 tools/run-action-head-smoke.py \
  --family pi05 \
  --binary build/vlacpp-pi0 \
  --work-dir /tmp/vlacpp-pi05-action-head-smoke
```

The pi0.5 smoke defaults to the Tacoin repo's HF config and norm_stats, so the
converted GGUF carries the policy's action horizon/dimension and action
normalization metadata for that restricted shard.

This performs:

1. HF safetensors header mapping for real OpenPI state/action-head tensors.
2. Remote range-read conversion of the mapped tensors to GGUF.
3. GGUF inspection.
4. `vlacpp-pi0` action-head forward and flow sampling.

The smoke intentionally converts only the action-head tensor subset; it does not
claim full model parity.

Latest verified smoke outputs:

- pi0:
  `hf://maxqualia/openpi-pi0-corkinbox100-1882950e/model.safetensors`,
  `tensor_count=10`, `action_count=1024`, first actions
  `[2.37179, 0.0943651, 3.36542]`. This run used tensor-shape inference for
  `state_dim` and `action_dim`; only `action_horizon` was supplied.
  A header-only real pi0 `pi0-full` manifest check against the same HF file
  verified 777/777 mapped tensors, including 767/767
  `paligemma_with_expert` backbone tensors preserved by name and the supported
  head/state tensors aliased to runtime names.
  `ms://lerobot/pi0/model.safetensors` uses a ModelScope-style `model.` prefix;
  the action-head mapper now aliases those prefixed tensor names, and a
  range-read restricted GGUF smoke produced `horizon=32`, `action_dim=32`, and
  capability `restricted-pi0-state-action-head`.
  `summarize-openpi-graph.py` on the same ModelScope source reports action
  width 1024, action_dim 32, 27 contiguous vision tower layers, 18 contiguous
  PaliGemma language layers, and 18 contiguous action expert layers.
- pi0.5:
  `hf://Tacoin/openpi-pi0.5-libero-onnx/checkpoints/pi05_libero_pytorch/model.safetensors`,
  `tensor_count=8`, `action_count=320`, first actions
  `[-0.250423, 0.00458899, -0.101771]`. This run used tensor-shape inference
  for `action_dim` plus the HF config/norm_stats defaults for
  `action_horizon` and normalization metadata.

## Known Gaps

- Full OpenPI tensor mapping is incomplete. Current real-checkpoint mapping
  covers the state projection and action-head subset only. A real pi0 HF header
  inventory for
  `hf://maxqualia/openpi-pi0-corkinbox100-1882950e/model.safetensors` maps
  10/777 tensors; a real pi0.5 HF header inventory for
  `hf://Tacoin/openpi-pi0.5-libero-onnx/checkpoints/pi05_libero_pytorch/model.safetensors`
  maps 8/812 tensors. The remaining tensors are `paligemma_with_expert`.
  pi0 subgroup inventory breaks that down into
  `paligemma_with_expert.paligemma.model` (602 tensors),
  `paligemma_with_expert.gemma_expert.model.layers.*` (162 tensors), and a
  small number of head/norm tensors.
  `map-openpi-tensors.py --family all` can generate a 777/777 identity manifest
  for the real pi0 header. `--family pi0-full` and `--family pi05-full` preserve
  all header tensors while aliasing the currently implemented head tensors, but
  the runtime still cannot execute the preserved backbone tensors.
- Runtime does not implement SigLIP image encoder, PaliGemma/Gemma prefix/suffix
  transformer, KV cache execution, or full pi0.5 AdaRMS conditioning. The
  restricted pi0.5 path only approximates the time-conditioned action head.
- `tools/summarize-openpi-graph.py` now identifies the graph dimensions needed
  for that implementation from a full tensor manifest/source, but it is only a
  planning/introspection tool and does not execute the graph.
- `tools/compare-openpi-policy.py` has a real official OpenPI API path and an
  explicit missing-OpenPI skip mode. It rejects restricted vlacpp models unless
  `--allow-restricted-vlacpp` is passed for harness tests, but it has not been
  run against an installed OpenPI environment and a fully converted checkpoint.
- Passing default tests verifies implemented subsets only; it is not evidence of
  complete pi0/pi0.5 inference parity.
