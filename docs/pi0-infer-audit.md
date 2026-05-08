# pi0 Inference Progress Audit

This audit tracks the user goal: initialize a branch and continue pi0 inference
end to end from a small HF checkpoint through GGUF conversion, model forward,
sampling, and OpenPI comparison.

## Success Criteria

| Requirement | Current Evidence | Status |
| --- | --- | --- |
| Git branch initialized | `git branch --show-current` reports `pi0-infer`; baseline commit is `cd35c54` (`pi0: add restricted gguf inference path`). | Done |
| HF checkpoint discovery/download path | `tools/inspect-safetensors.py` and `tools/run-action-head-smoke.py` consume `hf://maxqualia/openpi-pi0-corkinbox100-1882950e/model.safetensors` with range reads; pi0.5 action-head smoke was verified against `hf://Tacoin/openpi-pi0.5-libero-onnx/checkpoints/pi05_libero_pytorch/model.safetensors`; `convert-openpi-to-gguf.py --norm-stats` consumes HF pi0.5 Libero config/norm stats. | Partial |
| GGUF conversion | `tools/convert-openpi-to-gguf.py` writes GGUF for tiny velocity tensors, tiny safetensors, local/remote mapped OpenPI action-head tensors, BF16 remote tensors, and F16 local tensors. Tensor-map manifests preserve `vlacpp.metadata` from safetensors headers when present, so mapped local fixtures infer dimensions without command-line overrides. Metadata-less mapped OpenPI action-head shards infer `action_dim` from action projection tensors and `state_dim` from `state_proj.weight` when available. `tools/inspect-gguf.py` verifies emitted tensor names/shapes. Real pi0 HF inventory currently maps 10/777 tensors; 767 unmapped tensors are under `paligemma_with_expert`. | Partial |
| Model forward | `src/models/pi0.cpp` implements mock/tiny velocity forward, restricted pi0 state/action-head forward, and restricted pi0.5 action/time-head forward. Full SigLIP/PaliGemma/Gemma backbone is not implemented. | Partial |
| Flow sampling | `src/sampling/flow.cpp` Euler flow sampler is wired into mock, tiny velocity, and action-head paths. | Done for implemented paths |
| OpenPI comparison | `tools/compare-openpi-reference.py` compares tiny OpenPI-style math; `tools/compare-openpi-policy.py` can call official OpenPI policy API when installed and requires `full-openpi` capability by default; `tests/run_fake_openpi_policy_compare.py` validates both the restricted-model rejection and explicit subset-test override. Real official checkpoint parity has not been executed. | Partial |

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
- `vlacpp-action-head-gguf-convert`: small mapped action-head JSON checkpoint to GGUF.
- `vlacpp-action-head-cli`: action-head GGUF runtime shape/finite smoke.
- `vlacpp-action-head-gguf-inspect`: GGUF tensor name/shape inspection.
- `vlacpp-action-head-reference-compare`: action-head runtime vs independent math reference.
- `vlacpp-action-head-safetensors-create`: generated OpenPI-named action-head
  safetensors fixture.
- `vlacpp-action-head-safetensors-map`: local OpenPI-named safetensors header to
  vlacpp tensor-map manifest, including embedded `vlacpp.metadata`.
- `vlacpp-action-head-safetensors-inventory`: local tensor inventory with mapped
  coverage emitted by `map-openpi-tensors.py --include-inventory`.
- `vlacpp-action-head-safetensors-inventory-check`: verifies local inventory
  mapped/unmapped counts and groups.
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
- `vlacpp-pi05-f16-safetensors-create`: generated F16 pi0.5 safetensors fixture.
- `vlacpp-pi05-f16-safetensors-convert`: F16 pi0.5 safetensors to F32 GGUF.
- `vlacpp-pi05-f16-safetensors-cli`: F16-converted pi0.5 GGUF runtime shape/finite smoke.
- CLI `--info` and `vlacpp_model_capability()` report which restricted runtime
  subset is active, so subset smoke tests are not mistaken for full-model parity.
  CLI `--require-full` fails for these restricted subsets.
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
  10/777 tensors; the remaining 767 tensors are `paligemma_with_expert`.
- Runtime does not implement SigLIP image encoder, PaliGemma/Gemma prefix/suffix
  transformer, KV cache execution, or full pi0.5 AdaRMS conditioning. The
  restricted pi0.5 path only approximates the time-conditioned action head.
- `tools/compare-openpi-policy.py` has a real official OpenPI API path and an
  explicit missing-OpenPI skip mode. It rejects restricted vlacpp models unless
  `--allow-restricted-vlacpp` is passed for harness tests, but it has not been
  run against an installed OpenPI environment and a fully converted checkpoint.
- Passing default tests verifies implemented subsets only; it is not evidence of
  complete pi0/pi0.5 inference parity.
