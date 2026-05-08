# Third-party dependencies

`third_party/llama.cpp` is a git submodule:

```sh
git submodule update --init --recursive
```

The current runtime still builds without the submodule, but when it is present
`VLACPP_USE_LLAMA_CPP=ON` links the `ggml` target for the pi0 graph work.
