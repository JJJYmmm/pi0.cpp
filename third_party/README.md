# Third-party dependencies

`third_party/llama.cpp` is expected to be a git submodule:

```sh
git submodule add https://github.com/ggerganov/llama.cpp.git third_party/llama.cpp
git submodule update --init --recursive
```

The current scaffold builds without the submodule so the public API, converter
shape, and smoke tests can be developed before the real ggml graph is wired in.
