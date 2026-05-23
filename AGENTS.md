# AGENTS.md

## Cursor Cloud specific instructions

### Environment overview

SGLang is a high-performance LLM/multimodal serving framework. The main components are:

| Component | Path | Language | Build command |
|-----------|------|----------|---------------|
| SGLang Runtime | `python/` | Python | `uv pip install -e "python[dev]" --index-strategy unsafe-best-match --prerelease allow` |
| Model Gateway | `sgl-model-gateway/` | Rust | `CXX=g++ cargo build --features vendored-openssl` |
| sgl-kernel | `sgl-kernel/` | CUDA/C++ | Requires GPU; install pre-built from PyPI |
| Rust gRPC bridge | `rust/sglang-grpc/` | Rust | Built automatically during Python editable install via setuptools-rust |

### Cloud VM gotchas

- **No GPU available.** The Cloud Agent VM has no NVIDIA GPU. CUDA-dependent features (model serving, sgl-kernel compilation, flashinfer JIT) will not work. CPU tests and the Rust gateway work fine.
- **Default C++ compiler is clang**, which cannot find `<cstdint>` headers. Set `CXX=g++` when building the Rust gateway to avoid `esaxx-rs` compilation failures. The linker also needs `libstdc++.so` symlinked to `/usr/lib/x86_64-linux-gnu/` (already done in snapshot, but if lost: `sudo ln -sf /usr/lib/gcc/x86_64-linux-gnu/13/libstdc++.so /usr/lib/x86_64-linux-gnu/libstdc++.so`).
- **uv is installed to `~/.local/bin`**. Ensure `PATH` includes `/home/ubuntu/.local/bin`. For system-wide installs, use `sudo -E env "PATH=$PATH" UV_SYSTEM_PYTHON=1 UV_LINK_MODE=copy uv pip install ...`.

### Running lint, tests, and builds

- **Lint**: `SKIP=no-commit-to-branch pre-commit run --all-files` (see `.pre-commit-config.yaml` for hooks)
- **CPU tests**: `cd test && python3 run_suite.py --hw cpu --suite stage-a-test-cpu` (see `test/README.md` for full details)
- **Gateway tests**: `cd sgl-model-gateway && CXX=g++ cargo test --features vendored-openssl` (1 redis test requires `redis-server` binary)
- **Gateway build**: `cd sgl-model-gateway && CXX=g++ cargo build --features vendored-openssl`
- **Gateway run**: `./sgl-model-gateway/target/debug/sgl-model-gateway --port 30000 --host 127.0.0.1`

### LLM server (requires GPU)

The SGLang server requires a GPU. Launch with: `python -m sglang.launch_server --model-path <MODEL>` or `sglang serve --model <MODEL>`. This is not possible on the Cloud Agent VM.
