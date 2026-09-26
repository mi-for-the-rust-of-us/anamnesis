# Tests

## Integration Tests

Every `tests/*.rs` file is its own test crate. "Feature gate" is the
file-level `#![cfg(...)]`; "none" means the file always compiles (some
of those still gate individual tests or arms on a format feature, noted
in the description). Files marked *ad hoc* are `#[ignore]`d measurement
harnesses, run explicitly with `--ignored --nocapture`, never by plain
`cargo test`.

### Correctness and cross-validation

| File | Feature gate | What it tests |
|------|-------------|---------------|
| `cli.rs` | `cli` | CLI binary (`anamnesis`/`amn`): subcommand routing, argument parsing, output format, error messages, `--version` |
| `cli_convert.rs` | `cli` | `amn convert` smoke tests: argument parsing, dispatch routing and end-to-end conversion through the CLI |
| `cross_validation.rs` | none | FP8 dequantization: bit-exact comparison against PyTorch on 7 real models (3 FP8 schemes × 3 scale dtypes) |
| `cross_validation_gptq.rs` | `gptq` | GPTQ dequantization: bit-exact comparison against PyTorch on 4 fixtures from 2 models (2 quantizers × 2 bit widths) |
| `cross_validation_awq.rs` | `awq` | AWQ dequantization: bit-exact comparison against PyTorch on 2 real models (AutoAWQ GEMM, 4-bit) |
| `cross_validation_bnb.rs` | `bnb` | BitsAndBytes dequantization: bit-exact comparison against PyTorch on 7 fixtures from 4 models (Llama-3.2-1B NF4, FP4, double-quant NF4 and INT8; Qwen3 FP4; Qwen2.5-1.5B and Phi-3.5-mini double-quant NF4) |
| `cross_validation_bnb_encode.rs` | `bnb` | BitsAndBytes **encode**, validated against the same `bnb_reference/` fixtures as the decode suite |
| `cross_validation_gguf.rs` | `gguf` | GGUF block-quant dequantization: bit-exact comparison against the `gguf` Python package (`ggml-org` reference) on **all 22 kernels** from 4 real models (SmolLM2-135M, TinyLlama-1.1B, Mistral-7B-v0.3, Qwen2.5-0.5B) plus 3 synthetic fixtures. Since v0.7.3 each kernel is checked at **both `BF16` and full `F32` width**, the latter with no tolerance at all |
| `cross_validation_ollama.rs` | `gguf` | GGUF dequantization on an Ollama-distributed blob (`llama3.2:1b`, `Q8_0`) against the `gguf` Python package |
| `cross_validation_pth.rs` | `pth` | PyTorch `.pth` parsing: byte-exact comparison against PyTorch on 3 AlgZoo models (RNN + Transformer, both newer and older ZIP prefix formats) |
| `cross_validation_npz.rs` | `npz` | `parse_npz` against NumPy on a Gemma Scope 2B SAE slice (5 `F32` arrays) |
| `cross_validation_safetensors.rs` | none | safetensors header parsing (slice and reader entry points) against the upstream `safetensors` Python library, one fixture per detected scheme (FP8, GPTQ, AWQ, BnB NF4); the non-FP8 cases are gated per test |
| `cross_validation_convert.rs` | `npz`, `pth`, `gguf`, `bnb` (all four) | Cross-format `convert` round trips: every conversion pair through the low-level writers, plus the Phase 6.14 cells through the public `convert` entry point |
| `remember_orientation.rs` | none | `remember_to_bytes` emits every `nn.Linear` weight in `[out_features, in_features]` orientation; tests gated per test on `gptq` / `awq` / `bnb` |

### Contracts and invariants

| File | Feature gate | What it tests |
|------|-------------|---------------|
| `no_panic.rs` | none | No public parse/inspect entry point panics on malformed input (owned-bytes, reader and path/mmap variants); per-format tests gated on `gguf` / `pth` / `npz` |
| `panic_profile.rs` | none | `[profile.release]` builds with `panic = "abort"`, `[profile.python]` with unwind |
| `parallel_contract.rs` | none | Compile-time `Send`/`Sync` guards for the parsed-model types the parallel dispatch shares across threads |
| `parse_owned_path.rs` | none | Path/mmap, `parse_*_bytes` and `parse_*_from_reader` parse identically; malformed bytes and tight `ParseLimits` yield a clean `Err` |
| `python_ownership_contract.rs` | none | Owned extraction outlives a dropped `Parsed*` (the Phase 8 `PyO3` data-ownership contract) |

### Peak-heap assertions (`dhat`, *ad hoc*)

See [`peak_heap_README.md`](peak_heap_README.md) for the claims, thresholds and baselines.

| File | Feature gate | What it measures |
|------|-------------|------------------|
| `peak_heap_gptq.rs` | `gptq` | GPTQ dequant peak heap is `output_size + O(out_features)` |
| `peak_heap_awq.rs` | `awq` | Same claim for AWQ |
| `peak_heap_bnb_dq.rs` | `bnb` | BnB double-quant dequant allocates no intermediate byte buffer |
| `peak_heap_gguf.rs` | `gguf` | GGUF dequant peak equals the output size, per output dtype |
| `peak_heap_zip_metadata.rs` | `pth` | Vendored ZIP central-directory reader vs the `zip` crate: resident metadata heap |

### Benchmarks (*ad hoc*)

| File | Feature gate | What it measures |
|------|-------------|------------------|
| `bench_convert_adhoc.rs` | `npz`, `gguf`, `bnb` (all three) | `dhat` peak/cumulative heap of three `convert` routes (synthetic fixtures) |
| `bench_dequant_adhoc.rs` | none (arms gated on `gguf` / `bnb` / `gptq` / `awq`) | Dequantization kernel throughput on synthetic layer-sized inputs |
| `bench_gguf_convert_adhoc.rs` | `gguf` | Phase 7.2 scaling of the parallel GGUF-input `convert` path, and the calibration of `parallel::MIN_PARALLEL_BYTES` |
| `bench_gguf_inspect_adhoc.rs` | `gguf` | `parse_gguf(path).inspect()` (mmap) vs `inspect_gguf_from_reader` on the downloaded GGUF models |
| `bench_npz_adhoc.rs` | `npz` | `parse_npz` on the real 302 MB Gemma Scope file |
| `bench_parse_adhoc.rs` | none | `anamnesis::parse` + `inspect` on a multi-GB safetensors file from the local `HuggingFace` cache |
| `bench_pass2_adhoc.rs` | none (arms gated on `gguf`) | Phase 7 Stage 0: the shared pass-2 `f32 -> BF16` writer and the kernels feeding it |
| `bench_pth_adhoc.rs` | `pth` | `.pth` parsing and phase profiling on three torchvision checkpoints (see `pth_benchmark/` below) |
| `bench_pth_inspect_adhoc.rs` | `pth` | `parse_pth(path).inspect()` (mmap) vs `inspect_pth_from_reader`, on the in-tree AlgZoo fixtures and optionally a corpus named by `ANAMNESIS_ALGZOO_DIR` |

### Shared helpers: `tests/common/`

Cargo does not treat a subdirectory as a test target, so code several
test crates need lives in `tests/common/` and is pulled in with
`mod common;`. Only code that was identical across its copies lives
there:

| Module | Contents |
|--------|----------|
| `common/fixture.rs` | Little-endian readers for the checked-in binary fixture containers |
| `common/bf16.rs` | `BF16` golden comparison and synthesis |
| `common/bnb.rs` | The `AMNB` v2 BnB fixture container, shared by the decode and encode suites |
| `common/gguf.rs` | GGUF type mapping and `BF16` check shared by `cross_validation_gguf.rs` and `cross_validation_ollama.rs` |
| `common/builders.rs` | In-memory synthetic fixture builders for the `convert` suites |
| `common/heap.rs` | `dhat` helpers for the `peak_heap_*.rs` binaries |
| `common/bench.rs` | Timing and synthetic-fixture helpers for the `bench_*_adhoc.rs` harnesses |

## Fixtures

Most `fixtures/<scheme>_reference/` directories contain:

- a Python generator script (names vary per directory, see the table) that extracts a small slice from a real quantized model, dequantizes it with the canonical library (PyTorch, bitsandbytes, `gguf`, NumPy), and writes a binary fixture file
- `*.bin`: binary fixture files containing packed input tensors + the expected output

The fixture format is scheme-specific (documented in each generator script). All committed fixtures are in the repo so that `cargo test` works without downloading models or running Python. The exceptions are `pth_benchmark/` and `gguf_reference/models/`, which are gitignored and only needed by the *ad hoc* benchmarks.

| Directory | Models | Slice size | Generator(s) |
|-----------|--------|-----------|-----------|
| `fp8_reference/` | 7 models (EXAONE, Qwen3, Llama, Ministral) | 256x256 | `generate.py`; `regolden_f32.py` appends the `F32` golden to the committed `.bin` files in place |
| `gptq_reference/` | 4 fixtures from 2 models (Falcon3, Llama-3.2) | 256x256 | `generate_gptq.py` |
| `awq_reference/` | 2 models (Llama-3.2, Falcon3) | 256x256 | `generate_awq.py` |
| `bnb_reference/` | 7 fixtures from 4 models (Llama-3.2-1B NF4/FP4/NF4-double-quant/INT8, Qwen3 FP4, Qwen2.5-1.5B and Phi-3.5-mini double-quant NF4), each with a `.timing.json` Python-timing sidecar | 4096 elements (NF4/FP4), 256x256 (INT8) | `generate_bnb.py` |
| `npz_reference/` | Gemma Scope 2B SAE (small slice) | 5 F32 arrays | `generate_npz.py` |
| `gguf_reference/` | 23 fixtures: 20 from 4 models (SmolLM2-135M, TinyLlama-1.1B, Mistral-7B-v0.3, Qwen2.5-0.5B) plus 3 synthetic (`TQ1_0`, `TQ2_0`, `MXFP4`) | 65 536 elements (2 048 / 256 blocks) | `generate_gguf.py`; `generate_gguf_dequant_timings.py` writes the `*.dequant.timing.json` Python baselines. The source `.gguf` files go in `models/` (gitignored) |
| `ollama_reference/` | 1 model (`llama3.2:1b`, `Q8_0`, from the local Ollama cache) | 65 536 elements | `generate_ollama_fixture.py` |
| `pth_reference/` | 3 AlgZoo models (see below) | Full model (10–432 params) | `generate_pth_reference.py`; `bench_python_inspect.py` is the Python-side timing baseline |
| `safetensors_reference/` | 4 synthetic `.safetensors` files (FP8, GPTQ, AWQ, BnB NF4), each with a `<scheme>.expected.json` reference from the `safetensors` Python library | Tiny tensors (header-level test) | `generate.py` |
| `convert_reference/` | No fixtures: optional `*.timing.json` Python-timing sidecars for `cross_validation_convert.rs` (see its `README.md`) | n/a | `generate_convert_timings.py` |
| `pth_benchmark/` | 3 torchvision checkpoints (`resnet18`, `resnet50`, `vit_b_16`); **gitignored**, fetched once for `bench_pth_adhoc.rs` (recipe in that file's module doc) | Full model | none (one-line `torch.save` recipe) |

### AlgZoo `.pth` Fixtures

The `.pth` fixtures are real models from [AlgZoo](https://github.com/alignment-research-center/alg-zoo)
(Alignment Research Center, MIT-0 license), downloaded from their public
GCS bucket (`gs://arc-ml-public/alg/zoo/`). All are F32, tiny (2–3.5 KB):

| Fixture | AlgZoo source | Architecture | ZIP prefix | Params |
|---------|--------------|-------------|-----------|--------|
| `algzoo_rnn_small.pth` | `2nd_argmax_2_2_1073741824_2` | RNN (3 tensors) | `archive/` (newer PyTorch) | 10 |
| `algzoo_transformer_small.pth` | `longest_cycle_2_3_1073741824_0` | Transformer (7 tensors) | `archive/` (newer PyTorch) | 50 |
| `algzoo_rnn_blog.pth` | `one_layer_16_hidden_94_acc_2nd_argmax` | RNN (3 tensors) | `{model_name}/` (older PyTorch) | 432 |

Reference JSON manifests are generated by `generate_pth_reference.py` (requires
`torch` and `safetensors`). The Rust tests compare tensor names, shapes, dtypes,
and **raw bytes** against the Python-generated reference (0 byte difference expected).

## Regenerating Fixtures

Fixtures only need regeneration if the dequantization formula changes. To regenerate:

1. Download test models via `hf-fm` (see generator scripts for model IDs)
2. Run the directory's generator from the table above, e.g. `python tests/fixtures/gptq_reference/generate_gptq.py` or `python tests/fixtures/fp8_reference/generate.py`
3. Commit the updated `.bin` files

## Running Tests

```powershell
# All tests (requires all features)
cargo test --all-features

# Specific scheme
cargo test --features gptq --test cross_validation_gptq

# Release mode with AVX2 (for timing comparison)
$env:RUSTFLAGS="-C target-cpu=native"; cargo test --release --all-features -- --nocapture; $env:RUSTFLAGS=$null
```

In bash, the last line is `RUSTFLAGS="-C target-cpu=native" cargo test --release --all-features -- --nocapture`.

The CLI suites (`cli.rs`, `cli_convert.rs`) locate the binary through
`CARGO_BIN_EXE_*`, so Cargo builds it for them: no prebuilt binary is
needed, only the `cli` feature (included in `--all-features`).
