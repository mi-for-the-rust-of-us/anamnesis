# anamnesis

[![CI](https://github.com/mi-for-the-rust-of-us/anamnesis/actions/workflows/ci.yml/badge.svg)](https://github.com/mi-for-the-rust-of-us/anamnesis/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/anamnesis.svg)](https://crates.io/crates/anamnesis) [![docs.rs](https://docs.rs/anamnesis/badge.svg)](https://docs.rs/anamnesis) [![MSRV](https://img.shields.io/badge/MSRV-1.88-blue.svg)](https://www.rust-lang.org) [![license](https://img.shields.io/crates/l/anamnesis.svg)](https://github.com/mi-for-the-rust-of-us/anamnesis#license) [![unsafe: deny](https://img.shields.io/badge/unsafe-deny_(mmap_only)-blue.svg)](https://github.com/rust-secure-code/safety-dance/)

**ἀνάμνησις**: *Parse any format, recover any precision.*

A framework-agnostic Rust **library and CLI** for tensor-file work: parse
`.safetensors`, `.gguf`, `.npz`, and PyTorch `.pth`; **dequantize** quantized
weights back to bit-exact `BF16` (FP8 · GPTQ · AWQ · BitsAndBytes · all 22 GGUF
block types); inspect any format header-only without loading weights; and convert
between formats, all hardened for untrusted input.

*Published on crates.io; pre-1.0, so the API may still evolve; see [CHANGELOG](CHANGELOG.md) and [ROADMAP](ROADMAP.md).*

## Table of Contents

- [Install](#install)
- [CLI Commands](#cli-commands)
- [Try it](#try-it)
- [Library quick start](#library-quick-start)
- [Parsing untrusted input](#parsing-untrusted-input)
- [Formats & quantization support](#formats--quantization-support)
- [Performance & validation](#performance--validation)
- [Documentation](#documentation)
- [What's next](#whats-next)
- [Used by](#used-by)
- [License](#license)
- [Development](#development)

> **New to anamnesis?**
> - **I have a model file and want to see inside it** → [`amn inspect`](#cli-commands), or the [Inspect before you parse](docs/tutorials/inspect-before-you-parse.md) tutorial: format, tensors, shapes, dtypes, size, header-only.
> - **I want to recover full precision from a quantized model** → [Dequantize a GGUF model to BF16](docs/tutorials/dequantize-a-gguf-model.md): a k-quant becomes standard `BF16` safetensors, loadable in candle / burn / tch.
> - **I want to move a model to another format** → [Convert a model between formats](docs/tutorials/convert-between-formats.md): any input → `safetensors` / `gguf` / `bnb-nf4` in one command, quantized inputs auto-chained through `BF16`.
> - **I'm parsing untrusted / user-uploaded files on a server** → [Parsing untrusted input](#parsing-untrusted-input): the `inspect → check → parse-under-limits` recipe: typed errors, no panic, no `SIGBUS`.
> - **I'm calling anamnesis from Rust** → [Library quick start](#library-quick-start).
> - **I'm waiting for Python** → `pip install anamnesis` lands in **v0.8.0** ([What's next](#whats-next)); the [interop contract](docs/python-interop.md) is already frozen.
>
> Common questions live in the [FAQ](docs/FAQ.md); every command and flag is in the [CLI Reference](docs/cli-reference.md).

## Install

```sh
cargo install anamnesis --features cli,pth,gguf
```

Installs both `anamnesis` and `amn` (short alias). Pick the formats and schemes
you need via feature flags (`gptq`, `awq`, `bnb`, `npz`, `pth`, `gguf`, `ollama`,
`indicatif`); the full list is in the [CLI Reference](docs/cli-reference.md#install).

## CLI Commands

| Command | |
|---------|---|
| `amn parse <file>` | Parse and summarize a model file (`.safetensors`, `.pth`, `.npz`, `.gguf`) |
| `amn inspect <file>` | Show format, tensor counts, size estimates, dtypes, byte order |
| `amn remember <file>` | Dequantize to BF16 (safetensors) or convert `.pth`/`.gguf` → `.safetensors` |
| `amn convert <file> --to <target>` | Convert any input to `safetensors` / `gguf` / `bnb-nf4` through one dispatch |

Aliases: `amn info` = `amn inspect`, `amn dequantize` = `amn remember`. Format
detection is automatic (by extension, then magic bytes for `.bin`). Every flag,
the full convert matrix, output-path rules, and the `ollama:` URL scheme are in
the [CLI Reference](docs/cli-reference.md).

## Try it

```
$ amn inspect model-fp8.safetensors       # what precision is in here?
Format:      Fine-grained FP8 (E4M3), 128x128 blocks
Quantized:   1 tensors (weights) + 1 scale tensors (F32)
Passthrough: 1 tensors (norms, embeddings)
Size:        96 B (FP8) -> 144 B (BF16)
Lethe took:  ~48 B of precision

$ amn remember model-fp8.safetensors      # recover it: FP8 → bit-exact BF16
Parsing...  3 tensors, Fine-grained FP8 (E4M3), 128x128 blocks
Recalling... 1 tensors
Output: model-bf16.safetensors (144 B)

$ amn inspect ollama:llama3.2:1b          # header-only on a real GGUF, no weights loaded
Format:      GGUF v3
Arch:        llama
Tensors:     147
Total size:  1.22 GB
Dtypes:      F32, Q8_0
Alignment:   32 bytes

$ amn convert model.npz --to safetensors  # any input → any target, one dispatch
Converting model.npz -> model-bf16.safetensors (NPZ -> safetensors)
  Wrote 5 tensors -> model-bf16.safetensors
```

The `ollama:` scheme (built with `--features ollama`) resolves
`ollama:<model>:<tag>` straight to the local Ollama cache's GGUF blob; see the
[CLI Reference](docs/cli-reference.md#ollama-url-scheme).

## Library quick start

```toml
[dependencies]
anamnesis = { version = "0.7", features = ["gguf", "pth"] }
```

Inspect any format header-only (no weight data loaded):

```rust
let info = anamnesis::parse_gguf("model.gguf")?.inspect();
println!("{} tensors, {} bytes", info.tensor_count, info.total_bytes);
```

Dequantize a quantized model to bit-exact `BF16` safetensors, in memory:

```rust
use anamnesis::{parse, TargetDtype};
let bytes = parse("model-fp8.safetensors")?.remember_to_bytes(TargetDtype::BF16)?;
```

For untrusted input, prefer the copy-based `parse_bytes` / `parse_*_from_reader`
entry points (see below). Full API on [docs.rs](https://docs.rs/anamnesis).

## Parsing untrusted input

A `.safetensors`, `.npz`, `.pth`, or `.gguf` file is attacker-controllable: it
can *declare* arbitrary tensor counts, shapes, lengths, or expansion ratios.
anamnesis is **parse-first** so a multi-tenant or edge host can reject a hostile
or too-large file against limits **it** chooses, before committing memory. The
recipe is **inspect → check policy → parse under limits**:

```rust
use anamnesis::{inspect_npz_from_reader, parse_npz_with_limits, ParseLimits};

// 1. Inspect cheaply: header-only, bounded; never materialises tensor data.
let info = inspect_npz_from_reader(&mut reader)?;

// 2. Reject early against YOUR environment's policy: no parse, no allocation.
if info.total_bytes > my_ram_budget || info.tensors.len() > my_tensor_cap {
    return Err(/* 413 Too Large */);
}

// 3. Parse under a `ParseLimits` matched to this worker, enforced fail-fast,
//    before each allocation.
let limits = ParseLimits::default()
    .with_max_total_bytes(my_ram_budget)
    .with_max_item_count(my_tensor_cap)
    .with_max_decompression_ratio(1000); // reject a DEFLATE entry inflating >1000×
let tensors = parse_npz_with_limits(path, &limits)?;
```

Every parser has a `parse_*_with_limits` sibling. The four `ParseLimits` axes
(all opt-in; `ParseLimits::default()` is unbounded, **tighten-only** over the
built-in per-format floors):

| Axis | Builder | Bounds |
|---|---|---|
| Single allocation | `with_max_single_alloc` | the largest single header-declared buffer |
| Cumulative heap | `with_max_total_bytes` | the running sum across the whole file (many-small-items blow-up) |
| Item count | `with_max_item_count` | declared tensors / arrays / KV entries / archive members |
| Decompression ratio | `with_max_decompression_ratio` | a compressed entry's uncompressed:compressed ratio (`.npz` zip-bomb cap) |

**Error taxonomy.** A rejection's *kind* is its `AnamnesisError` variant, so a
host branches without string-matching: a budget/cap breach (a `ParseLimits` axis
**or** an always-on permanent floor like `MAX_PKL_SIZE`) is `LimitExceeded { limit, message }`
(→ *413*); a malformed/truncated file is `Parse` (→ *400*); a `.pth` pickle
referencing a `GLOBAL` outside the `torch.*` allowlist is `DisallowedGlobal { module, name }`
(a security signal); a recognised-but-unimplemented format/dtype is `Unsupported`;
I/O failures are `Io`. The v0.8.0 Python bindings map these one-to-one:

| `AnamnesisError` | Python exception |
|---|---|
| `Parse` | `ParseError` |
| `Unsupported` | `UnsupportedError` |
| `LimitExceeded` | `LimitExceededError` |
| `DisallowedGlobal` | `SecurityError` |
| `Io` | builtin `OSError` |

**No panic, no abort.** No public parse/inspect entry point panics or aborts on
any input: a hostile file is always a clean `Err`, never an unwinding panic and
never a `SIGBUS` (the copy-based `parse_bytes` / `parse_*_from_reader` paths use
no memory map). Pinned in stable CI by `tests/no_panic.rs` and the `cargo fuzz`
harness.

Walkthrough: [Inspect before you parse](docs/tutorials/inspect-before-you-parse.md).
The per-version hardening history is in [Validation → robustness timeline](docs/validation.md#robustness-hardening-timeline).

## Formats & quantization support

| Format | Inspect | Parse | Dequantize → BF16 | Quantize | Convert to |
|---|:---:|:---:|---|---|---|
| **safetensors** | ✓ | ✓ | ✓ FP8 · GPTQ · AWQ · BitsAndBytes | ✓ BnB-NF4 (Lethe) | safetensors · gguf · bnb-nf4 |
| **GGUF** | ✓ | ✓ | ✓ all 22 block types | n/a | safetensors · gguf · bnb-nf4 |
| **NPZ** | ✓ | ✓ | n/a *(already full precision)* | n/a | safetensors · gguf · bnb-nf4 |
| **PyTorch `.pth`** | ✓ | ✓ | n/a | n/a | safetensors · gguf · bnb-nf4 |

Every dequant and quant kernel is **bit-exact (0 ULP)** against the canonical
library's *own* code (`bitsandbytes`, AutoAWQ, GPTQModel, PyTorch's fp8 cast,
`gguf-py`), not a hand-rolled oracle. Header-only inspection works over any
reader (in-memory, HTTP-range, custom transport): safetensors needs only `Read`;
the ZIP/GGUF formats need `Read + Seek`. `amn convert` routes every
`(input × target)` pair through an in-memory **BF16 hub**, so a quantised input
auto-dequantises on the way to any target and `gguf → gguf` recovers precision in
place (preserving the source metadata KV). Non-safetensors formats and each scheme
are feature-gated (`gguf`, `npz`, `pth`, `gptq`, `awq`, `bnb`).

→ Full tested-model tables, per-kernel speeds, conversion benchmarks, and
methodology: **[Validation & tested models](docs/validation.md)**.

## Performance & validation

Representative measured results (release build, `target-cpu=native`, best-of-5):

- **Dequantization:** 2.7–54× faster than the reference Python/PyTorch path, bit-exact (0 ULP) across FP8 / GPTQ / AWQ / BnB / 22 GGUF block types.
- **PyTorch `.pth` parsing:** **11–31× faster** than `torch.load()` on torchvision models; **NPZ** at **3.6 GB/s** (17.7× the `npyz` crate).
- **Conversion:** `npz → safetensors` 6.75×, `pth → safetensors` 5.18×, `safetensors → BnB-NF4` 2.67× vs the Python ecosystem default.
- **Multi-threaded dequant:** whole-model `remember` / `convert` run **~3–4× faster** across CPU cores via the default-on `parallel` feature (`std::thread::scope`, no new dependency), giving byte-identical output at any thread count, with a modest `min(cores, 4)` default (opt out with `default-features = false`, or tune with `--threads`). Since v0.7.2 the `GGUF`-input path is parallelised too, at **~1.9×** on the reader stage, lower than the safetensors path because the conversion hub owns one buffer per tensor, [measured and explained here](docs/perf-experiments.md).
- **vs the Python `GGUF` stack:** whole-model `GGUF` dequantisation is **17–28×** faster than [`gguf-py`](https://pypi.org/project/gguf/) single-threaded and **34–53×** at the default thread budget. **Not a like-for-like output:** `gguf-py` returns `float32`, anamnesis returns `BF16`, which is half the bytes and the narrower type. What is verified is that anamnesis's `BF16` is **bit-identical to `gguf-py`'s `float32` correctly rounded to `BF16`** (0 ULP, all 22 kernels), so the two agree on the numbers and differ only in the delivered width. Since that width difference is itself worth ~2× of memory traffic on a bandwidth-bound workload, halving `gguf-py`'s time as a generous correction still leaves ~9–14× and ~17–26×.

These are guarded against regression by [CodSpeed](https://codspeed.io/) continuous
benchmarking in CI, plus dev-only tracks: [Criterion runtime
benchmarks](benches/README.md), [`dhat-rs` peak-heap assertions](tests/peak_heap_README.md)
that hold each kernel to its documented `# Memory` ceiling, and an Ollama
cross-validation, none of which ship in the published crate.

Numbers, fixtures, and method: [Validation & tested models](docs/validation.md)
and [`docs/perf-experiments.md`](docs/perf-experiments.md).

## Documentation

| Doc | |
|-----|---|
| [CLI Reference](docs/cli-reference.md) | Every subcommand, flag, the convert matrix, output-path rules, the `ollama:` scheme |
| [FAQ](docs/FAQ.md) | Common questions: install, feature flags, formats, `inspect` vs `parse`, dequantizing/converting, untrusted input |
| [Validation & tested models](docs/validation.md) | Cross-validation tables, per-kernel speeds, conversion benchmarks, peak-heap assertions, hardening timeline |
| [Tutorial: Inspect before you parse](docs/tutorials/inspect-before-you-parse.md) | The `inspect → check → parse` safety pattern; rejecting a hostile file; bounding memory with `ParseLimits` |
| [Tutorial: Dequantize a GGUF model to BF16](docs/tutorials/dequantize-a-gguf-model.md) | `inspect` → `remember` → verify: a GGUF k-quant becomes standard `BF16` safetensors |
| [Tutorial: Convert a model between formats](docs/tutorials/convert-between-formats.md) | `amn convert --to`: any input → `safetensors` / `gguf` / `bnb-nf4` through the BF16 hub, with GGUF-KV stamping |
| [Python interop contract](docs/python-interop.md) | The frozen v0.8.0 binding contract: panic-safety, the NumPy/BF16 ownership model |
| [Performance experiments](docs/perf-experiments.md) | Measured perf hypotheses, methods, and outcomes (incl. rejected ones) |

## What's next

- **Phase 7.3, caller-chosen output dtype for `convert` (v0.7.3):** the dequantisation output type becomes a parameter (`BF16` / `F32` / `F16`), monomorphised like a C++ template argument, with a runtime dispatch at the API edge. `F32` drops the narrowing step entirely, so the output is the reference `f32` itself (today's `BF16` rounds 80–97 % of values), cross-validated against `gguf-py` with no rounding in between for the first time.
- **Phase 7.4, the same for `remember` (v0.7.4):** `TargetDtype` gains `F32` and `F16`, and the FP8 / GPTQ / AWQ / BnB kernels become generic over the output type. Lands before the bindings freeze a dtype contract, so a Python caller can ask for a plain `np.float32` array without an optional dependency.
- **Phase 8, Python bindings (v0.8.0):** `pip install anamnesis`, with typed exceptions, owned NumPy arrays, `ml_dtypes.bfloat16`. The [interop contract](docs/python-interop.md) is already frozen.

Full plan in [ROADMAP.md](ROADMAP.md); progress in [CHANGELOG.md](CHANGELOG.md).

## Used by

- [candle-mi](https://github.com/mi-for-the-rust-of-us/candle-mi): Mechanistic interpretability toolkit for language models
- [hf-fetch-model](https://github.com/mi-for-the-rust-of-us/hf-fetch-model): Download, inspect, and compare HuggingFace models from Rust; uses anamnesis to parse cached tensor-file metadata (`.safetensors` / GGUF / NPZ / PyTorch `.pth`) for `hf-fm inspect --cached`

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE)
or [MIT License](LICENSE-MIT) at your option.

## Development

- Exclusively developed with [Claude Code](https://claude.com/product/claude-code)
- Continuous performance benchmarking with [CodSpeed](https://codspeed.io/)
- Git workflow managed with [Fork](https://fork.dev/)
