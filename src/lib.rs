// SPDX-License-Identifier: MIT OR Apache-2.0

//! **ἀνάμνησις** — parse any tensor format, recover any precision.
//!
//! `anamnesis` is a framework-agnostic Rust library for dequantizing
//! quantized model weights and parsing tensor archives. It handles
//! `.safetensors` (memory-mapped, classify, dequantize to `BF16`),
//! `.npz` (bulk extraction at near-I/O speed), and `PyTorch` `.pth`
//! (zero-copy mmap with lossless safetensors conversion, 11–31× faster
//! than `torch.load()`).
//!
//! # Supported Quantization Schemes (decode — [`remember`])
//!
//! | Scheme | Feature gate | Speedup vs `PyTorch` CPU (AVX2) |
//! |--------|-------------|-------------------------------|
//! | `FP8` `E4M3` (fine-grained, per-channel, per-tensor) | *(always on)* | 2.7–9.7× |
//! | `GPTQ` (`INT4`/`INT8`, group-wise, `g_idx`) | `gptq` | 6.5–12.2× |
//! | `AWQ` (`INT4`, per-group, activation-aware) | `awq` | 4.7–5.7× |
//! | `BitsAndBytes` `NF4`/`FP4` (lookup + per-block absmax) | `bnb` | 18–54× |
//! | `BitsAndBytes` `INT8` (`LLM.int8()`, per-row absmax) | `bnb` | 1.2× |
//!
//! All schemes produce **bit-exact** output (0 ULP difference) against the
//! canonical quantization libraries' own dequantization code —
//! `bitsandbytes` (`dequantize_4bit` / `int8_vectorwise_dequant`), `AutoAWQ`
//! (`unpack_awq` + `reverse_awq_order`), `GPTQModel` (`dequantize_weight`
//! plus its v1→v2 zero-point conversion), and `PyTorch`'s native `fp8`
//! cast — verified on real-model fixtures. Hand-rolled reference
//! reimplementations are banned from the fixture generators (v0.6.4 rule;
//! see `docs/dogfooding-feedbacks/` for the circular-validation incident
//! that motivated it).
//!
//! # Supported Quantization Schemes (encode — [`lethe`])
//!
//! Phase 5 introduces the encode side as the architectural inverse of
//! [`remember`]. Each kernel here takes the `BF16` bytes that
//! [`remember`] produces and writes the corresponding quantised bytes
//! plus the per-block / per-row metadata (`absmax`, `SCB`, …), so
//! `encode(decode(q, scale)) == q` holds bit-exactly for every codebook-
//! `LUT` family (`NF4`, `FP4`) and within `i8` representation error
//! plus the documented clamp at `± 127` for `INT8`.
//!
//! | Scheme | Feature gate | Cross-validation contract |
//! |--------|-------------|---------------------------|
//! | `BitsAndBytes` `NF4`/`FP4` encode | `bnb` | byte-exact vs `bitsandbytes`' on-disk bytes on every fixture |
//! | `BitsAndBytes` `INT8` encode | `bnb` | byte-exact vs `bitsandbytes`' on-disk bytes on every fixture |
//!
//! Subsequent encode-kernel families (`FP8`, `GGUF` legacy / `K-quants`
//! / `IQ` / `TQ` / `MXFP4`) land in Phase 7.5 and reuse the
//! `lethe::round_trip` harness introduced here.
//!
//! # Format Conversion Pipeline (Phase 6, v0.6.0)
//!
//! `amn convert <input> --to <target>` routes any v0.6.0-available
//! format pair through a single CLI dispatch. The same pipeline is
//! exposed as a library through three new helper families:
//!
//! - `write_gguf` / `write_gguf_to_writer` / `GgufWriteTensor` — the
//!   format-symmetric inverse of `parse_gguf`. Phase 6 emits scalar
//!   dtypes only (`F32`, `F16`, `BF16`, `F64`, `I8`–`I64`); quantised
//!   emit (`Q*`, `IQ*`, `TQ*`, `MXFP4`) lands in Phase 7.5 through
//!   the same writer scaffold. Behind the `gguf` feature.
//! - `npz_to_safetensors` / `npz_to_safetensors_bytes` — lossless
//!   `NPZ → safetensors` conversion. Every `NpzDtype` variant maps
//!   directly to its `safetensors::Dtype` counterpart. Behind the
//!   `npz` feature.
//! - `write_bnb_nf4_safetensors` / `write_bnb_nf4_safetensors_bytes`
//!   / `BnbWriteInput` / `classify_inputs` / `is_eligible_for_nf4`
//!   — end-to-end `BF16 → BnB-NF4 safetensors file` path. Wraps the
//!   `encode_bnb4_compute_absmax` kernel into the four-tensor
//!   on-disk companion layout (`weight`, `weight.absmax`,
//!   `weight.quant_map`, `weight.quant_state.bitsandbytes__nf4`).
//!   2-D tensors only; 1-D biases / norms / embeddings pass through
//!   unchanged in `BF16`. Behind the `bnb` feature.
//!
//! | Conversion | anamnesis (CPU) | Python baseline (CPU) | Ratio |
//! |---|---:|---:|---:|
//! | `NPZ → safetensors` (4096×4096 F32) | 11.2 ms | 75.7 ms (numpy + safetensors-py) | 6.75× |
//! | `PTH → safetensors` (4096×4096 BF16) | 5.7 ms | 29.6 ms (torch.load + safetensors.torch) | 5.18× |
//! | `safetensors-BF16 → GGUF` (4096×4096 BF16) | 13.6 ms | 15.1 ms (gguf-py) | 1.11× |
//! | `safetensors-BF16 → BnB-NF4` (4096×4096 BF16) | 141 ms | 376.8 ms (bitsandbytes CPU) | 2.67× |
//!
//! Headline numbers measured by `t14_perf_vs_python_size_matched` in
//! `tests/cross_validation_convert.rs` at `target-cpu=native`, release,
//! best-of-5 median. Full table including PyTorch-CPU equivalents for
//! the two non-PyTorch paths is in the project README.
//!
//! # `Ollama` integration (Phase 6.5)
//!
//! Feature-gated behind `ollama` (implies `gguf`). Adds **no
//! third-party dependency** — pure stdlib + `serde_json` (already a
//! runtime dep). Exposes one function:
//!
//! - `resolve_ollama_model("llama3.2:1b") -> PathBuf` — reads the
//!   `Ollama` manifest at
//!   `~/.ollama/models/manifests/registry.ollama.ai/library/<name>/<tag>`
//!   and returns the `GGUF` blob path
//!   (`~/.ollama/models/blobs/sha256-<hash>`). The `OLLAMA_MODELS`
//!   env var overrides the cache root. Accepts the `ollama:name:tag`
//!   URL-scheme form for `amn` CLI integration.
//!
//! The `amn` CLI's `parse` / `inspect` / `remember` / `convert`
//! subcommands recognise the `ollama:` URL scheme prefix and resolve
//! transparently:
//!
//! ```text
//! amn inspect ollama:llama3.2:1b
//! amn remember ollama:gemma2:2b --to bf16 -o gemma2.safetensors
//! amn convert ollama:qwen2.5-coder:7b-instruct --to safetensors
//! ```
//!
//! `cross_validation_ollama.rs` cross-validates anamnesis's `GGUF`
//! dequant byte-exactly against the `gguf-py` reference on a slice
//! pulled from a real `Ollama`-cached blob — the same kernel anamnesis
//! already validates against bartowski / `TheBloke` quantisations,
//! now also validated on the dominant local-LLM distribution channel.
//!
//! # Validation infrastructure (Phase 6.5, dev-only)
//!
//! Phase 6.5 ships three dev-only validation tracks. None of them
//! affect the published crate (`benches/`, `tests/peak_heap_*.rs`,
//! and the `dhat` / `criterion` dev-dependencies are excluded from
//! the published tarball by Cargo's defaults).
//!
//! 1. **Criterion runtime benchmarks** (`benches/dequant.rs`,
//!    `benches/parsing.rs`) — throughput baselines per kernel family
//!    plus a real-world bench on the Ollama-cached `llama3.2:1b`
//!    `Q8_0` slice. Run via `cargo bench --features
//!    gptq,awq,bnb,gguf,npz,pth`. See
//!    `benches/README.md` for run commands + machine-spec baselines.
//! 2. **`dhat-rs` peak-heap assertions** (`tests/peak_heap_gptq.rs`,
//!    `tests/peak_heap_awq.rs`, `tests/peak_heap_bnb_dq.rs`) — three
//!    `#[ignore]`d test binaries that wrap the global allocator and
//!    assert observed peak heap stays within the documented
//!    `output_size + O(out_features)` (`GPTQ` / `AWQ`) or
//!    `output_size + num_blocks × 4 + block_size × 4` (`BnB`
//!    double-quant) ceiling. Each kernel's scratch matches the
//!    documented `# Memory` claim to the byte on the reference
//!    machine. See `tests/peak_heap_README.md` for calibration and
//!    failure-interpretation guidance.
//! 3. **Ollama-fixture cross-validation** (`tests/cross_validation_ollama.rs`)
//!    — bit-exact `Q8_0` dequant against `gguf-py` on a real Ollama
//!    blob (see the `Ollama` integration section above).
//!
//! # `NPZ`/`NPY` Parsing
//!
//! Feature-gated behind `npz`. Custom `NPY` header parser with bulk
//! `read_exact` — zero per-element deserialization for LE data on LE
//! machines. Supports `F16`, `BF16`, `F32`, `F64`, all integer types,
//! and `Bool`. **3,586 MB/s** on a 302 MB file (1.3× raw I/O overhead).
//!
//! # `PyTorch` `.pth` Parsing
//!
//! Feature-gated behind `pth`. Minimal pickle VM (~36 opcodes) with a
//! security allowlist (only `PyTorch` tensor-reconstruction globals are
//! permitted — the VM never invokes callables, so there is no code-execution
//! path) and **working-set governance**: the value stack, memo clones, and
//! nesting depth are charged against permanent floors
//! (`MAX_PICKLE_WORKING_SET`, `MAX_PICKLE_VM_DEPTH`) and the caller's
//! [`ParseLimits`], so a crafted `.pth` cannot drive multi-GiB heap or a
//! recursive-`Drop` stack overflow even on the cheap `inspect_pth_from_reader`
//! pre-filter. Memory-mapped I/O with zero-copy `Cow::Borrowed` tensor data.
//! Lossless `.pth` → `.safetensors` conversion. **11–31× faster** than
//! `torch.load()` on torchvision models.
//!
//! # Panic safety
//!
//! No public parse/inspect entry point panics or aborts on **any** input — a
//! malformed, truncated, or hostile artefact is always a clean `Result::Err`
//! ([`AnamnesisError`]), never an unwinding panic and never a `SIGBUS` (use the
//! copy-based [`parse_bytes`] / `parse_*_from_reader` paths for untrusted input,
//! which take an owned buffer instead of a memory map). The lint floor
//! (`unwrap_used` / `expect_used` / `panic` / `indexing_slicing` all denied) plus
//! `checked_*` arithmetic on every header-derived value enforce this in the
//! source; `tests/no_panic.rs` (a `catch_unwind` battery, run in debug so
//! integer-overflow panics are in scope) and the `cargo fuzz` harness pin it.
//!
//! Library and CLI release builds use `panic = "abort"` (a *reachable* panic
//! should fail the process closed, not unwind). The Phase 8 Python extension is
//! built with a separate `panic = "unwind"` profile so `PyO3` can surface a panic
//! as a catchable `PanicException`; see `docs/python-interop.md`.
//!
//! # Quick Start
//!
//! Path-based dequantisation (FP8 → BF16):
//!
//! ```rust,no_run
//! use anamnesis::{parse, TargetDtype};
//!
//! let model = parse("model-fp8.safetensors")?;
//! let info = model.inspect();
//! println!("{info}");
//!
//! model.remember("model-bf16.safetensors", TargetDtype::BF16)?;
//! # Ok::<(), anamnesis::AnamnesisError>(())
//! ```
//!
//! Reader-generic inspection over any `Read + Seek` substrate (in-memory
//! `Cursor`, `HTTP`-range-backed adapter, custom transport). The example
//! below uses a `std::fs::File`; an `HTTP`-range adapter from a
//! downstream crate (e.g. `hf-fm`'s `HttpRangeReader`) plugs in
//! identically — anamnesis itself stays HTTP-free. Four reader-generic
//! entry points cover the supported tensor formats:
//!
//! ```rust,no_run
//! # #[cfg(all(feature = "npz", feature = "pth", feature = "gguf"))]
//! # fn run() -> anamnesis::Result<()> {
//! use anamnesis::{
//!     inspect_gguf_from_reader, inspect_npz_from_reader,
//!     inspect_pth_from_reader, parse_safetensors_header_from_reader,
//! };
//!
//! let st_header = parse_safetensors_header_from_reader(
//!     std::fs::File::open("shard.safetensors")?,
//! )?;
//! let npz_info = inspect_npz_from_reader(std::fs::File::open("weights.npz")?)?;
//! let gguf_info = inspect_gguf_from_reader(std::fs::File::open("model.gguf")?)?;
//! let pth_info = inspect_pth_from_reader(std::fs::File::open("model.pth")?)?;
//! # let _ = (st_header, npz_info, gguf_info, pth_info);
//! # Ok(()) }
//! ```
//!
//! # Architecture
//!
//! - [`parse()`] — memory-map a `.safetensors` file into a
//!   [`ParsedModel`]. Inspect-only workflows touch only the header
//!   (~1 MiB) regardless of file size; full dequantisation pages
//!   tensor bytes in lazily.
//! - [`ParsedModel::inspect`] — derive format, tensor counts, and size
//!   estimates from the parsed header (zero further I/O)
//! - [`ParsedModel::remember`] — dequantize all quantized tensors to `BF16`
//!   and write a standard `.safetensors` file
//! - [`ParsedModel::remember_to_bytes`] — the same dequant, returning the
//!   `.safetensors` bytes in memory instead of writing a file (no disk
//!   round-trip for an embedder)
//! - [`parse_safetensors_header`] / [`parse_safetensors_header_from_reader`]
//!   — header-only safetensors parsing. The reader-generic variant accepts
//!   any `Read` substrate (in-memory `Cursor`, `HTTP`-range-backed adapter,
//!   …) and reads only the 8-byte length prefix plus the `JSON` header,
//!   so a multi-GB shard's metadata can be inspected with a single
//!   ~1 MiB sequential fetch.
//! - `parse_npz()` — read an `.npz` archive into a `HashMap<String, NpzTensor>`
//!   (requires `npz` feature)
//! - `inspect_npz()` / `inspect_npz_from_reader()` — header-only `NPZ`
//!   inspection. The reader-generic variant accepts any `Read + Seek`
//!   substrate (in-memory `Cursor`, HTTP-range-backed adapter, …) so callers
//!   can extract tensor metadata without materialising the data segment
//!   (requires `npz` feature)
//! - `parse_gguf()` / `inspect_gguf_from_reader()` — `GGUF` parsing /
//!   inspection. The path-based variant memory-maps the file and returns a
//!   `ParsedGguf` with zero-copy tensor views; the reader-generic variant
//!   accepts any `Read + Seek` substrate and returns just the
//!   `GgufInspectInfo` summary, so a multi-GB quantised `GGUF`'s metadata
//!   can be inspected in a few range fetches over the front-loaded header
//!   without downloading the data section (requires `gguf` feature)
//! - `parse_pth()` / `inspect_pth_from_reader()` — `PyTorch` `.pth` parsing
//!   / inspection. The path-based variant memory-maps the file and returns
//!   a `ParsedPth` with zero-copy `tensors()`; the reader-generic variant
//!   accepts any `Read + Seek` substrate and returns just the
//!   `PthInspectInfo` summary, so a torchvision-class `.pth` is inspectable
//!   in a single `<100 KiB` range fetch over the ZIP central directory and
//!   `data.pkl` entry — no tensor-data files inside the archive are read
//!   (requires `pth` feature)
//! - `pth_to_safetensors()` / `pth_to_safetensors_bytes()` — lossless
//!   `.pth` → `.safetensors` conversion (requires `pth` feature)
//! - `npz_to_safetensors()` / `npz_to_safetensors_bytes()` — lossless
//!   `.npz` → `.safetensors` conversion (requires `npz` feature; Phase 6)
//! - `write_gguf()` / `write_gguf_to_writer()` — emit a `.gguf` file
//!   from scalar-dtype tensors plus a metadata `KV` table; the
//!   format-symmetric inverse of `parse_gguf` (requires `gguf` feature;
//!   Phase 6)
//! - `write_bnb_nf4_safetensors()` / `write_bnb_nf4_safetensors_bytes()`
//!   — end-to-end `BF16 → BnB-NF4 safetensors` path with the four-tensor
//!   companion layout (`weight`, `weight.absmax`, `weight.quant_map`,
//!   `weight.quant_state.bitsandbytes__nf4`) (requires `bnb` feature;
//!   Phase 6)
//!
//! The [`remember`] module contains one submodule per quantization family
//! ([`remember::fp8`] always-on; `remember::gptq`, `remember::awq`,
//! `remember::bnb` feature-gated independently under `gptq` / `awq` /
//! `bnb`).
//!
//! The [`lethe`] module mirrors that layout on the encode side. Phase 5
//! ships `lethe::bnb` (feature-gated behind `bnb`) plus the
//! always-on `lethe::round_trip` validation harness. Encoding a fresh
//! `BF16` source into `BnB-NF4`:
//!
//! ```rust,no_run
//! # #[cfg(feature = "bnb")]
//! # fn run() -> anamnesis::Result<()> {
//! use anamnesis::{encode_bnb4_compute_absmax, NF4_CODEBOOK};
//!
//! // 64 BF16 elements arranged as one 64-element block.
//! let bf16_bytes: Vec<u8> = vec![0u8; 64 * 2];
//! let codebook_bytes: Vec<u8> =
//!     NF4_CODEBOOK.iter().flat_map(|v| v.to_le_bytes()).collect();
//! let (weight, absmax) =
//!     encode_bnb4_compute_absmax(&bf16_bytes, &codebook_bytes, 64, 64)?;
//! assert_eq!(weight.len(), 32);   // 64 elements / 2 nibbles per byte
//! assert_eq!(absmax.len(), 4);    // 1 block × F32 LE
//! # Ok(()) }
//! ```
//!
//! Writing a `GGUF` file (Phase 6 — scalar dtypes only):
//!
//! ```rust,no_run
//! # #[cfg(feature = "gguf")]
//! # fn run() -> anamnesis::Result<()> {
//! use std::collections::HashMap;
//! use anamnesis::{write_gguf, GgufType, GgufWriteTensor};
//!
//! // Two BF16 tensors. `shape` is most-significant-first, matching
//! // `parse_gguf` on the read side.
//! let w_data: Vec<u8> = vec![0u8; 8 * 2];
//! let b_data: Vec<u8> = vec![0u8; 4 * 2];
//! let tensors = [
//!     GgufWriteTensor { name: "w", shape: &[4, 2], dtype: GgufType::BF16, data: &w_data },
//!     GgufWriteTensor { name: "b", shape: &[4],    dtype: GgufType::BF16, data: &b_data },
//! ];
//! // Metadata is optional; `general.alignment` is injected if absent.
//! write_gguf("out.gguf", &tensors, &HashMap::new())?;
//! # Ok(()) }
//! ```

// `deny` (not `forbid`) allows feature-gated modules to opt in to unsafe
// where required by external APIs (e.g., memmap2 in the `pth` module).
// See CONVENTIONS.md "// SAFETY:" rules for the policy.
#![deny(unsafe_code)]
#![deny(warnings)]
// Allow unknown lint names so that `#[allow(clippy::newer_lint)]` in test
// modules does not become an error when built with MSRV clippy (which may
// not recognise lints added in later releases). Without this, every new
// clippy lint suppression is a potential MSRV CI break.
#![allow(unknown_lints)]

/// Command-line interface implementation shared by the `anamnesis` and
/// `amn` binaries. Feature-gated behind `cli`; pulls in [`clap`] only
/// when enabled.
// Raw-byte backing store (memory map or owned copy) shared by every
// parsed-model type. Crate-internal: the `Backing` enum distinguishes the
// trusted mmap fast path from the owned-copy untrusted-input path.
mod backing;
#[cfg(feature = "cli")]
pub mod cli;
pub mod convert;
pub mod error;
pub mod inspect;
pub mod lethe;
pub mod limits;
pub mod model;
pub mod parse;
pub mod remember;

pub use convert::{convert, ConvertOptions, ConvertStats, ConvertTarget};
pub use error::{AnamnesisError, Result};
pub use inspect::{format_bytes, InspectInfo};
#[cfg(feature = "bnb")]
pub use lethe::{
    classify_inputs, encode_bnb4, encode_bnb4_compute_absmax, encode_bnb4_double_quant,
    encode_bnb_int8, encode_bnb_int8_compute_scb, is_eligible_for_nf4, write_bnb_nf4_safetensors,
    write_bnb_nf4_safetensors_bytes, BnbNf4WriteStats, BnbWriteInput, FP4_CODEBOOK, NF4_BLOCK_SIZE,
    NF4_CODEBOOK,
};
pub use limits::ParseLimits;
pub use model::{
    parse, parse_bytes, parse_bytes_with_limits, parse_from_reader, parse_from_reader_with_limits,
    parse_with_limits, ParsedModel, RememberOptions, TargetDtype,
};
#[cfg(feature = "ollama")]
pub use parse::resolve_ollama_model;
#[cfg(feature = "gguf")]
pub use parse::{
    inspect_gguf_from_reader, parse_gguf, parse_gguf_bytes, parse_gguf_bytes_with_limits,
    parse_gguf_from_reader, parse_gguf_from_reader_with_limits,
    parse_gguf_front_matter_from_reader, parse_gguf_front_matter_from_reader_with_limits,
    parse_gguf_with_limits, write_gguf, write_gguf_to_writer, GgufFrontMatter, GgufInspectInfo,
    GgufMetadataArray, GgufMetadataValue, GgufTensor, GgufTensorInfo, GgufType, GgufWriteTensor,
    ParsedGguf,
};
#[cfg(feature = "npz")]
pub use parse::{
    inspect_npz, inspect_npz_from_reader, parse_npz, parse_npz_with_limits, NpzDtype,
    NpzInspectInfo, NpzTensor, NpzTensorInfo,
};
#[cfg(feature = "pth")]
pub use parse::{
    inspect_pth_from_reader, parse_pth, parse_pth_bytes, parse_pth_bytes_with_limits,
    parse_pth_from_reader, parse_pth_from_reader_with_limits, parse_pth_with_limits, ParsedPth,
    PthDtype, PthInspectInfo, PthTensor, PthTensorInfo,
};
pub use parse::{
    parse_safetensors_header, parse_safetensors_header_from_reader,
    parse_safetensors_header_from_reader_with_limits, parse_safetensors_header_with_limits,
    AwqCompanions, AwqConfig, Bnb4Companions, BnbConfig, Dtype, GptqCompanions, GptqConfig,
    QuantScheme, SafetensorsHeader, TensorEntry, TensorRole,
};
#[cfg(feature = "awq")]
pub use remember::dequantize_awq_to_bf16;
#[cfg(feature = "gptq")]
pub use remember::dequantize_gptq_to_bf16;
#[cfg(feature = "bnb")]
pub use remember::{dequantize_bnb4_to_bf16, dequantize_bnb_int8_to_bf16};
pub use remember::{
    dequantize_fp8_to_bf16, dequantize_per_channel_fp8_to_bf16, dequantize_per_tensor_fp8_to_bf16,
};
#[cfg(feature = "gguf")]
pub use remember::{dequantize_gguf_blocks_to_bf16, dequantize_gguf_to_bf16};
#[cfg(feature = "npz")]
pub use remember::{npz_to_safetensors, npz_to_safetensors_bytes};
#[cfg(feature = "pth")]
pub use remember::{pth_to_safetensors, pth_to_safetensors_bytes};
