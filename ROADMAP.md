# Roadmap: anamnesis — Tensor Format Transformation for Rust

> *Parse any format, recover any precision.*

**Date:** March 20, 2026 (updated June 21, 2026)
**Status:** Phases 1–6.13 complete (v0.6.8 released). FP8/GPTQ/AWQ/BnB dequantization + NPZ parsing + PyTorch `.pth` parsing + GGUF parsing & dequantization — all 22 of 22 production block-quant kernels — plus reader-generic inspection across every format (`inspect_npz_from_reader<R: Read + Seek>`, `parse_safetensors_header_from_reader<R: Read>` cross-validated against the upstream `safetensors` Python library, and the GGUF / `.pth` counterparts from Phases 4.9–4.10), an mmap-based always-on `parse()` (~3236× faster on a 11.6 GiB safetensors shard), and the completed v0.6.x security-hardening line: caller-configurable `ParseLimits` untrusted-input governance (Phase 6.8), a working-set-bounded pickle VM (Phase 6.11), a vendored read-only ZIP container reader (Phase 6.12), and Python-readiness hardening — copy-based mmap-free parse paths, a finer `AnamnesisError` taxonomy (`LimitExceeded` / `DisallowedGlobal`), a panic/abort-freedom invariant + `unwind` extension profile, and the locked `NumPy`/`BF16` data-ownership contract (Phase 6.13). **Next:** Phase 6.14 (convert-matrix completion, v0.6.9) → Phase 7 (CPU SIMD pass — runtime AVX2/NEON dispatch, v0.7.0) → Phase 8 (Python bindings / PyO3, v0.8.0) → Phase 8.5 (Lethe encode completion, v0.8.5).
**Context:** The Rust ML ecosystem (candle, burn, tch) cannot load quantized models (FP8, GPTQ, AWQ) or NumPy weight archives (NPZ/NPY for SAEs). The only workaround is a Python script. anamnesis fills this gap: a framework-agnostic, pure-Rust crate that parses tensor formats and recovers precision when needed. Used by hf-fetch-model (download + transform pipeline) and candle-mi (MI framework).

---

## Table of Contents

- [1. Landscape](#1-landscape)
  - [1.1 The Problem](#11-the-problem)
  - [1.2 Existing Solutions](#12-existing-solutions)
- [2. Architecture](#2-architecture)
  - [2.1 Parsing as Foundation](#21-parsing-as-foundation)
  - [2.2 Module Structure](#22-module-structure)
  - [2.3 Ecosystem Fit](#23-ecosystem-fit)
- [3. Phased Development Plan](#3-phased-development-plan)
  - [3.0 Git Workflow](#30-git-workflow)
  - [Phase 1: FP8 Dequantization](#phase-1-fp8-dequantization)
  - [Phase 2: Additional Quantization Schemes](#phase-2-additional-quantization-schemes)
  - [Phase 3: NPZ/NPY Parsing](#phase-3-npznpy-parsing)
  - [Phase 3.5: PyTorch `.pth` Parsing](#phase-35-pytorch-pth-parsing)
  - [Phase 4: GGUF Parsing & Dequantization](#phase-4-gguf-parsing--dequantization)
  - [Phase 4 patch: API polish + dogfooding (v0.4.1)](#phase-4-patch-api-polish--dogfooding-v041)
  - [Phase 4.5: GGUF Completeness](#phase-45-gguf-completeness)
  - [Phase 4.7: Remote-only NPZ inspection (Reader-generic API)](#phase-47-remote-only-npz-inspection-reader-generic-api)
  - [Phase 4.8: Reader-generic safetensors header parsing](#phase-48-reader-generic-safetensors-header-parsing)
  - [Phase 4.9: Reader-generic GGUF inspection](#phase-49-reader-generic-gguf-inspection)
  - [Phase 4.10: Reader-generic PTH inspection](#phase-410-reader-generic-pth-inspection)
  - [Phase 5: Quantization (Lethe)](#phase-5-quantization-lethe)
  - [Phase 6: Format Conversion Matrix](#phase-6-format-conversion-matrix)
  - [Phase 6.5: Benchmarking & Performance Validation](#phase-65-benchmarking--performance-validation)
  - [Phase 6.6: Security Hardening — Unguarded Allocations](#phase-66-security-hardening--unguarded-allocations)
  - [Phase 6.7: Security Audit Hardening + Fuzzing](#phase-67-security-audit-hardening--fuzzing)
  - [Phase 6.8: Untrusted-Input Resource Governance](#phase-68-untrusted-input-resource-governance)
  - [Phase 6.9: Quantized Dequant Bit-Exactness (Dogfooding Fixes)](#phase-69-quantized-dequant-bit-exactness-dogfooding-fixes)
  - [Phase 6.10: Standard-Orientation Output for AWQ/GPTQ (Dogfooding Fix #2)](#phase-610-standard-orientation-output-for-awqgptq-dogfooding-fix-2)
  - [Phase 6.11: Pickle-VM Working-Set Governance](#phase-611-pickle-vm-working-set-governance)
  - [Phase 6.12: Vendored ZIP Container Reader](#phase-612-vendored-zip-container-reader)
  - [Phase 6.13: Python-Readiness Hardening](#phase-613-python-readiness-hardening)
  - [Phase 6.14: Convert-Matrix Completion (BF16 Hub)](#phase-614-convert-matrix-completion-bf16-hub)
  - [Phase 7: CPU SIMD Pass](#phase-7-cpu-simd-pass)
  - [Phase 8: Python Bindings (PyO3)](#phase-8-python-bindings-pyo3)
  - [Phase 8.5: Lethe Encode Completion](#phase-85-lethe-encode-completion)
  - [Phase 9: Emerging Quantization Formats](#phase-9-emerging-quantization-formats)
  - [Phase 10: Streaming Output](#phase-10-streaming-output)
  - [Future Directions](#future-directions)
- [4. Key Design Decisions](#4-key-design-decisions)
- [5. Relationship to Other Projects](#5-relationship-to-other-projects)

---

## 1. Landscape

### 1.1 The Problem

Quantized models are the norm. Mistral ships Ministral 3B as FP8 safetensors. Community quantizers (TheBloke, NeuralMagic, etc.) distribute Llama and other models in GPTQ and AWQ. SAE weights (Gemma Scope) ship as NPZ archives. None of these load in any Rust ML framework:

```rust
let vb = VarBuilder::from_mmaped_safetensors(&[path], DType::BF16, &device)?;
// → Error: unsupported safetensor dtype F8_E4M3
```

This blocks candle, candle-mi, burn, and tch. The entire Rust ML ecosystem stops at the file format boundary.

### 1.2 Existing Solutions

*Surveyed March 2026. No framework-agnostic tensor format transformation library exists in any language — not in Rust, not in Python, not in C++. Dequantization is universally implemented ad-hoc inside inference engines.*

#### Rust ecosystem

| Solution | What it does | Why it's not enough |
|---|---|---|
| [`float8`](https://crates.io/crates/float8) v0.7.0 | FP8 primitive types (E4M3, E5M2) with `to_f32()` / `from_f32()` conversions | Type-level only. No tensors, no safetensors, no scale factors, no block-wise dequant. A useful **building block** — anamnesis should depend on it for FP8 ↔ f32 element conversion. |
| [candle](https://github.com/huggingface/candle) v0.9.2+ | Added `F8E4M3` DType. [DeepSeek V3 PR #2745](https://github.com/huggingface/candle/pull/2745) implements block-wise FP8 dequant with `scale_inv` | Dequant logic is **model-specific** (DeepSeek V3 only), embedded in `candle-transformers`. Not a reusable library. No E5M2. Still fails on generic FP8 safetensors: `unsupported safetensor dtype F8_E4M3`. |
| [mistralrs-quant](https://lib.rs/crates/mistralrs-quant) | GPTQ, AWQ, HQQ, BnB, FP8 via `QuantMethod` trait with `dequantize_w()` | **CUDA-only** for GPTQ/AWQ. Tightly coupled to mistral.rs inference (742K SLoC, depends on candle-core, tokio, rayon). Not reusable as a standalone library. |
| [pmetal-gguf](https://docs.rs/pmetal-gguf) | Standalone GGUF dequantization with SIMD (`dequant::dequantize()`) | **GGUF-specific.** Does not handle safetensors FP8, GPTQ, AWQ, or NPZ. |
| [`gguf-rs-lib`](https://crates.io/crates/gguf-rs-lib) v0.2.5 | Type-safe GGUF reader/writer. Pure safe Rust. | Parsing only — no dequantization. Potential Phase 4 dependency (TBD). |
| [`llama-gguf`](https://crates.io/crates/llama-gguf) v0.14.0 | Pure Rust llama.cpp reimplementation with GPU dequant kernels (CUDA/Metal/DX12/Vulkan). | Full inference engine, not a standalone library. K-quant dequant math is a useful reference. |
| [`npyz`](https://crates.io/crates/npyz) v0.8.4 | Mature NPY/NPZ parser (4.6M downloads). f16 support via `half` feature. Full read/write. | **No bf16.** Initial Phase 3 approach wrapped `npyz`, but benchmarking revealed 23× overhead vs raw I/O. Replaced with anamnesis's own fast parser (17.7× faster). |
| [`ndarray-npy`](https://crates.io/crates/ndarray-npy) v0.10.0 | Most popular NPY/NPZ crate (7.2M downloads). Actively maintained. | **No f16/bf16.** Tightly coupled to ndarray. Less suitable than `npyz` for framework-agnostic use. |
| [safetensors](https://crates.io/crates/safetensors) | Reads/writes safetensors files | No quantization awareness. Pure format I/O. |
| [burn](https://github.com/tracel-ai/burn) v0.14.0+ | INT8 quantization (beta) | No FP8, no GPTQ, no AWQ. Own internal format only. |
| tch-rs | Defines FP8 variants (`Float8e4m3fn`, `Float8e5m2`) | Wraps PyTorch C++ API. Requires full libtorch installation. No Rust-native dequant. |
| candle-mi's NPZ parser | Loads Gemma Scope SAE weights from `.npz` | Tightly coupled to candle. Not reusable outside candle-mi. |

#### Python ecosystem

| Solution | What it does | Why it's not enough |
|---|---|---|
| [compressed-tensors](https://github.com/vllm-project/compressed-tensors) (vLLM) | Safetensors extension for sparse/quantized storage. Closest conceptual competitor to anamnesis. | Python-only, vLLM-coupled, decompression-to-bf16 still immature ([users requesting basic examples](https://huggingface.co/moonshotai/Kimi-K2-Thinking/discussions/2) as of late 2025). |
| [optimum-quanto](https://github.com/huggingface/optimum-quanto) (HuggingFace) | In-memory quantize/dequantize on PyTorch tensors | Being merged into `optimum`. In-memory only — no file-to-file conversion. PyTorch-coupled. |
| [TorchAO](https://github.com/pytorch/ao) | INT4, INT8, FP8 quantization within PyTorch | PyTorch tensors only, not a file format tool. |
| transformers `dequantize=True` | Loads fine-grained FP8 with dequantization during model load | Embedded in model loading pipeline. Not a standalone operation. |
| AutoGPTQ → [GPTQModel](https://pypi.org/project/GPTQModel/) | GPTQ quantization/dequantization | Dequant happens in fused CUDA kernels during inference. No standalone API. AutoGPTQ archived April 2025. |
| AutoAWQ, bitsandbytes | AWQ and NF4/INT8 quantization | Same pattern: dequant is implicit during inference, not exposed as a file conversion. |
| [llm-compressor](https://github.com/vllm-project/llm-compressor) (vLLM) | "Model-free PTQ" on safetensors (FP8) | **Quantization direction only.** No dequantization. |

#### C/C++ ecosystem

| Solution | What it does | Why it's not enough |
|---|---|---|
| llama.cpp | Converts safetensors → GGUF (Python scripts); dequantizes GGUF during inference (C++) | GGUF-specific. Not a library. Conversion scripts are Python. |
| NVIDIA TensorRT | `IQuantizeLayer` / `IDequantizeLayer` in graph IR | Operates within TensorRT's graph representation, not on raw files. NVIDIA-coupled. |
| NVIDIA TransformerEngine | FP8 compute library (matmul kernels) | Compute library, not a format conversion tool. |

#### The gap

No tool in any language provides **file → transform → file** as a first-class, framework-agnostic operation for quantized tensor formats. Dequantization is always buried inside inference engines or model loading pipelines. anamnesis is the first library designed for this purpose.

---

## 2. Architecture

### 2.1 Parsing as Foundation

Every operation in anamnesis begins with parsing. You cannot remember what you have not first parsed. You cannot quantize what you have not first parsed. You cannot inspect what you have not first parsed.

Parsing is the act of making contact with the weights: decoding their structure, validating their format, understanding what is present and what Lethe took. Sometimes parsing alone is enough (NPZ — nothing was forgotten). Other times, parsing reveals that precision recovery is needed (FP8 — Lethe took something), and remembering follows.

### 2.2 Module Structure

```
anamnesis (library crate)
│
├── parse/              ← decode + validate any tensor format
│   ├── safetensors        safetensors (including quantized metadata)
│   ├── npz                NPZ/NPY archives (feature-gated)
│   ├── pth                PyTorch .pth state_dict (feature-gated, Phase 3.5)
│   └── gguf               GGUF (feature-gated, Phase 4)
│
├── remember/           ← built on parse: precision recovery (dequantize)
│   ├── fp8                fine-grained, per-channel, per-tensor FP8 (E4M3, E5M2)
│   ├── gptq               GPTQ dequantization
│   ├── awq                AWQ dequantization
│   ├── bnb                BitsAndBytes (NF4, INT8) dequantization
│   ├── pth                PyTorch .pth → safetensors conversion (Phase 3.5)
│   └── gguf               GGUF K-quant dequantization (Phase 4)
│
├── lethe/              ← built on parse: precision reduction (quantize)
│   ├── bnb                BnB encode (NF4, FP4, INT8) (Phase 5)
│   ├── round_trip         Bit-exact decode↔encode validation harness (Phase 5)
│   ├── fp8                FP8 encode (E4M3, E5M2; per-tensor / per-channel / fine-grained) (Phase 8.5)
│   ├── gguf_legacy        GGUF legacy block encode (Q4_0–Q8_1) (Phase 8.5)
│   ├── gguf_kquants       GGUF K-quant encode (Q2_K–Q8_K) (Phase 8.5)
│   ├── gguf_iq            GGUF IQ-quant encode (IQ1/IQ2/IQ3/IQ4) (Phase 8.5)
│   ├── gguf_tq            GGUF TQ encode (TQ1_0, TQ2_0) (Phase 8.5)
│   └── mxfp4              MXFP4 encode (Phase 8.5)
│
├── inspect             ← built on parse: report without transforming
│
└── bin/
    └── main.rs         ← CLI binary (feature-gated behind "cli")
                           installed as both `anamnesis` and `amn`
                           subcommands: parse, inspect/info, remember/dequantize, forget/quantize
```

### 2.3 Ecosystem Fit

```
safetensors / npz / pth / gguf  ← file formats
    ↓
anamnesis (library)     ← parse, remember, lethe
anamnesis / amn (CLI)   ← amn parse, amn inspect, amn remember, amn forget
    ↓
hf-fetch-model          ← download + transform pipeline
                           (hf-fm --dequantize bf16, download_and_parse_npz)
candle / candle-mi      ← load + run + interpret
burn / tch / ...        ← any Rust ML consumer
```

hf-fetch-model depends on anamnesis for two things:
1. `--dequantize bf16` calls `anamnesis::parse()` + `.remember()` after download.
2. `download_and_parse_npz()` calls `download()` + `anamnesis::parse_npz()`.

---

## 3. Phased Development Plan

### 3.0 Git Workflow

Single `main` branch. Each task ends with a commit. Phases end with a push and a version tag. Same workflow as candle-mi and hf-fetch-model.

Commit style: imperative mood, lowercase, no trailing period. Examples:
- `add safetensors header parsing`
- `implement fine-grained FP8 dequantization`
- `validate FP8 dequantization against EXAONE-4.0-1.2B`

### Phase 1: FP8 Dequantization

**Goal:** The flagship use case — parse an FP8 safetensors file, dequantize to BF16, write a standard safetensors file loadable by any Rust ML framework. Proves the parse → remember architecture end-to-end.

- [x] Scaffold crate:
  - `git init` + GitHub repo creation + remote setup
  - `cargo init --lib`
  - `Cargo.toml` with metadata (name, version `0.1.0`, description, license MIT OR Apache-2.0, keywords, categories), feature gates (`cli = ["dep:clap"]`, `npz = ["dep:npyz"]`), two `[[bin]]` targets (`anamnesis` and `amn`, both pointing to `src/bin/main.rs`, `required-features = ["cli"]`) following the hf-fetch-model pattern
  - `src/lib.rs` with `#![forbid(unsafe_code)]`, `#![deny(warnings)]`
  - `CLAUDE.md` — "Before writing or modifying any Rust code, read and follow CONVENTIONS.md." (same as hf-fetch-model)
  - `LICENSE-MIT` + `LICENSE-APACHE` — dual license files
  - `.gitignore` — standard Rust (`/target`, etc.)
  - `.github/workflows/ci.yml` + `.github/workflows/publish.yml` — reused from hf-fetch-model's pattern with `actions/checkout@v5`: format check, clippy (`--all-targets` + `--all-features`), tests, doc check on publish, tag-triggered `cargo publish`
  — **commit**
- [x] Safetensors parsing foundation (`src/parse/safetensors.rs`) — read header JSON, extract tensor names, shapes, dtypes, byte offsets. Identify quantized tensors (FP8) vs passthrough (BF16/F32 norms, embeddings). Detect fine-grained FP8 metadata (scale factor tensors, block structure). This is the "make contact" layer — **commit**
- [x] Inspect (`src/inspect.rs`) — built on parse. Report format, tensor count, quantized vs passthrough breakdown, size estimate (current and dequantized). Matches the flagship CLI output in `amn-flagship-v2.md` — **commit**
- [x] Fine-grained FP8 dequantization (`src/remember/fp8.rs`) — E4M3 with 128×128 block scale factors. Scalar implementation written for auto-vectorization: contiguous `&[u8]` → `&mut [u16]` slices, no branches in the hot path (bitwise select for NaN/subnormal), `chunks_exact(128)` matching block size, scale factor hoisted before inner loop. Verify with `cargo-show-asm` that the compiler emits SIMD instructions; annotate with `// VECTORIZED:`. Built on the parsed representation from `parse/safetensors` — **commit**
- [x] Per-tensor FP8 dequantization — single scale factor per tensor (simpler case). Same module, same SIMD-friendly loop structure, different scale broadcast — **commit**
- [x] Parse-first public API (`src/lib.rs`) — `parse(path)` returns a `ParsedModel` struct holding header metadata + byte data. `ParsedModel::inspect()` returns format info. `ParsedModel::remember(output_path, TargetDtype)` dequantizes and writes a standard safetensors file. No file is re-read after the initial parse. The Rust API in `amn-flagship-v2.md` should work — **commit**
- [x] CLI binary (`src/bin/main.rs`) — thin `clap` wrapper over the library API. Subcommands: `parse`, `inspect` (alias `info`), `remember` (alias `dequantize`). Each subcommand calls `anamnesis::parse()` then the appropriate method. Progress output via `indicatif` (optional, behind `indicatif` feature). Same binary serves both `anamnesis` and `amn` names — **commit**
- [x] Download FP8 test models via `hf-fetch-model` (v0.8.1, with `list-files`). 7 models from 5 quantization tools, covering 3 FP8 schemes discovered during validation:

  | Model | Size | FP8 scheme | Scale dtype | Quantizer |
  |---|---|---|---|---|
  | `LGAI-EXAONE/EXAONE-4.0-1.2B-FP8` | 1.39 GB | **Fine-grained** | BF16 | LG AI |
  | `Qwen/Qwen3-1.7B-FP8` | 2.47 GB | **Fine-grained** | BF16 | Qwen |
  | `Qwen/Qwen3-4B-Instruct-2507-FP8` | 4.83 GB | **Fine-grained** | **F16** | Qwen |
  | `mistralai/Ministral-3-3B-Instruct-2512` | 4.35 GB | **Per-tensor** (scalar) | BF16 | Mistral |
  | `RedHatAI/Llama-3.2-1B-Instruct-FP8` | 1.88 GB | **Per-tensor** | BF16 | RedHat |
  | `RedHatAI/Llama-3.2-1B-Instruct-FP8-dynamic` | 1.89 GB | **Per-channel** `[N,1]` | BF16 | RedHat |
  | `nvidia/Llama-3.1-8B-Instruct-FP8` (shard 1) | 4.65 GB | **Per-tensor** (scalar) | F32 | NVIDIA |

  Discoveries during validation:
  - **Per-channel FP8** — a third scheme (one scale per row, shape `[N,1]`), used by RedHat/vLLM dynamic quantization. New `dequantize_per_channel_fp8_to_bf16()` and `PerChannelFp8` scheme variant.
  - **F16 scales** exist in the wild (Qwen3-4B-Instruct-2507-FP8, Qwen/Alibaba), undocumented in the ecosystem. All three scale dtypes (F32, BF16, F16) now supported.
  - **Scheme detection by scale shape**, not name suffix — both fine-grained and per-tensor use `_scale_inv`, distinguished by scale tensor dimensionality.
  — **commits** (multiple: BF16 scale fix, scheme detection fix, F16 + per-channel support)

- [x] Validation against FP8 test models — cross-validated against PyTorch (`torch.float8_e4m3fn` → `torch.bfloat16`) on 256×256 slices from all 7 models. **Bit-exact match** (0 ULP difference on 65,536 elements per fixture). Fixtures generated by `tests/fixtures/fp8_reference/generate.py`, Rust tests in `tests/cross_validation.rs`. Auto-vectorization verified with `cargo-show-asm`: SSE2 (default), AVX2 (`target-cpu=native`). **2.7–9.7× faster than PyTorch** (AVX2). — **commit** — **PUSH**

**Deliverable:** `anamnesis` v0.1.0 — FP8 dequantization works. — **PUSH + tag `v0.1.0`**

**Dependencies:** `safetensors`, `half`, `float8` (FP8 ↔ f32 element conversion). CLI: `clap` 4 with `derive` (behind `cli` feature), `indicatif` (behind `indicatif` feature, optional progress bars).

**Note on cached test assets:** The local HuggingFace cache (`~/.cache/huggingface/hub/`) contains 21 models, none with FP8/GPTQ/AWQ quantization. The one asset relevant to anamnesis is **google/gemma-scope-2b-pt-res** which contains `params.npz` (302 MB, 5 F32 arrays: `W_dec`, `W_enc`, `b_dec`, `b_enc`, `threshold`) — this will be used for Phase 3 NPZ validation.

**Note on candle-mi auto-config:** Validating EXAONE-4.0 and Qwen3 after dequantization requires extending candle-mi's auto-config with `exaone4` and `qwen3` `model_type` support. These are both text-only decoder transformers close to existing supported families:
- `exaone4` — LLaMA-like with alternating sliding window / full attention layers ("LLLG" pattern). Similar to Gemma 2's alternating window scheme.
- `qwen3` — extends `qwen2` with QK LayerNorm (additional `q_norm` / `k_norm` tensors per layer) and thinking mode support. The core architecture is otherwise identical to `qwen2`.

### Phase 2: Additional Quantization Schemes

**Goal:** Extend `remember/` to cover the major quantization formats. Each scheme adds a new submodule under `remember/` and extends the parsing layer to detect its metadata.

- [x] GPTQ dequantization (`src/remember/gptq.rs`) — INT4/INT8 with group-wise scale + zero-point. Parse GPTQ metadata from safetensors, reconstruct full-precision weights. Bit-exact against PyTorch on 4 real models (2 quantizers × 2 bit widths), 6.5–12.2× faster than CPU PyTorch (AVX2). Loop fission for full AVX2 vectorization — **commit**
- [x] AWQ dequantization (`src/remember/awq.rs`) — activation-aware INT4 with per-group scales, no +1 zero-point offset. Packs along out_features (columns), unlike GPTQ (rows). Bit-exact against PyTorch on 2 real models (AutoAWQ GEMM), 4.7–5.7× faster than CPU PyTorch (AVX2). Loop fission applied from the start. Note: 8-bit AWQ path is unit-tested but no real 8-bit AWQ models exist in the standard AutoAWQ `.qweight` format — all "8-bit AWQ" on HuggingFace use `compressed-tensors` (vLLM) or are mislabeled 4-bit — **commit**
- [x] BitsAndBytes dequantization (`src/remember/bnb.rs`) — NF4, FP4, double-quant NF4/FP4, and INT8 (LLM.int8()). 4-bit uses 16-entry lookup table + per-block absmax; INT8 uses per-row absmax / 127.0. Bit-exact against PyTorch on 4 real models, 18–54× faster for NF4/FP4 (AVX2), 1.2× for INT8 (near bandwidth limit). Loop fission for NF4/FP4; single-pass AVX2 for INT8 (`vpmovsxbd` → `vcvtdq2ps` → `vmulps`). Format discovery via `hf-fm inspect` (remote header probing) — **commit**
- [x] Feature gates — each scheme behind its own feature flag (`gptq`, `awq`, `bnb`). Default features: `fp8` only — **commit**
- [x] Validation against real models for each scheme — **commits as needed** — **PUSH**

**Deliverable:** `anamnesis` v0.2.0 — all major quantization schemes supported. — **PUSH + tag `v0.2.0`**

**New dependencies:** None expected (pure bit manipulation), but feature-gated if any arise.

### Pre-Phase 3: BnB4 output shape recovery

**Goal:** BnB4 weights are stored flattened (`[N, 1]`). The current dequantization emits a 1D `[total_elements]` output shape (see `model.rs`). The original 2D shape (`[out_features, in_features]`) must be recovered from `config.json` for downstream consumers that expect shaped weight tensors.

- [x] Recover original weight shape for `BnB4` tensors from `quant_state.bitsandbytes__nf4`/`__fp4` companion tensor (`JSON` blob with `"shape"` field). No `config.json` needed — the shape is stored inside the safetensors file itself — **commit**

### Phase 3: NPZ/NPY Parsing

**Goal:** Extend the `parse/` layer to NumPy archives. This is the case where parsing alone suffices — nothing was forgotten, the weights just need extracting from a foreign container. Migrates candle-mi's tightly-coupled NPZ parser into anamnesis as a reusable, framework-agnostic module.

**Own the fast path.** Initial implementation wrapped `npyz` (v0.8.4), but benchmarking revealed 23× overhead vs raw I/O due to per-element deserialization. Replaced with a custom `NPY` header parser (~80 lines) and bulk `read_exact` — for LE data on LE machines (>99% of ML files), the raw bytes ARE the correct output. Uses `zip` v2 directly for `ZIP` extraction. Result: 84 ms for 302 MB (1.3× raw I/O overhead), **17.7× faster** than `npyz`, **4.9× faster** than candle-mi's custom parser.

- [x] NPZ integration module (`src/parse/npz.rs`) — custom `NPY` header parser with bulk `read_exact` data extraction. `parse_npz(path) -> HashMap<String, NpzTensor>`. Zero per-element deserialization for LE data on LE machines. BF16 interpretation layer (JAX `V2` void dtype). **3,586 MB/s** on 302 MB file (1.3× raw I/O overhead, 17.7× faster than `npyz`) — **commit**
- [x] `NpzTensor` type — framework-agnostic struct: `name: String`, `shape: Vec<usize>`, `dtype: NpzDtype`, `data: Vec<u8>`. The `NpzDtype` enum includes `BF16` which `NumPy` cannot represent natively — **commit**
- [x] Feature-gated behind `npz` feature (adds `zip` v2 with `deflate`) — **commit**
- [x] Validation against Gemma Scope SAE weights — already cached locally at `~/.cache/huggingface/hub/models--google--gemma-scope-2b-pt-res/` (`params.npz`, 302 MB, 5 F32 arrays: `W_dec` [16384×2304], `W_enc` [2304×16384], `b_dec` [2304], `b_enc` [16384], `threshold` [16384]). Parse, verify shapes and values match Python reference — **commit** — **PUSH**

**Deliverable:** `anamnesis` v0.3.0 — NPZ parsing works. candle-mi can migrate its NPZ dependency from internal to `anamnesis`. — **PUSH + tag `v0.3.0`**

**New dependencies:** `zip` v2 with `deflate` feature (direct dependency, no `npyz`). Feature-gated behind anamnesis's `npz` feature.

### Phase 3.5: PyTorch `.pth` Parsing

**Goal:** Add PyTorch `.pth` (state_dict) parsing and lossless conversion to safetensors. This enables loading pre-trained weights from repositories that only ship `.pth` files (e.g., AlgZoo's 400+ tiny models on GCS, plus thousands of older HuggingFace models). Unlike Phases 1–2, no dequantization is needed — `.pth` tensors are already full-precision (F32/F64/BF16/F16).

**Approach:** Implement a minimal pickle VM (~36 opcodes, not the full protocol) that interprets the `data.pkl` stream inside the ZIP archive, extracts tensor metadata (shape, dtype, storage reference), and reads raw bytes from `data/{index}` entries. Security boundary: explicit GLOBAL allowlist rejects non-`torch.*` callables. Lossless format conversion to safetensors via `safetensors::tensor::serialize_to_file`. Feature-gated behind `pth`.

**Detailed plan:** See `PLAN-PTH-v0.3.1.md`.

- [x] Extract `byteswap_inplace` from `npz.rs` to shared `src/parse/utils.rs` — **commit**
- [x] Pickle VM + tensor extraction (`src/parse/pth.rs`) — minimal stack machine, GLOBAL allowlist, storage cache for shared storage, stride handling — **commit**
- [x] Test fixtures — 3 AlgZoo `.pth` models (2–3.5 KB each) from GCS + Python-generated JSON reference manifests — **commit**
- [x] Cross-validation tests (`tests/cross_validation_pth.rs`) — byte-exact comparison against `PyTorch` on 3 AlgZoo models (RNN + Transformer, newer and older ZIP formats) — **commit**
- [x] Safetensors conversion (`src/remember/pth.rs`) — lossless format writer, byte-exact roundtrip verified — **commit**
- [x] `ParsedPth` + `PthInspectInfo` — inspect/to_safetensors methods, Display impl — **commit**
- [x] CLI integration (`src/bin/main.rs`) — format dispatch by extension + ZIP magic, `parse`/`inspect`/`remember` for `.pth` — **commit**
- [x] Module wiring — `mod.rs`, `lib.rs`, `Cargo.toml` feature gate, `error.rs` cfg update — done incrementally across prior commits

**Deliverable:** `anamnesis` v0.3.1 — PyTorch `.pth` state_dict parsing + safetensors conversion. — **PUSH + tag `v0.3.1`**

**New dependencies:** None (reuses `zip` v2 from `npz`). Feature-gated behind `pth`.

### Phase 4: GGUF Parsing & Dequantization

**Status:** Complete (`v0.4.0` published). Parser + all 12 block-quant dequantisation kernels + streaming API + cross-validation against `llama.cpp` reference (10/12 kernels bit-exact, 6.3–31.3× faster than Python reference).

**Goal:** Add support for GGUF, the dominant format for local inference (~166,000 models on HuggingFace as of March 2026). GGUF is to llama.cpp/Ollama/LM Studio what safetensors is to HuggingFace Transformers. No standalone, framework-agnostic Rust crate exists for GGUF dequantization — candle-core has GGUF support but it is tightly coupled to candle's tensor types and inference pipeline.

**Approach:** Parse GGUF metadata and tensor layout. Dequantize K-quant types (`Q2_K`–`Q8_K`) and legacy types (`Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q8_1`) to BF16 using the same loop-fission + auto-vectorization strategy as GPTQ/AWQ/BnB. Feature-gated behind `gguf`.

**Key projects studied:**
- **ggml-org/ggml** (`src/ggml-common.h`, `src/ggml-quants.c`) — canonical block struct layouts with `_Static_assert` byte counts and scalar `dequantize_row_*` reference implementations. **Used as the ground truth for every kernel in this phase** (ported verbatim with CONVENTIONS annotations applied).
- **candle-core `quantized`** (`k_quants.rs`) — secondary math reference for K-quants; not depended on.
- **gguf-rs-lib** — considered as a parsing dependency but rejected in favour of a lean in-house parser (matches the `.npy` and `.pth` pattern, keeps all error paths inside `AnamnesisError`).
- **llama-gguf (Lexmata)** — pure Rust reimplementation of llama.cpp with GPU dequant kernels; not used.

- [x] **GGUF file parser** (`src/parse/gguf.rs`, commit `2acaf1a`) — lean in-house parser for `GGUF` v2 and v3. Memory-mapped via `memmap2` (reused from `pth`), returns `Cow::Borrowed` tensor views with zero per-tensor heap allocation. Reads header, metadata KV pairs (all 13 value types including nested `ARRAY`), and the tensor info table; resolves absolute tensor-data offsets from the `general.alignment` metadata key (default 32 B). Adversarial-input DoS guards on tensor count (1 M), KV count (1 M), string length (16 MiB), array length (16 M), array nesting depth (4), tensor dimensions (8), and element product (1 T). Public types: `GgufType`, `GgufMetadataValue`, `GgufMetadataArray`, `GgufTensor`, `GgufTensorInfo`, `GgufInspectInfo`, `ParsedGguf`. **21 unit tests.** **commit**
- [x] **K-quant dequantization** (`src/remember/gguf.rs`, commit `cc4ecf8`) — scalar reference kernels for `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_K` (256-element super-blocks). Each block is processed with two-pass loop fission: pass 1 unpacks the packed-bit storage into an `[f32; 256]` stack scratch buffer (L1-resident), pass 2 walks the scratch and emits `BF16` bytes via the shared `f32_bits_to_bf16_bits` helper from `remember::fp8`. `get_scale_min_k4` (Q4_K/Q5_K's 6-bit packed-scale extractor) and `q3_k_unpack_scales` (Q3_K's `kmask1`/`kmask2` permute) are ported verbatim as private `#[inline]` helpers with their own unit tests — the two most error-prone bit-packing pieces are verifiable in isolation. **commit**
- [x] **Legacy quant types** (`src/remember/gguf.rs`, commit `cc4ecf8`) — scalar kernels for `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q8_1` (32-element blocks). `Q8_1` is a trivial variant of `Q8_0` (same `d * qs[j]` formula; the auxiliary `d × Σ qs` field is ignored for reconstruction). Landed together with the K-quants in a single commit — the loop-fission template is identical; only the per-type pass-1 unpack differs. **commit**
- [x] **Streaming dequant API** (`dequantize_gguf_blocks_to_bf16`, commit `b6610fe`) — `FnMut(&[u8]) -> Result<()>` sink closure receives one block's worth of `BF16` bytes at a time (64 B legacy, 512 B K-quant). Peak heap is O(one scratch + one block buffer) ≈ 1.5 KB, independent of tensor size — enabling dequantisation of 70 B-parameter models on modest-RAM machines by streaming directly to disk. The `Vec`-returning `dequantize_gguf_to_bf16` is now a thin wrapper around the streaming variant; both entry points share the same validation and the same scalar kernels via a generic `run_legacy_kernel` / `run_super_kernel` outer-loop helper.
- [x] **Algorithmic audit fixes + consistency pass** (commits `b6610fe`, `3156104`) — `Vec::with_capacity` + `extend_from_slice` instead of `vec![0u8; n]` (no zero-init memset); `chunks_exact` outer loop eliminates per-block bounds checks; infallible `read_f16_bytes([u8; 2]) -> f32` / `read_f32_bytes([u8; 4]) -> f32` replace the old `Result`-returning readers; `n_elements.checked_mul(2)` overflow guard catches 32-bit targets with > 2 GiB of `BF16` output; `read_scales12` helper deduplicates the 12-byte packed-scale load across Q3_K/Q4_K/Q5_K. **30 unit tests** after the refactor.
- [x] **Feature-gated behind `gguf` feature** — added in commit `2acaf1a` as part of the parser commit (`gguf = ["dep:memmap2"]` in `Cargo.toml`). Reuses `memmap2` (already a `pth` dep) and `half` (already mandatory). No new third-party crate in any Phase 4 commit.
- [x] **Cross-validation against `llama.cpp` reference dequantization** (commit `73f8e74`) — 10 of 12 production kernels bit-exact (0 ULP) against the `gguf` Python package's `dequantize` function (the official `ggml-org` reference mirroring `ggml-quants.c`'s `dequantize_row_*`). Legacy quants: `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`. K-quants: `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`. Fixtures from bartowski SmolLM2-135M-Instruct and TheBloke TinyLlama-1.1B-Chat (65 536-element slices). `Q8_1` and `Q8_K` are not shipped by any real model (internal `llama.cpp` activation quant types) and are already covered by unit tests. **6.3–31.3× faster** than the Python reference (AVX2). **10 integration tests.** — **PUSH + tag `v0.4.0`**

**Deferred follow-ups:** every out-of-scope item is rehomed to **Phase 4.5: GGUF Completeness** (`IQ*`/`TQ*`/`MXFP4` kernels — targeting `v0.4.2`) and **Phase 7: CPU SIMD Pass** (cross-format AVX2/NEON pass-2 SIMD). User-facing polish (GGUF CLI subcommands, `ParsedGguf::dequantize_tensor`) is pulled into v0.4.1 alongside the in-memory safetensors API from the candle-mi dogfooding report. Big-endian GGUF v3 support has no committed target — the parser detects byte-swapped magic and returns a clear `Unsupported` error, which is enough until a real big-endian model ships.

**Deliverable:** `anamnesis` v0.4.0 — GGUF parsing + dequantization works end-to-end with bit-exact `llama.cpp`-validated output. anamnesis becomes the only Rust crate that can parse *both* safetensors-based (GPTQ/AWQ/BnB/FP8) and GGUF-based quantized models and dequantize everything to BF16. — **PUSH + tag `v0.4.0`**

**New dependencies:** None. The `gguf` feature pulls in `memmap2` (already used by `pth`) and relies on `half` (already mandatory). No third-party GGUF parser in the dependency tree.

### Phase 4 patch: API polish + dogfooding (v0.4.1)

**Goal:** Quick backward-compatible release bundling two GGUF polish items pulled forward from Phase 4.5 and one in-memory safetensors API addition requested by candle-mi dogfooding. Unblocks candle-mi stoicheia module (agnostic `.pth` loading) and gives GGUF users CLI + convenience-method access without waiting for the full `IQ*`/`TQ*`/`MXFP4` kernel work.

**Dogfooding report:** [`docs/dogfooding-feedbacks/in-memory-safetensors-for-candle-mi.md`](docs/dogfooding-feedbacks/in-memory-safetensors-for-candle-mi.md) — candle-mi's stoicheia module needs an in-memory path from `.pth` bytes to safetensors bytes (`VarBuilder::from_buffered_safetensors`). The existing `pth_to_safetensors` writes to disk; the new `pth_to_safetensors_bytes` returns `Vec<u8>`.

- [x] **`pth_to_safetensors_bytes` + `ParsedPth::to_safetensors_bytes`** — in-memory `.pth` → safetensors conversion returning `Vec<u8>` instead of writing to disk. Enables downstream crates (candle-mi) to load `.pth` files without a temp file round-trip. Re-export from `lib.rs`. **5 new tests** (2 unit + 3 integration roundtrip-via-bytes). ~50 lines — **commit**
- [x] **`ParsedGguf::dequantize_tensor` convenience method** — thin wrapper that infers `n_elements` from `shape.iter().try_fold(checked_mul)` and slices the mmap at `data_offset..data_offset + byte_len` before delegating to `dequantize_gguf_to_bf16`. ~40 lines — **commit**
- [x] **GGUF CLI subcommands** — extend `src/bin/main.rs` format detection to recognise the `"GGUF"` magic and add `amn parse model.gguf`, `amn inspect model.gguf`, and `amn remember model.gguf --to bf16 -o out.safetensors`. Dequantizes quantized tensors to `BF16`, passes through non-quantized tensors with their original dtype, reverses GGUF MSB-first shape to safetensors row-major. ~130 lines — **commit** — **PUSH + tag `v0.4.1`**

**Deliverable:** `anamnesis` v0.4.1 — in-memory safetensors API for `.pth`, GGUF CLI subcommands, and `dequantize_tensor` convenience method. ~100 lines total, fully backward compatible. — **PUSH + tag `v0.4.1`**

### Phase 4.5: GGUF Completeness

**Goal:** Close the last remaining coverage gap in GGUF dequantisation so that anamnesis can handle every GGUF block type shipping on HuggingFace today. `v0.4.0` covers the 12 block types that back most mainstream models (`Q4_0`–`Q8_1`, `Q2_K`–`Q8_K`), but a growing fraction of 2025–2026 GGUF uploads use the newer `IQ*` family — `IQ4_XS` in particular has become a common "small but accurate" quant for many Llama 3, Qwen 2.5, and Mistral Nemo variants — plus a handful of models ship `TQ*` or `MXFP4`. Without those kernels, anamnesis rejects real files with `AnamnesisError::Unsupported`.

**Approach:** Port the scalar `dequantize_row_*` reference functions for the remaining block types from `ggml-quants.c` using the same loop-fission template as Phase 4. Most of the work is verbatim translation of the reference kernels plus verbatim copying of the lattice / codebook constant tables that the `IQ*` family uses (`iq2xxs_grid: [u64; 256]`, `iq3s_grid`, `iq1s_grid`, and siblings). Each variant has a distinct block struct layout documented in `ggml-common.h`; the test playbook is unchanged from Phase 4 — hand-constructed single-block fixtures per type, plus bit-exact cross-validation against `llama.cpp` reference output.

- [x] **`IQ4_NL` and `IQ4_XS` dequant kernels** (`src/remember/gguf.rs`, commits `486fa85` + `89abaa6`) — non-linear 4-bit quants (`IQ4_NL` 32-element at 18 B/block, `IQ4_XS` 256-element super-block at 136 B/block) sharing a single 16-entry `kvalues_iq4nl` codebook constant. Ported verbatim from `ggml-quants.c` using the Phase 4 two-pass loop-fission template. Bit-exact (0 ULP) against the `gguf` Python reference on 65 536-element slices from `bartowski/SmolLM2-135M-Instruct-GGUF` (6 new unit tests + 2 new cross-validation tests). First `IQ*` kernels in the crate; `IQ4_XS` is the most widely used member of the `IQ*` family on HuggingFace.
- [x] **`IQ2_XXS`, `IQ2_XS`, `IQ2_S` dequant kernels** (`src/remember/gguf.rs`, commits `cc83de8` + `6385c73`) — three 2-bit super-quants (66 / 74 / 82 B/block) sharing a lattice-codebook design: `IQ2XXS_GRID: [u64; 256]`, `IQ2XS_GRID: [u64; 512]`, `IQ2S_GRID: [u64; 1024]`, plus the `ksigns_iq2xs` / `kmask_iq2xs` sign tables (~14 KB total), all ported verbatim from `ggml-common.h` into a private `iq_grids` submodule at the bottom of the file. Bit-exact (0 ULP) against the `gguf` Python reference on 65 536-element slices from `bartowski/Mistral-7B-Instruct-v0.3-GGUF` (`IQ2_XXS` + `IQ2_XS`) and `bartowski/Qwen2.5-0.5B-Instruct-GGUF` IQ2_M mix (`IQ2_S`). Discovered along the way: `Mistral-7B-v0.3-IQ2_S.gguf` is misleadingly named — it ships `IQ2_XS` + `IQ3_S`, no `IQ2_S`. 7 new unit tests + 3 new cross-validation tests.
- [x] **`IQ3_XXS` and `IQ3_S` dequant kernels** (`src/remember/gguf.rs`, commits `acdcfcd` + `62b4aa3`) — two 3-bit super-quants (98 / 110 B/block) sharing two new codebook grids: `IQ3XXS_GRID: [u32; 256]` (1 KB) and `IQ3S_GRID: [u32; 512]` (2 KB). Both reuse the Phase 4.5 step 2 `write_signed_grid` helper — the combined-grid/signs packing format is shared across every sign-masked `IQ*` kernel. `IQ3_S` introduces the unusual odd-integer scale formula `d × (1 + 2·nibble)` and pairs low/high nibbles of `scales[outer]` across two consecutive sub-blocks. Bit-exact (0 ULP) against the `gguf` Python reference on 65 536-element slices from `bartowski/Mistral-7B-Instruct-v0.3-GGUF` — a single 2.64 GB download (`Mistral-7B-Instruct-v0.3-IQ3_XXS.gguf`) covers both fixtures because that file ships 96 `IQ3_XXS` tensors + 33 `IQ3_S` tensors. 6 new unit tests + 2 new cross-validation tests (3.32× / 4.37× faster than Python reference, release-mode best-of-5).
- [x] **`IQ1_S` and `IQ1_M` dequant kernels** (`src/remember/gguf.rs`, commits `16bd880` + `ecaca11` + `891e4b6`) — two 1-bit super-quants, smallest footprint in the IQ family. `IQ1_S` (50 B/block, top-level `d: f16`, 11-bit grid index, ±`IQ1S_DELTA = 0.125` additive bias); `IQ1_M` (56 B/block, **no top-level `d`** — super-block scale reconstructed from a scattered 16-bit pattern across `scales[8]` reinterpreted as `f16`). Both share the 2048-entry `IQ1S_GRID: [u64; 2048]` codebook of signed `i8` 8-element vectors (~16 KB, the largest single grid in the crate). Inner-loop math is `dl × (grid[j] + delta)` — additive bias instead of IQ2/IQ3's multiplicative ±1 sign — needing the new `write_delta_grid` helper. Bit-exact (0 ULP) against the `gguf` Python reference on 65 536-element slices from `bartowski/Mistral-7B-Instruct-v0.3-GGUF` (separate `IQ1_S.gguf` and `IQ1_M.gguf` files; no shared-file trick available like step 3). 6 new unit tests + 2 new cross-validation tests (13.77× / 7.88× faster than Python reference, release-mode best-of-5 — fastest IQ kernels in the crate).
- [x] **`TQ1_0` and `TQ2_0` dequant kernels** (`src/remember/gguf.rs`, commits `72f8e3c` + `111de18` + `53dd3ab`) — two ternary super-quants (54 / 66 B/block) decoding to `{-d, 0, +d}`. `TQ1_0` uses base-3 packing (5 ternaries per `qs` byte, 4 per `qh` byte) decoded via the `pow3 = [1, 3, 9, 27, 81, 243]` multiplication trick. `TQ2_0` is plain 2-bit packing. New `decode_pow3_ternary` helper alongside the existing `write_signed_grid` / `write_delta_grid` family. **First step using synthetic fixtures** — only ~15 BitNet-derivative GGUFs ship TQ types on HuggingFace, but Python `gguf.quants.quantize()` implements both, so a deterministic synthetic random tensor (seed=42, scale=0.1) is the practical fixture source. Bit-exact (0 ULP) against the `gguf` Python reference. 5 new unit tests + 2 new cross-validation tests (35.59× / 26.31× faster than Python reference, release-mode best-of-5 — fastest GGUF kernels in the crate, beating Q5_0's 31.3× record).
- [x] **`MXFP4` dequant kernel** (`src/remember/gguf.rs`, commits `b87d877` + `acfe8a6` + `89e596d`) — 32-element microscaling FP4 (OCP MX standard, added to `ggml` in 2024), 17 B/block: `e: u8` (E8M0 byte exponent) + `qs[16]` (4-bit packed). Distinct from the `IQ*` / `TQ*` family because it uses a standardised IEEE-like sub-block exponent (`E8M0`) rather than a learned codebook, decoded by a new `e8m0_to_fp32_half` helper that matches `llama.cpp`'s `ggml_e8m0_to_fp32_half` (no NaN guard for `e == 0xFF`, deviating from raw OCP MX spec). 16-entry signed `i8` codebook (`K_VALUES_MXFP4`, storing 2× the OCP E2M1 magnitudes) with the doubling cancelled by the half-scale exponent. Same low/high split-nibble layout as `Q4_0` / `IQ4_NL` via the existing `run_legacy_kernel` runner. Bit-exact (0 ULP) against the `gguf` Python reference on a deterministic synthetic fixture (Python `gguf.quants.quantize()` supports MXFP4 too — same synthetic-fixture path as TQ1_0/TQ2_0 from step 5). 4 new unit tests + 1 new cross-validation test (30.14× faster than Python reference, release-mode best-of-5).
- [x] **Cross-validation extension** — landed incrementally with steps 1–6: every `IQ*` / `TQ*` / `MXFP4` kernel ships its own cross-validation test against the `gguf` Python reference (mirrors `ggml-quants.c`). Bit-exactness contract holds at 0 ULP for all 22 of 22 production kernels — **PUSH + tag `v0.4.2`**.

**Deliverable:** `anamnesis` v0.4.2 — anamnesis now dequantises every GGUF block type shipping on HuggingFace. `IQ*`/`TQ*`/`MXFP4` are the last meaningful coverage gap; after this release anamnesis is a drop-in replacement for `llama.cpp`'s CPU dequant path. — **PUSH + tag `v0.4.2`**

**New dependencies:** None. Reuses the `gguf` feature gate, `memmap2`, and `half` from Phase 4.

**Explicitly out of scope:**

- **Big-endian GGUF v3 support** — the parser detects byte-swapped magic and returns `Unsupported`. A reader rewrite can reuse `parse::utils::byteswap_inplace` if demand ever materialises, but no mainstream model is distributed big-endian today. No committed target.
- **CPU SIMD optimisation of pass 2** — cross-cutting optimisation that would benefit FP8, GPTQ, AWQ, and BnB equally, not just GGUF. See [Phase 7: CPU SIMD Pass](#phase-7-cpu-simd-pass).
- **Remote-only NPZ inspection (HTTP-range probe)** — rehomed to its own scheduled milestone, [Phase 4.7](#phase-47-remote-only-npz-inspection-reader-generic-api), shipping in `v0.4.3` ahead of Phase 5.

### Phase 4.7: Remote-only NPZ inspection (Reader-generic API)

**Goal:** Eliminate the bandwidth cliff candle-mi v0.1.10 dogfooding (GemmaScope `open()` flow, 2026-04-30) hit when a 288 MiB layer-0 NPZ download was wasted just to read 16 bytes of `W_enc` shape metadata. Add a reader-generic `inspect_npz_from_reader<R: Read + Seek>` so callers can plug in any positional source — including an HTTP-range-backed adapter — without anamnesis itself taking on a network or TLS dependency. Cuts candle-mi's GemmaScope `open()` cold-start from ~30 s on a 100 Mbps link to <1 s. Call site reference: `candle-mi/src/clt/mod.rs::open_gemmascope` (TODO comment marks the spot).

**Approach:** Lift the file-opening boilerplate out of the existing `inspect_npz(path)` and re-express it as a thin wrapper over a new `inspect_npz_from_reader<R: Read + Seek>(reader: R) -> Result<NpzInspectInfo>`. ZIP places its central directory at the end of the file and `zip::ZipArchive::new` already accepts any `R: Read + Seek`; the existing per-entry `parse_npy_header` flow translates verbatim — only the I/O substrate changes. Anamnesis stays HTTP-free; remote inspection is composed at the call site by feeding an HTTP-range-backed `Read + Seek` adapter (which `hf-fm` already implements for safetensors and can extend to NPZ). Architectural fit: this is the same "parse from reader" doctrine the rest of the crate already follows (`parse_npy_header(&mut impl Read)`, the safetensors header parser, the GGUF mmap reader) — the public API surface just hadn't been pushed up to the inspect layer yet.

- [x] **`inspect_npz_from_reader<R: Read + Seek>` public function** (`src/parse/npz.rs`, commit `660850a`) — accepts any `Read + Seek` source, returns `NpzInspectInfo` exactly as today. `inspect_npz(path)` is now a two-line wrapper that opens a `std::fs::File` and delegates. `NpzInspectInfo` / `NpzTensorInfo` / `NpzDtype` unchanged (already public from v0.3.0). Re-exported from `parse/mod.rs` and `lib.rs` alongside the existing `inspect_npz`.
- [x] **In-memory cursor coverage test** (`src/parse/npz.rs`, `tests/cross_validation_npz.rs`, commit `660850a`) — `inspect_path_and_reader_agree_on_gemma_scope_fixture` asserts field-for-field parity between `inspect_npz(path)` and `inspect_npz_from_reader(Cursor::new(&bytes))` on the real Gemma Scope SAE fixture. Plus three in-module unit tests: `inspect_from_reader_matches_path` (multi-array in-memory archive), `inspect_from_reader_empty_archive`, `inspect_from_reader_rejects_fortran_order`.
- [x] **Range-read access pattern documentation** (`src/parse/npz.rs` rustdoc, commit `660850a`) — the rustdoc on `inspect_npz_from_reader` documents the three logical fetches an HTTP-range adapter must serve (EOCD scan ~64 KiB, central directory ~few KiB, per-entry local file header + NPY header ~512 B), and the prefetch-on-first-seek caching shape that amortises round trips to ~7 small range requests on a typical 5-array Gemma Scope `params.npz` (well under 100 KiB instead of the full ~300 MiB download).

**Phase complete.** The three implementation steps above shipped in commit `660850a`; `v0.4.3` was tagged on 2026-05-01 (commit `9468d03`) and the publish workflow pushed it to crates.io (run `25206873389`, 54 s). — **PUSH + tag `v0.4.3`** ✓

**Deliverable:** `anamnesis` v0.4.3 — reader-generic NPZ inspection. candle-mi (and any other downstream consumer) can now inspect a remote NPZ archive without materialising the data segment by composing `inspect_npz_from_reader` with an HTTP-range-backed `Read + Seek` adapter. Backward compatible: the existing `inspect_npz(path)` entry-point is unchanged (now implemented as a 2-line wrapper). No new dependencies, no feature gate — the whole change is a public-API extension on the already-shipping `npz` feature. — **PUSH + tag `v0.4.3`**

**New dependencies:** None. Reuses the existing `npz` feature gate, the existing `zip` v2 dep, and the existing `NpzInspectInfo` types.

**Why reader-generic, not built-in HTTP:** A `npz-remote` feature with an embedded HTTP/TLS transport (e.g., `ureq` + rustls) was considered and rejected. Reasons: (a) anamnesis's dep tree is consciously lean — adding TLS would expand the audit surface and re-trigger the full publish dry-run gauntlet for a single download-orchestration concern; (b) HTTP belongs in `hf-fm`, which already range-reads safetensors and is the natural home for a `Read + Seek` HTTP adapter; (c) the reader-generic API has the additional payoff of in-memory `Cursor<&[u8]>` inspection, mirroring the in-memory path the v0.4.1 candle-mi dogfooding round opened on the `.pth` side. The HTTP convenience layer can land later in a dedicated crate or as an opt-in feature if multiple downstream users request it.

**Explicitly out of scope:**

- **Built-in HTTP transport** — see "Why reader-generic, not built-in HTTP" above. The HTTP-range-backed `Read + Seek` adapter lives in `hf-fm` (or any other downstream caller), not in anamnesis.
- **Remote `parse_npz` (data-fetching variant)** — only `inspect_npz_from_reader` (metadata-only) is in scope. A reader-generic `parse_npz_from_reader` is a natural follow-up if a concrete downstream need arises, but the current dogfooding signal is exclusively about inspection (16 bytes of shape metadata), not data extraction.

### Phase 4.8: Reader-generic safetensors header parsing

**Goal:** Mirror Phase 4.7's reader-generic doctrine on the safetensors path. Add `parse_safetensors_header_from_reader<R: Read>` so any caller bringing a `Read` source can extract the header without materialising the full file. Unblocks `hf-fm` v0.11.1 (the planned retirement of `hf-fm`'s bespoke `fetch_header_bytes` parser) and removes the only remaining duplicated format-knowledge between the two crates.

**Approach:** The safetensors header is sequential at the start of the file: 8-byte little-endian length prefix, then exactly that many bytes of `JSON`. **No `Seek` is needed** — a plain `Read` suffices. Lift the file-opening boilerplate out of the existing always-on mmap-based `parse()` and expose a reader-generic primitive that reads the prefix, reads the header bytes, and delegates to the existing `parse_safetensors_header(&[u8])`. The `ParsedModel` struct continues to be created exclusively by `parse(path)` (which still mmaps); the new entry-point returns just `SafetensorsHeader` for inspect-only flows. Backward compatible: `parse(path)` and `parse_safetensors_header(&[u8])` are unchanged.

- [x] **`parse_safetensors_header_from_reader<R: Read>` public function** (`src/parse/safetensors.rs`, commit `9a79973`) — reads the 8-byte LE length prefix, reads `header_len` bytes of `JSON` header, then bypasses `safetensors::SafeTensors::read_metadata` (which would require the data section to be present in the buffer) by deserialising the `JSON` directly via `serde_json::from_slice` into `safetensors::tensor::Metadata`. The slice-based and reader-based paths share a private `build_header_from_metadata` helper so both produce field-for-field identical `SafetensorsHeader`s. A 100 MiB sanity cap on the declared header length bounds the worst-case allocation an adversarial source can trigger. Both `parse_safetensors_header` and the new function are now re-exported from the crate root, joining the existing path-based `parse`. ~150 LOC including the helper, the cap constant, and rustdoc.
- [x] **Cross-validation against the upstream `safetensors` Python library** (`tests/fixtures/safetensors_reference/`, `tests/cross_validation_safetensors.rs`, commit `5531720`) — the spec wording "across the existing `FP8` / `GPTQ` / `AWQ` / `BnB` fixtures" is realised as four small `.safetensors` fixtures (340 B–2 KiB each) plus a sibling `<scheme>.expected.json` reference for each, recording exactly what `safetensors.safe_open` reports about that file's header. The references are produced by `generate.py` (`PyTorch` for `FP8` weights via `torch.float8_e4m3fn`, `numpy` for everything else) and triple-checked: raw 8-byte length prefix + `JSON` parse per spec, cross-checked against `safetensors.safe_open`. Five new integration tests (one per scheme + one on-disk `File` reader cover) assert that **both** anamnesis entry points reproduce the Python-sourced reference field-for-field on `header_size`, file metadata, and per-tensor `name`/`dtype`/`shape`/`data_offsets`. A final `headers_must_agree` step locks the fields the Python library does not expose (anamnesis-detected scheme + scheme-specific configs + role classifications). Plus 5 new unit tests in `parse::safetensors::tests` covering minimal substrate equivalence and the three error paths (truncated prefix → `Io`, oversized declared length → `Parse` without allocating, truncated `JSON` tail → `Io`). Total fixture footprint ~5 KiB, all `include_bytes!`/`include_str!`-baked.
- [x] **Range-read access pattern documentation + source-context convention** (`src/parse/safetensors.rs` rustdoc, commit `9a79973`) — full rustdoc on `parse_safetensors_header_from_reader` covering `# Range-read access pattern` (the two contiguous fetches an `HTTP`-range adapter must serve: 8 bytes at offset 0, then `header_len` bytes at offset 8 — total transfer ≈ header size, ~1 MiB on a multi-GB shard, vs the full file), `# Errors`, `# Source context` (the convention fixed in commit `b3b7df8`), and `# Memory`. New "Safetensors Header Inspection" section in [`README.md`](README.md) mirrors the existing `NPZ` entry, and the `lib.rs` crate-level docs surface all three header-parsing entry points (path / slice / reader) in one place.

**Deliverable:** `anamnesis` v0.4.4 — reader-generic safetensors header parsing with proper external cross-validation. Unblocks `hf-fm` v0.11.1 (remote safetensors via anamnesis), retiring the duplicated bespoke parser in `hf-fm/src/inspect.rs::fetch_header_bytes`. — **PUSH + tag `v0.4.4`**

**New dependencies:** None.

**Why no `Seek`:** the safetensors header lives at file offsets `[0, 8 + length)` — exactly the bytes a sequential `Read` produces in order. A `Read + Seek` constraint would force HTTP-range adapters to handle seek-back even though no parser code seeks backwards. Keeping the constraint at `Read` means the simplest possible HTTP-range implementation works (one connection, two contiguous range fetches, never seek-back).

**Source-context convention** *(applies across Phases 4.8 / 4.9 / 4.10)*: errors from the reader-generic primitives describe the **format-level problem** (e.g., *"safetensors header length read failed"*, *"GGUF tensor info malformed"*), **not** the source identity (filename, URL, repo id). The functions are reader-agnostic — the source could be a file, an in-memory `Cursor`, or an HTTP-range adapter. Callers that have a source name should wrap the returned error with that context (that is hf-fm's role in v0.11.1+). This matches anamnesis's existing convention (`parse_safetensors_header(&[u8])` and `inspect_npz_from_reader` already return source-agnostic errors). The recommended rustdoc shape for the new entry-point makes this explicit:

```rust
/// Parses a safetensors header from any `Read` source.
///
/// # Errors
///
/// Returns [`AnamnesisError::Io`] if the reader fails to produce the
/// requested bytes (8-byte length prefix or `length` bytes of JSON).
/// Returns [`AnamnesisError::Parse`] if the header bytes are malformed.
///
/// # Source context
///
/// Errors describe the **format-level problem**, not the source
/// identity. The function is reader-agnostic — the source could be a
/// file, an in-memory `Cursor`, or an HTTP-range adapter. Callers that
/// have a source name (filename, URL, etc.) should wrap the returned
/// error with that context. See the crate-level docs for the
/// recommended pattern.
```

The same `# Source context` block (with format name swapped) should appear on `inspect_gguf_from_reader` and `inspect_pth_from_reader`, so all four reader-generic primitives (`inspect_npz_from_reader` already shipped + the three new ones) document the convention uniformly.

### Phase 4.9: Reader-generic GGUF inspection

**Goal:** Add `inspect_gguf_from_reader<R: Read + Seek>` so any caller bringing a positional source can extract GGUF metadata (header, KV pairs, tensor info table) without materialising the data segment. Unblocks `hf-fm` v0.11.2 (remote GGUF inspection — the originally-promised v0.11.0 feature, now landing on the `HttpRangeReader` adapter `hf-fm` builds in v0.11.0).

**Approach:** The current `parse_gguf` walks a `Cursor<&[u8]>` over a `memmap2::Mmap`. Generalise the cursor to `Read + Seek` so the same adversarial-input-guarded logic works against any positional source. The path-based `parse_gguf(path)` becomes a thin wrapper that mmaps the file and delegates. The existing public types (`ParsedGguf`, `GgufInspectInfo`, `GgufTensor`, `GgufTensorInfo`, `GgufType`, `GgufMetadataValue`, `GgufMetadataArray`) all carry over unchanged — only the I/O substrate is parameterised. Adversarial-input guards (caps on tensor count, KV count, string length, array length, nesting depth, dimension count, element product) are preserved verbatim.

- [x] **Refactor `parse_gguf` cursor pattern off `memmap2::Mmap`** (`src/parse/gguf.rs`, commit `87720ca`) — the slice-based `Cursor<'a>` over `memmap2::Mmap` is replaced by a generic `GgufReader<R: Read + Seek>` running over any positional substrate. The path-based `parse_gguf(path)` is now a thin wrapper that mmaps the file and delegates to a new `parse_gguf_from_reader` core via `std::io::Cursor` over the mapping. The `ParsedGguf` zero-copy tensor-data contract and every adversarial-input guard (caps on tensor count, KV count, string length, array length, nesting depth, dimension count, element product, per-tensor alignment, end-of-data bounds) are preserved verbatim; all 17 prior `parse::gguf::tests` pass unchanged.
- [x] **`inspect_gguf_from_reader<R: Read + Seek>` public function** (`src/parse/gguf.rs`, `src/parse/mod.rs`, `src/lib.rs`, commit `e94c0ed`) — accepts any `Read + Seek` source, returns `GgufInspectInfo` field-for-field identical to `parse_gguf(path).inspect()`. A shared `build_inspect_info` helper consumes the parsed front matter so the two entry points are guaranteed substrate-equivalent — every field of the resulting `GgufInspectInfo` is computed by the same code regardless of which entry point produced it. Re-exported from `parse/mod.rs` and the crate root, joining the existing reader-generic family (`parse_safetensors_header_from_reader`, `inspect_npz_from_reader`).
- [x] **In-memory cursor coverage + range-pattern docs** (`src/parse/gguf.rs`, `tests/cross_validation_gguf.rs`, commit `a8909f2` + real-fixture follow-up commit `c0f7a06`) — 4 in-module unit tests on synthetic GGUFs (`inspect_from_reader_matches_path_minimal`, `inspect_from_reader_matches_path_mixed_dtypes`, `inspect_from_reader_accepts_header_only_file`, `inspect_from_reader_propagates_parse_errors`) plus an `#[ignore]`-gated integration test (`substrate_equivalence_real_gguf_models`) that walks every `*.gguf` under `tests/fixtures/gguf_reference/models/` and asserts equivalence against the path-based variant. Locally-confirmed substrate-equivalent on 17 of 17 real GGUFs spanning 4 architectures × 11 distinct dtypes × 84 MiB to 2.7 GiB: `bartowski/SmolLM2-135M-Instruct` (8 quants), `bartowski/Mistral-7B-Instruct-v0.3` (5 quants), `bartowski/Qwen2.5-{0.5,1.5}B-Instruct-IQ2_M`, `TheBloke/TinyLlama-1.1B-chat-v1.0` (Q2_K, Q5_0). Full rustdoc on `inspect_gguf_from_reader` covers the `# Range-read access pattern`, `# Source context`, and `# Memory` sections, mirroring the Phase 4.7 / 4.8 conventions. **Bonus: Tier 1 perf win (commit `c1b85bb`)** — the user-supplied reader is wrapped internally in `BufReader<R>` (64 KiB), collapsing the parser's many small `read_exact` calls into one syscall per buffer-fill on a `File` substrate. Reader/mmap ratio collapsed from median 51.7× / mean 56.6× (slower) to median 1.0× / mean 1.0× (parity). `tests/bench_gguf_inspect_adhoc.rs` lands the regression-detection harness; `docs/perf-experiments.md` Experiment 5 records method + numbers + trade-offs accepted. **PUSH + tag `v0.4.5`**

**Deliverable:** `anamnesis` v0.4.5 — reader-generic GGUF inspection. Unblocks `hf-fm` v0.11.2 (remote GGUF inspect). A 2 GiB quantised GGUF inspectable in a few range requests fetching the front-loaded metadata — no weight data downloaded. — **PUSH + tag `v0.4.5`**

**New dependencies:** None. Reuses the existing `gguf` feature gate, `memmap2`, and the existing public types.

**Why `Read + Seek` (and not just `Read`) for GGUF:** unlike safetensors's prefix-then-JSON layout, GGUF's parser reads back-and-forth across the front matter (e.g., it computes the absolute tensor-data offset by combining the relative offsets in the tensor-info table with the post-tensor-info `data_section_start`, then validates per-tensor alignment relative to that anchor). The simplest correct refactor preserves the existing positional access pattern via `Seek`. A pure-`Read` reformulation is possible but would require restructuring the parser into a strict forward pass — out of scope for this phase.

### Phase 4.10: Reader-generic PTH inspection

**Goal:** Add `inspect_pth_from_reader<R: Read + Seek>` so any caller bringing a positional source can extract `PyTorch` `.pth` tensor metadata (names, dtypes, shapes, sizes) without materialising the tensor data files inside the ZIP archive. Unblocks `hf-fm` v0.11.3 — closing the remote-inspect matrix across all four tensor formats anamnesis supports.

**Approach:** Largest of the three Phase 4.8 / 4.9 / 4.10 lifts. PTH's metadata lives in `data.pkl` mid-archive; the existing pickle VM zero-copies `Cow::Borrowed` slices from `memmap2::Mmap`. The reader-based path: range-fetch the ZIP central directory, locate `data.pkl`'s entry, materialise its bytes (typically <100 KiB even on torchvision-class 300 MB models), then run the existing pickle interpreter on that owned buffer. The local-file `parse_pth(path)` zero-copy contract is preserved — the reader-based entry-point returns just metadata (`PthInspectInfo`), not the zero-copy `ParsedPth` (which requires the mmap to outlive the borrowed tensor data slices).

- [x] **`inspect_pth_from_reader<R: Read + Seek>` public function** (`src/parse/pth.rs`) — range-fetches the ZIP central directory via `zip::ZipArchive::new(reader)` (which already accepts any `R: Read + Seek`, same as Phase 4.7's NPZ path), locates `data.pkl` and the optional `byteorder` entry by walking the central directory once and stripping the archive prefix (so both `archive/data.pkl` and `{model_name}/data.pkl` are accepted), materialises `data.pkl`'s bytes (with a 100 MiB cap to defend against adversarial central directories), and runs the existing pickle interpreter on that owned `Vec<u8>`. Returns `PthInspectInfo` (metadata only — no tensor data). Re-exported from `parse/mod.rs` and the crate root, joining the existing reader-generic family (`parse_safetensors_header_from_reader`, `inspect_npz_from_reader`, `inspect_gguf_from_reader`).
- [x] **Pickle VM contract preserved** — the local-file (`parse_pth`) zero-copy contract (`Cow::Borrowed` slices into the mmap) is unchanged; `ParsedPth::tensors()` and the existing `.pth` → `.safetensors` round-trip continue to work identically. Two shared private helpers — `interpret_pickle_to_meta(&[u8]) -> Vec<TensorMeta>` and `build_pth_inspect_info(&[TensorMeta], bool) -> PthInspectInfo` — are now used by both entry points so the pickle interpreter's security allowlist (`is_allowed_global`) and the inspect-summary field arithmetic are guaranteed identical across the path-based and reader-generic paths. The 9 existing cross-validation tests in `tests/cross_validation_pth.rs` continue to pass byte-for-byte against the `PyTorch`-generated reference on the 3 AlgZoo fixtures.
- [x] **In-memory cursor coverage + range-pattern docs** (`src/parse/pth.rs`, `tests/cross_validation_pth.rs`) — 8 in-module unit tests on synthetic `.pth` archives covering substrate-equivalence (`inspect_from_reader_matches_path_empty_dict`), the optional `byteorder` entry (`inspect_from_reader_honours_byteorder_entry`), older-style prefixes (`inspect_from_reader_accepts_older_prefix`), and every documented rejection branch (legacy pre-1.6 pickle, wrong magic, too-small file, missing `data.pkl`, oversized `byteorder`). Plus the integration test `substrate_equivalence_algzoo_fixtures` in `tests/cross_validation_pth.rs` which asserts field-for-field equivalence of `PthInspectInfo` across the three substrates `parse_pth(path).inspect()`, `inspect_pth_from_reader(File::open(path)?)`, and `inspect_pth_from_reader(Cursor::new(fs::read(path)?))` on every AlgZoo fixture. Full rustdoc on `inspect_pth_from_reader` covers the `# Range-read access pattern` (EOCD scan + central directory read + one bulk read of `data.pkl` + optional one bulk read of `byteorder`, ~<100 KiB total even on a 300 MB torchvision `.pth`), `# Why metadata-only (no parse_pth_from_reader)`, `# Performance` (no internal `BufReader` — the I/O pattern is bulk-oriented through `zip`, unlike `GGUF`'s many small `read_exact` calls), `# Errors`, `# Source context`, and `# Memory` sections, mirroring the Phase 4.7 / 4.8 / 4.9 conventions. — **PUSH + tag `v0.4.6`**

**Deliverable:** `anamnesis` v0.4.6 — reader-generic `.pth` inspection. Unblocks `hf-fm` v0.11.3, closing the remote-inspect matrix (`safetensors` / `NPZ` / `GGUF` / `PTH` all remotely inspectable through the same `HttpRangeReader` substrate). — **PUSH + tag `v0.4.6`**

**New dependencies:** None. Reuses the existing `pth` feature gate, `zip` v2, and `memmap2`. The reader-based path goes through `zip::ZipArchive::new(reader)` (already accepts any `R: Read + Seek`).

**Why metadata-only (no `parse_pth_from_reader`):** the local-file `parse_pth(path)` returns a `ParsedPth` with `Cow::Borrowed` zero-copy slices into the mmap. A reader-based equivalent that preserved that contract would need the reader to outlive every borrowed slice — workable for some `Read + Seek` substrates but awkward for HTTP-range adapters where each tensor read is a fresh fetch. Constraining the reader-based entry-point to **inspect-only** (returns owned `PthInspectInfo`) sidesteps the borrowing complexity for the v0.11 use case (browsing tensor metadata before downloading), while leaving room for a future `parse_pth_from_reader` if a streaming-data use case develops.

**Explicitly out of scope (across Phases 4.8 / 4.9 / 4.10):**

- **HTTP transport in anamnesis** — same as Phase 4.7. anamnesis stays HTTP-free; the network layer lives in `hf-fm`'s `HttpRangeReader` adapter (or any other downstream consumer's equivalent).
- **Reader-generic data-fetching variants** (`parse_pth_from_reader`, `parse_gguf_from_reader` returning full data) — only metadata-only inspect paths are in scope for this matrix-completion arc. Reader-generic full-data paths are natural follow-ups if concrete downstream needs arise (the current need is exclusively inspection).
- **Reader-generic remember/dequant** — full dequantisation requires touching every byte of every quantised tensor; the structural advantage of reader-based access (only fetch what you touch) collapses. Out of scope unless a streaming-dequant story develops in parallel ([Phase 10: Streaming Output](#phase-10-streaming-output) territory).

### Phase 5: Quantization (Lethe)

**Goal:** The opposite direction — take full-precision weights and quantize them. Built on parse (read the source) + lethe (compress). With Phase 4 complete, this enables **cross-format conversion** via the dequantize-then-requantize path (e.g., GPTQ safetensors → BF16 → GGUF Q4_K_M). No Rust tool can do this today.

**Strategy:** Phase 5 ships a deliberately narrow first cut and **claims its own minor version (`v0.5.0`)** because Lethe is the architectural inverse of Anamnesis (Phases 1–4.5) and deserves a minor-version namespace of its own — even when the initial coverage is one quant family rather than the full encode-side matrix. The minimum viable end-to-end pipeline ships **BnB encode** as step 1 (this Phase), then proceeds through Phase 6 (Format Conversion Matrix), Phase 6.5 (Benchmarking), and Phase 8 (Python Bindings) to validate the complete `parse → remember → forget → convert → bind` chain on a single quant family. Once the architecture is proven end-to-end, the remaining encode kernel families (FP8, GGUF legacy block, GGUF K-quants, etc.) land in **Phase 8.5+** as a focused encode-completion milestone.

**Sequencing — three sub-steps, one minor version (`v0.5.0`):**

- [x] **Step 1a — `BnB` encode (`NF4` + `FP4` + `INT8`, plain) + round-trip harness** (`src/lethe/bnb.rs` + `src/lethe/round_trip.rs`, shipped commit `a5c452d`) — three encode kernels mirroring the Phase 2 BnB decode kernels: `encode_bnb4` (codebook-supplied, drives both `NF4` and `FP4`) uses 16-entry codebook lookup with nearest-codebook-entry search (linear scan, exact-bit-match priority for the `+0`/`-0` disambiguation); `encode_bnb_int8` is per-row absmax-derived scale + `round(x / scale)` with `[-128, 127]` clamp. `pub const NF4_CODEBOOK` and `pub const FP4_CODEBOOK` exposed as drop-in `quant_map` substitutes for callers without a reference file. Plus the convenience `encode_bnb4_compute_absmax` / `encode_bnb_int8_compute_scb` entry points for Phase-6's "quantize from BF16 source" path. The round-trip harness (`assert_bnb4_decode_encode_round_trip`, `assert_bnb_int8_decode_encode_round_trip`) asserts `encode(decode(q, scale, codebook)) == q` across the entire valid index range — the codebook itself is the oracle, no external Python reference needed. Cross-validation against PyTorch `bitsandbytes` on the existing 4 Llama-arch BnB fixtures: re-encode the decoded BF16 and assert byte equality with the original `weight_data`. **Result: byte-exact round-trip on every Llama fixture (NF4, FP4, INT8 — including FP4 thanks to the sign-of-zero preservation discovery; see below).**

- [x] **Step 1b — Cross-architecture plain-FP4 fixture (Qwen3)** (shipped commit `24cba42`) — downloaded `ema1234/qwen_mcqa_bnb_fp4` via `hf-fm`, extended `generate_bnb.py` with a Qwen3 entry, added `cross_validate_encode_qwen3_mcqa_fp4` to `cross_validation_bnb_encode.rs`. Result: byte-exact round-trip on Qwen3 (0 / 2048 byte diffs), confirming the sign-of-zero preservation rule generalises beyond a single org's quantization pipeline. PyTorch quantize-timing sidecars wired in for every fixture.

- [x] **Step 1c — `encode_bnb4_double_quant` + cross-architecture DQ NF4 fixtures (Qwen2.5 + Phi-3.5)** — implemented the fourth encode kernel: a `recover_double_quant_absmax` helper mirrors the decode-side nested-absmax recovery (`nested_quant_map[absmax_byte] * nested_absmax`), then delegates to the existing `encode_bnb4_core` with the recovered `f32` absmax. Validated first against the existing `medmekk/Llama-3.2-1B-Instruct-bnb-nf4-double-quant` fixture (byte-exact: 0 / 2048 diffs), then against two new cross-architecture fixtures: `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit` and `unsloth/Phi-3.5-mini-instruct-bnb-4bit` (both confirmed double-quant via `hf-fm inspect` *before* download — saved ~3.8 GiB of speculative bytes). Cross-validation result: **byte-exact round-trip on all three DQ fixtures across three architecture families (Llama, Qwen2.5, Phi-3.5)**. — **commit** — **PUSH + tag `v0.5.0`**.

**Boundary-pushing finding (sign-of-zero preservation, surfaced in Step 1a):** The on-disk `bitsandbytes` Python `FP4` `quant_map` stores `+0.0` at *both* index 0 *and* index 8 — collapsing the `±0` pair. This makes the naive `decode → encode` round-trip mathematically impossible *under the existing decode contract* (our decode was required to bit-exactly match `bitsandbytes`' Python decode, which is itself lossy on the sign of zero). Rather than accept decode-equivalence as the operative contract for `FP4`, Step 1a introduces a narrow, principled tweak in [`dequantize_bnb4_to_bf16`](src/remember/bnb.rs): when a looked-up codebook entry is exactly `+0.0` AND the nibble has its high bit set (`nibble & 0x8 != 0`), the emitted `BF16` is `-0.0` instead of `+0.0`. This recovers the sign information `bitsandbytes`' Python decode discards, so `encode_bnb4` can recover the original nibble byte-exactly. The divergence from `bitsandbytes`' Python is arithmetically invisible (`+0` and `-0` are IEEE 754 equivalent), affects `0.2 %` of `FP4` elements, and is a no-op for `NF4` and every codebook whose upper-half indices hold non-zero entries. The existing `cross_validation_bnb.rs` `compare_bf16` helper now treats `±0` as equivalent so the decode test stays green. Step 1b's Qwen3 fixture is the cross-architecture validation that the rule generalises.

**Ecosystem finding (surfaced during Step 1b candidate hunt, motivates Step 1c):** Cross-architecture BnB-4bit candidate selection via `hf-fm search` + `hf-fm inspect` (HTTP-range header fetches, no download) revealed that **every** non-Llama BnB-NF4 candidate inspected uses double-quant: `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit`, `unsloth/Phi-3.5-mini-instruct-bnb-4bit`, `Blackwood416/Qwen3-0.6B-BNB-NF4`, `medmekk/Qwen2.5-Coder-0.5B-Instruct-bnb-4bit-nf4` — all expose `nested_absmax` / `nested_quant_map` companion tensors. This is not coincidence: bitsandbytes' default 4-bit-NF4 setting IS double-quant (33 % memory saving over plain NF4), so virtually all non-Llama `bnb-4bit` uploads in the wild are DQ. **Plain (non-DQ) NF4 is effectively a Llama-fixture-only phenomenon in the current ecosystem.** Consequence: `encode_bnb4_double_quant` is not a polish item — it's the difference between "anamnesis can encode toy fixtures" and "anamnesis can encode real-world BnB models". Promoted from a deferred follow-up to a required Step 1c gate on `v0.5.0`.

**Deliverable:** `anamnesis` v0.5.0 — **Lethe lands with full BnB encode coverage.** Phase 5 ships `BnB` encode (`NF4` / `FP4` / `INT8`, plain + double-quant) + the round-trip validation harness that every subsequent encode kernel will reuse + the sign-of-zero preservation tweak in the decode path + cross-architecture fixtures (Llama / Qwen3 / Qwen2.5 / Phi-3.5) proving the rules generalise beyond a single org's quantization pipeline. Lethe gets its own minor version even though the initial coverage is one quant family — the architectural commitment (`src/lethe/`, the round-trip oracle, the encode-side public API surface, the encode side proven against the full ecosystem) is what `v0.5.0` represents. The remaining encode kernel families (FP8, GGUF legacy block, GGUF K-quants, IQ4_NL/XS, MXFP4, TQ) move to **Phase 8.5: Lethe Encode Completion**, landing after the full chain has been validated through Python bindings in Phase 8. — **PUSH + tag `v0.5.0`**

### Phase 6: Format Conversion Matrix

**Goal:** Wire the conversion pipeline — `parse → remember → forget → write` end-to-end via a single CLI primitive. With Phase 5 complete (BnB encode), v0.6.0 ships every decode-side conversion (any input format → safetensors BF16) plus the BnB encode path (BF16-safetensors → BnB safetensors; full input coverage for the `bnb-nf4` target lands in [Phase 6.14](#phase-614-convert-matrix-completion-bf16-hub)); the remaining encode-side rows of the matrix (GGUF / FP8 / IQ / TQ / MXFP4) light up at v0.8.5 once Phase 8.5 lands the missing encode kernels, accessible through the same `convert()` API and the same `amn convert` subcommand.

**Key conversions unlocked (cumulative across Phase 6 and Phase 8.5):**

| From | To | Use case | Available |
|------|----|----------|-----------|
| AWQ safetensors | safetensors BF16 | Remove quantization for fine-tuning | v0.6.0 |
| GGUF | safetensors BF16 | Load llama.cpp models in candle/burn | v0.6.0 |
| NPZ | safetensors | Migrate JAX weights to HuggingFace ecosystem | v0.6.0 |
| PyTorch .pth | safetensors | Migrate legacy PyTorch weights | v0.3.1 (shipped) |
| safetensors BF16 | safetensors BnB-NF4 | Compact BnB-quantized model from BF16 source | v0.6.0 (Phase 5 forget) |
| GPTQ safetensors | GGUF Q4_K_M | Deploy HuggingFace models in llama.cpp/Ollama | **v0.8.5** (Phase 8.5) |
| safetensors BF16 | GGUF Q4_K_M | Quantize for local inference | **v0.8.5** (Phase 8.5) |
| safetensors BF16 | GGUF IQ4_XS | Compact "small but accurate" GGUF for local inference | **v0.8.5** (Phase 8.5) |

- [x] GGUF output writer (header + tensor info + F32 / BF16 passthrough emit) — full quantized-block emit (K-quant / IQ / TQ / MXFP4) unlocks once Phase 8.5 encoders land and plug into the same writer scaffold — **commit `6768ee0`**
- [x] `amn convert` CLI subcommand — `amn convert model.gguf --to safetensors -o model.safetensors` (and `--to bnb-nf4` from v0.6.0); `--to gguf-q4km` and other quantized targets light up at v0.8.5 via the same dispatch — **commit `f5cdee2`**
- [x] Cross-format round-trip validation — 13 byte-exact integration tests in `tests/cross_validation_convert.rs` covering every v0.6.0-available format pair, both directions where reversible (NPZ→ST, PTH→ST, ST-BF16↔GGUF byte-exact loop, BnB-NF4 byte-exact vs Python `bitsandbytes` fixture, BnB-NF4 idempotency, multi-hop NPZ→ST→GGUF, mixed-dtype, multi-dim 1-D/3-D/4-D, non-default alignment, empty-GGUF). Plus a size-matched `t14_perf_vs_python_size_matched` (`#[ignore]`d) that reports rust µs against six checked-in Python sidecars (numpy / gguf-py / bitsandbytes / torch.load + safetensors.torch, plus PyTorch-CPU equivalents for the two non-PyTorch paths) at the same 4096×4096 shape — commits `eb030a9` + `2897344` + `d00435b` + `895e4bc`.

**Deliverable:** `anamnesis` v0.6.0 — the convert primitive plus every decode-side conversion (any input → safetensors BF16) and the BnB encode path (BF16 safetensors → BnB-NF4 safetensors; full input coverage for the `bnb-nf4` target lands in [Phase 6.14](#phase-614-convert-matrix-completion-bf16-hub)). No Rust or Python tool offers this convert primitive today. The remaining encode-side rows complete in Phase 8.5. — **PUSH + tag `v0.6.0`**

### Phase 6.5: Benchmarking & Performance Validation

**Goal:** Establish reproducible performance baselines before the Python bindings expose anamnesis to a much larger audience. Benchmark every dequantization kernel (throughput), validate memory efficiency claims (peak heap), and cross-validate against the dominant real-world GGUF distribution channel (Ollama) — not just bartowski/TheBloke quantisations. All benchmarking and validation infrastructure is dev-only — zero impact on the published crate.

**Approach:** Use [criterion](https://github.com/bheisler/criterion.rs) for runtime throughput benchmarks (statistical rigor, regression detection, baselines). Use [dhat-rs](https://github.com/nnethercote/dhat-rs) for peak-heap assertions in integration tests. Both are `[dev-dependencies]` only — bench code in `benches/`, memory tests in `tests/`. Ollama-pulled GGUF blobs become shared real-world fixtures consumed by both the criterion benches (one extra row alongside the synthetic 4096×11008 fixtures) and a new `tests/cross_validation_ollama.rs` cross-validation suite — same fixture serves double duty.

**Real-world cross-validation (`tests/`) — Ollama track (begin here):** The dequant correctness claim has been validated against `gguf-py` reference on bartowski/TheBloke fixtures; extending the validation to a real Ollama-distributed blob proves it on the dominant local-LLM distribution channel. Ollama stays a test-time external tool (invoked via `Command::new` / pre-cached blobs); anamnesis itself takes on no Go dependency.

- [x] **Fixture target: `llama3.2:1b` (Q8_0, 1.23 GiB on disk).** Selected after probing the local Ollama 0.24.0 cache for the smallest available model: Ollama defaults to Q8_0 for the 1B size tier, so the first Ollama cross-validation hits the Q8_0 kernel (anamnesis already validates Q8_0 against bartowski's SmolLM2 — this becomes "same kernel, second real-world distribution channel"). Slice 65,536 elements from `blk.0.attn_q.weight` (shape `[2048, 2048]`, Q8_0, structurally stable across Llama-arch releases): 2,048 Q8_0 blocks × 34 bytes = 68 KiB raw + 128 KiB BF16 reference, total ~200 KiB committed — same shape as the existing `tests/fixtures/gguf_reference/smollm2_q8_0.bin` slice fixtures, well under the 10 MiB published-tarball cap. A `generate_ollama_fixture.py` refresh script (alongside the slice) mirrors the pattern of `tests/fixtures/convert_reference/generate_convert_timings.py`; running it on a different Ollama install or after `ollama pull` regenerates both the raw slice and the gguf-py reference dequant output. Larger K-quant Ollama fixtures (`Q4_K_M`) deferred as a polish item — adding one is a `ollama pull llama3.2:1b-instruct-q4_K_M` + script re-run, no test-side changes — **commit `899e601`**
- [x] `cross_validation_ollama.rs` — cross-validate dequant byte-exact against the committed gguf-py reference on the Ollama-pulled slice, mirroring `cross_validation_gguf.rs`'s shape. Validates that anamnesis correctly consumes real-world Ollama distribution channels, not just bartowski/TheBloke quantisations. **Result: 0 ULP mismatches on the 65,536-element Q8_0 slice from `llama3.2:1b`** — **commit `6a13bde`**
- [x] `resolve_ollama_model(name)` helper behind a new `ollama` feature flag — reads `~/.ollama/models/manifests/registry.ollama.ai/library/<name>/<tag>` JSON, follows the `application/vnd.ollama.image.model` layer's `digest` to `~/.ollama/models/blobs/sha256-<hash>`, returns the `PathBuf`. Wire into `detect_format` so `amn inspect ollama:llama3.2:1b` works without manual blob-hash lookup. Pure path resolution, no network, no Go interop. End-to-end CLI smoke verified: `amn inspect ollama:llama3.2:1b` returns the parsed GGUF inspect info. **No third-party dependency added** — pure stdlib + `serde_json` (already a runtime dep) — **commit `e671dc1`**

**Runtime benchmarks (`benches/`):**

- [x] Dequantization throughput — 7 synthetic-fixture bench groups in `benches/dequant.rs` (FP8 per-tensor + fine-grained, GPTQ INT4 g128, AWQ INT4 g128, BnB NF4 b64, BnB INT8, GGUF Q4_K), all on a transformer-layer-sized 4096×11008 fixture, reported as Gelem/s. Quick-sanity range on the dev machine: **657 Melem/s → 2.01 Gelem/s** — **commit `3c216fa`**
- [x] Dequantization throughput on the Ollama fixture — bench group `dequant_gguf_q8_0_ollama` in `benches/dequant.rs` runs on the **same** 65,536-element Q8_0 slice the [`cross_validation_ollama`](tests/cross_validation_ollama.rs) suite validates byte-exactly against the `gguf-py` reference. Throughput number paired with a correctness guarantee on real Ollama distribution data: **3.92 Gelem/s** on the dev machine — **commit `3c216fa`**
- [x] Parsing throughput — `benches/parsing.rs` with five bench groups: an `fs::read` baseline plus header-only parses for safetensors, NPZ, PTH (AlgZoo fixture), GGUF. Baseline + each parser report `Throughput::Bytes(fixture_size)` so the README can frame each parser as a multiplier over raw I/O. Dev-machine sanity: `fs::read` 4.30 GiB/s, `parse_safetensors_header` 193 µs, `inspect_npz` 1.56 ms, `inspect_pth` 167 µs, `inspect_gguf` 98 µs — **commit `a163327`**

**Peak-memory validation (`tests/`):**

- [x] Peak-heap assertions for GPTQ/AWQ dequantization — `dhat-rs` instrumented tests at [tests/peak_heap_gptq.rs](tests/peak_heap_gptq.rs) and [tests/peak_heap_awq.rs](tests/peak_heap_awq.rs). Two `#[ignore]`d assertions per file (1M-element + 45M-element fixtures) that compare the dhat-observed `max_bytes` against the decomposed ceiling `output_size + K × out_features × 4` with K=5 slack. Result: **observed scratch matches the documented `3 × out_features × 4` claim to the byte at both scales**, on commits `6d53a77` (GPTQ) and `9d24eb9` (AWQ)
- [x] Peak-heap assertions for BnB double-quant — `dhat-rs` instrumented test at [tests/peak_heap_bnb_dq.rs](tests/peak_heap_bnb_dq.rs). Uses a tighter noise-tolerance assertion (4 KiB above the documented `num_blocks × 4 + block_size × 4` scratch) rather than a ratio multiplier, so a single accidental `Vec<u8>[num_blocks × 4]` intermediate would trip immediately. Result: **zero excess observed** beyond the documented scratch at both 1M and 45M element scales, on commit `5190a5d`

**New dev-dependencies:** `criterion` (runtime), `dhat` (peak heap). Not compiled into the published crate. `ollama` feature flag (for the path-resolution helper) is library-side and adds no third-party dependency — pure stdlib + `serde_json` (already a runtime dep).

**Out of scope (deferred or never):**

- **Architecture-aware GGUF metadata *auto-generation* — making `amn convert --to gguf` emit an inference-loadable model by itself.** Synthesising the Llama-architecture KV metadata (`llama.context_length`, `llama.embedding_length`, the 32K-string `tokenizer.ggml.tokens` array, …) cannot be derived from the tensors; it is a model-packaging layer, not a tensor-format-conversion layer, and belongs in a downstream crate (e.g. `hf-fetch-model`, which already reads `config.json`), not anamnesis. *(The format-level half — a caller-supplied GGUF metadata **pass-through**, where the caller provides the KV and anamnesis writes it verbatim — IS in scope, in [Phase 6.14](#phase-614-convert-matrix-completion-bf16-hub).)*
- **Ollama as a perf comparison column in the README.** `ollama create` is the full conversion pipeline (Go converter + llama.cpp quantize); it does not expose a "just the convert math" timer. Any timing comparison would mostly measure orchestration overhead, not the work anamnesis does. The cross-validation suite proves correctness; the synthetic-fixture criterion benches plus the Ollama-fixture criterion bench prove throughput.

**Deliverable:** Baselines recorded, CI can detect regressions, performance claims in README are backed by reproducible numbers, dequant correctness validated against a real Ollama-distributed model. No version bump — this is infrastructure, not a user-facing release.

### Phase 6.6: Security Hardening — Unguarded Allocations

**Goal:** Close the two unguarded-allocation findings surfaced by auditing anamnesis's parsers against the [`candle #3533`](https://github.com/huggingface/candle/issues/3533) DoS pattern. Patch-level release: user-facing behaviour is unchanged for legitimate files; malicious or malformed inputs fail fast with a clear error rather than driving multi-GiB allocations. No new public API, no breaking changes.

**Origin — `candle #3533` and the audit it triggered:** [`candle #3533`](https://github.com/huggingface/candle/issues/3533) (filed 2026-05-14) disclosed an unguarded-allocation DoS pattern in candle's GGUF parser: a length / count / dimension field is read from a binary file header as `u32` or `u64`, then used directly as the size argument of `vec![T; n]`, `Vec::with_capacity(n)`, or `for _ in 0..n` with no upper-bound check. A 37-byte malicious GGUF declaring `n_dimensions = 2^32 - 1` drives the parser into ~32 GiB immediate allocation. Same bug class as [CVE-2025-66960](https://nvd.nist.gov/vuln/detail/CVE-2025-66960) (ollama `fs/ggml/gguf.go`, fixed) — CWE-770 (Allocation of Resources Without Limits) + CWE-1284 (Improper Validation of Specified Quantity) + CWE-400 (Uncontrolled Resource Consumption). The candle issue notes that the sibling `safetensors` crate caps with `MAX_HEADER_SIZE` — the upstream pattern anamnesis already mirrors on its own safetensors path.

A hf-fm-side audit (2026-05-19) cross-checked all four anamnesis parsers against the candle #3533 sites:

- **GGUF** ([src/parse/gguf.rs](src/parse/gguf.rs)) — **fully capped.** All five candle sites have explicit gates (`MAX_STRING_LEN = 16 MiB`, `MAX_ARRAY_LEN = 16 M`, `MAX_TENSOR_COUNT = 1 M`, `MAX_KV_COUNT = 1 M`, `MAX_TENSOR_DIMS = 8`) plus `PREALLOC_SOFT_CAP = 256` defeating the "declared 1 M, pre-allocated immediately" `with_capacity` sub-pattern. No change required.
- **Safetensors** ([src/parse/safetensors.rs](src/parse/safetensors.rs)) — **fully capped.** Reader-path 100 MiB header cap at [src/parse/safetensors.rs:1068-1077](src/parse/safetensors.rs#L1068-L1077); slice-path delegates to the upstream `safetensors` crate's own `MAX_HEADER_SIZE`. No change required.
- **NPZ** ([src/parse/npz.rs](src/parse/npz.rs)) — **one unguarded site** (NPY `header_len`, finding A below).
- **PTH** ([src/parse/pth.rs](src/parse/pth.rs)) — **one mmap-path inconsistency** (`MAX_PKL_SIZE` applied in reader path only, finding B below).

**Sequencing — two findings, one patch version (`v0.6.1`):**

- [x] **Step 1 — Finding B (PTH mmap path):** apply the existing `MAX_PKL_SIZE = 100 MiB` constant (defined at [src/parse/pth.rs:40](src/parse/pth.rs#L40)) in `parse_pth` before slicing the mmap. Currently only the reader-based `inspect_pth_from_reader` path consults this cap; the mmap path slices `data.pkl` from the ZIP central directory bounded only by file size. On a 10 GiB ZIP whose `data.pkl` entry claims 10 GiB, the pickle VM would process it. Shipped as a shared `enforce_pkl_size_cap` helper called from both entry points (so the bound is identical by construction), plus a `checked_add` on the mmap slice end. — **commit `714af0d`**

- [x] **Step 2 — Finding A (NPZ `header_len`):** introduce `const NPY_MAX_HEADER_BYTES: usize = 1 << 20;` (1 MiB) and reject `header_len > NPY_MAX_HEADER_BYTES` before the `vec![0u8; header_len]` in `parse_npy_header`. Closes the NPY-v2/v3 4 GiB declared-header window: `header_len` is decoded as `u16` (v1, 64 KiB max — already safe by datatype) or `u32` (v2/v3, can reach 4 GiB). Real NPY headers are <1 KiB; 1 MiB is generous enough for any plausible legitimate file. The single site covers every NPZ path (NPZ extraction is always reader-based). — **commit `4d6a005`**

- [x] **Step 3 (defence-in-depth) — NPZ shape-product cap:** the existing `checked_mul` at [src/parse/npz.rs](src/parse/npz.rs) prevents the `usize` overflow case (catches `shape = (usize::MAX, 1)` correctly), but rejection only happens at the arithmetic-overflow boundary. Added `const NPZ_MAX_ARRAY_BYTES: u64 = 1 << 33;` (8 GiB, matching real tensor sizes on consumer GPUs; `u64` so the literal compiles on 32-bit targets) and an explicit reject before the `vec!` in `read_array_data`, making rejection deterministic well below overflow. Same shape as candle #3533's recommended `GGUF_MAX_ARRAY_ELEMENTS`. Defence in depth, not a fix for a reachable bug. — **commit `7ceb7ce`**

- [x] **Step 4 (defence-in-depth) — Pickle 4-byte-length per-allocation cap:** the 4-byte-length pickle opcodes (`BINUNICODE`, `BINSTRING`, `BINBYTES`) each clone an owned `String`/`Vec<u8>` of the declared length from the `pkl_data` slice. Step 1 closes the gross case (multi-GiB stream), but a single ~99 MiB payload inside an otherwise-legitimate 100 MiB pickle would still allocate in one shot. Added `const MAX_PICKLE_PAYLOAD: usize = 64 * 1024 * 1024;` (64 MiB — far above any realistic `state_dict` key or storage-class name) and an `enforce_pickle_payload_cap` guard on all three opcodes, in the shared `interpret_pickle_to_meta` path so it protects **both** the mmap and reader entry points (not just mmap). — **commit `d7d5f11`**

- [x] **Step 5 — Regression tests:** for each fix above, a test constructs a minimal malicious fixture and asserts the parser returns `Err` rather than panicking or OOM-ing, mirroring candle #3533's "tiny PoC → expect `Err`" pattern. Landed as in-module `#[cfg(test)]` tests (rather than `tests/parse_dos_*.rs`) because the guards live in private functions and `zip` is an optional runtime dependency unavailable to integration-test crates — the in-module tests build fixtures in-process via the existing `zip` dev usage and exercise the public reader/mmap paths directly: NPY `header_len = u32::MAX` (+ boundary), oversized declared shape, oversized `BINBYTES`/`BINUNICODE` end-to-end through `parse_pth` / `inspect_pth_from_reader`, an `enforce_pkl_size_cap` boundary test, and an `#[ignore]`d >100 MiB `data.pkl` mmap PoC. No checked-in binaries, no Python sidecar. — **commit `9975964`**

**No new runtime dependencies.** Three new module-level constants (`NPY_MAX_HEADER_BYTES`, `NPZ_MAX_ARRAY_BYTES`, `MAX_PICKLE_PAYLOAD`) plus two private guard helpers (`enforce_pkl_size_cap`, `enforce_pickle_payload_cap`); zero new types or public API. The `MAX_PKL_SIZE` constant already existed; Step 1 just applies it in an additional code path via the shared helper.

**Out of scope (deferred or never):**

- **CVE assignment.** anamnesis hasn't shipped a publicly-known exploitable issue under its own name; the candle #3533 disclosure is upstream. Internal hardening before public Python exposure ([Phase 8's](#phase-8-python-bindings-pyo3) `pip install anamnesis`) is the right framing — there's no CVE to assign because there's no known in-the-wild attack. If a downstream consumer reports active exploitation against a pre-v0.6.1 deployment, revisit.
- **Audit boundary at third-party crates.** `safetensors 0.4`, `zip 2.4` (`default-features = false`, `features = ["deflate"]`), and `serde_json 1` are the runtime parsing dependencies. ZIP-format safety remains the `zip` crate's responsibility; anamnesis pre-validates `data_start + data_len ≤ raw.len()` before indexing into a `zip`-exposed entry, which is the right defensive shape downstream of a trusted crate. JSON-header safety remains `serde_json`'s responsibility; the upstream 100 MiB cap bounds nesting / array depth implicitly. Re-auditing these crates' internals is out of scope for v0.6.1.
- **General `cargo fuzz` harness.** A fuzz campaign over each parser would catch more bug classes than the targeted regression tests in Step 5, but is a larger investment and belongs in a dedicated phase, not v0.6.1. Earmark for a future Phase 6.7 if downstream demand surfaces.

**Deliverable:** `anamnesis` v0.6.1 — patch-level security hardening triggered by the candle #3533 audit. Two unguarded-allocation findings closed (Steps 1–2), two defence-in-depth caps optionally added (Steps 3–4), regression tests pinning each fix (Step 5). No user-facing behaviour change for legitimate files; no API breakage. — **PUSH + tag `v0.6.1`**

### Phase 6.7: Security Audit Hardening + Fuzzing

**Goal:** Before exposing anamnesis to the Python ecosystem in [Phase 8](#phase-8-python-bindings-pyo3) — which multiplies the audience ~100× and adds an FFI trust boundary — run a full security audit of every parser and transform layer, close what it surfaces, and stand up a `cargo fuzz` harness for ongoing coverage. Patch-level release (`v0.6.2`): no new public API, no behaviour change for legitimate files.

**Origin — `candle #3556` and the pre-Phase-7 audit:** Re-reading the fix PR for [`candle #3533`](https://github.com/huggingface/candle/issues/3533), [`PR #3556`](https://github.com/huggingface/candle/pull/3556), showed its substantive addition beyond constant caps is a **remaining-bytes check** — capture the stream length once, reject any declared length exceeding the bytes physically left. anamnesis's `GGUF` reader already had this (`read_into` threads `file_len`); the audit asked where the *rest* of the codebase stands against the same bug class (CWE-770 / CWE-1284 / CWE-400) ahead of `pip install anamnesis`.

**Audit method (recorded because it shaped the result):** three parallel read-only agents for breadth, then **solo verification of every claim against the source**. The agents were a useful candidate generator but unreliable as arbiters — two of three "HIGH" findings were false positives that collapsed when the cited lines were read (`remember/bnb.rs` guards `block_size == 0` and `nested_block_size == 0`; `remember/fp8.rs` bounds its edge block via `.get_mut(...).ok_or_else`). The `remember/gguf.rs` coverage gap the agents left (only spot-checked) was closed by a solo read: that file is clean — its `validate_dequant_input` gate cross-checks `data.len() == n_blocks × type_size` (the very pattern PR #3556 champions) and all seven `IQ`-grid indices are bit-mask-bounded to their exact grid lengths (256 / 512 / 1024 / 2048). **Net: no high-severity / memory-unsafety bug; three verified low-to-medium findings, all hardening/consistency.**

**Sequencing — three fixes + harness, one patch version (`v0.6.2`):**

- [x] **Step 1 — Finding A (NPZ array bytes vs ZIP entry size):** `read_array_data` now takes the ZIP entry's declared uncompressed `size()` and rejects a shape-derived `data_bytes` exceeding it **before** `vec![0u8; data_bytes]`. Previously the only bound was the 8 GiB `NPZ_MAX_ARRAY_BYTES` cap, so a tiny entry declaring an enormous shape — or a `DEFLATE` entry whose declared shape would balloon the allocation — could drive a multi-GiB transient alloc / decompression before `read_exact` failed. This adopts the `data.len() == expected` cross-check the `GGUF` dequant path already performs. `PTH` is immune on the mmap path (the `build_entry_index` `Stored`-only filter) and bounded to 100 MiB on the reader path (`enforce_pkl_size_cap` before `read_to_end`). — **commit `798450a`**
- [x] **Step 2 — Finding B (GGUF `read_bytes` ordering):** factor the remaining-bytes check out of `read_into` into `ensure_remaining(n)` and call it at the top of `read_bytes`, before `vec![0u8; n]`. A tiny file declaring a string up to the type cap (`MAX_STRING_LEN` = 16 MiB) no longer allocates before the EOF check — the candle #3556 ordering. — **commit `c562027`**
- [x] **Step 3 — Finding C (unchecked `shape.iter().product()`):** added a shared `checked_num_elements(shape) -> Option<usize>` helper (`src/parse/utils.rs`, now always-compiled with `byteswap_inplace` kept `npz`/`pth`-gated) and routed the two production sites through it: `src/model.rs` `shape_to_rows_cols` (overflow → `Parse`) and `src/lethe/bnb_writer.rs` `is_eligible_for_nf4` (overflow → not eligible). Verified clippy-clean on no-features, `bnb`-only, and all-features. — **commit `5ea4c17`**
- [x] **Step 4 — CONVENTIONS invariant:** documented **"validate a declared size against the container's stated size"** under "When Parsing Untrusted Input" — a third defence distinct from constant caps and checked arithmetic, with `GGUF`'s `data.len() == n_blocks × type_size` / `ensure_remaining` and NPZ's `entry.size()` check as exemplars, plus the shared-helper rule. — **commit `c1016e4`**
- [x] **Step 5 — `cargo fuzz` harness:** a `fuzz/` cargo-fuzz workspace member (excluded from the published crate via `exclude = ["/fuzz"]`; nightly + libFuzzer, not in the stable CI matrix) with six targets: four reader/inspect (`fuzz_safetensors` / `fuzz_npz` / `fuzz_gguf` / `fuzz_pth` — headers + the pickle VM) plus two path/parse (`fuzz_npz_parse` / `fuzz_pth_parse` — the data-extraction paths, materialising the input to a temp file so `read_array_data` + the `entry.size()` guard and `parse_pth` tensor extraction are reached). libFuzzer doesn't build on Windows-MSVC, so it was **authored and verified under WSL2 Ubuntu** (nightly `rustc` + `cargo-fuzz` 0.13): all six compile; smoke campaigns ran ~20 s unseeded per reader target (~0.2–0.8 M runs) plus a 62 s seeded `fuzz_pth` run (213 k runs, cov 1412 / ft 5372, corpus 675, RSS steady ~489 MB) and 30 s seeded runs of the two path targets (102 k / 103 k runs) — **zero crashes** across all six. The split valve was not needed. A scheduled Linux fuzz CI job is a candidate follow-up. — **commits `b7499ea`, `5d0aca0`**
- [x] **Step 6 — Regression tests + docs:** in-module malicious-fixture tests — NPZ over-declared-shape-vs-`entry.size()` (landed with Step 1), `GGUF` oversize-string reject-before-alloc (declared 1 MiB < cap but > remaining bytes → `ensure_remaining` rejects pre-alloc), and `checked_num_elements` overflow (Step 3) — plus the `CHANGELOG` `### Security` entry and this section checked off. — **commit** — **PUSH + tag `v0.6.2`**

**Verified safe (no change required):** `remember/bnb.rs` divisor guards; `remember/fp8.rs` edge index; `remember/` output allocations (`checked_mul(2)`); the whole of `remember/gguf.rs` (validation gate + mask-bounded `IQ` grids); the `safetensors` slice/reader header caps from Phase 4.8 / 6.6.

**Out of scope (deferred or never):**

- **NPZ compression-ratio cap / lowering the 8 GiB budget.** The `entry.size()` cross-check (Step 1) is the precise fix; a ratio cap is extra defence-in-depth to revisit if a concrete need surfaces.
- **`type_size` single-source-of-truth in `remember/gguf.rs`.** The per-kernel hard-coded `type_size` duplicates `dtype.type_size()`; test-covered and not attacker-controllable, so informational only.
- **CLI `convert` file re-read inside the tensor loop.** A performance inefficiency, not a memory-safety issue.

**Deliverable:** `anamnesis` v0.6.2 — pre-Phase-8 security-audit hardening. Three audit findings closed (NPZ entry-size cross-check, `GGUF` reject-before-allocate, checked shape-product), the cross-check discipline codified in `CONVENTIONS.md`, and a `cargo fuzz` harness for ongoing coverage. No user-facing behaviour change for legitimate files; no API breakage. — **PUSH + tag `v0.6.2`**

### Phase 6.8: Untrusted-Input Resource Governance

**Goal:** Give callers a way to bound parser resource use to *their* environment, before [Phase 8](#phase-8-python-bindings-pyo3) multiplies the audience ~100× and makes "parse an untrusted, user-uploaded file" the normal operating mode. Phases 6.6–6.7 closed the *unbounded* DoS class; the fixed caps that remain are tuned for server / GPU hosts and still exceed what a memory-constrained edge device (a drone companion computer, an MCU-class board) or a per-worker slot in a multi-tenant model-serving / fine-tuning backend can absorb. This phase makes the limits *caller-configurable* and adds the two governance pieces a multi-tenant host needs — an aggregate budget and a decompression-ratio cap — plus documents the parse-first policy-gate pattern. Library-side and language-agnostic (it benefits every consumer, not only the Python bindings); ships as `v0.6.3` and lands **before** Phase 8 so the bindings expose an already-governed parser.

**Threat model — why fixed caps are not enough for these hosts:** the per-item caps (`NPZ_MAX_ARRAY_BYTES` 8 GiB, `MAX_PKL_SIZE` 100 MiB, `MAX_STRING_LEN` 16 MiB, the GGUF `MAX_*` family) are (a) **per-item, not aggregate** — a file declaring 50 000 tensors each just under a cap blows up in total while every individual check passes; (b) **fixed at server scale** — a 2 GB worker or edge board `OOM`s on a file that is well under the 8 GiB array cap; and (c) **silent on amplification** — the `entry.size()` cross-check (6.7) bounds a `DEFLATE` array to its *honestly declared* uncompressed size, but a few-KB compressed entry can honestly declare gigabytes. On a server an `OOM` kills one worker (DoS for co-tenants; cheap amplification → cloud-bill / outage); on an edge host it is a dead process (with `panic = "abort"`, any panic aborts instantly). Both want to *reject early with a clean `Err`* against a budget they choose.

**Sequencing — three governance pieces + a dependency-hygiene change + docs, one patch version (`v0.6.3`):**

- [x] **Step 1 — `ParseLimits` (caller-supplied budget):** a `ParseLimits` value (max single allocation, max *aggregate* declared bytes, max total tensor / array / KV count) threaded through every parser entry point and enforced fail-fast with `AnamnesisError::Parse` *before* allocation. The existing fixed caps become the permissive defaults (`ParseLimits::default()` == today's behaviour → no breaking change); callers tighten them to their environment (a drone sets MB-scale ceilings; an MLaaS worker sets per-slot-RAM ceilings). Subsumes the edge-deployment idea formerly in Future Directions — edge and MLaaS are the same feature at different numbers. — **commit**
- [x] **Step 2 — Aggregate accounting:** track a running total of declared bytes / counts across the whole file and reject when it crosses the `ParseLimits` aggregate budget, closing the many-small-items blow-up the per-item caps miss. — **commit**
- [x] **Step 3 — Decompression-ratio cap (`NPZ` / zip `DEFLATE`):** reject an entry whose uncompressed-to-compressed ratio exceeds a configurable bound, killing the cheap-amplification economics of a zip bomb. Promoted from the 6.7 "review-again" list — the multi-tenant track is the concrete need. Implemented through `zip`'s **public** API — no container rewrite: `ZipFile::size()` / `compressed_size()` give the declared ratio *before* allocation (the honest-but-huge bomb), checked in the `parse_npz` loop via the new `ParseLimits::max_decompression_ratio` axis. The `entry.take(cap)` read-bound the ROADMAP also listed proved **redundant for NPZ** — `read_array_data` already `read_exact`s exactly `data_bytes` (≤ `entry.size()`), so actual inflation is bounded by the existing read; the metadata ratio check is the operative defense. — **commit**
- [x] **Step 4 — Inflate-only `zip` (drop the unused DEFLATE compressor):** switch the `zip` dependency from the umbrella `deflate` feature to `["deflate-flate2", "flate2"]` + a direct `flate2 = { default-features = false, features = ["rust_backend"] }`, and add `dep:flate2` to the `npz`/`pth` features. This removes `zopfli` (+ `bumpalo`, `log`) — a DEFLATE *compressor* this read-only consumer never runs — from the tree while keeping `miniz_oxide` inflate. The `["flate2"]` zip feature is load-bearing: `deflate-flate2` enables zip's inflate *code* but not zip's optional `flate2` *dependency*; the implicit `flate2` feature flips the dep and the direct `rust_backend` dep supplies the backend via unification. Measured during planning: the three crates leave the `npz,pth` tree and a `numpy.savez_compressed` fixture still inflates bit-exactly. Pure dependency hygiene; `CHANGELOG` `Changed`. — **commit**
- [x] **Step 5 — Document the inspect-before-parse policy gate + tests:** the parse-first payoff — the bounded, header-only `inspect_*_from_reader` paths (already shipped) report total declared bytes / tensor count, so a host should `inspect` → check against policy → only then `parse`. Make this the documented, front-of-README pattern, ensure each inspect path surfaces the totals a gate needs, and add regression + `cargo fuzz` coverage for the `ParseLimits`, aggregate, and ratio paths. — **commit**
- [x] **Step 6 — `remember_to_bytes` (companion deliverable, *non-security* `Added`):** an in-memory twin of `ParsedModel::remember` — `ParsedModel::remember_to_bytes(target) -> Result<Vec<u8>>` returns the dequantised `BF16` safetensors as bytes instead of writing a file, so an embedder (e.g. candle-mi's quantized loader → `from_buffered_safetensors`) dequantises without a disk round-trip. Shares all dequant + companion-grouping with `remember` (factor the `views`-building into a shared helper: `remember` → `serialize_to_file`, `remember_to_bytes` → `serialize`). Completes the file/bytes pairing every other serializer already has (`npz_to_safetensors[_bytes]`, `pth_to_safetensors[_bytes]`, `write_bnb_nf4_safetensors[_bytes]`). Build it **first** in the phase so candle-mi isn't gated on the governance work; it rides in `v0.6.3` rather than a one-method release. Peak heap is the eager `O(2·n_params)` full model — the streaming-peak variants are [Phase 10](#phase-10-streaming-output). — **commit** — **PUSH + tag `v0.6.3`**

**Naming — the `remember_to_<destination>` family:** `remember` recovers precision and emits to a destination; each variant names the destination, so the set stays uniform and extensible. `remember_to_bytes` (this phase) and the streaming `remember_to_{writer,sink}` ([Phase 10](#phase-10-streaming-output)) are the *same operation at different peak-memory profiles*.

| Variant | Destination | Phase |
|---|---|---|
| `remember(path, target)` | file (safetensors) | shipped |
| `remember_to_bytes(target)` | eager in-memory `Vec<u8>` (safetensors) | **6.8** |
| `remember_to_writer(w: impl Write, target)` | streamed safetensors to any writer | [10](#phase-10-streaming-output) |
| `remember_to_sink(target, FnMut(&str, &[usize], &[u8]))` | per-tensor, no framing (embedder assembles) — the `RememberSink` | [10](#phase-10-streaming-output) |

`ParsedPth::to_safetensors_bytes` stays as-is — a different operation (lossless passthrough, not dequant), so a different verb is correct.

**Decision coupled to this phase — vendor the ZIP container? (evaluated → deferred).** anamnesis delegates ZIP-container parsing (`.npz`, `.pth`) to the `zip` crate. The two motivations for vendoring a minimal reader were **measured during 6.8 planning and both dissolved**:

1. *"`zip`'s `deflate` feature also pulls a DEFLATE compressor (`zopfli`) a read-only consumer never uses"* — true, but removed by a **feature change, not a rewrite**. `zopfli` arrives via the `deflate-zopfli` *sub-*feature; selecting `["deflate-flate2", "flate2"]` + a direct `flate2/rust_backend` drops `zopfli` (+ `bumpalo`, `log`) from the tree while keeping `miniz_oxide` inflate. This is now **Step 4 above** (measured: the three crates leave the `npz,pth` tree and a `numpy.savez_compressed` fixture still inflates bit-exactly).
2. *"6.8's ratio cap wants to live inside the decompress loop, which the `zip` API makes awkward"* — **false**. `ZipFile::size()` / `compressed_size()` expose the declared expansion ratio *before* allocation (rejecting the honest-but-huge bomb at the metadata gate), and `ZipFile: Read` lets `entry.take(cap).read_to_end()` bound the *actual* inflated bytes (rejecting the lying-header bomb) — both through the **public** API, proven against `zip 2.4.2`. Step 3's ratio cap therefore needs no vendoring.

With both triggers gone, what a vendored *reader* would still remove is only `zip`'s **non-codec** parsing (`indexmap`, `memchr`, `crc32fast`, `displaydoc`); the `unsafe`-heavy, *fuzzed* surface — `miniz_oxide` inflate — **stays regardless** (NPZ reads DEFLATE; `np.savez_compressed`; **never hand-roll the codec**). Owning the container is therefore **not a security win** on these grounds — a `cargo geiger` line-delta was not taken because the dependency tree already shows the codec is retained, and replacing a mature, fuzzed parser with hand-rolled code would *raise* risk. **Deferred to Future Directions**, gated on an actual `cargo geiger` measurement and, if ever pursued, scoped **PTH-first** (STORED-only — `build_entry_index` already skips non-`Stored` — so **zero codec**, the clean pilot) **with a `fuzz_zip` target from day one** — never folded into a security tag, since adding hand-rolled, less-battle-tested parser surface inside a security release works against the release's own goal.

**Reopened by measurement (Phase 6.8 Steps 1–3 audit, 2026-06-08).** The deferral above weighed only `unsafe`/codec surface and missed a **resource-governance** axis — which is precisely Phase 6.8's mandate. A `dhat` measurement during the Steps 1–3 consistency audit found that `zip::ZipArchive::new` eagerly materialises the **whole central directory** into a fat per-entry `ZipFileData` (~25 fields, the filename stored 2–3× across `file_name` / `file_name_raw` / the internal name map): a **4.4 MB / 50 000-tiny-entry archive → ~25 MB peak (~5.7×, ~500 B/entry)**, versus the **~40 B/entry** anamnesis actually needs (`build_entry_index`'s `name → (offset, size)`). This amplification is bounded (entries are file-grounded) but is *not* reachable through `zip`'s API and *not* bounded by `ParseLimits`; only a lean vendored central-directory reader bounds it (~12× metadata reduction, measured). This is the concrete, measured trigger the original deferral lacked → **planned as Phase 6.12 (Vendored ZIP container reader), after the `v0.6.6` pickle-VM patch, before Phase 8**, PTH-first + `fuzz_zip`, NPZ keeping `miniz_oxide` for inflate.

**Complement, not duplication:** bounding *peak* heap regardless of model size is [Phase 10 (Streaming Output)](#phase-10-streaming-output)'s job (stream blocks to disk instead of materialising); 6.8 bounds what a parser will *accept* before it starts, Phase 10 bounds what it *holds* while running. Together they let an untrusted-input host reject hostile declarations early **and** cap honest-but-large work.

**Out of scope:** model authenticity / signing (detecting the injected file in the first place) and process / resource isolation (`rlimit`, cgroup, sandboxed subprocess) are the caller's links of the defence-in-depth chain — anamnesis owns only defensive parsing and the clean `Err` the host fails safe on.

**Deliverable:** `anamnesis` v0.6.3 — the security core (caller-configurable `ParseLimits` per-item + aggregate, a decompression-ratio cap, the documented inspect-before-parse policy gate) **plus** the `remember_to_bytes` in-memory dequant companion (a non-security `Added` that rides in the same tag so we don't ship a one-method release) **plus** the inflate-only `zip` dependency tightening (drops the unused `zopfli` compressor — a non-security `Changed`). Default behaviour unchanged; no breaking change. The `CHANGELOG` carries a `### Security` section (the governance work), an `### Added` section (`remember_to_bytes`), and a `### Changed` section (inflate-only `zip`), keeping the security narrative honest. Lands before Phase 8 so `pip install anamnesis` exposes an already-governed parser. — **PUSH + tag `v0.6.3`**

### Phase 6.9: Quantized Dequant Bit-Exactness (Dogfooding Fixes)

**Goal (inserted 2026-06-10, stop-everything insertion ahead of the planned pickle-VM patch).** Fix the exact-parity defects surfaced by dogfooding `v0.6.3` inside candle-mi (full report: `docs/dogfooding-feedbacks/bnb-nibble-order-and-circular-fixture-validation.md` (internal dogfooding report, not tracked in the repo)), and fix the **methodology hole that let them ship green**: the BnB/AWQ/GPTQ cross-validation fixtures were generated by hand-rolled PyTorch reimplementations that shared the Rust code's blind spots — circular validation dressed as external. Re-anchoring every fixture on the canonical library's own code surfaced **three real correctness bugs** (the report predicted one confirmed + two suspects):

1. **BnB NF4/FP4 nibble order swapped** (decode + encode): anamnesis was low-nibble-first; bitsandbytes is high-nibble-first → element-permuted weights (confirmed end-to-end by candle-mi's forward-parity gate).
2. **BnB double-quant `nested_offset` dropped** (decode + encode + parse): bitsandbytes recovers `absmax = nested_dequant(...) + offset` (offset = mean of absmax, stored in the `quant_state` JSON blob); anamnesis ignored it → every recovered absmax biased low by 0.05–0.08 on real unsloth/medmekk DQ checkpoints. Found by this phase's re-anchoring, not by the original report.
3. **AWQ GEMM interleave missing** (decode): AutoAWQ packs nibbles in the order `[0, 2, 4, 6, 1, 3, 5, 7]` (both `qweight` and `qzeros`); anamnesis unpacked sequentially → column-permuted output (44 468 / 65 536 fixture elements wrong). 8-bit AWQ is now rejected as `Unsupported` — AutoAWQ's GEMM format is 4-bit only, so no canonical interleave exists to anchor against.

**GPTQ was confirmed correct** (sequential LSB-first + the `+1` zero-point offset matches GPTQModel's loader-side v1→v2 conversion + dequant) — but its fixtures were re-anchored anyway, because a true negative still needs a genuine external anchor to stay true.

**Methodology rule adopted (the real fix):** a dequant fixture's "expected" **must** come from the canonical library's own code — `bitsandbytes.functional.dequantize_4bit` / `int8_vectorwise_dequant` (CUDA kernel at generation time; fixtures stay CPU-checkable), AutoAWQ `unpack_awq` + `reverse_awq_order`, GPTQModel `TorchLinear.dequantize_weight` + `convert_gptq_v1_to_v2_format_module` — never a reimplementation of the formula. All BnB/AWQ/GPTQ cross-validation now passes at **0 ULP** (tightened from 1), with red-before/green-after proven against the unfixed code for both bug families.

- [x] Re-anchor `generate_bnb.py` on real bitsandbytes (7 fixtures, new `nested_offset` header field); prove red (6/6 NF4-FP4 fixtures fail vs the old decode, INT8 passes — exactly as predicted). — **commit**
- [x] Fix BnB nibble order (decode core + double-quant + encoder mirror + unit tests) and double-quant `nested_offset` (new parameter on `dequantize_bnb4_double_quant_to_bf16` / `encode_bnb4_double_quant`, parsed from the `quant_state` blob in `model.rs`). **Breaking API change**, flagged in the CHANGELOG. — **commit**
- [x] Re-anchor `generate_awq.py` on AutoAWQ's `packing_utils`; prove red (68 % of elements wrong); fix the interleave in `remember::awq` (`AWQ_ORDER`/`AWQ_REVERSE_ORDER`); reject `bits != 4`. — **commit**
- [x] Re-anchor `generate_gptq.py` on GPTQModel's module pipeline; convention confirmed; tighten to 0 ULP. — **commit**
- [x] Audit every other format's fixture anchor (GGUF/Ollama: real `gguf-py`; NPZ/PTH/safetensors: the format's own library; FP8: torch's native fp8 cast) — all genuinely external, no action needed. — **commit** — **PUSH + tag `v0.6.4`**

**Deliverable:** `anamnesis` `v0.6.4` — BnB NF4/FP4 (plain + double-quant), BnB INT8, AWQ, and GPTQ dequant are bit-exact (0 ULP) against their canonical libraries' own output on real-model fixtures; the encode side reproduces `bitsandbytes`' on-disk bytes byte-exactly; the circular-fixture methodology hole is closed crate-wide. Confirmed downstream by candle-mi's `bnb_nf4_llama_forward_parity` gate going green. — **PUSH + tag `v0.6.4`**

### Phase 6.10: Standard-Orientation Output for AWQ/GPTQ (Dogfooding Fix #2)

**Goal (inserted 2026-06-10, the second stop-everything dogfooding insertion).** Fix the output-contract defect surfaced by the second candle-mi dogfooding report (`docs/dogfooding-feedbacks/awq-gptq-dequant-transpose-orientation.md`, internal, not tracked in the repo): `remember` / `remember_to_bytes` emitted AWQ and GPTQ projection weights in the **GEMM-native `[in_features, out_features]`** orientation, but a standard `nn.Linear.weight` safetensors is **`[out_features, in_features]`** — so every 2-D projection weight came out transposed and no standard consumer (candle, `transformers` as a plain model) could load the output. The dequant **values** were already correct (Phase 6.9's 0-ULP anchoring); the **layout for standard consumption** was not, and the value-only cross-validation is structurally blind to it — the same class of gap as Phase 6.9's circular fixtures, one property up.

**The fix (validated against the canonical library):** GPTQModel's own `dequantize_model` applies `.T` to its GEMM-native `dequantize_weight()` when assigning to a plain `nn.Linear` — the transpose belongs at the output-contract boundary. The kernels keep their GEMM-native orientation (their 0-ULP anchoring and every Phase 6.9 fixture stay byte-for-byte untouched); `remember`'s AWQ/GPTQ arms transpose each quantized weight (cache-blocked `transpose_bf16` in `remember::quant_utils`) and emit `[out, in]`. Passthrough tensors (norms, biases, 2-D `embed_tokens` / `lm_head` kept in BF16) are untouched. Audit of every other format's emitted orientation: BnB4 (shape from the `quant_state` blob), BnB INT8, FP8 ×3, GGUF→safetensors (`shape.reverse()` at the CLI dispatch), and the NPZ/PTH/safetensors passthroughs are all already standard — only AWQ and GPTQ were wrong.

- [x] Output-orientation contract tests (`tests/remember_orientation.rs`): every quantized scheme loaded through the public `parse → remember_to_bytes → safetensors::deserialize` path on a synthetic **non-square** model, asserting emitted shape `[out, in]` **and** element mapping (`W_std[o][i] == W_native[i][o]`), plus passthrough byte-identity. Proven red against the unfixed code (AWQ/GPTQ emitted `[8, 16]` for an `in=8, out=16` projection), green after. GGUF↔safetensors reversal pinned with an absolute-orientation assert in the convert round-trip. — **commit**
- [x] `transpose_bf16` (tiled, `remember::quant_utils`) wired into the `remember` AWQ/GPTQ arms; kernel docs state the GEMM-native return layout explicitly; `# Memory` docs note the transient per-tensor transpose buffer. **Breaking behavior change** flagged in the CHANGELOG. — **commit** — **PUSH + tag `v0.6.5`**

**Deliverable:** `anamnesis` `v0.6.5` — AWQ/GPTQ `remember` output loads as a standard model (shape `[out, in]`, values unchanged); orientation is a tested contract for every scheme. Confirmed downstream by candle-mi's `validate_quantized_loading::{awq,gptq}_llama_forward_parity` gates going green. — **PUSH + tag `v0.6.5`**

### Phase 6.11: Pickle-VM Working-Set Governance

**Goal:** Close the resource-governance hole an **independent security audit surfaced in `v0.6.3`** (2026-06-09, re-verified against the code). The `.pth` pickle virtual machine (`src/parse/pth.rs`, `PickleVm::execute`) bounds only the `data.pkl` byte length (`MAX_PKL_SIZE` 100 MiB) and string/bytes payloads — the **value stack, memo clones, and nesting depth are charged to nothing** — so a small crafted pickle forces multi-GB heap or a recursive-`Drop` stack overflow → **DoS by process abort** (the crate builds `panic = "abort"`, so any of these aborts instantly). It is reachable from `parse_pth` **and** from `inspect_pth_from_reader` — the very path the README's inspect-before-parse recipe tells a host to run on untrusted input as the *cheap, safe* pre-filter — so the `v0.6.3` "untrusted-input resource governance" story has a real hole on its own advertised safe path. The finding directly refutes the in-code rationale at `parse_pth_with_limits` ("every value the VM builds is driven by an opcode in that bounded stream"): the 100 MiB cap bounds the opcode *stream*, not the heap each opcode produces. Ships as **`v0.6.6`**, before [Phase 8](#phase-8-python-bindings-pyo3), so `pip install anamnesis` exposes a VM that is actually governed. *(Originally slated for `v0.6.4`, then `v0.6.5`; the two stop-everything dogfooding fixes — [Phase 6.9](#phase-69-quantized-dequant-bit-exactness-dogfooding-fixes) and [Phase 6.10](#phase-610-standard-orientation-output-for-awqgptq-dogfooding-fix-2) — took those tags.)*

**Threat model — three confirmed amplification vectors** (byte-level constructions in the audit; `max_total_bytes` mitigates none today because no site routes through `charge_alloc`):

- **(a) Flat stack** — 100 MiB of `N` (`NONE`) opcodes → ~5.9 GB heap (`sizeof(PickleValue)` ≈ 56 B, dominated by the `Global { module, name }` variant; ~56× amplification). No per-push charge, no stack-depth cap.
- **(b) Memo replay** — `BINGET` / `LONG_BINGET` deep-`clone()` a memoised list and push, unbudgeted; a ~1 MB file forces multi-GB heap because clone size is decoupled from the 2-byte opcode. This is *the* specific reason the documented invariant fails.
- **(c) Nesting / `Drop`** — `BINPERSID` (`Q`) `Box`-wraps one layer per byte with no depth check; `PickleValue`'s derived `Drop` is recursive, so a ~0.5 MB `Q`-chain overflows the call stack on drop. `MAX_PICKLE_NESTING` only guards *post-execution* traversal, not `execute()` or `Drop`.

The same audit **confirmed the rest of the parser surface safe** — the `is_allowed_global` allowlist is airtight (the VM never invokes callables → no RCE), and the dequant kernels, `ParseLimits`/`Budget` ordering, the safetensors `tensor_data` `.get()` backstop, NPZ + zip container, and GGUF all hold. So this is **one well-scoped gap, not a systemic failure**.

**Sequencing — three governance pieces, then tests/fuzz + the audit's one non-security nit, one patch version (`v0.6.6`):**

Steps 1–3 were unified into a **single `O(1)`-per-opcode accounting choke point** (a parallel `SlotMeta` depth/size stack alongside the value stack): `push_value` is the only way the stack grows, so the permanent floors and the caller budget are enforced at one site. A per-push *deep* walk would have been an `O(n²)` CPU-DoS in the guard itself (a `BINPERSID` chain), so depth is tracked incrementally and the deep-size walk runs only on the memo-clone opcodes (bounded by early budget rejection).

- [x] **Step 1 — Always-on structural cap:** permanent `MAX_PICKLE_WORKING_SET` (512 MiB) **and** `MAX_PICKLE_VM_DEPTH` (256) floors enforced *inside* `execute()` via `push_value` / the depth metadata, bounding (a) and (c) even under `ParseLimits::default()`. — **commit**
- [x] **Step 2 — Wire the working set to `Budget`:** every value-push and every `BINPUT`/`LONG_BINPUT`/`MEMOIZE`/`BINGET`/`LONG_BINGET` clone charges through the existing `Budget::charge_alloc`, the deep size on the memo clones — bounding (b) and giving `max_total_bytes` real teeth over the VM. `enforce_pickle_payload_cap` keeps only the per-item `MAX_PICKLE_PAYLOAD` pre-allocation cap; the cumulative charge moved to the single choke point. — **commit**
- [x] **Step 3 — Nesting-depth cap during `execute()`:** `MAX_PICKLE_VM_DEPTH` rejected at construction, so an over-deep value never forms; recursive `Drop` and every recursive walk (`Clone`, deep-size, the post-execution extraction traversal) stay ≤ 256 frames. Iterative `Drop` proved unnecessary — the depth cap is the cleaner, more obviously-correct control. — **commit**
- [x] **Step 4 — Regression tests + fuzz + the BnB nit:** four new tests assert each construction (`N`-flood, memo-replay, over-deep `Q`-chain) and the permanent byte floor return a clean `Err` *quickly* (the `Q`-chain trips under `UNBOUNDED_LIMITS` in ~260 bytes); the 81 existing `G1`–`G25` + fixture tests stay green; a real WSL2 `fuzz_pth` campaign ran clean (RSS-limited, so an unbounded-heap regression would surface as a libFuzzer OOM); and the BnB odd-`block_size` guard landed across all decode/encode entry points with tests. — **commit** — **PUSH + tag `v0.6.6`**

**Fuzz toolchain — confirmed available, just run it (verified 2026-06-09 on this machine's WSL2).** Nightly `rustc` + `cargo-fuzz` 0.13.1 + a C linker + the `llvm-tools` / `rust-src` nightly components are all installed, and `cargo +nightly fuzz list` enumerates every target (incl. `fuzz_pth`, `fuzz_pth_parse`, `fuzz_pth_limits`, and the Phase 6.13 owned-bytes `fuzz_*_bytes` set). **Do not hesitate or re-provision** — from a WSL shell, `cd /mnt/c/Users/Eric\ JACOPIN/Documents/Code/Source/anamnesis` and run e.g. `cargo +nightly fuzz run fuzz_pth -- -max_total_time=600`; seed from `tests/fixtures/pth_reference/*.pth`. Only caveat: `llvm-symbolizer` is absent (optional — `sudo apt-get install llvm` for symbolized crash traces; libFuzzer still catches every crash without it). If WSL won't start, it's the BIOS `SVM` (AMD virtualization) setting, not the toolchain.

**Out of scope:** CPU-time DoS (the VM stays `O(opcodes)` — a wall-clock budget is the host's job), and the **Vendored ZIP container reader** — that is its own [Phase 6.12](#phase-612-vendored-zip-container-reader) (`v0.6.7`, the separate, measured `zip::ZipArchive::new` ~5.7× central-directory amplification track; see the *Reopened by measurement* note under Phase 6.8).

**Deliverable:** `anamnesis` `v0.6.6` — the pickle VM's working set (value stack, memo clones, nesting) is bounded by both a permanent structural floor **and** the caller's `ParseLimits`, closing the audit's P0 and making the inspect-before-parse pre-filter safe to run on untrusted input. The `CHANGELOG` carries a `### Security` section. No API change; default behaviour unchanged for honest files. Found by the independent-review process stood up after `v0.6.3` — a process worth repeating each security tag. — **PUSH + tag `v0.6.6`**

### Phase 6.12: Vendored ZIP Container Reader

**Goal:** Replace `zip::ZipArchive::new` (the `.pth` / `.npz` central-directory parser) with a lean, vendored, **read-only** central-directory reader. The trigger is a measured resource-governance gap — full analysis in the *Decision coupled to this phase — vendor the ZIP container?* and *Reopened by measurement* notes under [Phase 6.8](#phase-68-untrusted-input-resource-governance): `ZipArchive::new` eagerly materialises the **whole** central directory into a fat per-entry `ZipFileData`, **~5.7× the file (~500 B/entry)** for a many-tiny-entry archive versus the **~40 B/entry** anamnesis actually needs (`build_entry_index`'s `name → (offset, size)`). That amplification is bounded (entries are file-grounded) but is *not* reachable through `zip`'s API and *not* bounded by `ParseLimits` — only an owned reader bounds it. (Distinct from the [Phase 6.11](#phase-611-pickle-vm-working-set-governance) finding: that one is *unbounded* heap → DoS; this is a *bounded* ~5.7× constant-factor efficiency cost.) Ships as **`v0.6.7`**, after the `v0.6.6` pickle-VM patch ([Phase 6.11](#phase-611-pickle-vm-working-set-governance)), before [Phase 8](#phase-8-python-bindings-pyo3).

**Scope & guardrails (carried from the Phase 6.8 decision — do not re-litigate):**

- **PTH-first.** `.pth` entries are STORED-only (`build_entry_index` already skips non-`Stored`), so the pilot reader is **zero codec** — pure central-directory + local-header offset parsing, the cleanest surface to own.
- **Never hand-roll the codec.** NPZ keeps `miniz_oxide` for DEFLATE inflate (the `unsafe`-heavy, upstream-*fuzzed* surface **stays**); only `zip`'s **non-codec** container parsing (`indexmap` / `memchr` / `crc32fast` / `displaydoc`) is replaced. Owning the container is a *resource-governance* win, not a codec-safety one.
- **`fuzz_zip` from day one** — the dedicated fuzz target lands *with* the reader, not after.
- `zip` moves to `[dev-dependencies]` (every `ZipWriter` use is already `#[cfg(test)]`-only fixture writing).

**Sequencing — one patch version (`v0.6.7`):**

- [x] **Step 1 — Vendored `.pth` central-directory reader:** STORED-only, builds the `name → (offset, size)` index bounds-checked against the mmap, behind the existing `parse_pth` API. Ship `fuzz_zip` alongside. Delivered with **full ZIP64 support** (EOCD record + locator + `0x0001` extra field — no regression on >4 GiB / >65 535-entry checkpoints) and an in-crate **differential oracle** (vendored index vs the `zip` crate across STORED/DEFLATE/ZIP64/multi-entry + a 256-archive randomized sweep). — **commit**
- [x] **Step 2 — NPZ over the vendored reader,** keeping `miniz_oxide` inflate for DEFLATE entries (codec untouched). Adds a `Read + Seek` `ReaderSource` + a `BoundedReader` entry-window; `inspect_npz` / `parse_npz` and the `.pth` reader path (`inspect_pth_from_reader`) now share `read_central_directory`. DEFLATE entries inflate in the consumer via `flate2` (the vendored reader stays codec-free); a non-`STORED`/`DEFLATE` `.npy` entry fails fast with `Unsupported`. — **commit**
- [x] **Step 3 — Drop the `zip` runtime dependency** (moved to `[dev-dependencies]`; `npz`/`pth` now pull only `flate2`); measured the metadata reduction with `dhat` against a 50 000-tiny-entry fixture (`tests/peak_heap_zip_metadata.rs`). **Result: 337 → 41 B/entry resident, 8.07× (3.12× peak)** — the 41 B/entry hits the projected ~40 B/entry target; the ~12× projection assumed `zip` ≈ 500 B/entry, but it measures 337 on short entry names, so ~8× is the real ceiling and the index (a sorted, `shrink_to_fit`-trimmed `Vec<(Box<str>, usize, usize)>`) sits at it. — **commit** — **PUSH + tag `v0.6.7`**

**Deliverable:** `anamnesis` `v0.6.7` — `.pth` / `.npz` container parsing on a lean owned reader (metadata reduction **measured: 337 → 41 B/entry resident, 8.07×**), the `zip` crate off the runtime path, `miniz_oxide` retained for inflate, `fuzz_zip` shipped. A pre-release security self-review folded in two hardening fixes on the new reader: the container reader now honours the caller's `ParseLimits` (CWE-770 — entry-count + central-directory allocation checked fail-fast; mmap path reads the directory zero-copy), and `DEFLATE` inflation on the `.pth` reader path is bounded to the declared, capped `uncompressed_size` (CWE-409 zip-bomb guard). No public API or behaviour change for legitimate files. — **PUSH + tag `v0.6.7`**

### Phase 6.13: Python-Readiness Hardening

**Goal:** Make the Rust *core* honestly able to back the binding-level safety promises [Phase 8](#phase-8-python-bindings-pyo3) makes — *before* those promises are frozen into a published `pip` API. Phase 8's hardening bullet pledges "every parser error surfaces as a catchable Python exception, never a process `abort`", an inspect-before-parse policy gate, and `ParseLimits` exposure — but four gaps live **below** the binding, in anamnesis itself, where a wheel cannot route around them: **(1)** the always-on `parse()` is mmap-backed, and a faulting mmap raises **`SIGBUS` / `SIGSEGV` — an OS signal Python `try/except` cannot catch** — so the "never a dead worker" pledge is unbacked for the exact hostile-upload case Phase 8 targets; **(2)** `AnamnesisError` has just three variants (`Parse` / `Unsupported` / `Io`), collapsing every DoS-cap rejection *and* every pickle-allowlist rejection into one opaque `Parse` string, so a multi-tenant backend cannot map "payload too large (limit hit)" vs "malformed" vs "disallowed pickle global" to different HTTP responses or alert channels; **(3)** a PyO3 extension built with the `panic = "abort"` mode the parser docs assume converts every Rust panic into an **uncatchable process abort**, voiding the catchable-exception pledge at the *profile* level; and **(4)** the `NumPy` / `BF16` data-ownership contract is undecided, and the obvious zero-copy implementation — a `NumPy` array borrowing Rust-owned mmap memory — is a **use-after-free reachable from pure Python** once the owning `ParsedModel` drops. All four are core-level, language-agnostic, and **one-way doors** once a Python API depends on them. Ships as **`v0.6.8`**, after [Phase 6.12](#phase-612-vendored-zip-container-reader) and before [Phase 8](#phase-8-python-bindings-pyo3).

**Why a separate phase (not folded into Phase 8):** the same reason [Phase 6.8](#phase-68-untrusted-input-resource-governance) shipped `ParseLimits` as a library feature *before* the bindings — these are *library* guarantees that benefit every consumer (candle-mi, hf-fetch-model, any Rust caller), not Python-only glue. Freezing the Python exception hierarchy, the untrusted-input entry point, and the returned-array ownership model against a core that cannot yet honour them would bake the gaps into a published `pip` contract that is expensive to change post-`1.0`.

**Sequencing — four core hardenings, one patch version (`v0.6.8`):**

- [x] **Step 1 — Copy-based (`no-mmap`) full-parse path, made the recommended untrusted entry.** `CONVENTIONS.md`'s accepted-`unsafe` table records the gap verbatim: *"Full mmap-based parsing has no non-`unsafe` equivalent and is exempt."* The reader-generic *inspection* APIs (`parse_safetensors_header_from_reader`, `inspect_npz_from_reader`, `inspect_pth_from_reader`, and the GGUF counterpart from [Phase 4.9](#phase-49-reader-generic-gguf-inspection)) cover **headers only**; the tensor-**data** read for the three mmap formats (safetensors always-on, `.pth`, `.gguf` — NPZ is already copy-based and absent from the mmap table) still goes through `memmap2::Mmap::map`, whose documented invariant is a read-only artefact with no concurrent writer. A truncated upload, a network / FUSE filesystem, or a concurrent writer faults the mapping → **`SIGBUS`**, an abort no caller can catch. Add a `parse_*_from_reader` / `parse_bytes` family that reads the artefact fully into owned memory (bounded by the same `ParseLimits` budget, via the same `enforce_*_cap` helpers so the two paths cannot drift) and parses with **zero `unsafe` and zero mmap**; document it as *the* entry point for untrusted input, leaving the mmap `parse()` as the explicit trusted-local-file fast path. Update the accepted-`unsafe` table's "Non-`unsafe` parity where it exists" clause — full parse now *has* parity and is no longer exempt. — **commit**

- [x] **Step 2 — Finer-grained `AnamnesisError`, and the canonical Rust→Python exception map.** `AnamnesisError` is already `#[non_exhaustive]`, so new variants are **additive** (no breaking change for wildcard matchers). Add **`LimitExceeded`** — every `ParseLimits` rejection and every permanent structural floor (`MAX_PKL_SIZE`, `MAX_PICKLE_WORKING_SET`, `MAX_PICKLE_VM_DEPTH`, `NPY_MAX_HEADER_BYTES`, `MAX_STRING_LEN`, the GGUF `MAX_*` family, the vendored-zip entry-count / central-directory caps, the `DEFLATE` ratio cap, and the `max_total_bytes` aggregate) returns it, carrying the limit name and the offending-vs-allowed values — so a host can answer *413 Payload Too Large* distinctly from *400 Bad Request*. Reclassify the **pickle-allowlist** rejection (a non-`torch.*` `GLOBAL`) from `Unsupported` into a dedicated **`DisallowedGlobal`** security-signal variant a backend can log / alert on as a potentially hostile upload. Keep `Parse` (generic malformed), `Unsupported` (recognised-but-unimplemented format / feature), and `Io`. Document — in `lib.rs` and the README — the **frozen** Rust→Python mapping the Phase 8 binding will implement: base `AnamnesisError(Exception)` → `ParseError` ← `Parse`, `UnsupportedError` ← `Unsupported`, `LimitExceededError` ← `LimitExceeded`, `SecurityError` ← `DisallowedGlobal`, and `Io` → builtin `OSError`. Migrating the construction sites is mechanical; the `CHANGELOG` carries a `### Changed` note (the pickle-global reclassification is the one observable change, acceptable pre-`1.0` and pre-binding). — **commit**

- [x] **Step 3 — Panic / abort-freedom as a tested invariant + the `unwind` build requirement.** Two parts. **(a) Profile:** establish and CI-enforce that the Python extension `cdylib` builds with **`panic = "unwind"`**, never `panic = "abort"` — PyO3's panic boundary (Rust panic → Python `PanicException`) works *only* while panics unwind; under `abort` any panic is an uncatchable process kill that silently voids Phase 8's central pledge. *(As built: `[profile.release] panic = "abort"` is now a committed profile, and this step adds `[profile.python]` — `inherits = "release"`, `panic = "unwind"` — the profile maturin builds the Phase 8 `cdylib` with via `--profile python`; `tests/panic_profile.rs` asserts both so the contract cannot regress before the `cdylib` exists.)* **(b) Invariant:** extend the `cargo fuzz` harness (all 13 targets, every format — incl. the three owned-bytes `fuzz_*_bytes` targets) to assert the stronger property — *every input yields `Result::Err`, never a panic or abort* — promoting the existing "runs clean" campaigns into an explicit no-panic contract, and audit that the `#![deny(...)]` lint floor (`unwrap_used`, `expect_used`, `panic`, `indexing_slicing`) plus the `checked_*` arithmetic discipline (`CONVENTIONS.md` § *When Parsing Untrusted Input*) leaves no reachable panic on any public entry point, including debug-mode arithmetic-overflow panics. *Done:* `tests/no_panic.rs` (a stable-CI `catch_unwind` battery over adversarial inputs, run in debug so overflow panics are in scope) plus the three owned-bytes fuzz campaigns (≈75.9 M executions, zero crashes); `docs/python-interop.md` records the panic-safety + unwind contract. — **commit `1ddbb70`**

- [x] **Step 4 — Decide and lock the `NumPy` / `BF16` data-ownership contract.** Two coupled decisions Phase 8's `numpy` interop hard-depends on, settled here so the binding cannot ship an unsafe shape. **Ownership:** a returned array must either *own* its bytes (a copy) or borrow Rust memory through a lifetime-safe capsule that keeps the owner alive — never a bare view into an mmap the owning `ParsedModel` can drop, which is a use-after-free callable from Python. Step 1's copy-based path makes owned-bytes the natural default. **`BF16`:** `NumPy` has no native `bfloat16`, so fix the contract now — return `ml_dtypes.bfloat16` arrays when that (optional) Python dependency is present, else raw `bytes` + an explicit dtype string (`NpzTensor` already models a `BF16` that `NumPy` cannot — see `CONVENTIONS.md` field-docs example). Capture the decision in a short `docs/python-interop.md` design note so Phase 8 *implements* rather than re-litigates it. *Done:* locked **owned-copy by default** (zero-copy `PyCapsule` a documented future opt-in) and **`ml_dtypes.bfloat16` else `bytes` + dtype string** (never a silent upcast); recorded in `docs/python-interop.md`, pinned by `tests/python_ownership_contract.rs` (owned extraction outlives a dropped `Parsed*`) + FFI-boundary doc-notes on `GgufTensor` / `PthTensor`. No new core API — the contract documents and tests what already exists. — **commit** — **PUSH + tag `v0.6.8`**

**Out of scope:**

- **The wheel perf cliff — now resolved by the Phase 7/8 ordering, not here.** Every headline speedup (2.7–54× dequant, 17.7× NPZ) is measured with `target-cpu=native` → AVX2; the repo ships **no `.cargo/config.toml`**, so a PyPI wheel — which must target a fixed baseline — would otherwise lose all of it. This is a *performance / packaging* concern, not a *safety* one, so it lives outside this hardening phase. It is addressed by the now-preceding [Phase 7 (CPU SIMD Pass)](#phase-7-cpu-simd-pass) — a runtime AVX2/NEON dispatcher on the shared pass-2 writer — together with [Phase 8 (Python Bindings)](#phase-8-python-bindings-pyo3)'s wheel-baseline decision (ship a primary `x86-64-v3` AVX2 wheel + an SSE2 fallback). Both land before the first wheel, so the launch benchmark reflects the advertised numbers.
- **All PyO3 / `maturin` / wheel / GIL wiring** stays in [Phase 8](#phase-8-python-bindings-pyo3) — this phase ships no `python` feature and adds no dependency.

**Deliverable:** `anamnesis` `v0.6.8` — the core can honestly back Phase 8's safety promises: a copy-based, mmap-free, `ParseLimits`-bounded full-parse path that cannot `SIGBUS` on a hostile upload (and is documented as the recommended untrusted-input entry point); a finer `AnamnesisError` taxonomy (`LimitExceeded`, `DisallowedGlobal`) with a frozen Rust→Python exception map; a CI-enforced `panic = "unwind"` extension profile plus a fuzz-asserted no-panic invariant; and a locked `NumPy` / `BF16` ownership contract. Language-agnostic — every Rust consumer benefits — and no behaviour change for honest files on the existing mmap path. The `CHANGELOG` carries `### Added` (reader-path parse, error variants) and `### Changed` (pickle-global reclassification) sections. — **PUSH + tag `v0.6.8`**

**New dependencies:** None. (`ml_dtypes` is a Phase 8 *Python*-side optional, not a Rust dependency; `pyo3` / `numpy` still arrive in Phase 8.)

### Phase 6.14: Convert-Matrix Completion (BF16 Hub)

**Goal:** Make `amn convert` complete and honest for the three targets that ship today (`safetensors`/`bf16`, scalar `gguf`, `bnb-nf4`) by routing every conversion through the in-memory **BF16 hub** anamnesis already has the pieces for — `parse → remember` (to BF16, in memory) `→ forget`/`write` (to target) — instead of the per-cell handlers that reject uncovered pairs. v0.6.0 ([Phase 6](#phase-6-format-conversion-matrix)) wired the common cells but left genuine holes: the `bnb-nf4` target accepts **only** a plain-BF16 safetensors source (`npz` / `.pth` / `gguf` → `bnb-nf4` all return `Unsupported`), `gguf → gguf` is unimplemented, and a quantised input → an encode target fails with a "dequantise first" two-hop the user must stage by hand. **None of these need new kernels** — they are composition of shipped primitives — so closing them *before* [Phase 8](#phase-8-python-bindings-pyo3) exposes `convert()` to a ~100× audience makes the matrix complete and the docs honest at low risk. Ships as **`v0.6.9`**, after [Phase 6.13](#phase-613-python-readiness-hardening), before [Phase 7](#phase-7-cpu-simd-pass).

Also lands the **caller-supplied GGUF metadata pass-through** the convert story needs. `write_gguf` already takes a `metadata: &HashMap<String, GgufMetadataValue>` map, but `amn convert --to gguf` passes none — so the output is a tensor-container GGUF with no KV (no `Arch:` line). A `--gguf-metadata <file>` / repeatable `--gguf-kv key=value` flag wires that *existing* writer parameter through to the CLI: anamnesis stamps whatever KV it is handed and stays strictly format-level; the *model knowledge* (architecture hyperparameters, tokenizer) stays the caller's responsibility (see **Out of scope**).

**The matrix today (verified `v0.6.7`) and the cells this phase wires:**

| input → target | `safetensors`/`bf16` | scalar `gguf` | `bnb-nf4` |
|---|---|---|---|
| safetensors (plain BF16) | ✓ | ✓ | ✓ |
| safetensors (FP8/GPTQ/AWQ/BnB) | ✓ dequant | **wire** (auto-chain) | **wire** (auto-chain) |
| GGUF | ✓ dequant | **wire** (dequant-in-place) | **wire** |
| NPZ | ✓ | ✓ | **wire** |
| `.pth` | ✓ | ✓ | **wire** |

**Sequencing — one patch version (`v0.6.9`):**

- [x] **Step 1 — BF16-hub dispatch core:** refactor `convert` so any input that can reach BF16 (full-precision `.pth` / NPZ / plain safetensors, or a dequantised quantised input) routes `parse → remember`-in-memory `→ forget`/`write`(target), replacing the per-cell `Unsupported` handlers. The eager in-memory hub is `O(2× model)` peak (same profile as `remember_to_bytes`; the streaming variant is [Phase 10](#phase-10-streaming-output)). *Done:* promoted out of the CLI into a new **library** module `src/convert.rs` (always compiled, publicly re-exported) — the dispatch previously lived in `src/cli.rs` behind the `cli` feature, so no library consumer could convert and [Phase 8](#phase-8-python-bindings-pyo3) could not expose `convert()` without a later refactor; `cli.rs` shrank 1440 → 701 lines. Format detection moved with it (one source of truth for the CLI's parse/inspect/remember too). **Dtype policy: the hub is a BF16 *pivot*, not a BF16 *floor*** — quantised tensors dequantise to `BF16`, scalar tensors keep their original dtype, so `.pth → safetensors` and `NPZ`-`F32` → `GGUF` stay lossless; `BF16` is materialised for a scalar only where the target demands it. Verified behaviour-preserving: `convert --to safetensors` is **byte-identical** to the old `remember` path on the FP8 and `.pth` fixtures. — **commit `6d4c223`**
- [x] **Step 2 — Fill the `bnb-nf4` column:** `npz → bnb-nf4`, `.pth → bnb-nf4`, `gguf → bnb-nf4` via the hub plus the existing [Phase 5](#phase-5-quantization-lethe) BnB-NF4 encoder (2-D weights encoded; biases / norms / embeddings pass through as BF16 — the encoder's existing contract). *Done:* the column opened structurally with Step 1's generic dispatch (`npz → bnb-nf4` flipped from `Unsupported` to working, turning its negative test positive); this step pins **both halves** of the encoder contract — `gguf → bnb-nf4` on a 64-element (one-block) tensor asserts the NF4 companions, and `.pth → bnb-nf4` on the committed AlgZoo fixture (`[2,1]`/`[2,2]`/`[2,2]`, all below the 64-element block) asserts the **passthrough** path: written through as `BF16`, no companions. Plus a `CONVENTIONS.md` audit of the Step-1 code (`ConvertOptions` constructors made `const fn`). — **commit `4d280fd`**
- [x] **Step 3 — `gguf → gguf` (dequantise-in-place) + auto-chain the two-hop:** a quantised GGUF → scalar GGUF (recover precision, re-emit as a scalar GGUF), and quantised-safetensors → `gguf` / `bnb-nf4` chained through BF16 internally so the user no longer stages a temp file. The old "dequantise first" error becomes the actual behaviour. *Done:* both already **ran** after Step 1 (the per-cell `Unsupported` handlers went away with the refactor — verified: FP8 safetensors → `gguf` now reports its dequant step). The real defect was quieter: `gguf → gguf` **silently dropped the source KV** (the writer passed an empty metadata map), re-emitting a bare tensor container no runtime can load. The hub now carries the source KV (`Hub.gguf_metadata`), which Step 4 merges caller KV over. *Plan correction:* the plan claimed `ParsedGguf::metadata` was private and needed a new accessor — it was already `pub const fn metadata()`; the duplicate was caught by the compiler and removed, so **no new public API** was required. Tests assert a `String` **and** a typed `U32` key survive the round trip with types intact. — **commit `b348d13`**
- [x] **Step 4 — `--gguf-metadata <file>` / `--gguf-kv key=value` pass-through:** wire the existing `write_gguf` `metadata` parameter through the CLI; KV is written verbatim. A `.json` file or repeated `--gguf-kv` are both accepted; values typed per the GGUF metadata-value grammar. **No architecture awareness** — anamnesis writes what it is given. *Done:* precedence runs **inherited source KV → JSON file → `--gguf-kv`**, so `convert --to gguf` finally emits an `Arch:` line instead of a KV-less tensor container. Plain JSON values are inferred (string, bool, integer → `u32`, float → `f32`, array typed from its first element); an explicit `{"type": "i32", "value": …}` / `{"type": "array", "item_type": "i32", "value": […]}` pins an exact width. *Why the escape hatch earns its keep:* inference cannot be right for every key — `tokenizer.ggml.token_type` is `Array<I32>` by llama.cpp convention but a JSON array of non-negative integers infers `Array<U32>`, and special-casing that key would import exactly the model knowledge this layer refuses; a test asserts **both** halves. `--gguf-kv` is always `String`-valued (split on the first `=`). API: `ConvertOptions` gains `gguf_metadata` + `with_gguf_metadata`; `new()` / `with_limits` lose `const` because the `gguf` build carries a `HashMap` (compiler-decided, per the `CONVENTIONS.md` const rule). 6 grammar unit tests (inference, explicit forms, and every malformed case) + CLI tests for the three-way merge and a malformed `--gguf-kv`. — **commit `f3c0849`**

  A follow-up `CONVENTIONS.md` pass over Steps 1–4 caught a real defect: `ConvertStats.passthrough` double-counted the dequantised tensors against its own documented contract (*"tensors written in their incoming dtype"*), so the three counters now **partition** the written set, pinned by an invariant test. — **commit `5cf800e`**
- [x] **Step 5 — Cross-format round-trip tests + docs:** extend `tests/cross_validation_convert.rs` with the newly-wired cells (byte-exact where reversible); refresh the README convert section, the FAQ convert answer, and the planned `convert-between-formats` tutorial to document the completed matrix. *Done:* the docs sweep landed earlier (README `Formats & quantization support` convert column, FAQ convert answer, `docs/cli-reference.md` matrix ❌→✅ + footnotes — commit `a3cadf8`); this step adds **six `convert()`-level tests** (`t15`–`t20`) that pin the *dispatch* rather than the primitives `t1`–`t14` already cover: the quantised-safetensors→gguf auto-chain, `gguf→gguf` byte-exact-scalar + typed-KV survival (the Step 3 regression), caller-KV-over-source merge, and `gguf`/`npz`/`.pth`→`bnb-nf4` (the cells that returned `Unsupported` before Step 1). Plus a `docs/tutorials/convert-between-formats.md` walkthrough with **real captured output** (`RedHatAI/Llama-3.2-1B-Instruct-FP8` → `bnb-nf4`/`gguf`; `gguf→gguf` KV inheritance on `Qwen2.5-0.5B-Instruct-GGUF`), linked from the FAQ. The Phase 6 goal's "any input → BnB" wording was corrected retroactively. — **commit `6adaa27`** — **PUSH + tag `v0.6.9`**

**Out of scope:**

- **Quantised GGUF *target* columns (`gguf-q4km`, FP8, IQ, TQ, MXFP4).** These need the encode kernels from [Phase 8.5](#phase-85-lethe-encode-completion); this phase completes *input coverage* for the three *current* targets, not the quantised target columns. The full input×target grid closes when 8.5 lands.
- **Architecture-aware GGUF metadata auto-generation.** Synthesising `llama.context_length`, `attention.head_count_kv`, the multi-thousand-token `tokenizer.ggml.tokens` array, chat templates, and per-architecture hyperparameters cannot be derived from the tensors — it comes from the source model's `config.json` / `tokenizer.json` (which `convert` does not ingest) and tracks llama.cpp's evolving per-arch conventions. That is a **model-packaging layer** and belongs in a downstream crate (e.g. `hf-fetch-model`, which already reads `config.json` and could supply the KV that Step 4's flag then writes). Step 4 provides the *mechanism* (write caller KV); it deliberately does not provide the *model knowledge*.

**Deliverable:** `anamnesis` `v0.6.9` — `amn convert` routes every shipped (input × current-target) pair through the BF16 hub: the `bnb-nf4` target now accepts NPZ / `.pth` / GGUF / quantised-safetensors sources, `gguf → gguf` dequantises in place, and quantised inputs auto-chain through BF16 (no hand-staged temp file). Plus a `--gguf-metadata` / `--gguf-kv` pass-through so a caller can write the KV that makes a GGUF inference-loadable, while anamnesis stays format-level. No new kernels; the quantised GGUF target columns still land at Phase 8.5. The Phase 6 deliverable's "any input → BnB" wording is corrected to "BF16-safetensors → BnB" retroactively. — **PUSH + tag `v0.6.9`**

**New dependencies:** None. Reuses the [Phase 5](#phase-5-quantization-lethe) BnB encoder, the [Phase 6](#phase-6-format-conversion-matrix) `write_gguf` writer (including its existing `metadata` parameter), and the in-memory `remember` path.

### Phase 7: CPU SIMD Pass

> **⚠️ Measured update (2026-07, branch `phase-7-cpu-simd`, [`docs/perf-experiments.md`](docs/perf-experiments.md) Experiment 10).** A measurement-first Stage 0 **disproved this phase's core thesis.** A bit-exact, hand-written AVX2 `f32x8_to_bf16x8` — the shared-writer helper this phase is built around — measured **1.02× (native) / 1.04× (SSE2): null.** Root causes, all measured: the shared pass-2 writer is **bandwidth-bound** (~0.85× the single-core memcpy ceiling), so no ISA beats DRAM; the compiler **already** auto-vectorizes the FP8/GPTQ/AWQ arithmetic (confirmed via `cargo asm` index-0 disassembly); and the slowest kernels (IQ2/IQ3 lattice codebooks) are **gather-bound** (`vpgatherdd` is slow on Zen 3). **Conclusion: explicit SIMD is exhausted with evidence — no product `unsafe` ships** (per [`CLAUDE.md`](CLAUDE.md) "no measured win → no commit"). The AVX2 prototype is preserved in the `#[ignore]` harness [`tests/bench_pass2_adhoc.rs`](tests/bench_pass2_adhoc.rs) as reproducible proof. The **largest measured lever is multi-threading** (every kernel is single-threaded on a 16-core CPU; dequant is embarrassingly parallel), so **Phase 7 is re-scoped to multi-threaded dequant**, governed by the new [`CONVENTIONS.md`](CONVENTIONS.md) "When Parallelizing Work" section. The original SIMD plan below is retained as the record of what was planned and empirically tested.

**Goal (original — superseded by the measured finding above):** Replace the scalar pass-2 `f32 → BF16` loop in every dequantiser with an explicit, runtime-dispatched AVX2 / NEON vectorised version, giving a 4–8× speedup on the hot path without touching the correctness-verified scalar kernels. This is the first time anamnesis commits to explicit SIMD intrinsics — up to `v0.6.x` the project relied entirely on auto-vectorisation through the loop-fission + `chunks_exact_mut(2)` pattern. Runtime dispatch (`is_x86_feature_detected!`) is the load-bearing addition: it lets a *single distributed artefact* — most importantly a PyPI wheel, which cannot be built with `target-cpu=native` — select the AVX2 path at run time instead of baking one ISA in at compile time.

**Why this is a single phase and not per-format work:** Phases 1–4.5 all share the same `f32_bits_to_bf16_bits` pattern (defined once at [src/remember/fp8.rs:101](src/remember/fp8.rs#L101) and reused verbatim by `gptq.rs`, `awq.rs`, `bnb.rs`, and `gguf.rs`). A single `f32x8_to_bf16x8` SIMD helper retrofits every existing dequantiser at once, with one `unsafe` module, one `// SAFETY:` contract, one benchmark harness, and one cross-format round of cross-validation. Splitting the work per format would duplicate the intrinsic surface area across five modules and violate `CONVENTIONS.md`'s "unsafe lives in a single, dedicated module" rule.

The Phase 5 / 8.5 encode kernels write quantized bytes (not BF16), so they don't share this exact pass — but their absmax-then-round-and-pack inner loops are similarly auto-vectorisation friendly and could pick up SIMD intrinsics in a follow-up to this phase if the later encode-side benchmarks (Phase 6.5 + Phase 8.5 cross-validation results) demand it.

**Why now (moved ahead of the Python bindings):** the original plan deferred this until benchmark pressure from real users demanded it; the [Phase 8](#phase-8-python-bindings-pyo3) Python bindings invert that. A `pip install` exposes anamnesis to a ~100× audience that will benchmark it on day one, and a distributed wheel cannot use `target-cpu=native` — so without a runtime-dispatched SIMD path a default wheel ships the x86-64 SSE2 baseline and silently loses the AVX2 throughput every headline number (2.7–54× dequant, 17.7× NPZ) is measured at. Phase 8 ships the *packaging* half of the answer (an `x86-64-v3` AVX2 wheel + an SSE2 fallback — see Phase 8's wheel-baseline decision); this phase supplies the complementary *code* half — the runtime AVX2/NEON dispatcher on the shared pass-2 writer, which keeps even the SSE2 fallback wheel fast on capable CPUs and gives every Rust embedder the same single-source-of-truth SIMD helper regardless of how they set their own `target-cpu`. Settling it *before* the bindings means the launch benchmarks are measured against the final shipped code paths, not retrofitted after users report the gap. A `TODO(phase4-followup)` comment already sits on [`write_scratch_to_bf16`](src/remember/gguf.rs) flagging the intent.

**Approach:** Add a `simd` feature flag. Under that flag, introduce `src/remember/simd.rs` — a single dedicated module containing `#[target_feature(enable = "avx2")] unsafe fn f32x8_to_bf16x8(...)` and its NEON equivalent. The `write_scratch_to_bf16` helper (currently in `src/remember/gguf.rs` but logically shared across all dequantisers) moves to this module and becomes a runtime dispatcher: `is_x86_feature_detected!("avx2")` → AVX2 path, `std::arch::is_aarch64_feature_detected!("neon")` → NEON path, else fall through to the existing scalar loop. The scalar path stays the canonical correctness reference; the SIMD paths are tested bit-exactly against it via golden-vector cross-checks.

Per [`CONVENTIONS.md`](CONVENTIONS.md)'s accepted-unsafe table, all of the following must hold:

1. `unsafe` lives in a **single, dedicated module** (`src/remember/simd.rs`) — never scattered.
2. Every `unsafe` block carries a `// SAFETY:` comment documenting the invariants (lane count, alignment, target-feature preconditions).
3. The module is gated behind `#[cfg(feature = "simd")]` — users who don't enable the feature get `#![forbid(unsafe_code)]` unchanged.
4. The safe scalar fallback exists and is tested identically to the SIMD path.

> **The 8 checkboxes below are the SUPERSEDED SIMD plan** — retained as the tested record (measured null, see the banner and Experiment 10); **not pursued, no `unsafe` shipped.** The plan that actually shipped is "Phase 7 as shipped" after the deliverable.

- [ ] **`simd` feature flag + `remember::simd` module scaffold** — new `Cargo.toml` entry, `#[cfg(feature = "simd")] pub mod simd;` in `remember/mod.rs`, `cfg_attr(feature = "simd", allow(unsafe_code))` in `lib.rs`, and a new row in the `CONVENTIONS.md` accepted-unsafe table — **commit**
- [ ] **`f32x8_to_bf16x8` AVX2 intrinsic** — 8-wide round-to-nearest-even `f32 → BF16` lane-for-lane equivalent to the scalar `f32_bits_to_bf16_bits`. Bit-exact against the scalar reference (golden-vector test) — **commit**
- [ ] **`f32x4_to_bf16x4` NEON intrinsic** — ARM64 equivalent for Apple Silicon and AWS Graviton. Same bit-exactness requirement as the AVX2 path — **commit**
- [ ] **Runtime dispatch in `write_scratch_to_bf16`** — move the shared helper out of `src/remember/gguf.rs` into `src/remember/simd.rs` (or a new `remember::bf16_writer` shim). Retrofit FP8, GPTQ, AWQ, BnB, and GGUF pass-2 loops to call the shared dispatcher. Delete the per-kernel scalar copies — single source of truth — **commit**
- [ ] **Criterion bench harness** — benchmarks comparing scalar vs SIMD vs PyTorch CPU on the existing FP8/GPTQ/AWQ/BnB/GGUF cross-validation fixtures. Target: ≥ 4× speedup on `BF16`-writing pass 2 for tensors ≥ 1 MB. Publish the numbers in the crate README — **commit**
- [ ] **`copy_to_contiguous` flat-index pass-2 conversion** — replace the cross-iteration `coords[]` carry chain in [`src/parse/pth.rs::copy_to_contiguous`](src/parse/pth.rs) with the stateless `c_i = (flat_idx / stride_i) % shape_i` formula plus specialised fast paths for `ndim ∈ {1, 2, 3}`. Verify with `cargo-show-asm` that the loop emits SIMD strided gathers on AVX2/NEON. Path is rare (<0.1% of `state_dict` files) so this is a polish item, not a hot-path priority. Bit-exact unit tests already cover the function — **commit**
- [ ] **AWQ/GPTQ pass-2 zip-chain audit** — verify whether the four-way `chunks_exact_mut(2).zip(unpacked).zip(zeros).zip(scales)` pattern in [`src/remember/awq.rs`](src/remember/awq.rs) and [`src/remember/gptq.rs`](src/remember/gptq.rs) actually pessimises auto-vectorisation under stable LLVM. If `cargo-show-asm` shows scalar fallback, refactor to an indexed `for i in 0..len { … }` loop over pre-validated equal-length slices (per [`CONVENTIONS.md`](CONVENTIONS.md) "Reconciling bounds checking with vectorization"). Bit-exactness against PyTorch must hold at 0 ULP after refactor — **commit**
- [ ] **Cross-format correctness sweep** — re-run every Phase 1–4.5 cross-validation test with `--features simd` and assert bit-identical output against the scalar path. No format may regress — **commit** — **PUSH + tag `v0.7.0`**

**Deliverable:** `anamnesis` v0.7.0 — `BF16`-writing pass 2 runs 4–8× faster on AVX2 and NEON CPUs, scalar fallback preserved for `forbid(unsafe_code)` users, with cross-kernel consistency (FP8 / GPTQ / AWQ / BnB / GGUF all benefit from the same single-source-of-truth SIMD helper), and a runtime dispatcher that lets the [Phase 8](#phase-8-python-bindings-pyo3) wheels deliver AVX2 throughput without a `target-cpu=native` build. — **PUSH + tag `v0.7.0`**

**New dependencies:** None at runtime — uses `std::arch` intrinsics directly. Optional dev-dependency on `criterion` for the bench harness.

---

#### Phase 7 as shipped: multi-threaded per-tensor dequant

Replacing the null SIMD plan above with the **measured** lever (~3–4×, [`docs/perf-experiments.md`](docs/perf-experiments.md) Experiment 11). No `unsafe`, no new runtime dependency.

- [x] **Measurement-first investigation** — roofline + `cargo asm` audit + GGUF kernel sweep + a bit-exact hand-written AVX2 `f32x8_to_bf16x8` prototype proving explicit SIMD is null (1.02×/1.04×); harness `tests/bench_pass2_adhoc.rs`, Experiments 10–11 — **commit `ae619aa`**
- [x] **`CONVENTIONS.md` "When Parallelizing Work"** — determinism mandatory, disjoint outputs, caller-owned thread budget bounded by hardware (never file-declared counts), `// PARALLEL:` annotation — **commit `ae619aa`**
- [x] **`parallel` Cargo feature (default-ON) + `RememberOptions` / `with_threads`** + `remember_with_options` / `remember_to_bytes_with_options`; existing `remember` signatures unchanged; budget `None → min(available_parallelism, 4)`, `Some(n) → n.max(1)` — **commit `934bf81`**
- [x] **Parallel `ParsedModel::dequantize_all`** — per-tensor `std::thread::scope` over disjoint chunks, results sorted by header index → **byte-identical for any thread count**; determinism test across `{1, 2, 4}` (feature on and off); kernels untouched, bit-exact-vs-PyTorch suite green — **commit `934bf81`**
- [x] **`ConvertOptions.threads` / `with_threads`** — the safetensors input path parallelises for free (reuses the model dequant) — **commit `934bf81`**
- [ ] **GGUF-reader parallelisation** — *deferred follow-up:* `read_gguf`'s own per-tensor loop over mmap-borrowed `ParsedGguf` needs `ParsedGguf: Sync` + a loop restructure before it can join the scoped-thread pool
- [ ] **README / docs throughput note + release** — document the `parallel` feature and the `min(cores, 4)` default; then the release checklist — **PUSH + tag `v0.7.0`**

**Deliverable:** `anamnesis` v0.7.0 — whole-model `remember` / `convert` dequantise **~3–4× faster** on multi-core CPUs (DRAM-bandwidth-bound, so a modest `min(cores, 4)` default captures ~90 % of the win while leaving the host's cores free), **byte-identical output at any thread count**, opt-out via `default-features = false` for sequential/wasm builds. Explicit SIMD was measured and rejected — no product `unsafe`. — **PUSH + tag `v0.7.0`**

### Phase 8: Python Bindings (PyO3)

**Goal:** Expose anamnesis to the Python ecosystem via `pip install anamnesis`. Phases 1–7 give anamnesis the fastest dequantization (2.7–54× vs PyTorch), the fastest NPZ parser (17.7× vs npyz), PyTorch `.pth` parsing, GGUF support, BnB encoding, the `convert()` pipeline scaffold, and — from the immediately preceding [Phase 7](#phase-7-cpu-simd-pass) — a runtime-dispatched SIMD path so that speed survives wheel distribution. Python bindings multiply the audience by ~100×, replacing ad-hoc dequantization scripts across the ML community. By shipping after the format conversion scaffold, `pip install anamnesis` exposes `convert()` from day one; the remaining encode-side targets (GGUF / FP8 / IQ / TQ / MXFP4) light up at v0.8.5 / Phase 8.5 through the same Python API once those kernels land.

**Approach:** Use [PyO3](https://pyo3.rs/) + [maturin](https://github.com/PyO3/maturin) to build a native Python extension. The Python API should mirror the Rust library API closely: `parse()`, `inspect()`, `remember()`, `forget()`, `convert()`, `parse_npz()`. Returns NumPy arrays (via `numpy` interop) or raw bytes, following the ownership / `BF16` contract frozen in [Phase 6.13](#phase-613-python-readiness-hardening) Step 4. Ships as a wheel on PyPI.

- [ ] PyO3 module scaffold (`src/python.rs`) — feature-gated behind `python` — **commit**
- [ ] `parse_npz()` binding — returns `dict[str, NpzTensor]` with NumPy-compatible arrays — **commit**
- [ ] Safetensors `parse()` + `inspect()` bindings — **commit**
- [ ] `remember()` / `forget()` / `convert()` bindings — dequantize, quantize, and convert from Python — **commit**
- [ ] **Hardening for untrusted input (binding-specific) — wiring the [Phase 6.13](#phase-613-python-readiness-hardening) core into PyO3:** map each `AnamnesisError` variant to its Python exception (`ParseError` / `UnsupportedError` / `LimitExceededError` / `SecurityError` / `OSError`) per the 6.13-frozen hierarchy, so a multi-tenant backend distinguishes *413* (limit hit) from *400* (malformed) from a flagged hostile pickle — never a dead worker; route the binding's untrusted entry through 6.13's copy-based `parse_*_from_reader` (no mmap → no uncatchable `SIGBUS`) and build the `cdylib` with `panic = "unwind"` (6.13 Step 3); expose `inspect()` prominently as the recommended *first* call (the [Phase 6.8](#phase-68-untrusted-input-resource-governance) inspect-before-parse policy gate) and surface `ParseLimits` to Python callers; **release the GIL** during large `parse`/`remember`/`convert` so one tenant's parse does not stall others; add a binding-level fuzz/smoke test asserting *malformed bytes → Python exception, never a process exit*. — **commit**
- [ ] **Wheel baseline / multiversioning decision — `x86-64-v3` primary + SSE2 fallback:** a PyPI wheel cannot use `target-cpu=native`, so a default x86-64 wheel ships the SSE2 baseline and loses the AVX2 throughput the headline numbers are measured at. Build the **primary x86-64 wheel for the `x86-64-v3` micro-architecture level** (`-C target-cpu=x86-64-v3` → AVX2 / FMA / BMI via auto-vectorisation across *every* kernel, not just the Phase 7 pass-2 writer), and ship a **second `x86-64-v1` (SSE2) fallback wheel** for pre-2013 CPUs; document the support policy ("requires AVX2; SSE2 fallback wheel available") in the README and wheel metadata. The Phase 7 runtime-dispatched writer keeps the fallback wheel fast on the shared pass-2 hot path even on the v1 build. **ARM / aarch64** wheels need no special handling — NEON is baseline in aarch64, so auto-vectorisation (and Phase 7's NEON path) apply automatically. — **commit**
- [ ] `maturin` build config, PyPI packaging, CI workflow — **commit**
- [ ] Python test suite — validate against PyTorch reference on the same fixtures — **commit**
- [ ] **User docs — Python FAQ section + `pip`-to-NumPy tutorial (mirroring [hf-fetch-model](https://github.com/PCfVW/hf-fetch-model)'s `docs/` system):** add a **Python section** to `docs/FAQ.md` (install; *"why is the wheel slow on my old CPU?"* → the `x86-64-v3` + SSE2-fallback policy from the wheel-baseline step; which exception to catch, per the [Phase 6.13](#phase-613-python-readiness-hardening) `ParseError` / `LimitExceededError` / `SecurityError` hierarchy; how to get a `bf16` NumPy array via `ml_dtypes`) plus a `docs/tutorials/` walkthrough (`pip install` → `inspect()` → first dequantized NumPy array). Built on the `docs/python-interop.md` design note from [Phase 6.13](#phase-613-python-readiness-hardening) Step 4, and reusing the FAQ/tutorial conventions (`STYLE CONVENTIONS` block, freshness marker, 30-second answer, bash + PowerShell commands) the sibling hf-fetch-model docs already follow. *(The general `docs/FAQ.md` + `docs/tutorials/` scaffold for Rust/CLI users lands earlier as a non-versioned docs chore; this step adds the Python chapters.)* — **commit** — **PUSH**

**Deliverable:** `anamnesis` v0.8.0 — `pip install anamnesis` works, with the full conversion matrix exposed, an `x86-64-v3` wheel that delivers the advertised AVX2 throughput (SSE2 fallback wheel for older CPUs), and Python user docs (FAQ section + `pip`-to-NumPy tutorial). — **PUSH + tag `v0.8.0`**

**New dependencies:** `pyo3`, `numpy` (PyO3 interop). The optional Python-side `ml_dtypes` backs the `BF16` return contract from [Phase 6.13](#phase-613-python-readiness-hardening) Step 4. Feature-gated behind `python`.

### Phase 8.5: Lethe Encode Completion

**Goal:** Close the encode-side coverage gap left by Phase 5's narrow first cut. `v0.5.0` ships **BnB** encode (`NF4` / `FP4` / `INT8`) plus the round-trip validation harness; this phase extends that harness to every other codebook-style kernel family already supported on the decode side: **FP8** (Phase 1), **GGUF legacy block** + **K-quants** (Phase 4), and **GGUF IQ-quants** + **TQ** + **MXFP4** (Phase 4.5). After this release, anamnesis can encode into every block-quant format it can decode — closing the loop that the Phase 6 conversion matrix opened against the BnB-only encode side, and giving Phase 8's Python bindings a complete encode surface.

**Approach:** Mirror the Phase 4 / Phase 4.5 playbook on the encode side. Every kernel here is a codebook-LUT inversion (find the nearest entry in a fixed table) plus a per-block scale derivation (per-row absmax, per-block max, or sub-block stats). The encode loop body is structurally identical across families; only the codebook constants and the per-block scale formula differ — exactly the same pattern that justified collapsing six decode-kernel families into Phase 4.5.

Each kernel reuses the generic `assert_bit_exact_decode_encode` round-trip harness from Phase 5 (the codebook itself is the oracle for codebook-LUT kernels — no external Python reference needed for the round-trip). Each kernel ALSO ships a cross-validation test against the same Python reference used on the decode side: `gguf.quants.quantize()` for the GGUF families (legacy block, K-quants, IQ, TQ, MXFP4), and `torch.float8_e4m3fn` / `torch.float8_e5m2` casts for FP8. Fixtures are the existing `bartowski/...` GGUF slices and the seven FP8 model slices from Phase 1 — no new downloads.

**Why now (not earlier):** sequencing this *after* Phase 8 means the full `parse → remember → forget → convert → bind` chain has already been validated end-to-end through Python on a single quant family (BnB). Architectural mistakes in `lethe/` would have been caught by then. Phase 8.5 is then a focused encode-completion sweep against a stable architecture rather than co-evolving with one.

- [ ] **Step 1 — `FP8` encode (`E4M3` + `E5M2`, per-tensor / per-channel / fine-grained block schemes)** (`src/lethe/fp8.rs`) — three encode flows mirroring the v0.1.0 decode kernels: per-tensor (single scale), per-channel (one scale per row, `[N,1]`), and fine-grained 128×128 block scales. For each scheme, `scale = absmax / fp8_max` at the appropriate granularity, divide and round-to-nearest-even via `float8::F8E4M3::from_f32` (and `F8E5M2::from_f32` for the E5M2 variant). The round-trip harness asserts `decode(encode(x)) ≈ x` to within FP8 representation error (FP8 is lossy by construction); the *re-encode* path on the existing 7 FP8 fixtures DOES require bit-exactness — `encode(decode(q)) == q` once the scales stabilise. Cross-validation against `torch.float8_e4m3fn` cast on 256×256 slices from the existing `Llama-3.2-1B-Instruct-FP8`, `Qwen3-1.7B-FP8`, and `EXAONE-4.0-1.2B-FP8` fixtures. ~600 LOC, ~3 effort units — **commit**

- [ ] **Step 2 — GGUF legacy block encode (`Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q8_1`)** (`src/lethe/gguf_legacy.rs`) — six encode kernels for the 32-element legacy block family. Per-block absmax → `d` (block scale); divide and round each element to the nearest signed integer in the block-type's bit width; pack low/high nibbles for the 4-bit and 5-bit variants exactly inverting the decode-side unpack. `Q*_1` variants additionally derive `m` (block min) for asymmetric quantisation, mirroring the `(scale, min)` pair the decode side reads. Reuses the legacy-block runner that Phase 4 introduced (`run_legacy_kernel`) on the encode side. Cross-validation against `gguf.quants.quantize()` on the same `bartowski/SmolLM2-135M-Instruct-GGUF` slices used for decode-side validation. ~800 LOC, ~4 effort units — **commit**

- [ ] **Step 3 — GGUF K-quants encode (`Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_K`)** (`src/lethe/gguf_kquants.rs`) — six encode kernels for the 256-element super-block family. Per-block absmax → top-level `d`; per-sub-block `(scale, min)` derivation matching the decode-side `get_scale_min_k4` 6-bit packed-scale layout. `Q3_K` requires the inverse of the `kmask1`/`kmask2` permute (`q3_k_pack_scales` mirroring the existing `q3_k_unpack_scales`); a private `#[inline]` helper with its own unit tests, same pattern as the decode side. Reuses every constant table already ported from `ggml-common.h`. Cross-validation against `gguf.quants.quantize()` on the existing `bartowski/TinyLlama-1.1B-Chat` and `SmolLM2-135M-Instruct` slices. ~1200 LOC, ~5 effort units — **commit**

- [ ] **Step 4 — IQ4 + IQ2 family encode (`IQ4_NL`, `IQ4_XS`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`)** (`src/lethe/gguf_iq.rs`) — five codebook-LUT encode kernels. `IQ4_NL` / `IQ4_XS` walk the 16-entry `kvalues_iq4nl` codebook finding the nearest signed value per element, then pack into the existing low/high-nibble layout. `IQ2_XXS` / `IQ2_XS` / `IQ2_S` walk the 256 / 512 / 1024-entry signed-vector grids (`IQ2XXS_GRID`, `IQ2XS_GRID`, `IQ2S_GRID`) finding the nearest 8-element signed-vector entry per source chunk, then pack the index plus the matching `ksigns_iq2xs` sign mask into the block's `qs` / `qh` / `scales` fields. Lattice search is the new primitive; a single `nearest_signed_vector_index` helper covers all three IQ2 variants by varying grid size. Cross-validation against `gguf.quants.quantize()` on the existing `bartowski/Mistral-7B-Instruct-v0.3-GGUF` slices. ~1500 LOC, ~7 effort units — **commit**

- [ ] **Step 5 — IQ3 + IQ1 family encode (`IQ3_XXS`, `IQ3_S`, `IQ1_S`, `IQ1_M`)** (`src/lethe/gguf_iq.rs`) — four codebook-LUT encode kernels with two new wrinkles. `IQ3_XXS` / `IQ3_S` reuse the lattice-search helper from step 4 against the 256-entry `IQ3XXS_GRID` and 512-entry `IQ3S_GRID`; `IQ3_S` additionally inverts the unusual odd-integer scale formula `d × (1 + 2·nibble)` and the low/high nibble pairing across two consecutive sub-blocks. `IQ1_S` / `IQ1_M` use the 2048-entry `IQ1S_GRID` plus the `±IQ1S_DELTA = 0.125` *additive* bias — encode subtracts the bias before nearest-grid lookup, instead of IQ2/IQ3's multiplicative ±1 sign. `IQ1_M`'s scattered-`f16` super-scale layout (no top-level `d`) requires the inverse pack of the four-byte `scales[8]`-as-`f16` reconstruction. Cross-validation on the existing `bartowski/Mistral-7B-Instruct-v0.3-GGUF` slices (`IQ3_XXS` and `IQ3_S` ride the same shared file as decode-side step 3; `IQ1_S` / `IQ1_M` use their dedicated fixtures). ~1300 LOC, ~6 effort units — **commit**

- [ ] **Step 6 — TQ + MXFP4 encode (`TQ1_0`, `TQ2_0`, `MXFP4`)** (`src/lethe/gguf_tq.rs` + `src/lethe/mxfp4.rs`) — three encode kernels sharing the synthetic-fixture validation pattern the decode side established for the same kernels. **TQ:** per-block absmax → `d`; map each element to `{-1, 0, +1}` via threshold `d/2`; for `TQ1_0` re-pack five ternaries per byte using base-3 multiplication (inverse of the `pow3` decode trick); for `TQ2_0` re-pack four ternaries per byte using plain 2-bit packing. **MXFP4:** per-block absmax → `e: u8` (E8M0 byte exponent) via a new `f32_to_e8m0_half` helper (inverse of the existing `e8m0_to_fp32_half`); divide each element by the half-scale, find nearest entry in the 16-entry signed `K_VALUES_MXFP4` codebook, pack low/high nibbles. Reuses the `run_legacy_kernel` outer-loop runner. Synthetic-fixture cross-validation against `gguf.quants.quantize()` (Python supports both, mirrors Phase 4.5 step 5/6). ~600 LOC, ~3 effort units — **commit**

- [ ] **Step 7 — `forget()` dispatch + `forget` CLI subcommand + cumulative cross-validation** (`src/lib.rs` + `src/bin/main.rs`) — extend the `TargetKernel` enum and `ParsedModel::forget()` dispatch (introduced in Phase 5 for BnB) to recognise every kernel landed in steps 1–6, plus the `amn forget model.safetensors --to <kernel> -o out.gguf` CLI subcommand (alias `quantize`). The cross-validation suite is the cumulative roll-up of every per-step cross-validation test — bit-exactness contract holds at 0 ULP for the codebook-LUT kernels (steps 2–6) and within FP8 representation error for the FP8 path (step 1). Mirrors the v0.4.2 closeout step from Phase 4.5. ~300 LOC, ~2 effort units — **commit** — **PUSH + tag `v0.8.5`**

**Deliverable:** `anamnesis` v0.8.5 — Lethe encode completion. Every block-quant format anamnesis can decode out of can now also be encoded into. Combined with Phase 6's any-to-any conversion CLI and Phase 8's Python bindings, `pip install anamnesis` ships a complete read/write/convert matrix for the GGUF + FP8 + BnB universe — the first time any tool, Rust or Python, has covered all three families end-to-end. — **PUSH + tag `v0.8.5`**

**New dependencies:** None. Reuses the `lethe/` namespace (introduced in Phase 5), the `gguf` and `bnb` feature gates, `half`, and `float8`. Codebook constants from Phase 4.5 are reused verbatim — encode-side helpers live alongside them in the existing `iq_grids` submodule.

**Explicitly out of scope:**

- **GPTQ encoding** — requires a calibration dataset and OBQ/Hessian-based weight optimisation. Calibration-aware quantisation is a fundamentally different architecture (gradient computation, per-layer quant search) and belongs in its own dedicated phase, not in the codebook-LUT family of Phase 8.5.
- **AWQ encoding** — same reason as GPTQ; needs activation statistics from a calibration loop.
- **Big-endian GGUF v3 emit** — mirrors the Phase 4.5 deferral on the decode side. No committed target.

### Phase 9: Emerging Quantization Formats

**Goal:** Extend coverage to newer quantization formats that currently have zero Rust implementations. Prioritized by ecosystem adoption (HuggingFace model count as of March 2026).

**Landscape (March 2026):**

| Format | HF Models | Ecosystem | Rust Status |
|--------|-----------|-----------|-------------|
| EXL2 | ~4,775 | ExLlamaV2-only (GPU, hobbyist/local) | None |
| HQQ | ~190 | Transformers, vLLM (calibration-free) | mistral.rs only (inference-coupled) |
| AQLM | ~120 | Transformers, vLLM (academic, sub-2-bit) | None |

**Note:** EXL2's successor (EXL3) is emerging. HQQ and AQLM have strong technical merits but very low adoption. These formats are monitored; implementation is deferred until adoption grows or a concrete downstream need arises.

- [ ] EXL2 dequantization — mixed-bitrate (2–8 bpw), ExLlamaV2 binary format — **commit**
- [ ] HQQ dequantization — half-quadratic quantization, 1–8 bit, calibration-free — **commit**
- [ ] AQLM dequantization — additive quantization, sub-2-bit with vector codebooks — **commit**
- [ ] Cross-validation for each format — **commit** — **PUSH**

**Deliverable:** `anamnesis` v0.9.0 — emerging format coverage. — **PUSH + tag `v0.9.0`**

### Phase 10: Streaming Output

**Goal:** Drop peak heap from `O(total_dequantised_size)` to `O(largest_tensor_BF16)` for whole-model dequantisation paths (`ParsedModel::remember` and the `amn remember model.gguf -o out.safetensors` CLI). Unblocks 70 B+ model conversion on commodity hardware (≤ 32 GB) where the current eager-buffering approach OOMs. This phase also completes the `remember_to_<destination>` family begun in [Phase 6.8](#phase-68-untrusted-input-resource-governance): it adds the streaming `remember_to_writer` and the per-tensor `remember_to_sink` (`RememberSink`), so an embedder like candle-mi can load a dequantised model with peak ≈ one tensor — the streaming complement to 6.8's eager `remember_to_bytes`.

**Background:** The dequantisation kernels already provide streaming entry points — [`dequantize_gguf_blocks_to_bf16`](src/remember/gguf.rs) is `O(one block)` per call, and the `FP8`/`GPTQ`/`AWQ`/`BnB` kernels emit one tensor at a time. The orchestrators do not stream: [`ParsedModel::remember_bf16_inner`](src/model.rs) accumulates every dequantised tensor's `Vec<u8>` into a `dequantized_data: Vec<(name, bytes, shape)>` and then hands all `TensorView`s to `safetensors::serialize_to_file` simultaneously. The same pattern lives in [`src/bin/main.rs::run_remember_gguf`](src/bin/main.rs). The `safetensors` 0.4 crate's writer **already streams tensor bodies** (one `BufWriter::write_all` per tensor inside `serialize_to_file`) — the eager buffering is ours, not the crate's.

**Approach:** Implement a custom `View` impl whose `data()` lazily dequantises a single tensor on demand, paired with an iterator that yields `(name, LazyView)` derived from the parsed model. The crate's `prepare()` step only needs `data_len()` / `dtype()` / `shape()` for the header — all deterministic and cheap. Hand the iterator to `safetensors::serialize_to_file`; the writer pulls bytes per tensor and drops them after each `write_all`. Peak heap drops to `O(largest_tensor_BF16)` — for a 70 B model's biggest layer (~500 MB BF16) this fits on commodity hardware. True per-block O(1) streaming is **out of scope** for Phase 10 — it would require a custom safetensors writer (the `View` trait returns a single `Cow<[u8]>`, not a stream) and the per-tensor variant is sufficient for every model size in current circulation.

- [ ] **`LazyDequantView<'a>` type** in `src/remember/streaming.rs` — implements `safetensors::tensor::View`, captures a borrow into the parsed model + the per-tensor metadata needed to dispatch into the right kernel, and returns `Cow::Owned(dequantised_bytes)` from `data()` on demand — **commit**
- [ ] **`ParsedModel::remember_to_writer<W: Write>`** — streams lazy views to any writer (file via `serialize_to_file`, or any `Write`), peak `O(largest_tensor_BF16)`. Rewrite `remember()` (file) and 6.8's eager `remember_to_bytes` to delegate here so existing call sites pick up the win automatically. Names the streaming member of the `remember_to_<destination>` family introduced in [Phase 6.8](#phase-68-untrusted-input-resource-governance) — **commit**
- [ ] **`ParsedModel::remember_to_sink` (the `RememberSink`)** — the per-tensor, *no-framing* member: a `FnMut(&str name, &[usize] shape, &[u8] bf16) -> Result<()>` callback invoked once per dequantised tensor, so an embedder (candle-mi) assembles directly into its own tensors with **no intermediate safetensors buffer** — peak ≈ one tensor + the host's model. The streaming-in-memory complement to 6.8's eager `remember_to_bytes`; mirrors the existing [`dequantize_gguf_blocks_to_bf16`](src/remember/gguf.rs) sink shape — **commit**
- [ ] **GGUF CLI streaming path** — refactor `run_remember_gguf` in `src/bin/main.rs` to drop the `tensor_data: Vec<(...)>` accumulator in favour of a lazy iterator. Same `LazyDequantView` infrastructure — **commit**
- [ ] **Peak-heap regression tests via `dhat-rs`** — assert peak heap ≤ `2 × largest_tensor_BF16 + small_constant` on a representative 7 B fixture (FP8 + GGUF). Reuses Phase 6.5 dhat-rs infrastructure — **commit**
- [ ] **Cross-format correctness sweep** — re-run every Phase 1–4.5 cross-validation test against the streaming output path; output bytes must be byte-identical to the eager path on the same input — **commit** — **PUSH + tag `v0.10.0`**

**Deliverable:** `anamnesis` v0.10.0 — peak heap drops from `O(model_size)` to `O(largest_tensor)` for whole-model dequantisation. 70 B+ models become convertible on commodity hardware. All existing public APIs preserved (`remember()` becomes a thin wrapper). — **PUSH + tag `v0.10.0`**

**New dependencies:** None at runtime. Reuses the Phase 6.5 `dhat-rs` dev-dependency for peak-heap assertions.

**Out of scope:**

- **True per-block O(1) streaming** — requires a custom safetensors writer (the `View::data()` method returns `Cow<[u8]>`, not a stream). Per-tensor lazy dequant covers every plausible model size; per-block remains a Future Directions item if real-world peak-heap measurements after Phase 10 ever show insufficient headroom.
- **`WASM32` linear-memory adaptation** — WASM32's 4 GB memory cap is independent of `usize` typing; supporting it requires WASM64 or a fundamentally different I/O model. Listed under Future Directions.

### Future Directions

The following are potential extensions beyond Phase 10, listed for context. They will be promoted to full phases when a concrete need arises.

- **Model surgery** — extract specific layers, merge LoRA adapters with base weights at the file level, split/shard for distributed loading
- **Quantization quality analysis** — per-layer distortion reports, optimal mixed-precision selection, diff two quantized versions of the same model
- **WASM target** — compile anamnesis to WASM for browser-based model inspection (drop a safetensors file, see tensor layout and quantization scheme instantly)
- **GPU-accelerated dequantization** — compute shader (wgpu/Vulkan) kernels for dequantization hot loops, for cases where the Phase 7 CPU SIMD pass is still insufficient (e.g., bulk 70 B-parameter conversions on a workstation with an idle GPU)
- **Pickle GLOBAL safety scan** — the `.pth` parser already enforces a strict GLOBAL allowlist (`is_allowed_global`, weights-only-equivalent) and **fails closed** on a disallowed callable. Expose a *non-failing* scan path that enumerates every GLOBAL the pickle references (allowed + disallowed) so a consumer can render a load-safety verdict — *"only tensor-reconstruction globals (safe)"* vs *"references `posix.system` — arbitrary-code-execution risk under `torch.load(weights_only=False)`"*. Pure metadata over the already-parsed opcode stream (no extra I/O); the Phase 6.11/6.12 resource governance means the scan itself can't be DoS'd by a crafted file. Surfaced by the June 2026 ecosystem survey — `llm_hunter` explicitly *disclaims* pickle-opcode/threat analysis and HuggingFace's scanning is Python/server-side, so this is the Rust-native, offline piece. **Consumer:** hf-fetch-model's `inspect <repo> file.pth` safety line (tracked in its cache-management roadmap)

---

## 4. Key Design Decisions

### Framework-agnostic output

anamnesis outputs standard safetensors files (for dequantization) and raw byte arrays with shape/dtype metadata (for NPZ). It never depends on candle, burn, or tch. Any framework can consume its output.

### Parse-first architecture

Every code path begins with parsing. The `parse/` module is the foundation; `remember/`, `lethe/`, and `inspect` are built on top. This is not incidental — it reflects the design principle that you cannot remember what you have not first parsed.

### Feature gates per format

Each quantization scheme and container format is behind its own feature flag. Users pay only for what they need. Default: `fp8` only.

### No GPU dependency

All operations are pure CPU. FP8 dequantization is bit manipulation, not matrix multiplication. This keeps the crate lightweight and universally deployable.

### SIMD strategy

Conversion loops process billions of elements and are embarrassingly parallel
(each element is independent). The strategy is layered:

1. **Phase 1: SIMD-friendly scalar code.** Write loops that follow the
   CONVENTIONS.md SIMD-friendly rules (contiguous slices, no branches,
   `chunks_exact`, hoisted invariants, separate input/output buffers).
   Verify auto-vectorization with `cargo-show-asm`. This requires **no
   `unsafe`**, no nightly, no extra dependencies, and builds on stable Rust.
   The compiler typically emits AVX2 on x86-64 (Intel + AMD) and NEON on ARM.

2. **If benchmarks demand more: hand-written intrinsics** behind a `simd`
   feature gate. `#[target_feature(enable = "avx2")]` + runtime
   `is_x86_feature_detected!` dispatch, with the scalar path as fallback.
   One dedicated module (`src/remember/simd.rs`), tested against the scalar
   reference. Requires `unsafe` (see CONVENTIONS.md `// SAFETY:` rules).

3. **Not pursued: `std::simd` (portable SIMD).** Requires nightly Rust, which
   would virally infect downstream crates (candle-mi, hf-fetch-model). If
   `std::simd` stabilizes, it becomes the preferred path and replaces both
   the scalar and intrinsic implementations.

Coverage: AVX2 covers all Intel + AMD CPUs since ~2015. NEON covers Apple
Silicon and ARM servers. The scalar fallback covers everything else.

### Error type

A single `AnamnesisError` enum with variants per failure category:
- `Parse { reason }` — format decoding failures (including `NPZ`/`NPY` and `ZIP` errors)
- `Unsupported { format, detail }` — recognized but unimplemented format
- `Io(std::io::Error)` — file system errors

---

## 5. Relationship to Other Projects

### hf-fetch-model

Download crate. Depends on anamnesis for `--dequantize` and `download_and_parse_npz()`. anamnesis provides the format intelligence; hf-fetch-model provides the network I/O. See hf-fetch-model v0.8.0 roadmap.

### candle-mi

MI framework. anamnesis intersects with candle-mi in two ways:

1. **NPZ migration.** candle-mi contained a tightly-coupled NPZ parser for Gemma Scope SAE weights. Phase 3 of anamnesis replaces it. Migration design: `candle-mi/design/migrate-npz-to-anamnesis.md`. `anamnesis` v0.3.0+ provides `parse_npz()` with a `From<AnamnesisError>` bridge to `MIError`. Target: candle-mi v0.1.5.

2. **Auto-config extensions.** The FP8 test models chosen for anamnesis Phase 1 validation introduce two new architectures that candle-mi should support:
   - **`exaone4`** (`LGAI-EXAONE/EXAONE-4.0-1.2B-FP8`) — LLaMA-like with alternating sliding window / full attention ("LLLG" pattern). Close to Gemma 2's alternating scheme. Requires a new `parse_exaone4()` in `config.rs`.
   - **`qwen3`** (`Qwen/Qwen3-1.7B-FP8`) — extends `qwen2` with QK LayerNorm (`q_norm` / `k_norm` per layer). Requires a new `parse_qwen3()` in `config.rs` and handling the extra norm tensors in the forward pass.

   These extensions benefit candle-mi independently of anamnesis — both model families are mainstream and worth supporting regardless.

### candle

HuggingFace's Rust ML framework. anamnesis outputs standard safetensors that candle can load directly. No dependency in either direction — anamnesis reads the safetensors format, not candle's tensor types.
