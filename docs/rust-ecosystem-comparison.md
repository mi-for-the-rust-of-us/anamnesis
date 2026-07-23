# Rust Ecosystem Comparison

**Date surveyed:** 2026-07-23
**anamnesis version at survey:** v0.6.9 (Phase 6.14 complete; tag `v0.6.9`, commit `4fa44dd`)
**Competitor versions:** every version pin, release date, and star count below was re-verified on 2026-07-23 against the live `crates.io` JSON API (`crates.io/api/v1/crates/<name>`) and the GitHub REST API (`api.github.com/repos/<owner>/<repo>`) — not from memory. Source URLs are listed per tool and in the Sources section.
**Scope:** Rust crates and binaries only. Python tooling (`bitsandbytes`, `autogptq`, `autoawq`, `GPTQModel`, `llama.cpp quantize`, `compressed-tensors`, `safetensors` Python bindings) is out of scope by design — this matrix exists to help downstream Rust consumers (`candle`, `mistral.rs`, `burn`, `candle-mi`, plus any new HF-quantised-model loader) evaluate whether `anamnesis` covers what they need vs the alternatives in their own ecosystem.

> **What changed since the 2026-06-15 survey (v0.6.7 + Phase 6.12).** The 15 capability axes are unchanged — no competitor added a multi-family dequant primitive, a standalone GGUF writer, or a multi-format convert dispatch, and no new direct Rust competitor appeared (see "New-competitor sweep" below). The substantive movement is:
> - **(a) Version/maintenance drift across every competitor**, re-pinned here. The notable bumps: `candle-core` **0.10.2 → 0.11.0** (2026-06-26); `mistral.rs` repo release **v0.8.3 → v0.9.0** (2026-07-07); the `aprender`/`apr-cli` monorepo **0.41/0.49 → 0.60.0** (2026-07-06); `ollama` **v0.30.8 → v0.32.1** (2026-07-16); `burn` steady at 0.21.0 but now **15.6k★**. Nobody shipped a new capability that touches an anamnesis axis.
> - **(b) The two `candle` parser DoS issues** of the class anamnesis hardens against have both progressed: the **GGUF** parser ([#3533](https://github.com/huggingface/candle/issues/3533)) and its follow-up allocation site ([#3585](https://github.com/huggingface/candle/issues/3585)) are now **closed**; the **`.pth` pickle-VM** working set ([#3617](https://github.com/huggingface/candle/issues/3617)) remains **open**. All three were contributed to upstream by the anamnesis author, with anamnesis's already-shipped GGUF allocation caps and v0.6.6 pickle-VM governance cited as the reference fixes.
> - **(c) anamnesis's v0.6.8→v0.6.9 work.** Phase 6.13 (v0.6.8): copy-based `no-mmap` full-parse entry points (`parse_*_bytes` / `parse_*_from_reader`, the **recommended path for untrusted input** — a truncated source yields a clean `Err`, not a `SIGBUS`), panic/abort-freedom promoted to a tested CI invariant (`tests/no_panic.rs`) plus a `[profile.python]` unwind profile for the Phase 8 PyO3 `cdylib`, a finer `AnamnesisError` taxonomy (`LimitExceeded` / `DisallowedGlobal`), and a `NumPy`/`BF16` data-ownership contract recorded ahead of the bindings. Phase 6.14 (v0.6.9): the **convert matrix is now complete via an in-memory `BF16` hub** — every `(input × target)` pair routes through one hub, quantized inputs **auto-chain** through `BF16`, `gguf → gguf` dequantizes in place preserving source KV, and `convert` is now a **public library API** (`anamnesis::convert`), not just a CLI subcommand; `--gguf-metadata`/`--gguf-kv` stamp a typed GGUF KV table.
> - **(d) A conceptual (non-Rust) competitor surfaced:** vLLM's [`compressed-tensors`](https://github.com/vllm-project/compressed-tensors) — a Safetensors extension for sparse/quantized storage, the closest *conceptual* analog to anamnesis's parse-and-recover shape, but Python-only and vLLM-coupled (see "Beyond Rust").
>
> These are robustness/correctness/coverage properties, not new matrix columns; they are folded into the "quality properties" note, the Convert/GGUFWrite cells, and the "Where anamnesis stands" section. Roadmap note: the deferred quantised encode kernels (FP8 / GGUF block families) renumbered from **Phase 7.5 → Phase 8.5** ("Lethe Encode Completion", after Phase 7 CPU-SIMD and Phase 8 PyO3).

---

## What anamnesis does (the comparison reference)

`anamnesis` is a Rust library AND CLI binary for **parse-first tensor format work**: parse any HuggingFace-ecosystem tensor archive, inspect metadata without materialising weights, dequantise to `BF16` for downstream consumption, encode `BF16` back into compact quantised formats (Phase 5+), and dispatch any available format pair through a single `convert` primitive (Phase 6). Framework-agnostic by design — produces raw bytes, not bound to a specific tensor library's types. The 15 capability areas used as comparison axes:

| # | Header | Capability |
|---|---|---|
| 1 | **STParse** | `.safetensors` header parsing — path-based (mmap) **and** reader-generic over any `Read + Seek` substrate (HTTP-range friendly), including dequant-companion-tensor awareness (`.qzeros`, `.absmax`, `.quant_map`, `.SCB`, `.g_idx`, …) |
| 2 | **NPZParse** | NumPy `.npz` archive parsing — bulk-read NPY parser with `F16`/`BF16`/`F32`/`F64`/integer support, 3,586 MB/s on real fixtures (17.7× `npyz`) |
| 3 | **PTHParse** | PyTorch `.pth` parsing — minimal pickle VM (~36 opcodes) with security allowlist, zero-copy `Cow::Borrowed` mmap, 11–31× faster than `torch.load()` on torchvision models |
| 4 | **GGUFParse** | `.gguf` parsing — header + metadata + tensor-info, reader-generic inspection variant |
| 5 | **Inspect** | header-only metadata across **all four** formats (zero further I/O after header read; reader-generic for `.safetensors`, `.npz`, `.gguf`, `.pth`) |
| 6 | **DQ-FP8** | `FP8 E4M3` dequant: fine-grained 128×128 block-scale, per-channel `[N,1]`, per-tensor; 2.7–9.7× vs PyTorch CPU |
| 7 | **DQ-GPTQ** | `GPTQ` `INT4`/`INT8` dequant: group-wise scales, `g_idx`, zero-points; 6.5–12.2× vs PyTorch CPU |
| 8 | **DQ-AWQ** | `AWQ` `INT4` dequant: per-group activation-aware, canonical `AWQ_ORDER` nibble interleave; 4.7–5.7× vs PyTorch CPU |
| 9 | **DQ-BnB** | `BitsAndBytes` `NF4`/`FP4`/`INT8` dequant including double-quant (`nested_offset`-correct); 18–54× vs PyTorch CPU for 4-bit |
| 10 | **DQ-GGUF** | `GGUF` dequant: legacy block (`Q4_0`/`Q4_1`/`Q5_0`/`Q5_1`/`Q8_0`/`Q8_1`), K-quants (`Q2_K`–`Q8_K`), `IQ` family (`IQ1_S`/`IQ1_M`/`IQ2_*`/`IQ3_*`/`IQ4_NL`/`IQ4_XS`), `TQ1_0`/`TQ2_0`, `MXFP4` |
| 11 | **EncBnB** | `BitsAndBytes` **encode** — `NF4`/`FP4`/`INT8` plain **and** double-quant, plus sign-of-zero preservation rule; shipped in Phase 5 commits `a5c452d` / `24cba42` / `ab4e735` |
| 12 | **GGUFWrite** | `.gguf` **writing** — the format-symmetric inverse of `parse_gguf`: emits 24-byte header + metadata KV table + tensor-info table + aligned tensor data. Scalar dtypes only (`F32`/`F16`/`BF16`/`F64`/`I8`–`I64`); quantised emit (`Q*`/`IQ*`/`TQ*`/`MXFP4`) reserved for Phase 8.5 through the same writer scaffold. Phase 6, commit `6768ee0` |
| 13 | **Convert** | multi-format conversion dispatch — CLI `amn convert <input> --to <target>` **and** library `anamnesis::convert` (`ConvertTarget` / `ConvertOptions` / `ConvertStats`, re-exported since Phase 6.14). Targets: `safetensors` (alias `bf16`), `gguf` (unquantised passthrough, with `--gguf-metadata`/`--gguf-kv` typed KV stamping), `bnb-nf4`. Since Phase 6.14 every `(input × target)` pair routes through one in-memory **`BF16` hub**: any of `.safetensors`/`.npz`/`.pth`/`.gguf` reaches any target, a **quantized input auto-chains** through `BF16` (no manual dequantize-first hop), and `gguf → gguf` dequantizes in place while inheriting the source metadata KV; combinations still outside the matrix return clear `Unsupported` rather than silent fall-through. Measured **1.11×–6.75× faster than the closest Python ecosystem default** (numpy / gguf-py / bitsandbytes CPU) at 4096×4096, release, CPU; **2.17×–8.24× faster than PyTorch-CPU equivalents** for the two non-PyTorch paths. Phase 6 commit `f5cdee2`, hub completion Phase 6.14 (`src/convert.rs`) |
| 14 | **Lib** | embeddable Rust library API |
| 15 | **CLI** | CLI binary (`anamnesis` + alias `amn`) |

Quality properties shared across every kernel (not table columns — would otherwise mark all-✓ for anamnesis trivially):

- **Bit-exact** (0 ULP) validation against PyTorch reference on real model fixtures. As of v0.6.4 the cross-validation fixtures are anchored on **the canonical libraries' own code** (`bitsandbytes.functional.dequantize_4bit`, AutoAWQ `unpack_awq`, GPTQModel `dequantize_weight`) rather than re-implemented formulas — a methodology fix that surfaced and closed three real BnB/AWQ nibble-order/offset bugs that had validated green behind circular fixtures (v0.6.4), and an output-orientation contract bug (`[in,out]` vs `nn.Linear` `[out,in]`, v0.6.5).
- **`#![deny(unsafe_code)]`** at the crate root with a documented, tightly-scoped opt-in for `memmap2::Mmap::map` only.
- **Hardened against untrusted input** (v0.6.1→v0.6.9, the [`candle #3533`](https://github.com/huggingface/candle/issues/3533) DoS class): caller-configurable `ParseLimits` budgets (single-alloc / cumulative-heap / item-count / decompression-ratio, all fail-fast before allocation, each rejection now a typed `LimitExceeded { limit, message }` since Phase 6.13), permanent always-on floors (pickle-VM working-set `512 MiB` + depth `256`; `ZIP_MAX_ENTRIES` `1 048 576`), a **vendored read-only ZIP container reader** with full `ZIP64` support and a compression allowlist (`zip` is now a dev-only dependency), copy-based `no-mmap` entry points (`parse_*_bytes` / `parse_*_from_reader`) that turn a truncated/concurrently-written source into a clean `Err` instead of an uncatchable `SIGBUS` — the recommended path for untrusted input — and a `cargo-fuzz` campaign with a target per parser (including the Phase 6.13 owned-byte `fuzz_*_bytes` targets) and an in-crate differential oracle against the `zip` crate.
- **No panic, no abort on any input** (Phase 6.13, `tests/no_panic.rs`): every public parse/inspect entry point is a tested `catch_unwind` invariant over synthetic malformed shapes and bit-flipped fixtures — promoted from an implicit property (a panic would crash the fuzzer) to an explicit CI contract. Release artefacts build `panic = "abort"` (a reachable panic is treated as a process kill in the threat model); the Phase 8 PyO3 `cdylib` builds a dedicated `[profile.python]` with `panic = "unwind"` so a panic surfaces as a catchable Python `PanicException`.

---

## Per-tool detail

### A. Direct competitors (multi-family Rust dequantisation libraries)

**None known.** No other Rust crate exposes dequantisation across multiple families (`FP8` + `GPTQ` + `AWQ` + `BnB` + `GGUF`) as a standalone primitive. A fresh 2026-07-23 sweep (crates.io / GitHub / lib.rs for "multi-family dequant library", "standalone Rust GGUF writer", "tensor convert CLI") found no new entrant: the closest adjacent items are `gguf-rs-lib` (Rust, GGUF read+write only, 6★, last release 2025-09-02) and `ggufy` (a SafeTensors→GGUF converter written in **Zig**, not Rust). The nearest *conceptual* competitor to appear is vLLM's `compressed-tensors`, but it is Python-only and vLLM-coupled (see "Beyond Rust"). Dequant in the current Rust ecosystem still lives **inside** inference frameworks (see category C), tightly coupled to those frameworks' tensor types and loading paths. `anamnesis` remains the first crate to make dequant a framework-agnostic library that produces raw `BF16` bytes any downstream tensor system can consume. Phase 5 extends this to the encode side: **no other Rust crate ships `BnB` encode kernels at all** — quantisation in Rust today means "call `bitsandbytes` from Python via PyO3" or "use `mistral.rs quantize` and accept its custom UQFF format". Phase 6 extends further to **standalone GGUF writing** (scalar dtypes today, quantised types at Phase 8.5 through the same scaffold) and a **multi-format convert dispatch** — `amn convert any.safetensors --to bnb-nf4` / `amn convert weights.npz --to gguf` / `amn convert model.pth --to gguf` are one-call conversions (CLI or the `anamnesis::convert` library API) with no analog elsewhere in Rust, and since Phase 6.14 a quantized input auto-chains through the `BF16` hub to any target. The only adjacent Rust convert primitive, `apr-cli convert --quantize q4_k`, is lossy by construction (it cannot pass through `BF16` unchanged) and is tightly bound to the GGUF/APR target.

### B. Tensor format parsers / inspectors (single-format or narrow scope)

**safetensors** — [github.com/safetensors/safetensors](https://github.com/safetensors/safetensors) | [crates.io/crates/safetensors](https://crates.io/crates/safetensors)
The canonical format crate. **v0.8.0 (2026-06-09)**, ~3.8k stars (3,812), actively maintained (last push 2026-07-22). Library-only, no CLI. Header read/write, lazy mmap-backed views, plus — new since the prior survey's 0.7.x snapshot — a non-mmap `pread(2)` backend, Apple-Silicon Metal/MPS direct loading, GIL-free serialization, and additional FP8 dtypes (`float8_e4m3fnuz`/`float8_e5m2fnuz`). **Does not** dequantise, doesn't expose dtype histograms, doesn't know about quant companion tensors (`.qzeros` / `.absmax` / `.quant_map`); those are anamnesis-side concepts.

**safetensors_explorer** — [github.com/EricLBuehler/safetensors_explorer](https://github.com/EricLBuehler/safetensors_explorer) | [crates.io/crates/safetensors_explorer](https://crates.io/crates/safetensors_explorer)
Interactive TUI inspector for `.safetensors` AND `.gguf`. **v0.2.0 (2025-07-28)**, 59 stars; no release in ~12 months but the repo saw a recent push (last push 2026-06-20). Tree view, fuzzy search, sharded model index detection. CLI/TUI-only (no library API), inspect-only (no dequant, no encode, no format conversion). The closest analog to `anamnesis inspect` but on metadata only.

**safetensors-cli** — [github.com/gzsombor/safetensors-cli](https://github.com/gzsombor/safetensors-cli) | [crates.io/crates/safetensors-cli](https://crates.io/crates/safetensors-cli)
**v0.1.0 (2023-06-17)**, 1 star, functionally frozen (~3 years at one release); the only recent activity is renovate-bot dependency bumps (last push 2026-06-10). Local-file `.safetensors` header dump only.

**gguf-rs-lib / gguf-cli** — [github.com/ThreatFlux/gguf](https://github.com/ThreatFlux/gguf) | [crates.io/crates/gguf-rs-lib](https://crates.io/crates/gguf-rs-lib)
GGUF parsing library + `gguf-cli` binary (`info`/`tensors`/`metadata`/`validate`). **v0.2.5 (2025-09-02)**, 6 stars, active (last push 2026-07-20). Zero-copy parsing, optional mmap, async (Tokio). GGUF only — no `.safetensors`/`.npz`/`.pth` support, no dequantisation. The crate description advertises "reading **and writing** GGUF files," but the README demonstrates no writer/encode path and no dequant — treat write support as claimed-in-metadata, not demonstrated.
*(Naming note: the crates.io name `gguf-rs` is a **different, unrelated** project — zackshen's GGUF parser, [crates.io/crates/gguf-rs](https://crates.io/crates/gguf-rs) v0.1.8 (2026-06-15), repo [github.com/zackshen/gguf](https://github.com/zackshen/gguf), 21★. The 2026-05-21 survey conflated the two; the `gguf-cli` binary and the 0.2.5 version belong to ThreatFlux's `gguf-rs-lib`.)*

**inspector-gguf** — [github.com/FerrisMind/inspector-gguf](https://github.com/FerrisMind/inspector-gguf) | [docs.rs/inspector-gguf](https://docs.rs/inspector-gguf)
GUI (egui drag-and-drop) + CLI + library. **v0.3.1 (2025-12-15)**, 5 stars, frozen (last push 2025-12-15). Exports analysis as CSV/YAML/Markdown/HTML/PDF. Tokenizer + chat-template surface. GGUF only, inspect only, heavyweight footprint.

**npyz** — [crates.io/crates/npyz](https://crates.io/crates/npyz) | [docs.rs/npyz](https://docs.rs/npyz)
`.npy` / `.npz` parser. **v0.9.1 (2026-05-02)**, 33 stars (`ExpHP/npyz`), recently maintained. Per-element deserialisation (slower than anamnesis's bulk `read_exact` path on LE data; the anamnesis README documents a 17.7× speed delta on a 302 MB fixture). No dequantisation, no `.safetensors` / `.pth` / `.gguf` support.

**ndarray-npy** — [crates.io/crates/ndarray-npy](https://crates.io/crates/ndarray-npy)
`.npy` ↔ `ndarray::Array` adapter. **v0.10.0 (2025-12-15)**, 71 stars, low-activity but alive. Useful when the downstream type is `ndarray`; not framework-agnostic.

**serde-pickle** — [crates.io/crates/serde-pickle](https://crates.io/crates/serde-pickle)
General Python pickle decoder via the serde ecosystem. **v1.2.0 (2024-11-22)**, 224 stars, dormant (no code pushes since the 1.2.0 release). Pure VM with no `.pth`-specific opcode allowlist, no zero-copy tensor extraction, no security-allowlist on `GLOBAL` references — usable as a primitive, but **substantial gap** between "pickle parser" and "safe `.pth` loader" that anamnesis closes (and which anamnesis v0.6.6 hardened further with pickle-VM working-set + recursion-depth floors).

### C. Inference frameworks (parse + dequant + inference bundled)

**candle** — [github.com/huggingface/candle](https://github.com/huggingface/candle)
Hugging Face's Rust ML framework. `candle-core` **v0.11.0 (2026-06-26)**, **20,709 stars**, active (last push 2026-07-23). Parses `.safetensors` and `.gguf` natively; dequant happens **inside** the tensor-loading code path (`candle-quantized`, `quantized-var-builder`), tightly coupled to `candle::Tensor`. GGUF dequant (llama.cpp quantized types) is the most complete family on the candle side; no `FP8`, no first-class `GPTQ`/`AWQ`/`BnB` loaders (those exist in downstream candle-* crates like `candle-transformers` but as model-specific glue, not standalone dequant primitives). `.pth` loading via the `tch` bridge or hand-rolled paths; no equivalent of anamnesis's pickle-VM-with-allowlist. No encode side. **Robustness note:** candle's parsers have carried two unbounded-allocation/recursion DoS issues of the same class anamnesis hardens against, and the anamnesis author contributed to both. (1) The **GGUF** parser DoS ([#3533](https://github.com/huggingface/candle/issues/3533), the CVE-2025-66960 / Ollama class — a 37-byte file driving multi-GB allocations): the fix ([#3556](https://github.com/huggingface/candle/pull/3556), size caps + remaining-byte validation + dim/recursion limits) merged 2026-06-06, and anamnesis's follow-up flagged a further allocation site (`TensorInfo::read`'s unchecked element product) beyond that fix's reach — routed to [#3585](https://github.com/huggingface/candle/issues/3585), with anamnesis's GGUF parser (checked product + `MAX_TENSOR_ELEMENTS` cap + mmap byte-range check) cited as the reference. **Both #3533 and #3585 are now closed.** (2) The **`.pth` pickle-VM** working-set DoS ([#3617](https://github.com/huggingface/candle/issues/3617), CWE-1325 memo-replay amplification + CWE-674 recursive `Drop`), filed by the anamnesis author as the direct analog of anamnesis's v0.6.6 pickle-VM governance (`MAX_PICKLE_WORKING_SET` 512 MiB + `MAX_PICKLE_VM_DEPTH` 256) and still **open** as of this survey. anamnesis shipped and `cargo-fuzz`-ed both controls before the upstream reports.

**mistral.rs** — [github.com/EricLBuehler/mistral.rs](https://github.com/EricLBuehler/mistral.rs)
LLM inference engine. Repo release **v0.9.0 (2026-07-07)**, **7,514 stars**, very active (last push 2026-07-23); the published `mistralrs` crate lags at 0.8.1 (2026-04-02) — the repo tags ahead of the crate release. Internally handles `BnB`/`GPTQ` (Marlin kernels)/`AWQ` (expanded)/`GGUF`/`ISQ`/`AFQ`/`MXFP4`/`F8Q8` for its own model loading; ships a `mistralrs quantize` / `--quant` smart-quant path that emits its own **UQFF** format (an ISQ-serialized, candle-friendly bundle), not `BnB` or `GGUF`. Dequant is not exposed as a reusable library primitive; the quantisation/dequantisation kernels are entangled with the inference graph. Closest "all-format" coverage on the inference side but the wrong shape if you want raw `BF16` bytes for a different framework.

**burn** — [github.com/tracel-ai/burn](https://github.com/tracel-ai/burn)
Multi-backend training+inference framework. **v0.21.0 (2026-05-07)**, **15,627 stars**, active (last push 2026-07-23). Weight loading is in-code via `PyTorch`/`Safetensors`/`ONNX` import; no standalone CLI for parse/inspect/dequant. v0.21.0 is infrastructure-focused (distributed stack, `burn.toml`, CPU backend) — does **not** cover the `GPTQ`/`AWQ`/`BnB`/`GGUF` quantised families as first-class loaders. Out of scope for the comparison axes here.

**tract** — ONNX-only inference. Out of scope (no `.safetensors`/`.pth`/`.gguf`/quant-family overlap).

**llm** — superseded by `mistral.rs` / `candle`; no active development.

**apr-cli** — [docs.rs/apr-cli](https://docs.rs/crate/apr-cli/latest) | part of [paiml/aprender](https://github.com/paiml/aprender)
`apr-cli` **v0.60.0 (2026-07-06)**; the `aprender` monorepo (~70–80 workspace crates) is likewise at **v0.60.0 (2026-07-06)**, 109 stars, very active (last push 2026-07-07). The `apr-cli` crate is a CLI binary (no library API for the format tooling); the `aprender` crate *is* a published Rust library, but it exposes ML algorithms, not the tensor-format parse/convert/quantize internals. Subcommands: `inspect` / `list` / `tui` / `diff` / `chat` / `quantize` / `serve` / `qa`. Reads APR (native), GGUF, and SafeTensors. Documentation explicitly says it "specialises in quantization rather than dequantization" — no dequant primitive is exposed. The most relevant overlap: `apr convert model.safetensors --quantize q4_k -o model.gguf` — a one-shot SafeTensors → GGUF/APR Q4_K conversion. This is the **closest existing Rust analog** to anamnesis's Phase 5 / Phase 6 / Phase 8.5 encode + convert flow, but with a different output format and lossy-by-construction (it cannot pass `BF16` through unchanged). Positioning is different: apr-cli is heavyweight ML/inference machinery (`chat`/`serve`/`tui`/`qa` as the primary value); anamnesis is a thin framework-agnostic library that produces raw bytes. Also ships `apr serve plan` for the VRAM-fit verdict.

### D. PyTorch `.pth` ecosystem

**tch** — [github.com/LaurentMazare/tch-rs](https://github.com/LaurentMazare/tch-rs)
Rust bindings for `libtorch`. **v0.24.0 (2026-03-26)** (targets libtorch 2.11.0), 5,453 stars, active (last push 2026-07-17). Loads `.pth` natively because it ships the full PyTorch C++ runtime. ~250 MB native dependency, opaque tensor type (`tch::Tensor`), wrong shape if you want a thin parse-and-convert tool. No quant-family dequant exposed.

**pickle** / **serde-pickle** — see category B.

### E. Encode (quantisation) Rust tools

**`mistral.rs quantize`** — emits UQFF (mistral.rs's custom packaging of candle tensors). Not a general-purpose quantisation primitive; tightly bound to mistral.rs's inference path.

**`apr-cli quantize`** — emits GGUF/APR Q4_K (and int8/int4/fp16). Lossy by construction, no `BnB`/`FP8` on-disk layout, no library API.

**No other Rust crate** ships dequant-family encode kernels in the *source* libraries' on-disk layouts (`BnB`/`GPTQ`/`AWQ`). Quantisation in Rust today is otherwise "run Python and import the bytes back". `anamnesis` Phase 5 ships the first standalone `BnB` encode kernel set; Phase 8.5 extends to `FP8`/`GGUF` (legacy/K-quants/`IQ`/`TQ`/`MXFP4`) through the same `lethe` round-trip harness.

---

## Comparison matrix

Cells: ✓ = present · ◐ = partial / framework-coupled / narrow · — = absent.

| Tool | STParse | NPZParse | PTHParse | GGUFParse | Inspect | DQ-FP8 | DQ-GPTQ | DQ-AWQ | DQ-BnB | DQ-GGUF | EncBnB | GGUFWrite | Convert | Lib | CLI |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **anamnesis v0.6.9** *(reference)* | ✓ path + reader + no-mmap | ✓ | ✓ pickle-VM | ✓ | ✓ all 4 fmts | ✓ E4M3 3 modes | ✓ INT4/INT8 + g_idx | ✓ INT4 | ✓ NF4/FP4/INT8 + DQ | ✓ block/K/IQ/TQ/MXFP4 | ✓ NF4/FP4/INT8 + DQ | ✓ scalar dtypes (Phase 8.5: quantised) | ✓ BF16-hub, full input×target + lib `convert` | ✓ | ✓ `anamnesis`/`amn` |
| safetensors 0.8.0 | ✓ lib only | — | — | — | — no CLI | — | — | — | — | — | — | — | — | ✓ | — |
| safetensors_explorer 0.2.0 | ◐ inspect TUI | — | — | ◐ inspect TUI | ✓ ST + GGUF metadata | — | — | — | — | — | — | — | — | — bin only | ✓ TUI |
| safetensors-cli 0.1.0 | ◐ minimal | — | — | — | ◐ minimal local | — | — | — | — | — | — | — | — | — | ✓ |
| gguf-rs-lib 0.2.5 (ThreatFlux) | — | — | — | ✓ | ◐ GGUF only | — | — | — | — | — | — | ◐ "write" in crate metadata, not demonstrated | — | ✓ | ✓ `gguf-cli` |
| inspector-gguf 0.3.1 | — | — | — | ✓ | ◐ GGUF only, GUI+CLI | — | — | — | — | — | — | — | — | ✓ | ✓ |
| npyz 0.9.1 | — | ✓ slower 1× | — | — | ◐ NPY metadata | — | — | — | — | — | — | — | — | ✓ | — |
| ndarray-npy 0.10.0 | — | ◐ `ndarray`-bound | — | — | — | — | — | — | — | — | — | — | — | ✓ | — |
| serde-pickle 1.2.0 | — | — | ◐ generic pickle, no `.pth` opcode allowlist or zero-copy tensor extraction | — | — | — | — | — | — | — | — | — | — | ✓ | — |
| candle 0.11.x | ✓ via `candle-core` | — | ◐ via `tch` bridge | ✓ via `candle-quantized` | ◐ in-code only | — | ◐ model-specific glue | ◐ model-specific glue | ◐ model-specific glue | ◐ Q\*\_\* + K-quants | — | — | — | ✓ framework | — no central CLI |
| mistral.rs 0.9.0 | ✓ for inference | — | — | ✓ for inference | ◐ via `doctor` | ◐ F8Q8, inference-coupled | ◐ for inference (Marlin), tensor-coupled | ◐ for inference, tensor-coupled | ◐ for inference, tensor-coupled | ◐ for inference, tensor-coupled | ◐ UQFF only, not BnB | — | ◐ UQFF target | ✓ | ✓ `mistralrs` |
| burn 0.21.0 | ◐ via importer | — | ◐ via importer | — | — | — | — | — | — | — | — | — | — | ✓ framework | — |
| apr-cli 0.60.0 | ✓ read for inference | — | — | ✓ read + write | ◐ APR-centric | — | — | — | — | — | — bound to GGUF/APR Q4\_K | ◐ Q4\_K / Q4\_0 only (lossy by construction) | ◐ ST→GGUF/APR Q4\_K (lossy) | — bin only | ✓ |
| tch 0.24.0 | — | — | ✓ via `libtorch` | — | — | — | — | — | — | — | — | — | — | ✓ | — |

---

## Where anamnesis stands

- **Multi-format parser**: anamnesis is the only Rust crate covering all four of `.safetensors` / `.npz` / `.pth` / `.gguf` in one library with a unified `inspect` story.
- **Multi-family dequant**: anamnesis is the only Rust crate exposing dequantisation across **all five** of `FP8` / `GPTQ` / `AWQ` / `BnB` / `GGUF` as standalone, framework-agnostic primitives that produce raw `BF16` bytes. Every other tool in the matrix either covers one family (gguf-rs-lib) or covers many families but tightly coupled to a specific tensor library's loader path (candle, mistral.rs). The 2026-06-15 new-competitor sweep confirmed no Rust entrant has changed this.
- **Encode side (Phase 5)**: anamnesis is **the first** Rust crate to ship `BnB` encode kernels at all. The closest adjacents are `mistral.rs quantize` (targets its own UQFF format) and `apr-cli quantize` (targets GGUF/APR Q4_K) — neither produces the `bitsandbytes` on-disk layout.
- **GGUF write + multi-format convert (Phase 6 → 6.14)**: anamnesis is **the first** Rust crate to ship a standalone GGUF writer (scalar dtypes today, quantised dtypes at Phase 8.5 through the same scaffold) AND a multi-format conversion dispatch, exposed both as `amn convert <input> --to <target>` and — since Phase 6.14 — as a public library API (`anamnesis::convert` / `ConvertTarget` / `ConvertOptions` / `ConvertStats`). Since Phase 6.14 every `(input × target)` pair routes through one in-memory **`BF16` hub**, so any of `.safetensors` / `.npz` / `.pth` / `.gguf` reaches any target (`safetensors` / `gguf` / `bnb-nf4`), a **quantized input auto-chains** through `BF16` with no manual dequantize-first hop, and `gguf → gguf` dequantizes in place while inheriting the source metadata KV (`--gguf-metadata`/`--gguf-kv` stamp a typed KV table). The closest adjacents are `apr-cli convert` (lossy-by-construction, always quantises to GGUF/APR Q4_K, `.safetensors → .gguf` only) and `gguf-rs-lib` (advertises GGUF write but does not demonstrate it). anamnesis's convert primitive is measured **1.11×–6.75× faster than the closest Python ecosystem default** (numpy / gguf-py / bitsandbytes CPU) at 4096×4096, release, CPU, and **2.17×–8.24× faster than PyTorch-CPU equivalents** for the two non-PyTorch baselines — see `tests/cross_validation_convert.rs::t14_perf_vs_python_size_matched`. The Phase 6.14 copy-elimination pass also lowered convert peak heap (`bnb-nf4` −39 %, `npz → *` −49.6 %, `gguf → gguf` −20 % cumulative) with no output change.
- **`.pth` loading without `libtorch`**: anamnesis's pickle-VM-with-security-allowlist is the only pure-Rust safe `.pth` loader; the alternatives are (a) `tch` (pulls in ~250 MB of native `libtorch`) or (b) `serde-pickle` (general pickle, no `.pth`-specific opcode allowlist, no zero-copy tensor extraction, no security model for `GLOBAL` references, dormant since 2024-11).
- **CLI + Lib parity**: anamnesis ships both. `safetensors_explorer` / `safetensors-cli` / `apr-cli` are bin-only; `safetensors` / `tch` / `burn` are lib-only; `candle` has no central CLI for parse/inspect work.
- **Robustness against untrusted input**: anamnesis treats malformed/malicious files as a first-class threat model. v0.6.1→v0.6.9 ship caller-configurable `ParseLimits` (fail-fast before allocation, each rejection a typed `LimitExceeded`), always-on pickle-VM working-set + recursion-depth floors, a vendored `ZIP64`-aware container reader with a compression allowlist (dropping `zip` to a dev-only dependency), copy-based `no-mmap` full-parse entry points (`parse_*_bytes` / `parse_*_from_reader`) that make a truncated source a clean `Err` rather than a `SIGBUS` — the recommended path for untrusted input — panic/abort-freedom as a tested CI invariant (`tests/no_panic.rs`), and a `cargo-fuzz` campaign with a target per parser (including Phase 6.13's owned-byte `fuzz_*_bytes`) plus a differential oracle. The same DoS class has surfaced as real issues in `candle` — the GGUF parser ([#3533](https://github.com/huggingface/candle/issues/3533) + follow-up [#3585](https://github.com/huggingface/candle/issues/3585), both now closed) and the `.pth` pickle VM ([#3617](https://github.com/huggingface/candle/issues/3617), open) — all contributed to upstream by the anamnesis author, with anamnesis's already-shipped GGUF allocation caps and v0.6.6 pickle-VM governance cited as the reference fixes. No other tool in the matrix exposes a comparable, reproducible-from-the-tree hardening story.
- **Validation infrastructure (Phase 6.5)**: anamnesis ships dev-only `criterion` runtime benchmarks (`benches/dequant.rs`, `benches/parsing.rs`) AND `dhat-rs`-instrumented peak-heap regression tests (`tests/peak_heap_{gptq,awq,bnb_dq,zip_metadata}.rs`) that assert observed allocations stay within the documented `# Memory` ceilings — the v0.6.7 vendored ZIP reader is pinned by `peak_heap_zip_metadata.rs` at 41 B/entry vs the `zip` crate's 337 B/entry (8.07× resident reduction). No other crate in the matrix exposes both perf and memory-efficiency claims as reproducible-from-the-tree assertions — `mistralrs` carries internal benchmarks but they're inference-coupled; `gguf-rs-lib` / `safetensors_explorer` / `apr-cli` ship no perf-validation infrastructure visible in their repos.

### Closest competitor footprint

The closest competitor coverage requires a **union of four or five tools**: `safetensors` (parse only) + `safetensors_explorer` (inspect TUI, ST+GGUF) + `gguf-rs-lib` (GGUF parse + CLI inspect) + `mistral.rs` (inference-coupled BnB/GPTQ/AWQ/GGUF dequant + UQFF encode) + optionally `apr-cli` (SafeTensors → GGUF/APR Q4_K quantise + convert). Even that union still does not cover:

- Cross-format **lossless** conversion across **all** available pairs (anamnesis: `.pth` → `.safetensors`, `.npz` → `.safetensors`, `.npz` → `.gguf`, `.pth` → `.gguf`, `.safetensors` → `.gguf`, `.safetensors` → `.bnb-nf4` are all single CLI calls; apr-cli's `convert` is always lossy via quantize and only handles `.safetensors` → `.gguf/.apr Q4_K/Q4_0`)
- Standalone dequant primitives (mistral.rs dequant is bound to its inference graph; apr-cli explicitly doesn't expose dequant)
- `BnB` encode in `BnB`'s on-disk layout (mistral.rs encodes to UQFF, apr-cli to GGUF/APR Q4_K — neither produces BnB)
- Standalone GGUF writer (apr-cli writes GGUF but only as the output of its quantize pipeline; gguf-rs-lib advertises but does not demonstrate write; anamnesis's `write_gguf` accepts any caller-supplied `(name, shape, dtype, data)` tensors and writes a self-describing GGUF v3, with quantised dtypes reserved for the Phase 8.5 scaffold extension)
- `.pth` loading without a `libtorch` dependency
- Reader-generic header parsing (HTTP-range remote inspect over any `Read + Seek` substrate)
- Library API for embedders (`apr-cli` / `safetensors_explorer` / `gguf-cli` are bin-only; embedders cannot reuse the orchestration)
- A documented untrusted-input threat model with caller-tunable `ParseLimits` and a fuzzed parser surface

### Beyond Rust (context — not in the matrix by design)

**ollama** ([github.com/ollama/ollama](https://github.com/ollama/ollama)) — **v0.32.1 (2026-07-16)**, ~176.7k stars, written in Go (66 %) + C (27 %, mostly llama.cpp glue) + TS (3 %). The dominant local-LLM CLI in the broader ecosystem. Most relevant for our scope: ships a `convert/` directory in Go with `reader_safetensors.go` and `reader_torch.go` plus architecture-specific converters (Llama / Llama4 / Gemma / Qwen / Mistral / Mixtral / Phi3 / …) — independent re-implementations of the same parse work `anamnesis` does in Rust, targeting the same GGUF output that `apr-cli` (and `anamnesis` Phase 8.5) targets. The pipeline is `HF SafeTensors → ollama's Go converter → GGUF intermediate → llama.cpp quantize → final GGUF`. **No dequantisation primitives exposed** — entirely bundled with the inference path, same pattern as `candle` / `mistral.rs`. **No library API for Rust embedders** (Python + JS clients only).

**compressed-tensors** ([github.com/vllm-project/compressed-tensors](https://github.com/vllm-project/compressed-tensors)) — **306 stars**, active (last push 2026-07-23), Python. A Safetensors extension defining on-disk layouts for sparse/quantized storage (FP8, INT4/INT8 pack-quant, 2:4 sparsity), the storage format the vLLM/`llm-compressor` stack produces and loads. This is the closest *conceptual* competitor to anamnesis's parse-and-recover shape — it standardises quantized weights on top of safetensors, exactly the surface anamnesis reads. But it is **Python-only and vLLM-coupled**: decompression to `BF16` runs inside the vLLM/PyTorch load path, not as a framework-agnostic primitive, and there is no Rust API. It belongs here as ecosystem context, not in the Rust matrix. (An anamnesis reader for the `compressed-tensors` config schema is a plausible future coverage item, not a shipped one.)

The parse + quantise pipeline is being solved *somewhere* in every active local-LLM ecosystem:

| Ecosystem | Tool | Quantise output | Convert / write |
|---|---|---|---|
| Python | `bitsandbytes` + `autogptq` + `autoawq` + `llama.cpp quantize` | BnB, GPTQ, AWQ, GGUF (each via separate tool) | NPZ↔ST via `safetensors-py`; PTH→ST via `safetensors.torch`; ST→GGUF via `gguf-py` |
| Go | `ollama convert` (+ llama.cpp) | GGUF only | ST/PTH → GGUF (bundled with inference, no standalone primitive) |
| Zig | `ggufy` | GGUF (Q2_K…Q8_0) | ST → GGUF (diffusion-model focused; not Rust) |
| Rust | `apr-cli` quantize | GGUF / APR Q4_K / Q4_0 only | ST → GGUF/APR Q4_K (lossy by construction) |
| Rust | `mistral.rs quantize` | UQFF only | ST/PTH/GGUF → UQFF (inference-coupled) |
| Rust | `anamnesis` (Phase 5+6) | BnB plain + DQ (Phase 5); GGUF scalar passthrough (Phase 6); FP8 + GGUF block families at Phase 8.5 | **`amn convert` / `anamnesis::convert` BF16-hub dispatch** covering ST/NPZ/PTH/GGUF → ST/GGUF/BnB-NF4, every input×target pair with quantized-input auto-chaining (lossless where the pipeline permits) |

What's distinctive about `anamnesis` isn't *that* it exists — it's the **library-first + framework-agnostic + multi-family-dequant + standalone convert dispatch** shape. None of the cross-language analogs ship dequant as a reusable primitive consumable from outside their own inference path, and none ship a unified convert primitive that routes any of four input formats through any of three target formats via a single CLI subcommand.

---

## Sources

All version pins, dates, and star counts verified 2026-07-23 via the crates.io JSON API and GitHub REST API.

- [safetensors GitHub](https://github.com/safetensors/safetensors) / [crates.io](https://crates.io/crates/safetensors) — v0.8.0 (2026-06-09), 3,812★
- [safetensors_explorer GitHub](https://github.com/EricLBuehler/safetensors_explorer) / [crates.io](https://crates.io/crates/safetensors_explorer) — v0.2.0 (2025-07-28), 59★
- [safetensors-cli GitHub](https://github.com/gzsombor/safetensors-cli) / [crates.io](https://crates.io/crates/safetensors-cli) — v0.1.0 (2023-06-17), 1★
- [gguf-rs-lib GitHub (ThreatFlux)](https://github.com/ThreatFlux/gguf) / [crates.io](https://crates.io/crates/gguf-rs-lib) — v0.2.5 (2025-09-02), 6★ · (distinct from [gguf-rs/zackshen](https://crates.io/crates/gguf-rs) v0.1.8 (2026-06-15), 21★)
- [inspector-gguf GitHub (FerrisMind)](https://github.com/FerrisMind/inspector-gguf) / [docs.rs](https://docs.rs/inspector-gguf) — v0.3.1 (2025-12-15), 5★
- [npyz docs.rs](https://docs.rs/npyz) / [crates.io](https://crates.io/crates/npyz) — v0.9.1 (2026-05-02), 33★
- [ndarray-npy crates.io](https://crates.io/crates/ndarray-npy) — v0.10.0 (2025-12-15), 71★
- [serde-pickle crates.io](https://crates.io/crates/serde-pickle) — v1.2.0 (2024-11-22), 224★
- [candle GitHub](https://github.com/huggingface/candle) / [candle-core crates.io](https://crates.io/crates/candle-core) — v0.11.0 (2026-06-26), 20,709★ · GGUF DoS [#3533](https://github.com/huggingface/candle/issues/3533) / fix [#3556](https://github.com/huggingface/candle/pull/3556) (merged 2026-06-06) + follow-up [#3585](https://github.com/huggingface/candle/issues/3585) — **both closed** · pickle-VM DoS [#3617](https://github.com/huggingface/candle/issues/3617) (open)
- [mistral.rs GitHub](https://github.com/EricLBuehler/mistral.rs) — repo release v0.9.0 (2026-07-07), 7,514★ · crate `mistralrs` 0.8.1 (2026-04-02)
- [burn GitHub](https://github.com/tracel-ai/burn) — v0.21.0 (2026-05-07), 15,627★
- [apr-cli docs.rs](https://docs.rs/crate/apr-cli/latest) / [aprender monorepo](https://github.com/paiml/aprender) — apr-cli v0.60.0 (2026-07-06) / aprender v0.60.0 (2026-07-06), 109★
- [tch-rs GitHub](https://github.com/LaurentMazare/tch-rs) / [crates.io](https://crates.io/crates/tch) — v0.24.0 (2026-03-26), 5,453★
- [ollama GitHub](https://github.com/ollama/ollama) — v0.32.1 (2026-07-16), ~176.7k★ (Go, ecosystem context only)
- [compressed-tensors GitHub (vLLM)](https://github.com/vllm-project/compressed-tensors) — 306★, active (Python, ecosystem context only)

---

## Open questions

These items remain genuinely undecided and worth a follow-up pass before wider circulation:

- **`NoUnsafe`, `BitExact`, `Hardened`, and `NoPanic` as table columns?** Currently relegated to the "quality properties" note since they're properties rather than capabilities, but a "rigour" column (no-unsafe + bit-exact + fuzzed/limit-bounded parsers + tested panic/abort-freedom) would be a real differentiator vs e.g. candle's recently-patched GGUF-parser DoS and unsafe-heavy framework loaders. The candle #3533/#3585 (now closed) and #3617 (open) episodes, plus Phase 6.13's promotion of panic-freedom to a CI invariant, strengthen the case for surfacing this in the matrix proper.
- **Per-tool detail length.** Kept brief (~1–4 sentences each); could expand each tool's section with reverse-dep notes / download counts if needed for wider circulation.

## Resolved scope decisions

These are recorded so the survey does not loop back to them in future passes:

- **gguf-rs naming.** The crates.io name `gguf-rs` (zackshen, v0.1.7, a GGUF parser) is **not** the same project as ThreatFlux's `gguf-rs-lib` (v0.2.5, which owns the `gguf-cli` binary). The 2026-05-21 survey conflated them; the matrix now tracks `gguf-rs-lib` (the one with the CLI + write claim) and footnotes the zackshen crate.
- **No new direct competitor (2026-07-23 sweep).** A fresh search for a standalone multi-family Rust dequant library, a standalone Rust GGUF writer, or a multi-format Rust convert CLI found none. Closest adjacents: `gguf-rs-lib` (Rust, GGUF-only, read+write, 6★) and `ggufy` (Zig, not Rust). Neither overlaps anamnesis's multi-family-dequant or multi-format-convert shape.
- **compressed-tensors is context, not a matrix row (2026-07-23).** vLLM's `compressed-tensors` (306★) is the closest *conceptual* competitor — it standardises quantized weights on top of safetensors — but it is Python-only and vLLM-coupled, with decompression bound to the vLLM/PyTorch load path and no Rust API. It is recorded in the "Beyond Rust" section as ecosystem context. A future anamnesis reader for its config schema is a plausible coverage item, tracked but not shipped.
- **Adjacent CLI tools considered and excluded:**
  - `llmserve` ([github.com/AlexsJones/llmserve](https://github.com/AlexsJones/llmserve)) — v0.0.8 (2026-04-20), 280★. Rust TUI launcher for existing inference engines (llama-server, KoboldCpp, LocalAI, MLX, Ollama, vLLM, LM Studio) plus model-file discovery. Out of scope: does not parse, dequantise, or convert tensor formats — strictly discovery + launch. Different niche (find + serve) from anamnesis (parse + transform).
  - `ollama` ([github.com/ollama/ollama](https://github.com/ollama/ollama)) — Go, covered in the "Beyond Rust" footer above.
  - `hf-hub` ([crates.io/crates/hf-hub](https://crates.io/crates/hf-hub)) — v1.0.0-rc.1 (2026-05-07). Rust port of `huggingface_hub` for downloading + caching model files. Different layer: pre-anamnesis (you use `hf-hub` to fetch the `.safetensors` shard, then anamnesis transforms it). Complement, not competitor.
- **Phase 8.5 forward-looking column.** Already visible through the **GGUFWrite** column whose anamnesis cell reads "scalar dtypes (Phase 8.5: quantised)". A separate planned-feature column would be redundant once Phase 8.5 lands; the deferred `FP8`/`IQ`/`TQ`/`MXFP4` encode kernels will appear in **EncBnB** generalised to a multi-family **Enc-\*** column at that point. (The encode roadmap renumbered 7.5 → 8.5 when Phase 7 CPU-SIMD and Phase 8 PyO3 were slotted ahead of it.)
