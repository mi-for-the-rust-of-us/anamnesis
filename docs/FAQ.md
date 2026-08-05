# Frequently Asked Questions

<!-- Last updated: 2026-07-22, anamnesis v0.6.9 -->

<!--
STYLE CONVENTIONS for editing this FAQ — keep growth consistent.
(Adapted from the sibling hf-fetch-model FAQ so the two read alike.)

1. Tone: conversational, matching the project's README voice. Address the
   reader as "you". Prefer short paragraphs over bullet points.
2. Question format: "### How do I …?" or "### What is …?" as the heading.
   Use natural-language questions — GitHub's anchor generator produces
   usable slugs from them. Keep them in the Contents list too.
3. Answer length: 2–4 sentences, plus at most one small code block with a
   concrete command. Anything longer is a tutorial, not an FAQ entry —
   promote it to docs/tutorials/ and link out instead.
4. Shell context: the project's primary shell is PowerShell on Windows
   (see CLAUDE.md). When showing env vars, give both variants side by
   side — PowerShell `$env:VAR="…"; amn …` first, then bash/zsh
   `VAR=… amn …` — so neither audience is left guessing.
5. "MSRV" is spelled out the first time as "Minimum Rust Version (MSRV)";
   the acronym is OK on reuse.
6. Freshness marker: update the "Last updated" date and version at the top
   whenever any answer text changes — not for typo fixes or new entries
   that don't touch existing answers.
7. Scope: answer questions about features that actually ship today. Do not
   pre-document unshipped work (Python bindings, encode-side kernels) —
   those get dedicated entries when they land. The one exception is a
   single forward-pointer under "Python" so users know it is coming.
8. Grouping: if a section grows past ~5 entries, consider splitting it. If
   an entry grows past ~6 sentences, promote it to docs/tutorials/.
-->

A living list of the questions we and our early users have actually run into. If your question is not here, please open an issue on [GitHub](https://github.com/PCfVW/anamnesis/issues) — we add entries as real questions arrive.

## Contents

- [About anamnesis](#about-anamnesis)
  - [What is anamnesis? How is it different from the `safetensors` crate?](#what-is-anamnesis-how-is-it-different-from-the-safetensors-crate)
  - [Why can't I just load a quantized model in candle or burn?](#why-cant-i-just-load-a-quantized-model-in-candle-or-burn)
  - [Why are there two binary names, `anamnesis` and `amn`?](#why-are-there-two-binary-names-anamnesis-and-amn)
  - [What do "remember", "forget", and "Lethe" mean?](#what-do-remember-forget-and-lethe-mean)
  - [Is it stable? What does a `0.7.x` version mean?](#is-it-stable-what-does-a-07x-version-mean)
- [Installation](#installation)
  - [How do I install the CLI? What is the Minimum Rust Version?](#how-do-i-install-the-cli-what-is-the-minimum-rust-version)
  - [Which feature flags do I need?](#which-feature-flags-do-i-need)
- [Formats and inspection](#formats-and-inspection)
  - [Which file formats can anamnesis read?](#which-file-formats-can-anamnesis-read)
  - [What's the difference between `parse` and `inspect`?](#whats-the-difference-between-parse-and-inspect)
  - [How do I inspect an Ollama model without hunting for the blob path?](#how-do-i-inspect-an-ollama-model-without-hunting-for-the-blob-path)
- [Dequantizing and converting](#dequantizing-and-converting)
  - [How do I dequantize a quantized model to BF16?](#how-do-i-dequantize-a-quantized-model-to-bf16)
  - [What does "Lethe took ~N B of precision" mean?](#what-does-lethe-took-n-b-of-precision-mean)
  - [How do I convert between formats?](#how-do-i-convert-between-formats)
- [Parsing untrusted input](#parsing-untrusted-input)
  - [Is it safe to parse a model file from a stranger?](#is-it-safe-to-parse-a-model-file-from-a-stranger)
  - [How do I bound memory when parsing untrusted files?](#how-do-i-bound-memory-when-parsing-untrusted-files)
  - [Can a malformed file crash the process (panic or abort)?](#can-a-malformed-file-crash-the-process-panic-or-abort)
- [Python](#python)
  - [Is there a `pip install anamnesis`?](#is-there-a-pip-install-anamnesis)

## About anamnesis

### What is anamnesis? How is it different from the `safetensors` crate?

anamnesis is a framework-agnostic, pure-Rust library (plus a CLI) that *parses* tensor formats and *recovers precision* from quantized ones. The `safetensors` crate reads and writes the safetensors container but has no quantization awareness — it hands you the raw FP8/GPTQ/AWQ/NF4 bytes as-is. anamnesis builds on top: it detects the quantization scheme and dequantizes the weights back to `BF16` that any Rust ML framework can load, and it also reads `.gguf`, `.npz`, and PyTorch `.pth` files that `safetensors` does not touch.

### Why can't I just load a quantized model in candle or burn?

Because the Rust ML frameworks stop at the file-format boundary for quantized weights — `VarBuilder::from_mmaped_safetensors` fails on an FP8 tensor with `unsupported safetensor dtype F8_E4M3`, and there is no loader for GPTQ/AWQ packing or GGUF k-quants. anamnesis is the missing step: it turns a quantized file into a standard `BF16` safetensors file (or in-memory bytes) that candle, burn, or tch loads with no special support.

### Why are there two binary names, `anamnesis` and `amn`?

They are the same binary — `amn` is just a short alias for the people who type it many times a day. Use whichever you prefer; every example in the docs works with both.

### What do "remember", "forget", and "Lethe" mean?

They are the project's names for the two directions of precision change. **Remember** recovers precision (dequantize — the FP8/GPTQ/AWQ/BnB/GGUF → `BF16` path); **forget** (a.k.a. Lethe, after the river of forgetting) reduces it (quantize). The CLI subcommand is `amn remember` (alias `amn dequantize`); `amn inspect` reports how much precision "Lethe took" when a model was quantized.

### Is it stable? What does a `0.7.x` version mean?

`0.7.x` is pre-`1.0`: the format coverage and dequantization correctness are production-grade (bit-exact against each canonical library), but the public API may still evolve before `1.0`. The preceding `0.6.x` line centered on a security-hardening pass for untrusted input, capped by the convert-matrix completion in `0.6.9`. The `0.7.x` line is about throughput and API polish ahead of the Python bindings: `0.7.0` made whole-model dequantisation **multi-threaded** (~3–4×, byte-identical at any thread count), and `0.7.1` added reader-generic full `GGUF` front matter. Pin a version in `Cargo.toml` and read `CHANGELOG.md` before upgrading.

Note that `0.7.0` was originally planned as a CPU **SIMD** pass. It shipped as multi-threading instead: explicit SIMD was prototyped, measured, and *rejected* — a bit-exact hand-written AVX2 `f32x8_to_bf16x8` scored 1.02×, because the shared `f32 → BF16` writer is memory-bandwidth-bound rather than compute-bound. The measurements are in [`perf-experiments.md`](perf-experiments.md) (Experiments 10–11).

## Installation

### How do I install the CLI? What is the Minimum Rust Version?

Install from crates.io with the `cli` feature enabled:

```
cargo install anamnesis --features cli,pth,npz,gguf,bnb,awq,gptq
```

The Minimum Rust Version (MSRV) is **1.88**. The library itself (no CLI) is a normal `cargo add anamnesis` dependency.

### Which feature flags do I need?

`cli` builds the `anamnesis`/`amn` binaries; FP8 safetensors support is always on, but the other formats are feature-gated so you only compile what you use: `pth`, `npz`, `gguf`, `bnb`, `awq`, `gptq`, and `ollama` (adds the `ollama:` URL scheme, implies `gguf`). Enable the ones matching the files you handle — e.g. `--features cli,gguf` if you only work with GGUF.

## Formats and inspection

### Which file formats can anamnesis read?

`.safetensors` (including FP8 / GPTQ / AWQ / BitsAndBytes-quantized), `.gguf`, `.npz` (NumPy archives), and PyTorch `.pth` / `.pt` (via a minimal, allowlisted pickle VM). A `.bin` file is probed for ZIP/GGUF magic so PyTorch, GGUF, and safetensors payloads are distinguished automatically — you do not pass a format flag.

### What's the difference between `parse` and `inspect`?

`amn inspect` is a fast, header-only summary — format, tensor counts, dtypes, size estimate, byte order — and for `.gguf`/`.npz`/`.pth` it does not read the weight bodies at all. `amn parse` does the full parse and lists every tensor with its name, dtype, and shape. Reach for `inspect` first to decide whether a file is worth the full parse — this is also the recommended safety gate for untrusted input (walkthrough: [Inspect before you parse](tutorials/inspect-before-you-parse.md)).

### How do I inspect an Ollama model without hunting for the blob path?

Build with the `ollama` feature and pass an `ollama:<model>:<tag>` URL — anamnesis reads the Ollama manifest and resolves it to the local GGUF blob for you:

```
amn inspect ollama:llama3.2:1b
```

If your Ollama cache is in a non-default location, point it there with the `OLLAMA_MODELS` environment variable:

```
# PowerShell
$env:OLLAMA_MODELS="D:\ollama\models"; amn inspect ollama:llama3.2:1b

# bash / zsh
OLLAMA_MODELS=/data/ollama/models amn inspect ollama:llama3.2:1b
```

## Dequantizing and converting

### How do I dequantize a quantized model to BF16?

Run `amn remember` (alias `amn dequantize`) and point `-o` at the output file — it detects the scheme and writes standard `BF16` safetensors:

```
amn remember model-fp8.safetensors -o model-bf16.safetensors
```

The full walkthrough, including a GGUF example with real output, is in [Dequantize a GGUF model to BF16](tutorials/dequantize-a-gguf-model.md).

### What does "Lethe took ~N B of precision" mean?

It is the estimated number of bytes of precision that quantization (Lethe) discarded relative to the dequantized `BF16` size — a quick gauge of how lossy the source quantization was. A near-`~0 B` figure (as you sometimes see for NF4 against a small fixture) means the round-trip is essentially exact at that size; a large figure means the source threw away a lot. It is a reporting aid in `inspect`, not a correctness claim.

### How do I convert between formats?

`amn convert <file> --to <target>` routes every input through one in-memory **BF16 hub**, so **every input reaches every current target** (`safetensors`/`bf16`, `gguf`, `bnb-nf4`). A quantized input dequantizes automatically — the old "dequantize first, then re-run" two-hop is gone — and `gguf → gguf` recovers precision in place while preserving the source's metadata KV so the result stays loadable.

```
amn convert model.gguf --to bnb-nf4     # dequantize + re-encode, one command
```

Scalar dtypes are preserved (so `.pth → safetensors` and `NPZ`-`F32` → `GGUF` stay lossless); only quantized tensors become `BF16`. Stamp your own GGUF metadata with `--gguf-metadata <file.json>` / `--gguf-kv key=value` (anamnesis writes it verbatim). The full walkthrough with real output is in [Convert a model between formats](tutorials/convert-between-formats.md); the full matrix, the metadata grammar, and what stays out of scope until the encode kernels land are in the [CLI reference](cli-reference.md#amn-convert-file---to-target).

## Parsing untrusted input

### Is it safe to parse a model file from a stranger?

A tensor archive is attacker-controllable, so anamnesis treats every parser entry point as a hardened boundary: checked arithmetic on header-derived sizes, allocation caps before any `vec!`, a strict allowlist in the `.pth` pickle VM (it never invokes Python callables), and a vendored read-only ZIP reader. The recommended pattern is **inspect → check against your policy → parse** — run the cheap `amn inspect` (or the reader-based `inspect_*_from_reader` library calls) first and only commit to a full parse if the declared sizes look sane. Step-by-step walkthrough: [Inspect before you parse](tutorials/inspect-before-you-parse.md); the README's "Parsing untrusted input" section has the full policy.

### How do I bound memory when parsing untrusted files?

The library API takes a caller-supplied `ParseLimits` budget (max single allocation, max aggregate declared bytes, max item count, max decompression ratio) threaded through every `parse_*_with_limits` entry point and enforced fail-fast *before* allocation. `ParseLimits::default()` is permissive (today's behaviour); tighten it to your environment — a memory-constrained edge board sets MB-scale ceilings, a multi-tenant worker sets per-slot ceilings — and a hostile declaration is rejected with a clean `AnamnesisError::LimitExceeded` (carrying the breached limit's name) instead of an OOM. Note that the **always-on permanent per-format caps** (the 100 MiB safetensors header, `MAX_PKL_SIZE`, the `GGUF` counts, …) already return `LimitExceeded` *even under the default budget* — tightening only lowers the thresholds, it is not what makes `LimitExceeded` reachable. A malformed file is `Parse`, and a `.pth` referencing a non-`torch.*` pickle global is `DisallowedGlobal` — so a host can branch on the error *kind*, not the message.

### Can a malformed file crash the process (panic or abort)?

No. No public parse/inspect entry point panics or aborts on any input — a malformed, truncated, or hostile file is always a clean `Result::Err`, never an unwinding panic and never a `SIGBUS` (the copy-based `parse_bytes` / `parse_*_from_reader` paths use no memory map). It's enforced in the source (the `unwrap`/`expect`/`panic`/indexing lints are denied crate-wide, and every header-derived size uses checked arithmetic) and pinned in CI by `tests/no_panic.rs` plus the `cargo fuzz` harness. Library/CLI release builds abort on panic (fail-closed); the future Python wheel is built to *unwind* so even an unexpected panic becomes a catchable `PanicException` rather than a dead worker.

## Python

### Is there a `pip install anamnesis`?

Not yet — Python bindings (PyO3) are planned for **v0.8.0** ([Phase 8](../ROADMAP.md#phase-8-python-bindings-pyo3) on the roadmap), after the throughput work in the `0.7.x` line so the published wheels actually deliver the advertised speed. That ordering is why `0.7.0` ships **multi-threading** rather than the SIMD pass originally planned: threads work regardless of the wheel's `target-cpu`, whereas compile-time SIMD would have been left on the table by any generic wheel a user `pip install`s. When the bindings land, this FAQ gains a Python section (installation, the exception hierarchy, and how to get a `bf16` NumPy array). Until then, use the CLI or the Rust library.
