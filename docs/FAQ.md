# Frequently Asked Questions

<!-- Last updated: 2026-08-15, anamnesis v0.7.4 -->

<!--
STYLE CONVENTIONS for editing this FAQ. Keep growth consistent.
(Adapted from the sibling hf-fetch-model FAQ so the two read alike.)

1. Tone: conversational, matching the project's README voice. Address the
   reader as "you". Prefer short paragraphs over bullet points.
2. Question format: "### How do I …?" or "### What is …?" as the heading.
   Use natural-language questions, because GitHub's anchor generator produces
   usable slugs from them. Keep them in the Contents list too.
3. Answer length: 2–4 sentences, plus at most one small code block with a
   concrete command. Anything longer is a tutorial, not an FAQ entry:
   promote it to docs/tutorials/ and link out instead.
4. Shell context: the project's primary shell is PowerShell on Windows
   (see CLAUDE.md). When showing env vars, give both variants side by
   side, PowerShell `$env:VAR="…"; amn …` first, then bash/zsh
   `VAR=… amn …`, so neither audience is left guessing.
5. "MSRV" is spelled out the first time as "Minimum Rust Version (MSRV)";
   the acronym is OK on reuse.
6. Freshness marker: update the "Last updated" date and version at the top
   whenever any answer text changes, but not for typo fixes or new entries
   that don't touch existing answers.
7. Scope: answer questions about features that actually ship today. Do not
   pre-document unshipped work (Python bindings, encode-side kernels):
   those get dedicated entries when they land. The one exception is a
   single forward-pointer under "Python" so users know it is coming.
8. Grouping: if a section grows past ~5 entries, consider splitting it. If
   an entry grows past ~6 sentences, promote it to docs/tutorials/.
-->

A living list of the questions we and our early users have actually run into. If your question is not here, please open an issue on [GitHub](https://github.com/mi-for-the-rust-of-us/anamnesis/issues); we add entries as real questions arrive.

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
  - [Can anamnesis read an `.npz` saved from a transposed array?](#can-anamnesis-read-an-npz-saved-from-a-transposed-array)
- [Dequantizing and converting](#dequantizing-and-converting)
  - [How do I dequantize a quantized model to BF16?](#how-do-i-dequantize-a-quantized-model-to-bf16)
  - [What does "Lethe took ~N B of precision" mean?](#what-does-lethe-took-n-b-of-precision-mean)
  - [How do I convert between formats?](#how-do-i-convert-between-formats)
  - [Can I control how many threads anamnesis uses?](#can-i-control-how-many-threads-anamnesis-uses)
  - [Can I cancel a long-running `remember` or `convert`?](#can-i-cancel-a-long-running-remember-or-convert)
  - [Why is the output `BF16` and not `float32`?](#why-is-the-output-bf16-and-not-float32)
  - [Which output dtype should I ask for?](#which-output-dtype-should-i-ask-for)
  - [Why is my `f32` output twice the size of the `bf16` one?](#why-is-my-f32-output-twice-the-size-of-the-bf16-one)
  - [Does asking for `f32` rewrite every tensor as `float32`?](#does-asking-for-f32-rewrite-every-tensor-as-float32)
  - [Is anamnesis still bit-exact against PyTorch at `f32`?](#is-anamnesis-still-bit-exact-against-pytorch-at-f32)
- [Parsing untrusted input](#parsing-untrusted-input)
  - [Is it safe to parse a model file from a stranger?](#is-it-safe-to-parse-a-model-file-from-a-stranger)
  - [How do I bound memory when parsing untrusted files?](#how-do-i-bound-memory-when-parsing-untrusted-files)
  - [Can a malformed file crash the process (panic or abort)?](#can-a-malformed-file-crash-the-process-panic-or-abort)
- [Python](#python)
  - [Is there a `pip install` for Python?](#is-there-a-pip-install-for-python)

## About anamnesis

### What is anamnesis? How is it different from the `safetensors` crate?

anamnesis is a framework-agnostic, pure-Rust library (plus a CLI) that *parses* tensor formats and *recovers precision* from quantized ones. The `safetensors` crate reads and writes the safetensors container but has no quantization awareness: it hands you the raw FP8/GPTQ/AWQ/NF4 bytes as-is. anamnesis builds on top: it detects the quantization scheme and dequantizes the weights back to `BF16` that any Rust ML framework can load, and it also reads `.gguf`, `.npz`, and PyTorch `.pth` files that `safetensors` does not touch.

### Why can't I just load a quantized model in candle or burn?

Because the Rust ML frameworks stop at the file-format boundary for quantized weights: `VarBuilder::from_mmaped_safetensors` fails on an FP8 tensor with `unsupported safetensor dtype F8_E4M3`, and there is no loader for GPTQ/AWQ packing or GGUF k-quants. anamnesis is the missing step: it turns a quantized file into a standard `BF16` safetensors file (or in-memory bytes) that candle, burn, or tch loads with no special support.

### Why are there two binary names, `anamnesis` and `amn`?

They are the same binary; `amn` is just a short alias for the people who type it many times a day. Use whichever you prefer; every example in the docs works with both.

### What do "remember", "forget", and "Lethe" mean?

They are the project's names for the two directions of precision change. **Remember** recovers precision (dequantize, the FP8/GPTQ/AWQ/BnB/GGUF → `BF16` path); **forget** (a.k.a. Lethe, after the river of forgetting) reduces it (quantize). The CLI subcommand is `amn remember` (alias `amn dequantize`); `amn inspect` reports how much precision "Lethe took" when a model was quantized.

### Is it stable? What does a `0.7.x` version mean?

`0.7.x` is pre-`1.0`: the format coverage and dequantization correctness are production-grade (bit-exact against each canonical library), but the public API may still evolve before `1.0`. The preceding `0.6.x` line centered on a security-hardening pass for untrusted input, capped by the convert-matrix completion in `0.6.9`. The `0.7.x` line is about throughput and API polish ahead of the Python bindings: `0.7.0` made whole-model dequantisation **multi-threaded** (~3–4×, byte-identical at any thread count), `0.7.1` added reader-generic full `GGUF` front matter, `0.7.2` brought the `GGUF` conversion path under that same thread pool (~1.9× on the reader stage) and exposed the budget as `--threads`, `0.7.3`/`0.7.4` made the dequantisation output dtype caller-chosen (`BF16`/`F32`/`F16`) across the `convert` and `remember` paths respectively, and `0.7.5` closed the last reader-generic gap with full `.pth` front matter — the direct counterpart to `0.7.1`'s `GGUF` one. Pin a version in `Cargo.toml` and read `CHANGELOG.md` before upgrading.

Note that `0.7.0` was originally planned as a CPU **SIMD** pass. It shipped as multi-threading instead: explicit SIMD was prototyped, measured, and *rejected*: a bit-exact hand-written AVX2 `f32x8_to_bf16x8` scored 1.02×, because the shared `f32 → BF16` writer is memory-bandwidth-bound rather than compute-bound. The measurements are in [`perf-experiments.md`](perf-experiments.md) (Experiments 10–11).

## Installation

### How do I install the CLI? What is the Minimum Rust Version?

Install from crates.io with the `cli` feature enabled:

```
cargo install anamnesis --features cli,pth,npz,gguf,bnb,awq,gptq
```

The Minimum Rust Version (MSRV) is **1.88**. The library itself (no CLI) is a normal `cargo add anamnesis` dependency.

### Which feature flags do I need?

`cli` builds the `anamnesis`/`amn` binaries; FP8 safetensors support is always on, but the other formats are feature-gated so you only compile what you use: `pth`, `npz`, `gguf`, `bnb`, `awq`, `gptq`, and `ollama` (adds the `ollama:` URL scheme, implies `gguf`). Enable the ones matching the files you handle, e.g. `--features cli,gguf` if you only work with GGUF.

## Formats and inspection

### Which file formats can anamnesis read?

`.safetensors` (including FP8 / GPTQ / AWQ / BitsAndBytes-quantized), `.gguf`, `.npz` (NumPy archives), and PyTorch `.pth` / `.pt` (via a minimal, allowlisted pickle VM). A `.bin` file is probed for ZIP/GGUF magic so PyTorch, GGUF, and safetensors payloads are distinguished automatically; you do not pass a format flag.

### What's the difference between `parse` and `inspect`?

`amn inspect` is a fast, header-only summary (format, tensor counts, dtypes, size estimate, byte order), and for `.gguf`/`.npz`/`.pth` it does not read the weight bodies at all. `amn parse` does the full parse and lists every tensor with its name, dtype, and shape. Reach for `inspect` first to decide whether a file is worth the full parse; this is also the recommended safety gate for untrusted input (walkthrough: [Inspect before you parse](tutorials/inspect-before-you-parse.md)).

### Can anamnesis read an `.npz` saved from a transposed array?

Yes, since **v0.7.6**. `np.savez("w.npz", w=x.T)` writes `fortran_order: True`, because NumPy records the memory order it finds and a transposed view is column-major. Earlier versions rejected such an archive outright with an `Unsupported` error — defensible when the audience was Rust consumers of C-order SAE archives, and a poor welcome for anyone whose file NumPy had written from an ordinary two-line script.

The array is now rewritten into row-major order on the way out, so `NpzTensor::data` keeps the row-major promise every downstream consumer relies on. That is deliberate rather than a flag on the tensor: `npz_to_safetensors`, the `convert` hub, and any framework loading the result all assume row-major, and an order flag a caller might ignore would be a silent-orientation bug — plausible numbers in the wrong places, not a crash. C-order archives, which is nearly all of them, pay nothing.

### How do I inspect an Ollama model without hunting for the blob path?

Build with the `ollama` feature and pass an `ollama:<model>:<tag>` URL, and anamnesis reads the Ollama manifest and resolves it to the local GGUF blob for you:

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

Run `amn remember` (alias `amn dequantize`) and point `-o` at the output file; it detects the scheme and writes standard `BF16` safetensors:

```
amn remember model-fp8.safetensors -o model-bf16.safetensors
```

The full walkthrough, including a GGUF example with real output, is in [Dequantize a GGUF model to BF16](tutorials/dequantize-a-gguf-model.md).

### What does "Lethe took ~N B of precision" mean?

It is the estimated number of bytes of precision that quantization (Lethe) discarded relative to the dequantized `BF16` size, a quick gauge of how lossy the source quantization was. A near-`~0 B` figure (as you sometimes see for NF4 against a small fixture) means the round-trip is essentially exact at that size; a large figure means the source threw away a lot. It is a reporting aid in `inspect`, not a correctness claim.

### How do I convert between formats?

`amn convert <file> --to <target>` routes every input through one in-memory **BF16 hub**, so **every input reaches every current target** (`safetensors`/`bf16`, `gguf`, `bnb-nf4`). A quantized input dequantizes automatically (the old "dequantize first, then re-run" two-hop is gone), and `gguf → gguf` recovers precision in place while preserving the source's metadata KV so the result stays loadable.

```
amn convert model.gguf --to bnb-nf4     # dequantize + re-encode, one command
```

Scalar dtypes are preserved (so `.pth → safetensors` and `NPZ`-`F32` → `GGUF` stay lossless); only quantized tensors become `BF16`. From the library, `convert_bytes` does the same thing without touching the filesystem (added in v0.7.6, for callers holding a download or crossing an `FFI` boundary); it detects the input format from magic bytes and returns the output bytes, byte-identical to what the file path writes. Stamp your own GGUF metadata with `--gguf-metadata <file.json>` / `--gguf-kv key=value` (anamnesis writes it verbatim). The full walkthrough with real output is in [Convert a model between formats](tutorials/convert-between-formats.md); the full matrix, the metadata grammar, and what stays out of scope until the encode kernels land are in the [CLI reference](cli-reference.md#amn-convert-file---to-target).

### Can I control how many threads anamnesis uses?

Yes: pass `--threads N` to `amn remember` or `amn convert`, or use `RememberOptions::new().with_threads(n)` / `ConvertOptions::new().with_threads(n)` from the library. The default is `min(cpu cores, 4)`: dequantization is memory-bandwidth bound and plateaus at roughly 3–4× by about four threads, so a bigger number mostly just takes cores away from whatever else you are running. Output is **byte-identical whatever you pass**, because the thread count is a performance knob and never a correctness variable, and below a 4 MiB input floor the sequential path runs regardless, because spawning a pool costs more than the work saves.

```
amn convert model-Q4_K_M.gguf --to safetensors --threads 8
```

### Can I cancel a long-running `remember` or `convert`?

Yes, from the library, since **v0.7.6**. Attach a `CancelToken` through `RememberOptions::with_cancel` or `ConvertOptions::with_cancel`, keep a clone, and call `cancel()` from any thread — a signal handler, a watchdog, a request-timeout task:

```rust
use anamnesis::{CancelToken, RememberOptions, TargetDtype, parse};

let token = CancelToken::new();
let worker = token.clone();
// ... hand `worker` to whatever decides to stop the run ...

let model = parse("model-fp8.safetensors")?;
model.remember_with_options(
    "out.safetensors",
    TargetDtype::BF16,
    RememberOptions::new().with_cancel(token),
)?;
```

The run stops at the next tensor boundary and returns `AnamnesisError::Cancelled` — a variant of its own, so a host can tell a user-initiated abort from a bad file. **No output file is written**: every path builds its result in memory before serialising, so the check lands before any byte reaches the filesystem and there is nothing to clean up. Cancellation is cooperative, so a worker already inside a tensor finishes that tensor; the bound is one tensor, not the whole model. A token you never cancel costs nothing and changes no output byte.

The CLI does not expose this yet — `Ctrl-C` on `amn` is the usual process signal. The token exists for embedders, and for the v0.8.0 Python bindings, where releasing the GIL around a long call would otherwise make `KeyboardInterrupt` undeliverable until the call returned.

### Why is the output `BF16` and not `float32`?

It does not have to be. `amn convert model.gguf --to safetensors --out-dtype f32` emits `float32`, and `--out-dtype f16` emits IEEE half. `bf16` remains the default, so nothing changes unless you ask.

`BF16` is the dtype the safetensors / Hugging Face ecosystem serves weights in, and at 2 bytes per element it halves the memory traffic on a path that is bandwidth-bound end to end. It is, though, lossy against the *exact* dequantized value: a `Q8_0` value is an `f16` scale times an `int8`, needing up to ~18 bits of significand where `BF16` holds 8. Measured on `SmolLM2-135M-Q4_K_M`, only 3–20 % of values land exactly on a `BF16` grid point and the rest round by at most half a ULP (≈ 0.39 % relative). That also scopes the project's "bit-exact, 0 ULP" claim precisely: it is 0 ULP against the reference **rounded to `BF16`**, which is how every cross-validation fixture is built, not against the true value, for which you need `float32`.

`--out-dtype f32` is the option that removes anamnesis's own narrowing step entirely, so the value you get is the `f32` that `gguf-py` itself produces. Expect it to be *slower* than `bf16`, not faster: it doubles the output bytes on a path that is bandwidth-bound, which is the honest cost of the precision rather than a defect.

`f16` is not simply "the better 2-byte option". It buys 3 significand bits over `bf16` (11 versus 8) and pays a far narrower exponent range: `bf16` shares `f32`'s range, while `f16` overflows to infinity above 65504 and flushes to zero below about `2⁻²⁴`. anamnesis follows plain IEEE semantics there rather than saturating, so its output matches what NumPy and PyTorch produce for the same conversion.

Since v0.7.4 the `remember` path offers the same choice, spelled `--to`:

```
amn remember model-fp8.safetensors --to f32
```

Until then `remember` was `bf16`-only, because its four kernel families (`FP8`, `GPTQ`, `AWQ`, `BnB`) fused the narrowing step into their inner loops. Those loops are now generic over the output width, so every format anamnesis reads can be dequantized at any of the three.

### Which output dtype should I ask for?

Stay on `bf16` unless you have a specific reason not to. It is what the safetensors and Hugging Face ecosystem serves weights in, it is what candle, burn, and tch expect, and at 2 bytes per element it keeps memory traffic down on a path that is bandwidth-bound.

Ask for `f32` when you want the reference value itself rather than a rounded copy of it: cross-validating against PyTorch, debugging a numerical discrepancy, or feeding a downstream pipeline that computes in `float32` anyway. It costs double the output bytes and runs slower, which is the price of the precision.

Ask for `f16` only when a consumer specifically requires IEEE half and you know your values fit inside its range. It is not the better 2-byte option by default, for the reasons in the entry above.

The longer version, with the decision written out and the numbers behind it, is in [Choosing an output dtype](tutorials/choosing-an-output-dtype.md).

### Why is my `f32` output twice the size of the `bf16` one?

Because `float32` is 4 bytes per element and `bfloat16` is 2. The doubling is the dtype, not overhead: nothing is being padded or duplicated.

Only the tensors anamnesis **dequantizes** double. Passthrough tensors keep their source dtype and their exact bytes, so a real model grows by somewhat less than 2x overall, depending on how much of it was quantized in the first place.

Expect it to be slower as well as bigger. These kernels are bandwidth-bound, so writing twice the bytes costs roughly what you would guess. If you want the size before you commit to the run, ask for it at the width you actually intend — `amn inspect --to f32` from the command line, or `InspectOptions` from the library:

```rust
use anamnesis::{InspectOptions, TargetDtype, parse};

let model = parse("model-fp8.safetensors")?;
let info = model.inspect_with_options(
    &InspectOptions::new().with_output_dtype(TargetDtype::F32),
);
println!("{}", info.dequantized_size);
```

*(`InspectOptions` is taken by reference since v0.7.6, when it gained a `limits` field and stopped being `Copy`. Upgrading from v0.7.4? Add the `&`.)*

### Does asking for `f32` rewrite every tensor as `float32`?

No. `--to` on `remember` and `--out-dtype` on `convert` govern the tensors anamnesis **dequantizes**, and nothing else.

Both commands have always produced mixed-dtype files: dequantized tensors came out `BF16`, while passthrough tensors (norms, biases, embeddings, anything not block-quantized) kept whatever dtype the source held. Asking for a wider output changes the first group only. An `F16` norm in the source is still an `F16` norm in the output, and an `F32` tensor stays byte-identical.

That is deliberate. A passthrough tensor is copied, never decoded, so widening it would invent precision that was never in the file while doubling its size. If you want a single-dtype file, what you want is a cast pass, which is a different operation from dequantization.

### Is anamnesis still bit-exact against PyTorch at `f32`?

Yes, and as of v0.7.4 that is tested rather than assumed. Every kernel family is cross-validated at full `f32` width against the canonical library's own output, compared bit for bit with no tolerance: `FP8` against PyTorch, `GPTQ` against GPTQModel, `AWQ` against AutoAWQ, `BnB` against bitsandbytes, and `GGUF` against `gguf-py`.

This mattered more than it sounds, because exactness at `BF16` never implied exactness at `f32`. Rounding the reference to `BF16` before comparing discards 16 mantissa bits, and in these fixtures 38 to 98 percent of values carry bits `BF16` cannot represent, most families sitting above 77 percent. The comparison was throwing away most of the available signal.

Widening it found a real defect. The `BnB` `INT8` kernel computed `w * (SCB / 127)` where bitsandbytes computes `(w * SCB) * (1 / 127)`. Those are the same real number and the same `BF16`, but they differ by 1 ULP on 26.9 percent of elements at `f32`. Five releases of `BF16` cross-validation had reported 0 mismatches. v0.7.4 matches the canonical association, so the kernel is now exact at every output width.

## Parsing untrusted input

### Is it safe to parse a model file from a stranger?

A tensor archive is attacker-controllable, so anamnesis treats every parser entry point as a hardened boundary: checked arithmetic on header-derived sizes, allocation caps before any `vec!`, a strict allowlist in the `.pth` pickle VM (it never invokes Python callables), and a vendored read-only ZIP reader. The recommended pattern is **inspect → check against your policy → parse**: run the cheap `amn inspect` (or the reader-based `inspect_*_from_reader` library calls) first and only commit to a full parse if the declared sizes look sane. Since **v0.7.6** that first call takes your budget too — the `_with_options` forms accept an `InspectOptions` carrying a `ParseLimits` — so the call you are told to make *first* is one you can tighten, which it was not before. Step-by-step walkthrough: [Inspect before you parse](tutorials/inspect-before-you-parse.md); the README's "Parsing untrusted input" section has the full policy.

### How do I bound memory when parsing untrusted files?

The library API takes a caller-supplied `ParseLimits` budget (max single allocation, max aggregate declared bytes, max item count, max decompression ratio) threaded through every `parse_*_with_limits` entry point — and, since **v0.7.6**, through the `inspect_*_with_options` and `detect_format_from_bytes_with_limits` entry points as well — and enforced fail-fast *before* allocation. `ParseLimits::default()` is permissive (today's behaviour); tighten it to your environment (a memory-constrained edge board sets MB-scale ceilings, a multi-tenant worker sets per-slot ceilings), and a hostile declaration is rejected with a clean `AnamnesisError::LimitExceeded` (carrying the breached limit's name) instead of an OOM. Note that the **always-on permanent per-format caps** (the 100 MiB safetensors header, `MAX_PKL_SIZE`, the `GGUF` counts, …) already return `LimitExceeded` *even under the default budget*; tightening only lowers the thresholds, it is not what makes `LimitExceeded` reachable. A malformed file is `Parse`, and a `.pth` referencing a non-`torch.*` pickle global is `DisallowedGlobal`, so a host can branch on the error *kind*, not the message.

### Can a malformed file crash the process (panic or abort)?

No. No public parse/inspect entry point panics or aborts on any input: a malformed, truncated, or hostile file is always a clean `Result::Err`, never an unwinding panic and never a `SIGBUS` (the copy-based `parse_bytes` / `parse_*_from_reader` paths use no memory map). It's enforced in the source (the `unwrap`/`expect`/`panic`/indexing lints are denied crate-wide, and every header-derived size uses checked arithmetic) and pinned in CI by `tests/no_panic.rs` plus the `cargo fuzz` harness. Library/CLI release builds abort on panic (fail-closed); the future Python wheel is built to *unwind* so even an unexpected panic becomes a catchable `PanicException` rather than a dead worker.

## Python

### Is there a `pip install` for Python?

Not yet. Python bindings (PyO3) are planned for **v0.8.0** ([Phase 8](../ROADMAP.md#phase-8-python-bindings-pyo3) on the roadmap), after the throughput work in the `0.7.x` line so the published wheels actually deliver the advertised speed. That ordering is why `0.7.0` ships **multi-threading** rather than the SIMD pass originally planned: threads work regardless of the wheel's `target-cpu`, whereas compile-time SIMD would have been left on the table by any generic wheel a user `pip install`s. The package will be **`anamnesis-quant`** rather than `anamnesis`: the bare name is already taken on PyPI by an unrelated project, so the distribution name differs from the Rust crate, which is unchanged. Which output dtypes are offered is **no longer open** — v0.7.4 settled it on both paths, so a Python caller will be able to ask for `bf16`, `f32` or `f16` and get a native NumPy array for the latter two with no optional dependency. When the bindings land, this FAQ gains a Python section (installation, the exception hierarchy, and how the returned arrays map onto `NumPy` dtypes). Until then, use the CLI or the Rust library.
