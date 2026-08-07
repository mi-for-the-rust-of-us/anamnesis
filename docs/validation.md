# Validation & tested models

<!-- Last updated: 2026-06-25, anamnesis v0.6.8 -->

The evidence behind anamnesis's correctness, performance, and robustness claims —
the per-scheme cross-validation tables, the conversion / parsing benchmarks, the
peak-heap assertions, and the untrusted-input hardening timeline. The
[README](../README.md) carries the summary; this is the detail.

> **Validation provenance (v0.6.4):** "Cross-validated" below means the expected
> output comes from the **canonical library's own code** — `bitsandbytes`
> (`dequantize_4bit` / `int8_vectorwise_dequant`), AutoAWQ (`unpack_awq` +
> `reverse_awq_order`), GPTQModel (`dequantize_weight` + its v1→v2 zero-point
> conversion), PyTorch's native fp8 cast, and `gguf-py`'s `dequantize` — never a
> hand-rolled reimplementation of the formula. v0.6.4 adopted this rule after
> dogfooding in candle-mi exposed a circularly-validated nibble-order bug (full
> post-mortem: `docs/dogfooding-feedbacks/bnb-nibble-order-and-circular-fixture-validation.md`,
> internal, not tracked in the repo) and fixed three real defects (BnB nibble
> order, BnB double-quant `nested_offset`, AWQ GEMM interleave). The "vs PyTorch"
> speed columns are historical CPU-PyTorch timings (pre-v0.6.4 baselines); they
> document throughput, not correctness.

## Contents

- [Dequantization](#dequantization) — FP8 · GPTQ · AWQ · BitsAndBytes · GGUF block-quant
- [Quantization (Lethe)](#quantization-lethe--phase-5)
- [Format conversion pipeline](#format-conversion-pipeline-phase-6-v060)
- [Whole-model throughput](#whole-model-throughput-phase-7--72) — thread scaling · vs `gguf-py`
- [Parsing & inspection](#parsing--inspection)
- [Performance & peak-heap validation](#performance--peak-heap-validation-phase-65)
- [Robustness hardening timeline](#robustness-hardening-timeline)

## Dequantization

### FP8

Cross-validated against PyTorch's native fp8→f32 cast on 7 real FP8 models from 5 quantization tools. Bit-exact output (0 ULP difference). Auto-vectorized: SSE2 on any x86-64, AVX2 with `target-cpu=native`.

| Model | Quantizer | Scheme | Scales | vs PyTorch (AVX2) |
|---|---|---|---|---|
| EXAONE-4.0-1.2B-FP8 | LG AI | Fine-grained | BF16 | 6.0x faster |
| Qwen3-1.7B-FP8 | Qwen | Fine-grained | BF16 | 3.9x faster |
| Qwen3-4B-Instruct-2507-FP8 | Qwen | Fine-grained | F16 | 3.0x faster |
| Ministral-3-3B-Instruct-2512 | Mistral | Per-tensor | BF16 | 9.7x faster |
| Llama-3.2-1B-Instruct-FP8 | RedHat | Per-tensor | BF16 | 3.9x faster |
| Llama-3.2-1B-Instruct-FP8-dynamic | RedHat | Per-channel | BF16 | 2.7x faster |
| Llama-3.1-8B-Instruct-FP8 | NVIDIA | Per-tensor | F32 | 6.3x faster |

### GPTQ

Cross-validated against GPTQModel's own dequant pipeline (`TorchLinear.dequantize_weight` + its loader-side v1→v2 zero-point conversion) on 4 real GPTQ models from 2 quantizers (AutoGPTQ, GPTQModel). Bit-exact output (0 ULP difference). Loop fission for full AVX2 vectorization. `remember` emits the standard `nn.Linear` `[out_features, in_features]` orientation (transposed from the GEMM-native packed layout, the same `.T` GPTQModel's `dequantize_model` applies) — pinned by the orientation contract tests in `tests/remember_orientation.rs` (v0.6.5).

| Model | Quantizer | Bits | vs PyTorch (AVX2) |
|---|---|---|---|
| Falcon3-1B-Instruct-GPTQ-Int4 | AutoGPTQ | 4 | 6.5x faster |
| Llama-3.2-1B-Instruct-GPTQ | AutoGPTQ | 4 | 12.2x faster |
| Falcon3-1B-Instruct-GPTQ-Int8 | AutoGPTQ | 8 | 7.0x faster |
| Llama-3.2-1B-gptqmodel-8bit | GPTQModel | 8 | 7.9x faster |

### AWQ

Cross-validated against AutoAWQ's own unpack + reorder (`packing_utils.unpack_awq` + `reverse_awq_order`, including the GEMM nibble interleave `[0, 2, 4, 6, 1, 3, 5, 7]`) on 2 real AWQ models. Bit-exact output (0 ULP difference). 4-bit only — AutoAWQ's GEMM format defines no other width, so anamnesis rejects the rest rather than guess an interleave. Loop fission for full AVX2 vectorization. `remember` emits the standard `nn.Linear` `[out_features, in_features]` orientation (transposed from the GEMM-native packed layout) — pinned by the orientation contract tests in `tests/remember_orientation.rs` (v0.6.5).

| Model | Quantizer | Bits | vs PyTorch (AVX2) |
|---|---|---|---|
| llama-3.2-1b-instruct-awq | AutoAWQ | 4 | 5.7x faster |
| Falcon3-1B-Instruct-AWQ | AutoAWQ | 4 | 4.7x faster |

### BitsAndBytes

Cross-validated against real `bitsandbytes` (`functional.dequantize_4bit` / `int8_vectorwise_dequant`) on 7 real BitsAndBytes models across 4 architecture families (NF4, FP4, double-quant incl. the `nested_offset` absmax recovery, INT8). Bit-exact output (0 ULP difference). Loop fission for AVX2 on NF4/FP4; single-pass AVX2 on INT8 (`vpmovsxbd` → `vcvtdq2ps` → `vmulps`).

| Model | Format | Elements | vs PyTorch (AVX2) |
|---|---|---|---|
| Llama-3.2-1B-Instruct-bnb-nf4 | NF4 | 4,096 | 21.8x faster |
| Llama-3.2-1B-BNB-FP4 | FP4 | 4,096 | 18.0x faster |
| Llama-3.2-1B-Instruct-bnb-nf4-double-quant | NF4 double-quant | 4,096 | 54.0x faster |
| Llama-3.2-1B-BNB-INT8 | INT8 | 65,536 | 1.2x faster |

> **Note:** INT8 speedup is modest because the operation is trivially simple (`i8→f32→multiply`). Both PyTorch and anamnesis are near memory bandwidth limits at ~0.7–0.8 ns/element. The AVX2 hot loop is fully vectorized — the 1.2× reflects the inherent ceiling, not a missed optimization.

### GGUF block-quant

Cross-validated against the `gguf` Python package (`ggml-org` reference, mirrors `ggml-quants.c`) on **22 block-quant kernels** from 4 real models (bartowski SmolLM2-135M-Instruct, TheBloke TinyLlama-1.1B-Chat, bartowski Mistral-7B-Instruct-v0.3, bartowski Qwen2.5-0.5B-Instruct) plus 3 synthetic fixtures (`TQ1_0` / `TQ2_0` / `MXFP4` — only ~15 BitNet-derivative GGUFs ship the `TQ*` types on HuggingFace, and mainstream `MXFP4` only ships inside the 11 GB `gpt-oss-20b` upload, so a deterministic random tensor is the practical fixture source). Bit-exact output (0 ULP difference). **All 22 of 22 GGUF block types supported** — Phase 4.5 closed in step 6 (MXFP4). Feature-gated behind `gguf`.

| Kernel | Model | vs `gguf` Python (AVX2) |
|---|---|---|
| Q4_0 | SmolLM2-135M | 6.9x faster |
| Q4_1 | SmolLM2-135M | 6.3x faster |
| Q5_0 | TinyLlama-1.1B | 31.3x faster |
| Q5_1 | SmolLM2-135M | 11.4x faster |
| Q8_0 | SmolLM2-135M | 6.3x faster |
| IQ4_NL | SmolLM2-135M | 12.2x faster |
| Q2_K | TinyLlama-1.1B | 6.7x faster |
| Q3_K | SmolLM2-135M | 10.9x faster |
| Q4_K | SmolLM2-135M | 8.1x faster |
| Q5_K | SmolLM2-135M | 11.6x faster |
| Q6_K | SmolLM2-135M | 26.6x faster |
| IQ4_XS | SmolLM2-135M | 12.6x faster |
| IQ2_XXS | Mistral-7B-v0.3 | 3.45x faster |
| IQ2_XS | Mistral-7B-v0.3 | 2.84x faster |
| IQ2_S | Qwen2.5-0.5B | 4.10x faster |
| IQ3_XXS | Mistral-7B-v0.3 | 3.32x faster |
| IQ3_S | Mistral-7B-v0.3 | 4.37x faster |
| IQ1_S | Mistral-7B-v0.3 | 15.00x faster |
| IQ1_M | Mistral-7B-v0.3 | 7.85x faster |
| TQ1_0 | synthetic | 35.59x faster |
| TQ2_0 | synthetic | 26.31x faster |
| MXFP4 | synthetic | 30.14x faster |

> **Note:** `Q8_1` and `Q8_K` are internal `llama.cpp` activation quant types, not shipped as model weights — they are covered by unit tests only. Speedup measured on 65,536 elements (release build, `target-cpu=native`, best-of-5 per kernel). The `IQ2_*` and `IQ3_*` kernels land in the 2.8×–4.4× range rather than the 6×–31× range of the pure-arithmetic `Q*` kernels because their pass 1 involves a codebook LUT gather and a per-element sign branch — neither of which the auto-vectoriser can eliminate. The `IQ1_*` kernels are notably faster (7.9×–15.0×) because their inner loop replaces the per-element sign branch with a single scalar `±delta` per 8-element group, and the codebook gather is a plain `[u64; 2048]` table lookup. The ternary `TQ*` kernels are the **fastest in the crate** (26×–36×) — no codebook lookup at all, just bit shifts (`TQ2_0`) or a base-3 multiplication trick (`TQ1_0`) decoding directly to `{-d, 0, +d}`. `MXFP4` lands at 30× — structurally identical to `IQ4_NL` (12.2×) but with a tighter 17 B/block layout (1 B `E8M0` exponent vs 2 B `f16`) and a smaller codebook (16 entries × 4-bit nibble lookup) that the auto-vectoriser handles cleanly. Phase 7 (CPU SIMD pass) will further address the IQ2/IQ3 case with hand-written AVX2 intrinsics.

> **Limitations (peak heap):** Whole-model dequantisation via `ParsedModel::remember` or `amn remember model.gguf -o out.safetensors` retains every dequantised tensor in heap memory simultaneously until the underlying `safetensors::serialize_to_file` call returns. Peak heap is `O(total_BF16_output_size)` ≈ `2 × n_parameters` bytes — comfortable for **≤7 B** models on a 32 GB system, **tight at 13 B**, **OOMs at 70 B+**. The single-tensor kernel `dequantize_gguf_blocks_to_bf16` is already streaming (O(one block)); the orchestrator-level streaming output path is planned for Phase 10 — see [ROADMAP.md](../ROADMAP.md).

## Quantization (Lethe — Phase 5)

The inverse direction. Phase 5 ships the `lethe` namespace alongside `remember`: `encode_bnb4` / `encode_bnb4_double_quant` / `encode_bnb_int8` plus the bit-exact `round_trip` validation harness. Cross-validated against real `bitsandbytes` on **7 fixtures across 4 architecture families** (Llama 3.2 / Qwen3 / Qwen2.5 / Phi-3.5): every fixture round-trips **byte-exact** (0 byte diffs) against the original `bitsandbytes`-quantised on-disk bytes — including the high-nibble-first packing order and the double-quant `nested_offset` (both fixed in v0.6.4).

| Fixture | Format | Elements | Byte-exact round-trip | vs PyTorch quantize (CPU) |
|---|---|---|---|---|
| Llama-3.2-1B-Instruct-bnb-nf4 | NF4 plain | 4,096 | ✓ 0 / 2048 | 0.22× (slower) |
| Llama-3.2-1B-BNB-FP4 | FP4 plain | 4,096 | ✓ 0 / 2048 | 0.24× (slower) |
| Llama-3.2-1B-Instruct-bnb-nf4-double-quant | NF4 double-quant | 4,096 | ✓ 0 / 2048 | 0.22× (slower) |
| Llama-3.2-1B-BNB-INT8 | INT8 | 65,536 | ✓ 0 / 65536 | 0.03× (32× slower) |
| ema1234/qwen_mcqa_bnb_fp4 | FP4 plain (Qwen3) | 4,096 | ✓ 0 / 2048 | 0.20× (slower) |
| unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit | NF4 double-quant (Qwen2.5) | 4,096 | ✓ 0 / 2048 | 0.18× (slower) |
| unsloth/Phi-3.5-mini-instruct-bnb-4bit | NF4 double-quant (Phi-3.5) | 4,096 | ✓ 0 / 2048 | 0.18× (slower) |

> **Sign-of-zero preservation finding (FP4):** The on-disk `bitsandbytes` Python `FP4` `quant_map` stores `+0.0` at *both* index 0 *and* index 8 — collapsing the `±0` pair. A naive `decode → encode` round-trip would be mathematically impossible under that codebook. Phase 5 introduces a narrow, principled tweak in `dequantize_bnb4_to_bf16`: when a codebook entry is exactly `+0.0` AND the nibble has its high bit set (`nibble & 0x8 != 0`), the emitted `BF16` is `-0.0`. This recovers the sign information `bitsandbytes`' Python decode discards. Arithmetically invisible (both are IEEE 754 zero), affects `0.2 %` of `FP4` elements, no-op for `NF4`. The encoder mirrors the rule with `apply_sign_magnitude_encode_correction`. Confirmed to generalise: the Qwen3 FP4 fixture shows the same `+0.0` / `+0.0` codebook collapse and round-trips byte-exact under the rule.

> **Ecosystem finding (NF4 double-quant):** `hf-fm inspect` HTTP-range probes during cross-architecture candidate selection revealed that **every** non-Llama BnB-NF4 model checked uses double-quant — bitsandbytes' default. Plain NF4 is effectively a Llama-fixture-only phenomenon. Without `encode_bnb4_double_quant` (Step 1c), anamnesis would only encode a tiny corner of real-world BnB-4bit models. Promoted from deferred polish to required Step 1c gate on `v0.5.0`.

> **On the "slower than PyTorch" column:** The encode kernels are 4–6× slower than PyTorch's broadcast-vectorised quantize on `BnB4`, 32× slower on `INT8`. This is expected — PyTorch encode uses a single broadcast tensor op (`(blocks.unsqueeze(-1) - codebook).abs().argmin(dim=-1)`) that vectorises across the whole tensor; the Rust encode loop is currently scalar per element. **Phase 7 (CPU SIMD pass) is the natural target** — this table makes the gap visible. The same loop-fission + `target-cpu=native` infrastructure that gave the decode path its 18–54× wins is the candidate retrofit on the encode side.

## Format conversion pipeline (Phase 6, v0.6.0)

`amn convert <input> --to <target>` routes any v0.6.0-available format pair through a single dispatch. Targets at v0.6.0: `safetensors` (alias `bf16`), `gguf` (unquantised passthrough), `bnb-nf4`. Quantised GGUF targets (`gguf-q4km`, …) land in Phase 6.14 / 7.5 through the same dispatch.

End-to-end runtime, **release build, `target-cpu=native`, single 4096×4096 tensor (32 MiB BF16 or 64 MiB F32) on CPU**, vs the closest Python equivalents (best-of-5 median, PyTorch 2.10.0 with the CPU device, NumPy 2.4, safetensors-py 0.7, gguf-py 0.18, bitsandbytes 0.49.1's CPU kernel):

| Conversion | anamnesis | Python ecosystem default | PyTorch-CPU equivalent |
|---|---:|---:|---:|
| `npz → safetensors` | **11.2 ms** | 75.7 ms (numpy + safetensors-py) — **6.75× faster** | 92.5 ms (torch.from_numpy + safetensors.torch) — **8.24× faster** |
| `pth → safetensors` | **5.7 ms** | — | 29.6 ms (torch.load + safetensors.torch) — **5.18× faster** |
| `safetensors-BF16 → GGUF` | **13.6 ms** | 15.1 ms (gguf-py) — **1.11× faster** | 29.6 ms (safetensors.torch.load + gguf-py) — **2.17× faster** |
| `safetensors-BF16 → BnB-NF4` | **141 ms** | 376.8 ms (bitsandbytes CPU) — **2.67× faster** | — *(bitsandbytes is the PyTorch-native path)* |

> **What "PyTorch-CPU equivalent" means here:** the two non-PyTorch baselines (NPZ via NumPy, GGUF via gguf-py) get a second row that routes the data through `torch.from_numpy` / `safetensors.torch.load_file` before the writer step — the way Python practitioners typically pipeline these formats when they already work in PyTorch tensors. The PyTorch row is consistently 1.2–1.6× slower than the ecosystem default because the extra `torch.from_numpy` / `tensor.numpy()` hop is non-zero work, not because PyTorch is inefficient. We report both so the table is honest about Python's choice space, not just the fastest available Python path.

> **Methodology:** Each row's number is the median of 5 release-mode runs at the same 4096×4096 shape both sides agree on, captured in the `tests/fixtures/convert_reference/*.timing.json` sidecars and re-validated by `t14_perf_vs_python_size_matched` in [`tests/cross_validation_convert.rs`](../tests/cross_validation_convert.rs). Run `cargo test --release --all-features --test cross_validation_convert -- --ignored --nocapture t14` to reproduce on your own machine, and re-generate the sidecars via `tests/fixtures/convert_reference/generate_convert_timings.py`. The 13 byte-exact round-trip tests in the same file (default, not `--ignored`) exercise every conversion pair at small synthetic fixtures so the correctness suite stays fast.

> **Limitations:** The single-tensor `O(input size)` shape of the perf measurement applies only to the convert primitive itself, not to whole-model conversion. Multi-shard models retain every dequantised tensor in heap until `safetensors::serialize_to_file` returns — same constraint as `ParsedModel::remember`, see the GGUF block-quant peak-heap note above. Phase 10 (streaming output) is the planned remedy.

## Whole-model throughput (Phase 7 / 7.2)

The convert table above measures the **convert primitive on one tensor**, and says so in its
Limitations note. This section covers the other axis: **whole-model** dequantisation, where
multi-threading applies and where the comparison against the Python stack is most meaningful.

Release build, `target-cpu=native`, best-of-5 median, AMD Ryzen 9 5950X / Windows 11. Method and
full discussion in [`perf-experiments.md`](perf-experiments.md) Experiment 12.

### Thread scaling — the `GGUF` reader stage (`convert::read_hub`)

| Model | 1 thread | 4 threads (default) | 16 threads |
|---|---:|---:|---:|
| `SmolLM2-135M-Instruct-Q4_K_M` (100.6 MiB) | 103.88 ms | **54.70 ms — 1.90×** | 59.45 ms — 1.75× |
| `tinyllama-1.1b-chat-v1.0.Q5_0` (731.5 MiB) | 694.57 ms | **355.76 ms — 1.95×** | 319.71 ms — 2.17× |

Output is **byte-identical at every thread count** — asserted across `{1, 2, 4, 8, 16}` and the
hardware-resolved default, on 269 MB and 2.2 GB outputs, before any timing is reported.

**Why ~1.9× and not the ~3–4× the safetensors path reaches:** the conversion hub owns one `Vec`
per tensor, which is structurally the `Vec`-per-task case Experiment 11 measured a 2.2× ceiling
for (versus 2.9× for pre-allocated disjoint slices). The reader lands exactly there. Going faster
means redesigning `Hub` toward one contiguous model-wide buffer — a memory-layout change with an
API blast radius — so it is recorded as the known ceiling, not attempted.

**End-to-end `convert()` gains less again — 1.19–1.36× — and that is Amdahl, not a defect.**
Writing the multi-`GiB` output file is ~60 % of the wall clock and is inherently sequential. Quote
the reader figure for the reader and the end-to-end figure for `convert()`; they answer different
questions.

### Whole-model `GGUF` dequantisation vs `gguf-py`

Same fixtures, against [`gguf-py`](https://pypi.org/project/gguf/) 0.18.0 — the only
general-purpose `GGUF` dequantiser on PyPI, `NumPy`-vectorised rather than naive Python loops, and
the library anamnesis is cross-validated against:

| Model | `gguf-py` | anamnesis 1 thread | anamnesis 4 threads |
|---|---:|---:|---:|
| `SmolLM2-135M-Instruct-Q4_K_M` | 2894.6 ms | 103.9 ms — **27.9×** | 54.7 ms — **52.9×** |
| `tinyllama-1.1b-chat-v1.0.Q5_0` | 12031.4 ms | 694.6 ms — **17.3×** | 355.8 ms — **33.8×** |

> **This is not a like-for-like output, and the numbers should not be quoted without it.**
> `gguf-py` returns `float32`; anamnesis returns `BF16` — half the bytes, and the narrower type.
> What is verified is that anamnesis's `BF16` is bit-identical to `gguf-py`'s `float32`
> **correctly rounded to `BF16`** (0 ULP, all 22 kernels), so the two agree on the values and
> differ only in delivered width; both do their block arithmetic in `f32` internally. Since that
> width difference is itself worth ~2× of memory traffic on a bandwidth-bound workload, halving
> `gguf-py`'s time as a generous correction still leaves ~8.7–14× and ~17–26×. The like-for-like
> threading row is the 1-thread column — `gguf-py` is single-threaded.

> **Methodology:** anamnesis numbers from the `#[ignore]`d `hub_scaling_bench` in
> [`src/convert.rs`](../src/convert.rs) (in-crate because `read_hub` is private) and
> [`tests/bench_gguf_convert_adhoc.rs`](../tests/bench_gguf_convert_adhoc.rs). The `gguf-py`
> baseline is regenerated by
> [`generate_gguf_dequant_timings.py`](../tests/fixtures/gguf_reference/generate_gguf_dequant_timings.py);
> its sidecar JSONs are committed, so the comparison prints without a Python environment. The
> models themselves are gitignored (~13 GiB).

## Parsing & inspection

Header-only **inspection** ships for every format — read the declared tensor metadata (names, shapes, dtypes, totals) without materialising tensor data, over any reader substrate (in-memory `Cursor`, an HTTP-range adapter, a custom transport). anamnesis takes on no network or TLS dependency; downstream crates plug in their own adapter. Reader requirement per format:

| Format | Path entry | Reader-generic entry | Reader bound |
|---|---|---|---|
| safetensors | `parse(path)` | `parse_safetensors_header_from_reader<R: Read>` | **`Read`** — layout is prefix-then-JSON, two contiguous reads, never seek-back |
| NPZ | `inspect_npz(path)` | `inspect_npz_from_reader<R: Read + Seek>` | **`Read + Seek`** — ZIP central directory lives at end-of-file |
| GGUF | `parse_gguf(path)` | `inspect_gguf_from_reader<R: Read + Seek>` · `parse_gguf_front_matter_from_reader<R: Read + Seek>` | **`Read + Seek`** — tensor offsets resolved against a captured stream length |
| `.pth` | `parse_pth(path)` | `inspect_pth_from_reader<R: Read + Seek>` | **`Read + Seek`** — ZIP central directory + seek-back to local headers |

GGUF carries two reader-generic entry points because the two useful outputs differ in size, not in work: `inspect_gguf_from_reader` reduces to the aggregate `GgufInspectInfo` (version, arch, tensor count, totals, distinct dtypes) for the cheap inspect-before-parse policy gate, while `parse_gguf_front_matter_from_reader` (v0.7.1) keeps the whole `GgufFrontMatter` — the full metadata table plus the per-tensor name/shape/dtype/offset list, the same detail `parse_gguf(path)` exposes over mmap. Both delegate to one internal core over the same bytes, so they are substrate- and limits-equivalent by construction. The full-detail variant is what lets a remote consumer render a complete tensor table without downloading the data segment.

See the rustdoc on each `*_from_reader` for the exact access pattern an adapter must satisfy.

### NPZ / NPY

Feature-gated behind `npz`. Custom NPY header parser with bulk `read_exact` — zero per-element deserialization for little-endian data on little-endian machines. Cross-validated byte-exact against NumPy on Gemma Scope 2B SAE weights.

| Metric | Value |
|---|---|
| Throughput (302 MB Gemma Scope, F32) | **3,586 MB/s** |
| Overhead vs raw I/O | 1.3x |
| vs `npyz` crate | **17.7x faster** |
| Supported dtypes | F16, BF16, F32, F64, Bool, U8–U64, I8–I64 |

BF16 support via JAX `V2` void-dtype convention. Big-endian NPY files handled with in-place byte-swap.

### PyTorch `.pth`

Feature-gated behind `pth`. Minimal pickle VM (~36 opcodes) with security allowlist. Memory-mapped I/O with zero-copy tensor access (`Cow::Borrowed` from mmap). Cross-validated byte-exact against PyTorch `torch.load()` on 3 [AlgZoo](https://github.com/alignment-research-center/alg-zoo) models (MIT-0 license).

| Model | Size | Tensors | vs `torch.load` |
|---|---|---|---|
| torchvision ResNet-18 | 45 MB | 102 | **11.2x faster** |
| torchvision ResNet-50 | 98 MB | 267 | **12.7x faster** |
| torchvision ViT-B/16 | 330 MB | 152 | **30.8x faster** |

Lossless `.pth` → `.safetensors` conversion preserving original dtypes (F16, BF16, F32, F64, I8–I64, U8, Bool), writing directly from mmap slices to the output file — zero intermediate copies. Handles both newer (`archive/` prefix) and older (`{model_name}/` prefix) PyTorch ZIP conventions; legacy (pre-1.6) raw-pickle files are rejected with a clear error.

**Inspection** (header-only, full 6 960-file [AlgZoo](https://github.com/alignment-research-center/alg-zoo) corpus; best-of-5 release-mode median per file, `target-cpu=native`, PyTorch 2.10.0+cu130):

| Substrate | Median per file | vs `torch.load` |
|---|---:|---:|
| `parse_pth(path).inspect()` (mmap) | 124.0 µs | **4.07x faster** |
| `inspect_pth_from_reader(File)` (reader) | 168.7 µs | **2.99x faster** |
| `torch.load(weights_only=True)` (PyTorch) | 504.3 µs | baseline |

PyTorch has no separate inspect-only primitive — `torch.load(weights_only=True)` fully materialises every tensor before the caller can iterate the `state_dict`, so the speedup is a **lower bound** that grows by orders of magnitude on larger models (the reader path stays bounded by `data.pkl` size while `torch.load` scales linearly in total tensor-data size). Per-family breakdown and full method: [`docs/perf-experiments.md`](perf-experiments.md) Experiment 6. A 300 MB torchvision `.pth` is inspectable through an HTTP-range adapter in well under 100 KiB of network transfer (ZIP central directory + the `data.pkl` entry only), instead of 300 MB.

## Performance & peak-heap validation (Phase 6.5)

Three dev-only validation tracks — none affect the published crate (excluded from the tarball by Cargo's defaults). Each is a regression detector that catches drift before it reaches a release:

- **[Criterion runtime benchmarks](../benches/README.md)** — `benches/dequant.rs` covers 7 synthetic-layer-sized kernel groups (FP8, GPTQ, AWQ, BnB NF4, BnB INT8, GGUF Q4_K, plus FP8 fine-grained) at `4096 × 11008`, reporting **657 Melem/s → 2.01 Gelem/s** on the dev machine. `benches/parsing.rs` covers header-only parses for all four formats vs an `fs::read` baseline. `benches/convert.rs` (v0.7.2) covers the **whole-model threaded** paths — quantised `GGUF` → safetensors and `FP8` → `BF16` `remember` — each at **1 and 4 threads**, so a dispatch that silently serialises shows up as the two converging. Plus a real-world group on the Ollama-cached `llama3.2:1b` `Q8_0` slice (**3.92 Gelem/s**). Run with `cargo bench --features gptq,awq,bnb,gguf,npz,pth`.
- **[`dhat-rs` peak-heap assertions](../tests/peak_heap_README.md)** — three `#[ignore]`d test binaries that wrap the global allocator and assert observed peak heap stays within the documented ceiling. **Every kernel's observed scratch matches the documented `# Memory` claim to the byte**: GPTQ/AWQ at `3 × out_features × 4`, BnB-DQ at `num_blocks × 4 + block_size × 4`. Run with `cargo test --release --features gptq --test peak_heap_gptq -- --ignored --nocapture` (and similarly for `awq` / `bnb_dq`). The v0.6.7 vendored ZIP reader is pinned by `peak_heap_zip_metadata.rs` at **41 B/entry vs the `zip` crate's 337 B/entry** (8.07× resident reduction).
- **[Ollama cross-validation](../tests/cross_validation_ollama.rs)** — bit-exact `Q8_0` dequant against the `gguf-py` reference on a slice extracted from the local Ollama cache's `llama3.2:1b` blob. Result: **0 ULP mismatches**.

## Robustness hardening timeline

The untrusted-input hardening line, version by version. Every parser bounds its header-derived allocations against the [`candle #3533`](https://github.com/huggingface/candle/issues/3533) unguarded-allocation DoS class ([CWE-770](https://cwe.mitre.org/data/definitions/770.html) / [CWE-1284](https://cwe.mitre.org/data/definitions/1284.html) / [CWE-400](https://cwe.mitre.org/data/definitions/400.html)): a length / count / dimension field read from a file header is validated against an explicit cap *before* it reaches `vec!` / `Vec::with_capacity` / the pickle VM.

**v0.6.1** — GGUF and safetensors were already capped; v0.6.1 closes the NPZ `header_len` window (NPY v2/v3 decode it as a `u32`, reachable to 4 GiB) and the PyTorch `.pth` mmap-path `data.pkl` size gate (previously bounded only by file size), and adds NPZ array-byte and per-opcode pickle payload caps as defence in depth.

**v0.6.2** — a full pre-Phase-7 security audit (all four parsers plus the dequant/encode layers) adds a *container-size cross-check*: NPZ rejects a declared array size larger than the ZIP entry's own uncompressed `size()` before allocating, and the GGUF reader validates each variable-length read against the bytes physically remaining. A shared checked shape-product helper removes a debug-panic / release-wrap on adversarial shapes. Backing the audit is a coverage-guided [`cargo fuzz`](../fuzz/README.md) harness — one libFuzzer target per parser, dev-only, focused on the pickle VM — run with zero crashes.

**v0.6.3** — makes those fixed, server-scale caps *caller-configurable* (Phase 6.8). A new `ParseLimits` budget — `max_single_alloc`, `max_total_bytes`, `max_item_count`, `max_decompression_ratio` — threads through every parser via `parse_*_with_limits`, enforced fail-fast and tighten-only; `ParseLimits::default()` is unbounded, so default behaviour is byte-for-byte unchanged. Also drops the unused `zopfli` DEFLATE *compressor*.

**v0.6.6** — governs the `.pth` pickle VM's *working set* (Phase 6.11), closing a P0 an independent audit surfaced. The VM's value stack, memoised clones, and nesting depth were charged to nothing — so a small crafted pickle could amplify into multi-GiB heap ([CWE-1325](https://cwe.mitre.org/data/definitions/1325.html)) or a recursive-`Drop` stack overflow ([CWE-674](https://cwe.mitre.org/data/definitions/674.html)). Every value-creating opcode now flows through one `O(1)`-per-opcode accounting choke point: each pushed value and the deep size of each memo clone is charged to a permanent **512 MiB working-set floor** *and* the caller's `max_total_bytes`, and construction nesting is capped at a permanent **depth of 256**. Both floors are always-on (they bite even under `ParseLimits::default()`). A 385k-run RSS-limited `fuzz_pth` campaign runs clean. Bundled fix: BnB4 dequant/encode now reject an odd `block_size`.

**v0.6.7** — replaces the `zip` crate's container parser with a lean, vendored, **read-only** central-directory reader (Phase 6.12), closing a *bounded* metadata-amplification gap (`zip::ZipArchive::new` materialises **337 B/entry** vs the **~40 B/entry** anamnesis needs). The vendored reader owns just the container (EOCD + central directory + local-header offsets, full **ZIP64** support); `DEFLATE` `.npz` entries still inflate through `flate2` / `miniz_oxide`. Bounds-checked cursor reads, `checked_*` offset arithmetic, a permanent entry-count + per-name cap, a `data_start + size ≤ file_len` cross-check, a compression allowlist, and it honours the caller's `ParseLimits` ([CWE-770](https://cwe.mitre.org/data/definitions/770.html)). `DEFLATE` inflation on the `.pth` reader path is bounded with `Read::take(uncompressed_size)` ([CWE-409](https://cwe.mitre.org/data/definitions/409.html)). The `.pth` entry index moved to a sorted, `shrink_to_fit`-trimmed `Vec<(Box<str>, …)>` — **41 B/entry, 8.07× reduction**. `zip` left the runtime dependency tree (now dev-only + the differential fuzz oracle); `fuzz_zip` landed.

**v0.6.8** — Python-readiness hardening (Phase 6.13), library-side. A **copy-based, mmap-free full-parse path** (`parse_bytes` / `parse_*_from_reader`) becomes the recommended entry point for untrusted input: owned `Vec<u8>`, no mmap, so a truncated or concurrently-written source is a clean `Err` instead of an uncatchable **`SIGBUS` / `SIGSEGV`** (an OS signal Python `try/except` cannot catch). `AnamnesisError` gains a finer taxonomy — `LimitExceeded { limit }` and `DisallowedGlobal { module, name }` split out of the formerly-opaque `Parse` — both reachable even under `ParseLimits::default()` (allowlist + permanent floors are always-on). Panic/abort-freedom is now a *tested* invariant — a stable-CI `catch_unwind` battery over every parse/inspect entry point plus owned-path `cargo fuzz` targets (~76 M executions, zero crashes) — and the Phase 8 PyO3 `cdylib` is pinned to a `panic = "unwind"` profile. The `NumPy` / `BF16` data-ownership contract is locked ([python-interop.md](python-interop.md)): returned arrays own their bytes by default (no use-after-free reachable from Python), `BF16` via `ml_dtypes.bfloat16` else raw bytes + dtype.
