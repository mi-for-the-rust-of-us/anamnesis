# Lethe — Encode-Side Walkthrough

**Status:** v0.7.8. The `BnB` encode kernels shipped in v0.5.0 (Phase 5 step 1a/1b/1c) and the `BnB-NF4` safetensors writer behind `amn convert --to bnb-nf4` in v0.6.0 (Phase 6). Covers `BnB` encode only: `FP8` / `GGUF` / `IQ` / `TQ` / `MXFP4` encode land in Phase 8.5.

This document walks through the `lethe` namespace — the encode-side inverse of [`remember`](../src/remember/). Three audiences:

- **Round-trip consumers** (testing tools, fixture validators): use the strict-mirror API where you supply the same metadata the decoder originally read.
- **Fresh-quantise consumers** (Phase 6 conversion path, end-user CLIs): use the `_compute_*` convenience variants that derive metadata from the source `BF16`.
- **Downstream embedders** (`candle-mi`, the Python bindings planned for Phase 8, v0.8.0): consume the kernels as library primitives that produce raw bytes, no framework coupling.

---

## API surface at a glance

| Function | Direction | Input metadata | Output |
|---|---|---|---|
| [`encode_bnb4`](../src/lethe/bnb.rs) | strict mirror of decode | caller-supplied `absmax_data` + `quant_map_data` | packed nibbles (`Vec<u8>`) |
| [`encode_bnb4_compute_absmax`](../src/lethe/bnb.rs) | quantise from BF16 source | derives per-block absmax internally; caller still supplies `quant_map_data` | `(packed_weight, absmax_bytes)` |
| [`encode_bnb4_double_quant`](../src/lethe/bnb.rs) | strict mirror, DQ | caller-supplied `absmax_data` (U8) + `quant_map_data` + `nested_absmax_data` + `nested_quant_map_data` | packed nibbles |
| [`encode_bnb_int8`](../src/lethe/bnb.rs) | strict mirror | caller-supplied `scb_data` | `i8` bytes |
| [`encode_bnb_int8_compute_scb`](../src/lethe/bnb.rs) | quantise from BF16 source | derives per-row `SCB` internally | `(weight_bytes, scb_bytes)` |
| [`NF4_CODEBOOK`](../src/lethe/bnb.rs) | constant | — | `[f32; 16]` |
| [`FP4_CODEBOOK`](../src/lethe/bnb.rs) | constant | — | `[f32; 16]` (preserves `-0.0` at index 8) |
| [`write_bnb_nf4_safetensors`](../src/lethe/bnb_writer.rs) | quantise from BF16 source, whole file | derives per-block absmax (single-quant, `block_size` 64) | a `.safetensors` file on disk |
| [`write_bnb_nf4_safetensors_bytes`](../src/lethe/bnb_writer.rs) | as above, in memory | as above | serialised safetensors bytes (`Vec<u8>`) |
| [`is_eligible_for_nf4`](../src/lethe/bnb_writer.rs) | eligibility test | a tensor shape | `bool`: 2-D, at least 64 elements, a multiple of 64 |
| [`classify_inputs`](../src/lethe/bnb_writer.rs) | dry-run count | a slice of `BnbWriteInput` | `BnbNf4WriteStats` (`quantized`, `passthrough`) |

All functions are `#[cfg(feature = "bnb")]`. Add `bnb` to your `[dependencies]` features list:

```toml
[dependencies]
anamnesis = { version = "0.7", features = ["bnb"] }
```

---

## Walkthrough 1 — Encode a fresh `BF16` source to `BnB NF4`

The Phase 6 "quantise from `BF16` source" use case. The caller has a `BF16` tensor in memory and wants the BnB on-disk layout (packed weight + `F32` absmax + `F32` quant_map).

```rust
use anamnesis::{encode_bnb4_compute_absmax, NF4_CODEBOOK};

// Step 1: have your BF16 source as raw bytes (e.g., from a safetensors parse).
// 4096 BF16 elements arranged as 64 blocks of block_size = 64.
let bf16_bytes: Vec<u8> = /* ... your source ... */ vec![0u8; 4096 * 2];

// Step 2: serialize the canonical NF4 codebook to bytes (one-time, ~64 bytes).
let codebook_bytes: Vec<u8> = NF4_CODEBOOK
    .iter()
    .flat_map(|v| v.to_le_bytes())
    .collect();

// Step 3: encode — derives per-block absmax internally, returns packed weight + absmax bytes.
let (weight_bytes, absmax_bytes) = encode_bnb4_compute_absmax(
    &bf16_bytes,
    &codebook_bytes,
    /* total_elements = */ 4096,
    /* block_size = */ 64,
)?;

assert_eq!(weight_bytes.len(), 2048);  // 4096 elements / 2 nibbles per byte
assert_eq!(absmax_bytes.len(), 256);   // 64 blocks * F32 LE
# Ok::<(), anamnesis::AnamnesisError>(())
```

Write the three buffers (`weight`, `absmax`, `quant_map`) into a `.safetensors` file alongside the original config, matching the bitsandbytes companion-tensor naming convention:

- `<layer>.weight` ← `weight_bytes` (`U8` dtype)
- `<layer>.weight.absmax` ← `absmax_bytes` (`F32` dtype, shape `[num_blocks]`)
- `<layer>.weight.quant_map` ← `codebook_bytes` (`F32` dtype, shape `[16]`)

For `FP4`, substitute `FP4_CODEBOOK`. The bitsandbytes Python `quant_map` collapses `-0.0` to `+0.0` at index 8 — anamnesis's hardcoded `FP4_CODEBOOK` preserves `-0.0` distinct from `+0.0` (so the round-trip is byte-exact under our codebook, and only deviates by the sign-of-zero bit pattern on `0.2 %` of elements when read back through bitsandbytes' Python decode).

---

## Walkthrough 2 — Round-trip a `BnB`-quantised file

The fixture-validation / cross-check use case. The caller has the original on-disk `weight_data` and wants to confirm that re-encoding the decoded `BF16` reproduces the original bytes.

```rust
use anamnesis::{encode_bnb4, remember::bnb::dequantize_bnb4_to_bf16};

// Inputs read from the on-disk .safetensors file:
let weight_data: &[u8]    = /* bitsandbytes-quantised packed nibbles */;
let absmax_data: &[u8]    = /* F32 LE absmax tensor */;
let quant_map_data: &[u8] = /* F32[16] codebook tensor */;
let total_elements: usize = /* num blocks * block_size */;
let block_size: usize     = 64;

// Decode → BF16 → re-encode (using the same metadata).
let bf16 = dequantize_bnb4_to_bf16(
    weight_data, absmax_data, quant_map_data,
    total_elements, block_size,
)?;
let re_encoded = encode_bnb4(
    &bf16, absmax_data, quant_map_data,
    total_elements, block_size,
)?;

assert_eq!(re_encoded, weight_data, "round-trip should be byte-exact");
# Ok::<(), anamnesis::AnamnesisError>(())
```

This pattern is the **bit-exact round-trip contract**: for any codebook with distinct entries, `encode(decode(weight, metadata), metadata) == weight` byte-for-byte. The contract holds:

- **NF4 plain**: codebook entries are distinct → unconditional.
- **FP4 plain**: codebook collapses `±0` at indices 0/8 → would normally fail; the sign-of-zero rule in `dequantize_bnb4_to_bf16` recovers the lost sign info, so the round-trip is byte-exact under both anamnesis's `FP4_CODEBOOK` constant *and* the on-disk bitsandbytes Python codebook.
- **INT8**: per-row affine quant, recoverable up to the `[-128, 127]` clamp on the edges.
- **NF4 double-quant**: nested absmax recovery is deterministic; same contract holds.

Tested at 0-byte-diff bit-exactness on 7 fixtures spanning 4 architectures (Llama 3.2 / Qwen3 / Qwen2.5 / Phi-3.5) — see [`tests/cross_validation_bnb_encode.rs`](../tests/cross_validation_bnb_encode.rs).

---

## Walkthrough 3 — Encode `BnB INT8` from `BF16` source

```rust
use anamnesis::encode_bnb_int8_compute_scb;

// 256 rows × 256 columns = 65536 BF16 elements.
let bf16_bytes: Vec<u8> = vec![0u8; 256 * 256 * 2];

let (weight_bytes, scb_bytes) = encode_bnb_int8_compute_scb(
    &bf16_bytes,
    /* out_features = */ 256,
    /* in_features = */ 256,
)?;

assert_eq!(weight_bytes.len(), 256 * 256);  // one i8 byte per element
assert_eq!(scb_bytes.len(), 256 * 4);       // one F32 SCB per row
# Ok::<(), anamnesis::AnamnesisError>(())
```

`SCB` = per-row absmax (`F32`). The encoder derives it as `max(|x|)` over each row, then writes each element as `round(x * 127.0 / SCB)` clamped to `[-128, 127]`. The clamp matters at the boundary: an exact-`SCB` value rounds to `+127`, not `+128` (which would overflow `i8`).

---

## Walkthrough 4 — Encode `BnB NF4` with double-quant

The strict-mirror variant for fixtures that already have all the double-quant metadata. Phase 5 step 1c. The caller supplies the `U8` quantised absmax bytes and the nested-quant metadata.

```rust
use anamnesis::encode_bnb4_double_quant;

let bf16_data: &[u8]            = /* BF16 decoded earlier */;
let absmax_data: &[u8]          = /* U8 quantised absmax */;
let quant_map_data: &[u8]       = /* F32[16] main codebook */;
let nested_absmax_data: &[u8]   = /* F32 per-nested-block scale */;
let nested_quant_map_data: &[u8] = /* F32[256] nested codebook */;

let packed_weight = encode_bnb4_double_quant(
    bf16_data,
    absmax_data,
    quant_map_data,
    nested_absmax_data,
    nested_quant_map_data,
    /* total_elements = */ 4096,
    /* block_size = */ 64,
    /* nested_block_size = */ 64,
)?;
# Ok::<(), anamnesis::AnamnesisError>(())
```

The encoder recovers the per-block `F32` absmax internally via `nested_quant_map[absmax_byte] * nested_absmax[nested_block_idx]` — the same formula the decoder applies — then delegates to the inner `encode_bnb4_core`. Round-trip is byte-exact when the supplied metadata matches what the decoder originally read.

> **Note:** there is still no `encode_bnb4_double_quant_compute_*` convenience entry point. The fresh-quantise-from-`BF16`-source path that shipped in Phase 6 (`amn convert --to bnb-nf4`, `write_bnb_nf4_safetensors`) writes single-quant absmax and does not need one. Deriving absmax, nested_absmax, and the nested codebook from a fresh source is encode-completion work (Phase 8.5).

---

## How the kernels work (one-line each)

- **`encode_bnb4*`** — for each pair of consecutive `BF16` elements, divide by the block's absmax, find the nearest entry in the 16-entry codebook via linear scan (exact-bit-match priority for `±0` disambiguation), pack the two 4-bit indices into one `U8`. Inverse of `dequantize_bnb4_to_bf16`.
- **`encode_bnb_int8*`** — for each `BF16` element, divide by the row's `SCB / 127.0` scale, round-to-nearest, clamp to `[-128, 127]`, store as `i8` (`u8` two's-complement). Inverse of `dequantize_bnb_int8_to_bf16`.
- **`apply_sign_magnitude_encode_correction`** — mirrors the decode-side sign-of-zero rule. When the source value is sign-negative AND the nearest-search returned a lower-half nibble AND the corresponding upper-half codebook entry has the same bits as the chosen entry, shift the nibble to the upper half. This recovers the sign-magnitude convention bitsandbytes' encode kernel uses for `FP4` with the collapsed `+0/+0` codebook.

For a deeper read see [`src/lethe/bnb.rs`](../src/lethe/bnb.rs) (the module-level `//!` doc) and the [round-trip harness](../src/lethe/round_trip.rs).

---

## What anamnesis does *not* do (yet) on the encode side

- **No CLI `quantize` / `forget` subcommand yet.** Encoding from the command line goes through `convert`, which has existed since v0.6.0: `amn convert model.safetensors --to bnb-nf4 -o quantised.safetensors` (any input format, quantised inputs dequantised first through the hub). It emits plain single-quant `NF4` only; `FP4`, `INT8`, and double-quant encode remain library-only. A `forget` dispatch and subcommand are Phase 8.5 step 7.
- **No `encode_bnb4_double_quant_compute_*` convenience** (deferred to Phase 8.5; see the note under Walkthrough 4).
- **No FP8 / GPTQ / AWQ / GGUF / IQ / TQ / MXFP4 encode.** All are targeted at Phase 8.5 ("Lethe Encode Completion"), shipping after the BnB encode pipeline has been validated end-to-end through the Python bindings in Phase 8.
- **No Python bindings yet.** Phase 8 (PyO3, v0.8.0) exposes the encode + decode + convert primitives to the Python ecosystem.
- **No SIMD on encode hot paths.** Encode kernels are currently 4–6× slower than PyTorch's broadcast-vectorised quantize on `BnB4`, 32× slower on `INT8`. Phase 7 was the CPU SIMD pass, and on the decode side it measured null: a bit-exact hand-written AVX2 `f32 → BF16` writer gained 1.02×, because the compiler already vectorises and the writer is bandwidth-bound ([Experiment 10](perf-experiments.md)). Phase 7 shipped multi-threading instead. The encode loops' nearest-codebook scan has not been measured the same way, so whether explicit SIMD would pay there is an open question rather than a plan.

See [`ROADMAP.md`](../ROADMAP.md) for the full sequencing.

---

## See also

- [`docs/validation.md`](validation.md#quantization-lethe--phase-5): the Lethe section with the cross-architecture fixture table
- [`CHANGELOG.md`](../CHANGELOG.md): `[0.5.0]` entry block (the kernels) and `[0.6.0]` (the `bnb-nf4` convert target)
- [`ROADMAP.md`](../ROADMAP.md) — Phase 5 step 1a/1b/1c (shipped) + Phase 8.5 (deferred encode kernels)
- [`docs/rust-ecosystem-comparison.md`](rust-ecosystem-comparison.md) — where anamnesis's encode-side coverage stands in the wider Rust + cross-language landscape
- [`docs/perf-experiments.md`](perf-experiments.md) — case-study entry for the sign-of-zero preservation rule (Experiment 7)
