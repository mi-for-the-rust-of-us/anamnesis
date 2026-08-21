# Lethe — Encode-Side Walkthrough

**Status:** v0.5.0 (Phase 5 step 1a/1b/1c shipped). Covers `BnB` encode only — `FP8` / `GGUF` / `IQ` / `TQ` / `MXFP4` encode land in Phase 8.5.

This document walks through the `lethe` namespace — the encode-side inverse of [`remember`](../src/remember/). Three audiences:

- **Round-trip consumers** (testing tools, fixture validators): use the strict-mirror API where you supply the same metadata the decoder originally read.
- **Fresh-quantise consumers** (Phase 6 conversion path, end-user CLIs): use the `_compute_*` convenience variants that derive metadata from the source `BF16`.
- **Downstream embedders** (`candle-mi`, future Python bindings in Phase 7): consume the kernels as library primitives that produce raw bytes, no framework coupling.

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

All functions are `#[cfg(feature = "bnb")]`. Add `bnb` to your `[dependencies]` features list:

```toml
[dependencies]
anamnesis = { version = "0.5", features = ["bnb"] }
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

> **Note:** there is no `encode_bnb4_double_quant_compute_*` convenience entry point in v0.5.0. The fresh-quantise-from-`BF16`-source path (Phase 6 conversion matrix) requires deriving absmax + nested_absmax + the nested codebook from the source — that work is deferred to the Phase 6 conversion-CLI commit.

---

## How the kernels work (one-line each)

- **`encode_bnb4*`** — for each pair of consecutive `BF16` elements, divide by the block's absmax, find the nearest entry in the 16-entry codebook via linear scan (exact-bit-match priority for `±0` disambiguation), pack the two 4-bit indices into one `U8`. Inverse of `dequantize_bnb4_to_bf16`.
- **`encode_bnb_int8*`** — for each `BF16` element, divide by the row's `SCB / 127.0` scale, round-to-nearest, clamp to `[-128, 127]`, store as `i8` (`u8` two's-complement). Inverse of `dequantize_bnb_int8_to_bf16`.
- **`apply_sign_magnitude_encode_correction`** — mirrors the decode-side sign-of-zero rule. When the source value is sign-negative AND the nearest-search returned a lower-half nibble AND the corresponding upper-half codebook entry has the same bits as the chosen entry, shift the nibble to the upper half. This recovers the sign-magnitude convention bitsandbytes' encode kernel uses for `FP4` with the collapsed `+0/+0` codebook.

For a deeper read see [`src/lethe/bnb.rs`](../src/lethe/bnb.rs) (the module-level `//!` doc) and the [round-trip harness](../src/lethe/round_trip.rs).

---

## What anamnesis does *not* do (yet) on the encode side

- **No CLI `quantize` / `forget` / `convert` subcommand yet.** Phase 6 will ship `amn convert model.safetensors --to bnb-nf4 -o quantised.safetensors`. Today the kernels are library-only.
- **No `encode_bnb4_double_quant_compute_*` convenience** (deferred to the Phase 6 conversion-CLI work).
- **No FP8 / GPTQ / AWQ / GGUF / IQ / TQ / MXFP4 encode** — all targeted at Phase 8.5 ("Lethe Encode Completion"), shipping after the BnB encode pipeline has been validated end-to-end through Python bindings in Phase 7.
- **No Python bindings yet.** Phase 7 (PyO3) exposes the encode + decode + convert primitives to the Python ecosystem.
- **No SIMD on encode hot paths.** Encode kernels are currently 4–6× slower than PyTorch's broadcast-vectorised quantize on `BnB4`, 32× slower on `INT8`. Phase 9 (CPU SIMD pass) is the natural target — the same loop-fission + `target-cpu=native` infrastructure that gave the decode path its 18–54× wins is the candidate retrofit on the encode side.

See [`ROADMAP.md`](../ROADMAP.md) for the full sequencing.

---

## See also

- [`README.md`](../README.md) — "BitsAndBytes Quantization (Lethe — Phase 5)" section with the cross-architecture fixture table
- [`CHANGELOG.md`](../CHANGELOG.md) — `[0.5.0]` entry block
- [`ROADMAP.md`](../ROADMAP.md) — Phase 5 step 1a/1b/1c (shipped) + Phase 8.5 (deferred encode kernels)
- [`docs/rust-ecosystem-comparison.md`](rust-ecosystem-comparison.md) — where anamnesis's encode-side coverage stands in the wider Rust + cross-language landscape
- [`docs/perf-experiments.md`](perf-experiments.md) — case-study entry for the sign-of-zero preservation rule (Experiment 7)
