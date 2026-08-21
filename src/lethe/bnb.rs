// SPDX-License-Identifier: MIT OR Apache-2.0

//! `BitsAndBytes` quantization (`NF4` / `FP4` 4-bit and `INT8`) — encode side.
//!
//! The inverse of [`remember::bnb`](mod@crate::remember::bnb). `NF4` and
//! `FP4` use a 16-entry codebook lookup (the codebook lives in the
//! `.weight.quant_map` companion tensor on disk; this module also exposes
//! canonical [`NF4_CODEBOOK`] and [`FP4_CODEBOOK`] constants for callers
//! that need to encode from scratch without a reference file). `INT8`
//! (`LLM.int8()`) uses per-row absmax with `round(x / scale)` clamped to
//! `[-128, 127]`.
//!
//! Every encode kernel is the byte-for-byte inverse of its decode
//! counterpart: feeding the `BF16` output of [`dequantize_bnb4_to_bf16`]
//! back through [`encode_bnb4`] with the same `absmax_data` and
//! `quant_map_data` reproduces the original `weight_data` bit-exactly.
//! This contract is encoded by the harness in [`round_trip`](mod@crate::lethe::round_trip)
//! and verified against real `bitsandbytes`-quantised fixtures in
//! `tests/cross_validation_bnb_encode.rs`.
//!
//! # References
//!
//! - Dettmers et al., "`LLM.int8()`: 8-bit Matrix Multiplication for
//!   Transformers at Scale", `NeurIPS` 2022 (`arXiv:2208.07339`)
//! - Dettmers et al., "`QLoRA`: Efficient Finetuning of Quantized Large
//!   Language Models", `NeurIPS` 2023 (`arXiv:2305.14314`)
//!
//! [`dequantize_bnb4_to_bf16`]: crate::remember::bnb::dequantize_bnb4_to_bf16

use crate::error::AnamnesisError;

// ---------------------------------------------------------------------------
// Canonical codebooks (bitsandbytes reference values)
// ---------------------------------------------------------------------------

/// Canonical `NF4` (4-bit `NormalFloat`) lookup table — 16 entries,
/// ascending sorted, derived from the quantiles of a standard normal
/// distribution.
///
/// Source: `bitsandbytes.functional.create_normal_map(offset=0.9677083)`
/// (the reference Python implementation). The same 16 values appear in
/// every `bitsandbytes`-quantised `.weight.quant_map` tensor for `NF4`
/// models, so this constant matches the on-disk codebook bit-exactly
/// and may be used as a drop-in `quant_map_data` substitute when
/// encoding from a raw `BF16` source that has no companion file.
///
/// Ascending sort is by IEEE 754 numeric order. Index `7` is exactly
/// `0.0`; indices `0..7` are negative, indices `8..16` are positive.
pub const NF4_CODEBOOK: [f32; 16] = [
    -1.0,
    -0.696_192_8,
    -0.525_073_05,
    -0.394_917_5,
    -0.284_441_38,
    -0.184_773_43,
    -0.091_050_036,
    0.0,
    0.079_580_3,
    0.160_930_2,
    0.246_112_3,
    0.337_915_24,
    0.440_709_83,
    0.562_617,
    0.722_956_84,
    1.0,
];

/// Canonical `FP4` (4-bit floating point) lookup table — 16 entries in
/// the `bitsandbytes` storage order (indexed directly by the 4-bit
/// value, not by numeric magnitude).
///
/// Source: `bitsandbytes` `FP4` decode kernel constants. The encoding
/// is sign (1 bit) + exponent-mantissa (3 bits) for `E2M1`-style `FP4`
/// with `bitsandbytes`-specific subnormal mapping. Indices `0..8` are
/// non-negative; indices `8..16` mirror them with the sign flipped
/// (note that index `8` is `-0.0`, distinct from index `0`'s `+0.0` in
/// the IEEE 754 bit pattern).
///
/// The `+0.0` / `-0.0` pair is the only place where two codebook entries
/// share an absolute value; the encoder's nearest-entry search uses
/// exact-bit-match priority to disambiguate, preserving the original
/// nibble across a `decode → encode` round trip.
pub const FP4_CODEBOOK: [f32; 16] = [
    0.0,
    0.005_208_333_5,
    0.666_666_7,
    1.0,
    0.333_333_34,
    0.5,
    0.166_666_67,
    0.25,
    -0.0,
    -0.005_208_333_5,
    -0.666_666_7,
    -1.0,
    -0.333_333_34,
    -0.5,
    -0.166_666_67,
    -0.25,
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Reads a little-endian `f32` from a byte slice at the given offset.
///
/// Mirrors `read_f32_le` in [`remember::bnb`](mod@crate::remember::bnb).
/// Duplicated locally rather than shared to keep the two modules
/// independently re-readable as decode / encode mirrors.
///
/// Returns `None` if the slice does not contain 4 bytes at `offset`.
fn read_f32_le(data: &[u8], offset: usize) -> Option<f32> {
    let bytes: &[u8] = data.get(offset..offset + 4)?;
    let arr: [u8; 4] = bytes.try_into().ok()?;
    Some(f32::from_le_bytes(arr))
}

/// Converts a `BF16` bit pattern to an `f32` value. Lossless — `BF16` is
/// exactly the upper 16 bits of an `f32` for finite values.
///
/// Inverse of `f32_bits_to_bf16_bits` in [`remember::fp8`](mod@crate::remember::fp8).
#[inline]
#[must_use]
fn bf16_bits_to_f32(bits: u16) -> f32 {
    // BITWISE: BF16 → f32 by shifting into upper 16 bits of IEEE 754
    f32::from_bits(u32::from(bits) << 16)
}

/// Applies the sign-magnitude correction to a chosen nibble, the
/// inverse of the decode-side `apply_sign_magnitude_zero` rule in
/// [`remember::bnb`](mod@crate::remember::bnb).
///
/// When the source `value` has its sign bit set (`x.is_sign_negative()`)
/// AND the nearest-search returned a lower-half index AND the
/// corresponding upper-half codebook entry has the same IEEE 754 bits
/// as the chosen entry, the nibble is shifted to the upper half (the
/// `bitsandbytes` sign-magnitude convention).
///
/// In practice this only fires for negative inputs that round to a
/// codebook-`+0.0` entry whose `+8` counterpart is also `+0.0` — i.e.,
/// the `FP4` `quant_map[8] == +0.0` pathological case. For every other
/// codebook layout the conditional collapses to a no-op:
///
/// - `NF4`: `codebook[7] = 0.0` but `codebook[15] = 1.0` → bits
///   differ → no shift.
/// - `FP4` with strictly-distinct upper / lower halves: `codebook[i+8]
///   ≠ codebook[i]` for `i ≠ 0` → no shift.
/// - Positive inputs: `is_sign_negative()` is false → no shift.
#[inline]
#[must_use]
fn apply_sign_magnitude_encode_correction(value: f32, nibble: u8, codebook: &[f32; 16]) -> u8 {
    if value.is_sign_negative() && nibble < 8 {
        let upper = nibble + 8;
        // INDEX: nibble < 8 ⇒ upper < 16; codebook has 16 entries
        #[allow(clippy::indexing_slicing, clippy::as_conversions)]
        let upper_bits = codebook[upper as usize].to_bits();
        // INDEX: nibble < 8; codebook has 16 entries
        #[allow(clippy::indexing_slicing, clippy::as_conversions)]
        let chosen_bits = codebook[nibble as usize].to_bits();
        if upper_bits == chosen_bits {
            return upper;
        }
    }
    nibble
}

/// Finds the index of the codebook entry nearest to `value`.
///
/// Linear scan over 16 entries — branch-predictable, fits in registers,
/// faster than binary search for this size. Ties between equally-near
/// entries break by **exact IEEE 754 bit-match priority**: if any
/// codebook entry has the same bit pattern as `value`, that entry wins
/// regardless of any other candidate's distance. This disambiguates the
/// `+0.0` / `-0.0` pair in `FP4` (both have distance zero from each
/// other; the exact-bit rule preserves sign across a round trip).
///
/// If no exact-bit match exists, the smallest index with the smallest
/// `(value - codebook[i]).abs()` wins (standard nearest-neighbour
/// convention with `<` rather than `<=`, so lower indices win equal-
/// distance ties).
#[inline]
#[must_use]
fn nearest_codebook_index(value: f32, codebook: &[f32; 16]) -> u8 {
    let val_bits = value.to_bits();
    let mut best_idx: u8 = 0;
    let mut best_dist = f32::INFINITY;
    let mut best_exact = false;
    for (i, &entry) in codebook.iter().enumerate() {
        let exact = entry.to_bits() == val_bits;
        let dist = (value - entry).abs();
        // EXPLICIT: exact-bit-match priority dominates strict-distance
        // comparison so that +0/-0 round-trip correctly under FP4.
        let take = if exact && !best_exact {
            true
        } else if !exact && best_exact {
            false
        } else {
            dist < best_dist
        };
        if take {
            // CAST: i is 0..16, fits u8 trivially
            #[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
            let i_u8 = i as u8;
            best_idx = i_u8;
            best_dist = dist;
            best_exact = exact;
        }
    }
    best_idx
}

/// Parses 64 bytes of `F32` little-endian codebook into `[f32; 16]`.
fn parse_codebook(quant_map_data: &[u8]) -> crate::Result<[f32; 16]> {
    if quant_map_data.len() != 64 {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "BnB4 quant_map must be 64 bytes (16xF32), got {}",
                quant_map_data.len()
            ),
        });
    }
    let mut codebook = [0.0f32; 16];
    for (i, slot) in codebook.iter_mut().enumerate() {
        *slot = read_f32_le(quant_map_data, i * 4).ok_or_else(|| AnamnesisError::Parse {
            reason: "BnB4 quant_map read out of bounds".into(),
        })?;
    }
    Ok(codebook)
}

/// Parses an absmax byte slice into a freshly allocated `Vec<f32>`.
///
/// The on-disk absmax tensor is `F32` little-endian; this is the same
/// byte layout the decode side reads. `num_blocks * 4` bytes expected.
fn parse_absmax(absmax_data: &[u8], num_blocks: usize) -> crate::Result<Vec<f32>> {
    let expected_bytes = num_blocks
        .checked_mul(4)
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "absmax byte count overflow".into(),
        })?;
    if absmax_data.len() != expected_bytes {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "BnB4 absmax byte count mismatch: expected {expected_bytes}, got {}",
                absmax_data.len()
            ),
        });
    }
    let mut absmax = vec![0.0f32; num_blocks];
    for (i, slot) in absmax.iter_mut().enumerate() {
        *slot = read_f32_le(absmax_data, i * 4).ok_or_else(|| AnamnesisError::Parse {
            reason: format!("BnB4 absmax read out of bounds at block {i}"),
        })?;
    }
    Ok(absmax)
}

// ---------------------------------------------------------------------------
// NF4/FP4 encode (4-bit, lookup-table based)
// ---------------------------------------------------------------------------

/// Core `NF4` / `FP4` encode: accepts a pre-parsed `f32` absmax slice
/// and a pre-parsed codebook.
///
/// Mirrors `dequantize_bnb4_core` in [`remember::bnb`](mod@crate::remember::bnb).
/// Callers are responsible for validation; this function assumes inputs
/// are dimensionally consistent.
fn encode_bnb4_core(
    bf16_data: &[u8],
    absmax: &[f32],
    codebook: &[f32; 16],
    total_elements: usize,
    block_size: usize,
) -> crate::Result<Vec<u8>> {
    // --- Allocate output ---
    let out_byte_len = total_elements
        .checked_div(2)
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "BnB4 encode total_elements must be even".into(),
        })?;
    let mut output = vec![0u8; out_byte_len];

    // --- Per-block encoding with loop fission ---
    let bytes_per_block = block_size / 2;
    // Scratch buffer for unpacked f32 values (one block at a time, fits in L1)
    let mut scratch = vec![0.0f32; block_size];

    for (block_idx, &block_absmax) in absmax.iter().enumerate() {
        // Pre-slice validated ranges (two-level bounds checking per CONVENTIONS.md)
        let bf16_byte_start = block_idx
            .checked_mul(block_size)
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!("BnB4 encode bf16 start overflow at block {block_idx}"),
            })?;
        let bf16_byte_end =
            bf16_byte_start
                .checked_add(block_size * 2)
                .ok_or_else(|| AnamnesisError::Parse {
                    reason: format!("BnB4 encode bf16 end overflow at block {block_idx}"),
                })?;
        let bf16_block = bf16_data
            .get(bf16_byte_start..bf16_byte_end)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!("BnB4 encode bf16 block {block_idx} out of bounds"),
            })?;
        let o_start = block_idx * bytes_per_block;
        let o_end = o_start + bytes_per_block;
        let out_block = output
            .get_mut(o_start..o_end)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!("BnB4 encode output block {block_idx} out of bounds"),
            })?;

        // --- Pass 1 (unpack): BF16 bytes → f32 scratch ---
        // Branch-free shift; pre-slice was validated above.
        // INDEX: scratch.len() == block_size, guaranteed by vec![0.0f32; block_size]
        #[allow(clippy::indexing_slicing)]
        let scratch_block = &mut scratch[..block_size];
        // VECTORIZED: scalar fallback — Pass-1 of fission; BF16-byte
        // unpacking crosses byte/integer/float domains and is
        // intentionally scalar (mirrors the decode-side Pass-1 rule in
        // src/remember/bnb.rs).
        for (bf16_pair, slot) in bf16_block.chunks_exact(2).zip(scratch_block.iter_mut()) {
            // INDEX: chunks_exact(2) guarantees exactly 2 bytes per pair
            #[allow(clippy::indexing_slicing)]
            let bits = u16::from_le_bytes([bf16_pair[0], bf16_pair[1]]);
            *slot = bf16_bits_to_f32(bits);
        }

        // --- Pass 2 (normalize + lookup): f32 / absmax → nearest codebook → pack ---
        // The codebook search is a gather over 16 entries; not currently
        // auto-vectorised, but the divide-and-search inner kernel is
        // identical for every kernel in the Phase 8.5 family and is the
        // natural target for an explicit SIMD pass in a future phase.
        // INDEX: scratch_view.len() == block_size, paired with chunks_exact(2)
        #[allow(clippy::indexing_slicing)]
        let scratch_view = &scratch[..block_size];
        // VECTORIZED: scalar fallback — the codebook search does not
        // auto-vectorise and will not in this shape. Verified in `--emit=asm`,
        // x86-64 target-cpu=native, opt-level=3: `encode_bnb4_core` emits **no**
        // packed float instruction for this loop and instead makes two `call`s
        // to `nearest_codebook_index`, which stays out of line because it is a
        // data-dependent linear search over 16 entries. The `%ymm` traffic that
        // does appear in this function (`vpmovzxwd` + `vpslld`) is pass 1's
        // `BF16` → `f32` widening, not this loop.
        //
        // Escalating would mean an explicit gather or a 16-way blend tree
        // (`CONVENTIONS.md` § when auto-vectorization is not enough, rung 1),
        // which Experiment 10 says to attempt only after a roofline shows the
        // loop is compute-bound. It is on the encode path, which no benchmark
        // currently reports as hot, so the honest state is `scalar fallback`
        // rather than a speculative rewrite.
        for (pair, out_byte) in scratch_view.chunks_exact(2).zip(out_block.iter_mut()) {
            // INDEX: chunks_exact(2) guarantees exactly 2 f32 per pair
            #[allow(clippy::indexing_slicing)]
            let (val_first, val_second) = (pair[0], pair[1]);
            // If absmax is zero, every value collapses to zero;
            // divide-by-zero would propagate NaN/inf into the search
            // and produce undefined nibbles. Decode treats absmax = 0
            // as a zero block; the encoder mirrors that by emitting
            // nibble 0 (the canonical zero-yielding entry for both
            // NF4 (codebook[7] == 0) and FP4 (codebook[0] == +0));
            // we use the codebook itself to find the lowest-index
            // exact-zero entry so the choice is codebook-driven, not
            // hard-coded.
            let (norm_first, norm_second) = if block_absmax == 0.0 {
                (0.0_f32, 0.0_f32)
            } else {
                (val_first / block_absmax, val_second / block_absmax)
            };
            let first_raw = nearest_codebook_index(norm_first, codebook);
            let second_raw = nearest_codebook_index(norm_second, codebook);
            // Sign-magnitude correction: mirror the decode-side
            // sign-of-zero preservation. For codebooks where
            // codebook[i] == codebook[i+8] in bits (FP4's +0/+0 pair),
            // a negative-sign input that rounds to the lower-half
            // index is shifted to the upper-half index so the round
            // trip preserves the nibble byte-exactly.
            let first_nibble =
                apply_sign_magnitude_encode_correction(norm_first, first_raw, codebook);
            let second_nibble =
                apply_sign_magnitude_encode_correction(norm_second, second_raw, codebook);
            // BITWISE: pack the FIRST element in the high nibble (bits [7:4])
            // and the SECOND in the low nibble (bits [3:0]) — the bitsandbytes
            // order, inverse of the decode-side unpack (`byte >> 4` first,
            // `byte & 0x0F` second).
            *out_byte = (first_nibble << 4) | (second_nibble & 0x0F);
        }
    }

    Ok(output)
}

/// Encodes `BF16` weights to `BitsAndBytes` `NF4` / `FP4` packed nibbles.
///
/// The inverse of [`dequantize_bnb4_to_bf16`]. Each pair of consecutive
/// `BF16` elements is encoded into one byte: the first element's
/// nearest-codebook nibble in the high position (`byte >> 4`), the
/// second in the low position (`byte & 0x0F`) — the `bitsandbytes`
/// packing order. The caller supplies the
/// codebook (`quant_map_data` — usually the `.weight.quant_map`
/// companion tensor read from the source `.safetensors` file) and the
/// per-block absmax values (`absmax_data` — usually freshly derived
/// from the source `BF16`; see [`encode_bnb4_compute_absmax`] for the
/// computed-absmax convenience).
///
/// # Arguments
///
/// - `bf16_data` — `BF16` little-endian bytes (`total_elements * 2` bytes).
/// - `absmax_data` — `F32` little-endian per-block absmax values
///   (`num_blocks * 4` bytes, where `num_blocks = total_elements / block_size`).
/// - `quant_map_data` — `F32` little-endian 16-entry codebook
///   (`64` bytes). Pass [`NF4_CODEBOOK`] or [`FP4_CODEBOOK`] serialised
///   to bytes if no on-disk file is available.
/// - `total_elements` — total number of `BF16` elements (= output bytes × 2).
/// - `block_size` — elements per absmax block (typically 64).
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if:
/// - `block_size` is zero, or `total_elements` is not divisible by
///   `block_size`,
/// - `bf16_data.len() != total_elements * 2`,
/// - `absmax_data.len() != num_blocks * 4`,
/// - `quant_map_data.len() != 64`.
///
/// # Memory
///
/// Allocates `total_elements / 2` bytes for the packed output, plus a
/// scratch buffer of `block_size * 4` bytes for loop fission (fits in
/// L1 cache).
///
/// [`dequantize_bnb4_to_bf16`]: crate::remember::bnb::dequantize_bnb4_to_bf16
pub fn encode_bnb4(
    bf16_data: &[u8],
    absmax_data: &[u8],
    quant_map_data: &[u8],
    total_elements: usize,
    block_size: usize,
) -> crate::Result<Vec<u8>> {
    // --- Validation ---
    if block_size == 0 {
        return Err(AnamnesisError::Parse {
            reason: "BnB encode block_size must be > 0".into(),
        });
    }
    // Odd block_size truncates `bytes_per_block = block_size / 2` in
    // `encode_bnb4_core` → mis-aligned blocks → wrong packed output. Mirror of
    // the decode-side guard in `remember::bnb`.
    if !block_size.is_multiple_of(2) {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "BnB4 encode block_size must be even (two nibbles per byte), got {block_size}"
            ),
        });
    }
    if !total_elements.is_multiple_of(2) {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "BnB4 encode total_elements ({total_elements}) must be even \
                 (two nibbles per byte)"
            ),
        });
    }
    let expected_bf16_bytes =
        total_elements
            .checked_mul(2)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: "BnB4 encode bf16 byte count overflow".into(),
            })?;
    if bf16_data.len() != expected_bf16_bytes {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "BnB4 encode bf16 byte count mismatch: expected {expected_bf16_bytes} for \
                 {total_elements} elements, got {}",
                bf16_data.len()
            ),
        });
    }
    if !total_elements.is_multiple_of(block_size) {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "BnB4 encode total_elements ({total_elements}) not divisible by \
                 block_size ({block_size})"
            ),
        });
    }
    let num_blocks = total_elements / block_size;
    let codebook = parse_codebook(quant_map_data)?;
    let absmax = parse_absmax(absmax_data, num_blocks)?;
    encode_bnb4_core(bf16_data, &absmax, &codebook, total_elements, block_size)
}

/// Convenience wrapper: derives per-block absmax from the `BF16` input,
/// then encodes via [`encode_bnb4`].
///
/// Use this when quantising a fresh `BF16` source tensor (Phase 6
/// `safetensors BF16 → BnB-NF4` conversion path). The returned pair is
/// `(packed_weight_bytes, absmax_bytes)` — both little-endian.
///
/// # Arguments
///
/// - `bf16_data` — `BF16` little-endian bytes.
/// - `quant_map_data` — 16-entry codebook (64 bytes), `F32` LE.
/// - `total_elements` — total number of `BF16` elements.
/// - `block_size` — elements per absmax block (typically 64).
///
/// # Errors
///
/// Same as [`encode_bnb4`], plus the input dimensional checks.
///
/// # Memory
///
/// Allocates `total_elements / 2` bytes for the packed output, plus
/// `num_blocks * 4` bytes for the derived absmax, plus a scratch
/// buffer of `block_size * 4` bytes (fits in L1 cache).
pub fn encode_bnb4_compute_absmax(
    bf16_data: &[u8],
    quant_map_data: &[u8],
    total_elements: usize,
    block_size: usize,
) -> crate::Result<(Vec<u8>, Vec<u8>)> {
    if block_size == 0 {
        return Err(AnamnesisError::Parse {
            reason: "BnB encode block_size must be > 0".into(),
        });
    }
    // Odd block_size truncates `bytes_per_block = block_size / 2` in
    // `encode_bnb4_core` → mis-aligned blocks → wrong packed output. Mirror of
    // the decode-side guard in `remember::bnb`.
    if !block_size.is_multiple_of(2) {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "BnB4 encode block_size must be even (two nibbles per byte), got {block_size}"
            ),
        });
    }
    if !total_elements.is_multiple_of(block_size) {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "BnB4 encode total_elements ({total_elements}) not divisible by \
                 block_size ({block_size})"
            ),
        });
    }
    let expected_bf16_bytes =
        total_elements
            .checked_mul(2)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: "BnB4 encode bf16 byte count overflow".into(),
            })?;
    if bf16_data.len() != expected_bf16_bytes {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "BnB4 encode bf16 byte count mismatch: expected {expected_bf16_bytes} for \
                 {total_elements} elements, got {}",
                bf16_data.len()
            ),
        });
    }
    let num_blocks = total_elements / block_size;
    let mut absmax = vec![0.0f32; num_blocks];
    for (block_idx, slot) in absmax.iter_mut().enumerate() {
        let bf16_byte_start = block_idx * block_size * 2;
        let bf16_byte_end = bf16_byte_start + block_size * 2;
        let bf16_block = bf16_data
            .get(bf16_byte_start..bf16_byte_end)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!("BnB4 encode bf16 block {block_idx} out of bounds"),
            })?;
        let mut max_abs = 0.0_f32;
        for pair in bf16_block.chunks_exact(2) {
            // INDEX: chunks_exact(2) guarantees exactly 2 bytes per pair
            #[allow(clippy::indexing_slicing)]
            let bits = u16::from_le_bytes([pair[0], pair[1]]);
            let v = bf16_bits_to_f32(bits).abs();
            if v > max_abs {
                max_abs = v;
            }
        }
        *slot = max_abs;
    }
    let absmax_bytes: Vec<u8> = absmax.iter().flat_map(|v| v.to_le_bytes()).collect();
    let codebook = parse_codebook(quant_map_data)?;
    let weight_bytes = encode_bnb4_core(bf16_data, &absmax, &codebook, total_elements, block_size)?;
    Ok((weight_bytes, absmax_bytes))
}

// ---------------------------------------------------------------------------
// NF4/FP4 double-quant encode (4-bit, lookup-table based, nested absmax)
// ---------------------------------------------------------------------------

/// Recovers the per-block `f32` absmax values from `bitsandbytes` double-quant
/// metadata.
///
/// Mirrors the recovery step inside
/// [`dequantize_bnb4_double_quant_to_bf16`](crate::remember::bnb::dequantize_bnb4_double_quant_to_bf16):
/// for each block `i`, reads the `U8` quantised absmax byte, looks up the
/// corresponding entry in the 256-entry nested codebook, multiplies by
/// the per-nested-block `nested_absmax` scale, and adds the
/// `nested_offset` (the `bitsandbytes` absmax-mean compression bias).
/// The recovered `Vec<f32>` is the same value the decoder uses, so
/// encoding `BF16` produced by decode through `encode_bnb4_core` with
/// this recovered absmax round-trips byte-exactly.
fn recover_double_quant_absmax(
    absmax_data: &[u8],
    nested_absmax_data: &[u8],
    nested_quant_map_data: &[u8],
    nested_offset: f32,
    nested_block_size: usize,
) -> crate::Result<Vec<f32>> {
    if nested_block_size == 0 {
        return Err(AnamnesisError::Parse {
            reason: "BnB nested_block_size must be > 0".into(),
        });
    }
    if nested_quant_map_data.len() != 1024 {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "BnB4 nested_quant_map must be 1024 bytes (256xF32), got {}",
                nested_quant_map_data.len()
            ),
        });
    }
    let num_blocks = absmax_data.len();
    let num_nested_blocks = if num_blocks.is_multiple_of(nested_block_size) {
        num_blocks / nested_block_size
    } else {
        num_blocks / nested_block_size + 1
    };
    let expected_nested_absmax_bytes =
        num_nested_blocks
            .checked_mul(4)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: "BnB4 encode nested absmax byte count overflow".into(),
            })?;
    if nested_absmax_data.len() != expected_nested_absmax_bytes {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "BnB4 encode nested_absmax byte count mismatch: expected \
                 {expected_nested_absmax_bytes}, got {}",
                nested_absmax_data.len()
            ),
        });
    }

    // Pre-load nested codebook (256 entries).
    let mut nested_codebook = [0.0_f32; 256];
    for (i, slot) in nested_codebook.iter_mut().enumerate() {
        *slot = read_f32_le(nested_quant_map_data, i * 4).ok_or_else(|| AnamnesisError::Parse {
            reason: "BnB4 encode nested_quant_map read out of bounds".into(),
        })?;
    }

    // Recover per-block absmax: nested_codebook[absmax_byte] * nested_absmax[nested_block_idx].
    let mut recovered = vec![0.0_f32; num_blocks];
    for (i, (&absmax_byte, slot)) in absmax_data.iter().zip(recovered.iter_mut()).enumerate() {
        let nested_block_idx = i / nested_block_size;
        let nested_absmax_val =
            read_f32_le(nested_absmax_data, nested_block_idx * 4).ok_or_else(|| {
                AnamnesisError::Parse {
                    reason: format!(
                        "BnB4 encode nested_absmax read out of bounds at block {nested_block_idx}"
                    ),
                }
            })?;
        // CAST: u8 -> usize, byte value 0-255 used as lookup index
        #[allow(clippy::as_conversions)]
        let idx = absmax_byte as usize;
        // INDEX: idx is 0-255, nested_codebook has 256 entries
        #[allow(clippy::indexing_slicing)]
        let entry = nested_codebook[idx];
        *slot = entry * nested_absmax_val + nested_offset;
    }
    Ok(recovered)
}

/// Encodes `BF16` weights to `BitsAndBytes` `NF4` / `FP4` packed nibbles
/// using the **double-quant** absmax layout.
///
/// The inverse of
/// [`dequantize_bnb4_double_quant_to_bf16`](crate::remember::bnb::dequantize_bnb4_double_quant_to_bf16).
/// The caller supplies the already-quantised `U8` absmax bytes plus the
/// nested-quant metadata (`nested_absmax`, `nested_quant_map`); this
/// function recovers the per-block `f32` absmax using the same formula
/// the decoder applies, then re-encodes `BF16` to packed nibbles via the
/// same nearest-codebook search as [`encode_bnb4`]. Round-trip is
/// byte-exact when the supplied metadata matches the metadata the
/// decoder originally read: `encode_bnb4_double_quant(decode(weight,
/// absmax, qm, n_absmax, n_qm)) == weight`.
///
/// This strict mirror is the round-trip API. A future
/// `encode_bnb4_double_quant_compute_*` convenience that derives `absmax`,
/// `nested_absmax`, and the nested codebook from a fresh `BF16` source —
/// needed by the Phase 6 "any input -> BnB-NF4 safetensors" conversion
/// path — is intentionally out of scope for Phase 5 step 1c.
///
/// # Arguments
///
/// - `bf16_data` — `BF16` little-endian bytes (`total_elements * 2` bytes).
/// - `absmax_data` — `U8` quantised absmax indices (one byte per block,
///   `total_elements / block_size` bytes total).
/// - `quant_map_data` — `F32` little-endian 16-entry main codebook
///   (`64` bytes). Pass [`NF4_CODEBOOK`] or [`FP4_CODEBOOK`] serialised
///   to bytes if no on-disk file is available.
/// - `nested_absmax_data` — `F32` little-endian per-nested-block scale.
/// - `nested_quant_map_data` — `F32` little-endian 256-entry nested codebook
///   (`1024` bytes).
/// - `nested_offset` — additive absmax offset from the `quant_state` blob
///   (`bitsandbytes` `QuantState.offset`); must equal the value the decoder
///   used for the round trip to close.
/// - `total_elements` — total number of `BF16` elements.
/// - `block_size` — elements per absmax block (typically 64).
/// - `nested_block_size` — absmax indices per nested-absmax block (typically 256).
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if:
/// - `block_size` or `nested_block_size` is zero, or `total_elements` is
///   not divisible by `block_size`,
/// - `bf16_data.len() != total_elements * 2`,
/// - `absmax_data.len() != total_elements / block_size`,
/// - `nested_quant_map_data.len() != 1024`,
/// - `nested_absmax_data.len() != ceil(num_blocks / nested_block_size) * 4`,
/// - `quant_map_data.len() != 64`.
///
/// # Memory
///
/// Allocates `total_elements / 2` bytes for the packed output, an `f32`
/// recovered-absmax array (`num_blocks * 4` bytes), and a scratch buffer
/// of `block_size * 4` bytes for loop fission (fits in L1 cache). No
/// intermediate byte serialisation.
#[allow(clippy::too_many_arguments)]
pub fn encode_bnb4_double_quant(
    bf16_data: &[u8],
    absmax_data: &[u8],
    quant_map_data: &[u8],
    nested_absmax_data: &[u8],
    nested_quant_map_data: &[u8],
    nested_offset: f32,
    total_elements: usize,
    block_size: usize,
    nested_block_size: usize,
) -> crate::Result<Vec<u8>> {
    // --- Validation ---
    if block_size == 0 {
        return Err(AnamnesisError::Parse {
            reason: "BnB encode block_size must be > 0".into(),
        });
    }
    // Odd block_size truncates `bytes_per_block = block_size / 2` in
    // `encode_bnb4_core` → mis-aligned blocks → wrong packed output. Mirror of
    // the decode-side guard in `remember::bnb`.
    if !block_size.is_multiple_of(2) {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "BnB4 encode block_size must be even (two nibbles per byte), got {block_size}"
            ),
        });
    }
    if !total_elements.is_multiple_of(2) {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "BnB4 encode total_elements ({total_elements}) must be even \
                 (two nibbles per byte)"
            ),
        });
    }
    let expected_bf16_bytes =
        total_elements
            .checked_mul(2)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: "BnB4 encode bf16 byte count overflow".into(),
            })?;
    if bf16_data.len() != expected_bf16_bytes {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "BnB4 encode bf16 byte count mismatch: expected {expected_bf16_bytes} for \
                 {total_elements} elements, got {}",
                bf16_data.len()
            ),
        });
    }
    if !total_elements.is_multiple_of(block_size) {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "BnB4 encode total_elements ({total_elements}) not divisible by \
                 block_size ({block_size})"
            ),
        });
    }
    let num_blocks = total_elements / block_size;
    if absmax_data.len() != num_blocks {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "BnB4 double-quant encode: absmax byte count mismatch: expected \
                 {num_blocks} (one byte per block), got {}",
                absmax_data.len()
            ),
        });
    }

    // --- Recovery + delegation ---
    let recovered_absmax = recover_double_quant_absmax(
        absmax_data,
        nested_absmax_data,
        nested_quant_map_data,
        nested_offset,
        nested_block_size,
    )?;
    let codebook = parse_codebook(quant_map_data)?;
    encode_bnb4_core(
        bf16_data,
        &recovered_absmax,
        &codebook,
        total_elements,
        block_size,
    )
}

// ---------------------------------------------------------------------------
// INT8 encode (LLM.int8(), per-row absmax)
// ---------------------------------------------------------------------------

/// Encodes `BF16` weights to `BitsAndBytes` `INT8` (`LLM.int8()`) per-row
/// quantisation.
///
/// The inverse of [`dequantize_bnb_int8_to_bf16`]. For each row, the
/// per-row scale is `scb / 127.0`; each `BF16` element becomes
/// `round(value / scale)` clamped to the `i8` range `[-128, 127]` and
/// stored as a `u8` two's-complement byte. The caller supplies the
/// per-row `SCB` values (`scb_data` — usually freshly derived from the
/// source `BF16`; see [`encode_bnb_int8_compute_scb`] for the
/// computed-`SCB` convenience).
///
/// The clamp matters: a `BF16` value at exact `± SCB` could round-trip
/// to `± 128` under naive truncation, but `i8` maxes at `+127`. The
/// asymmetric `i8` range is the reason `bitsandbytes` uses `± 127` (not
/// `± 128`) in the decode scale formula.
///
/// # Arguments
///
/// - `bf16_data` — `BF16` little-endian bytes (`out_features * in_features * 2` bytes).
/// - `scb_data` — `F32` little-endian per-row absmax values
///   (`out_features * 4` bytes).
/// - `out_features` — number of output rows.
/// - `in_features` — number of input columns.
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if any of the input byte counts
/// disagree with the declared `out_features` / `in_features`.
///
/// # Memory
///
/// Allocates `out_features * in_features` bytes for the `i8` output (one
/// byte per element). No scratch buffer — the inner loop is 1:1 (`BF16`
/// → `i8`).
///
/// [`dequantize_bnb_int8_to_bf16`]: crate::remember::bnb::dequantize_bnb_int8_to_bf16
pub fn encode_bnb_int8(
    bf16_data: &[u8],
    scb_data: &[u8],
    out_features: usize,
    in_features: usize,
) -> crate::Result<Vec<u8>> {
    let total_elements =
        out_features
            .checked_mul(in_features)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: "BnB INT8 encode element count overflow".into(),
            })?;
    let expected_bf16_bytes =
        total_elements
            .checked_mul(2)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: "BnB INT8 encode bf16 byte count overflow".into(),
            })?;
    if bf16_data.len() != expected_bf16_bytes {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "BnB INT8 encode bf16 byte count mismatch: expected {expected_bf16_bytes}, got {}",
                bf16_data.len()
            ),
        });
    }
    let expected_scb_bytes = out_features
        .checked_mul(4)
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "BnB INT8 encode SCB byte count overflow".into(),
        })?;
    if scb_data.len() != expected_scb_bytes {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "BnB INT8 encode SCB byte count mismatch: expected {expected_scb_bytes}, got {}",
                scb_data.len()
            ),
        });
    }

    let mut output = vec![0u8; total_elements];

    for row in 0..out_features {
        let scb_val = read_f32_le(scb_data, row * 4).ok_or_else(|| AnamnesisError::Parse {
            reason: format!("BnB INT8 encode SCB read out of bounds at row {row}"),
        })?;
        // Per-row scale: SCB / 127.0 (identical to the decode side).
        let scale = scb_val / 127.0;

        // Pre-slice for branch-free inner loop (two-level bounds checking)
        let bf16_byte_start =
            row.checked_mul(in_features * 2)
                .ok_or_else(|| AnamnesisError::Parse {
                    reason: format!("BnB INT8 encode bf16 row start overflow at row {row}"),
                })?;
        let bf16_byte_end = bf16_byte_start + in_features * 2;
        let bf16_row = bf16_data
            .get(bf16_byte_start..bf16_byte_end)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!("BnB INT8 encode bf16 row {row} out of bounds"),
            })?;
        let o_start = row * in_features;
        let o_end = o_start + in_features;
        let out_row = output
            .get_mut(o_start..o_end)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!("BnB INT8 encode output row {row} out of bounds"),
            })?;

        // Hot loop: 1:1 BF16 → i8, scale hoisted.
        // VECTORIZED: confirmed AVX2 vdivps + vroundps + vmaxps + vminps +
        // vpackusdw on %ymm (8-wide) in `--emit=asm`, x86-64
        // target-cpu=native, opt-level=3. The whole encode kernel vectorises:
        // `vpmovzxwd` + `vpslld` widen `BF16` to `f32`, `vdivps` applies the
        // hoisted row scale, `vroundps $11` is round-to-nearest-even (mode 3
        // with the precision-exception mask), `vmaxps`/`vminps` are the
        // ±127 clamp, and `vpackusdw` narrows back to bytes.
        //
        // The `scale == 0.0` test inside the loop body is loop-invariant, so
        // `LLVM` hoists it and emits two loop versions rather than a per-element
        // branch; the residual scalar float instructions in this function are
        // that second version plus the ragged tail, not the main path.
        //
        // This annotation had read `pending` since the kernel was written, which
        // `CONVENTIONS.md` makes a release blocker at the next `vX.Y.0`. It is
        // resolved here rather than at v0.8.0 so the Python bindings do not
        // inherit an open one.
        for (bf16_pair, out_byte) in bf16_row.chunks_exact(2).zip(out_row.iter_mut()) {
            // INDEX: chunks_exact(2) guarantees exactly 2 bytes per pair
            #[allow(clippy::indexing_slicing)]
            let bits = u16::from_le_bytes([bf16_pair[0], bf16_pair[1]]);
            let v = bf16_bits_to_f32(bits);
            let scaled = if scale == 0.0 {
                // SCB == 0 ⇒ decode would produce 0 for the whole row;
                // mirror that by encoding 0 for every element (rather than
                // propagating NaN from 0/0).
                0.0_f32
            } else {
                v / scale
            };
            // Round to nearest (banker's not required — `bitsandbytes`
            // uses `round_()` which is round-half-to-even on PyTorch CPU,
            // but in practice no real BF16 value lands exactly on a half
            // because BF16's mantissa resolution is coarser than 0.5; so
            // `round` ties effectively never fire here).
            let rounded = scaled.round();
            // Clamp to i8 range.
            let clamped = rounded.clamp(-128.0, 127.0);
            // CAST: f32 → i8, value is in [-128, 127] after clamp,
            // never NaN (caller guards against NaN BF16 input via the
            // dimensional checks above; if a NaN slipped in, clamp
            // preserves NaN, but `as i8` on NaN is implementation-defined
            // → defensive cast via i32 first).
            #[allow(
                clippy::as_conversions,
                clippy::cast_possible_truncation,
                clippy::cast_possible_wrap
            )]
            let signed = clamped as i32 as i8;
            // CAST: i8 → u8 by two's-complement reinterpretation (the
            // safetensors file stores I8 as raw bytes; -1 → 0xFF, etc.)
            *out_byte = signed.cast_unsigned();
        }
    }

    Ok(output)
}

/// Convenience wrapper: derives per-row `SCB` from the `BF16` input,
/// then encodes via [`encode_bnb_int8`].
///
/// The returned pair is `(weight_bytes, scb_bytes)` — both little-endian.
///
/// # Arguments
///
/// - `bf16_data` — `BF16` little-endian bytes (`out_features * in_features * 2` bytes).
/// - `out_features` — number of output rows.
/// - `in_features` — number of input columns.
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if input dimensions overflow or
/// `bf16_data` length is wrong.
///
/// # Memory
///
/// Allocates `out_features * in_features` bytes for the `i8` output plus
/// `out_features * 4` bytes for the derived `SCB`.
pub fn encode_bnb_int8_compute_scb(
    bf16_data: &[u8],
    out_features: usize,
    in_features: usize,
) -> crate::Result<(Vec<u8>, Vec<u8>)> {
    let total_elements =
        out_features
            .checked_mul(in_features)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: "BnB INT8 encode element count overflow".into(),
            })?;
    let expected_bf16_bytes =
        total_elements
            .checked_mul(2)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: "BnB INT8 encode bf16 byte count overflow".into(),
            })?;
    if bf16_data.len() != expected_bf16_bytes {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "BnB INT8 encode bf16 byte count mismatch: expected {expected_bf16_bytes}, got {}",
                bf16_data.len()
            ),
        });
    }
    let mut scb = vec![0.0f32; out_features];
    for (row, slot) in scb.iter_mut().enumerate() {
        let bf16_byte_start = row * in_features * 2;
        let bf16_byte_end = bf16_byte_start + in_features * 2;
        let bf16_row = bf16_data
            .get(bf16_byte_start..bf16_byte_end)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!("BnB INT8 encode bf16 row {row} out of bounds"),
            })?;
        let mut max_abs = 0.0_f32;
        for pair in bf16_row.chunks_exact(2) {
            // INDEX: chunks_exact(2) guarantees exactly 2 bytes per pair
            #[allow(clippy::indexing_slicing)]
            let bits = u16::from_le_bytes([pair[0], pair[1]]);
            let v = bf16_bits_to_f32(bits).abs();
            if v > max_abs {
                max_abs = v;
            }
        }
        *slot = max_abs;
    }
    let scb_bytes: Vec<u8> = scb.iter().flat_map(|v| v.to_le_bytes()).collect();
    let weight_bytes = encode_bnb_int8(bf16_data, &scb_bytes, out_features, in_features)?;
    Ok((weight_bytes, scb_bytes))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(
    clippy::panic,
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::float_cmp,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap
)]
mod tests {
    use super::*;
    use crate::lethe::round_trip::{
        assert_bnb_int8_decode_encode_round_trip, assert_bnb4_decode_encode_round_trip,
    };
    use crate::remember::bnb::{dequantize_bnb_int8_to_bf16, dequantize_bnb4_to_bf16};

    fn f32_to_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn bf16_bytes_from_f32(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|v| {
                let bits = v.to_bits();
                let lsb = (bits >> 16) & 1;
                let rounding_bias = 0x7FFF_u32 + lsb;
                let bf16 = (bits.wrapping_add(rounding_bias) >> 16) as u16;
                bf16.to_le_bytes()
            })
            .collect()
    }

    // --- nearest_codebook_index ---

    #[test]
    fn nearest_picks_closest_entry() {
        // Exactly-representable f32 spacing (powers of 2) so equidistant
        // values are unambiguous ties under IEEE 754.
        let cb = [
            -0.5, -0.375, -0.25, -0.125, 0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0,
            1.125, 1.25, 1.375,
        ];
        assert_eq!(nearest_codebook_index(0.0, &cb), 4);
        // 0.0625 is exactly between cb[4]=0.0 and cb[5]=0.125 → tie, lower index.
        assert_eq!(nearest_codebook_index(0.0625, &cb), 4);
        // 0.07 (0.0625 + a tick) is closer to cb[5]=0.125.
        assert_eq!(nearest_codebook_index(0.07, &cb), 5);
        // 1.3125 is exactly between cb[14]=1.25 and cb[15]=1.375 → tie, lower index.
        assert_eq!(nearest_codebook_index(1.3125, &cb), 14);
        // 1.32 is closer to cb[15]=1.375.
        assert_eq!(nearest_codebook_index(1.32, &cb), 15);
        // Below the lowest entry clamps to index 0.
        assert_eq!(nearest_codebook_index(-10.0, &cb), 0);
        // Above the highest entry clamps to index 15.
        assert_eq!(nearest_codebook_index(10.0, &cb), 15);
    }

    #[test]
    fn nearest_preserves_signed_zero() {
        // Pathological FP4 case: codebook has +0 and -0 in distinct slots.
        let mut cb = [0.0f32; 16];
        cb[0] = 0.0;
        cb[8] = -0.0;
        for (i, slot) in cb.iter_mut().enumerate().take(8).skip(1) {
            *slot = i as f32 * 0.1;
        }
        for (i, slot) in cb.iter_mut().enumerate().take(16).skip(9) {
            *slot = -(i as f32 - 8.0) * 0.1;
        }
        assert_eq!(nearest_codebook_index(0.0, &cb), 0);
        assert_eq!(nearest_codebook_index(-0.0, &cb), 8);
    }

    // --- encode_bnb4 round-trip on synthetic inputs ---

    #[test]
    fn encode_bnb4_round_trips_linear_codebook() {
        // codebook[i] = (i - 7.5) * 0.1, all distinct.
        let mut cb = [0.0f32; 16];
        for (i, slot) in cb.iter_mut().enumerate() {
            *slot = (i as f32 - 7.5) * 0.1;
        }
        assert_bnb4_decode_encode_round_trip(
            &cb,
            &[1.0, 2.0, 0.5, 8.0],
            32,
            dequantize_bnb4_to_bf16,
            encode_bnb4,
        )
        .unwrap();
    }

    #[test]
    fn encode_bnb4_round_trips_nf4_codebook() {
        assert_bnb4_decode_encode_round_trip(
            &NF4_CODEBOOK,
            &[1.0, 0.5, 2.0, 0.0123],
            32,
            dequantize_bnb4_to_bf16,
            encode_bnb4,
        )
        .unwrap();
    }

    #[test]
    fn encode_bnb4_round_trips_fp4_codebook() {
        // FP4 has -0 and +0 in distinct slots; exact-bit-match priority
        // must preserve the sign across decode→encode.
        assert_bnb4_decode_encode_round_trip(
            &FP4_CODEBOOK,
            &[1.0, 0.5, 2.0, 0.0123],
            32,
            dequantize_bnb4_to_bf16,
            encode_bnb4,
        )
        .unwrap();
    }

    #[test]
    fn encode_bnb4_round_trips_collapsed_fp4_codebook() {
        // The on-disk bitsandbytes Python FP4 quant_map collapses -0.0
        // to +0.0 (codebook[0].to_bits() == codebook[8].to_bits() == 0).
        // Round-trip byte-exactness on this codebook depends on the
        // joint operation of:
        //   - decode-side apply_sign_magnitude_zero (emits -0 BF16 for
        //     nibble 8 / codebook +0), AND
        //   - encode-side apply_sign_magnitude_encode_correction
        //     (lifts -0 BF16 input to nibble 8 instead of nibble 0).
        let mut collapsed = FP4_CODEBOOK;
        collapsed[8] = 0.0; // collapse the -0 entry to +0 (matches bitsandbytes' on-disk layout)
        assert_eq!(
            collapsed[0].to_bits(),
            collapsed[8].to_bits(),
            "test pre-condition: indices 0 and 8 must share bits",
        );
        assert_bnb4_decode_encode_round_trip(
            &collapsed,
            &[1.0, 0.5, 2.0, 0.0123],
            32,
            dequantize_bnb4_to_bf16,
            encode_bnb4,
        )
        .unwrap();
    }

    #[test]
    fn apply_sign_magnitude_encode_correction_lifts_to_upper_when_duplicated() {
        // Mirrors FP4 fixture: codebook[0] == codebook[8] == +0.0.
        let mut cb = FP4_CODEBOOK;
        cb[8] = 0.0;
        // -0 input with chosen lower-half nibble 0 → lifts to 8.
        assert_eq!(apply_sign_magnitude_encode_correction(-0.0, 0, &cb), 8);
        // +0 input with chosen nibble 0 → stays 0 (positive sign).
        assert_eq!(apply_sign_magnitude_encode_correction(0.0, 0, &cb), 0);
        // -tiny non-zero with chosen nibble 0 → still lifts (correction
        // is sign-bit-based, not magnitude-based; this matches
        // bitsandbytes' x < 0 ? 8 : 0 sign-magnitude convention).
        assert_eq!(apply_sign_magnitude_encode_correction(-1e-30, 0, &cb), 8);
        // -value rounding to upper-half nibble → no change.
        assert_eq!(apply_sign_magnitude_encode_correction(-1.0, 11, &cb), 11);
    }

    #[test]
    fn apply_sign_magnitude_encode_correction_noop_when_bits_differ() {
        // NF4-style: codebook[0] and codebook[8] have different bits.
        assert_eq!(
            apply_sign_magnitude_encode_correction(-0.0, 7, &NF4_CODEBOOK),
            7,
            "NF4 codebook has codebook[15]=1.0 ≠ codebook[7]=0.0; \
             correction must not fire",
        );
    }

    #[test]
    fn encode_bnb4_uniform_codebook_zero_byte() {
        // All-zero bf16 input + uniform codebook should produce all-zero bytes.
        let cb: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let cb_bytes = f32_to_bytes(&cb);
        let absmax_bytes = f32_to_bytes(&[1.0]);
        let bf16 = bf16_bytes_from_f32(&[0.0; 4]);
        let out = encode_bnb4(&bf16, &absmax_bytes, &cb_bytes, 4, 4).unwrap();
        // codebook[0] = 0.0 → nibble 0 (both halves) → byte 0x00
        assert_eq!(out, vec![0x00, 0x00]);
    }

    #[test]
    fn encode_bnb4_nibble_extraction_inverse() {
        // codebook = [0, 1, 2, …, 15]; bf16 input [1, 3, 2, 4] with absmax=1.0
        // → nibbles [1, 3, 2, 4] → bytes [0x13, 0x24] (bitsandbytes order:
        // first element packs into the HIGH nibble).
        let cb: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let cb_bytes = f32_to_bytes(&cb);
        let absmax_bytes = f32_to_bytes(&[1.0]);
        let bf16 = bf16_bytes_from_f32(&[1.0, 3.0, 2.0, 4.0]);
        let out = encode_bnb4(&bf16, &absmax_bytes, &cb_bytes, 4, 4).unwrap();
        assert_eq!(out, vec![0x13, 0x24]);
    }

    #[test]
    fn encode_bnb4_validation_errors() {
        let cb_bytes = f32_to_bytes(&[0.0; 16]);
        let absmax_bytes = f32_to_bytes(&[1.0]);
        // block_size = 0
        assert!(encode_bnb4(&[0; 4], &absmax_bytes, &cb_bytes, 2, 0).is_err());
        // bf16 byte mismatch
        assert!(encode_bnb4(&[0; 2], &absmax_bytes, &cb_bytes, 4, 4).is_err());
        // wrong codebook size
        assert!(encode_bnb4(&[0; 8], &absmax_bytes, &[0; 32], 4, 4).is_err());
        // odd total_elements
        assert!(encode_bnb4(&[0; 6], &absmax_bytes, &cb_bytes, 3, 3).is_err());
        // odd block_size — rejected with an even-ness message
        let err = encode_bnb4(&[0; 4], &absmax_bytes, &cb_bytes, 8, 3).unwrap_err();
        assert!(
            matches!(err, AnamnesisError::Parse { ref reason } if reason.contains("even")),
            "expected even-block_size rejection, got: {err}"
        );
    }

    // --- encode_bnb4_double_quant ---

    #[test]
    fn encode_bnb4_double_quant_round_trips_synthetic() {
        // Construct a minimal double-quant scenario:
        // - 4 elements, block_size = 2 → 2 blocks
        // - nested_block_size = 256 → 1 nested block covering both
        // - absmax_data: two bytes (one per block); pick indices that hit
        //   distinct nested_codebook entries so each block's recovered
        //   absmax differs.
        // - nested_codebook[64] = 0.5, nested_codebook[200] = 1.5
        //   (other entries are 0.0 → un-exercised, valid).
        // - nested_absmax: single F32 = 2.0 (one nested block).
        // - nested_offset = 0.5 (exercises the bitsandbytes absmax-mean bias).
        // - Recovered absmax = [0.5 * 2.0 + 0.5, 1.5 * 2.0 + 0.5] = [1.5, 3.5].
        // - Codebook: NF4 (distinct entries, no sign-of-zero ambiguity).
        let mut nested_cb = [0.0_f32; 256];
        nested_cb[64] = 0.5;
        nested_cb[200] = 1.5;
        let nested_cb_bytes: Vec<u8> = nested_cb.iter().flat_map(|v| v.to_le_bytes()).collect();
        let nested_absmax_bytes = f32_to_bytes(&[2.0]);
        let nested_offset = 0.5_f32;
        let absmax_data = vec![64u8, 200u8];
        let codebook_bytes = f32_to_bytes(&NF4_CODEBOOK);

        // Build a weight_data with diverse nibbles, decode, then encode → assert byte equality.
        // 4 elements packed in 2 bytes: bytes 0x53, 0x9C → nibble pairs
        // (high=5, low=3) and (high=9, low=12) in bitsandbytes order.
        let weight_data = vec![0x53u8, 0x9Cu8];
        let bf16 = crate::remember::bnb::dequantize_bnb4_double_quant_to_bf16(
            &weight_data,
            &absmax_data,
            &codebook_bytes,
            &nested_absmax_bytes,
            &nested_cb_bytes,
            nested_offset,
            4,
            2,
            256,
        )
        .unwrap();
        let re_encoded = encode_bnb4_double_quant(
            &bf16,
            &absmax_data,
            &codebook_bytes,
            &nested_absmax_bytes,
            &nested_cb_bytes,
            nested_offset,
            4,
            2,
            256,
        )
        .unwrap();
        assert_eq!(re_encoded, weight_data);
    }

    #[test]
    fn encode_bnb4_double_quant_validation_errors() {
        let codebook_bytes = f32_to_bytes(&NF4_CODEBOOK);
        let nested_cb_bytes = f32_to_bytes(&[0.0_f32; 256]);
        let nested_absmax_bytes = f32_to_bytes(&[1.0]);
        let absmax = vec![0u8];

        // block_size = 0
        assert!(
            encode_bnb4_double_quant(
                &[0; 4],
                &absmax,
                &codebook_bytes,
                &nested_absmax_bytes,
                &nested_cb_bytes,
                0.0,
                2,
                0,
                256,
            )
            .is_err()
        );

        // total_elements odd
        assert!(
            encode_bnb4_double_quant(
                &[0; 6],
                &absmax,
                &codebook_bytes,
                &nested_absmax_bytes,
                &nested_cb_bytes,
                0.0,
                3,
                3,
                256,
            )
            .is_err()
        );

        // wrong nested_quant_map size
        assert!(
            encode_bnb4_double_quant(
                &[0; 4],
                &absmax,
                &codebook_bytes,
                &nested_absmax_bytes,
                &[0; 512],
                0.0,
                2,
                2,
                256,
            )
            .is_err()
        );

        // absmax byte count mismatch (1 block expected, 2 given)
        assert!(
            encode_bnb4_double_quant(
                &[0; 4],
                &[0u8, 0u8],
                &codebook_bytes,
                &nested_absmax_bytes,
                &nested_cb_bytes,
                0.0,
                2,
                2,
                256,
            )
            .is_err()
        );

        // nested_block_size = 0
        assert!(
            encode_bnb4_double_quant(
                &[0; 4],
                &absmax,
                &codebook_bytes,
                &nested_absmax_bytes,
                &nested_cb_bytes,
                0.0,
                2,
                2,
                0,
            )
            .is_err()
        );
    }

    #[test]
    fn encode_bnb4_compute_absmax_round_trips_self() {
        // Build a bf16 source, compute absmax, encode; decode with the
        // returned absmax + same codebook; encode again; bytes must match.
        let cb_bytes = f32_to_bytes(&NF4_CODEBOOK);
        // 64 elements (2 blocks of 32): values spanning [-1.0, 1.0]
        let values: Vec<f32> = (0..64).map(|i| (i as f32 - 31.5) / 31.5).collect();
        let bf16 = bf16_bytes_from_f32(&values);
        let (weight, absmax) = encode_bnb4_compute_absmax(&bf16, &cb_bytes, 64, 32).unwrap();
        // Re-decode then re-encode using the returned absmax.
        let decoded = dequantize_bnb4_to_bf16(&weight, &absmax, &cb_bytes, 64, 32).unwrap();
        let re_encoded = encode_bnb4(&decoded, &absmax, &cb_bytes, 64, 32).unwrap();
        assert_eq!(weight, re_encoded);
    }

    // --- encode_bnb_int8 round-trip ---

    #[test]
    fn encode_bnb_int8_round_trips_every_i8() {
        assert_bnb_int8_decode_encode_round_trip(dequantize_bnb_int8_to_bf16, encode_bnb_int8)
            .unwrap();
    }

    #[test]
    fn encode_bnb_int8_basic_inverse() {
        // 2×2 matrix, SCB = [127.0, 254.0]
        // bf16 input matching the decode test's outputs: [[1.0, -1.0], [4.0, -4.0]]
        // → i8 = [[1, -1], [2, -2]] (scaled back by SCB/127).
        let scb_bytes = f32_to_bytes(&[127.0, 254.0]);
        let bf16 = bf16_bytes_from_f32(&[1.0, -1.0, 4.0, -4.0]);
        let out = encode_bnb_int8(&bf16, &scb_bytes, 2, 2).unwrap();
        assert_eq!(out, vec![1u8, 0xFF, 2u8, 0xFE]);
    }

    #[test]
    fn encode_bnb_int8_clamps_positive_overflow() {
        // BF16 value at exact SCB → scaled = 127.0 exactly; rounds + clamps to +127.
        let scb_bytes = f32_to_bytes(&[1.0]);
        let bf16 = bf16_bytes_from_f32(&[1.0, 0.5]);
        let out = encode_bnb_int8(&bf16, &scb_bytes, 1, 2).unwrap();
        // 1.0 / (1.0/127.0) = 127.0 → +127 (0x7F); 0.5 / scale = 63.5 → 64 (0x40)
        assert_eq!(out[0], 0x7F);
        assert_eq!(out[1], 0x40);
    }

    #[test]
    fn encode_bnb_int8_clamps_negative_overflow() {
        // BF16 value below -SCB → scaled < -128; clamps to -128 (0x80).
        let scb_bytes = f32_to_bytes(&[1.0]);
        // -2.0 / (1.0/127.0) = -254.0 → clamp to -128 (0x80).
        let bf16 = bf16_bytes_from_f32(&[-2.0]);
        let out = encode_bnb_int8(&bf16, &scb_bytes, 1, 1).unwrap();
        assert_eq!(out[0], 0x80);
    }

    #[test]
    fn encode_bnb_int8_zero_scb() {
        // SCB = 0 → every element encodes to 0 (mirrors decode collapsing
        // a zero-SCB row to zero).
        let scb_bytes = f32_to_bytes(&[0.0]);
        let bf16 = bf16_bytes_from_f32(&[1.0, -1.0, 100.0]);
        let out = encode_bnb_int8(&bf16, &scb_bytes, 1, 3).unwrap();
        assert_eq!(out, vec![0, 0, 0]);
    }

    #[test]
    fn encode_bnb_int8_validation_errors() {
        let scb_bytes = f32_to_bytes(&[1.0]);
        // bf16 length mismatch
        assert!(encode_bnb_int8(&[0; 2], &scb_bytes, 1, 2).is_err());
        // scb length mismatch
        assert!(encode_bnb_int8(&[0; 4], &scb_bytes, 2, 1).is_err());
    }

    #[test]
    fn encode_bnb_int8_compute_scb_round_trips_self() {
        // Build a bf16 source, derive SCB, encode → decode → re-encode;
        // bytes must match.
        let values: Vec<f32> = (-16..16).map(|i| i as f32 * 0.5).collect(); // 32 values
        let bf16 = bf16_bytes_from_f32(&values);
        let (weight, scb) = encode_bnb_int8_compute_scb(&bf16, 2, 16).unwrap();
        let decoded = dequantize_bnb_int8_to_bf16(&weight, &scb, 2, 16).unwrap();
        let re_encoded = encode_bnb_int8(&decoded, &scb, 2, 16).unwrap();
        assert_eq!(weight, re_encoded);
    }
}
