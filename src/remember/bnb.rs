// SPDX-License-Identifier: MIT OR Apache-2.0

//! `BitsAndBytes` dequantization (`NF4`/`FP4` 4-bit and `INT8`).
//!
//! `NF4`/`FP4` uses a 16-entry lookup table + per-block absmax scaling.
//! `INT8` (`LLM.int8()`) uses per-row absmax with linear `I8` quantization.
//!
//! # Output width
//!
//! Since v0.7.4 [`dequantize_bnb4`], `dequantize_bnb4_double_quant` and
//! [`dequantize_bnb_int8`] are generic over
//! [`OutputElement`]; the `*_to_bf16` names remain as
//! `Bf16Out` wrappers. Both kernels end in
//! [`OutputElement::write_scratch`]: the `NF4`/`FP4` path already had an `f32`
//! block scratch and simply stopped narrowing inside pass 2, and the `INT8`
//! path gained a per-row scratch so it has the same shape.
//!
//! # References
//!
//! - Dettmers et al., "`LLM.int8()`: 8-bit Matrix Multiplication for
//!   Transformers at Scale", `NeurIPS` 2022 (`arXiv:2208.07339`)
//! - Dettmers et al., "`QLoRA`: Efficient Finetuning of Quantized Large
//!   Language Models", `NeurIPS` 2023 (`arXiv:2305.14314`)

// MEASURED-REVERT: clippy::chunks_exact_to_as_chunks (new in Rust 1.98).
// Not taken here **despite a favourable reading**, which is the honest reason
// rather than a convenient one. Migrating this loop measured -7.6 % on
// `dequant_bnb_int8` (i.e. faster), but that sits inside this host's own noise
// floor: the same bench run reported +10.6 % at p = 0.00 on `dequant_bnb_nf4`
// with byte-identical source. A reading smaller than the instrument's error is
// not evidence. Meanwhile the structurally identical `GPTQ` and `AWQ` loops --
// same four-way zip, same `VECTOR_TILE * E::BYTES` output that cannot migrate --
// cost +51 % and +74 % respectively. Reverted with its twins pending a
// measurement on hardware that can resolve the difference.
// See CONVENTIONS.md § MEASURED-REVERT Annotation, and § Benchmark evidence for why a
// criterion baseline alone cannot settle this.
#![allow(clippy::chunks_exact_to_as_chunks)]

use crate::error::AnamnesisError;
use crate::remember::output::{Bf16Out, OutputElement, VECTOR_TILE};

/// Reciprocal of the `INT8` scale denominator, applied as `bitsandbytes` does.
///
/// `bitsandbytes`' `int8_vectorwise_dequant` is
/// `A * stats.view(-1, 1) * 7.874015718698502e-3`. That decimal literal and
/// `1.0 / 127.0` round to the **same** `f32` (`0x3C01_0204`), asserted below,
/// so writing the exact quotient here costs no fidelity and reads honestly.
/// What actually mattered was the multiply *order*, not the constant — see
/// [`dequantize_bnb_int8`].
const INV_127: f32 = 1.0 / 127.0;

// The claim above is checked at compile time rather than trusted: if a future
// bitsandbytes changed the literal, the assertion is where that surfaces.
//
// `excessive_precision` is allowed on purpose: the literal is quoted verbatim
// from `bitsandbytes/_ops.py`, and its excess digits over `f32` are precisely
// what the assertion exists to discharge. Trimming it to what `f32` can hold
// would make the check tautological.
#[allow(clippy::excessive_precision)]
const _: () = {
    assert!(INV_127.to_bits() == 0x3C01_0204);
    assert!((7.874_015_718_698_502_e-3_f32).to_bits() == INV_127.to_bits());
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Reads a little-endian `f32` from a byte slice at the given offset.
///
/// # Errors
///
/// Returns `None` if the slice does not contain 4 bytes at `offset`.
fn read_f32_le(data: &[u8], offset: usize) -> Option<f32> {
    let bytes: &[u8] = data.get(offset..offset + 4)?;
    let arr: [u8; 4] = bytes.try_into().ok()?;
    Some(f32::from_le_bytes(arr))
}

/// Applies the sign-magnitude zero-preservation rule to one looked-up
/// codebook entry.
///
/// When the entry is exactly `+0.0` (IEEE 754 bits `0x00000000`) AND
/// the nibble has its high bit set (`nibble & 0x8 != 0`), returns
/// `-0.0`. Otherwise returns the entry unchanged.
///
/// The rule implements `BnB`-style sign-magnitude convention: in
/// `bitsandbytes` `FP4`, the high nibble bit encodes the sign of the
/// quantised value, but the on-disk `quant_map` stores `+0.0` at both
/// index 0 and index 8 (a lossy compression of the codebook). Without
/// this tweak, our decoded `BF16` would emit `0x0000` for both nibbles,
/// destroying the sign information. With it, nibble 8 decodes to
/// `0x8000` (negative zero), so a subsequent encode can recover the
/// original nibble byte-exactly.
///
/// No-op for any codebook whose upper-half (indices 8..16) has no
/// `+0.0` entry — `NF4` (index-8 entry is `0.0795…`), every `GGUF`
/// codebook, etc.
#[inline]
#[must_use]
fn apply_sign_magnitude_zero(entry: f32, nibble: usize) -> f32 {
    // BITWISE: detect IEEE 754 +0.0 via exact bit equality (treats only
    // +0, not -0, as the trigger — -0 is already what we'd emit).
    if entry.to_bits() == 0 && (nibble & 0x8) != 0 {
        -0.0_f32
    } else {
        entry
    }
}

// ---------------------------------------------------------------------------
// NF4/FP4 dequantization (4-bit, lookup-table based)
// ---------------------------------------------------------------------------

/// Core `NF4`/`FP4` dequant: accepts pre-decoded `f32` absmax values directly.
///
/// Shared by both the plain and double-quant public entry points.
/// Callers are responsible for validation; this function assumes inputs
/// are dimensionally consistent.
fn dequantize_bnb4_core<E: OutputElement>(
    weight_data: &[u8],
    absmax: &[f32],
    quant_map: &[f32; 16],
    total_elements: usize,
    block_size: usize,
) -> crate::Result<Vec<u8>> {
    // --- Allocate output ---
    let out_byte_len =
        total_elements
            .checked_mul(E::BYTES)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: "BnB4 output byte count overflow".into(),
            })?;
    let mut output = vec![0u8; out_byte_len];

    // --- Per-block dequantization with loop fission ---
    let bytes_per_block = block_size / 2;
    // Scratch buffer for unpacked f32 values (one block at a time, fits in L1)
    let mut scratch = vec![0.0f32; block_size];

    for (block_idx, &block_absmax) in absmax.iter().enumerate() {
        // Pre-slice validated ranges (two-level bounds checking per CONVENTIONS.md)
        let w_start = block_idx * bytes_per_block;
        let w_end = w_start + bytes_per_block;
        let weight_block =
            weight_data
                .get(w_start..w_end)
                .ok_or_else(|| AnamnesisError::Parse {
                    reason: format!("BnB4 weight block {block_idx} out of bounds"),
                })?;
        let o_start = block_idx * block_size * E::BYTES;
        let o_end = o_start + block_size * E::BYTES;
        let out_block = output
            .get_mut(o_start..o_end)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!("BnB4 output block {block_idx} out of bounds"),
            })?;

        // --- Pass 1 (unpack): byte → 2 nibbles → table lookup → f32 scratch ---
        // Each byte produces two f32 values via the quant_map lookup.
        // Nibble order is HIGH-first (`byte >> 4` → element 2i, `byte & 0x0F`
        // → element 2i+1), matching the bitsandbytes kernel. The opposite
        // (low-first) order shipped in ≤ v0.6.3 and produced element-permuted
        // output; see docs/dogfooding-feedbacks/
        // bnb-nibble-order-and-circular-fixture-validation.md.
        //
        // Sign-of-zero preservation (FP4-style sign-magnitude codebooks):
        // when the looked-up codebook entry is exactly +0.0 AND the
        // nibble has its high bit set (n & 0x8 != 0), we substitute
        // -0.0. This recovers the sign information that bitsandbytes'
        // Python on-disk `FP4` `quant_map` discards (its index-8 entry
        // is stored as +0.0 instead of -0.0). The arithmetic value is
        // unchanged — both are IEEE 754 zero — but the BF16 sign bit
        // is preserved across a decode→encode round trip, so the
        // round-trip is byte-exact rather than only decode-equivalent.
        // This is a no-op for `NF4` (codebook[8] = 0.0795…, never +0)
        // and for any codebook whose upper half lacks a +0.0 entry.
        // VECTORIZED: scalar fallback — Pass-1 of fission; the codebook lookup
        // `quant_map[nibble]` is data-dependent indexing, which the compiler
        // cannot turn into packed loads. Confirmed in `--emit=asm`, x86-64
        // target-cpu=native, opt-level=3: this loop emits scalar `vmovss` table
        // loads assembled lane-by-lane with `vinsertps` (12 of them), and no
        // `vgatherdps` at all — consistent with Experiment 10's finding that the
        // gather-bound kernels are the ones SIMD cannot help. Pass 2 below is
        // the pass that vectorises, and carries its own annotation.
        //
        // (Before v0.7.4 this comment also described pass 2 as doing the `BF16`
        // convert. That stopped being true when the narrowing moved into pass 3
        // and became width-generic; it is stated once, below, where it happens.)
        // INDEX: scratch.len() == block_size, guaranteed by vec![0.0f32; block_size]
        #[allow(clippy::indexing_slicing)]
        let scratch_block = &mut scratch[..block_size];
        for (&byte, pair) in weight_block.iter().zip(scratch_block.chunks_exact_mut(2)) {
            // BITWISE: extract high nibble (bits [7:4]) and low nibble (bits [3:0])
            // CAST: u8 → usize, nibble values 0-15 used as lookup indices
            #[allow(clippy::as_conversions)]
            let high = (byte >> 4) as usize;
            #[allow(clippy::as_conversions)]
            let low = (byte & 0x0F) as usize;
            // INDEX: high and low are 0-15, quant_map has 16 entries
            #[allow(clippy::indexing_slicing)]
            {
                pair[0] = apply_sign_magnitude_zero(quant_map[high], high);
                pair[1] = apply_sign_magnitude_zero(quant_map[low], low);
            }
        }

        // --- Pass 2 (scale): f32 scratch × absmax, in place ---
        // Pure float multiply, no narrowing: the output width is applied once
        // per block by pass 3. Scaling in place rather than into a second
        // buffer keeps the block's working set at one L1-resident array; the
        // read and the write are the same element, so there is no aliasing
        // ambiguity for the vectorizer to give up on.
        // VECTORIZED: confirmed AVX2 vmulps on %ymm (8-wide) in `--emit=asm`,
        // x86-64 target-cpu=native, opt-level=3, for all three `E`; the
        // narrowing that follows is `write_scratch`'s own confirmed loop
        // (`vpaddd`/`vpsrld` at `Bf16Out`, `vmovups` at `F32Out`, `vcvtps2ph`
        // at `F16Out`). Pass 1 above stays a table lookup and is annotated
        // separately.
        //
        // Measured 25.88 -> 26.96 ms on a 45 M-element `NF4` fixture against
        // the v0.7.3 binary (best-of-5 min of 4 interleaved rounds, release,
        // target-cpu=native), i.e. 1.04x slower — the smallest cost of the four
        // families, because this kernel already had an `f32` block scratch and
        // only the narrowing moved out of the loop.
        for val in scratch_block.iter_mut() {
            *val *= block_absmax;
        }

        // --- Pass 3: narrow to the caller's output width ---
        E::write_scratch(scratch_block, out_block);
    }

    Ok(output)
}

/// Dequantizes `BitsAndBytes` `NF4`/`FP4` quantized weights to `BF16`.
///
/// The [`Bf16Out`] special case of [`dequantize_bnb4`], kept so that every
/// pre-v0.7.4 caller compiles unchanged.
///
/// # Errors
///
/// See [`dequantize_bnb4`].
///
/// # Memory
///
/// See [`dequantize_bnb4`]; at `BF16` the output buffer is
/// `total_elements × 2` bytes.
#[inline]
pub fn dequantize_bnb4_to_bf16(
    weight_data: &[u8],
    absmax_data: &[u8],
    quant_map_data: &[u8],
    total_elements: usize,
    block_size: usize,
) -> crate::Result<Vec<u8>> {
    dequantize_bnb4::<Bf16Out>(
        weight_data,
        absmax_data,
        quant_map_data,
        total_elements,
        block_size,
    )
}

/// Dequantizes `BitsAndBytes` `NF4`/`FP4` quantized weights into `E`.
///
/// Each byte in `weight_data` packs two 4-bit values: high nibble first
/// (`byte >> 4` → element `2i`), low nibble second (`byte & 0x0F` →
/// element `2i + 1`) — the `bitsandbytes` kernel convention. Each nibble
/// indexes into `quant_map_data` (a 16-entry `F32` lookup table). The
/// looked-up value is then scaled by the block's absmax.
///
/// # Sign-of-zero preservation
///
/// When a looked-up codebook entry is exactly `+0.0` (bits `0x00000000`)
/// AND the nibble has its high bit set (`nibble & 0x8 != 0`), the
/// emitted `BF16` is `-0.0` (bits `0x8000`) rather than `+0.0`. This
/// recovers the sign information that `bitsandbytes`' Python on-disk
/// `FP4` `quant_map` discards (its index-8 entry is stored as `+0.0`
/// instead of `-0.0`). The arithmetic value is unchanged — both `+0`
/// and `-0` are IEEE 754 zero — but the sign bit propagates through
/// the encode round trip so [`encode_bnb4`](crate::encode_bnb4) can
/// recover the original nibble byte-exactly.
///
/// This is a **deliberate divergence** from `bitsandbytes`' Python
/// decode (which always emits `+0` for nibble 8 under that codebook).
/// The divergence is arithmetically invisible (zero is zero); the only
/// observable difference is the sign bit on a small fraction of
/// elements (`8 / 4096 = 0.2 %` on the existing `FP4` fixture). It is
/// a no-op for `NF4` and any other codebook whose upper-half indices
/// hold non-zero entries.
///
/// # Arguments
///
/// - `weight_data` — `U8` bytes, two `NF4`/`FP4` values per byte.
/// - `absmax_data` — `F32` per-block absmax values (little-endian bytes).
/// - `quant_map_data` — `F32[16]` lookup table.
/// - `total_elements` — total number of dequantized elements (= weight bytes × 2).
/// - `block_size` — elements per absmax block (typically 64).
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if tensor dimensions are inconsistent.
///
/// # Memory
///
/// Allocates `total_elements × E::BYTES` bytes of output, plus a scratch
/// buffer of `block_size × 4` bytes for loop fission (fits in L1 cache).
pub fn dequantize_bnb4<E: OutputElement>(
    weight_data: &[u8],
    absmax_data: &[u8],
    quant_map_data: &[u8],
    total_elements: usize,
    block_size: usize,
) -> crate::Result<Vec<u8>> {
    // --- Validation ---
    if block_size == 0 {
        return Err(AnamnesisError::Parse {
            reason: "BnB block_size must be > 0".into(),
        });
    }
    // An odd block_size silently truncates `bytes_per_block = block_size / 2`,
    // mis-aligning every block after the first and producing wrong (not
    // out-of-bounds) output. Two nibbles pack into one byte, so a real BnB4
    // block_size is always even (64 in practice).
    if !block_size.is_multiple_of(2) {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "BnB4 block_size must be even (two nibbles per byte), got {block_size}"
            ),
        });
    }
    let expected_weight_bytes = if total_elements.is_multiple_of(2) {
        Some(total_elements / 2)
    } else {
        None
    };
    if expected_weight_bytes != Some(weight_data.len()) {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "BnB4 weight byte count mismatch: expected {} for {} elements, got {}",
                expected_weight_bytes.unwrap_or(0),
                total_elements,
                weight_data.len()
            ),
        });
    }
    if !total_elements.is_multiple_of(block_size) {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "BnB4 total_elements ({total_elements}) not divisible by block_size ({block_size})"
            ),
        });
    }
    let num_blocks = total_elements / block_size;
    let expected_absmax_bytes = num_blocks
        .checked_mul(4)
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "absmax byte count overflow".into(),
        })?;
    if absmax_data.len() != expected_absmax_bytes {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "BnB4 absmax byte count mismatch: expected {expected_absmax_bytes}, got {}",
                absmax_data.len()
            ),
        });
    }
    // quant_map must be exactly 16 F32 values = 64 bytes
    if quant_map_data.len() != 64 {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "BnB4 quant_map must be 64 bytes (16×F32), got {}",
                quant_map_data.len()
            ),
        });
    }

    // --- Pre-load quant_map (16 entries) ---
    let mut quant_map = [0.0f32; 16];
    for (i, val) in quant_map.iter_mut().enumerate() {
        *val = read_f32_le(quant_map_data, i * 4).ok_or_else(|| AnamnesisError::Parse {
            reason: "BnB4 quant_map read out of bounds".into(),
        })?;
    }

    // --- Decode absmax bytes → f32 slice ---
    let mut absmax_f32 = vec![0.0f32; num_blocks];
    for (i, val) in absmax_f32.iter_mut().enumerate() {
        *val = read_f32_le(absmax_data, i * 4).ok_or_else(|| AnamnesisError::Parse {
            reason: format!("BnB4 absmax read out of bounds at block {i}"),
        })?;
    }

    dequantize_bnb4_core::<E>(
        weight_data,
        &absmax_f32,
        &quant_map,
        total_elements,
        block_size,
    )
}

/// Dequantizes `BitsAndBytes` `NF4`/`FP4` with double quantization to `BF16`.
///
/// The [`Bf16Out`] special case of [`dequantize_bnb4_double_quant`], kept so
/// that every pre-v0.7.4 caller compiles unchanged.
///
/// # Errors
///
/// See [`dequantize_bnb4_double_quant`].
///
/// # Memory
///
/// See [`dequantize_bnb4_double_quant`]; at `BF16` the output buffer is
/// `total_elements × 2` bytes.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn dequantize_bnb4_double_quant_to_bf16(
    weight_data: &[u8],
    absmax_data: &[u8],
    quant_map_data: &[u8],
    nested_absmax_data: &[u8],
    nested_quant_map_data: &[u8],
    nested_offset: f32,
    total_elements: usize,
    block_size: usize,
    nested_block_size: usize,
) -> crate::Result<Vec<u8>> {
    dequantize_bnb4_double_quant::<Bf16Out>(
        weight_data,
        absmax_data,
        quant_map_data,
        nested_absmax_data,
        nested_quant_map_data,
        nested_offset,
        total_elements,
        block_size,
        nested_block_size,
    )
}

/// Dequantizes `BitsAndBytes` `NF4`/`FP4` with double quantization into `E`.
///
/// First dequantizes the nested absmax values (themselves quantized to `U8`),
/// then uses the recovered `F32` absmax values for the main `NF4`/`FP4` dequant.
///
/// The recovery formula is the `bitsandbytes` one:
///
/// ```text
/// absmax[i] = nested_quant_map[absmax_u8[i]] × nested_absmax[i / nested_block_size]
///             + nested_offset
/// ```
///
/// The additive `nested_offset` (the mean of the original absmax values,
/// subtracted by `bitsandbytes` before nested quantization to centre the
/// distribution) is stored in the `quant_state` `JSON` blob as
/// `"nested_offset"`. Versions ≤ v0.6.3 omitted it, biasing every recovered
/// absmax low by the offset; see
/// `docs/dogfooding-feedbacks/bnb-nibble-order-and-circular-fixture-validation.md`.
///
/// # Arguments
///
/// - `weight_data` — `U8` bytes, two 4-bit values per byte.
/// - `absmax_data` — `U8` quantized absmax values (one per block).
/// - `quant_map_data` — `F32[16]` main lookup table.
/// - `nested_absmax_data` — `F32` absmax for the nested quantization.
/// - `nested_quant_map_data` — `F32[256]` lookup table for the nested quantization.
/// - `nested_offset` — additive absmax offset from the `quant_state` blob
///   (`bitsandbytes` `QuantState.offset`); `0.0` only for synthetic inputs
///   that were never offset-compressed.
/// - `total_elements` — total number of dequantized elements.
/// - `block_size` — elements per absmax block (typically 64).
/// - `nested_block_size` — elements per nested absmax block (typically 256).
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if tensor dimensions are inconsistent.
///
/// # Memory
///
/// Allocates `total_elements × E::BYTES` bytes of output, plus an `f32`
/// absmax array (`num_blocks × 4` bytes) and a scratch buffer
/// (`block_size × 4` bytes). No intermediate byte serialization.
#[allow(clippy::too_many_arguments)]
pub fn dequantize_bnb4_double_quant<E: OutputElement>(
    weight_data: &[u8],
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
    if block_size == 0 || nested_block_size == 0 {
        return Err(AnamnesisError::Parse {
            reason: "BnB block_size and nested_block_size must be > 0".into(),
        });
    }
    // Odd block_size truncates `bytes_per_block = block_size / 2` → mis-aligned
    // blocks → wrong output (see the plain-decode guard above).
    if !block_size.is_multiple_of(2) {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "BnB4 block_size must be even (two nibbles per byte), got {block_size}"
            ),
        });
    }
    if !total_elements.is_multiple_of(block_size) {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "BnB4 total_elements ({total_elements}) not divisible by block_size ({block_size})"
            ),
        });
    }
    let num_blocks = total_elements / block_size;
    if absmax_data.len() != num_blocks {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "BnB4 double-quant: absmax byte count mismatch: expected {num_blocks}, got {}",
                absmax_data.len()
            ),
        });
    }
    // nested_quant_map must be exactly 256 F32 values = 1024 bytes
    if nested_quant_map_data.len() != 1024 {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "BnB4 nested_quant_map must be 1024 bytes (256×F32), got {}",
                nested_quant_map_data.len()
            ),
        });
    }

    // --- Pre-load nested quant_map (256 entries) ---
    let mut nested_quant_map = [0.0f32; 256];
    for (i, val) in nested_quant_map.iter_mut().enumerate() {
        *val = read_f32_le(nested_quant_map_data, i * 4).ok_or_else(|| AnamnesisError::Parse {
            reason: "BnB4 nested_quant_map read out of bounds".into(),
        })?;
    }

    // --- Dequantize nested absmax: U8 → F32 via nested lookup × nested_absmax ---
    let num_nested_blocks = if num_blocks.is_multiple_of(nested_block_size) {
        num_blocks / nested_block_size
    } else {
        // Partial last block: round up
        num_blocks / nested_block_size + 1
    };
    let expected_nested_absmax_bytes =
        num_nested_blocks
            .checked_mul(4)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: "nested absmax byte count overflow".into(),
            })?;
    if nested_absmax_data.len() != expected_nested_absmax_bytes {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "BnB4 nested_absmax byte count mismatch: expected {expected_nested_absmax_bytes}, got {}",
                nested_absmax_data.len()
            ),
        });
    }

    let mut dequantized_absmax = vec![0.0f32; num_blocks];
    for (i, &absmax_byte) in absmax_data.iter().enumerate() {
        let nested_block_idx = i / nested_block_size;
        let nested_absmax_val =
            read_f32_le(nested_absmax_data, nested_block_idx * 4).ok_or_else(|| {
                AnamnesisError::Parse {
                    reason: format!(
                        "BnB4 nested_absmax read out of bounds at block {nested_block_idx}"
                    ),
                }
            })?;
        // CAST: u8 → usize, absmax_byte is 0-255 used as lookup index
        #[allow(clippy::as_conversions)]
        let idx = absmax_byte as usize;
        // INDEX: idx is 0-255, nested_quant_map has 256 entries;
        //        i < num_blocks, dequantized_absmax has num_blocks entries
        #[allow(clippy::indexing_slicing)]
        {
            dequantized_absmax[i] = nested_quant_map[idx] * nested_absmax_val + nested_offset;
        }
    }

    // --- Pre-load quant_map (16 entries) ---
    if quant_map_data.len() != 64 {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "BnB4 quant_map must be 64 bytes (16×F32), got {}",
                quant_map_data.len()
            ),
        });
    }
    let mut quant_map = [0.0f32; 16];
    for (i, val) in quant_map.iter_mut().enumerate() {
        *val = read_f32_le(quant_map_data, i * 4).ok_or_else(|| AnamnesisError::Parse {
            reason: "BnB4 quant_map read out of bounds".into(),
        })?;
    }

    // --- Delegate to core dequant with recovered f32 absmax directly ---
    // No intermediate serialization: dequantized_absmax is passed as &[f32].
    dequantize_bnb4_core::<E>(
        weight_data,
        &dequantized_absmax,
        &quant_map,
        total_elements,
        block_size,
    )
}

// ---------------------------------------------------------------------------
// INT8 dequantization (LLM.int8(), per-row absmax)
// ---------------------------------------------------------------------------

/// Dequantizes `BitsAndBytes` `INT8` (`LLM.int8()`) quantized weights to `BF16`.
///
/// The [`Bf16Out`] special case of [`dequantize_bnb_int8`], kept so that every
/// pre-v0.7.4 caller compiles unchanged.
///
/// # Errors
///
/// See [`dequantize_bnb_int8`].
///
/// # Memory
///
/// See [`dequantize_bnb_int8`]; at `BF16` the output buffer is
/// `out_features × in_features × 2` bytes.
#[inline]
pub fn dequantize_bnb_int8_to_bf16(
    weight_data: &[u8],
    scb_data: &[u8],
    out_features: usize,
    in_features: usize,
) -> crate::Result<Vec<u8>> {
    dequantize_bnb_int8::<Bf16Out>(weight_data, scb_data, out_features, in_features)
}

/// Dequantizes `BitsAndBytes` `INT8` (`LLM.int8()`) quantized weights into `E`.
///
/// Each `I8` weight value is dequantized via
/// `value = (weight_i8 × SCB) × (1 / 127)`, where `SCB` is the per-row absolute
/// maximum and the reciprocal is the `INV_127` constant.
///
/// **The association is load-bearing, not incidental.** Until v0.7.4 this
/// kernel hoisted `SCB / 127` per row and computed `weight_i8 × scale`, which
/// is the same real number but a different `f32` evaluation. Both round to the
/// same `BF16`, so the `BF16` cross-validation reported 0/65536 mismatches and
/// the divergence stayed invisible; comparing at `F32` against
/// `bitsandbytes`' `int8_vectorwise_dequant` showed **17610/65536 (26.9 %)**
/// elements off by exactly 1 `ULP`. Matching `bitsandbytes`' order restores
/// bit-exactness at every output width. See `tests/cross_validation_bnb.rs`.
///
/// # Arguments
///
/// - `weight_data` — `I8` bytes, one per element.
/// - `scb_data` — `F32` per-row absmax values (one per `out_features`).
/// - `out_features` — number of output rows.
/// - `in_features` — number of input columns.
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if tensor dimensions are inconsistent.
///
/// # Memory
///
/// Allocates `out_features × in_features × E::BYTES` bytes of output, plus one
/// `in_features × 4`-byte row scratch. The scratch is new in v0.7.4: it is what
/// lets the narrowing move out of the arithmetic loop and into
/// [`OutputElement::write_scratch`], and it is L1-resident for any realistic
/// row width.
pub fn dequantize_bnb_int8<E: OutputElement>(
    weight_data: &[u8],
    scb_data: &[u8],
    out_features: usize,
    in_features: usize,
) -> crate::Result<Vec<u8>> {
    // --- Validation ---
    let total_elements =
        out_features
            .checked_mul(in_features)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: "BnB INT8 element count overflow".into(),
            })?;
    if weight_data.len() != total_elements {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "BnB INT8 weight byte count mismatch: expected {total_elements}, got {}",
                weight_data.len()
            ),
        });
    }
    let expected_scb_bytes = out_features
        .checked_mul(4)
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "SCB byte count overflow".into(),
        })?;
    if scb_data.len() != expected_scb_bytes {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "BnB INT8 SCB byte count mismatch: expected {expected_scb_bytes}, got {}",
                scb_data.len()
            ),
        });
    }

    // --- Allocate output ---
    let out_byte_len =
        total_elements
            .checked_mul(E::BYTES)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: "BnB INT8 output byte count overflow".into(),
            })?;
    let mut output = vec![0u8; out_byte_len];

    // One register-sized tile, reused for every row. Before v0.7.4 this kernel
    // was a single fused pass, which is also why it was `BF16`-only; the split
    // is what lets `OutputElement::write_scratch` own the narrowing for every
    // width. The tile is [`VECTOR_TILE`]-sized rather than row-sized for the
    // reason that constant documents: a 44 KB row scratch measured 1.115×
    // against the pre-v0.7.4 fused kernel, because the f32s reached memory
    // between the two passes.
    let mut tile = [0.0_f32; VECTOR_TILE];

    // --- Per-row dequantization ---
    // Scale is constant per row → hoisted.
    for row in 0..out_features {
        let scb_val = read_f32_le(scb_data, row * 4).ok_or_else(|| AnamnesisError::Parse {
            reason: format!("BnB INT8 SCB read out of bounds at row {row}"),
        })?;
        // No per-row `SCB / 127` hoist here: the multiply order is part of the
        // contract with `bitsandbytes` (see this function's doc comment), and
        // folding the reciprocal into a row scale is exactly the reassociation
        // that cost 26.9 % of elements 1 ULP at `F32`. `INV_127` is loop-
        // invariant and lands in a register, so the cost is one extra packed
        // multiply per tile rather than a per-element division.

        // Pre-slice for branch-free inner loop (two-level bounds checking)
        let w_start = row * in_features;
        let w_end = w_start + in_features;
        let row_weights = weight_data
            .get(w_start..w_end)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!("BnB INT8 weight row {row} out of bounds"),
            })?;
        let o_start = row * in_features * E::BYTES;
        let o_end = o_start + in_features * E::BYTES;
        let out_row = output
            .get_mut(o_start..o_end)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!("BnB INT8 output row {row} out of bounds"),
            })?;
        // Two passes per tile: `I8` → `f32` × scale, then narrow. Tiling at
        // `VECTOR_TILE` keeps the intermediate `f32`s in registers, so the
        // split costs no memory traffic over the pre-v0.7.4 fused loop.
        let w_tiles = row_weights.chunks_exact(VECTOR_TILE);
        // Read before the iterator is consumed: `remainder` borrows.
        let tail_w = w_tiles.remainder();
        let mut o_tiles = out_row.chunks_exact_mut(VECTOR_TILE * E::BYTES);

        // VECTORIZED: confirmed AVX2 vpmovsxbd + vcvtdq2ps + vmulps on %ymm
        // (8-wide) via cargo-show-asm, x86-64 target-cpu=native, opt-level=3,
        // for all three `E` — each dumps 6 vpmovsxbd, 6 vcvtdq2ps and 12
        // vmulps, i.e. exactly the two packed multiplies (× SCB, × INV_127)
        // per sign-extending load, with no vdivps/vdivss anywhere: the v0.7.3
        // per-row division is gone rather than merely hoisted. The 2 vmulss in
        // each dump belong to the ragged-tail loop below, which is annotated
        // scalar fallback. The narrowing that follows is `write_scratch`'s own
        // confirmed loop.
        //
        // Cost of the association fix, measured because adding a multiply to a
        // bandwidth-bound loop is a hypothesis until it is not: 16.82 -> 17.48
        // ms at 4096 x 11008 (min over 10 interleaved before/after rounds of
        // best-of-5, release, target-cpu=native), i.e. 1.04x slower. Accepted
        // deliberately — it buys bit-exactness against bitsandbytes at F32,
        // where the old association was 1 ULP out on 26.9 % of elements.
        //
        // Measured 15.27 -> 16.98 ms at 4096 x 11008 against the v0.7.3 binary
        // (best-of-5 min of 4 interleaved rounds, release, target-cpu=native),
        // i.e. 1.11x slower. v0.7.3 fused all of this into one loop with the
        // BF16 store inline; the split is what buys F32/F16, and the register
        // tiling is what keeps the cost to ~11% rather than the 44 KB row
        // scratch's measured 1.115x plus L2 traffic.
        for (w_tile, o_tile) in w_tiles.zip(o_tiles.by_ref()) {
            for (&w_byte, value) in w_tile.iter().zip(tile.iter_mut()) {
                // CAST: u8 (from I8 two's complement) → i8 → f32
                #[allow(clippy::as_conversions, clippy::cast_possible_wrap)]
                let w_i8 = w_byte as i8;
                *value = f32::from(w_i8) * scb_val * INV_127;
            }
            E::write_scratch(&tile, o_tile);
        }

        // Edge tile (< VECTOR_TILE elements). `tail_w.len() < VECTOR_TILE ==
        // tile.len()`, so the sub-slice is always `Some`; `get_mut` rather than
        // an index keeps the no-panic floor structural.
        if let Some(tail_tile) = tile.get_mut(..tail_w.len()) {
            // VECTORIZED: scalar fallback — ragged-tail path, at most
            // `VECTOR_TILE - 1` elements per row with no constant trip count
            // to unroll. The full tiles above carry the confirmed 8-wide loop.
            for (&w_byte, value) in tail_w.iter().zip(tail_tile.iter_mut()) {
                // CAST: u8 (from I8 two's complement) → i8 → f32
                #[allow(clippy::as_conversions, clippy::cast_possible_wrap)]
                let w_i8 = w_byte as i8;
                *value = f32::from(w_i8) * scb_val * INV_127;
            }
            E::write_scratch(tail_tile, o_tiles.into_remainder());
        }
    }

    Ok(output)
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
    clippy::cast_precision_loss
)]
mod tests {
    use super::*;

    /// Helper: build F32 LE bytes from a slice of f32 values.
    fn f32_to_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// Helper: read a BF16 value from output bytes at element index.
    fn read_bf16(output: &[u8], idx: usize) -> f32 {
        let offset = idx * 2;
        let bits = u16::from_le_bytes([output[offset], output[offset + 1]]);
        let f32_bits = u32::from(bits) << 16;
        f32::from_bits(f32_bits)
    }

    // --- Sign-of-zero preservation (FP4-style collapsed codebooks) ---

    #[test]
    fn apply_sign_magnitude_zero_flips_only_when_codebook_is_plus_zero() {
        // +0.0 entry + nibble high bit set → emit -0.0.
        let out = apply_sign_magnitude_zero(0.0, 8);
        assert_eq!(
            out.to_bits(),
            0x8000_0000,
            "expected -0.0 (bits 0x80000000)"
        );
        // +0.0 entry + nibble high bit clear → unchanged (still +0.0).
        let out = apply_sign_magnitude_zero(0.0, 0);
        assert_eq!(out.to_bits(), 0x0000_0000, "expected +0.0");
        // -0.0 entry already → unchanged regardless of nibble.
        let out = apply_sign_magnitude_zero(-0.0, 8);
        assert_eq!(out.to_bits(), 0x8000_0000);
        let out = apply_sign_magnitude_zero(-0.0, 0);
        assert_eq!(out.to_bits(), 0x8000_0000);
        // Non-zero entry → unchanged regardless of nibble.
        assert_eq!(apply_sign_magnitude_zero(0.5, 8), 0.5);
        assert_eq!(apply_sign_magnitude_zero(-0.5, 11), -0.5);
        // NF4 codebook[7] = 0.0; nibble 7 has high bit clear → unchanged.
        assert_eq!(apply_sign_magnitude_zero(0.0, 7).to_bits(), 0x0000_0000);
        // NF4 codebook[8] = 0.0795… (non-zero) → unchanged even with high bit set.
        let v = 0.079_580_3_f32;
        assert_eq!(apply_sign_magnitude_zero(v, 8).to_bits(), v.to_bits());
    }

    #[test]
    fn bnb4_decode_preserves_sign_on_collapsed_fp4_codebook() {
        // Build a codebook with +0 at both index 0 and 8 (the
        // bitsandbytes Python FP4 layout); nibble 0 should decode to +0,
        // nibble 8 should decode to -0.
        let mut cb = [0.0f32; 16];
        cb[0] = 0.0;
        cb[8] = 0.0; // collapsed: same bits as cb[0]
        cb[1] = 0.1; // arbitrary non-zero to avoid an all-zero codebook
        cb[9] = -0.1;
        let cb_bytes: Vec<u8> = cb.iter().flat_map(|v| v.to_le_bytes()).collect();
        // Weight bytes: 0x01 (high=0, low=1), 0x89 (high=8, low=9) —
        // bitsandbytes order: high nibble decodes first.
        // 4 elements, block_size=4, absmax=[1.0]. 1 block.
        let weight = vec![0x01u8, 0x89u8];
        let absmax = f32_to_bytes(&[1.0]);
        let out = dequantize_bnb4_to_bf16(&weight, &absmax, &cb_bytes, 4, 4).unwrap();
        let elem0 = u16::from_le_bytes([out[0], out[1]]);
        let elem1 = u16::from_le_bytes([out[2], out[3]]);
        let elem2 = u16::from_le_bytes([out[4], out[5]]);
        let elem3 = u16::from_le_bytes([out[6], out[7]]);
        assert_eq!(elem0, 0x0000, "nibble 0 → +0 BF16");
        assert_eq!(
            elem1 & 0x7FFF,
            0x3DCD & 0x7FFF,
            "nibble 1 → ~0.1 BF16 (magnitude check)"
        );
        assert!(elem1 & 0x8000 == 0, "nibble 1 → positive sign");
        assert_eq!(
            elem2, 0x8000,
            "nibble 8 → -0 BF16 (the new sign-preservation rule)"
        );
        assert!(elem3 & 0x8000 != 0, "nibble 9 → negative sign");
    }

    // --- NF4/FP4 tests ---

    #[test]
    fn bnb4_uniform_lookup() {
        // All bytes = 0x00 → both nibbles = 0 → quant_map[0] * absmax
        let quant_map: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let quant_map_bytes = f32_to_bytes(&quant_map);
        let block_size = 4;
        let num_bytes = 2; // 4 elements = 2 bytes
        let weight_data = vec![0x00u8; num_bytes];
        let absmax_bytes = f32_to_bytes(&[2.0]); // 1 block

        let out =
            dequantize_bnb4_to_bf16(&weight_data, &absmax_bytes, &quant_map_bytes, 4, block_size)
                .unwrap();

        // quant_map[0] = 0.0, so all outputs should be 0.0
        for i in 0..4 {
            assert_eq!(read_bf16(&out, i), 0.0, "element {i}");
        }
    }

    #[test]
    fn bnb4_nibble_extraction() {
        // bitsandbytes order: HIGH nibble decodes to the first element.
        // Byte 0x31 → high nibble = 3 (element 0), low nibble = 1 (element 1)
        // Byte 0x42 → high nibble = 4 (element 2), low nibble = 2 (element 3)
        let quant_map: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let quant_map_bytes = f32_to_bytes(&quant_map);
        let weight_data = vec![0x31, 0x42];
        let absmax_bytes = f32_to_bytes(&[1.0]); // 1 block, scale=1.0

        let out =
            dequantize_bnb4_to_bf16(&weight_data, &absmax_bytes, &quant_map_bytes, 4, 4).unwrap();

        // Element 0: quant_map[3] * 1.0 = 3.0
        assert_eq!(read_bf16(&out, 0), 3.0);
        // Element 1: quant_map[1] * 1.0 = 1.0
        assert_eq!(read_bf16(&out, 1), 1.0);
        // Element 2: quant_map[4] * 1.0 = 4.0
        assert_eq!(read_bf16(&out, 2), 4.0);
        // Element 3: quant_map[2] * 1.0 = 2.0
        assert_eq!(read_bf16(&out, 3), 2.0);
    }

    #[test]
    fn bnb4_absmax_scaling() {
        // quant_map[5] = 0.5, absmax = 4.0 → result = 2.0
        let mut quant_map = [0.0f32; 16];
        quant_map[5] = 0.5;
        let quant_map_bytes = f32_to_bytes(&quant_map);
        let weight_data = vec![0x55]; // both nibbles = 5
        let absmax_bytes = f32_to_bytes(&[4.0]);

        let out =
            dequantize_bnb4_to_bf16(&weight_data, &absmax_bytes, &quant_map_bytes, 2, 2).unwrap();

        assert_eq!(read_bf16(&out, 0), 2.0); // 0.5 * 4.0
        assert_eq!(read_bf16(&out, 1), 2.0);
    }

    #[test]
    fn bnb4_multi_block() {
        // Two blocks with different absmax values
        let quant_map: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let quant_map_bytes = f32_to_bytes(&quant_map);
        // Block 0: byte 0x10 → nibbles (high=1, low=0); Block 1: same
        let weight_data = vec![0x10, 0x10];
        let absmax_bytes = f32_to_bytes(&[1.0, 3.0]);

        let out = dequantize_bnb4_to_bf16(
            &weight_data,
            &absmax_bytes,
            &quant_map_bytes,
            4,
            2, // block_size = 2
        )
        .unwrap();

        // Block 0: quant_map[1]*1.0=1.0, quant_map[0]*1.0=0.0
        assert_eq!(read_bf16(&out, 0), 1.0);
        assert_eq!(read_bf16(&out, 1), 0.0);
        // Block 1: quant_map[1]*3.0=3.0, quant_map[0]*3.0=0.0
        assert_eq!(read_bf16(&out, 2), 3.0);
        assert_eq!(read_bf16(&out, 3), 0.0);
    }

    #[test]
    fn bnb4_validation_errors() {
        let quant_map_bytes = f32_to_bytes(&[0.0; 16]);
        let absmax_bytes = f32_to_bytes(&[1.0]);

        // block_size = 0
        assert!(dequantize_bnb4_to_bf16(&[0], &absmax_bytes, &quant_map_bytes, 2, 0).is_err());

        // Mismatched weight length
        assert!(dequantize_bnb4_to_bf16(&[0, 0], &absmax_bytes, &quant_map_bytes, 2, 2).is_err());

        // Wrong quant_map size
        assert!(dequantize_bnb4_to_bf16(&[0], &absmax_bytes, &[0; 32], 2, 2).is_err());

        // Odd block_size truncates bytes_per_block → rejected with an
        // even-ness message (fires before the weight/absmax checks).
        let err =
            dequantize_bnb4_to_bf16(&[0; 4], &absmax_bytes, &quant_map_bytes, 8, 3).unwrap_err();
        assert!(
            matches!(err, AnamnesisError::Parse { ref reason } if reason.contains("even")),
            "expected even-block_size rejection, got: {err}"
        );
    }

    // --- Double-quant tests ---

    #[test]
    fn bnb4_double_quant_basic() {
        // Nested: absmax U8 value 2 → nested_quant_map[2] * nested_absmax[0]
        let quant_map: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let quant_map_bytes = f32_to_bytes(&quant_map);

        let mut nested_quant_map = [0.0f32; 256];
        nested_quant_map[2] = 0.5; // absmax byte=2 → lookup=0.5
        let nested_quant_map_bytes = f32_to_bytes(&nested_quant_map);

        let nested_absmax_bytes = f32_to_bytes(&[4.0]); // nested scale = 4.0
        // Recovered absmax = nested_quant_map[2] * nested_absmax[0] = 0.5 * 4.0 = 2.0

        let absmax_data = vec![2u8]; // 1 block, absmax byte = 2
        let weight_data = vec![0x10]; // nibbles: high=1, low=0

        let out = dequantize_bnb4_double_quant_to_bf16(
            &weight_data,
            &absmax_data,
            &quant_map_bytes,
            &nested_absmax_bytes,
            &nested_quant_map_bytes,
            0.0, // nested_offset (none for this synthetic input)
            2,   // total_elements
            2,   // block_size
            256, // nested_block_size
        )
        .unwrap();

        // quant_map[1] * 2.0 = 2.0 (high nibble decodes first)
        assert_eq!(read_bf16(&out, 0), 2.0);
        // quant_map[0] * 2.0 = 0.0
        assert_eq!(read_bf16(&out, 1), 0.0);
    }

    #[test]
    fn bnb4_double_quant_applies_nested_offset() {
        // Same setup as `bnb4_double_quant_basic` but with a non-zero
        // nested_offset (the bitsandbytes absmax-mean compression bias):
        // recovered absmax = nested_quant_map[2] * nested_absmax[0] + offset
        //                  = 0.5 * 4.0 + 1.0 = 3.0
        let quant_map: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let quant_map_bytes = f32_to_bytes(&quant_map);

        let mut nested_quant_map = [0.0f32; 256];
        nested_quant_map[2] = 0.5;
        let nested_quant_map_bytes = f32_to_bytes(&nested_quant_map);
        let nested_absmax_bytes = f32_to_bytes(&[4.0]);

        let absmax_data = vec![2u8];
        let weight_data = vec![0x10]; // nibbles: high=1, low=0

        let out = dequantize_bnb4_double_quant_to_bf16(
            &weight_data,
            &absmax_data,
            &quant_map_bytes,
            &nested_absmax_bytes,
            &nested_quant_map_bytes,
            1.0, // nested_offset
            2,
            2,
            256,
        )
        .unwrap();

        // quant_map[1] * 3.0 = 3.0
        assert_eq!(read_bf16(&out, 0), 3.0);
        // quant_map[0] * 3.0 = 0.0
        assert_eq!(read_bf16(&out, 1), 0.0);
    }

    // --- INT8 tests ---

    #[test]
    fn bnb_int8_basic() {
        // 2×2 matrix, SCB = [127.0, 254.0]
        // weight_i8 = [[1, -1], [2, -2]]
        // dequant = weight_i8 * SCB[row] / 127.0
        let weight_data: Vec<u8> = vec![
            1u8,  // i8 = 1
            0xFF, // i8 = -1
            2u8,  // i8 = 2
            0xFE, // i8 = -2
        ];
        let scb_bytes = f32_to_bytes(&[127.0, 254.0]);

        let out = dequantize_bnb_int8_to_bf16(&weight_data, &scb_bytes, 2, 2).unwrap();

        // Row 0: scale = 127.0/127.0 = 1.0
        assert_eq!(read_bf16(&out, 0), 1.0); // 1 * 1.0
        assert_eq!(read_bf16(&out, 1), -1.0); // -1 * 1.0
        // Row 1: scale = 254.0/127.0 = 2.0
        assert_eq!(read_bf16(&out, 2), 4.0); // 2 * 2.0
        assert_eq!(read_bf16(&out, 3), -4.0); // -2 * 2.0
    }

    #[test]
    fn bnb_int8_zero_scale() {
        // SCB = 0.0 → all outputs should be 0.0
        let weight_data = vec![127u8, 1u8]; // i8 = 127, 1
        let scb_bytes = f32_to_bytes(&[0.0]);

        let out = dequantize_bnb_int8_to_bf16(&weight_data, &scb_bytes, 1, 2).unwrap();

        assert_eq!(read_bf16(&out, 0), 0.0);
        assert_eq!(read_bf16(&out, 1), 0.0);
    }

    #[test]
    fn bnb_int8_validation_errors() {
        let scb_bytes = f32_to_bytes(&[1.0]);

        // Mismatched weight length (2 elements but only 1 byte)
        assert!(dequantize_bnb_int8_to_bf16(&[0], &scb_bytes, 1, 2).is_err());

        // Mismatched SCB length (2 rows but only 1 SCB value)
        assert!(dequantize_bnb_int8_to_bf16(&[0; 4], &scb_bytes, 2, 2).is_err());
    }
}
