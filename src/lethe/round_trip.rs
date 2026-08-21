// SPDX-License-Identifier: MIT OR Apache-2.0

//! Bit-exact round-trip validation harness for codebook-`LUT` encode kernels.
//!
//! Every encode kernel in [`lethe`](mod@crate::lethe) whose forward map is
//! a nearest-codebook-entry lookup (`NF4`, `FP4`, every `GGUF` family
//! shipping in Phase 8.5) shares one validation property: the codebook
//! itself is the oracle. If the codebook entries are distinct, then for
//! every valid index `i` and every non-`NaN` scale `s`,
//!
//! ```text
//! encode(decode(i, s, codebook), s, codebook) == i
//! ```
//!
//! holds unconditionally — the encoder finds the nearest entry to
//! `codebook[i] * s` divided back by `s`, and that nearest entry is
//! `codebook[i]` itself. No external Python reference is required to
//! validate this contract; the codebook is the ground truth.
//!
//! For codebooks with non-injective entries (notably `bitsandbytes`'
//! Python `FP4` `quant_map`, where `quant_map[0] == quant_map[8] ==
//! +0.0` in bits), the property is rescued by the sign-of-zero
//! preservation rule in `dequantize_bnb4_to_bf16` (requires `bnb`
//! feature): the decoder emits `-0.0` when the codebook entry is
//! `+0.0` and the nibble's high bit is set, so the decoded `BF16` is
//! bit-distinct for nibbles 0 and 8 even when the codebook itself is
//! not. The round-trip then holds byte-exactly for every `BnB`-family
//! codebook anamnesis parses.
//!
//! This module hosts the harness primitives that Phase 5 (`BnB`) and
//! Phase 8.5 (`FP8`, `GGUF` …) wire their kernels through. Each helper
//! constructs a minimal synthetic input that exercises every valid index
//! at every supplied scale, then asserts byte-equal recovery.

use crate::error::AnamnesisError;

/// Asserts that every valid `BnB4` nibble round-trips bit-exactly through
/// `decode → encode` for the given codebook and set of scales.
///
/// The input is a 16-byte synthetic weight buffer where byte `i` packs
/// nibble `i` in both the low and high positions; both nibbles are
/// covered by a single block. One `block_size`-element block is used per
/// scale, with all 16 nibble values represented exactly twice per block
/// at minimum (`block_size >= 32` recommended; the harness asserts
/// `block_size >= 32` to guarantee coverage).
///
/// The harness accepts both the `decode` and `encode` functions as
/// closures so it can be invoked from unit tests inside the `bnb`
/// module (requires `bnb` feature) and from integration tests (passing
/// the public re-exports directly).
///
/// # Arguments
///
/// - `codebook` — 16-entry `f32` codebook. Entries must be pairwise
///   distinct (in IEEE 754 bit-pattern terms) for the harness to be
///   meaningful; `+0.0` and `-0.0` count as distinct here.
/// - `scales` — set of per-block absmax values to test. Each scale
///   produces a fresh decode-then-encode round-trip on the same
///   synthetic 32-element block.
/// - `block_size` — elements per absmax block. Must be a multiple of 32
///   and at most the harness's synthetic input length (`16 * 2 =
///   32` elements default — pass `block_size = 32`).
/// - `decode` — closure invoking the kernel-under-test's decode entry
///   point with the standard `(weight_bytes, absmax_bytes,
///   quant_map_bytes, total_elements, block_size)` signature.
/// - `encode` — closure invoking the kernel-under-test's encode entry
///   point with the standard `(bf16_bytes, absmax_bytes, quant_map_bytes,
///   total_elements, block_size)` signature.
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if the harness's own bounds are
/// violated (`block_size` not a multiple of 32, `block_size > 32`,
/// `scales` empty). Decode / encode errors propagate.
///
/// # Panics
///
/// Panics if the round-trip is not byte-equal — that is the harness's
/// purpose. Each panic message names the offending scale, the original
/// byte, and the round-tripped byte.
pub fn assert_bnb4_decode_encode_round_trip<D, E>(
    codebook: &[f32; 16],
    scales: &[f32],
    block_size: usize,
    decode: D,
    encode: E,
) -> crate::Result<()>
where
    D: Fn(&[u8], &[u8], &[u8], usize, usize) -> crate::Result<Vec<u8>>,
    E: Fn(&[u8], &[u8], &[u8], usize, usize) -> crate::Result<Vec<u8>>,
{
    if scales.is_empty() {
        return Err(AnamnesisError::Parse {
            reason: "round-trip harness needs at least one scale".into(),
        });
    }
    if block_size != 32 {
        return Err(AnamnesisError::Parse {
            reason: format!("round-trip harness requires block_size == 32 (got {block_size})"),
        });
    }

    // Synthetic input: 16 bytes; byte i = (i << 4) | i, packing nibble i twice.
    let mut synthetic = [0u8; 16];
    for (i, slot) in synthetic.iter_mut().enumerate() {
        // CAST: i is 0..16, fits u8 trivially
        #[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
        let nibble = i as u8;
        // BITWISE: pack nibble i twice (low + high) so a 16-byte input
        // exposes every nibble value 0..16 in both packed positions
        *slot = (nibble << 4) | nibble;
    }
    let total_elements: usize = 32; // 16 bytes × 2 nibbles per byte

    // Serialize the codebook to F32 LE bytes for the standard decode/encode signature.
    let mut codebook_bytes = [0u8; 64];
    for (i, &entry) in codebook.iter().enumerate() {
        let le = entry.to_le_bytes();
        // INDEX: 4*i+4 <= 64 since i < 16; codebook_bytes is exactly 64 bytes
        #[allow(clippy::indexing_slicing)]
        codebook_bytes[i * 4..i * 4 + 4].copy_from_slice(&le);
    }

    for (scale_idx, &scale) in scales.iter().enumerate() {
        // One scale, one block → 4 bytes of F32 LE absmax.
        let absmax_bytes = scale.to_le_bytes();

        let bf16 = decode(
            &synthetic,
            &absmax_bytes,
            &codebook_bytes,
            total_elements,
            block_size,
        )?;
        let recovered = encode(
            &bf16,
            &absmax_bytes,
            &codebook_bytes,
            total_elements,
            block_size,
        )?;

        assert_eq!(
            recovered.len(),
            synthetic.len(),
            "scale[{scale_idx}] = {scale}: round-trip byte count mismatch \
             (expected {}, got {})",
            synthetic.len(),
            recovered.len(),
        );
        for (byte_idx, (&orig, &back)) in synthetic.iter().zip(recovered.iter()).enumerate() {
            assert_eq!(
                back, orig,
                "scale[{scale_idx}] = {scale}, byte {byte_idx}: \
                 round-trip mismatch (expected 0x{orig:02X}, got 0x{back:02X})",
            );
        }
    }

    Ok(())
}

/// Asserts that every `i8` value round-trips bit-exactly through
/// `BnB INT8` `decode → encode` for the given `SCB` value.
///
/// The input is a 256-byte synthetic weight buffer (one row) where byte
/// `i` is `(i as i8) as u8` — covering every signed value in `[-128,
/// 127]`. `SCB = 127.0` makes the decode scale exactly `1.0`, ensuring
/// that every `i8` value decodes to an integer-valued `BF16` that
/// re-encodes to its original `i8` without ambiguity.
///
/// For non-`127` `SCB` values, edge `i8` values may suffer `BF16`
/// rounding such that `round(bf16 / scale)` saturates at `+127` rather
/// than recovering `-128` exactly; the harness restricts itself to
/// `SCB = 127.0` to keep the contract clean. The general-`SCB` runtime
/// behaviour (with clamp at `+/- 127`) is exercised by the cross-validation
/// tests against real fixtures.
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] from the decode or encode call.
///
/// # Panics
///
/// Panics if the round-trip is not byte-equal at every position.
pub fn assert_bnb_int8_decode_encode_round_trip<D, E>(decode: D, encode: E) -> crate::Result<()>
where
    D: Fn(&[u8], &[u8], usize, usize) -> crate::Result<Vec<u8>>,
    E: Fn(&[u8], &[u8], usize, usize) -> crate::Result<Vec<u8>>,
{
    let mut synthetic = [0u8; 256];
    for (i, slot) in synthetic.iter_mut().enumerate() {
        // CAST: i is 0..256, fits u8 trivially
        #[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
        let byte = i as u8;
        *slot = byte;
    }
    let scb = 127.0_f32.to_le_bytes();
    let bf16 = decode(&synthetic, &scb, 1, 256)?;
    let recovered = encode(&bf16, &scb, 1, 256)?;

    assert_eq!(
        recovered.len(),
        synthetic.len(),
        "INT8 round-trip byte count mismatch (expected {}, got {})",
        synthetic.len(),
        recovered.len(),
    );
    for (byte_idx, (&orig, &back)) in synthetic.iter().zip(recovered.iter()).enumerate() {
        assert_eq!(
            back, orig,
            "INT8 byte {byte_idx}: round-trip mismatch \
             (expected 0x{orig:02X}, got 0x{back:02X})",
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::float_cmp,
    // `toy_decode` / `toy_encode` must keep the Result<Vec<u8>> return
    // type to satisfy the harness closure signature; they never error.
    clippy::unnecessary_wraps
)]
mod tests {
    use super::*;

    /// A toy decode: reads 16 codebook bytes as F32, multiplies by the
    /// per-block absmax (read as F32), and writes BF16 bytes via the
    /// upstream `f32_bits_to_bf16_bits` helper.
    fn toy_decode(
        weight: &[u8],
        absmax: &[u8],
        codebook: &[u8],
        total_elements: usize,
        block_size: usize,
    ) -> crate::Result<Vec<u8>> {
        assert_eq!(weight.len() * 2, total_elements);
        let mut cb = [0.0f32; 16];
        for (i, slot) in cb.iter_mut().enumerate() {
            let arr: [u8; 4] = codebook[i * 4..i * 4 + 4].try_into().unwrap();
            *slot = f32::from_le_bytes(arr);
        }
        let mut out = vec![0u8; total_elements * 2];
        let num_blocks = total_elements / block_size;
        for block_idx in 0..num_blocks {
            let arr: [u8; 4] = absmax[block_idx * 4..block_idx * 4 + 4].try_into().unwrap();
            let scale = f32::from_le_bytes(arr);
            let w_start = block_idx * (block_size / 2);
            let w_end = w_start + block_size / 2;
            for (offset, &byte) in weight[w_start..w_end].iter().enumerate() {
                // bitsandbytes order: high nibble decodes to the first element.
                let high = (byte >> 4) as usize;
                let low = (byte & 0x0F) as usize;
                let val_first = cb[high] * scale;
                let val_second = cb[low] * scale;
                let bf16_first = (val_first.to_bits() >> 16) as u16;
                let bf16_second = (val_second.to_bits() >> 16) as u16;
                let o = (block_idx * block_size + offset * 2) * 2;
                out[o..o + 2].copy_from_slice(&bf16_first.to_le_bytes());
                out[o + 2..o + 4].copy_from_slice(&bf16_second.to_le_bytes());
            }
        }
        Ok(out)
    }

    /// A toy encode: for each BF16, divides by absmax, finds nearest
    /// codebook entry (linear scan with exact-bit-match priority), packs
    /// two nibbles per byte. Independent of `lethe::bnb` so the harness
    /// is exercised standalone.
    fn toy_encode(
        bf16: &[u8],
        absmax: &[u8],
        codebook: &[u8],
        total_elements: usize,
        block_size: usize,
    ) -> crate::Result<Vec<u8>> {
        assert_eq!(bf16.len(), total_elements * 2);
        let mut cb = [0.0f32; 16];
        for (i, slot) in cb.iter_mut().enumerate() {
            let arr: [u8; 4] = codebook[i * 4..i * 4 + 4].try_into().unwrap();
            *slot = f32::from_le_bytes(arr);
        }
        let mut out = vec![0u8; total_elements / 2];
        let num_blocks = total_elements / block_size;
        for block_idx in 0..num_blocks {
            let arr: [u8; 4] = absmax[block_idx * 4..block_idx * 4 + 4].try_into().unwrap();
            let scale = f32::from_le_bytes(arr);
            for pair_idx in 0..block_size / 2 {
                let e0 = block_idx * block_size + pair_idx * 2;
                let arr0: [u8; 2] = bf16[e0 * 2..e0 * 2 + 2].try_into().unwrap();
                let arr1: [u8; 2] = bf16[(e0 + 1) * 2..(e0 + 1) * 2 + 2].try_into().unwrap();
                let f0 = f32::from_bits(u32::from(u16::from_le_bytes(arr0)) << 16) / scale;
                let f1 = f32::from_bits(u32::from(u16::from_le_bytes(arr1)) << 16) / scale;
                let n0 = nearest_idx(f0, &cb);
                let n1 = nearest_idx(f1, &cb);
                let o = block_idx * (block_size / 2) + pair_idx;
                // bitsandbytes order: first element packs into the HIGH nibble.
                out[o] = (n0 << 4) | n1;
            }
        }
        Ok(out)
    }

    fn nearest_idx(v: f32, cb: &[f32; 16]) -> u8 {
        let val_bits = v.to_bits();
        let mut best = 0u8;
        let mut best_d = f32::INFINITY;
        let mut best_exact = false;
        for (i, &c) in cb.iter().enumerate() {
            let exact = c.to_bits() == val_bits;
            let d = (v - c).abs();
            let take = if exact && !best_exact {
                true
            } else if !exact && best_exact {
                false
            } else {
                d < best_d
            };
            if take {
                best = i as u8;
                best_d = d;
                best_exact = exact;
            }
        }
        best
    }

    #[test]
    fn harness_round_trips_linear_codebook() {
        // codebook[i] = (i as f32 - 7.5) * 0.1, distinct entries.
        let mut cb = [0.0f32; 16];
        for (i, slot) in cb.iter_mut().enumerate() {
            *slot = (i as f32 - 7.5) * 0.1;
        }
        let scales = [1.0, 2.0, 0.5, 0.123];
        assert_bnb4_decode_encode_round_trip(&cb, &scales, 32, toy_decode, toy_encode).unwrap();
    }

    #[test]
    fn harness_round_trips_signed_zero_codebook() {
        // codebook with -0.0 and +0.0 in distinct slots — the FP4
        // pathological case. Exact-bit-match priority disambiguates.
        let mut cb = [0.0f32; 16];
        cb[0] = 0.0;
        cb[8] = -0.0;
        for (i, slot) in cb.iter_mut().enumerate() {
            if i != 0 && i != 8 {
                *slot = (i as f32 - 7.5) * 0.1;
            }
        }
        let scales = [1.0, 2.5];
        assert_bnb4_decode_encode_round_trip(&cb, &scales, 32, toy_decode, toy_encode).unwrap();
    }

    #[test]
    fn harness_rejects_bad_block_size() {
        let cb = [0.0f32; 16];
        let scales = [1.0];
        assert!(
            assert_bnb4_decode_encode_round_trip(&cb, &scales, 16, toy_decode, toy_encode).is_err()
        );
    }

    #[test]
    fn harness_rejects_empty_scales() {
        let cb = [0.0f32; 16];
        let scales: [f32; 0] = [];
        assert!(
            assert_bnb4_decode_encode_round_trip(&cb, &scales, 32, toy_decode, toy_encode).is_err()
        );
    }
}
