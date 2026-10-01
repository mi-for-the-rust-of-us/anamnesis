// SPDX-License-Identifier: MIT OR Apache-2.0

//! NVIDIA `NVFP4` dequantisation, in the layout `TensorRT` Model Optimizer
//! (`modelopt`) writes to safetensors.
//!
//! A quantised linear layer `X` is stored as three tensors:
//!
//! - `X.weight`: `U8 [rows, cols / 2]`, two 4-bit `E2M1` codes per byte, the
//!   **low** nibble holding the even column and the high nibble the odd one;
//! - `X.weight_scale`: `F8_E4M3 [rows, cols / 16]`, one scale per 16 columns
//!   (a standard `float8_e4m3fn`, sign bit included);
//! - `X.weight_scale_2`: `F32`, one scale for the whole tensor.
//!
//! The value of column `c` of row `r` is
//!
//! ```text
//! E2M1[code(r, c)] × (E4M3(weight_scale[r, c / 16]) × weight_scale_2)
//! ```
//!
//! with the two scales multiplied **first**. That is `modelopt`'s own
//! association: `NVFP4QTensor.dequantize` computes
//! `per_block_scale = q_per_block_scale * per_block_quant_scale`, then one
//! multiply per value, and the two orders differ in the last bit at `F32`. The `E2M1` table is `modelopt`'s
//! `e2m1_values` verbatim, including the `+0.0` it stores for code 8 (the
//! "negative zero" code), which matters to a bit-exact `F32` comparison.
//!
//! This is a different on-disk format from `GGUF`'s `NVFP4`
//! (`crate::remember::gguf`): `ggml` splits each 16-value group into low and
//! high halves, uses an unsigned `UE4M3` scale, and folds a factor of two into a
//! doubled codebook. The two share only the 4-bit `E2M1` value set.
//!
//! Validated bit for bit, at `BF16` and `F32`, against `modelopt` 0.47.0's
//! `NVFP4QTensor.dequantize` on a real tensor of
//! `nvidia/Llama-3.1-8B-Instruct-NVFP4` (`tests/cross_validation_nvfp4.rs`).

use crate::error::AnamnesisError;
use crate::remember::fp8::e4m3_to_f32_bits;
use crate::remember::output::{Bf16Out, OutputElement};

/// Values per `NVFP4` scale block (`group_size` in `hf_quant_config.json`).
const BLOCK: usize = 16;

/// The 16 `E2M1` values, indexed by the 4-bit code: `modelopt`'s
/// `e2m1_values`, with `+0.0` for code 8 exactly as it stores it.
const E2M1_VALUES: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Dequantises a `modelopt` `NVFP4` weight to `BF16`.
///
/// The [`Bf16Out`] special case of [`dequantize_nvfp4`].
///
/// # Errors
///
/// See [`dequantize_nvfp4`].
///
/// # Memory
///
/// See [`dequantize_nvfp4`]: one output buffer of `rows × cols × 2` bytes.
#[inline]
pub fn dequantize_nvfp4_to_bf16(
    weight: &[u8],
    block_scales: &[u8],
    global_scale: f32,
    rows: usize,
    cols: usize,
) -> crate::Result<Vec<u8>> {
    dequantize_nvfp4::<Bf16Out>(weight, block_scales, global_scale, rows, cols)
}

/// Dequantises a `modelopt` `NVFP4` weight into `E`.
///
/// # Arguments
///
/// * `weight` — the packed `U8` weight, `rows × cols / 2` bytes, row-major;
///   the low nibble of each byte is the even column.
/// * `block_scales` — the `F8_E4M3` `weight_scale`, `rows × cols / 16` bytes,
///   row-major.
/// * `global_scale` — the `F32` `weight_scale_2`.
/// * `rows`, `cols` — the **logical** shape: `cols` counts values, not bytes.
///
/// # Returns
///
/// `rows × cols × E::BYTES` bytes, little-endian, row-major.
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if `cols` is not a multiple of 16, if
/// `rows × cols` (or the output size) overflows `usize`, or if `weight` or
/// `block_scales` does not have the length the shape implies.
///
/// # Memory
///
/// Allocates the output buffer, `rows × cols × E::BYTES` bytes, and nothing
/// else: one 16-value block is decoded at a time into a stack scratch.
pub fn dequantize_nvfp4<E: OutputElement>(
    weight: &[u8],
    block_scales: &[u8],
    global_scale: f32,
    rows: usize,
    cols: usize,
) -> crate::Result<Vec<u8>> {
    if !cols.is_multiple_of(BLOCK) {
        return Err(AnamnesisError::Parse {
            reason: format!("NVFP4 column count {cols} is not a multiple of {BLOCK}"),
        });
    }
    let n_values = rows
        .checked_mul(cols)
        .ok_or_else(|| AnamnesisError::Parse {
            reason: format!("NVFP4 shape {rows} × {cols} overflows usize"),
        })?;
    let out_len = n_values
        .checked_mul(E::BYTES)
        .ok_or_else(|| AnamnesisError::Parse {
            reason: format!("NVFP4 output size {n_values}×{} overflows usize", E::BYTES),
        })?;
    if weight.len() != n_values / 2 {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "NVFP4 weight is {} bytes, expected {} for {rows} × {cols} values",
                weight.len(),
                n_values / 2
            ),
        });
    }
    if block_scales.len() != n_values / BLOCK {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "NVFP4 weight_scale is {} bytes, expected {} for {rows} × {cols} values",
                block_scales.len(),
                n_values / BLOCK
            ),
        });
    }

    let mut out = vec![0u8; out_len];
    let mut scratch = [0.0_f32; BLOCK];
    // Every row is a whole number of blocks (`cols % 16 == 0`), so the
    // row-major weight, scales and output can be walked block by block without
    // tracking rows: block `b` is bytes `8b..8b + 8` of the weight, byte `b` of
    // the scales, and values `16b..16b + 16` of the output.
    //
    // VECTORIZED: scalar fallback, pass 1 of the loop-fission pair: the nibble
    // unpack and table lookup cross byte, integer and float domains. Pass 2 is
    // `OutputElement::write_scratch`, the shared narrowing writer.
    //
    // GENERIC-CONST: the output side cannot use `as_chunks_mut`, whose width is
    // a const generic argument: `as_chunks_mut::<{ BLOCK * E::BYTES }>()` is
    // rejected by stable Rust ("generic parameters may not be used in const
    // operations"). The input side has a plain constant width and does.
    #[allow(clippy::chunks_exact_to_as_chunks)]
    for ((packed, &scale), block_out) in weight
        .as_chunks::<{ BLOCK / 2 }>()
        .0
        .iter()
        .zip(block_scales)
        .zip(out.chunks_exact_mut(BLOCK * E::BYTES))
    {
        // The two scales first, as `modelopt` multiplies them.
        let block_scale = f32::from_bits(e4m3_to_f32_bits(scale)) * global_scale;
        for ([even, odd], &byte) in scratch.as_chunks_mut::<2>().0.iter_mut().zip(packed) {
            // BITWISE: low nibble is the even column, high nibble the odd one
            // INDEX: both nibbles are masked to 0..16, the table has 16 entries
            #[allow(clippy::indexing_slicing)]
            {
                *even = E2M1_VALUES[usize::from(byte & 0x0F)] * block_scale;
                *odd = E2M1_VALUES[usize::from(byte >> 4)] * block_scale;
            }
        }
        E::write_scratch(&scratch, block_out);
    }
    Ok(out)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing, clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::remember::output::F32Out;

    fn f32_words(bytes: &[u8]) -> Vec<f32> {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&w| f32::from_le_bytes(w))
            .collect()
    }

    /// Low nibble is the even column, high nibble the odd one, and each 16
    /// columns take their own scale.
    #[test]
    fn nibble_order_and_block_scales() {
        // One row of 32 values: two blocks. Byte 0 = codes (lo 1, hi 7):
        // columns 0 and 1 are 0.5 and 6.0 times the block scale.
        let mut weight = vec![0u8; 16];
        weight[0] = 0x71;
        weight[8] = 0x9A; // block 2, columns 16/17: codes 10 (-1.0) and 9 (-0.5)
        // E4M3 0x38 = 1.0, 0x40 = 2.0; global scale 0.25.
        let scales = [0x38, 0x40];
        let out = f32_words(&dequantize_nvfp4::<F32Out>(&weight, &scales, 0.25, 1, 32).unwrap());
        assert_eq!(out[0], 0.5 * 0.25);
        assert_eq!(out[1], 6.0 * 0.25);
        assert_eq!(out[16], -0.5);
        assert_eq!(out[17], -0.25);
        assert!(out[2..16].iter().all(|&v| v == 0.0));
    }

    /// Code 8 is `+0.0`, as `modelopt` stores it, not `-0.0`.
    #[test]
    fn code_8_is_positive_zero() {
        let weight = vec![0x88u8; 8];
        let out = f32_words(&dequantize_nvfp4::<F32Out>(&weight, &[0x38], 1.0, 1, 16).unwrap());
        assert!(out.iter().all(|v| v.to_bits() == 0), "{out:?}");
    }

    /// `v × (s1 × s2)`, not `(v × s1) × s2`: chosen so the two orders differ.
    #[test]
    fn scales_are_multiplied_first() {
        // E4M3 0x3B = 1.375; s2 chosen so 1.375 × s2 rounds.
        let s2 = 0.1_f32;
        let weight = vec![0x55u8; 8]; // code 5 = 3.0 everywhere
        let out = f32_words(&dequantize_nvfp4::<F32Out>(&weight, &[0x3B], s2, 1, 16).unwrap());
        let modelopt = 3.0_f32 * (1.375_f32 * s2);
        let other = (3.0_f32 * 1.375_f32) * s2;
        assert_ne!(
            modelopt.to_bits(),
            other.to_bits(),
            "test needs orders that differ"
        );
        assert_eq!(out[0].to_bits(), modelopt.to_bits());
    }

    #[test]
    fn rejects_inconsistent_shapes() {
        assert!(dequantize_nvfp4::<F32Out>(&[0; 8], &[0x38], 1.0, 1, 15).is_err());
        assert!(dequantize_nvfp4::<F32Out>(&[0; 7], &[0x38], 1.0, 1, 16).is_err());
        assert!(dequantize_nvfp4::<F32Out>(&[0; 8], &[0x38, 0x38], 1.0, 1, 16).is_err());
        assert!(dequantize_nvfp4::<F32Out>(&[0; 8], &[0x38], 1.0, usize::MAX, 16).is_err());
    }
}
