// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fine-grained `FP8` `E4M3` dequantization.
//!
//! Converts raw `F8_E4M3` bytes with 128×128 block scale factors into output
//! bytes of a caller-chosen width. The conversion pipeline is fully branchless
//! to enable auto-vectorization.
//!
//! # Output width
//!
//! Since v0.7.4 the three entry points are generic over
//! [`OutputElement`](crate::OutputElement): `Bf16Out` (the default and the only
//! width before v0.7.4), `F32Out` (no narrowing step at all, so the emitted
//! value *is* the one the kernel computed) and `F16Out`. The `*_to_bf16` names
//! remain as `#[inline]` `Bf16Out` wrappers, so no existing caller changes.
//!
//! The narrowing is **not** fused into the arithmetic. Each entry point tiles
//! its work into `BLOCK_SIZE`-element runs through a stack `f32` scratch
//! ([`fp8_run_to_output`]) and hands the scratch to
//! [`OutputElement::write_scratch`], which is where every output width in this
//! crate is produced. That keeps the kernel arithmetic identical across output
//! types and gives each width one loop to verify rather than three per family.

use crate::error::AnamnesisError;
use crate::parse::safetensors::Dtype;
use crate::remember::output::{Bf16Out, OutputElement};

/// Block size for fine-grained `FP8` quantization (128×128 elements per block).
const BLOCK_SIZE: usize = 128;

/// `F32` bit patterns for `E4M3` subnormal values (exponent = 0).
///
/// Index `m` (0..7) maps to the `f32` representation of `m × 2⁻⁹`.
/// These are precomputed to avoid branches or float arithmetic in the
/// hot loop.
// INDEX: mant is masked to 3 bits, always 0..7
#[allow(clippy::indexing_slicing)]
const SUBNORMAL_TABLE: [u32; 8] = [
    // BITWISE: E4M3 subnormal f32 bit patterns for mantissa 0..7
    // value = mant × 2^(-9), stored as IEEE 754 f32 bits
    0x0000_0000, // mant=0: 0.0
    0x3B00_0000, // mant=1: 1 × 2^(-9) = 0.001953125
    0x3B80_0000, // mant=2: 2 × 2^(-9) = 0.00390625
    0x3BC0_0000, // mant=3: 3 × 2^(-9) = 0.005859375
    0x3C00_0000, // mant=4: 4 × 2^(-9) = 0.0078125
    0x3C20_0000, // mant=5: 5 × 2^(-9) = 0.009765625
    0x3C40_0000, // mant=6: 6 × 2^(-9) = 0.01171875
    0x3C60_0000, // mant=7: 7 × 2^(-9) = 0.013671875
];

// ---------------------------------------------------------------------------
// Element-level conversion (branchless)
// ---------------------------------------------------------------------------

/// Converts a single `E4M3` byte to its IEEE 754 `f32` bit pattern.
///
/// Completely branchless: uses a const lookup table for subnormals and
/// bitwise select for all control flow. The result is a `u32` containing
/// the `f32` bit pattern (not an `f32` value) to stay in integer domain.
///
/// # Format
///
/// `E4M3`: 1 sign bit, 4 exponent bits (bias 7), 3 mantissa bits.
/// - Normal: `(-1)^s × 2^(exp-7) × (1 + mant/8)`
/// - Subnormal (exp=0): `(-1)^s × mant × 2^(-9)`
/// - `NaN`: exp=15, mant=7 (byte `0x7F` or `0xFF`)
#[must_use]
pub(crate) fn e4m3_to_f32_bits(byte: u8) -> u32 {
    let b = u32::from(byte);

    // BITWISE: extract sign bit from E4M3 byte (bit [7])
    let sign = b >> 7;
    // BITWISE: extract 4-bit exponent from E4M3 byte (bits [6:3])
    let exp = (b >> 3) & 0xF;
    // BITWISE: extract 3-bit mantissa from E4M3 byte (bits [2:0])
    let mant = b & 0x7;

    // --- Normal path (valid when exp > 0) ---
    // BITWISE: construct IEEE 754 f32 from E4M3 normal: sign | biased_exp | mantissa
    // f32 exponent = exp - 7 + 127 = exp + 120; mantissa shifted from 3 to 23 bits
    let normal_bits = (sign << 31) | ((exp + 120) << 23) | (mant << 20);

    // --- Subnormal path (valid when exp == 0) ---
    // BITWISE: look up precomputed f32 bits for subnormal mantissa, apply sign
    // INDEX: mant is masked to 3 bits (0..7), SUBNORMAL_TABLE has 8 entries
    // CAST: u32 → usize, mant is 0..7 and always fits
    #[allow(clippy::indexing_slicing, clippy::as_conversions)]
    let sub_bits = SUBNORMAL_TABLE[mant as usize] | (sign << 31);

    // --- Branchless select: subnormal vs normal ---
    // BITWISE: generate all-ones mask when exp==0 (subnormal), all-zeros otherwise
    // exp.wrapping_sub(1) underflows to 0xFFFF_FFFF when exp==0, so bit 31 is set
    let sub_flag = exp.wrapping_sub(1) >> 31;
    let sub_mask = 0u32.wrapping_sub(sub_flag);
    let selected = (sub_bits & sub_mask) | (normal_bits & !sub_mask);

    // --- NaN override ---
    // BITWISE: detect E4M3 NaN — bits [6:0] == 0x7F (exp=15, mant=7)
    let nan_check = (b & 0x7F) ^ 0x7F; // 0 when NaN
    let nan_flag = nan_check.wrapping_sub(1) >> 31; // 1 when NaN
    let nan_mask = 0u32.wrapping_sub(nan_flag);
    // BITWISE: canonical quiet NaN with original sign
    let nan_bits = (sign << 31) | 0x7FC0_0000;

    // BITWISE: final select — NaN overrides normal/subnormal result
    (nan_bits & nan_mask) | (selected & !nan_mask)
}

/// Converts an IEEE 754 `f32` bit pattern to a `BF16` bit pattern with
/// round-to-nearest-even.
///
/// Completely branchless. Takes the upper 16 bits of the `f32` with
/// proper rounding: when the value is exactly halfway between two `BF16`
/// representable values, it rounds to the one with an even least
/// significant bit.
#[must_use]
pub(crate) fn f32_bits_to_bf16_bits(bits: u32) -> u16 {
    // BITWISE: round-to-nearest-even for f32 → BF16
    // The rounding bias is 0x7FFF plus the LSB of the BF16 result.
    // This ensures ties round to even: if bit 16 (BF16 LSB) is 1 and
    // the truncated bits are exactly 0x8000, the +1 rounds up to even.
    let lsb = (bits >> 16) & 1;
    let rounding_bias = 0x7FFF_u32 + lsb;
    // CAST: u32 → u16, intentional truncation to extract upper 16 bits as BF16
    #[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
    let bf16 = (bits.wrapping_add(rounding_bias) >> 16) as u16;
    bf16
}

/// Converts a single `E4M3` byte to `f32`, multiplied by `scale`.
///
/// This is the hot-loop kernel: [`e4m3_to_f32_bits`] followed by one `f32`
/// multiply, branchless throughout. It deliberately stops at `f32` — the
/// narrowing to the caller's output width happens once per tile in
/// [`OutputElement::write_scratch`], not once per element here. Before v0.7.4
/// this function was `e4m3_to_scaled_bf16` and fused the narrowing in, which is
/// what made the family `BF16`-only.
#[must_use]
fn e4m3_to_scaled_f32(byte: u8, scale: f32) -> f32 {
    let value_bits = e4m3_to_f32_bits(byte);
    f32::from_bits(value_bits) * scale
}

/// Converts a run of `E4M3` bytes at one hoisted `scale` into `out`.
///
/// The shared tail of all three `FP8` entry points, and the only place the
/// family's output width is decided. Work is tiled into [`BLOCK_SIZE`]-element
/// runs through a stack scratch buffer (512 B, L1-resident, no allocation), so
/// pass 1 stays pure `E4M3` → `f32` arithmetic and pass 2 is
/// [`OutputElement::write_scratch`] verbatim.
///
/// # Preconditions
///
/// `out.len() == bytes.len() × E::BYTES`. The iteration pairs chunks of both
/// slices, so a short `out` produces short output rather than a panic and a
/// long one leaves the tail untouched. Every caller sizes both from the same
/// element count, so neither happens in practice.
#[inline]
fn fp8_run_to_output<E: OutputElement>(bytes: &[u8], scale: f32, out: &mut [u8]) {
    let mut scratch = [0.0_f32; BLOCK_SIZE];
    let in_tiles = bytes.chunks_exact(BLOCK_SIZE);
    // Read before the iterator is consumed: `remainder` borrows rather than
    // advances, so this is the edge run the loop below will not cover.
    let tail_in = in_tiles.remainder();
    let mut out_tiles = out.chunks_exact_mut(BLOCK_SIZE * E::BYTES);

    // VECTORIZED: pending cargo-show-asm verification
    for (in_tile, out_tile) in in_tiles.zip(out_tiles.by_ref()) {
        for (&byte, dst) in in_tile.iter().zip(scratch.iter_mut()) {
            *dst = e4m3_to_scaled_f32(byte, scale);
        }
        E::write_scratch(&scratch, out_tile);
    }

    // Edge run (< BLOCK_SIZE elements). `chunks_exact` hands back exactly the
    // elements that did not fill a tile, and `into_remainder` the bytes that
    // did not fill an output tile, so the two lengths correspond.
    let tail_out = out_tiles.into_remainder();
    // `tail_in.len() < BLOCK_SIZE == scratch.len()`, so this is always `Some`;
    // `get_mut` is used rather than a slice index so the crate's no-panic floor
    // is upheld by construction rather than by an argument in a comment.
    if let Some(tail_scratch) = scratch.get_mut(..tail_in.len()) {
        // VECTORIZED: pending cargo-show-asm verification
        for (&byte, dst) in tail_in.iter().zip(tail_scratch.iter_mut()) {
            *dst = e4m3_to_scaled_f32(byte, scale);
        }
        E::write_scratch(tail_scratch, tail_out);
    }
}

// ---------------------------------------------------------------------------
// Scale factor loading
// ---------------------------------------------------------------------------

/// Reads a scale factor from raw little-endian bytes at the given
/// block position. Supports `F32`, `BF16`, and `F16` scales.
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if the byte offset is out of bounds
/// or the dtype is unsupported.
fn load_scale(
    scale_data: &[u8],
    block_row: usize,
    block_col: usize,
    scale_cols: usize,
    scale_dtype: Dtype,
) -> crate::Result<f32> {
    let bps = scale_dtype.byte_size();
    let scale_idx = block_row
        .checked_mul(scale_cols)
        .and_then(|v| v.checked_add(block_col))
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "scale index overflow".into(),
        })?;
    let byte_offset = scale_idx
        .checked_mul(bps)
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "scale byte offset overflow".into(),
        })?;
    let end = byte_offset
        .checked_add(bps)
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "scale byte range overflow".into(),
        })?;
    let slice = scale_data
        .get(byte_offset..end)
        .ok_or_else(|| AnamnesisError::Parse {
            reason: format!(
                "scale data too short: need bytes {byte_offset}..{end}, have {}",
                scale_data.len()
            ),
        })?;
    read_scale_bytes(slice, scale_dtype)
}

/// Converts raw little-endian scale bytes to `f32` based on dtype.
fn read_scale_bytes(slice: &[u8], dtype: Dtype) -> crate::Result<f32> {
    match dtype {
        Dtype::F32 => {
            let arr: [u8; 4] = slice.try_into().map_err(|_| AnamnesisError::Parse {
                reason: "scale slice is not 4 bytes".into(),
            })?;
            Ok(f32::from_le_bytes(arr))
        }
        Dtype::BF16 => {
            let arr: [u8; 2] = slice.try_into().map_err(|_| AnamnesisError::Parse {
                reason: "scale slice is not 2 bytes".into(),
            })?;
            // BITWISE: BF16 → f32 by shifting into upper 16 bits of IEEE 754
            let f32_bits = u32::from(u16::from_le_bytes(arr)) << 16;
            Ok(f32::from_bits(f32_bits))
        }
        Dtype::F16 => {
            let arr: [u8; 2] = slice.try_into().map_err(|_| AnamnesisError::Parse {
                reason: "scale slice is not 2 bytes".into(),
            })?;
            // BITWISE: F16 → f32 via half crate's IEEE 754 conversion
            Ok(half::f16::from_le_bytes(arr).to_f32())
        }
        Dtype::F8E4M3
        | Dtype::F8E5M2
        | Dtype::F64
        | Dtype::Bool
        | Dtype::U8
        | Dtype::I8
        | Dtype::U16
        | Dtype::I16
        | Dtype::U32
        | Dtype::I32
        | Dtype::U64
        | Dtype::I64 => Err(AnamnesisError::Parse {
            reason: format!("unsupported scale dtype: {dtype}"),
        }),
    }
}

// ---------------------------------------------------------------------------
// Block-level dequantization (public API)
// ---------------------------------------------------------------------------

/// Dequantizes a fine-grained `FP8` `E4M3` weight tensor to `BF16`.
///
/// The [`Bf16Out`] special case of [`dequantize_fp8`], kept so that every
/// pre-v0.7.4 caller compiles unchanged.
///
/// # Errors
///
/// See [`dequantize_fp8`].
///
/// # Memory
///
/// See [`dequantize_fp8`]; at `BF16` the output buffer is `rows × cols × 2`
/// bytes.
#[inline]
pub fn dequantize_fp8_to_bf16(
    weight_data: &[u8],
    scale_data: &[u8],
    rows: usize,
    cols: usize,
    scale_dtype: Dtype,
) -> crate::Result<Vec<u8>> {
    dequantize_fp8::<Bf16Out>(weight_data, scale_data, rows, cols, scale_dtype)
}

/// Dequantizes a fine-grained `FP8` `E4M3` weight tensor into `E`.
///
/// Each 128×128 block of the weight tensor shares one `F32` scale factor.
/// The formula is: `E(FP8_to_f32(byte) × scale)`, where the only step that
/// depends on `E` is the final write. With [`F32Out`](crate::F32Out) there is
/// no narrowing step at all, so the emitted value is the one the kernel
/// computed.
///
/// # Arguments
///
/// * `weight_data` — raw `F8_E4M3` bytes in row-major order (1 byte per element).
/// * `scale_data` — raw scale factors in row-major order, little-endian.
///   Shape: `[⌈rows/128⌉, ⌈cols/128⌉]`.
/// * `rows` — number of rows in the weight tensor.
/// * `cols` — number of columns in the weight tensor.
/// * `scale_dtype` — dtype of the scale tensor (`F32` or `BF16`).
///
/// # Returns
///
/// A `Vec<u8>` of length `rows × cols × E::BYTES`, in little-endian byte
/// order, suitable for writing directly into a `.safetensors` output file
/// whose header declares `E::DTYPE`.
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if `weight_data` length does not match
/// `rows × cols`, or if `scale_data` is incompatible with the weight
/// dimensions and block size.
///
/// # Memory
///
/// Allocates a single output buffer of `rows × cols × E::BYTES` bytes, plus a
/// fixed 512-byte stack tile. No intermediate heap allocations. Peak memory is
/// input + output (~3× the `FP8` weight size at `BF16`, ~5× at `F32`).
pub fn dequantize_fp8<E: OutputElement>(
    weight_data: &[u8],
    scale_data: &[u8],
    rows: usize,
    cols: usize,
    scale_dtype: Dtype,
) -> crate::Result<Vec<u8>> {
    // --- Validation ---
    let bytes_per_scale = scale_dtype.byte_size();
    if bytes_per_scale == 0 {
        return Err(AnamnesisError::Parse {
            reason: format!("unsupported scale dtype: {scale_dtype}"),
        });
    }

    let expected_weight_len = rows
        .checked_mul(cols)
        .ok_or_else(|| AnamnesisError::Parse {
            reason: format!("rows × cols overflow: {rows} × {cols}"),
        })?;
    if weight_data.len() != expected_weight_len {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "weight data length {} != rows × cols {expected_weight_len}",
                weight_data.len()
            ),
        });
    }

    // Derive scale grid dimensions from the actual scale tensor data.
    // The scale tensor may be stored as 2D [scale_rows, scale_cols] or
    // 1D [scale_rows * scale_cols] — either way, the byte count tells us
    // the total number of scale elements.
    if !scale_data.len().is_multiple_of(bytes_per_scale) {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "scale data length {} is not a multiple of {bytes_per_scale} ({scale_dtype})",
                scale_data.len()
            ),
        });
    }
    let scale_elements = scale_data.len() / bytes_per_scale;
    let scale_rows = rows.div_ceil(BLOCK_SIZE);
    if scale_rows == 0 {
        return Err(AnamnesisError::Parse {
            reason: "zero rows".into(),
        });
    }
    if !scale_elements.is_multiple_of(scale_rows) {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "scale grid is not rectangular: {scale_elements} elements / {scale_rows} rows \
                 has remainder {}",
                scale_elements % scale_rows
            ),
        });
    }
    let scale_cols = scale_elements / scale_rows;
    let col_blocks_needed = cols.div_ceil(BLOCK_SIZE);
    if scale_cols < col_blocks_needed {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "scale has {scale_cols} column blocks but weight needs {col_blocks_needed} \
                 (cols={cols}, block_size={BLOCK_SIZE})"
            ),
        });
    }

    // --- Allocate output ---
    let out_byte_len =
        expected_weight_len
            .checked_mul(E::BYTES)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: "output size overflow".into(),
            })?;
    let mut output = vec![0u8; out_byte_len];

    // --- Row-by-row, column-block iteration ---
    for r in 0..rows {
        let block_row = r / BLOCK_SIZE;
        let row_offset = r.checked_mul(cols).ok_or_else(|| AnamnesisError::Parse {
            reason: "row offset overflow".into(),
        })?;
        let row_w = weight_data
            .get(row_offset..row_offset + cols)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!("weight row {r} out of bounds"),
            })?;
        let out_row_offset =
            row_offset
                .checked_mul(E::BYTES)
                .ok_or_else(|| AnamnesisError::Parse {
                    reason: "output row offset overflow".into(),
                })?;
        let row_o = output
            .get_mut(out_row_offset..out_row_offset + cols * E::BYTES)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!("output row {r} out of bounds"),
            })?;

        // Full 128-column blocks via chunks_exact
        let full_blocks = row_w.chunks_exact(BLOCK_SIZE);
        let remainder_w = full_blocks.remainder();

        for (block_col, w_chunk) in full_blocks.enumerate() {
            let scale = load_scale(scale_data, block_row, block_col, scale_cols, scale_dtype)?;
            let o_start = block_col * BLOCK_SIZE * E::BYTES;
            let o_chunk = row_o
                .get_mut(o_start..o_start + BLOCK_SIZE * E::BYTES)
                .ok_or_else(|| AnamnesisError::Parse {
                    reason: format!("output chunk at row {r}, block_col {block_col} out of bounds"),
                })?;

            // Hot inner run: 128 elements with a hoisted scale. The output
            // width lives entirely inside `fp8_run_to_output`.
            fp8_run_to_output::<E>(w_chunk, scale, o_chunk);
        }

        // Edge column block (< 128 columns)
        if !remainder_w.is_empty() {
            let last_block_col = cols / BLOCK_SIZE;
            let scale = load_scale(
                scale_data,
                block_row,
                last_block_col,
                scale_cols,
                scale_dtype,
            )?;
            let o_start = last_block_col * BLOCK_SIZE * E::BYTES;
            let o_chunk = row_o
                .get_mut(o_start..o_start + remainder_w.len() * E::BYTES)
                .ok_or_else(|| AnamnesisError::Parse {
                    reason: format!("output remainder at row {r} out of bounds"),
                })?;

            fp8_run_to_output::<E>(remainder_w, scale, o_chunk);
        }
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// Per-tensor dequantization (public API)
// ---------------------------------------------------------------------------

/// Dequantizes a per-tensor `FP8` `E4M3` weight tensor to `BF16`.
///
/// The [`Bf16Out`] special case of [`dequantize_per_tensor_fp8`], kept so that
/// every pre-v0.7.4 caller compiles unchanged.
///
/// # Errors
///
/// See [`dequantize_per_tensor_fp8`].
///
/// # Memory
///
/// See [`dequantize_per_tensor_fp8`]; at `BF16` the output buffer is
/// `weight_data.len() × 2` bytes.
#[inline]
pub fn dequantize_per_tensor_fp8_to_bf16(weight_data: &[u8], scale: f32) -> crate::Result<Vec<u8>> {
    dequantize_per_tensor_fp8::<Bf16Out>(weight_data, scale)
}

/// Dequantizes a per-tensor `FP8` `E4M3` weight tensor into `E`.
///
/// The entire tensor shares a single `F32` scale factor. This is the simpler
/// case compared to fine-grained (block-wise) dequantization.
/// The formula is: `E(FP8_to_f32(byte) × scale)`.
///
/// # Arguments
///
/// * `weight_data` — raw `F8_E4M3` bytes (1 byte per element).
/// * `scale` — single `F32` scale factor for the entire tensor.
///
/// # Returns
///
/// A `Vec<u8>` of length `weight_data.len() × E::BYTES`, in little-endian byte
/// order.
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if the output size overflows.
///
/// # Memory
///
/// Allocates a single output buffer of `weight_data.len() × E::BYTES` bytes,
/// plus a fixed 512-byte stack tile. Peak memory is input + output (~3× the
/// `FP8` weight size at `BF16`, ~5× at `F32`).
pub fn dequantize_per_tensor_fp8<E: OutputElement>(
    weight_data: &[u8],
    scale: f32,
) -> crate::Result<Vec<u8>> {
    let out_byte_len =
        weight_data
            .len()
            .checked_mul(E::BYTES)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: "output size overflow".into(),
            })?;
    let mut output = vec![0u8; out_byte_len];

    // One flat run over the whole tensor: the scale is a single value, so
    // there is nothing to hoist per block and the tiling inside
    // `fp8_run_to_output` is the only structure the loop needs.
    fp8_run_to_output::<E>(weight_data, scale, &mut output);

    Ok(output)
}

// ---------------------------------------------------------------------------
// Per-channel dequantization (public API)
// ---------------------------------------------------------------------------

/// Dequantizes a per-channel `FP8` `E4M3` weight tensor to `BF16`.
///
/// The [`Bf16Out`] special case of [`dequantize_per_channel_fp8`], kept so that
/// every pre-v0.7.4 caller compiles unchanged.
///
/// # Errors
///
/// See [`dequantize_per_channel_fp8`].
#[inline]
pub fn dequantize_per_channel_fp8_to_bf16(
    weight_data: &[u8],
    scale_data: &[u8],
    rows: usize,
    cols: usize,
    scale_dtype: Dtype,
) -> crate::Result<Vec<u8>> {
    dequantize_per_channel_fp8::<Bf16Out>(weight_data, scale_data, rows, cols, scale_dtype)
}

/// Dequantizes a per-channel `FP8` `E4M3` weight tensor into `E`.
///
/// Each row of the weight tensor has its own scale factor (shape `[rows, 1]`).
/// The formula is: `E(FP8_to_f32(weight[r, c]) × scale[r])`.
///
/// # Arguments
///
/// * `weight_data` — raw `F8_E4M3` bytes in row-major order (1 byte per element).
/// * `scale_data` — raw scale factor bytes in row-major order, one per row.
/// * `rows` — number of rows in the weight tensor.
/// * `cols` — number of columns in the weight tensor.
/// * `scale_dtype` — dtype of the scale tensor (`F32`, `BF16`, or `F16`).
///
/// # Returns
///
/// A `Vec<u8>` of length `rows × cols × E::BYTES`, in little-endian byte order.
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if dimensions or scale data are inconsistent.
///
/// # Memory
///
/// Allocates a single output buffer of `rows × cols × E::BYTES` bytes, plus a
/// fixed 512-byte stack tile. Peak memory is input + output (~3× the `FP8`
/// weight size at `BF16`, ~5× at `F32`).
pub fn dequantize_per_channel_fp8<E: OutputElement>(
    weight_data: &[u8],
    scale_data: &[u8],
    rows: usize,
    cols: usize,
    scale_dtype: Dtype,
) -> crate::Result<Vec<u8>> {
    let bps = scale_dtype.byte_size();
    let expected_weight_len = rows
        .checked_mul(cols)
        .ok_or_else(|| AnamnesisError::Parse {
            reason: format!("rows × cols overflow: {rows} × {cols}"),
        })?;
    if weight_data.len() != expected_weight_len {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "weight data length {} != rows × cols {expected_weight_len}",
                weight_data.len()
            ),
        });
    }
    let expected_scale_len = rows.checked_mul(bps).ok_or_else(|| AnamnesisError::Parse {
        reason: "scale byte count overflow".into(),
    })?;
    if scale_data.len() != expected_scale_len {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "per-channel scale data length {} != expected {expected_scale_len} \
                 (rows={rows}, {bps} bytes per scale)",
                scale_data.len()
            ),
        });
    }

    let out_byte_len =
        expected_weight_len
            .checked_mul(E::BYTES)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: "output size overflow".into(),
            })?;
    let mut output = vec![0u8; out_byte_len];

    // Per-row iteration: the scale is hoisted per row, and each row is one
    // run through `fp8_run_to_output`.
    for r in 0..rows {
        let scale_offset = r * bps;
        let scale_slice = scale_data
            .get(scale_offset..scale_offset + bps)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!("per-channel scale for row {r} out of bounds"),
            })?;
        let scale = read_scale_bytes(scale_slice, scale_dtype)?;

        let row_start = r * cols;
        let row_w = weight_data
            .get(row_start..row_start + cols)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!("weight row {r} out of bounds"),
            })?;
        let out_row_start = row_start * E::BYTES;
        let row_o = output
            .get_mut(out_row_start..out_row_start + cols * E::BYTES)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!("output row {r} out of bounds"),
            })?;

        fp8_run_to_output::<E>(row_w, scale, row_o);
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
    clippy::cast_possible_truncation
)]
mod tests {
    use super::*;

    // -- e4m3_to_f32_bits: individual known values ---------------------------

    fn bits_to_f32(bits: u32) -> f32 {
        f32::from_bits(bits)
    }

    #[test]
    fn e4m3_zero() {
        assert_eq!(bits_to_f32(e4m3_to_f32_bits(0x00)), 0.0);
    }

    #[test]
    fn e4m3_negative_zero() {
        let val = bits_to_f32(e4m3_to_f32_bits(0x80));
        assert!(val.is_sign_negative());
        assert_eq!(val, -0.0);
    }

    #[test]
    fn e4m3_one() {
        // exp=7, mant=0 → 2^(7-7) × (1 + 0/8) = 1.0
        assert_eq!(bits_to_f32(e4m3_to_f32_bits(0x38)), 1.0);
    }

    #[test]
    fn e4m3_negative_one() {
        assert_eq!(bits_to_f32(e4m3_to_f32_bits(0xB8)), -1.0);
    }

    #[test]
    fn e4m3_two() {
        // exp=8, mant=0 → 2^(8-7) × 1 = 2.0
        assert_eq!(bits_to_f32(e4m3_to_f32_bits(0x40)), 2.0);
    }

    #[test]
    fn e4m3_half() {
        // exp=6, mant=0 → 2^(6-7) × 1 = 0.5
        assert_eq!(bits_to_f32(e4m3_to_f32_bits(0x30)), 0.5);
    }

    #[test]
    fn e4m3_max_normal() {
        // exp=15, mant=6 → 2^(15-7) × (1 + 6/8) = 256 × 1.75 = 448.0
        assert_eq!(bits_to_f32(e4m3_to_f32_bits(0x7E)), 448.0);
    }

    #[test]
    fn e4m3_min_positive_normal() {
        // exp=1, mant=0 → 2^(1-7) × 1 = 2^(-6) = 0.015625
        assert_eq!(bits_to_f32(e4m3_to_f32_bits(0x08)), 0.015_625);
    }

    #[test]
    fn e4m3_min_positive_subnormal() {
        // exp=0, mant=1 → 1 × 2^(-9) = 0.001953125
        assert_eq!(bits_to_f32(e4m3_to_f32_bits(0x01)), 0.001_953_125);
    }

    #[test]
    fn e4m3_max_subnormal() {
        // exp=0, mant=7 → 7 × 2^(-9) = 0.013671875
        assert_eq!(bits_to_f32(e4m3_to_f32_bits(0x07)), 0.013_671_875);
    }

    #[test]
    fn e4m3_nan_positive() {
        assert!(bits_to_f32(e4m3_to_f32_bits(0x7F)).is_nan());
    }

    #[test]
    fn e4m3_nan_negative() {
        let val = bits_to_f32(e4m3_to_f32_bits(0xFF));
        assert!(val.is_nan());
    }

    // -- Exhaustive cross-validation against float8 crate --------------------

    #[test]
    fn exhaustive_cross_validation_with_float8() {
        for byte_val in 0u16..=255 {
            // CAST: u16 → u8, loop range is 0..=255
            #[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
            let byte = byte_val as u8;

            let our_f32 = bits_to_f32(e4m3_to_f32_bits(byte));
            let ref_f32 = float8::F8E4M3::from_bits(byte).to_f32();

            if ref_f32.is_nan() {
                assert!(
                    our_f32.is_nan(),
                    "byte {byte:#04X}: expected NaN, got {our_f32}"
                );
            } else {
                assert_eq!(
                    our_f32, ref_f32,
                    "byte {byte:#04X}: our={our_f32}, ref={ref_f32}"
                );
            }
        }
    }

    // -- f32_bits_to_bf16_bits -----------------------------------------------

    #[test]
    fn bf16_one() {
        assert_eq!(f32_bits_to_bf16_bits(1.0_f32.to_bits()), 0x3F80);
    }

    #[test]
    fn bf16_zero() {
        assert_eq!(f32_bits_to_bf16_bits(0.0_f32.to_bits()), 0x0000);
    }

    #[test]
    fn bf16_negative_one() {
        // -1.0 in f32 = 0xBF800000, BF16 = 0xBF80
        assert_eq!(f32_bits_to_bf16_bits((-1.0_f32).to_bits()), 0xBF80);
    }

    #[test]
    fn bf16_nan() {
        let nan_bits = f32::NAN.to_bits();
        let bf16 = f32_bits_to_bf16_bits(nan_bits);
        // Upper 16 bits of any NaN should still be NaN in BF16
        let reconstructed = f32::from_bits(u32::from(bf16) << 16);
        assert!(reconstructed.is_nan());
    }

    #[test]
    fn bf16_round_to_nearest_even() {
        // Test round-to-nearest-even: f32 value exactly halfway between
        // two BF16 values should round to the one with even LSB.
        //
        // BF16 1.0 = 0x3F80 (LSB=0), next BF16 = 0x3F81 (LSB=1)
        // Halfway point: 0x3F80_8000 in f32
        // Since BF16 LSB would be 0, it should stay at 0x3F80 (round down).
        assert_eq!(f32_bits_to_bf16_bits(0x3F80_8000), 0x3F80);

        // BF16 1.0078125 = 0x3F81 (LSB=1), next BF16 = 0x3F82 (LSB=0)
        // Halfway point: 0x3F81_8000 in f32
        // Since BF16 LSB would be 1, it should round up to 0x3F82 (round to even).
        assert_eq!(f32_bits_to_bf16_bits(0x3F81_8000), 0x3F82);
    }

    // -- e4m3_to_scaled_f32 --------------------------------------------------
    //
    // These assert the *unnarrowed* value, which is the whole point of the
    // v0.7.4 split: the kernel now stops at `f32` and the output width is
    // applied once per tile by `OutputElement::write_scratch`. The `BF16`
    // results the pre-v0.7.4 versions of these tests asserted are still
    // covered, one layer out, by the `dequantize_*_to_bf16` block tests below.

    #[test]
    fn scaled_f32_identity() {
        // scale=1.0: result should match unscaled conversion
        let byte = 0x38; // 1.0 in E4M3
        assert_eq!(e4m3_to_scaled_f32(byte, 1.0), 1.0);
    }

    #[test]
    fn scaled_f32_by_two() {
        // 1.0 × 2.0 = 2.0
        assert_eq!(e4m3_to_scaled_f32(0x38, 2.0), 2.0);
    }

    #[test]
    fn scaled_f32_nan_times_scale() {
        assert!(e4m3_to_scaled_f32(0x7F, 42.0).is_nan());
    }

    #[test]
    fn scaled_f32_zero_times_scale() {
        assert_eq!(e4m3_to_scaled_f32(0x00, 100.0), 0.0);
    }

    #[test]
    fn scaled_f32_keeps_bits_bf16_would_round() {
        // The reason the split exists. 0x39 is E4M3 1.125; times 1.0009766
        // the product needs more significand bits than BF16 holds, so the
        // f32 the kernel now returns is strictly more informative than the
        // BF16 the pre-v0.7.4 kernel returned.
        let scale = 1.0_f32 + 1.0 / 1024.0;
        let exact = e4m3_to_scaled_f32(0x39, scale);
        let narrowed = f32::from_bits(u32::from(f32_bits_to_bf16_bits(exact.to_bits())) << 16);
        assert_ne!(exact.to_bits(), narrowed.to_bits());
    }

    // -- fp8_run_to_output ---------------------------------------------------

    #[test]
    fn run_to_output_covers_the_tail_past_a_full_tile() {
        // BLOCK_SIZE + 3 elements: one full tile plus a 3-element remainder.
        // The remainder path is the one a `chunks_exact`-only implementation
        // would silently drop, so it is asserted explicitly at every width.
        let bytes = vec![0x38u8; BLOCK_SIZE + 3]; // all 1.0 in E4M3

        let mut bf16 = vec![0u8; bytes.len() * 2];
        fp8_run_to_output::<Bf16Out>(&bytes, 2.0, &mut bf16);
        for pair in bf16.chunks_exact(2) {
            assert_eq!(pair, &[0x00, 0x40], "BF16 2.0");
        }

        let mut f32_out = vec![0u8; bytes.len() * 4];
        fp8_run_to_output::<crate::F32Out>(&bytes, 2.0, &mut f32_out);
        for word in f32_out.chunks_exact(4) {
            let v = f32::from_le_bytes([word[0], word[1], word[2], word[3]]);
            assert_eq!(v, 2.0, "F32 2.0");
        }

        let mut f16_out = vec![0u8; bytes.len() * 2];
        fp8_run_to_output::<crate::F16Out>(&bytes, 2.0, &mut f16_out);
        for pair in f16_out.chunks_exact(2) {
            let v = half::f16::from_le_bytes([pair[0], pair[1]]);
            assert_eq!(f32::from(v), 2.0, "F16 2.0");
        }
    }

    // -- dequantize_fp8_to_bf16: block-level tests ---------------------------

    /// Helper: build scale data from a flat slice of f32 values.
    fn make_scale_bytes(scales: &[f32]) -> Vec<u8> {
        scales.iter().flat_map(|s| s.to_le_bytes()).collect()
    }

    #[test]
    fn single_block_128x128() {
        let rows = 128;
        let cols = 128;
        // All elements = 0x38 (1.0 in E4M3), scale = 2.0
        let weight_data = vec![0x38u8; rows * cols];
        let scale_data = make_scale_bytes(&[2.0]);

        let output =
            dequantize_fp8_to_bf16(&weight_data, &scale_data, rows, cols, Dtype::F32).unwrap();

        assert_eq!(output.len(), rows * cols * 2);
        // Every BF16 element should be 2.0 (0x4000 in LE = [0x00, 0x40])
        for chunk in output.chunks_exact(2) {
            assert_eq!(chunk, &[0x00, 0x40], "expected BF16 2.0");
        }
    }

    #[test]
    fn multi_block_256x256() {
        let rows = 256;
        let cols = 256;
        // 2×2 blocks, each with a different scale
        let weight_data = vec![0x38u8; rows * cols]; // all 1.0 in E4M3
        let scales = [1.0_f32, 2.0, 3.0, 4.0]; // 2×2 scale grid
        let scale_data = make_scale_bytes(&scales);

        let output =
            dequantize_fp8_to_bf16(&weight_data, &scale_data, rows, cols, Dtype::F32).unwrap();

        // Check a sample element from each block
        // Block (0,0) at position (0,0): scale=1.0, expect BF16(1.0)=0x3F80
        assert_eq!(&output[0..2], &[0x80, 0x3F]);
        // Block (0,1) at position (0,128): scale=2.0, expect BF16(2.0)=0x4000
        assert_eq!(&output[256..258], &[0x00, 0x40]);
        // Block (1,0) at position (128,0): scale=3.0, expect BF16(3.0)=0x4040
        let offset_10 = 128 * 256 * 2;
        assert_eq!(&output[offset_10..offset_10 + 2], &[0x40, 0x40]);
        // Block (1,1) at position (128,128): scale=4.0, expect BF16(4.0)=0x4080
        let offset_11 = offset_10 + 128 * 2;
        assert_eq!(&output[offset_11..offset_11 + 2], &[0x80, 0x40]);
    }

    #[test]
    fn edge_block_130x130() {
        let rows = 130;
        let cols = 130;
        // 2×2 scale grid (ceil(130/128) = 2 in each dimension)
        let weight_data = vec![0x38u8; rows * cols]; // all 1.0 in E4M3
        let scales = [1.0_f32, 2.0, 3.0, 4.0];
        let scale_data = make_scale_bytes(&scales);

        let output =
            dequantize_fp8_to_bf16(&weight_data, &scale_data, rows, cols, Dtype::F32).unwrap();
        assert_eq!(output.len(), rows * cols * 2);

        // Block (0,0): position (0,0), scale=1.0 → BF16(1.0)
        assert_eq!(&output[0..2], &[0x80, 0x3F]);
        // Block (0,1): position (0,128), scale=2.0 → BF16(2.0)
        assert_eq!(&output[256..258], &[0x00, 0x40]);
        // Edge element at (0,129): still block (0,1), scale=2.0 → BF16(2.0)
        assert_eq!(&output[258..260], &[0x00, 0x40]);
    }

    #[test]
    fn single_element_1x1() {
        let weight_data = vec![0x38u8]; // 1.0
        let scale_data = make_scale_bytes(&[3.0]);

        let output = dequantize_fp8_to_bf16(&weight_data, &scale_data, 1, 1, Dtype::F32).unwrap();
        assert_eq!(output.len(), 2);
        // 1.0 × 3.0 = 3.0 → BF16 0x4040 → LE [0x40, 0x40]
        assert_eq!(&output[..], &[0x40, 0x40]);
    }

    #[test]
    fn single_row_1x128() {
        let weight_data = vec![0x40u8; 128]; // all 2.0 in E4M3
        let scale_data = make_scale_bytes(&[0.5]);

        let output = dequantize_fp8_to_bf16(&weight_data, &scale_data, 1, 128, Dtype::F32).unwrap();
        // 2.0 × 0.5 = 1.0 → BF16 0x3F80 → LE [0x80, 0x3F]
        for chunk in output.chunks_exact(2) {
            assert_eq!(chunk, &[0x80, 0x3F]);
        }
    }

    // -- Validation error tests ----------------------------------------------

    #[test]
    fn validation_weight_length_mismatch() {
        let result = dequantize_fp8_to_bf16(&[0u8; 10], &[0u8; 4], 2, 6, Dtype::F32);
        assert!(result.is_err());
    }

    #[test]
    fn validation_scale_not_multiple_of_4() {
        // Scale data must be a multiple of 4 bytes (f32 elements).
        let result = dequantize_fp8_to_bf16(&[0u8; 4], &[0u8; 5], 2, 2, Dtype::F32);
        assert!(result.is_err());
    }

    #[test]
    fn validation_scale_too_small() {
        // 256×256 weight needs ceil(256/128)=2 column blocks, but scale
        // only provides 1 column block (1 element for 1 scale_row).
        let weight = vec![0u8; 256 * 256];
        let scale = vec![0u8; 4]; // 1 f32 element, scale_cols = 1/2 = 0
        let result = dequantize_fp8_to_bf16(&weight, &scale, 256, 256, Dtype::F32);
        assert!(result.is_err());
    }

    #[test]
    fn validation_zero_dimensions() {
        // 0×0 triggers "zero rows" error.
        let result = dequantize_fp8_to_bf16(&[], &[], 0, 0, Dtype::F32);
        assert!(result.is_err());
    }

    // -- dequantize_per_tensor_fp8_to_bf16 -----------------------------------

    #[test]
    fn per_tensor_all_ones_scale_one() {
        // 128 elements of 1.0 in E4M3 (0x38), scale=1.0
        let weight = vec![0x38u8; 128];
        let output = dequantize_per_tensor_fp8_to_bf16(&weight, 1.0).unwrap();
        assert_eq!(output.len(), 256);
        for chunk in output.chunks_exact(2) {
            assert_eq!(chunk, &[0x80, 0x3F]); // BF16 1.0
        }
    }

    #[test]
    fn per_tensor_scale_two() {
        // 1.0 × 2.0 = 2.0
        let weight = vec![0x38u8; 64];
        let output = dequantize_per_tensor_fp8_to_bf16(&weight, 2.0).unwrap();
        for chunk in output.chunks_exact(2) {
            assert_eq!(chunk, &[0x00, 0x40]); // BF16 2.0
        }
    }

    #[test]
    fn per_tensor_non_aligned_length() {
        // 130 elements — tests remainder handling (128 + 2)
        let weight = vec![0x40u8; 130]; // 2.0 in E4M3
        let output = dequantize_per_tensor_fp8_to_bf16(&weight, 0.5).unwrap();
        assert_eq!(output.len(), 260);
        // 2.0 × 0.5 = 1.0
        for chunk in output.chunks_exact(2) {
            assert_eq!(chunk, &[0x80, 0x3F]); // BF16 1.0
        }
    }

    #[test]
    fn per_tensor_single_element() {
        let output = dequantize_per_tensor_fp8_to_bf16(&[0x38], 3.0).unwrap();
        assert_eq!(output.len(), 2);
        assert_eq!(&output[..], &[0x40, 0x40]); // BF16 3.0
    }

    #[test]
    fn per_tensor_empty() {
        let output = dequantize_per_tensor_fp8_to_bf16(&[], 1.0).unwrap();
        assert!(output.is_empty());
    }

    #[test]
    fn per_tensor_nan_preserved() {
        let output = dequantize_per_tensor_fp8_to_bf16(&[0x7F], 42.0).unwrap();
        let bf16_bits = u16::from_le_bytes([output[0], output[1]]);
        let f = f32::from_bits(u32::from(bf16_bits) << 16);
        assert!(f.is_nan());
    }

    // -- dequantize_per_channel_fp8_to_bf16 -----------------------------------

    /// Helper: build BF16 scale data from a slice of f32 values.
    fn make_bf16_scale_bytes(scales: &[f32]) -> Vec<u8> {
        // BITWISE: f32 → BF16 by taking upper 16 bits (no rounding for exact values)
        scales
            .iter()
            .flat_map(|s| ((s.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    /// Helper: build F16 scale data from a slice of f32 values.
    fn make_f16_scale_bytes(scales: &[f32]) -> Vec<u8> {
        scales
            .iter()
            .flat_map(|s| half::f16::from_f32(*s).to_le_bytes())
            .collect()
    }

    #[test]
    fn per_channel_basic_f32_scale() {
        // 2 rows × 4 cols, each row has its own F32 scale
        let rows = 2;
        let cols = 4;
        let weight_data = vec![0x38u8; rows * cols]; // all 1.0 in E4M3
        // Row 0: scale=2.0, Row 1: scale=3.0
        let scale_data = make_scale_bytes(&[2.0, 3.0]);

        let output =
            dequantize_per_channel_fp8_to_bf16(&weight_data, &scale_data, rows, cols, Dtype::F32)
                .unwrap();

        assert_eq!(output.len(), rows * cols * 2);
        // Row 0: 1.0 × 2.0 = 2.0 → BF16 [0x00, 0x40]
        for chunk in output[..cols * 2].chunks_exact(2) {
            assert_eq!(chunk, &[0x00, 0x40], "row 0: expected BF16 2.0");
        }
        // Row 1: 1.0 × 3.0 = 3.0 → BF16 [0x40, 0x40]
        for chunk in output[cols * 2..].chunks_exact(2) {
            assert_eq!(chunk, &[0x40, 0x40], "row 1: expected BF16 3.0");
        }
    }

    #[test]
    fn per_channel_bf16_scale() {
        // 2 rows × 4 cols, BF16 scale factors
        let rows = 2;
        let cols = 4;
        let weight_data = vec![0x38u8; rows * cols]; // all 1.0 in E4M3
        let scale_data = make_bf16_scale_bytes(&[2.0, 3.0]);

        let output =
            dequantize_per_channel_fp8_to_bf16(&weight_data, &scale_data, rows, cols, Dtype::BF16)
                .unwrap();

        assert_eq!(output.len(), rows * cols * 2);
        // Row 0: 1.0 × 2.0 = 2.0
        for chunk in output[..cols * 2].chunks_exact(2) {
            assert_eq!(chunk, &[0x00, 0x40], "row 0: expected BF16 2.0");
        }
        // Row 1: 1.0 × 3.0 = 3.0
        for chunk in output[cols * 2..].chunks_exact(2) {
            assert_eq!(chunk, &[0x40, 0x40], "row 1: expected BF16 3.0");
        }
    }

    #[test]
    fn per_channel_f16_scale() {
        // 2 rows × 4 cols, F16 scale factors
        let rows = 2;
        let cols = 4;
        let weight_data = vec![0x38u8; rows * cols]; // all 1.0 in E4M3
        let scale_data = make_f16_scale_bytes(&[2.0, 3.0]);

        let output =
            dequantize_per_channel_fp8_to_bf16(&weight_data, &scale_data, rows, cols, Dtype::F16)
                .unwrap();

        assert_eq!(output.len(), rows * cols * 2);
        // Row 0: 1.0 × 2.0 = 2.0
        for chunk in output[..cols * 2].chunks_exact(2) {
            assert_eq!(chunk, &[0x00, 0x40], "row 0: expected BF16 2.0");
        }
        // Row 1: 1.0 × 3.0 = 3.0
        for chunk in output[cols * 2..].chunks_exact(2) {
            assert_eq!(chunk, &[0x40, 0x40], "row 1: expected BF16 3.0");
        }
    }

    #[test]
    fn per_channel_single_row() {
        let weight_data = vec![0x40u8; 128]; // 128 elements of 2.0 in E4M3
        let scale_data = make_scale_bytes(&[0.5]); // F32 scale = 0.5

        let output =
            dequantize_per_channel_fp8_to_bf16(&weight_data, &scale_data, 1, 128, Dtype::F32)
                .unwrap();

        // 2.0 × 0.5 = 1.0 → BF16 [0x80, 0x3F]
        for chunk in output.chunks_exact(2) {
            assert_eq!(chunk, &[0x80, 0x3F]);
        }
    }

    #[test]
    fn per_channel_nan_preserved() {
        // NaN element (0x7F) should stay NaN regardless of scale
        let weight_data = vec![0x7F]; // NaN in E4M3
        let scale_data = make_scale_bytes(&[42.0]);

        let output =
            dequantize_per_channel_fp8_to_bf16(&weight_data, &scale_data, 1, 1, Dtype::F32)
                .unwrap();

        let bf16_bits = u16::from_le_bytes([output[0], output[1]]);
        let f = f32::from_bits(u32::from(bf16_bits) << 16);
        assert!(f.is_nan());
    }

    #[test]
    fn per_channel_validation_weight_mismatch() {
        // weight length doesn't match rows × cols
        let result = dequantize_per_channel_fp8_to_bf16(&[0u8; 10], &[0u8; 8], 2, 6, Dtype::F32);
        assert!(result.is_err());
    }

    #[test]
    fn per_channel_validation_scale_mismatch() {
        // scale length doesn't match rows × bytes_per_scale
        let result = dequantize_per_channel_fp8_to_bf16(&[0u8; 8], &[0u8; 2], 2, 4, Dtype::F32);
        assert!(result.is_err());
    }

    // -- Fine-grained with non-F32 scale dtypes -------------------------------

    #[test]
    fn fine_grained_bf16_scale() {
        // 128×128 block with BF16 scale = 2.0
        let rows = 128;
        let cols = 128;
        let weight_data = vec![0x38u8; rows * cols]; // all 1.0 in E4M3
        let scale_data = make_bf16_scale_bytes(&[2.0]);

        let output =
            dequantize_fp8_to_bf16(&weight_data, &scale_data, rows, cols, Dtype::BF16).unwrap();

        assert_eq!(output.len(), rows * cols * 2);
        for chunk in output.chunks_exact(2) {
            assert_eq!(chunk, &[0x00, 0x40], "expected BF16 2.0");
        }
    }

    #[test]
    fn fine_grained_f32_scale() {
        // 128×128 block with F32 scale — already tested by single_block_128x128,
        // but this explicitly names the gap: F32 scale path for fine-grained.
        let rows = 128;
        let cols = 128;
        let weight_data = vec![0x38u8; rows * cols]; // all 1.0 in E4M3
        let scale_data = make_scale_bytes(&[3.0]);

        let output =
            dequantize_fp8_to_bf16(&weight_data, &scale_data, rows, cols, Dtype::F32).unwrap();

        assert_eq!(output.len(), rows * cols * 2);
        // 1.0 × 3.0 = 3.0 → BF16 [0x40, 0x40]
        for chunk in output.chunks_exact(2) {
            assert_eq!(chunk, &[0x40, 0x40], "expected BF16 3.0");
        }
    }

    #[test]
    fn fine_grained_f16_scale() {
        // 128×128 block with F16 scale = 2.0
        let rows = 128;
        let cols = 128;
        let weight_data = vec![0x38u8; rows * cols]; // all 1.0 in E4M3
        let scale_data = make_f16_scale_bytes(&[2.0]);

        let output =
            dequantize_fp8_to_bf16(&weight_data, &scale_data, rows, cols, Dtype::F16).unwrap();

        assert_eq!(output.len(), rows * cols * 2);
        for chunk in output.chunks_exact(2) {
            assert_eq!(chunk, &[0x00, 0x40], "expected BF16 2.0");
        }
    }

    #[test]
    fn fine_grained_f32_multi_block() {
        // 256×256 with F32 scales — 2×2 block grid, verifying F32 scale path
        // across multiple blocks
        let rows = 256;
        let cols = 256;
        let weight_data = vec![0x38u8; rows * cols]; // all 1.0 in E4M3
        let scales = [1.0_f32, 4.0, 2.0, 8.0];
        let scale_data = make_scale_bytes(&scales);

        let output =
            dequantize_fp8_to_bf16(&weight_data, &scale_data, rows, cols, Dtype::F32).unwrap();

        // Block (0,0): scale=1.0 → BF16(1.0)=0x3F80
        assert_eq!(&output[0..2], &[0x80, 0x3F]);
        // Block (0,1): scale=4.0 → BF16(4.0)=0x4080
        assert_eq!(&output[256..258], &[0x80, 0x40]);
        // Block (1,0): scale=2.0 → BF16(2.0)=0x4000
        let offset_10 = 128 * 256 * 2;
        assert_eq!(&output[offset_10..offset_10 + 2], &[0x00, 0x40]);
        // Block (1,1): scale=8.0 → BF16(8.0)=0x4100
        let offset_11 = offset_10 + 128 * 2;
        assert_eq!(&output[offset_11..offset_11 + 2], &[0x00, 0x41]);
    }
}
