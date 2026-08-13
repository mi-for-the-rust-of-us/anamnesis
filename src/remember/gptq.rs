// SPDX-License-Identifier: MIT OR Apache-2.0

//! `GPTQ` dequantization (INT4/INT8 with group-wise scale + zero-point).
//!
//! Converts packed integer weights with per-group scale factors and zero-points
//! into output bytes of a caller-chosen width. Supports both 4-bit and 8-bit
//! quantization, with optional activation-order group indices (`g_idx`).
//!
//! # Output width
//!
//! Since v0.7.4 [`dequantize_gptq`] is generic over
//! [`OutputElement`](crate::OutputElement); `dequantize_gptq_to_bf16` remains as
//! the `Bf16Out` wrapper. The kernel is a three-pass loop fission: unpack the
//! packed `I32` row into `f32`, apply `(q - zero) × scale` into a second `f32`
//! scratch, then hand that scratch to [`OutputElement::write_scratch`]. Before
//! v0.7.4 the last two passes were fused and narrowed to `BF16` inline, which
//! is what made the family `BF16`-only.
//!
//! Reference: Frantar et al., "GPTQ: Accurate Post-Training Quantization for
//! Generative Pre-trained Transformers", ICLR 2023 (arXiv:2210.17323).

use crate::error::AnamnesisError;
use crate::parse::safetensors::Dtype;
use crate::remember::output::{Bf16Out, OutputElement};
use crate::remember::quant_utils::{read_scale_f32, read_u32_le};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extracts an unsigned integer value from a packed `u32` at the given bit
/// position.
///
/// Branchless: a single shift + mask operation. Works for both 4-bit and 8-bit.
///
/// # Arguments
///
/// * `packed` — the packed `u32` containing multiple quantized values.
/// * `pos` — the position within the packed value (0-based).
/// * `shift` — precomputed `bits * pos` (number of bits to shift right).
/// * `mask` — precomputed `(1 << bits) - 1` (bitmask for one value).
#[must_use]
fn unpack_gptq(packed: u32, shift: u32, mask: u32) -> u32 {
    // BITWISE: extract unsigned integer at bit position `shift` with width `bits`
    (packed >> shift) & mask
}

// ---------------------------------------------------------------------------
// Per-group unpacking (lazy, cache-friendly)
// ---------------------------------------------------------------------------

/// Unpacks zero-point values (with +1 offset) for a single group into `buf`.
///
/// Fills `buf[0..out_features]` with the f32 zero-points for group `g`.
/// The +1 offset follows the standard `GPTQ` convention: `qzeros` are stored
/// as `actual_zero - 1` during packing, and `+ 1` is applied during unpacking.
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if `qzeros_data` is too short.
fn unpack_zeros_for_group(
    buf: &mut [f32],
    qzeros_data: &[u8],
    g: usize,
    out_features: usize,
    bits: u8,
) -> crate::Result<()> {
    // CAST: u8 → u32, bits is 4 or 8
    #[allow(clippy::as_conversions)]
    let bits_u32 = u32::from(bits);
    // BITWISE: mask for one quantized value, e.g. 0xF for 4-bit, 0xFF for 8-bit
    let mask = (1u32 << bits_u32) - 1;
    // CAST: u8 → usize, bits is 4 or 8
    #[allow(clippy::as_conversions)]
    let pack_factor = 32 / bits as usize;
    let packed_cols =
        out_features
            .checked_div(pack_factor)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: "pack_factor is zero".into(),
            })?;

    for (j, buf_val) in buf.iter_mut().enumerate() {
        let packed_col = j / pack_factor;
        let pos = j % pack_factor;
        // CAST: usize → u32, pos is at most 7 (4-bit) or 3 (8-bit)
        #[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
        let shift = bits_u32 * (pos as u32);

        let byte_offset = (g * packed_cols + packed_col)
            .checked_mul(4)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: "qzeros byte offset overflow".into(),
            })?;
        let packed = read_u32_le(qzeros_data, byte_offset)?;
        let qz = unpack_gptq(packed, shift, mask);

        // BITWISE: +1 offset per standard GPTQ convention (stored as actual-1)
        // CAST: u32 → f32, qz+1 is at most 16 (4-bit) or 256 (8-bit), exact in f32
        #[allow(clippy::as_conversions, clippy::cast_precision_loss)]
        {
            *buf_val = (qz + 1) as f32;
        }
    }

    Ok(())
}

/// Unpacks scale factors for a single group into `buf`.
///
/// Fills `buf[0..out_features]` with the f32 scales for group `g`.
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if `scales_data` is too short or the
/// dtype is unsupported.
fn unpack_scales_for_group(
    buf: &mut [f32],
    scales_data: &[u8],
    g: usize,
    out_features: usize,
    scale_dtype: Dtype,
) -> crate::Result<()> {
    let bps = scale_dtype.byte_size();
    let row_start = g
        .checked_mul(out_features)
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "scales group row offset overflow".into(),
        })?;

    for (j, buf_val) in buf.iter_mut().enumerate() {
        let byte_offset = row_start
            .checked_add(j)
            .and_then(|idx| idx.checked_mul(bps))
            .ok_or_else(|| AnamnesisError::Parse {
                reason: "scale byte offset overflow".into(),
            })?;
        *buf_val = read_scale_f32(scales_data, byte_offset, scale_dtype)?;
    }

    Ok(())
}

/// Parse the `g_idx` tensor into a `Vec<usize>` of group indices.
///
/// Each element maps an input feature to its group index.
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if the data length does not match
/// `in_features × 4` bytes.
fn parse_g_idx(g_idx_data: &[u8], in_features: usize) -> crate::Result<Vec<usize>> {
    let expected_len = in_features
        .checked_mul(4)
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "g_idx byte length overflow".into(),
        })?;
    if g_idx_data.len() != expected_len {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "g_idx data length {} != expected {expected_len} (in_features={in_features} × 4)",
                g_idx_data.len()
            ),
        });
    }

    let mut indices = Vec::with_capacity(in_features);
    for i in 0..in_features {
        let byte_offset = i.checked_mul(4).ok_or_else(|| AnamnesisError::Parse {
            reason: "g_idx byte offset overflow".into(),
        })?;
        let val = read_u32_le(g_idx_data, byte_offset)?;
        // CAST: u32 → usize, group index fits in usize
        #[allow(clippy::as_conversions)]
        let idx = val as usize;
        indices.push(idx);
    }

    Ok(indices)
}

// ---------------------------------------------------------------------------
// Main dequantization (public API)
// ---------------------------------------------------------------------------

/// Dequantizes a `GPTQ`-quantized weight tensor to `BF16`.
///
/// The [`Bf16Out`] special case of [`dequantize_gptq`], kept so that every
/// pre-v0.7.4 caller compiles unchanged.
///
/// # Errors
///
/// See [`dequantize_gptq`].
///
/// # Memory
///
/// See [`dequantize_gptq`]; at `BF16` the output buffer is
/// `in_features × out_features × 2` bytes.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn dequantize_gptq_to_bf16(
    qweight_data: &[u8],
    scales_data: &[u8],
    qzeros_data: &[u8],
    g_idx_data: Option<&[u8]>,
    in_features: usize,
    out_features: usize,
    group_size: usize,
    bits: u8,
    scale_dtype: Dtype,
) -> crate::Result<Vec<u8>> {
    dequantize_gptq::<Bf16Out>(
        qweight_data,
        scales_data,
        qzeros_data,
        g_idx_data,
        in_features,
        out_features,
        group_size,
        bits,
        scale_dtype,
    )
}

/// Dequantizes a `GPTQ`-quantized weight tensor into `E`.
///
/// Unpacks INT4 or INT8 values from packed `I32` tensors, applies per-group
/// scale factors and zero-points, and writes `E`. Supports both
/// sequential group assignment and activation-order via `g_idx`.
///
/// The standard `GPTQ` dequantization formula is:
/// `dequant[i, j] = (qweight[i, j] - (qzeros[g, j] + 1)) × scales[g, j]`
///
/// # Arguments
///
/// * `qweight_data` — packed `I32` weight bytes, row-major `[in_features/pack_factor, out_features]`.
/// * `scales_data` — scale factor bytes, row-major `[num_groups, out_features]`.
/// * `qzeros_data` — packed `I32` zero-point bytes, row-major `[num_groups, out_features/pack_factor]`.
/// * `g_idx_data` — optional `I32` group index bytes, `[in_features]`.
/// * `in_features` — number of input features (unpacked weight rows).
/// * `out_features` — number of output features (weight columns).
/// * `group_size` — number of input features per group (typically 128).
/// * `bits` — quantization bit width (4 or 8).
/// * `scale_dtype` — dtype of the scales tensor (`F16`, `BF16`, or `F32`).
///
/// # Returns
///
/// A `Vec<u8>` of length `in_features × out_features × E::BYTES`, in
/// little-endian byte order. Shape: `[in_features, out_features]`
/// — the **GEMM-native** orientation the canonical `GPTQModel` kernel
/// produces (and the cross-validation fixtures anchor). Note this is the
/// transpose of a standard `nn.Linear.weight` (`[out, in]`);
/// `ParsedModel::remember` / `remember_to_bytes` transpose to the standard
/// orientation before serializing, mirroring `GPTQModel`'s own
/// `dequantize_model` (`.T`).
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if tensor dimensions are inconsistent.
/// Returns [`AnamnesisError::Unsupported`] if the bit width or scale dtype
/// is not supported.
///
/// # Memory
///
/// Allocates per-group scratch buffers for zero-points and scales
/// (`out_features × 4` bytes each), an unpacking scratch buffer and a values
/// scratch buffer (`out_features × 4` bytes each), plus the output buffer
/// (`in_features × out_features × E::BYTES` bytes). Group data is computed
/// lazily — only the current group's row is live at any time. The values
/// scratch is new in v0.7.4: it is what lets the narrowing move out of the
/// arithmetic loop and into [`OutputElement::write_scratch`], and at
/// `out_features × 4` bytes it is L1-resident and independent of model size.
#[allow(clippy::too_many_arguments)]
pub fn dequantize_gptq<E: OutputElement>(
    qweight_data: &[u8],
    scales_data: &[u8],
    qzeros_data: &[u8],
    g_idx_data: Option<&[u8]>,
    in_features: usize,
    out_features: usize,
    group_size: usize,
    bits: u8,
    scale_dtype: Dtype,
) -> crate::Result<Vec<u8>> {
    // --- Validate bit width ---
    if bits != 4 && bits != 8 {
        return Err(AnamnesisError::Unsupported {
            format: "GPTQ".into(),
            detail: format!("{bits}-bit quantization not supported (expected 4 or 8)"),
        });
    }

    // CAST: u8 → usize, bits is 4 or 8
    #[allow(clippy::as_conversions)]
    let pack_factor = 32 / bits as usize;

    // --- Validate dimensions ---
    if in_features == 0 || out_features == 0 || group_size == 0 {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "zero dimension: in_features={in_features}, out_features={out_features}, \
                 group_size={group_size}"
            ),
        });
    }
    if !in_features.is_multiple_of(pack_factor) {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "in_features {in_features} is not a multiple of pack_factor {pack_factor}"
            ),
        });
    }
    if !out_features.is_multiple_of(pack_factor) {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "out_features {out_features} is not a multiple of pack_factor {pack_factor}"
            ),
        });
    }
    if !in_features.is_multiple_of(group_size) {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "in_features {in_features} is not a multiple of group_size {group_size}"
            ),
        });
    }

    let packed_rows = in_features / pack_factor;
    let packed_cols = out_features / pack_factor;
    let num_groups = in_features / group_size;

    // --- Validate tensor sizes ---
    let expected_qw_len = packed_rows
        .checked_mul(out_features)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "qweight byte length overflow".into(),
        })?;
    if qweight_data.len() != expected_qw_len {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "qweight data length {} != expected {expected_qw_len}",
                qweight_data.len()
            ),
        });
    }

    let expected_scales_len = num_groups
        .checked_mul(out_features)
        .and_then(|n| n.checked_mul(scale_dtype.byte_size()))
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "scales byte length overflow".into(),
        })?;
    if scales_data.len() != expected_scales_len {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "scales data length {} != expected {expected_scales_len}",
                scales_data.len()
            ),
        });
    }

    let expected_qzeros_len = num_groups
        .checked_mul(packed_cols)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "qzeros byte length overflow".into(),
        })?;
    if qzeros_data.len() != expected_qzeros_len {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "qzeros data length {} != expected {expected_qzeros_len}",
                qzeros_data.len()
            ),
        });
    }

    // --- Parse g_idx if present ---
    let g_idx = g_idx_data
        .map(|data| parse_g_idx(data, in_features))
        .transpose()?;

    // --- Pre-validate g_idx entries (fail-fast before the hot loop) ---
    if let Some(ref idx) = g_idx {
        for (i, &g) in idx.iter().enumerate() {
            if g >= num_groups {
                return Err(AnamnesisError::Parse {
                    reason: format!("g_idx[{i}] = {g} >= num_groups {num_groups}"),
                });
            }
        }
    }

    // --- Allocate output ---
    let out_byte_len = in_features
        .checked_mul(out_features)
        .and_then(|n| n.checked_mul(E::BYTES))
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "output size overflow".into(),
        })?;
    let mut output = vec![0u8; out_byte_len];

    // --- Precompute constants ---
    // CAST: u8 → u32, bits is 4 or 8
    #[allow(clippy::as_conversions)]
    let bits_u32 = u32::from(bits);
    // BITWISE: mask for one quantized value, e.g. 0xF for 4-bit, 0xFF for 8-bit
    let mask = (1u32 << bits_u32) - 1;

    // Pre-allocate scratch buffers for one row each (reused across iterations).
    // Lazy per-group: only `out_features` f32 values are live at a time,
    // instead of the full `num_groups × out_features` grid.
    let mut unpacked_buf = vec![0.0_f32; out_features];
    let mut zeros_buf = vec![0.0_f32; out_features];
    let mut scales_buf = vec![0.0_f32; out_features];
    let mut cached_group: Option<usize> = None;

    // --- Hot loop: row-by-row dequantization ---
    // Two-level bounds checking per CONVENTIONS.md: validate slices ONCE
    // before the inner loop, then iterate branch-free inside.
    for i in 0..in_features {
        // Determine group for this input feature.
        // INDEX: g < num_groups guaranteed by pre-validation above (g_idx path)
        // or by i < in_features ∧ in_features = num_groups × group_size (sequential path).
        let g = if let Some(ref idx) = g_idx {
            // INDEX: i < in_features, g_idx.len() == in_features (validated in parse_g_idx)
            idx.get(i).copied().ok_or_else(|| AnamnesisError::Parse {
                reason: format!("g_idx index {i} out of bounds"),
            })?
        } else {
            i / group_size
        };

        let packed_row = i / pack_factor;
        let pos = i % pack_factor;
        // CAST: usize → u32, pos is at most 7 (4-bit) or 3 (8-bit)
        #[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
        let shift = bits_u32 * (pos as u32);

        // --- Pre-validate slices ONCE (two-level bounds checking) ---

        // qweight row: out_features contiguous I32 values for this packed_row.
        let qw_row_start = packed_row
            .checked_mul(out_features)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| AnamnesisError::Parse {
                reason: "qweight row byte offset overflow".into(),
            })?;
        let qw_row_end = qw_row_start + out_features * 4;
        let qw_row =
            qweight_data
                .get(qw_row_start..qw_row_end)
                .ok_or_else(|| AnamnesisError::Parse {
                    reason: format!("qweight row {packed_row} out of bounds"),
                })?;

        // Lazy per-group unpacking: refill zeros/scales only when the group
        // changes. For sequential access (no g_idx), this fires once per
        // group_size rows. The scratch buffers are out_features-sized and
        // L1-resident.
        if cached_group != Some(g) {
            unpack_zeros_for_group(&mut zeros_buf, qzeros_data, g, out_features, bits)?;
            unpack_scales_for_group(&mut scales_buf, scales_data, g, out_features, scale_dtype)?;
            cached_group = Some(g);
        }
        let zeros_row = &zeros_buf[..];
        let scales_row = &scales_buf[..];

        // Output row: contiguous `E` bytes.
        let out_row_start = i
            .checked_mul(out_features)
            .and_then(|n| n.checked_mul(E::BYTES))
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!("output row {i} offset overflow"),
            })?;
        let out_row_end = out_features
            .checked_mul(E::BYTES)
            .and_then(|row_bytes| out_row_start.checked_add(row_bytes))
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!("output row {i} end overflow"),
            })?;
        let out_row =
            output
                .get_mut(out_row_start..out_row_end)
                .ok_or_else(|| AnamnesisError::Parse {
                    reason: format!("output row {i} out of bounds"),
                })?;

        // --- Unpack qweight row into contiguous f32 values ---
        // Separates byte→u32 extraction (hard to vectorize) from the
        // arithmetic (easy to vectorize). The compiler auto-vectorizes
        // the second loop: pure f32 sub+mul+convert pipeline.
        // INDEX: unpacked_buf.len() == out_features, allocated before the outer loop
        let unpacked_row =
            unpacked_buf
                .get_mut(..out_features)
                .ok_or_else(|| AnamnesisError::Parse {
                    reason: "unpacked buffer too short".into(),
                })?;
        #[allow(clippy::indexing_slicing)]
        for (j, qw_chunk) in qw_row.chunks_exact(4).enumerate() {
            // INDEX: chunks_exact(4) guarantees exactly 4 bytes per chunk;
            // j < out_features guaranteed by qw_row length validation above
            let packed = u32::from_le_bytes([qw_chunk[0], qw_chunk[1], qw_chunk[2], qw_chunk[3]]);
            // BITWISE: extract unsigned quantized value at bit position `shift`
            // CAST: u32 → f32, qw is at most 15 (4-bit) or 255 (8-bit), exact in f32
            #[allow(clippy::as_conversions, clippy::cast_precision_loss)]
            let qw = unpack_gptq(packed, shift, mask) as f32;
            unpacked_row[j] = qw;
        }

        // --- Pass 2: pure f32 arithmetic, BRANCH-FREE, IN PLACE ---
        // Contiguous f32 reads (unpacked, zeros, scales), no byte manipulation
        // and no narrowing — just sub + mul, the shape CONVENTIONS.md's
        // loop-fission rule asks for.
        //
        // **In place over `unpacked_row`, not into a second buffer.** The first
        // v0.7.4 draft wrote into a separate `out_features`-sized scratch, which
        // doubled the row working set to ~88 KB at `out_features = 11008` and
        // pushed it out of L1. CONVENTIONS.md's "separate input and output
        // slices" rule is about aliasing *between distinct slices* defeating the
        // vectorizer; element `i` here reads and writes only element `i` of the
        // same slice, which raises no aliasing question at all.
        // VECTORIZED: pending cargo-show-asm verification
        for ((value, &zero), &scale) in unpacked_row
            .iter_mut()
            .zip(zeros_row.iter())
            .zip(scales_row.iter())
        {
            *value = (*value - zero) * scale;
        }

        // --- Pass 3: narrow to the caller's output width ---
        E::write_scratch(unpacked_row, out_row);
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
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::float_cmp
)]
mod tests {
    use super::*;
    // The kernel itself no longer narrows (v0.7.4 moved that to
    // `OutputElement::write_scratch`), but the `BF16` expectations these tests
    // assert still have to be built the same way the writer builds them.
    use crate::remember::fp8::f32_bits_to_bf16_bits;

    // -- unpack_gptq ---------------------------------------------------------

    #[test]
    fn unpack_4bit_all_positions() {
        // Pack 8 nibbles into one u32: values 0,1,2,...,7
        // packed = 0x76543210
        let packed: u32 = 0x7654_3210;
        let mask = 0xF;
        for pos in 0..8u32 {
            let shift = 4 * pos;
            assert_eq!(unpack_gptq(packed, shift, mask), pos);
        }
    }

    #[test]
    fn unpack_4bit_max_value() {
        // All nibbles set to 15 (0xF)
        let packed: u32 = 0xFFFF_FFFF;
        let mask = 0xF;
        for pos in 0..8u32 {
            let shift = 4 * pos;
            assert_eq!(unpack_gptq(packed, shift, mask), 15);
        }
    }

    #[test]
    fn unpack_8bit_all_positions() {
        // Pack 4 bytes into one u32: values 0x10, 0x20, 0x30, 0x40
        let packed: u32 = 0x4030_2010;
        let mask = 0xFF;
        assert_eq!(unpack_gptq(packed, 0, mask), 0x10);
        assert_eq!(unpack_gptq(packed, 8, mask), 0x20);
        assert_eq!(unpack_gptq(packed, 16, mask), 0x30);
        assert_eq!(unpack_gptq(packed, 24, mask), 0x40);
    }

    #[test]
    fn unpack_8bit_max_value() {
        let packed: u32 = 0xFFFF_FFFF;
        let mask = 0xFF;
        for pos in 0..4u32 {
            let shift = 8 * pos;
            assert_eq!(unpack_gptq(packed, shift, mask), 255);
        }
    }

    // -- Small dequantization with known values ------------------------------

    /// Build a minimal 4-bit GPTQ test case.
    ///
    /// 4 input features, 8 output features, `group_size`=4, `bits`=4.
    /// All qweight nibbles = 5, all qzeros nibbles = 3 (stored, +1 = 4),
    /// all scales = 2.0 (F16).
    ///
    /// Expected: (5 - 4) × 2.0 = 2.0 for every element.
    #[test]
    fn dequant_4bit_uniform() {
        let in_features = 8;
        let out_features = 8;
        let group_size = 8;
        let bits: u8 = 4;

        // qweight: [1, 8] (1 packed row, 8 output features)
        // Each I32 has 8 nibbles, but we only use position 0..7 for each column.
        // We need pack_factor rows of input packed into 1 packed_row.
        // packed_rows = 8/8 = 1
        // For the single packed_row, each column j has one I32 whose 8 nibbles
        // are the 8 input features at output j. All nibbles = 5.
        // 0x55555555 = all nibbles are 5
        let qweight_i32 = 0x5555_5555u32;
        let mut qweight_data = Vec::new();
        for _j in 0..out_features {
            qweight_data.extend_from_slice(&qweight_i32.to_le_bytes());
        }

        // scales: [1, 8], all 2.0 in F16
        let scale_f16 = half::f16::from_f32(2.0).to_le_bytes();
        let mut scales_data = Vec::new();
        for _ in 0..out_features {
            scales_data.extend_from_slice(&scale_f16);
        }

        // qzeros: [1, 1] (1 group, 8/8=1 packed col)
        // All nibbles = 3 (stored zero-point; actual = 3 + 1 = 4)
        let qzeros_i32 = 0x3333_3333u32;
        let qzeros_data = qzeros_i32.to_le_bytes().to_vec();

        let output = dequantize_gptq_to_bf16(
            &qweight_data,
            &scales_data,
            &qzeros_data,
            None,
            in_features,
            out_features,
            group_size,
            bits,
            Dtype::F16,
        )
        .unwrap();

        assert_eq!(output.len(), in_features * out_features * 2);

        // Expected: (5 - 4) × 2.0 = 2.0 → BF16 0x4000 → LE [0x00, 0x40]
        for chunk in output.chunks_exact(2) {
            assert_eq!(chunk, &[0x00, 0x40], "expected BF16 2.0");
        }
    }

    #[test]
    fn dequant_4bit_with_g_idx() {
        let in_features = 8;
        let out_features = 8;
        let group_size = 4;
        let bits: u8 = 4;

        // 2 groups: group 0 and group 1
        // g_idx maps input features to groups: [1,1,1,1,0,0,0,0]
        // (reversed from sequential, to test act-order)
        let g_idx_values: Vec<u32> = vec![1, 1, 1, 1, 0, 0, 0, 0];
        let g_idx_data: Vec<u8> = g_idx_values.iter().flat_map(|v| v.to_le_bytes()).collect();

        // qweight: all nibbles = 10
        let qweight_i32 = 0xAAAA_AAAAu32; // 0xA = 10 in each nibble
        let mut qweight_data = Vec::new();
        for _j in 0..out_features {
            qweight_data.extend_from_slice(&qweight_i32.to_le_bytes());
        }

        // scales: [2, 8]
        // Group 0 scales = 1.0, Group 1 scales = 3.0
        let scale_1 = half::f16::from_f32(1.0).to_le_bytes();
        let scale_3 = half::f16::from_f32(3.0).to_le_bytes();
        let mut scales_data = Vec::new();
        for _ in 0..out_features {
            scales_data.extend_from_slice(&scale_1); // group 0
        }
        for _ in 0..out_features {
            scales_data.extend_from_slice(&scale_3); // group 1
        }

        // qzeros: [2, 1] (2 groups, 8/8=1 packed col)
        // Group 0: stored zero = 7 (actual = 8)
        // Group 1: stored zero = 4 (actual = 5)
        let qz_group0 = 0x7777_7777u32;
        let qz_group1 = 0x4444_4444u32;
        let mut qzeros_data = Vec::new();
        qzeros_data.extend_from_slice(&qz_group0.to_le_bytes());
        qzeros_data.extend_from_slice(&qz_group1.to_le_bytes());

        let output = dequantize_gptq_to_bf16(
            &qweight_data,
            &scales_data,
            &qzeros_data,
            Some(&g_idx_data),
            in_features,
            out_features,
            group_size,
            bits,
            Dtype::F16,
        )
        .unwrap();

        // Input features 0-3 → g_idx = 1 → scale=3.0, zero=5
        // (10 - 5) × 3.0 = 15.0 → BF16 0x4170 → LE [0x70, 0x41]
        let bf16_15 = f32_bits_to_bf16_bits(15.0_f32.to_bits());
        for i in 0..4 {
            for j in 0..out_features {
                let offset = (i * out_features + j) * 2;
                let actual = u16::from_le_bytes([output[offset], output[offset + 1]]);
                assert_eq!(actual, bf16_15, "element [{i},{j}]: expected BF16 15.0");
            }
        }

        // Input features 4-7 → g_idx = 0 → scale=1.0, zero=8
        // (10 - 8) × 1.0 = 2.0 → BF16 0x4000
        let bf16_2 = f32_bits_to_bf16_bits(2.0_f32.to_bits());
        for i in 4..8 {
            for j in 0..out_features {
                let offset = (i * out_features + j) * 2;
                let actual = u16::from_le_bytes([output[offset], output[offset + 1]]);
                assert_eq!(actual, bf16_2, "element [{i},{j}]: expected BF16 2.0");
            }
        }
    }

    #[test]
    fn dequant_8bit_uniform() {
        let in_features = 4;
        let out_features = 4;
        let group_size = 4;
        let bits: u8 = 8;

        // qweight: [1, 4] (4/4=1 packed row, 4 output features)
        // Each I32 has 4 bytes. All bytes = 100.
        let qweight_i32 = 0x6464_6464u32; // 0x64 = 100
        let mut qweight_data = Vec::new();
        for _j in 0..out_features {
            qweight_data.extend_from_slice(&qweight_i32.to_le_bytes());
        }

        // scales: [1, 4], all 0.5 in F16
        let scale_half = half::f16::from_f32(0.5).to_le_bytes();
        let mut scales_data = Vec::new();
        for _ in 0..out_features {
            scales_data.extend_from_slice(&scale_half);
        }

        // qzeros: [1, 1] (1 group, 4/4=1 packed col)
        // All bytes = 49 (stored zero; actual = 50)
        let qzeros_i32 = 0x3131_3131u32; // 0x31 = 49
        let qzeros_data = qzeros_i32.to_le_bytes().to_vec();

        let output = dequantize_gptq_to_bf16(
            &qweight_data,
            &scales_data,
            &qzeros_data,
            None,
            in_features,
            out_features,
            group_size,
            bits,
            Dtype::F16,
        )
        .unwrap();

        // Expected: (100 - 50) × 0.5 = 25.0 → BF16 0x41C8 → LE [0xC8, 0x41]
        let bf16_25 = f32_bits_to_bf16_bits(25.0_f32.to_bits());
        for chunk in output.chunks_exact(2) {
            let actual = u16::from_le_bytes([chunk[0], chunk[1]]);
            assert_eq!(actual, bf16_25, "expected BF16 25.0");
        }
    }

    // -- Validation errors ---------------------------------------------------

    #[test]
    fn validation_unsupported_bits() {
        let result = dequantize_gptq_to_bf16(&[], &[], &[], None, 8, 8, 8, 3, Dtype::F16);
        assert!(result.is_err());
    }

    #[test]
    fn validation_zero_dimensions() {
        let result = dequantize_gptq_to_bf16(&[], &[], &[], None, 0, 8, 8, 4, Dtype::F16);
        assert!(result.is_err());
    }

    #[test]
    fn validation_in_features_not_multiple_of_pack_factor() {
        // in_features=5 is not a multiple of pack_factor=8 for 4-bit
        let result = dequantize_gptq_to_bf16(&[], &[], &[], None, 5, 8, 5, 4, Dtype::F16);
        assert!(result.is_err());
    }

    #[test]
    fn validation_g_idx_out_of_range() {
        let in_features = 8;
        let out_features = 8;
        let group_size = 4;
        let bits: u8 = 4;

        // Build valid qweight, scales, qzeros for 2 groups
        let qweight_i32 = 0x5555_5555u32;
        let mut qweight_data = Vec::new();
        for _ in 0..out_features {
            qweight_data.extend_from_slice(&qweight_i32.to_le_bytes());
        }
        let scale_f16 = half::f16::from_f32(1.0).to_le_bytes();
        let mut scales_data = Vec::new();
        for _ in 0..2 * out_features {
            scales_data.extend_from_slice(&scale_f16);
        }
        let qz = 0x3333_3333u32;
        let mut qzeros_data = Vec::new();
        qzeros_data.extend_from_slice(&qz.to_le_bytes());
        qzeros_data.extend_from_slice(&qz.to_le_bytes());

        // g_idx with entry = 99 (>= num_groups=2) at position 5
        let g_idx_values: Vec<u32> = vec![0, 0, 0, 0, 1, 99, 1, 1];
        let g_idx_data: Vec<u8> = g_idx_values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let result = dequantize_gptq_to_bf16(
            &qweight_data,
            &scales_data,
            &qzeros_data,
            Some(&g_idx_data),
            in_features,
            out_features,
            group_size,
            bits,
            Dtype::F16,
        );

        let err = result.unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("g_idx[5]") && msg.contains("99"),
            "expected fail-fast g_idx error, got: {msg}"
        );
    }

    #[test]
    fn validation_qweight_length_mismatch() {
        // in_features=8, out_features=8, bits=4 → qweight should be 1×8×4=32 bytes
        let result = dequantize_gptq_to_bf16(
            &[0u8; 16], // wrong: should be 32
            &[0u8; 16], // scales: 1 group × 8 × 2 = 16
            &[0u8; 4],  // qzeros: 1 group × 1 × 4 = 4
            None,
            8,
            8,
            8,
            4,
            Dtype::F16,
        );
        assert!(result.is_err());
    }
}
