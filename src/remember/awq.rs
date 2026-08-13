// SPDX-License-Identifier: MIT OR Apache-2.0

//! `AWQ` dequantization (activation-aware, INT4 with per-group scales).
//!
//! Converts packed integer weights with per-group scale factors and zero-points
//! into output bytes of a caller-chosen width. `AWQ` packs along `out_features`
//! (columns), unlike `GPTQ` which packs along `in_features` (rows). No
//! `.g_idx` — groups are always sequential.
//!
//! # Output width
//!
//! Since v0.7.4 [`dequantize_awq`] is generic over
//! [`OutputElement`](crate::OutputElement); `dequantize_awq_to_bf16` remains as
//! the `Bf16Out` wrapper. The kernel is a three-pass loop fission matching
//! `GPTQ`'s: unpack the interleaved nibbles into `f32`, apply
//! `(q - zero) × scale` into a second `f32` scratch, then hand that scratch to
//! [`OutputElement::write_scratch`].
//!
//! # Packing order (the `AutoAWQ` GEMM interleave)
//!
//! Unlike `GPTQ`, the 8 nibbles inside each packed `I32` are NOT in
//! sequential LSB-first order. `AutoAWQ` packs logical column offset
//! `AWQ_ORDER[i] = [0, 2, 4, 6, 1, 3, 5, 7][i]` at bit position `4 × i`,
//! for BOTH `qweight` and `qzeros`; its dequant applies the inverse
//! permutation (`AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]`) after
//! shift-unpacking. Versions ≤ v0.6.3 unpacked sequentially, producing
//! column-permuted output that validated green against an equally
//! sequential hand-rolled fixture; see docs/dogfooding-feedbacks/
//! bnb-nibble-order-and-circular-fixture-validation.md.
//!
//! `AutoAWQ`'s GEMM format is 4-bit only (`raise NotImplementedError`
//! otherwise), so this module rejects every other bit width — there is no
//! canonical interleave definition to anchor an 8-bit decode against.
//!
//! Reference: Lin et al., "AWQ: Activation-aware Weight Quantization for LLM
//! Compression and Acceleration", `MLSys` 2024 (arXiv:2306.00978);
//! `AutoAWQ` `awq/utils/packing_utils.py` (`unpack_awq`, `reverse_awq_order`).

use crate::error::AnamnesisError;
use crate::parse::safetensors::Dtype;
use crate::remember::output::{Bf16Out, OutputElement};
use crate::remember::quant_utils::{read_scale_f32, read_u32_le};

/// `AWQ` 4-bit pack factor: 8 nibbles per packed `I32`.
const AWQ_PACK_FACTOR: usize = 8;

/// `AutoAWQ` packing order: the nibble at bit position `4 × i` holds the
/// logical column offset `AWQ_ORDER[i]` within its group of 8 columns.
/// Mirrors `AWQ_ORDER` in `awq/utils/packing_utils.py`.
const AWQ_ORDER: [usize; 8] = [0, 2, 4, 6, 1, 3, 5, 7];

/// Inverse of [`AWQ_ORDER`]: logical column offset `j` is stored at bit
/// position `4 × AWQ_REVERSE_ORDER[j]`. Mirrors `AWQ_REVERSE_ORDER` in
/// `awq/utils/packing_utils.py`.
const AWQ_REVERSE_ORDER: [usize; 8] = [0, 4, 1, 5, 2, 6, 3, 7];

// ---------------------------------------------------------------------------
// Per-group unpacking (lazy, cache-friendly)
// ---------------------------------------------------------------------------

/// Unpacks zero-point values (NO +1 offset) for a single group into `buf`.
///
/// Fills `buf[0..out_features]` with the f32 zero-points for group `g`.
/// Unlike `GPTQ`, `AWQ` does NOT add +1 to zero-points. The stored values
/// are used directly: `dequant = (qw - qz) × scale`. The nibble for
/// logical column offset `j % 8` sits at bit position
/// `4 × AWQ_REVERSE_ORDER[j % 8]` (the `AutoAWQ` interleave — applied to
/// `qzeros` exactly as to `qweight`).
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if `qzeros_data` is too short.
fn unpack_zeros_for_group(
    buf: &mut [f32],
    qzeros_data: &[u8],
    g: usize,
    out_features: usize,
) -> crate::Result<()> {
    // BITWISE: mask for one 4-bit quantized value
    let mask = 0xFu32;
    let packed_cols = out_features / AWQ_PACK_FACTOR;

    for (j, buf_val) in buf.iter_mut().enumerate() {
        let packed_col = j / AWQ_PACK_FACTOR;
        // INDEX: j % AWQ_PACK_FACTOR < 8, AWQ_REVERSE_ORDER has 8 entries
        #[allow(clippy::indexing_slicing)]
        let pos = AWQ_REVERSE_ORDER[j % AWQ_PACK_FACTOR];
        // CAST: usize → u32, pos is at most 7
        #[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
        let shift = 4 * (pos as u32);

        let byte_offset = (g * packed_cols + packed_col)
            .checked_mul(4)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: "qzeros byte offset overflow".into(),
            })?;
        let packed = read_u32_le(qzeros_data, byte_offset)?;
        // BITWISE: extract unsigned zero-point — NO +1 offset (AWQ convention)
        let qz = (packed >> shift) & mask;
        // CAST: u32 → f32, qz is at most 15, exact in f32
        #[allow(clippy::as_conversions, clippy::cast_precision_loss)]
        {
            *buf_val = qz as f32;
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

// ---------------------------------------------------------------------------
// Main dequantization (public API)
// ---------------------------------------------------------------------------

/// Dequantizes an `AWQ`-quantized weight tensor to `BF16`.
///
/// The [`Bf16Out`] special case of [`dequantize_awq`], kept so that every
/// pre-v0.7.4 caller compiles unchanged.
///
/// # Errors
///
/// See [`dequantize_awq`].
///
/// # Memory
///
/// See [`dequantize_awq`]; at `BF16` the output buffer is
/// `in_features × out_features × 2` bytes.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn dequantize_awq_to_bf16(
    qweight_data: &[u8],
    scales_data: &[u8],
    qzeros_data: &[u8],
    in_features: usize,
    out_features: usize,
    group_size: usize,
    bits: u8,
    scale_dtype: Dtype,
) -> crate::Result<Vec<u8>> {
    dequantize_awq::<Bf16Out>(
        qweight_data,
        scales_data,
        qzeros_data,
        in_features,
        out_features,
        group_size,
        bits,
        scale_dtype,
    )
}

/// Dequantizes an `AWQ`-quantized weight tensor into `E`.
///
/// `AWQ` packs along `out_features` (columns): `qweight` shape is
/// `[in_features, out_features / pack_factor]`. The dequantization formula is:
/// `dequant[i, j] = (qweight[i, j] - qzeros[g, j]) × scales[g, j]`
///
/// Unlike `GPTQ`, there is no +1 offset on zero-points and no `g_idx` —
/// groups are always sequential (`g = i / group_size`). Also unlike
/// `GPTQ`, the 8 nibbles inside each packed `I32` follow the `AutoAWQ`
/// GEMM interleave (`AWQ_ORDER` / `AWQ_REVERSE_ORDER`), not
/// sequential LSB-first order — for both `qweight` and `qzeros`.
///
/// # Arguments
///
/// * `qweight_data` — packed `I32` weight bytes, row-major `[in_features, out_features/pack_factor]`.
/// * `scales_data` — scale factor bytes, row-major `[num_groups, out_features]`.
/// * `qzeros_data` — packed `I32` zero-point bytes, row-major `[num_groups, out_features/pack_factor]`.
/// * `in_features` — number of input features (weight rows).
/// * `out_features` — number of output features (unpacked weight columns).
/// * `group_size` — number of input features per group (typically 128).
/// * `bits` — quantization bit width. Must be 4: `AutoAWQ`'s GEMM format
///   is 4-bit only, so no canonical packing interleave exists for any
///   other width.
/// * `scale_dtype` — dtype of the scales tensor (`F16`, `BF16`, or `F32`).
///
/// # Returns
///
/// A `Vec<u8>` of length `in_features × out_features × E::BYTES`, in
/// little-endian byte order. Shape: `[in_features, out_features]`
/// — the **GEMM-native** orientation the canonical `AutoAWQ` kernel
/// produces (and the cross-validation fixtures anchor). Note this is the
/// transpose of a standard `nn.Linear.weight` (`[out, in]`);
/// `ParsedModel::remember` / `remember_to_bytes` transpose to the standard
/// orientation before serializing.
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
pub fn dequantize_awq<E: OutputElement>(
    qweight_data: &[u8],
    scales_data: &[u8],
    qzeros_data: &[u8],
    in_features: usize,
    out_features: usize,
    group_size: usize,
    bits: u8,
    scale_dtype: Dtype,
) -> crate::Result<Vec<u8>> {
    // --- Validate bit width ---
    // AutoAWQ's GEMM format is 4-bit only (its packer raises
    // NotImplementedError for every other width), so there is no canonical
    // nibble interleave to anchor a non-4-bit decode against. Rejecting is
    // safer than emitting plausibly-permuted weights.
    if bits != 4 {
        return Err(AnamnesisError::Unsupported {
            format: "AWQ".into(),
            detail: format!("{bits}-bit quantization not supported (AutoAWQ GEMM is 4-bit only)"),
        });
    }

    let pack_factor = AWQ_PACK_FACTOR;

    // --- Validate dimensions ---
    if in_features == 0 || out_features == 0 || group_size == 0 {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "zero dimension: in_features={in_features}, out_features={out_features}, \
                 group_size={group_size}"
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

    let packed_cols = out_features / pack_factor;
    let num_groups = in_features / group_size;

    // --- Validate tensor sizes ---
    let expected_qw_len = in_features
        .checked_mul(packed_cols)
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

    // --- Allocate output + scratch buffers ---
    let out_byte_len = in_features
        .checked_mul(out_features)
        .and_then(|n| n.checked_mul(E::BYTES))
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "output size overflow".into(),
        })?;
    let mut output = vec![0u8; out_byte_len];

    // Pre-allocate scratch buffers for one row each (reused across iterations).
    // Lazy per-group: only `out_features` f32 values are live at a time,
    // instead of the full `num_groups × out_features` grid.
    let mut unpacked_buf = vec![0.0_f32; out_features];
    let mut values_buf = vec![0.0_f32; out_features];
    let mut zeros_buf = vec![0.0_f32; out_features];
    let mut scales_buf = vec![0.0_f32; out_features];
    let mut cached_group: Option<usize> = None;

    // --- Precompute constants ---
    // BITWISE: mask for one 4-bit quantized value
    let mask = 0xFu32;

    // --- Hot loop: row-by-row dequantization ---
    // Two-level bounds checking per CONVENTIONS.md: validate slices ONCE
    // before the inner loop, then iterate branch-free inside.
    // Loop fission per CONVENTIONS.md: separate byte unpacking from f32 arithmetic.
    for i in 0..in_features {
        let g = i / group_size;

        // --- Pre-validate slices ONCE ---

        // qweight row: packed_cols contiguous I32 values for this row.
        // AWQ packs along out_features: qweight[i, 0..packed_cols]
        let qw_row_start = i
            .checked_mul(packed_cols)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| AnamnesisError::Parse {
                reason: "qweight row byte offset overflow".into(),
            })?;
        let qw_row_end = qw_row_start + packed_cols * 4;
        let qw_row =
            qweight_data
                .get(qw_row_start..qw_row_end)
                .ok_or_else(|| AnamnesisError::Parse {
                    reason: format!("qweight row {i} out of bounds"),
                })?;

        // Lazy per-group unpacking: refill zeros/scales only when the group
        // changes. For sequential access, this fires once per group_size rows.
        // The scratch buffers are out_features-sized and L1-resident.
        if cached_group != Some(g) {
            unpack_zeros_for_group(&mut zeros_buf, qzeros_data, g, out_features)?;
            unpack_scales_for_group(&mut scales_buf, scales_data, g, out_features, scale_dtype)?;
            cached_group = Some(g);
        }
        let zeros_row = &zeros_buf[..];
        let scales_row = &scales_buf[..];

        // Output row.
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

        // Scratch buffer.
        let unpacked_row =
            unpacked_buf
                .get_mut(..out_features)
                .ok_or_else(|| AnamnesisError::Parse {
                    reason: "unpacked buffer too short".into(),
                })?;

        // --- Pass 1: unpack qweight row → f32 scratch buffer ---
        // AWQ packs along out_features: each I32 contains pack_factor output
        // features in the AutoAWQ interleave — the nibble at bit position
        // 4 × pos belongs at logical column offset AWQ_ORDER[pos].
        #[allow(clippy::indexing_slicing)]
        for (packed_col, qw_chunk) in qw_row.chunks_exact(4).enumerate() {
            // INDEX: chunks_exact(4) guarantees exactly 4 bytes per chunk
            let packed = u32::from_le_bytes([qw_chunk[0], qw_chunk[1], qw_chunk[2], qw_chunk[3]]);

            // Unpack pack_factor values from this I32
            for pos in 0..pack_factor {
                // CAST: usize → u32, pos is at most 7
                #[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
                let shift = 4 * (pos as u32);
                // BITWISE: extract unsigned quantized value
                let qw = (packed >> shift) & mask;
                // CAST: u32 → f32, exact for values ≤ 15
                #[allow(clippy::as_conversions, clippy::cast_precision_loss)]
                let qw_f32 = qw as f32;
                // INDEX: pos < pack_factor = 8, AWQ_ORDER has 8 entries;
                // packed_col * pack_factor + AWQ_ORDER[pos] < out_features
                // (packed_col < packed_cols, AWQ_ORDER[pos] < pack_factor,
                // packed_cols * pack_factor = out_features)
                unpacked_row[packed_col * pack_factor + AWQ_ORDER[pos]] = qw_f32;
            }
        }

        // --- Pass 2: pure f32 arithmetic into an f32 scratch ---
        // Contiguous f32 reads (unpacked, zeros, scales), contiguous f32
        // writes, no byte manipulation and no narrowing — just sub + mul.
        // INDEX: values_buf.len() == out_features, allocated before the outer loop
        let values_row =
            values_buf
                .get_mut(..out_features)
                .ok_or_else(|| AnamnesisError::Parse {
                    reason: "dequantised values buffer too short".into(),
                })?;
        // VECTORIZED: pending cargo-show-asm verification
        for (((value, &qw), &zero), &scale) in values_row
            .iter_mut()
            .zip(unpacked_row.iter())
            .zip(zeros_row.iter())
            .zip(scales_row.iter())
        {
            *value = (qw - zero) * scale;
        }

        // --- Pass 3: narrow to the caller's output width ---
        E::write_scratch(values_row, out_row);
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

    // -- Small dequantization with known values ------------------------------

    /// Build a minimal AWQ 4-bit test case.
    ///
    /// 8 input features, 8 output features, `group_size`=8, `bits`=4.
    /// `AWQ` packs along `out_features`: `qweight` shape `[8, 1]` (8/8=1 packed col).
    /// All qweight nibbles = 10, all qzeros nibbles = 3 (NO +1), all scales = 2.0.
    ///
    /// Expected: (10 - 3) × 2.0 = 14.0 for every element.
    #[test]
    fn dequant_4bit_uniform() {
        let in_features = 8;
        let out_features = 8;
        let group_size = 8;
        let bits: u8 = 4;

        // qweight: [8, 1] (8 rows, 1 packed col per row)
        // Each I32 packs 8 output features. All nibbles = 10 (0xA).
        let qweight_i32 = 0xAAAA_AAAAu32;
        let mut qweight_data = Vec::new();
        for _i in 0..in_features {
            qweight_data.extend_from_slice(&qweight_i32.to_le_bytes());
        }

        // scales: [1, 8], all 2.0 in F16
        let scale_f16 = half::f16::from_f32(2.0).to_le_bytes();
        let mut scales_data = Vec::new();
        for _ in 0..out_features {
            scales_data.extend_from_slice(&scale_f16);
        }

        // qzeros: [1, 1] (1 group, 1 packed col)
        // All nibbles = 3 (no +1 offset for AWQ)
        let qzeros_i32 = 0x3333_3333u32;
        let qzeros_data = qzeros_i32.to_le_bytes().to_vec();

        let output = dequantize_awq_to_bf16(
            &qweight_data,
            &scales_data,
            &qzeros_data,
            in_features,
            out_features,
            group_size,
            bits,
            Dtype::F16,
        )
        .unwrap();

        assert_eq!(output.len(), in_features * out_features * 2);

        // Expected: (10 - 3) × 2.0 = 14.0 → BF16 0x4160 → LE [0x60, 0x41]
        let bf16_14 = f32_bits_to_bf16_bits(14.0_f32.to_bits());
        for chunk in output.chunks_exact(2) {
            let actual = u16::from_le_bytes([chunk[0], chunk[1]]);
            assert_eq!(actual, bf16_14, "expected BF16 14.0");
        }
    }

    /// Verify that AWQ does NOT apply the +1 zero-point offset that GPTQ uses.
    #[test]
    fn dequant_4bit_no_plus_one_offset() {
        let in_features = 8;
        let out_features = 8;
        let group_size = 8;
        let bits: u8 = 4;

        // qweight: all nibbles = 5
        let qweight_i32 = 0x5555_5555u32;
        let mut qweight_data = Vec::new();
        for _i in 0..in_features {
            qweight_data.extend_from_slice(&qweight_i32.to_le_bytes());
        }

        // scales: all 1.0 in F16 (identity scale)
        let scale_f16 = half::f16::from_f32(1.0).to_le_bytes();
        let mut scales_data = Vec::new();
        for _ in 0..out_features {
            scales_data.extend_from_slice(&scale_f16);
        }

        // qzeros: all nibbles = 3
        let qzeros_i32 = 0x3333_3333u32;
        let qzeros_data = qzeros_i32.to_le_bytes().to_vec();

        let output = dequantize_awq_to_bf16(
            &qweight_data,
            &scales_data,
            &qzeros_data,
            in_features,
            out_features,
            group_size,
            bits,
            Dtype::F16,
        )
        .unwrap();

        // AWQ: (5 - 3) × 1.0 = 2.0 (NOT (5 - 4) × 1.0 = 1.0 like GPTQ would give)
        let bf16_2 = f32_bits_to_bf16_bits(2.0_f32.to_bits());
        for chunk in output.chunks_exact(2) {
            let actual = u16::from_le_bytes([chunk[0], chunk[1]]);
            assert_eq!(actual, bf16_2, "expected BF16 2.0 (AWQ: no +1 offset)");
        }
    }

    /// Pins the `AutoAWQ` GEMM interleave: packed nibbles `0x76543210`
    /// (value `i` at bit position `4 × i`) must unpack to logical column
    /// values `[0, 4, 1, 5, 2, 6, 3, 7]` — i.e. logical column `j` reads
    /// the nibble at position `AWQ_REVERSE_ORDER[j]`. A sequential
    /// (`GPTQ`-style) unpack would yield `[0, 1, 2, …, 7]` instead.
    #[test]
    fn dequant_4bit_interleave_order() {
        let in_features = 8;
        let out_features = 8;
        let group_size = 8;
        let bits: u8 = 4;

        // qweight: every row packs nibble value i at bit position 4×i.
        let qweight_i32 = 0x7654_3210u32;
        let mut qweight_data = Vec::new();
        for _i in 0..in_features {
            qweight_data.extend_from_slice(&qweight_i32.to_le_bytes());
        }

        // scales: all 1.0; qzeros: all 0 → output = unpacked nibble value.
        let scale_f16 = half::f16::from_f32(1.0).to_le_bytes();
        let mut scales_data = Vec::new();
        for _ in 0..out_features {
            scales_data.extend_from_slice(&scale_f16);
        }
        let qzeros_data = 0u32.to_le_bytes().to_vec();

        let output = dequantize_awq_to_bf16(
            &qweight_data,
            &scales_data,
            &qzeros_data,
            in_features,
            out_features,
            group_size,
            bits,
            Dtype::F16,
        )
        .unwrap();

        let expected: [f32; 8] = [0.0, 4.0, 1.0, 5.0, 2.0, 6.0, 3.0, 7.0];
        for i in 0..in_features {
            for (j, &exp) in expected.iter().enumerate() {
                let offset = (i * out_features + j) * 2;
                let actual = u16::from_le_bytes([output[offset], output[offset + 1]]);
                let exp_bf16 = f32_bits_to_bf16_bits(exp.to_bits());
                assert_eq!(actual, exp_bf16, "element [{i},{j}]: expected {exp}");
            }
        }
    }

    /// `AutoAWQ`'s GEMM format is 4-bit only — 8-bit must be rejected, not
    /// decoded with a guessed interleave.
    #[test]
    fn validation_8bit_unsupported() {
        let result = dequantize_awq_to_bf16(&[], &[], &[], 4, 4, 4, 8, Dtype::F16);
        let err = result.unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("4-bit only"),
            "expected 4-bit-only rejection, got: {msg}"
        );
    }

    #[test]
    fn dequant_4bit_two_groups() {
        let in_features = 8;
        let out_features = 8;
        let group_size = 4;
        let bits: u8 = 4;

        // 2 groups: rows 0-3 → group 0, rows 4-7 → group 1
        // qweight: all nibbles = 8
        let qweight_i32 = 0x8888_8888u32;
        let mut qweight_data = Vec::new();
        for _i in 0..in_features {
            qweight_data.extend_from_slice(&qweight_i32.to_le_bytes());
        }

        // scales: [2, 8]
        // Group 0: scale = 1.0, Group 1: scale = 3.0
        let mut scales_data = Vec::new();
        for _ in 0..out_features {
            scales_data.extend_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        }
        for _ in 0..out_features {
            scales_data.extend_from_slice(&half::f16::from_f32(3.0).to_le_bytes());
        }

        // qzeros: [2, 1]
        // Group 0: zero = 6, Group 1: zero = 2
        let qz_g0 = 0x6666_6666u32;
        let qz_g1 = 0x2222_2222u32;
        let mut qzeros_data = Vec::new();
        qzeros_data.extend_from_slice(&qz_g0.to_le_bytes());
        qzeros_data.extend_from_slice(&qz_g1.to_le_bytes());

        let output = dequantize_awq_to_bf16(
            &qweight_data,
            &scales_data,
            &qzeros_data,
            in_features,
            out_features,
            group_size,
            bits,
            Dtype::F16,
        )
        .unwrap();

        // Group 0 (rows 0-3): (8 - 6) × 1.0 = 2.0
        let bf16_2 = f32_bits_to_bf16_bits(2.0_f32.to_bits());
        for i in 0..4 {
            for j in 0..out_features {
                let offset = (i * out_features + j) * 2;
                let actual = u16::from_le_bytes([output[offset], output[offset + 1]]);
                assert_eq!(actual, bf16_2, "element [{i},{j}]: expected BF16 2.0");
            }
        }

        // Group 1 (rows 4-7): (8 - 2) × 3.0 = 18.0
        let bf16_18 = f32_bits_to_bf16_bits(18.0_f32.to_bits());
        for i in 4..8 {
            for j in 0..out_features {
                let offset = (i * out_features + j) * 2;
                let actual = u16::from_le_bytes([output[offset], output[offset + 1]]);
                assert_eq!(actual, bf16_18, "element [{i},{j}]: expected BF16 18.0");
            }
        }
    }

    // -- Validation errors ---------------------------------------------------

    #[test]
    fn validation_unsupported_bits() {
        let result = dequantize_awq_to_bf16(&[], &[], &[], 8, 8, 8, 3, Dtype::F16);
        assert!(result.is_err());
    }

    #[test]
    fn validation_zero_dimensions() {
        let result = dequantize_awq_to_bf16(&[], &[], &[], 0, 8, 8, 4, Dtype::F16);
        assert!(result.is_err());
    }

    #[test]
    fn validation_qweight_length_mismatch() {
        // in_features=8, out_features=8, bits=4 → qweight should be 8×1×4=32 bytes
        let result = dequantize_awq_to_bf16(
            &[0u8; 16], // wrong: should be 32
            &[0u8; 16], // scales: 1 group × 8 × 2 = 16
            &[0u8; 4],  // qzeros: 1 group × 1 × 4 = 4
            8,
            8,
            8,
            4,
            Dtype::F16,
        );
        assert!(result.is_err());
    }
}
