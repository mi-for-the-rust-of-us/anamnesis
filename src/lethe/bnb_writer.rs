// SPDX-License-Identifier: MIT OR Apache-2.0

//! High-level `BnB`-NF4 safetensors writer — the Phase 6 end-to-end
//! `BF16 → BnB-NF4 safetensors file` path.
//!
//! Phase 5 shipped the low-level
//! [`encode_bnb4_compute_absmax`] kernel: takes a `BF16` slice, returns
//! packed nibbles + per-block absmax. This module wraps that kernel into
//! the on-disk safetensors companion-tensor layout that `bitsandbytes`
//! and the [`remember::bnb`](crate::remember::bnb) decode path expect.
//!
//! For each eligible source tensor the writer emits **four** tensors:
//!
//! | Output tensor | Dtype | Shape | Content |
//! |---|---|---|---|
//! | `<base>.weight` | `U8` | `[total/2, 1]` | packed 4-bit nibbles |
//! | `<base>.weight.absmax` | `F32` | `[num_blocks]` | per-block absmax |
//! | `<base>.weight.quant_map` | `F32` | `[16]` | the `NF4_CODEBOOK` |
//! | `<base>.weight.quant_state.bitsandbytes__nf4` | `U8` | `[len]` | JSON blob |
//!
//! The JSON blob is the minimum `quant_state` that
//! [`parse_bnb_quant_state_shape`](crate::model) needs to recover the
//! original 2-D shape: `{"quant_type":"nf4","blocksize":64,"shape":[r,c],"nested":false}`.
//!
//! # Eligibility
//!
//! Phase 6 quantises **2-D tensors only**, matching real `bitsandbytes`
//! encoder behaviour. 1-D biases / norms and ≥3-D embeddings pass through
//! unchanged in their input dtype. Tensors whose element count is not a
//! multiple of `block_size = 64` also pass through (with a warning emitted
//! by the calling CLI layer).

use std::fmt::Write as _;
use std::path::Path;

use crate::error::AnamnesisError;
use crate::lethe::bnb::{NF4_CODEBOOK, encode_bnb4_compute_absmax};
use crate::parse::utils::checked_num_elements;

/// `bitsandbytes`-default block size for `NF4` encoding.
pub const NF4_BLOCK_SIZE: usize = 64;

/// A tensor going into [`write_bnb_nf4_safetensors`]. All payloads are
/// `BF16` little-endian; non-`BF16` inputs must be converted by the caller
/// (the CLI layer, behind the `cli` feature, handles this).
#[derive(Debug, Clone)]
pub struct BnbWriteInput<'a> {
    /// Tensor name as it should appear in the output safetensors file
    /// (already mapped from whatever convention the source uses; the
    /// writer treats this as opaque).
    pub name: &'a str,
    /// Tensor shape (row-major, safetensors convention — NOT MSB-first).
    pub shape: &'a [usize],
    /// Raw `BF16` little-endian bytes. Length must be `2 × product(shape)`.
    pub bf16_data: &'a [u8],
}

/// Classifies a source tensor as "quantize to NF4" vs "passthrough as BF16".
///
/// Eligibility: 2-D shape, `product(shape) >= NF4_BLOCK_SIZE`, and the
/// element count is a multiple of `NF4_BLOCK_SIZE`. Anything else
/// passes through unchanged so the writer never produces a tensor that
/// can't be decoded by the existing `remember::bnb` path.
#[must_use]
pub fn is_eligible_for_nf4(shape: &[usize]) -> bool {
    if shape.len() != 2 {
        return false;
    }
    // Checked product: an adversarial shape whose element count overflows
    // `usize` is treated as not eligible (pass-through) rather than panicking
    // in debug / wrapping in release.
    let Some(total) = checked_num_elements(shape) else {
        return false;
    };
    total >= NF4_BLOCK_SIZE && total.is_multiple_of(NF4_BLOCK_SIZE)
}

/// Codebook bytes — `NF4_CODEBOOK` flattened to LE `F32`. Pre-computed
/// once per `write_bnb_nf4_safetensors` call.
fn codebook_bytes() -> Vec<u8> {
    NF4_CODEBOOK.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Minimum `quant_state` JSON the decoder needs to recover the original
/// 2-D shape. Mirrors the keys [`parse_bnb_quant_state_shape`](crate::model)
/// reads.
fn quant_state_json_bytes(shape: &[usize]) -> Vec<u8> {
    // We hand-build the JSON to keep the field order stable and avoid
    // pulling in `serde_json::to_string` allocations for a fixed-shape
    // blob. `blocksize` is sourced from [`NF4_BLOCK_SIZE`] so that any
    // future change to the default block size flows through to the
    // emitted `quant_state` blob.
    let mut s = String::new();
    let _ = write!(
        &mut s,
        r#"{{"quant_type":"nf4","blocksize":{NF4_BLOCK_SIZE},"shape":["#
    );
    for (i, dim) in shape.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        // `write!` on a `String` never fails — `usize`'s `Display` impl
        // is infallible and `String`'s `fmt::Write` impl never returns
        // `Err`.
        let _ = write!(&mut s, "{dim}");
    }
    s.push_str(r#"],"nested":false}"#);
    s.into_bytes()
}

/// Encodes the eligible tensors to `BnB-NF4` and writes a safetensors file
/// containing the four companion tensors per quantised input plus the raw
/// `BF16` passthrough tensors.
///
/// Pipeline per quantisable input:
///
/// 1. `encode_bnb4_compute_absmax(bf16, codebook, total, 64)` → `(weight, absmax)`.
/// 2. Emit `<name>.weight` (`U8`, `[total/2, 1]`), `<name>.weight.absmax`
///    (`F32`, `[num_blocks]`), `<name>.weight.quant_map` (`F32`, `[16]`),
///    `<name>.weight.quant_state.bitsandbytes__nf4` (`U8`, the JSON
///    `quant_state` blob).
///
/// Inputs that fail [`is_eligible_for_nf4`] are emitted unchanged as a
/// `BF16` tensor (`<name>`, `[shape]`).
///
/// Tensor output order is sorted by tensor name lexicographically for
/// deterministic serialisation.
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] when an input's `bf16_data.len()`
/// disagrees with `2 × product(shape)`, when shape contains a zero
/// dimension, or when safetensors serialisation fails.
///
/// Returns [`AnamnesisError::Io`] if the output file cannot be written.
///
/// # Memory
///
/// For each quantised tensor: one packed-weight `Vec<u8>` (`total/2`
/// bytes), one `absmax` `Vec<u8>` (`num_blocks × 4` bytes), plus a 64-byte
/// codebook and a small `quant_state` JSON blob. All four are retained
/// simultaneously until `safetensors::serialize_to_file` returns — same
/// retention shape as `ParsedModel::remember`. Passthrough tensors borrow
/// their `bf16_data` slice without copying.
pub fn write_bnb_nf4_safetensors(
    inputs: &[BnbWriteInput<'_>],
    output: impl AsRef<Path>,
) -> crate::Result<()> {
    let bytes = write_bnb_nf4_safetensors_bytes(inputs)?;
    std::fs::write(output.as_ref(), &bytes).map_err(AnamnesisError::Io)
}

/// In-memory variant of [`write_bnb_nf4_safetensors`]. Returns the
/// serialised safetensors bytes.
///
/// # Errors
///
/// Same as [`write_bnb_nf4_safetensors`] except no `Io` arm — this
/// function never touches disk.
///
/// # Memory
///
/// Same as [`write_bnb_nf4_safetensors`] plus the final output buffer.
pub fn write_bnb_nf4_safetensors_bytes(inputs: &[BnbWriteInput<'_>]) -> crate::Result<Vec<u8>> {
    let mut owned_storage: Vec<(String, safetensors::Dtype, Vec<usize>, Vec<u8>)> = Vec::new();

    let codebook = codebook_bytes();

    // We need each tensor to share its name's quantisation state; collect
    // sorted (deterministic output) before processing.
    let mut sorted_inputs: Vec<&BnbWriteInput<'_>> = inputs.iter().collect();
    sorted_inputs.sort_by_key(|t| t.name);

    for input in sorted_inputs {
        // Validate bf16 byte count.
        let total_elements: usize = input
            .shape
            .iter()
            .copied()
            .try_fold(1usize, |acc, d| {
                if d == 0 {
                    return None;
                }
                acc.checked_mul(d)
            })
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!(
                    "BnB-NF4 write `{}`: shape {:?} element-count overflow or zero dimension",
                    input.name, input.shape
                ),
            })?;
        let expected_bytes =
            total_elements
                .checked_mul(2)
                .ok_or_else(|| AnamnesisError::Parse {
                    reason: format!("BnB-NF4 write `{}`: BF16 byte count overflow", input.name),
                })?;
        if input.bf16_data.len() != expected_bytes {
            return Err(AnamnesisError::Parse {
                reason: format!(
                    "BnB-NF4 write `{}`: bf16_data length {} != expected {expected_bytes} \
                     bytes (shape {:?})",
                    input.name,
                    input.bf16_data.len(),
                    input.shape
                ),
            });
        }

        if is_eligible_for_nf4(input.shape) {
            let (weight, absmax) = encode_bnb4_compute_absmax(
                input.bf16_data,
                &codebook,
                total_elements,
                NF4_BLOCK_SIZE,
            )?;
            // Weight shape `[total/2, 1]` matches the bitsandbytes
            // ecosystem convention; the original 2-D shape is recovered
            // by the decoder via the quant_state JSON blob.
            owned_storage.push((
                format!("{}.weight", input.name),
                safetensors::Dtype::U8,
                vec![total_elements / 2, 1],
                weight,
            ));
            let num_blocks = total_elements / NF4_BLOCK_SIZE;
            owned_storage.push((
                format!("{}.weight.absmax", input.name),
                safetensors::Dtype::F32,
                vec![num_blocks],
                absmax,
            ));
            owned_storage.push((
                format!("{}.weight.quant_map", input.name),
                safetensors::Dtype::F32,
                vec![16],
                codebook.clone(),
            ));
            let qs = quant_state_json_bytes(input.shape);
            let qs_len = qs.len();
            owned_storage.push((
                format!("{}.weight.quant_state.bitsandbytes__nf4", input.name),
                safetensors::Dtype::U8,
                vec![qs_len],
                qs,
            ));
        } else {
            // Passthrough: BF16, original shape, original bytes.
            owned_storage.push((
                input.name.to_owned(),
                safetensors::Dtype::BF16,
                input.shape.to_vec(),
                input.bf16_data.to_vec(),
            ));
        }
    }

    // Re-sort the OUTPUT tensors by name so the safetensors header is
    // deterministic regardless of which inputs were eligible for NF4.
    owned_storage.sort_by(|a, b| a.0.cmp(&b.0));

    let mut views: Vec<(String, safetensors::tensor::TensorView<'_>)> =
        Vec::with_capacity(owned_storage.len());
    for (name, dtype, shape, data) in &owned_storage {
        let view =
            safetensors::tensor::TensorView::new(*dtype, shape.clone(), data).map_err(|e| {
                AnamnesisError::Parse {
                    reason: format!("failed to create TensorView for `{name}`: {e}"),
                }
            })?;
        views.push((name.clone(), view));
    }

    // EXHAUSTIVE: SafeTensorError is a foreign type that may gain variants
    #[allow(clippy::wildcard_enum_match_arm)]
    safetensors::tensor::serialize(views, None).map_err(|e| AnamnesisError::Parse {
        reason: format!("failed to serialize BnB-NF4 safetensors: {e}"),
    })
}

/// Summary of how a list of `BnbWriteInput` was classified — useful for
/// CLI reporting before/during the write.
#[derive(Debug, Default, Clone, Copy)]
#[must_use]
pub struct BnbNf4WriteStats {
    /// Number of input tensors quantised to `BnB-NF4`.
    pub quantized: usize,
    /// Number of input tensors passed through as `BF16`.
    pub passthrough: usize,
}

/// Classifies inputs into (quantised, passthrough) without actually
/// performing the encoding. Useful for printing a one-line summary before
/// the conversion begins.
pub fn classify_inputs(inputs: &[BnbWriteInput<'_>]) -> BnbNf4WriteStats {
    let mut stats = BnbNf4WriteStats::default();
    for input in inputs {
        if is_eligible_for_nf4(input.shape) {
            stats.quantized += 1;
        } else {
            stats.passthrough += 1;
        }
    }
    stats
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::wildcard_enum_match_arm
)]
mod tests {
    use super::*;
    use crate::parse::safetensors::{QuantScheme, parse_safetensors_header};
    use crate::remember::bnb::dequantize_bnb4_to_bf16;

    fn synth_bf16(rows: usize, cols: usize) -> Vec<u8> {
        let n = rows * cols;
        let mut out = Vec::with_capacity(n * 2);
        for i in 0..n {
            // Build a smooth ramp from -1 to +1.
            let v = (i as f32) / (n as f32) * 2.0 - 1.0;
            // BF16 = upper 16 bits of f32 (truncate).
            let bits = (v.to_bits() >> 16) as u16;
            out.extend_from_slice(&bits.to_le_bytes());
        }
        out
    }

    #[test]
    fn eligibility_only_2d_multiples_of_64() {
        assert!(is_eligible_for_nf4(&[64, 1]));
        assert!(is_eligible_for_nf4(&[8, 8]));
        assert!(is_eligible_for_nf4(&[128, 256]));
        assert!(!is_eligible_for_nf4(&[63, 1])); // not multiple of 64
        assert!(!is_eligible_for_nf4(&[64])); // 1-D
        assert!(!is_eligible_for_nf4(&[4, 4, 4])); // 3-D
        assert!(!is_eligible_for_nf4(&[])); // 0-D
    }

    #[test]
    fn quant_state_json_shape_recovery() {
        let blob = quant_state_json_bytes(&[256, 64]);
        let s = std::str::from_utf8(&blob).unwrap();
        // Must parse via serde_json and expose the "shape" array.
        let v: serde_json::Value = serde_json::from_str(s).unwrap();
        let arr = v["shape"].as_array().unwrap();
        assert_eq!(arr[0].as_u64(), Some(256));
        assert_eq!(arr[1].as_u64(), Some(64));
        assert_eq!(v["quant_type"].as_str(), Some("nf4"));
        assert_eq!(v["blocksize"].as_u64(), Some(64));
        assert_eq!(v["nested"].as_bool(), Some(false));
    }

    #[test]
    fn write_then_parse_detects_bnb_nf4_scheme() {
        let bf16 = synth_bf16(64, 1);
        let inputs = vec![BnbWriteInput {
            name: "linear",
            shape: &[64, 1],
            bf16_data: &bf16,
        }];
        let bytes = write_bnb_nf4_safetensors_bytes(&inputs).unwrap();
        let header = parse_safetensors_header(&bytes).unwrap();
        assert_eq!(
            header.scheme,
            QuantScheme::Bnb4,
            "scheme should be detected as Bnb4"
        );
        // Expect 4 tensors per quantized weight.
        let names: Vec<&str> = header.tensors.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"linear.weight"));
        assert!(names.contains(&"linear.weight.absmax"));
        assert!(names.contains(&"linear.weight.quant_map"));
        assert!(names.contains(&"linear.weight.quant_state.bitsandbytes__nf4"));
    }

    #[test]
    fn passthrough_for_ineligible_shapes() {
        let bf16_1d = synth_bf16(1, 8); // 8 elements, will be reshaped to [8] below
        let inputs = vec![BnbWriteInput {
            name: "norm",
            shape: &[8],
            bf16_data: &bf16_1d,
        }];
        let bytes = write_bnb_nf4_safetensors_bytes(&inputs).unwrap();
        let header = parse_safetensors_header(&bytes).unwrap();
        // No quantized companions — just one BF16 tensor.
        let names: Vec<&str> = header.tensors.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, vec!["norm"]);
    }

    #[test]
    fn round_trip_decode_recovers_within_codebook_error() {
        // Encode → decode → compare against original BF16.
        let bf16 = synth_bf16(64, 1);
        let inputs = vec![BnbWriteInput {
            name: "linear",
            shape: &[64, 1],
            bf16_data: &bf16,
        }];
        let bytes = write_bnb_nf4_safetensors_bytes(&inputs).unwrap();
        // Recover absmax + weight + quant_map from the parsed safetensors.
        let parsed = safetensors::SafeTensors::deserialize(&bytes).unwrap();
        let weight = parsed.tensor("linear.weight").unwrap();
        let absmax = parsed.tensor("linear.weight.absmax").unwrap();
        let qmap = parsed.tensor("linear.weight.quant_map").unwrap();

        let total_elements = 64;
        let decoded = dequantize_bnb4_to_bf16(
            weight.data(),
            absmax.data(),
            qmap.data(),
            total_elements,
            NF4_BLOCK_SIZE,
        )
        .unwrap();

        // Re-encoding the decoded BF16 must produce identical weight bytes
        // (Phase 5 idempotency guarantee).
        let re_inputs = vec![BnbWriteInput {
            name: "linear",
            shape: &[64, 1],
            bf16_data: &decoded,
        }];
        let re_bytes = write_bnb_nf4_safetensors_bytes(&re_inputs).unwrap();
        let re_parsed = safetensors::SafeTensors::deserialize(&re_bytes).unwrap();
        let re_weight = re_parsed.tensor("linear.weight").unwrap();
        assert_eq!(
            weight.data(),
            re_weight.data(),
            "BnB-NF4 encode is not idempotent on already-quantized BF16"
        );
    }

    #[test]
    fn rejects_bf16_length_mismatch() {
        // shape [64, 1] needs 128 bytes; supply only 16.
        let bf16 = vec![0u8; 16];
        let inputs = vec![BnbWriteInput {
            name: "w",
            shape: &[64, 1],
            bf16_data: &bf16,
        }];
        let err = write_bnb_nf4_safetensors_bytes(&inputs).expect_err("should reject");
        match err {
            AnamnesisError::Parse { reason } => {
                assert!(
                    reason.contains("bf16_data length"),
                    "unexpected reason: {reason}"
                );
            }
            other => panic!("expected Parse, got {other:?}"),
        }
    }

    #[test]
    fn classify_counts() {
        let bf16_a = synth_bf16(64, 1);
        let bf16_b = synth_bf16(1, 8);
        let inputs = vec![
            BnbWriteInput {
                name: "w",
                shape: &[64, 1],
                bf16_data: &bf16_a,
            },
            BnbWriteInput {
                name: "b",
                shape: &[8],
                bf16_data: &bf16_b,
            },
        ];
        let stats = classify_inputs(&inputs);
        assert_eq!(stats.quantized, 1);
        assert_eq!(stats.passthrough, 1);
    }
}
