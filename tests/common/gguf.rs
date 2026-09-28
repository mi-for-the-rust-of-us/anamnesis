// SPDX-License-Identifier: MIT OR Apache-2.0

//! `GGUF` helpers shared by `cross_validation_gguf.rs` and
//! `cross_validation_ollama.rs`. The two suites read different fixture
//! containers (v2 `AMNG` against the older 16-byte header), so each keeps its
//! own parser; what they share is the type mapping and the `BF16` check.

use std::time::Instant;

use anamnesis::{GgufType, dequantize_gguf_to_bf16};

use super::bf16::compare_bf16;

/// Maps a `ggml_type` discriminant to a [`GgufType`].
///
/// Maps all 24 block types. The 22 production kernels are cross-validated by
/// `cross_validation_gguf.rs`; `Q8_1` and `Q8_K` (intermediate formats with no
/// shipped models) are covered by unit tests instead.
pub fn gguf_type_from_disc(disc: u32) -> GgufType {
    match disc {
        2 => GgufType::Q4_0,
        3 => GgufType::Q4_1,
        6 => GgufType::Q5_0,
        7 => GgufType::Q5_1,
        8 => GgufType::Q8_0,
        9 => GgufType::Q8_1,
        10 => GgufType::Q2_K,
        11 => GgufType::Q3_K,
        12 => GgufType::Q4_K,
        13 => GgufType::Q5_K,
        14 => GgufType::Q6_K,
        15 => GgufType::Q8_K,
        20 => GgufType::IQ4_NL,
        23 => GgufType::IQ4_XS,
        16 => GgufType::IQ2_XXS,
        17 => GgufType::IQ2_XS,
        22 => GgufType::IQ2_S,
        18 => GgufType::IQ3_XXS,
        21 => GgufType::IQ3_S,
        19 => GgufType::IQ1_S,
        29 => GgufType::IQ1_M,
        34 => GgufType::TQ1_0,
        35 => GgufType::TQ2_0,
        39 => GgufType::MXFP4,
        other => panic!("unknown ggml_type discriminant: {other}"),
    }
}

/// Dequantizes `raw_data` to `BF16`, times it, and asserts it matches
/// `expected_bf16` within `max_ulp` per element.
pub fn check_bf16_against_golden(
    name: &str,
    raw_data: &[u8],
    dtype: GgufType,
    n_elements: usize,
    expected_bf16: &[u8],
    max_ulp: u16,
) {
    let total = n_elements;

    // Dequantize with anamnesis and measure time.
    let start = Instant::now();
    let actual =
        dequantize_gguf_to_bf16(raw_data, dtype, n_elements).expect("dequantization failed");
    let elapsed = start.elapsed();

    assert_eq!(
        actual.len(),
        expected_bf16.len(),
        "{name}: output length mismatch"
    );

    let (mismatches, max_diff) = compare_bf16(&actual, expected_bf16, max_ulp);
    eprintln!(
        "{name}: {total} elements, {mismatches} mismatches, \
         max ULP diff = {max_diff}, anamnesis = {:.1} \u{b5}s",
        elapsed.as_secs_f64() * 1e6
    );
    assert_eq!(
        mismatches, 0,
        "{name}: {mismatches}/{total} elements differ by more than {max_ulp} ULP"
    );
}

/// One tensor-info record of a raw `GGUF` file.
pub struct RawTensorInfo<'a> {
    /// Tensor name.
    pub name: &'a str,
    /// Dimensions as stored on disk (innermost first).
    pub dims: &'a [u64],
    /// `ggml_type` discriminant (`0` = `F32`).
    pub ggml_type: u32,
    /// Offset relative to the start of the tensor-data section.
    pub offset: u64,
}

/// `GGUF` metadata value type discriminant for `UINT32`.
pub const GGUF_KV_U32: u32 = 4;

/// Writes a `GGUF` v3 file byte for byte, with none of the writer's validation:
/// the hostile-input tests need files the crate's own writer refuses to make
/// (aliased offsets, odd alignments). Each metadata entry is
/// `(key, value type discriminant, encoded value)`. The tensor-info table is
/// padded to `alignment`, then `data` follows.
pub fn raw_gguf(
    kvs: &[(&str, u32, &[u8])],
    tensors: &[RawTensorInfo<'_>],
    alignment: usize,
    data: &[u8],
) -> Vec<u8> {
    fn string(out: &mut Vec<u8>, text: &str) {
        out.extend_from_slice(&(text.len() as u64).to_le_bytes());
        out.extend_from_slice(text.as_bytes());
    }
    let mut out = b"GGUF".to_vec();
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
    for (key, value_type, value) in kvs {
        string(&mut out, key);
        out.extend_from_slice(&value_type.to_le_bytes());
        out.extend_from_slice(value);
    }
    for t in tensors {
        string(&mut out, t.name);
        out.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
        for dim in t.dims {
            out.extend_from_slice(&dim.to_le_bytes());
        }
        out.extend_from_slice(&t.ggml_type.to_le_bytes());
        out.extend_from_slice(&t.offset.to_le_bytes());
    }
    let pad = (alignment - out.len() % alignment) % alignment;
    out.resize(out.len() + pad, 0);
    out.extend_from_slice(data);
    out
}
