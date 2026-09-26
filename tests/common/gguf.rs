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
/// Covers all 22 dequantizable block types: Phase 4.5 step 6 closed the
/// `GGUF` coverage gap, so every block-quantised `GgufType` variant is
/// exercised by `cross_validation_gguf.rs`.
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
