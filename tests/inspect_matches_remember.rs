// SPDX-License-Identifier: MIT OR Apache-2.0

//! `inspect`'s `dequantized_size` must equal what `remember` actually produces.
//!
//! The inspect-before-parse gate a host builds on is only as good as this one
//! number, and until v0.7.9 it was 8× too small for 4-bit `GPTQ` / `AWQ` models:
//! the estimate counted packed `I32` elements of `.qweight`, not the weights
//! packed into them (Phase 7.9, audit finding M-1). No test compared the
//! estimate with the output, so nothing noticed. This one does, for every
//! reference fixture, at every output width.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use anamnesis::{InspectOptions, TargetDtype};

/// Asserts `inspect().dequantized_size` equals the byte length of the tensors
/// `remember_to_bytes` writes, at `BF16`, `F32` and `F16`.
fn assert_estimate_matches_output(fixture: &str) {
    let path = format!("tests/fixtures/safetensors_reference/{fixture}");
    let bytes = std::fs::read(&path).expect("reference fixture");
    let model = anamnesis::parse_bytes(bytes).unwrap();
    for dtype in [TargetDtype::BF16, TargetDtype::F32, TargetDtype::F16] {
        let estimate = model
            .inspect_with_options(&InspectOptions::new().with_output_dtype(dtype))
            .dequantized_size;
        let output = model.remember_to_bytes(dtype).unwrap();
        let header = anamnesis::parse_safetensors_header(&output).unwrap();
        let actual: u64 = header
            .tensors
            .iter()
            .map(|t| u64::try_from(t.byte_len()).unwrap())
            .sum();
        assert_eq!(
            estimate, actual,
            "{fixture} at {dtype:?}: inspect says {estimate} B, remember wrote {actual} B"
        );
    }
}

#[test]
fn fp8_estimate_matches_output() {
    assert_estimate_matches_output("fp8.safetensors");
}

#[cfg(feature = "gptq")]
#[test]
fn gptq_estimate_matches_output() {
    assert_estimate_matches_output("gptq.safetensors");
}

#[cfg(feature = "awq")]
#[test]
fn awq_estimate_matches_output() {
    assert_estimate_matches_output("awq.safetensors");
}

#[cfg(feature = "bnb")]
#[test]
fn bnb_nf4_estimate_matches_output() {
    assert_estimate_matches_output("bnb_nf4.safetensors");
}
