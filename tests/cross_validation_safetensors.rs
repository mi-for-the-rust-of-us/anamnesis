// SPDX-License-Identifier: MIT OR Apache-2.0

//! Cross-validation tests for safetensors header parsing against the
//! upstream `safetensors` Python library (`HuggingFace`).
//!
//! Four small `.safetensors` fixtures (one per quantization scheme
//! anamnesis detects: `FP8`, `GPTQ`, `AWQ`, `BnB` `NF4`) live under
//! `tests/fixtures/safetensors_reference/`. Each is paired with a
//! `<scheme>.expected.json` reference recording exactly what the Python
//! library reports about that file's header — see
//! `tests/fixtures/safetensors_reference/generate.py` for how the
//! references are produced and triple-checked (raw spec parse +
//! `safetensors.safe_open` cross-check).
//!
//! For every fixture, the tests below run **both** anamnesis entry
//! points — the slice-based [`parse_safetensors_header`] and the
//! reader-based [`parse_safetensors_header_from_reader`] — and assert
//! field-for-field equality with the Python reference. This is true
//! cross-validation: anamnesis output is compared against an external
//! oracle, not against another anamnesis path.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::wildcard_enum_match_arm
)]

use std::collections::HashMap;

use anamnesis::{
    Dtype, SafetensorsHeader, parse_safetensors_header, parse_safetensors_header_from_reader,
};
use serde::Deserialize;

// ---------------------------------------------------------------------------
// Reference parsing
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct ExpectedTensor {
    name: String,
    dtype: String,
    shape: Vec<usize>,
    data_offsets: (usize, usize),
}

#[derive(Debug, Deserialize)]
struct ExpectedHeader {
    /// Scheme name, exactly as written by `generate.py` (kept for
    /// readability of test failures; the Rust scheme detection is
    /// asserted separately by unit tests in `parse::safetensors`).
    #[allow(dead_code)]
    scheme: String,
    header_size: usize,
    #[serde(default)]
    metadata: HashMap<String, String>,
    tensors: Vec<ExpectedTensor>,
}

fn load_reference(json: &str) -> ExpectedHeader {
    serde_json::from_str(json).expect("malformed reference JSON")
}

fn dtype_to_string(dtype: Dtype) -> String {
    // EXHAUSTIVE: `Dtype` is `#[non_exhaustive]` so a wildcard is required;
    // the wildcard preserves any future-added variant by routing it through
    // the `Display` impl, which is the canonical name source for this enum.
    match dtype {
        Dtype::F8E4M3 => "F8_E4M3".to_owned(),
        Dtype::F8E5M2 => "F8_E5M2".to_owned(),
        Dtype::BF16 => "BF16".to_owned(),
        Dtype::F16 => "F16".to_owned(),
        Dtype::F32 => "F32".to_owned(),
        Dtype::F64 => "F64".to_owned(),
        Dtype::Bool => "BOOL".to_owned(),
        Dtype::U8 => "U8".to_owned(),
        Dtype::I8 => "I8".to_owned(),
        Dtype::U16 => "U16".to_owned(),
        Dtype::I16 => "I16".to_owned(),
        Dtype::U32 => "U32".to_owned(),
        Dtype::I32 => "I32".to_owned(),
        Dtype::U64 => "U64".to_owned(),
        Dtype::I64 => "I64".to_owned(),
        other => other.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Per-fixture validator
// ---------------------------------------------------------------------------

/// Cross-validate one fixture against its Python-sourced reference, via
/// both the slice-based and reader-based anamnesis entry points. Asserts
/// that **both** entry points produce the exact same `SafetensorsHeader`
/// fields the Python `safetensors` library reports.
fn cross_validate(fixture_label: &str, bytes: &[u8], reference_json: &str) {
    let expected = load_reference(reference_json);

    let slice_header = parse_safetensors_header(bytes)
        .unwrap_or_else(|e| panic!("[{fixture_label}] slice parse failed: {e}"));
    let reader_header = parse_safetensors_header_from_reader(std::io::Cursor::new(bytes))
        .unwrap_or_else(|e| panic!("[{fixture_label}] reader parse failed: {e}"));

    for (path_label, header) in [("slice", &slice_header), ("reader", &reader_header)] {
        assert_eq!(
            header.header_size, expected.header_size,
            "[{fixture_label}/{path_label}] header_size mismatch"
        );

        let actual_metadata: HashMap<String, String> = header.metadata.clone().unwrap_or_default();
        assert_eq!(
            actual_metadata, expected.metadata,
            "[{fixture_label}/{path_label}] file metadata mismatch"
        );

        assert_eq!(
            header.tensors.len(),
            expected.tensors.len(),
            "[{fixture_label}/{path_label}] tensor count mismatch (got {} vs expected {})",
            header.tensors.len(),
            expected.tensors.len()
        );

        for (actual, expected_t) in header.tensors.iter().zip(expected.tensors.iter()) {
            assert_eq!(
                actual.name, expected_t.name,
                "[{fixture_label}/{path_label}] tensor name mismatch"
            );
            assert_eq!(
                dtype_to_string(actual.dtype).as_str(),
                expected_t.dtype.as_str(),
                "[{fixture_label}/{path_label}] dtype mismatch for `{}`",
                actual.name
            );
            assert_eq!(
                actual.shape, expected_t.shape,
                "[{fixture_label}/{path_label}] shape mismatch for `{}`",
                actual.name
            );
            assert_eq!(
                actual.data_offsets, expected_t.data_offsets,
                "[{fixture_label}/{path_label}] data_offsets mismatch for `{}`",
                actual.name
            );
        }
    }

    headers_must_agree(fixture_label, &slice_header, &reader_header);
}

/// Substrate-equivalence cross-check: even after both paths matched the
/// Python reference, assert directly that they produced the same
/// `SafetensorsHeader`. This guards against a regression where both paths
/// might independently match the reference on the fields recorded above
/// while diverging on a field the reference does not cover (scheme,
/// scheme-specific config, role classification).
fn headers_must_agree(
    fixture_label: &str,
    slice_header: &SafetensorsHeader,
    reader_header: &SafetensorsHeader,
) {
    assert_eq!(
        slice_header.scheme, reader_header.scheme,
        "[{fixture_label}] scheme mismatch between slice and reader paths"
    );
    assert_eq!(
        slice_header.gptq_config, reader_header.gptq_config,
        "[{fixture_label}] gptq_config mismatch between slice and reader paths"
    );
    assert_eq!(
        slice_header.awq_config, reader_header.awq_config,
        "[{fixture_label}] awq_config mismatch between slice and reader paths"
    );
    assert_eq!(
        slice_header.bnb_config, reader_header.bnb_config,
        "[{fixture_label}] bnb_config mismatch between slice and reader paths"
    );
    for (a, b) in slice_header
        .tensors
        .iter()
        .zip(reader_header.tensors.iter())
    {
        assert_eq!(
            a.role, b.role,
            "[{fixture_label}] role mismatch for `{}`",
            a.name
        );
    }
}

// ---------------------------------------------------------------------------
// Per-scheme tests
// ---------------------------------------------------------------------------

#[test]
fn cross_validate_fp8_against_python_reference() {
    cross_validate(
        "FP8",
        include_bytes!("fixtures/safetensors_reference/fp8.safetensors"),
        include_str!("fixtures/safetensors_reference/fp8.expected.json"),
    );
}

#[test]
#[cfg(feature = "gptq")]
fn cross_validate_gptq_against_python_reference() {
    cross_validate(
        "GPTQ",
        include_bytes!("fixtures/safetensors_reference/gptq.safetensors"),
        include_str!("fixtures/safetensors_reference/gptq.expected.json"),
    );
}

#[test]
#[cfg(feature = "awq")]
fn cross_validate_awq_against_python_reference() {
    cross_validate(
        "AWQ",
        include_bytes!("fixtures/safetensors_reference/awq.safetensors"),
        include_str!("fixtures/safetensors_reference/awq.expected.json"),
    );
}

#[test]
#[cfg(feature = "bnb")]
fn cross_validate_bnb_nf4_against_python_reference() {
    cross_validate(
        "BnB-NF4",
        include_bytes!("fixtures/safetensors_reference/bnb_nf4.safetensors"),
        include_str!("fixtures/safetensors_reference/bnb_nf4.expected.json"),
    );
}

// ---------------------------------------------------------------------------
// On-disk reader-coverage test
// ---------------------------------------------------------------------------

/// Confirms `parse_safetensors_header_from_reader` works on a real
/// `std::fs::File` (not just an in-memory `Cursor`). The four
/// fixture-driven tests above use `Cursor`, so this exists to lock the
/// `File` substrate path that an HTTP-range adapter would resemble after
/// the prefix + JSON bytes have been replayed sequentially.
#[test]
fn cross_validate_fp8_reader_path_on_real_file() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/safetensors_reference/fp8.safetensors"
    );
    let expected = load_reference(include_str!(
        "fixtures/safetensors_reference/fp8.expected.json"
    ));

    let file = std::fs::File::open(path).expect("open fp8.safetensors failed");
    let reader_header =
        parse_safetensors_header_from_reader(file).expect("reader parse from file failed");

    assert_eq!(reader_header.header_size, expected.header_size);
    assert_eq!(reader_header.tensors.len(), expected.tensors.len());
    for (actual, expected_t) in reader_header.tensors.iter().zip(expected.tensors.iter()) {
        assert_eq!(actual.name, expected_t.name);
        assert_eq!(
            dtype_to_string(actual.dtype).as_str(),
            expected_t.dtype.as_str()
        );
        assert_eq!(actual.shape, expected_t.shape);
        assert_eq!(actual.data_offsets, expected_t.data_offsets);
    }
}
