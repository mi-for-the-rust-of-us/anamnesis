// SPDX-License-Identifier: MIT OR Apache-2.0

//! Cross-validation against `Ollama`-distributed `GGUF` blobs.
//!
//! Phase 6.5, real-world cross-validation track. The dequant correctness
//! claim has been validated against the `gguf` Python package's reference
//! implementation on bartowski / `TheBloke` fixtures in
//! [`cross_validation_gguf`](super::cross_validation_gguf). This file
//! extends the validation to the dominant local-LLM distribution channel:
//! `Ollama`-pulled blobs cached under
//! `~/.ollama/models/blobs/sha256-<hash>`.
//!
//! Each fixture is a 65 536-element slice extracted by
//! `tests/fixtures/ollama_reference/generate_ollama_fixture.py` from a
//! specific tensor in a specific `Ollama`-cached model, paired with the
//! `gguf` Python package's reference `BF16` dequant. Fixture file format
//! is byte-identical to `tests/fixtures/gguf_reference/*.bin` so the
//! parser below intentionally mirrors `cross_validation_gguf`'s
//! `parse_gguf_fixture` — same 16-byte header (discriminant, element
//! count, raw byte count, golden byte count) followed by raw quantised
//! block data and the golden `BF16` output.
//!
//! # Coverage
//!
//! - `llama3.2:1b` (`Q8_0`, `blk.0.attn_q.weight` slice). `Q8_0` is also
//!   exercised by the bartowski `SmolLM2-135M` fixture in
//!   `cross_validation_gguf`; this test proves the same kernel works on
//!   the Ollama distribution channel, not that `Q8_0` itself is correct
//!   (the latter is already covered).
//!
//! # Source-fixture refresh
//!
//! Run `python tests/fixtures/ollama_reference/generate_ollama_fixture.py`
//! after `ollama pull llama3.2:1b`. The script resolves the manifest
//! at `~/.ollama/models/manifests/registry.ollama.ai/library/<name>/<tag>`
//! to the blob path, slices the named tensor, and emits the fixture
//! `.bin` byte-identically across machines.

#![cfg(feature = "gguf")]
#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::wildcard_enum_match_arm
)]

use std::time::Instant;

use anamnesis::{GgufType, dequantize_gguf_to_bf16};

// ---------------------------------------------------------------------------
// Fixture parsing
// ---------------------------------------------------------------------------

/// Parsed fixture payload — same layout as `cross_validation_gguf`'s
/// `GgufFixture`. Kept local rather than shared via a `tests/common/`
/// module so the two cross-validation suites stay independently
/// readable; the duplication is intentional and minimal.
struct OllamaFixture {
    n_elements: usize,
    raw_data: Vec<u8>,
    expected_bf16: Vec<u8>,
}

fn read_u32_le(data: &[u8], offset: usize) -> u32 {
    // INDEX: caller passes offsets that are bounded by the 16-byte fixture
    // header; per-test fixture data is checked-in, so this is a test-side
    // assertion, not an attacker-controllable surface.
    let bytes: [u8; 4] = data[offset..offset + 4].try_into().unwrap();
    u32::from_le_bytes(bytes)
}

/// Maps a `ggml_type` discriminant to a [`GgufType`]. Currently only
/// `Q8_0` is exercised — extending coverage to other kernels (e.g.,
/// `Q4_K_M` from a future `ollama pull`) means adding one match arm
/// per new fixture.
fn gguf_type_from_disc(disc: u32) -> GgufType {
    match disc {
        8 => GgufType::Q8_0,
        other => panic!("unsupported Ollama fixture ggml_type discriminant: {other}"),
    }
}

fn parse_ollama_fixture(data: &[u8], expected_dtype: GgufType) -> OllamaFixture {
    let disc = read_u32_le(data, 0);
    let n_elements = read_u32_le(data, 4) as usize;
    let raw_data_len = read_u32_le(data, 8) as usize;
    let golden_len = read_u32_le(data, 12) as usize;

    let actual_dtype = gguf_type_from_disc(disc);
    assert_eq!(
        actual_dtype, expected_dtype,
        "fixture dtype mismatch: expected {expected_dtype:?}, got {actual_dtype:?} (disc={disc})"
    );

    let header_size = 16;
    let raw_start = header_size;
    let golden_start = raw_start + raw_data_len;

    OllamaFixture {
        n_elements,
        raw_data: data[raw_start..raw_start + raw_data_len].to_vec(),
        expected_bf16: data[golden_start..golden_start + golden_len].to_vec(),
    }
}

// ---------------------------------------------------------------------------
// BF16 comparison
// ---------------------------------------------------------------------------

/// Compare two `BF16` byte slices, allowing up to `max_ulp_diff` `ULP`
/// (unit in the last place) difference per element. Same shape as
/// `cross_validation_gguf::compare_bf16` — `NaN` counts as a match if
/// both sides are `NaN`, otherwise sub-`ULP` differences are tolerated
/// up to the supplied cap.
fn compare_bf16(actual: &[u8], expected: &[u8], max_ulp_diff: u16) -> (usize, u16) {
    assert_eq!(actual.len(), expected.len(), "output length mismatch");
    let mut mismatches = 0;
    let mut max_diff: u16 = 0;

    for (i, (a_pair, e_pair)) in actual
        .as_chunks::<2>()
        .0
        .iter()
        .zip(expected.as_chunks::<2>().0)
        .enumerate()
    {
        // INDEX: chunks_exact(2) guarantees exactly 2 bytes per pair
        let a_bits = u16::from_le_bytes([a_pair[0], a_pair[1]]);
        let e_bits = u16::from_le_bytes([e_pair[0], e_pair[1]]);

        // NaN equivalence — both NaN is a match.
        let a_is_nan = (a_bits & 0x7F80 == 0x7F80) && (a_bits & 0x007F != 0);
        let e_is_nan = (e_bits & 0x7F80 == 0x7F80) && (e_bits & 0x007F != 0);
        if a_is_nan && e_is_nan {
            continue;
        }
        if a_is_nan != e_is_nan {
            mismatches += 1;
            continue;
        }

        let diff = a_bits.abs_diff(e_bits);
        if diff > max_ulp_diff {
            mismatches += 1;
            if i < 5 {
                eprintln!(
                    "  element {i}: actual=0x{a_bits:04X}, expected=0x{e_bits:04X}, diff={diff} ULP"
                );
            }
        }
        if diff > max_diff {
            max_diff = diff;
        }
    }
    (mismatches, max_diff)
}

// ---------------------------------------------------------------------------
// Unified test runner
// ---------------------------------------------------------------------------

fn run_cross_validation(name: &str, data: &[u8], dtype: GgufType, max_ulp: u16) {
    let fixture = parse_ollama_fixture(data, dtype);
    let total = fixture.n_elements;

    let start = Instant::now();
    let actual = dequantize_gguf_to_bf16(&fixture.raw_data, dtype, fixture.n_elements)
        .expect("dequantization failed");
    let elapsed = start.elapsed();

    assert_eq!(
        actual.len(),
        fixture.expected_bf16.len(),
        "{name}: output length mismatch"
    );

    let (mismatches, max_diff) = compare_bf16(&actual, &fixture.expected_bf16, max_ulp);
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// `llama3.2:1b` ships `Q8_0` quantised weights (the default for the 1B
/// size tier in `Ollama` 0.24.0). The fixture slices 65 536 elements from
/// `blk.0.attn_q.weight` — the first attention-query weight, structurally
/// stable across `Llama`-arch releases. Bit-exact (`0 ULP`) against the
/// `gguf` Python reference, same contract as the bartowski `Q8_0` slice
/// in `cross_validation_gguf`.
#[test]
fn cross_validate_llama3_2_1b_q8_0_ollama() {
    run_cross_validation(
        "llama3.2:1b Q8_0 (Ollama)",
        include_bytes!("fixtures/ollama_reference/llama3_2_1b_q8_0.bin"),
        GgufType::Q8_0,
        0,
    );
}
