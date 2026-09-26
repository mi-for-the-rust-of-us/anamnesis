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

mod common;

use anamnesis::GgufType;

use common::fixture::read_u32_le;
use common::gguf::{check_bf16_against_golden, gguf_type_from_disc};

// ---------------------------------------------------------------------------
// Fixture parsing
// ---------------------------------------------------------------------------

/// Parsed fixture payload: the `BF16`-only subset of `cross_validation_gguf`'s
/// `GgufFixture`. The container differs (this one predates the v2 `AMNG`
/// header), so the struct and its parser stay local; the type mapping and the
/// `BF16` check are shared through `tests/common/gguf.rs`.
struct OllamaFixture {
    n_elements: usize,
    raw_data: Vec<u8>,
    expected_bf16: Vec<u8>,
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
// Unified test runner
// ---------------------------------------------------------------------------

fn run_cross_validation(name: &str, data: &[u8], dtype: GgufType, max_ulp: u16) {
    let fixture = parse_ollama_fixture(data, dtype);

    check_bf16_against_golden(
        name,
        &fixture.raw_data,
        dtype,
        fixture.n_elements,
        &fixture.expected_bf16,
        max_ulp,
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
