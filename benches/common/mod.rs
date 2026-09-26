// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fixture synthesis shared by the bench targets (`ab`, `convert`, `dequant`,
//! `parsing`).
//!
//! A directory module on purpose: Cargo auto-discovers `benches/*.rs` (and
//! `benches/*/main.rs`) as bench targets, but not `benches/common/mod.rs`, so
//! this file is compiled only into the targets that declare `mod common;`.
//!
//! **Why the filler is shared rather than copied.** Every harness must fill its
//! fixtures with byte-for-byte the same pattern. `ab.rs` and `dequant.rs` are
//! quoted side by side (see the `F16Out` cost table), and `CodSpeed` compares each
//! id against its own history, so a filler that drifted in one copy would give
//! different quantised values, different codebook indices and different
//! denormal counts: absolute times would stop being comparable while still
//! looking as though they compared. One definition makes that drift impossible.

// Each bench target compiles this whole module but calls only what it needs
// (`convert.rs` never calls `synth_bytes`, for instance), so a target can see
// a helper as unused. That is the per-target view, not a dead helper.
#![allow(dead_code)]

/// Fills a buffer with deterministic non-zero bytes via a Knuth
/// multiplicative hash on the index. Avoids the all-zero pathology
/// that some quantisation kernels short-circuit through their
/// fast-path branches, and keeps bit patterns stable across runs so a
/// `CodSpeed` comparison is not perturbed by fixture churn.
pub fn fill_deterministic(buf: &mut [u8]) {
    for (i, b) in buf.iter_mut().enumerate() {
        // CAST: usize → u8 by deliberate truncation to a byte pattern, not a
        // value-carrying conversion.
        #[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
        {
            *b = (i.wrapping_mul(2_654_435_761) & 0xFF) as u8;
        }
    }
}

/// Allocates `n` bytes filled by [`fill_deterministic`].
pub fn synth_bytes(n: usize) -> Vec<u8> {
    let mut v = vec![0u8; n];
    fill_deterministic(&mut v);
    v
}
