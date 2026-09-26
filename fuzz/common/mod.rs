// SPDX-License-Identifier: MIT OR Apache-2.0

//! Helpers shared by the fuzz targets, included with
//! `#[path = "../common/mod.rs"] mod common;`. A directory module outside
//! `fuzz_targets/`, so it can never be mistaken for a target.

/// Derives all four `ParseLimits` axes from the first 8 bytes of the input
/// (four little-endian `u16`s), so the fuzzer co-explores
/// `(malformed file × tightened limits)`. `0xFFFF` maps to unbounded so the
/// accept path stays reachable; small values keep the reject branches hot.
///
/// The decompression-ratio axis is set for every target even where the path
/// under test reads no `DEFLATE` entry: it is then simply inert, and one
/// mapping for all targets is what keeps them from drifting apart (they were
/// three hand-kept copies before v0.7.8).
pub fn derive_limits(prefix: &[u8]) -> anamnesis::ParseLimits {
    let axis = |i: usize| -> u64 {
        let lo = prefix.get(i * 2).copied().unwrap_or(0xFF);
        let hi = prefix.get(i * 2 + 1).copied().unwrap_or(0xFF);
        match u16::from_le_bytes([lo, hi]) {
            u16::MAX => u64::MAX,
            v => u64::from(v),
        }
    };
    anamnesis::ParseLimits::default()
        .with_max_single_alloc(axis(0))
        .with_max_total_bytes(axis(1))
        .with_max_item_count(axis(2))
        .with_max_decompression_ratio(axis(3))
}
