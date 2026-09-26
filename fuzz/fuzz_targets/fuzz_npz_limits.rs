// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

//! Fuzz target: `parse_npz_with_limits` over arbitrary bytes **under a
//! `ParseLimits` derived from the input**, so the fuzzer co-explores
//! `(malformed file × tightened limits)`. NPZ exercises all four axes — the
//! single-allocation cap, the cumulative-byte `Budget` (`checked_add`), the
//! item-count gate, and the decompression-ratio cap (`checked_mul`). The
//! limit-enforcement branches must never panic/OOM — `Ok` or clean `Err`.

use std::io::Write;

#[path = "../common/mod.rs"]
mod common;

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }
    let (prefix, body) = data.split_at(8);
    let limits = common::derive_limits(prefix);

    let mut f = match tempfile::NamedTempFile::new() {
        Ok(f) => f,
        Err(_) => return,
    };
    if f.write_all(body).and_then(|()| f.flush()).is_err() {
        return;
    }
    let _ = anamnesis::parse_npz_with_limits(f.path(), &limits);
});
