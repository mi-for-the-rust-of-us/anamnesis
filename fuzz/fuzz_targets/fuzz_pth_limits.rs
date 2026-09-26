// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

//! Fuzz target: `parse_pth_with_limits` over arbitrary bytes **under a
//! `ParseLimits` derived from the input**. Exercises the `.pth`
//! limit-enforcement branches: the `data.pkl` single-allocation cap and the
//! pickle-VM cumulative-byte `Budget` (`checked_add`) charged on each owned
//! string/bytes payload. Must never panic/OOM — `Ok` or clean `Err`.

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
    if let Ok(parsed) = anamnesis::parse_pth_with_limits(f.path(), &limits) {
        let _ = parsed.tensors();
    }
});
