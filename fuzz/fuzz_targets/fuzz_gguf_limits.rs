// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

//! Fuzz target: `parse_gguf_with_limits` over arbitrary bytes **under a
//! `ParseLimits` derived from the input**. Exercises the GGUF
//! limit-enforcement branches: the single-allocation cap + cumulative-byte
//! `Budget` (`checked_add`) on every variable-length read, the scalar
//! metadata-array charge, and the tensor / KV item-count gate. Must never
//! panic/OOM — `Ok` or clean `Err`.

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
    if let Ok(parsed) = anamnesis::parse_gguf_with_limits(f.path(), &limits) {
        let _ = parsed.tensors().count();
    }
});
