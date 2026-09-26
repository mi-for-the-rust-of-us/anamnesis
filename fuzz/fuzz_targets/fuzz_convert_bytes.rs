// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

//! Fuzz target: `convert_bytes`, the whole pipeline on untrusted bytes.
//!
//! Format detection, the matching parser, dequantisation where the input is
//! quantised, the in-memory hub, and one of the three writers, all under a
//! `ParseLimits` derived from the input. Byte 8 picks the target
//! (`safetensors`, `gguf`, `bnb-nf4`), so every writer sees hostile hubs.
//! Before v0.7.8 this path was covered only by the fixed corpus in
//! `tests/no_panic.rs`. Must return `Ok` or a clean `Err`, never panic or OOM.

#[path = "../common/mod.rs"]
mod common;

use libfuzzer_sys::fuzz_target;

const TARGETS: [&str; 3] = ["safetensors", "gguf", "bnb-nf4"];

fuzz_target!(|data: &[u8]| {
    if data.len() < 9 {
        return;
    }
    let (prefix, body) = data.split_at(9);
    let limits = common::derive_limits(&prefix[..8]);
    let Ok(target) =
        anamnesis::ConvertTarget::parse(TARGETS[usize::from(prefix[8]) % TARGETS.len()])
    else {
        return;
    };
    let options = anamnesis::ConvertOptions::new().with_limits(limits);
    let _ = anamnesis::convert_bytes(body, target, &options);
});
