// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

//! Fuzz target: safetensors under a `ParseLimits` derived from the input, from
//! the header all the way through dequantisation.
//!
//! Drives the slice header parse and the owned-bytes model parse under the
//! same budget, then, when the bytes parse, `inspect` and `remember_to_bytes`.
//! That last call runs every dequant kernel the header selects (`FP8`, `GPTQ`,
//! `AWQ`, `BnB`) on attacker-shaped tensors, directly on a parsed model
//! (`fuzz_convert_bytes` reaches the same kernels through the `convert` hub).
//! Every call must return `Ok` or a clean `Err`, never panic or OOM.

#[path = "../common/mod.rs"]
mod common;

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }
    let (prefix, body) = data.split_at(8);
    let limits = common::derive_limits(prefix);

    let _ = anamnesis::parse_safetensors_header_with_limits(body, &limits);
    if let Ok(model) = anamnesis::parse_bytes_with_limits(body.to_vec(), &limits) {
        let _ = model.inspect();
        let _ = model.remember_to_bytes(anamnesis::TargetDtype::BF16);
    }
});
