// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

//! Fuzz target: PyTorch `.pth` full front-matter parsing over arbitrary
//! bytes. Exercises the same reader-generic core as `fuzz_pth.rs`
//! (`inspect_pth_from_reader`) — the ZIP walk and the pickle VM — but
//! through the full-detail entry point that hands the caller every parsed
//! tensor name and shape directly, instead of reducing them to counts.
//! Must never panic/OOM/recurse-to-overflow — `Ok` or clean `Err`.

use std::io::Cursor;

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = anamnesis::parse_pth_front_matter_from_reader(Cursor::new(data));
});
