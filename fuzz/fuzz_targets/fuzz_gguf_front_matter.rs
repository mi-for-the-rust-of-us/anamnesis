// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

//! Fuzz target: GGUF full front-matter parsing over arbitrary bytes.
//! Exercises the same reader-generic core as `fuzz_gguf.rs`
//! (`inspect_gguf_from_reader`), but through the full-detail entry point
//! that hands the caller every parsed tensor name and metadata value
//! directly — higher exposure than the aggregate summary. Must never
//! panic/OOM — `Ok` or clean `Err`.

use std::io::Cursor;

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = anamnesis::parse_gguf_front_matter_from_reader(Cursor::new(data));
});
