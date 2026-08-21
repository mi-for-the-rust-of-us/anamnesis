// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

//! Fuzz target: magic-byte format detection over arbitrary caller bytes
//! ([`detect_format_from_bytes`], Phase 7.6 item 6).
//!
//! Detection is not a cheap byte comparison for every format: `.npz` and
//! `.pth` share the `ZIP` magic, so telling them apart means **walking the
//! central directory** — real parsing of hostile input, performed *before* any
//! other validation and before a caller has chosen a parser. That makes it its
//! own attack surface rather than a preamble to one, so it gets its own target.
//!
//! Must never panic / abort / OOM — only `Ok(Format)` or a clean `Err`.

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = anamnesis::detect_format_from_bytes(data);
});
