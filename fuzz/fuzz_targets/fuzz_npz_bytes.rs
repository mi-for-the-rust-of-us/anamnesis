// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

//! Fuzz target: the copy-based `NPZ` full-parse entry point
//! [`parse_npz_bytes`] (Phase 7.6 item 3) — the recommended path for untrusted
//! input. Drives the vendored `ZIP` central-directory walk, `DEFLATE` inflate,
//! the `NPY` header parser, and the Fortran-order transposition over owned
//! bytes (no mmap). Must never panic / abort / OOM — only `Ok` or a clean
//! `Err`.
//!
//! `NPZ` was the last format to get a byte-form parser, which is why this
//! target arrives three releases after its `safetensors` / `GGUF` / `.pth`
//! siblings rather than alongside them.

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = anamnesis::parse_npz_bytes(data.to_vec());
});
