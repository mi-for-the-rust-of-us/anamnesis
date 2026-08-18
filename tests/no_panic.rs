// SPDX-License-Identifier: MIT OR Apache-2.0

//! Panic-freedom invariant (Phase 6.13 Step 3): **no public parse/inspect entry
//! point may panic or abort on any input** — a malformed or hostile artefact is
//! always a clean `Ok`/`Err`, never an unwinding panic (which under the shipped
//! `panic = "abort"` profile would be an uncatchable process kill, and which the
//! Phase 8 bindings must be able to surface as a catchable `PanicException`).
//!
//! Every re-exported parse/inspect entry point of all four formats is driven
//! here — the owned-bytes, reader, **and path/mmap** variants, each under both
//! `ParseLimits::default()` and a deliberately hostile tight budget (so the
//! `check_alloc` / bounded-reader / `Budget::charge_alloc` / count / ratio
//! rejection arithmetic is on the hook too) — over a battery of adversarial
//! inputs (synthetic malformed shapes + truncations and bit-flips of the
//! committed fixtures), each call wrapped in [`std::panic::catch_unwind`] and
//! asserted never to unwind. The suite runs under the default (debug) test
//! profile, so debug-only integer-overflow panics are in scope.
//!
//! Two boundaries are deliberate, not omissions:
//! - **`SIGBUS` is out of scope by nature.** The `parse*` path/mmap variants map
//!   the file; a *post-map* truncation faults with `SIGBUS` — an OS signal
//!   `catch_unwind` cannot catch. That hazard is addressed by *recommending the
//!   copy-based `parse_bytes` / `parse_*_from_reader` paths* for untrusted input
//!   (Step 1), not by this test. Mapping a complete file exercises the identical
//!   `parsed_*_from_backing` parse core the owned paths use.
//! - **Exhaustive structural exploration is the fuzzer's job.** This test pins
//!   the entry-point surface against a fixed corpus in stable CI; the coverage-
//!   guided `cargo fuzz` harness (incl. the owned-bytes `fuzz_*_bytes` targets)
//!   drives deep-nesting / rare-opcode discovery the fixed battery cannot.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::wildcard_enum_match_arm
)]

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;

use anamnesis::ParseLimits;

/// Runs `f` and asserts it returns normally (`Ok`/`Err`) rather than unwinding.
/// The produced value is irrelevant — only "did it panic?" matters.
fn assert_no_panic<T>(label: &str, f: impl FnOnce() -> T) {
    let outcome = catch_unwind(AssertUnwindSafe(f));
    assert!(outcome.is_ok(), "PANICKED on input `{label}`");
}

/// A deliberately hostile budget (16-byte single-allocation ceiling): forces
/// every parser onto its `ParseLimits` rejection branches, so the checked
/// arithmetic that backs them must reject without panicking.
fn tight() -> ParseLimits {
    ParseLimits::default().with_max_single_alloc(16)
}

/// Writes `bytes` to `path` for the path/mmap entry points. A failure here is a
/// fault in the *test harness*, not the code under test, so it may panic.
fn stage(path: &Path, bytes: &[u8]) {
    std::fs::write(path, bytes).expect("test harness: write temp input");
}

/// Committed (always-present) fixtures, truncated and bit-flipped below to build
/// realistic "almost-valid" adversarial inputs.
const FIXTURES: &[&str] = &[
    "tests/fixtures/safetensors_reference/fp8.safetensors",
    "tests/fixtures/safetensors_reference/gptq.safetensors",
    "tests/fixtures/safetensors_reference/awq.safetensors",
    "tests/fixtures/safetensors_reference/bnb_nf4.safetensors",
    "tests/fixtures/pth_reference/algzoo_rnn_small.pth",
    "tests/fixtures/npz_reference/gemma_scope_small.npz",
];

/// The adversarial input battery: `(label, bytes)`.
fn adversarial_inputs() -> Vec<(String, Vec<u8>)> {
    let mut inputs: Vec<(String, Vec<u8>)> = vec![
        ("empty".to_owned(), Vec::new()),
        ("one-zero".to_owned(), vec![0u8]),
        ("zeros-64".to_owned(), vec![0u8; 64]),
        ("ones-64".to_owned(), vec![0xFFu8; 64]),
        // 8 bytes of 0xFF: a safetensors `u64` header length of `u64::MAX`.
        ("u64-max-prefix".to_owned(), vec![0xFFu8; 8]),
        ("u64-max-prefix+junk".to_owned(), {
            let mut b = vec![0xFFu8; 8];
            b.extend_from_slice(b"junk-after-an-absurd-length");
            b
        }),
        // Small declared length + a partial JSON header.
        ("small-len+partial-json".to_owned(), {
            let mut b = 2u64.to_le_bytes().to_vec();
            b.extend_from_slice(b"{");
            b
        }),
        // ZIP local-file magic + junk (drives the `.npz` / `.pth` container).
        ("zip-magic+junk".to_owned(), {
            let mut b = b"PK\x03\x04".to_vec();
            b.extend_from_slice(&[0xABu8; 64]);
            b
        }),
        // GGUF magic + junk.
        ("gguf-magic+junk".to_owned(), {
            let mut b = b"GGUF".to_vec();
            b.extend_from_slice(&[0xCDu8; 64]);
            b
        }),
        // Raw pickle protocol bytes (legacy `.pth` detection path), then a MARK
        // flood — pushes the pickle VM toward its nesting/stack guards.
        ("pickle-proto".to_owned(), vec![0x80, 0x02, b'}', b'.']),
        ("pickle-proto+mark-flood".to_owned(), {
            let mut b = vec![0x80u8, 0x02];
            b.extend_from_slice(&[0x28u8; 8192]); // `(` = MARK
            b
        }),
        // Deeply repetitive bytes — stress recursion / nesting guards.
        ("open-brackets-8k".to_owned(), vec![b'['; 8192]),
        ("repeating-2byte-100k".to_owned(), {
            b"\x80\x02".iter().copied().cycle().take(100_000).collect()
        }),
    ];

    // A couple of deterministic pseudo-random blobs (no RNG dependency).
    for seed in [0x9E37_79B9u32, 0x1234_5678u32] {
        let blob: Vec<u8> = (0u32..512)
            .map(|i| (i.wrapping_mul(2_654_435_761).wrapping_add(seed) >> 13) as u8)
            .collect();
        inputs.push((format!("prng-{seed:08x}"), blob));
    }

    // Truncations and single-byte flips of each committed fixture: "almost
    // valid" inputs are the ones most likely to slip past a length check into a
    // downstream panic.
    for path in FIXTURES {
        let Ok(bytes) = std::fs::read(path) else {
            continue;
        };
        let len = bytes.len();
        let name = fixture_basename(path);
        for cut in [1usize, 4, 8, 16, len / 4, len / 2, len.saturating_sub(1)] {
            let cut = cut.min(len);
            inputs.push((format!("{name}@trunc{cut}"), bytes[..cut].to_vec()));
        }
        for pos in [0usize, len / 2, len.saturating_sub(1)] {
            if pos < len {
                let mut flipped = bytes.clone();
                flipped[pos] ^= 0xFF;
                inputs.push((format!("{name}@flip{pos}"), flipped));
            }
        }
    }

    inputs
}

/// The label prefix used for inputs derived from a fixture path.
fn fixture_basename(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

/// Guards against a silently-stale `FIXTURES` path: if a fixture moves, its
/// truncation/flip inputs vanish and the battery weakens *invisibly* — so assert
/// every fixture actually contributed inputs, loudly.
#[test]
fn battery_includes_every_fixture() {
    let inputs = adversarial_inputs();
    for fixture in FIXTURES {
        let base = fixture_basename(fixture);
        assert!(
            inputs.iter().any(|(label, _)| label.starts_with(base)),
            "no inputs derived from `{fixture}` — a FIXTURES path is stale, \
             silently weakening coverage"
        );
    }
}

// ---------------------------------------------------------------------------
// safetensors (always-on)
// ---------------------------------------------------------------------------

#[test]
fn safetensors_entry_points_never_panic() {
    use anamnesis::{
        parse, parse_bytes, parse_bytes_with_limits, parse_from_reader,
        parse_from_reader_with_limits, parse_safetensors_header,
        parse_safetensors_header_from_reader, parse_safetensors_header_from_reader_with_limits,
        parse_safetensors_header_with_limits, parse_with_limits,
    };
    use std::io::Cursor;

    let tmp = tempfile::NamedTempFile::new().expect("temp file");
    let path = tmp.path();
    let tight = tight();

    for (label, bytes) in adversarial_inputs() {
        // owned bytes
        assert_no_panic(&format!("st parse_bytes / {label}"), || {
            parse_bytes(bytes.clone())
        });
        assert_no_panic(
            &format!("st parse_bytes_with_limits[tight] / {label}"),
            || parse_bytes_with_limits(bytes.clone(), &tight),
        );
        // reader
        assert_no_panic(&format!("st parse_from_reader / {label}"), || {
            parse_from_reader(Cursor::new(bytes.clone()))
        });
        assert_no_panic(
            &format!("st parse_from_reader_with_limits[tight] / {label}"),
            || parse_from_reader_with_limits(Cursor::new(bytes.clone()), &tight),
        );
        // header — bytes + reader
        assert_no_panic(&format!("st parse_safetensors_header / {label}"), || {
            parse_safetensors_header(&bytes)
        });
        assert_no_panic(
            &format!("st parse_safetensors_header_with_limits[tight] / {label}"),
            || parse_safetensors_header_with_limits(&bytes, &tight),
        );
        assert_no_panic(&format!("st header_from_reader / {label}"), || {
            parse_safetensors_header_from_reader(Cursor::new(bytes.clone()))
        });
        assert_no_panic(
            &format!("st header_from_reader_with_limits[tight] / {label}"),
            || parse_safetensors_header_from_reader_with_limits(Cursor::new(bytes.clone()), &tight),
        );
        // path / mmap
        stage(path, &bytes);
        assert_no_panic(&format!("st parse(path) / {label}"), || parse(path));
        assert_no_panic(
            &format!("st parse_with_limits(path)[tight] / {label}"),
            || parse_with_limits(path, &tight),
        );
    }
}

// ---------------------------------------------------------------------------
// GGUF
// ---------------------------------------------------------------------------

#[cfg(feature = "gguf")]
#[test]
fn gguf_entry_points_never_panic() {
    use anamnesis::{
        inspect_gguf_from_reader, parse_gguf, parse_gguf_bytes, parse_gguf_bytes_with_limits,
        parse_gguf_from_reader, parse_gguf_from_reader_with_limits,
        parse_gguf_front_matter_from_reader, parse_gguf_front_matter_from_reader_with_limits,
        parse_gguf_with_limits,
    };
    use std::io::Cursor;

    let tmp = tempfile::NamedTempFile::new().expect("temp file");
    let path = tmp.path();
    let tight = tight();

    for (label, bytes) in adversarial_inputs() {
        assert_no_panic(&format!("gguf parse_gguf_bytes / {label}"), || {
            parse_gguf_bytes(bytes.clone())
        });
        assert_no_panic(
            &format!("gguf parse_gguf_bytes_with_limits[tight] / {label}"),
            || parse_gguf_bytes_with_limits(bytes.clone(), &tight),
        );
        assert_no_panic(&format!("gguf parse_gguf_from_reader / {label}"), || {
            parse_gguf_from_reader(Cursor::new(bytes.clone()))
        });
        assert_no_panic(
            &format!("gguf parse_gguf_from_reader_with_limits[tight] / {label}"),
            || parse_gguf_from_reader_with_limits(Cursor::new(bytes.clone()), &tight),
        );
        assert_no_panic(&format!("gguf inspect_from_reader / {label}"), || {
            inspect_gguf_from_reader(Cursor::new(bytes.clone()))
        });
        // The full-detail sibling of `inspect_gguf_from_reader` (v0.7.1): same
        // core, but every parsed string and array is handed back to the caller
        // rather than reduced to counts, so it gets its own coverage here.
        assert_no_panic(
            &format!("gguf parse_gguf_front_matter_from_reader / {label}"),
            || parse_gguf_front_matter_from_reader(Cursor::new(bytes.clone())),
        );
        assert_no_panic(
            &format!("gguf parse_gguf_front_matter_from_reader_with_limits[tight] / {label}"),
            || parse_gguf_front_matter_from_reader_with_limits(Cursor::new(bytes.clone()), &tight),
        );
        stage(path, &bytes);
        assert_no_panic(&format!("gguf parse_gguf(path) / {label}"), || {
            parse_gguf(path)
        });
        assert_no_panic(
            &format!("gguf parse_gguf_with_limits(path)[tight] / {label}"),
            || parse_gguf_with_limits(path, &tight),
        );
    }
}

// ---------------------------------------------------------------------------
// PyTorch .pth
// ---------------------------------------------------------------------------

#[cfg(feature = "pth")]
#[test]
fn pth_entry_points_never_panic() {
    use anamnesis::{
        inspect_pth_from_reader, parse_pth, parse_pth_bytes, parse_pth_bytes_with_limits,
        parse_pth_from_reader, parse_pth_from_reader_with_limits,
        parse_pth_front_matter_from_reader, parse_pth_front_matter_from_reader_with_limits,
        parse_pth_with_limits,
    };
    use std::io::Cursor;

    let tmp = tempfile::NamedTempFile::new().expect("temp file");
    let path = tmp.path();
    let tight = tight();

    for (label, bytes) in adversarial_inputs() {
        assert_no_panic(&format!("pth parse_pth_bytes / {label}"), || {
            parse_pth_bytes(bytes.clone())
        });
        assert_no_panic(
            &format!("pth parse_pth_bytes_with_limits[tight] / {label}"),
            || parse_pth_bytes_with_limits(bytes.clone(), &tight),
        );
        assert_no_panic(&format!("pth parse_pth_from_reader / {label}"), || {
            parse_pth_from_reader(Cursor::new(bytes.clone()))
        });
        assert_no_panic(
            &format!("pth parse_pth_from_reader_with_limits[tight] / {label}"),
            || parse_pth_from_reader_with_limits(Cursor::new(bytes.clone()), &tight),
        );
        assert_no_panic(&format!("pth inspect_from_reader / {label}"), || {
            inspect_pth_from_reader(Cursor::new(bytes.clone()))
        });
        // The full-detail sibling of `inspect_pth_from_reader` (v0.7.5): same
        // core, but every parsed tensor name and shape is handed back to the
        // caller rather than reduced to counts, so it gets its own coverage
        // here.
        assert_no_panic(
            &format!("pth parse_pth_front_matter_from_reader / {label}"),
            || parse_pth_front_matter_from_reader(Cursor::new(bytes.clone())),
        );
        assert_no_panic(
            &format!("pth parse_pth_front_matter_from_reader_with_limits[tight] / {label}"),
            || parse_pth_front_matter_from_reader_with_limits(Cursor::new(bytes.clone()), &tight),
        );
        stage(path, &bytes);
        assert_no_panic(&format!("pth parse_pth(path) / {label}"), || {
            parse_pth(path)
        });
        assert_no_panic(
            &format!("pth parse_pth_with_limits(path)[tight] / {label}"),
            || parse_pth_with_limits(path, &tight),
        );
    }
}

// ---------------------------------------------------------------------------
// NPZ (path-based parse + reader/path inspect)
// ---------------------------------------------------------------------------

#[cfg(feature = "npz")]
#[test]
fn npz_entry_points_never_panic() {
    use anamnesis::{inspect_npz, inspect_npz_from_reader, parse_npz, parse_npz_with_limits};
    use std::io::Cursor;

    let tmp = tempfile::NamedTempFile::new().expect("temp file");
    let path = tmp.path();
    let default = ParseLimits::default();
    let tight = tight();

    for (label, bytes) in adversarial_inputs() {
        assert_no_panic(&format!("npz inspect_from_reader / {label}"), || {
            inspect_npz_from_reader(Cursor::new(bytes.clone()))
        });
        stage(path, &bytes);
        assert_no_panic(&format!("npz inspect_npz(path) / {label}"), || {
            inspect_npz(path)
        });
        assert_no_panic(&format!("npz parse_npz(path) / {label}"), || {
            parse_npz(path)
        });
        assert_no_panic(
            &format!("npz parse_npz_with_limits(path)[default] / {label}"),
            || parse_npz_with_limits(path, &default),
        );
        assert_no_panic(
            &format!("npz parse_npz_with_limits(path)[tight] / {label}"),
            || parse_npz_with_limits(path, &tight),
        );
    }
}
