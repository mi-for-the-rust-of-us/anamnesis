// SPDX-License-Identifier: MIT OR Apache-2.0

//! Guards the panic-strategy build contract the (future) `PyO3` bindings depend
//! on, so it cannot silently regress before Phase 8 wires up the `cdylib`:
//!
//! - `[profile.release]` builds **abort** — the CLI / library fail-closed `DoS`
//!   posture (a reachable panic on untrusted input becomes a clean process exit,
//!   the severity assumption the hardening work argues from).
//! - `[profile.python]` (maturin `--profile python`) builds **unwind**, so `PyO3`
//!   turns a Rust panic into a catchable Python `PanicException` rather than an
//!   uncatchable process abort — the bindings' "never a dead worker" pledge.
//!
//! See `docs/python-interop.md` and the `[profile.*]` comments in `Cargo.toml`.

#![allow(clippy::panic)]

/// The crate manifest, baked in at compile time.
const CARGO_TOML: &str = include_str!("../Cargo.toml");

/// Returns the value of `key` set inside the `[header]` profile section of a
/// `Cargo.toml`, ignoring comment lines (which quote settings like
/// `panic = "abort"` as prose) and stopping at the next section header.
///
/// `header` is matched as a whole trimmed line, so a `#`-comment that mentions
/// `[profile.python]` in passing does not start a section.
fn profile_value<'a>(toml: &'a str, header: &str, key: &str) -> Option<&'a str> {
    let mut in_section = false;
    for line in toml.lines() {
        let trimmed = line.trim();
        if !in_section {
            if trimmed == header {
                in_section = true;
            }
            continue;
        }
        if trimmed.starts_with('[') {
            break; // next section
        }
        if trimmed.starts_with('#') {
            continue; // comment, e.g. prose quoting `panic = "abort"`
        }
        if let Some((k, v)) = trimmed.split_once('=')
            && k.trim() == key
        {
            return Some(v.trim().trim_matches('"'));
        }
    }
    None
}

#[test]
fn release_profile_aborts_on_panic() {
    assert_eq!(
        profile_value(CARGO_TOML, "[profile.release]", "panic"),
        Some("abort"),
        "[profile.release] must set panic = \"abort\" (the fail-closed DoS posture)"
    );
}

#[test]
fn python_profile_unwinds_on_panic() {
    assert_eq!(
        profile_value(CARGO_TOML, "[profile.python]", "panic"),
        Some("unwind"),
        "[profile.python] must set panic = \"unwind\" so PyO3 can surface a panic \
         as a catchable PanicException instead of aborting the worker"
    );
    assert_eq!(
        profile_value(CARGO_TOML, "[profile.python]", "inherits"),
        Some("release"),
        "[profile.python] must inherit `release` (same codegen as the shipped wheel)"
    );
}

#[test]
fn profile_value_ignores_comment_prose() {
    // The `[profile.release]` comment block quotes `panic = "abort"` and mentions
    // `[profile.python]`; the parser must read the real setting, not the prose.
    let sample = "\
# panic = \"unwind\" is mentioned here in a comment about [profile.python]
[profile.release]
# another comment: panic = \"unwind\"
panic = \"abort\"

[profile.python]
inherits = \"release\"
panic = \"unwind\"
";
    assert_eq!(
        profile_value(sample, "[profile.release]", "panic"),
        Some("abort")
    );
    assert_eq!(
        profile_value(sample, "[profile.python]", "panic"),
        Some("unwind")
    );
}
