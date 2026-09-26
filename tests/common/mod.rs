// SPDX-License-Identifier: MIT OR Apache-2.0

//! Helpers shared by the integration-test crates under `tests/`.
//!
//! Cargo compiles every `tests/*.rs` file as its own crate but does **not**
//! treat a subdirectory as a test target, so `tests/common/mod.rs` is the
//! standard place for code several of those crates need. A consumer opts in
//! with `mod common;` and imports what it uses, e.g.
//! `use common::bf16::compare_bf16;`.
//!
//! Only code that was byte-for-byte identical (or identical modulo comments)
//! across its copies lives here. Where copies genuinely diverged, the variant
//! is either kept local or selected by an explicit, named parameter (see
//! [`bf16::SignedZero`]), never folded silently into one behaviour.
//!
//! Submodules are grouped by concern so a reader can find a helper by what it
//! does rather than by which test first needed it:
//!
//! - [`fixture`]: little-endian readers for the binary fixture containers.
//! - [`bf16`]: `BF16` golden comparison and synthesis.
//! - [`bnb`]: the `AMNB` v2 `BnB` fixture container.
//! - `gguf` (requires the `gguf` feature): `ggml_type` mapping and the
//!   `BF16` golden check shared by the `GGUF` and `Ollama` suites.
//! - [`builders`]: in-memory `safetensors` / `NPZ` builders and temp files.
//! - [`heap`]: `dhat` serialisation and deterministic synthesis for the
//!   `peak_heap_*` binaries.
//! - [`bench`]: timing and fixture helpers for the `bench_*_adhoc` harnesses.

// Each test crate compiles this whole module but calls only the helpers it
// needs, so any one crate sees most of them as unused. `dead_code` here is
// that per-crate view, not a genuinely dead helper; the accepted idiom for
// `tests/common` is to allow it at this one site rather than scatter
// `#[allow(dead_code)]` over every item.
#![allow(dead_code)]
// The same test-code allowances every `tests/*.rs` crate declares for itself.
// Declared here too because the consumers' own lists differ (not every
// consumer allows `as_conversions` or `cast_possible_truncation`, for
// instance) and a shared helper must build under each of them.
#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::same_item_push
)]

pub mod bench;
pub mod bf16;
pub mod bnb;
pub mod builders;
pub mod fixture;
#[cfg(feature = "gguf")]
pub mod gguf;
pub mod heap;
