// SPDX-License-Identifier: MIT OR Apache-2.0

//! Compile-time guards for the auto-trait bounds the parallel dispatch relies on.
//!
//! `src/parallel.rs`'s `map_indexed` shares `&self` across scoped worker threads,
//! which requires the shared parsed-model types to be `Sync`. That holds today
//! purely because every field happens to be `Sync` — no marker impl is written
//! anywhere, and none should be (`CONVENTIONS.md` "When Parallelizing Work":
//! *never reach for manual `Send`/`Sync` impls*).
//!
//! The risk is silent regression by omission: adding a `Cell`, `RefCell`, `Rc`,
//! or raw pointer field to `ParsedModel` or `ParsedGguf` would drop `Sync`, and
//! the failure would surface as an inscrutable closure-bound error deep inside
//! the `thread::scope` call, far from the field that caused it. These assertions
//! turn that into a one-line failure naming the type.
//!
//! `Send` is asserted alongside `Sync` because the `Send` requirement is what
//! lets the produced results move back to the calling thread, and both are
//! equally easy to lose.
//!
//! These are **compile-time** assertions: if the crate builds, the contract
//! holds. The `#[test]` wrapper exists only so the file is a runnable target and
//! shows up in the suite.

#![allow(clippy::panic, clippy::unwrap_used, clippy::expect_used)]

/// Fails to compile unless `T: Send + Sync`.
const fn assert_send_sync<T: Send + Sync>() {}

/// The safetensors model shared across workers by
/// `ParsedModel::dequantize_all` (Phase 7, v0.7.0).
#[test]
fn parsed_model_is_send_and_sync() {
    assert_send_sync::<anamnesis::ParsedModel>();
}

/// The `GGUF` model shared across workers by `convert::read_gguf`
/// (Phase 7.2, v0.7.2). The ROADMAP long recorded `ParsedGguf: Sync` as a
/// *blocker* for this parallelisation; it was in fact already satisfied — every
/// field (`Backing`, `u32`, the metadata `HashMap`, the tensor-info `Vec`) is
/// `Sync`, and `memmap2::Mmap` is `Send + Sync`. This assertion is what keeps
/// that true.
#[cfg(feature = "gguf")]
#[test]
fn parsed_gguf_is_send_and_sync() {
    assert_send_sync::<anamnesis::ParsedGguf>();
}

/// The tensor **views** are what actually cross into the workers (the dispatch
/// is handed a `&[GgufTensor<'_>]`), so their auto-traits matter as much as the
/// owner's.
#[cfg(feature = "gguf")]
#[test]
fn gguf_tensor_view_is_send_and_sync() {
    assert_send_sync::<anamnesis::GgufTensor<'static>>();
}

/// The option types travel by value into the library entry points and will
/// cross the `PyO3` boundary at Phase 8; losing `Send` here would be a silent
/// ergonomics regression for any caller driving conversions from a worker pool.
#[test]
fn option_types_are_send_and_sync() {
    assert_send_sync::<anamnesis::RememberOptions>();
    assert_send_sync::<anamnesis::ConvertOptions>();
    // Asserted in its own right, not merely through the options structs that
    // hold one: a `CancelToken` exists to be cloned to another thread and set
    // there, so if it ever stopped being `Send + Sync` the failure should name
    // it rather than surfacing as an inscrutable bound on an options type.
    assert_send_sync::<anamnesis::CancelToken>();
}
