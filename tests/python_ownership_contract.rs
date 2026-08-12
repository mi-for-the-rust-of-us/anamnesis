// SPDX-License-Identifier: MIT OR Apache-2.0

//! Data-ownership contract (Phase 6.13 Step 4): **owned extraction outlives a
//! dropped `Parsed*`.**
//!
//! The Phase 8 `PyO3` bindings return *owned* `NumPy` arrays (see
//! `docs/python-interop.md`): a returned array must never be a bare view into a
//! `Backing` the owning `Parsed*` can drop, or `del model` becomes a
//! use-after-free reachable from pure Python. The binding takes ownership before
//! building the array — `Cow::into_owned()` for the GGUF / `.pth` zero-copy
//! slices, already-owned `Vec<u8>` for npz and `remember_to_bytes`.
//!
//! Each test below performs that exact extraction inside an inner scope, lets the
//! `Parsed*` **drop at the end of the scope**, and then uses the collected data.
//! Two guarantees combine: *compile-time*, the borrow checker would reject a
//! result that kept a `Cow::Borrowed` / `&'a` field (it could not outlive the
//! `Parsed*`), so the inner scope only compiles because the extraction takes
//! ownership; *run-time*, using the data after the drop confirms the owned copy
//! is genuinely independent of the owner. This is the `PyO3`-free foundation the
//! binding's owned-copy contract rests on.

#![allow(clippy::expect_used, clippy::indexing_slicing)]

use std::fs;

// ---------------------------------------------------------------------------
// safetensors (always-on) — remember_to_bytes is owned
// ---------------------------------------------------------------------------

#[test]
fn safetensors_owned_dequant_outlives_model() {
    use anamnesis::{TargetDtype, parse_bytes};

    const ST_FP8: &str = "tests/fixtures/safetensors_reference/fp8.safetensors";

    // Dequantise to owned BF16 bytes, then drop the ParsedModel.
    let bf16: Vec<u8> = {
        let model = parse_bytes(fs::read(ST_FP8).expect("read fixture")).expect("parse");
        model
            .remember_to_bytes(TargetDtype::BF16)
            .expect("remember to bytes")
        // `model` (and its Backing) dropped here.
    };

    assert!(
        !bf16.is_empty(),
        "owned dequant bytes survive the model drop"
    );
}

// ---------------------------------------------------------------------------
// GGUF — tensors() borrows the Backing; into_owned() severs it
// ---------------------------------------------------------------------------

#[cfg(feature = "gguf")]
#[test]
fn gguf_owned_tensors_outlive_parsed() {
    use anamnesis::{GgufType, GgufWriteTensor, parse_gguf_bytes, write_gguf_to_writer};
    use std::collections::HashMap;
    use std::io::Cursor;

    // Self-contained fixture via the crate's own writer (no on-disk dependency).
    let gguf_bytes: Vec<u8> = {
        let f32_bytes: Vec<u8> = (0u32..6).flat_map(u32::to_le_bytes).collect();
        let shape = [2usize, 3];
        let tensors = [GgufWriteTensor {
            name: "w",
            shape: &shape,
            dtype: GgufType::F32,
            data: &f32_bytes,
        }];
        let mut cursor = Cursor::new(Vec::new());
        write_gguf_to_writer(&mut cursor, &tensors, &HashMap::new()).expect("write gguf");
        cursor.into_inner()
    };

    // Collect fully-owned (name, shape, data) tuples, then drop the ParsedGguf.
    // `GgufTensor` borrows the Backing (`name: &'a str`, `data: Cow<'a, [u8]>`),
    // so this only compiles because every field is taken by value / into_owned.
    let owned: Vec<(String, Vec<usize>, Vec<u8>)> = {
        let parsed = parse_gguf_bytes(gguf_bytes).expect("parse gguf bytes");
        parsed
            .tensors()
            .map(|t| (t.name.to_owned(), t.shape.to_vec(), t.data.into_owned()))
            .collect()
        // `parsed` (and its Backing) dropped here.
    };

    assert!(!owned.is_empty(), "tensors survive the ParsedGguf drop");
    assert!(owned.iter().all(|(_, _, data)| !data.is_empty()));
    assert_eq!(owned[0].0, "w");
}

// ---------------------------------------------------------------------------
// PyTorch .pth — tensors() data is Cow; into_owned() severs it
// ---------------------------------------------------------------------------

#[cfg(feature = "pth")]
#[test]
fn pth_owned_tensors_outlive_parsed() {
    use anamnesis::parse_pth_bytes;

    const PTH: &str = "tests/fixtures/pth_reference/algzoo_rnn_small.pth";

    // `PthTensor::name` / `shape` are already owned; only `data` is a `Cow`.
    let owned: Vec<(String, Vec<u8>)> = {
        let parsed =
            parse_pth_bytes(fs::read(PTH).expect("read fixture")).expect("parse pth bytes");
        parsed
            .tensors()
            .expect("tensors")
            .into_iter()
            .map(|t| (t.name, t.data.into_owned()))
            .collect()
        // `parsed` (and its Backing) dropped here.
    };

    assert!(!owned.is_empty(), "tensors survive the ParsedPth drop");
    assert!(owned.iter().all(|(_, data)| !data.is_empty()));
}

// ---------------------------------------------------------------------------
// NPZ — owned by construction (parse_npz retains no Backing)
// ---------------------------------------------------------------------------

#[cfg(feature = "npz")]
#[test]
fn npz_tensors_are_owned_by_construction() {
    use anamnesis::parse_npz;

    const NPZ: &str = "tests/fixtures/npz_reference/gemma_scope_small.npz";

    // `parse_npz` returns owned `NpzTensor`s and retains no `Backing`, so there
    // is nothing to outlive — the data is already detached at the API boundary.
    let data: Vec<Vec<u8>> = parse_npz(NPZ)
        .expect("parse npz")
        .into_values()
        .map(|t| t.data)
        .collect();

    assert!(!data.is_empty(), "npz yields owned tensor data");
    assert!(data.iter().all(|d| !d.is_empty()));
}
