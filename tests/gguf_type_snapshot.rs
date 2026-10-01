// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `GGUF` parser's type table, checked against upstream's.
//!
//! `tests/fixtures/ggml_types.txt` is the list of live `GGML_TYPE_*` entries in
//! llama.cpp's `ggml.h` at a pinned commit, produced by
//! `scripts/ggml-types.sh`. This test checks, offline and through the public
//! API, that the parser accepts **exactly** those discriminants, under those
//! names, and refuses every other one. The weekly
//! `.github/workflows/ggml-drift.yml` job compares the snapshot itself with
//! upstream `master`, so together the two close the loop issue #15 found open:
//! upstream added `NVFP4`, `Q1_0` and `Q2_0`, and nothing noticed until a
//! user's file was refused.

#![cfg(feature = "gguf")]
#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::wildcard_enum_match_arm
)]

mod common;

use std::collections::BTreeMap;

use anamnesis::AnamnesisError;

use common::gguf::{RawTensorInfo, raw_gguf};

/// The snapshot, as `discriminant → name without the GGML_TYPE_ prefix`.
fn snapshot() -> BTreeMap<u32, String> {
    let text = include_str!("fixtures/ggml_types.txt");
    let mut types = BTreeMap::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (name, disc) = line
            .split_once(" = ")
            .unwrap_or_else(|| panic!("malformed snapshot line: {line}"));
        let name = name
            .strip_prefix("GGML_TYPE_")
            .unwrap_or_else(|| panic!("snapshot name without prefix: {name}"));
        let disc: u32 = disc.parse().unwrap();
        assert!(
            types.insert(disc, name.to_owned()).is_none(),
            "discriminant {disc} appears twice in the snapshot"
        );
    }
    types
}

/// A one-tensor `GGUF` of `ggml_type`, 256 elements (a multiple of every
/// block size), over enough zero bytes for the widest type at that count.
fn one_tensor(ggml_type: u32) -> Vec<u8> {
    let dims = [256_u64];
    let info = RawTensorInfo {
        name: "t",
        dims: &dims,
        ggml_type,
        offset: 0,
    };
    raw_gguf(&[], &[info], 32, &[0u8; 256 * 8])
}

#[test]
fn snapshot_is_pinned_to_a_commit() {
    let text = include_str!("fixtures/ggml_types.txt");
    let commit = text
        .lines()
        .find_map(|l| l.strip_prefix("# Commit:"))
        .expect("the snapshot names the upstream commit it was taken from")
        .trim();
    assert_eq!(
        commit.len(),
        40,
        "a full commit hash, not a branch: {commit}"
    );
    assert!(commit.bytes().all(|b| b.is_ascii_hexdigit()), "{commit}");
}

#[test]
fn parser_accepts_exactly_the_upstream_types() {
    let upstream = snapshot();
    assert!(!upstream.is_empty());

    for disc in 0..=255_u32 {
        let result = anamnesis::parse_gguf_bytes(one_tensor(disc));
        match upstream.get(&disc) {
            Some(name) => {
                let parsed = result.unwrap_or_else(|e| {
                    panic!("upstream type {name} ({disc}) is refused by the parser: {e}")
                });
                let dtype = parsed.tensor_info()[0].dtype;
                assert_eq!(&dtype.to_string(), name, "discriminant {disc}");
            }
            None => match result {
                Err(AnamnesisError::Unsupported { detail, .. }) => assert_eq!(
                    detail,
                    format!("unknown ggml_type discriminant {disc}"),
                    "discriminant {disc}"
                ),
                Err(other) => panic!("discriminant {disc}: unexpected error {other:?}"),
                Ok(_) => panic!("discriminant {disc} is not an upstream type but parses"),
            },
        }
    }
}
