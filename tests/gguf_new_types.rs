// SPDX-License-Identifier: MIT OR Apache-2.0

//! Issue #15: a `GGUF` holding `ggml_type` 40 (`NVFP4`), 41 (`Q1_0`) or 42
//! (`Q2_0`) was refused outright, because the parser's type table stopped at
//! 39 (`MXFP4`). The rejection came while reading the tensor-info table, so
//! not even `inspect` could run on such a file.
//!
//! The fixture is the issue's reproducer: one one-dimensional tensor and no
//! metadata. Discriminant 39 is the control, and differs from the others only
//! in that `u32`. One deviation from the issue: it used 64 elements, which is
//! not a whole `Q1_0` block (128), so here every tensor has 256 elements,
//! a multiple of every block size. The 64-element `Q1_0` case is pinned
//! separately, because it must now fail for the right reason.

#![cfg(feature = "gguf")]
#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::wildcard_enum_match_arm
)]

mod common;

use std::io::Cursor;

use anamnesis::{AnamnesisError, GgufType};

use common::gguf::{RawTensorInfo, raw_gguf};

/// Elements per tensor: a multiple of every block size (32, 64, 128, 256).
const N: u64 = 256;

/// The issue's 160-byte reproducer, generalised to any element count.
fn reproducer(ggml_type: u32, elements: u64, data_len: usize) -> Vec<u8> {
    let dims = [elements];
    let info = RawTensorInfo {
        name: "blk.0.attn_q.weight",
        dims: &dims,
        ggml_type,
        offset: 0,
    };
    raw_gguf(&[], &[info], 32, &vec![0u8; data_len])
}

/// `(discriminant, variant, name, block size, bytes per block)` for the
/// control and the three types issue #15 added. Layouts from llama.cpp
/// `ggml-common.h` at `37b53fd`.
const CASES: [(u32, GgufType, &str, u64, u64); 4] = [
    (39, GgufType::MXFP4, "MXFP4", 32, 17),
    (40, GgufType::NVFP4, "NVFP4", 64, 36),
    (41, GgufType::Q1_0, "Q1_0", 128, 18),
    (42, GgufType::Q2_0, "Q2_0", 64, 18),
];

#[test]
fn every_entry_point_parses_the_new_types() {
    for (disc, dtype, name, block, type_size) in CASES {
        let byte_len = N / block * type_size;
        let bytes = reproducer(disc, N, usize::try_from(byte_len).unwrap());

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(format!("t{disc}.gguf"));
        std::fs::write(&path, &bytes).unwrap();

        let parsed = [
            anamnesis::parse_gguf(&path).unwrap(),
            anamnesis::parse_gguf_bytes(bytes.clone()).unwrap(),
            anamnesis::parse_gguf_from_reader(Cursor::new(&bytes)).unwrap(),
        ];
        for model in &parsed {
            let infos = model.tensor_info();
            assert_eq!(infos.len(), 1, "{name}");
            assert_eq!(infos[0].dtype, dtype, "{name}");
            assert_eq!(infos[0].dtype.to_string(), name);
            assert_eq!(infos[0].byte_len, Some(byte_len), "{name}");

            let info = model.inspect();
            assert_eq!(info.tensor_count, 1, "{name}");
            assert_eq!(info.dtypes, vec![dtype], "{name}");
            assert_eq!(info.total_bytes, byte_len, "{name}");
            assert_eq!(info.unknown_size_tensors, 0, "{name}");
        }

        // The header-only path the issue's `hf-fm inspect` run goes through.
        let info = anamnesis::inspect_gguf_from_reader(Cursor::new(&bytes)).unwrap();
        assert_eq!(info.dtypes, vec![dtype], "{name}");
        assert_eq!(info.total_bytes, byte_len, "{name}");
    }
}

#[test]
fn reserved_and_sentinel_discriminants_are_still_refused() {
    // 36 is commented out upstream, 43 is `GGML_TYPE_COUNT` (a sentinel, not a
    // type), and 255 is simply unknown. All three must fail closed, naming
    // the discriminant, exactly as 40-42 used to.
    for disc in [36_u32, 43, 255] {
        let bytes = reproducer(disc, N, 1024);
        let err = anamnesis::parse_gguf_bytes(bytes).unwrap_err();
        match err {
            AnamnesisError::Unsupported { format, detail } => {
                assert_eq!(format, "GGUF");
                assert_eq!(detail, format!("unknown ggml_type discriminant {disc}"));
            }
            other => panic!("discriminant {disc}: expected Unsupported, got {other:?}"),
        }
    }
}

#[test]
fn the_issue_64_element_q1_0_fixture_is_not_a_whole_block() {
    // The issue's `t41.gguf` declares 64 elements, half a `Q1_0` block. It is
    // now refused as malformed rather than as an unknown type.
    let bytes = reproducer(41, 64, 64);
    let err = anamnesis::parse_gguf_bytes(bytes).unwrap_err();
    assert!(
        matches!(&err, AnamnesisError::Parse { reason } if reason.contains("not a multiple of block size 128")),
        "{err:?}"
    );
}
