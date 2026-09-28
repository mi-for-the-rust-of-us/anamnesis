// SPDX-License-Identifier: MIT OR Apache-2.0

//! Regression tests for the Phase 7.9 security audit (2026-09-28).
//!
//! Each section carries the audit's finding ID. The inputs are the audit's
//! proofs of concept, rebuilt in memory by `tests/common`, and every test asserts
//! the property the finding broke: a clean `Err` (never a panic, an abort or an
//! allocation the input does not pay for), on **every** entry point that can
//! reach the code, because the audit's other lesson is that fixes drift between
//! the mmap, bytes and reader paths.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::as_conversions
)]

mod common;

// ---------------------------------------------------------------------------
// H-1: a `.pth` tensor view may not materialise more bytes than its storage
// holds. A 380-byte file with a zero stride once aborted the process with a
// 4 EiB allocation inside `tensors()` / `convert_bytes`.
// ---------------------------------------------------------------------------

#[cfg(feature = "pth")]
mod h1_expanded_pth_views {
    use std::io::Cursor;

    use anamnesis::{AnamnesisError, ConvertOptions, ConvertTarget, ParseLimits};

    use crate::common::pth::single_u8_view;

    /// The shapes the audit's proofs of concept used, all with every stride 0 over a 1-byte
    /// storage: capacity-overflow panic, allocation-failure abort, a 1 GiB
    /// "successful" amplification, and a 4 TiB two-dimensional variant.
    const EXPANDED: &[(&str, &[i64], &[i64])] = &[
        ("panic", &[1 << 62, 3], &[0, 0]),
        ("abort", &[1 << 62], &[0]),
        ("gib", &[1 << 30], &[0]),
        ("2d", &[1 << 20, 1 << 20], &[0, 0]),
    ];

    fn is_expanded_view_error(r: &Result<impl std::fmt::Debug, AnamnesisError>) -> bool {
        matches!(r, Err(AnamnesisError::Parse { reason }) if reason.contains("expanded views"))
    }

    #[test]
    fn every_entry_point_refuses_an_expanded_view() {
        let dir = tempfile::tempdir().unwrap();
        for (label, shape, strides) in EXPANDED {
            let bytes = single_u8_view(shape, strides, &[7]);
            let path = dir.path().join(format!("{label}.pth"));
            std::fs::write(&path, &bytes).unwrap();

            assert!(
                is_expanded_view_error(&anamnesis::parse_pth(&path)),
                "{label}: parse_pth"
            );
            assert!(
                is_expanded_view_error(&anamnesis::parse_pth_bytes(bytes.clone())),
                "{label}: parse_pth_bytes"
            );
            assert!(
                is_expanded_view_error(&anamnesis::parse_pth_from_reader(Cursor::new(&bytes))),
                "{label}: parse_pth_from_reader"
            );
            assert!(
                is_expanded_view_error(&anamnesis::inspect_pth_from_reader(Cursor::new(&bytes))),
                "{label}: inspect_pth_from_reader"
            );
            assert!(
                is_expanded_view_error(&anamnesis::parse_pth_front_matter_from_reader(
                    Cursor::new(&bytes)
                )),
                "{label}: parse_pth_front_matter_from_reader"
            );
            // The entry point Phase 8 hands to Python, with and without limits.
            for options in [
                ConvertOptions::new(),
                ConvertOptions::new()
                    .with_limits(ParseLimits::default().with_max_total_bytes(1 << 20)),
            ] {
                assert!(
                    is_expanded_view_error(&anamnesis::convert_bytes(
                        &bytes,
                        ConvertTarget::Safetensors,
                        &options
                    )),
                    "{label}: convert_bytes"
                );
            }
        }
    }

    #[test]
    fn a_zero_stride_on_a_size_one_dimension_still_parses() {
        // Hugging Face's `position_ids = arange(n).expand(1, -1)`: shape (1, 4),
        // strides (0, 1). Nothing is repeated, so the view is valid.
        let bytes = single_u8_view(&[1, 4], &[0, 1], &[1, 2, 3, 4]);
        let parsed = anamnesis::parse_pth_bytes(bytes).unwrap();
        let tensors = parsed.tensors().unwrap();
        assert_eq!(tensors[0].shape, vec![1, 4]);
        assert_eq!(&*tensors[0].data, &[1, 2, 3, 4]);
    }

    #[test]
    fn a_transposed_view_is_materialised_row_major() {
        // Storage [1..=6] read as a 2x3 view with strides (1, 2): the transpose
        // of the row-major 3x2 matrix [[1, 2], [3, 4], [5, 6]].
        let bytes = single_u8_view(&[2, 3], &[1, 2], &[1, 2, 3, 4, 5, 6]);
        let parsed = anamnesis::parse_pth_bytes(bytes).unwrap();
        let tensors = parsed.tensors().unwrap();
        assert_eq!(&*tensors[0].data, &[1, 3, 5, 2, 4, 6]);
    }

    #[test]
    fn a_view_past_the_end_of_its_storage_is_refused_at_parse_time() {
        // Contiguous, but 5 bytes from a 4-byte storage: used to fail only
        // later, in `tensors()`; now every entry point refuses it.
        let bytes = single_u8_view(&[5], &[1], &[1, 2, 3, 4]);
        assert!(is_expanded_view_error(&anamnesis::parse_pth_bytes(
            bytes.clone()
        )));
        assert!(is_expanded_view_error(&anamnesis::inspect_pth_from_reader(
            Cursor::new(&bytes)
        )));
    }
}
