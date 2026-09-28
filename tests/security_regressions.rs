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

// ---------------------------------------------------------------------------
// N-1 (found by the design review of the fixes): the `.pth` strided copy walked
// every dimension for every element, with no cap on rank, so `(N, 1, …, 1)`
// with odd strides on the size-1 dimensions cost `N × rank`, invisible to
// `inspect`. The `.pth` twin of H-3.
// ---------------------------------------------------------------------------

#[cfg(feature = "pth")]
mod n1_pth_rank {
    use std::io::Cursor;

    use anamnesis::AnamnesisError;

    use crate::common::pth::single_u8_view;

    fn is_rank_cap(r: &Result<impl std::fmt::Debug, AnamnesisError>) -> bool {
        matches!(r, Err(AnamnesisError::LimitExceeded { limit, .. }) if *limit == "PTH_MAX_DIMS")
    }

    #[test]
    fn more_than_64_dimensions_are_refused_on_every_path() {
        let shape = vec![1i64; 65];
        let strides = vec![1i64; 65];
        let bytes = single_u8_view(&shape, &strides, &[7]);
        assert!(is_rank_cap(&anamnesis::parse_pth_bytes(bytes.clone())));
        assert!(is_rank_cap(&anamnesis::inspect_pth_from_reader(
            Cursor::new(&bytes)
        )));
        assert!(is_rank_cap(&anamnesis::parse_pth_front_matter_from_reader(
            Cursor::new(&bytes)
        )));
    }

    #[test]
    fn size_one_dimensions_do_not_change_the_bytes() {
        // (2, 1, 3) row-major, but with an arbitrary stride on the size-1
        // dimension: PyTorch treats it as contiguous, and so must we.
        let storage = [1u8, 2, 3, 4, 5, 6];
        let odd =
            anamnesis::parse_pth_bytes(single_u8_view(&[2, 1, 3], &[3, 99, 1], &storage)).unwrap();
        let canonical =
            anamnesis::parse_pth_bytes(single_u8_view(&[2, 1, 3], &[3, 3, 1], &storage)).unwrap();
        assert_eq!(
            odd.tensors().unwrap()[0].data,
            canonical.tensors().unwrap()[0].data
        );
        assert_eq!(odd.tensors().unwrap()[0].shape, vec![2, 1, 3]);

        // Transposed, with size-1 dimensions interleaved at odd strides.
        let t = anamnesis::parse_pth_bytes(single_u8_view(&[1, 2, 1, 3], &[5, 1, 77, 2], &storage))
            .unwrap();
        assert_eq!(&*t.tensors().unwrap()[0].data, &[1, 3, 5, 2, 4, 6]);
    }

    #[test]
    fn the_maximum_rank_with_odd_unit_strides_materialises_quickly() {
        // 64 dimensions, 63 of them size 1 with odd strides: before the fix this
        // took the strided copy and walked all 64 dimensions per element.
        let n = 1i64 << 20;
        let mut shape = vec![n];
        shape.extend(std::iter::repeat_n(1i64, 63));
        let mut strides = vec![1i64];
        strides.extend(std::iter::repeat_n(3i64, 63));
        let storage = vec![9u8; 1 << 20];
        let parsed =
            anamnesis::parse_pth_bytes(single_u8_view(&shape, &strides, &storage)).unwrap();
        let started = std::time::Instant::now();
        let tensors = parsed.tensors().unwrap();
        assert_eq!(tensors[0].data.len(), 1 << 20);
        assert!(
            matches!(tensors[0].data, std::borrow::Cow::Borrowed(_)),
            "a contiguous tensor with odd unit strides takes the zero-copy path"
        );
        assert!(started.elapsed() < std::time::Duration::from_secs(5));
    }
}

// ---------------------------------------------------------------------------
// I-9: an empty non-contiguous `.pth` tensor was an error, because the offset
// bound computed `dim - 1` for a zero dimension.
// ---------------------------------------------------------------------------

#[cfg(feature = "pth")]
#[test]
fn i9_an_empty_non_contiguous_tensor_is_empty() {
    let bytes = common::pth::single_u8_view(&[0, 3], &[1, 5], &[1, 2, 3]);
    let parsed = anamnesis::parse_pth_bytes(bytes).unwrap();
    let tensors = parsed.tensors().unwrap();
    assert_eq!(tensors[0].shape, vec![0, 3]);
    assert!(tensors[0].data.is_empty());
}

// ---------------------------------------------------------------------------
// M-2: many `.pth` keys aliasing one storage. Each is materialised in full, so
// 117 KB of input produced 250 MiB of output, and no limit applied. Now a
// permanent ratio caps the total at 16x the distinct storage bytes, and the
// caller's `max_total_bytes` bounds what is materialised.
// ---------------------------------------------------------------------------

#[cfg(feature = "pth")]
mod m2_aliased_pth_storages {
    use std::io::Cursor;

    use anamnesis::{AnamnesisError, ConvertOptions, ConvertTarget, ParseLimits};

    use crate::common::pth::{TensorSpec, aliased_state_dict_pickle, pth_archive};

    const STORAGE: usize = 64 * 1024;

    fn aliased(count: usize, byteorder: Option<&[u8]>) -> Vec<u8> {
        let shape = [i64::try_from(STORAGE).unwrap()];
        let spec = TensorSpec {
            name: "t0",
            storage_class: "ByteStorage",
            storage_key: "0",
            offset: 0,
            shape: &shape,
            strides: &[1],
        };
        pth_archive(
            &aliased_state_dict_pickle(&spec, count),
            &[("0", &vec![1u8; STORAGE])],
            byteorder,
        )
    }

    fn is_limit(r: &Result<impl std::fmt::Debug, AnamnesisError>, which: &str) -> bool {
        matches!(r, Err(AnamnesisError::LimitExceeded { limit, .. }) if *limit == which)
    }

    #[test]
    fn the_permanent_ratio_refuses_the_audit_file_at_default_limits() {
        // The audit's file: 4000 keys over one 64 KiB storage.
        let bytes = aliased(4000, None);
        let ratio = "PTH_MAX_MATERIALISE_RATIO";
        assert!(is_limit(&anamnesis::parse_pth_bytes(bytes.clone()), ratio));
        assert!(is_limit(
            &anamnesis::inspect_pth_from_reader(Cursor::new(&bytes)),
            ratio
        ));
        assert!(is_limit(
            &anamnesis::convert_bytes(&bytes, ConvertTarget::Safetensors, &ConvertOptions::new()),
            ratio
        ));
    }

    #[test]
    fn the_ratio_boundary_is_sixteen() {
        assert!(anamnesis::parse_pth_bytes(aliased(16, None)).is_ok());
        assert!(is_limit(
            &anamnesis::parse_pth_bytes(aliased(17, None)),
            "PTH_MAX_MATERIALISE_RATIO"
        ));
    }

    #[test]
    fn tied_weights_convert_under_default_limits() {
        // The four-way tie of a shared embedding, as in T5.
        let (out, _) = anamnesis::convert_bytes(
            &aliased(4, None),
            ConvertTarget::Safetensors,
            &ConvertOptions::new(),
        )
        .unwrap();
        let header = anamnesis::parse_safetensors_header(&out).unwrap();
        assert_eq!(header.tensors.len(), 4);
    }

    #[test]
    fn caller_limits_bound_what_is_materialised() {
        let four = 4 * STORAGE as u64;
        let limits = |total| ParseLimits::default().with_max_total_bytes(total);

        // `convert` owns every tensor: 4 x 64 KiB.
        let bytes = aliased(4, None);
        let convert = |total| {
            anamnesis::convert_bytes(
                &bytes,
                ConvertTarget::Safetensors,
                &ConvertOptions::new().with_limits(limits(total)),
            )
        };
        assert!(convert(four).is_ok());
        assert!(is_limit(&convert(four - 1), "max_total_bytes"));

        // `tensors()` borrows little-endian contiguous data, so it owns nothing
        // and nothing is charged: a budget far below the 256 KiB of tensors,
        // though above what the parse itself charges, is enough...
        let parsed =
            anamnesis::parse_pth_bytes_with_limits(bytes.clone(), &limits(32 << 10)).unwrap();
        assert!(parsed.tensors().is_ok());

        // ...but a big-endian file is copied to be swapped, and that is charged.
        let big = aliased(4, Some(b"big"));
        let tight = anamnesis::parse_pth_bytes_with_limits(big.clone(), &limits(four - 1)).unwrap();
        assert!(is_limit(&tight.tensors(), "max_total_bytes"));
        let enough = anamnesis::parse_pth_bytes_with_limits(big, &limits(four)).unwrap();
        assert!(enough.tensors().is_ok());
    }
}

// ---------------------------------------------------------------------------
// L-3: `ParseLimits` bounded parsing only. Every materialising call now checks
// its output against `max_total_bytes` before allocating it, using the same
// figure `inspect` reports, so a host whose gate checks
// `dequantized_size <= max_total_bytes` is not refused later (parsing charges
// its own, much smaller, allocations against the same limit), and one byte less
// is refused before anything is dequantised.
//
// The inputs are synthetic and sized so that the output dwarfs what the parse
// charges: on the tiny reference fixtures the parse overhead exceeds the output,
// and the boundary could not be observed.
// ---------------------------------------------------------------------------

mod l3_gate_equivalence {
    use anamnesis::{AnamnesisError, ConvertOptions, ConvertStats, ConvertTarget, ParseLimits};

    use crate::common::builders;

    fn refused_before_materialising(r: &Result<impl std::fmt::Debug, AnamnesisError>) -> bool {
        matches!(
            r,
            Err(AnamnesisError::LimitExceeded { limit, message })
                if *limit == "max_total_bytes" && message.contains("would materialise")
        )
    }

    fn limits(total: u64) -> ParseLimits {
        ParseLimits::default().with_max_total_bytes(total)
    }

    fn convert_with_total(input: &[u8], total: u64) -> anamnesis::Result<(Vec<u8>, ConvertStats)> {
        anamnesis::convert_bytes(
            input,
            ConvertTarget::Safetensors,
            &ConvertOptions::new().with_limits(limits(total)),
        )
    }

    /// `convert` succeeds at exactly `inspect().dequantized_size` and is
    /// refused, before materialising, one byte below it.
    fn assert_convert_gate(label: &str, input: &[u8], gate: u64) {
        assert!(
            convert_with_total(input, gate).is_ok(),
            "{label} at the gate"
        );
        assert!(
            refused_before_materialising(&convert_with_total(input, gate - 1)),
            "{label} one byte below the gate"
        );
    }

    #[test]
    fn safetensors_convert_is_bounded_by_the_inspect_figure() {
        let mut inputs = vec![("fp8", builders::build_fp8_checkpoint())];
        if cfg!(feature = "gptq") {
            inputs.push(("gptq", builders::build_gptq_checkpoint()));
        }
        for (label, bytes) in inputs {
            let gate = anamnesis::parse_bytes(bytes.clone())
                .unwrap()
                .inspect()
                .dequantized_size;
            assert_convert_gate(label, &bytes, gate);
        }
    }

    #[test]
    fn safetensors_remember_charges_only_what_it_owns() {
        // `remember` borrows the passthrough tensor, so it charges only the
        // dequantised share: 256 x 256 elements at BF16.
        let bytes = builders::build_fp8_checkpoint();
        let dequantised = 256 * 256 * 2;
        let remember = |total| {
            anamnesis::parse_bytes_with_limits(bytes.clone(), &limits(total))
                .unwrap()
                .remember_to_bytes(anamnesis::TargetDtype::BF16)
        };
        assert!(remember(dequantised).is_ok());
        assert!(refused_before_materialising(&remember(dequantised - 1)));
    }

    #[cfg(feature = "pth")]
    #[test]
    fn pth_convert_is_bounded_by_the_inspect_figure() {
        let bytes = crate::common::pth::single_u8_view(&[65536], &[1], &vec![3u8; 65536]);
        let gate = anamnesis::parse_pth_bytes(bytes.clone())
            .unwrap()
            .inspect()
            .dequantized_size;
        assert_convert_gate("pth", &bytes, gate);
    }

    #[cfg(feature = "gguf")]
    #[test]
    fn gguf_convert_and_remember_are_bounded_by_the_inspect_figure() {
        let (gguf, _) = anamnesis::convert_bytes(
            &builders::build_fp8_checkpoint(),
            ConvertTarget::Gguf,
            &ConvertOptions::new(),
        )
        .unwrap();
        let gate = anamnesis::parse_gguf_bytes(gguf.clone())
            .unwrap()
            .inspect()
            .dequantized_size;
        assert_convert_gate("gguf", &gguf, gate);

        let remember = |total| {
            anamnesis::parse_gguf_bytes_with_limits(gguf.clone(), &limits(total))
                .unwrap()
                .remember_to_bytes(anamnesis::TargetDtype::BF16)
        };
        assert!(remember(gate).is_ok());
        assert!(refused_before_materialising(&remember(gate - 1)));
    }
}

// ---------------------------------------------------------------------------
// H-2: GGUF tensor-info records could alias one data range (each copy then
// dequantised in full: 276 KB in, 105 MB out), and the writer padded every
// tensor to whatever alignment the source declared (1 MiB in, 1 GB out, with
// `inspect` reporting 3.9 KB).
// ---------------------------------------------------------------------------

#[cfg(feature = "gguf")]
mod h2_gguf_aliasing_and_alignment {
    use std::collections::HashMap;
    use std::io::Cursor;

    use anamnesis::{AnamnesisError, ConvertOptions, ConvertTarget, GgufMetadataValue};

    use crate::common::gguf::{GGUF_KV_U32, RawTensorInfo, raw_gguf};

    /// A `GGUF` of one-dimensional `F32` tensors of `elements` each, at the
    /// given relative offsets, over `data`.
    fn gguf(
        kvs: &[(&str, u32, &[u8])],
        offsets: &[u64],
        elements: u64,
        alignment: usize,
        data: &[u8],
    ) -> Vec<u8> {
        let names: Vec<String> = (0..offsets.len()).map(|i| format!("t{i}")).collect();
        let dims = [elements];
        let infos: Vec<RawTensorInfo<'_>> = names
            .iter()
            .zip(offsets)
            .map(|(name, &offset)| RawTensorInfo {
                name,
                dims: &dims,
                ggml_type: 0,
                offset,
            })
            .collect();
        raw_gguf(kvs, &infos, alignment, data)
    }

    fn is_parse(r: &Result<impl std::fmt::Debug, AnamnesisError>, needle: &str) -> bool {
        matches!(r, Err(AnamnesisError::Parse { reason }) if reason.contains(needle))
    }

    #[test]
    fn aliased_tensor_data_is_refused_on_every_path() {
        // 100 tensors of 64 KiB, all at offset 0 of one 64 KiB region.
        let bytes = gguf(&[], &[0; 100], 16 * 1024, 32, &vec![0u8; 64 * 1024]);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("aliased.gguf");
        std::fs::write(&path, &bytes).unwrap();

        assert!(is_parse(&anamnesis::parse_gguf(&path), "overlap"));
        assert!(is_parse(
            &anamnesis::parse_gguf_bytes(bytes.clone()),
            "overlap"
        ));
        assert!(is_parse(
            &anamnesis::parse_gguf_from_reader(Cursor::new(&bytes)),
            "overlap"
        ));
        assert!(is_parse(
            &anamnesis::inspect_gguf_from_reader(Cursor::new(&bytes)),
            "overlap"
        ));
        assert!(is_parse(
            &anamnesis::parse_gguf_front_matter_from_reader(Cursor::new(&bytes)),
            "overlap"
        ));
        assert!(is_parse(
            &anamnesis::convert_bytes(&bytes, ConvertTarget::Safetensors, &ConvertOptions::new()),
            "overlap"
        ));
    }

    #[test]
    fn partly_overlapping_and_adjacent_tensors() {
        // Two 8-element F32 tensors (32 bytes each). At offsets 0 and 32 they
        // are adjacent, which is how every real file is laid out.
        let data = vec![0u8; 64];
        assert!(anamnesis::parse_gguf_bytes(gguf(&[], &[0, 32], 8, 32, &data)).is_ok());
        // At alignment 16, offsets 0 and 16 overlap by 16 bytes.
        let align16 = 16u32.to_le_bytes();
        let kv: [(&str, u32, &[u8]); 1] = [("general.alignment", GGUF_KV_U32, &align16)];
        assert!(is_parse(
            &anamnesis::parse_gguf_bytes(gguf(&kv, &[0, 16], 8, 16, &data)),
            "overlap"
        ));
    }

    #[test]
    fn the_reader_refuses_an_alignment_that_is_not_a_power_of_two() {
        for alignment in [48u32, 7] {
            let value = alignment.to_le_bytes();
            let kv: [(&str, u32, &[u8]); 1] = [("general.alignment", GGUF_KV_U32, &value)];
            let bytes = gguf(&kv, &[0], 1, alignment as usize, &[0u8; 4]);
            assert!(
                is_parse(&anamnesis::parse_gguf_bytes(bytes), "not a power of two"),
                "alignment {alignment}"
            );
        }
    }

    #[test]
    fn the_writer_refuses_an_alignment_that_is_not_a_power_of_two() {
        let st = crate::common::builders::build_fp8_checkpoint();
        for alignment in [48u32, 7] {
            let options = ConvertOptions::new().with_gguf_metadata(HashMap::from([(
                "general.alignment".to_owned(),
                GgufMetadataValue::U32(alignment),
            )]));
            let r = anamnesis::convert_bytes(&st, ConvertTarget::Gguf, &options);
            assert!(
                matches!(&r, Err(AnamnesisError::Unsupported { detail, .. }) if detail.contains("power of two")),
                "alignment {alignment}: {r:?}"
            );
        }
    }

    #[test]
    fn a_large_source_alignment_is_not_inherited() {
        // Eight one-element tensors, legitimately 64 KiB apart under a 64 KiB
        // alignment. Converting to GGUF used to pad every output tensor to
        // 64 KiB too; the output now uses the default alignment of 32.
        let big = 1u32 << 16;
        let value = big.to_le_bytes();
        let kv: [(&str, u32, &[u8]); 1] = [("general.alignment", GGUF_KV_U32, &value)];
        let offsets: Vec<u64> = (0..8).map(|i| i * u64::from(big)).collect();
        let bytes = gguf(
            &kv,
            &offsets,
            1,
            big as usize,
            &vec![0u8; 7 * big as usize + 4],
        );
        assert_eq!(
            anamnesis::parse_gguf_bytes(bytes.clone())
                .unwrap()
                .alignment(),
            big
        );

        let (out, _) =
            anamnesis::convert_bytes(&bytes, ConvertTarget::Gguf, &ConvertOptions::new()).unwrap();
        let converted = anamnesis::parse_gguf_bytes(out.clone()).unwrap();
        assert_eq!(converted.alignment(), 32);
        assert!(out.len() < 4096, "output is {} bytes", out.len());

        // An alignment the caller asks for explicitly is still honoured.
        let options = ConvertOptions::new().with_gguf_metadata(HashMap::from([(
            "general.alignment".to_owned(),
            GgufMetadataValue::U32(64),
        )]));
        let (out, _) = anamnesis::convert_bytes(&bytes, ConvertTarget::Gguf, &options).unwrap();
        assert_eq!(anamnesis::parse_gguf_bytes(out).unwrap().alignment(), 64);
    }
}

// ---------------------------------------------------------------------------
// H-3: the `Fortran`-order transposition walked every declared dimension for
// every element, and the rank was bounded only by the 1 MiB header cap. A 240 KB
// archive with 80000 size-1 dimensions took 4.2 s under tight limits while
// `inspect` saw a small array.
// ---------------------------------------------------------------------------

#[cfg(feature = "npz")]
mod h3_npy_rank {
    use std::io::Cursor;

    use anamnesis::AnamnesisError;

    use crate::common::builders::build_npz_raw;

    fn fortran_u8(shape: &str, data: &[u8]) -> Vec<u8> {
        let dict = format!("{{'descr': '|u1', 'fortran_order': True, 'shape': {shape}, }}");
        build_npz_raw(&[("w", &dict, data)])
    }

    fn is_rank_cap(r: &Result<impl std::fmt::Debug, AnamnesisError>) -> bool {
        matches!(r, Err(AnamnesisError::LimitExceeded { limit, .. }) if *limit == "NPY_MAX_DIMS")
    }

    #[test]
    fn the_audit_archive_is_refused_before_any_work() {
        // The audit's PoC at its smallest size: 40000 elements, rank 40001.
        let n = 40_000;
        let shape = format!("({n},{})", "1,".repeat(n));
        let bytes = fortran_u8(&shape, &vec![7u8; n]);
        let started = std::time::Instant::now();
        assert!(is_rank_cap(&anamnesis::parse_npz_bytes(bytes.clone())));
        assert!(is_rank_cap(&anamnesis::inspect_npz_from_reader(
            Cursor::new(&bytes)
        )));
        assert!(started.elapsed() < std::time::Duration::from_secs(5));
    }

    #[test]
    fn the_maximum_rank_with_size_one_dimensions_is_fast_and_correct() {
        // Shape (2, 1 x 62, 3) in Fortran order: the 2x3 matrix [[0, 1, 2],
        // [3, 4, 5]] is stored column by column as 0 3 1 4 2 5.
        let shape = format!("(2,{}3)", "1,".repeat(62));
        let bytes = fortran_u8(&shape, &[0, 3, 1, 4, 2, 5]);
        let arrays = anamnesis::parse_npz_bytes(bytes).unwrap();
        assert_eq!(arrays["w"].data, vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(arrays["w"].shape.len(), 64);
    }
}
