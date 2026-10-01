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

// ---------------------------------------------------------------------------
// NVFP4's per-tensor scale (Phase 7.10)
//
// NVIDIA's NVFP4 keeps a second, per-tensor `F32` scale beside each weight as
// `<stem>.scale`, which llama.cpp multiplies in at inference time. `remember`
// and `convert` fold it into the dequantised weight and leave it out of the
// output; `inspect` reports the size of that output.
// ---------------------------------------------------------------------------

mod nvfp4_scale_folding {
    use std::collections::BTreeMap;

    use anamnesis::{
        ConvertOptions, ConvertStats, ConvertTarget, Dtype, F32Out, GgufType, InspectOptions,
        TargetDtype,
    };

    use crate::common::gguf::{RawTensorInfo, raw_gguf};

    /// One tensor of a hand-built `GGUF`: name, dims (`GGUF` order), type, bytes.
    struct Tensor {
        name: &'static str,
        dims: Vec<u64>,
        ggml_type: u32,
        data: Vec<u8>,
    }

    /// Lays `tensors` out at 32-byte-aligned offsets, as a real file does.
    fn gguf(tensors: &[Tensor]) -> Vec<u8> {
        let mut data = Vec::new();
        let mut offsets = Vec::new();
        for t in tensors {
            offsets.push(u64::try_from(data.len()).unwrap());
            data.extend_from_slice(&t.data);
            data.resize(data.len().div_ceil(32) * 32, 0);
        }
        let infos: Vec<RawTensorInfo<'_>> = tensors
            .iter()
            .zip(&offsets)
            .map(|(t, &offset)| RawTensorInfo {
                name: t.name,
                dims: &t.dims,
                ggml_type: t.ggml_type,
                offset,
            })
            .collect();
        raw_gguf(&[], &infos, 32, &data)
    }

    /// `blocks` NVFP4 blocks of patterned bytes: every scale byte and nibble
    /// value turns up, including `0x7F` and `0xFF`.
    fn nvfp4(blocks: usize) -> Vec<u8> {
        (0..blocks * 36)
            .map(|i| u8::try_from((i * 37 + 11) % 256).unwrap())
            .collect()
    }

    fn f32s(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// The weight decoded by the kernel alone, at `f32`: what folding scales.
    fn raw(data: &[u8], n: usize) -> Vec<f32> {
        anamnesis::dequantize_gguf::<F32Out>(data, GgufType::NVFP4, n)
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&w| f32::from_le_bytes(w))
            .collect()
    }

    fn words(bytes: &[u8]) -> Vec<u32> {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&w| u32::from_le_bytes(w))
            .collect()
    }

    fn convert(bytes: &[u8], dtype: Dtype) -> (Vec<u8>, ConvertStats) {
        let options = ConvertOptions::new().with_output_dtype(dtype);
        anamnesis::convert_bytes(bytes, ConvertTarget::Safetensors, &options).unwrap()
    }

    /// Every tensor of a safetensors buffer, by name.
    fn tensors(st: &[u8]) -> BTreeMap<String, Vec<u8>> {
        safetensors::SafeTensors::deserialize(st)
            .unwrap()
            .tensors()
            .into_iter()
            .map(|(name, view)| (name, view.data().to_vec()))
            .collect()
    }

    /// `inspect`'s `dequantized_size` is the bytes the conversion wrote.
    fn assert_inspect_matches(
        bytes: &[u8],
        written: &BTreeMap<String, Vec<u8>>,
        dtype: TargetDtype,
    ) {
        let parsed = anamnesis::parse_gguf_bytes(bytes.to_vec()).unwrap();
        let estimate = parsed
            .inspect_with_options(&InspectOptions::new().with_output_dtype(dtype))
            .dequantized_size;
        let actual: u64 = written
            .values()
            .map(|d| u64::try_from(d.len()).unwrap())
            .sum();
        assert_eq!(
            estimate, actual,
            "inspect says {estimate} B, convert wrote {actual} B"
        );
    }

    #[test]
    fn a_one_element_scale_is_folded_into_its_weight() {
        let weight = nvfp4(4); // 256 elements
        let scale = 3.0e-4_f32; // not a power of two, so the product rounds
        let bytes = gguf(&[
            Tensor {
                name: "blk.0.attn_q.weight",
                dims: vec![64, 4],
                ggml_type: 40,
                data: weight.clone(),
            },
            Tensor {
                name: "blk.0.attn_q.scale",
                dims: vec![1],
                ggml_type: 0,
                data: f32s(&[scale]),
            },
            Tensor {
                name: "blk.0.attn_q.input_scale",
                dims: vec![1],
                ggml_type: 0,
                data: f32s(&[0.5]),
            },
        ]);
        let expected: Vec<f32> = raw(&weight, 256).iter().map(|&v| v * scale).collect();

        // F32: the product itself, bit for bit.
        let (st, stats) = convert(&bytes, Dtype::F32);
        let out = tensors(&st);
        let want: Vec<u32> = expected.iter().map(|v| v.to_bits()).collect();
        assert_eq!(words(&out["blk.0.attn_q.weight"]), want);
        assert!(
            !out.contains_key("blk.0.attn_q.scale"),
            "the folded scale is not written"
        );
        assert_eq!(
            out["blk.0.attn_q.input_scale"],
            f32s(&[0.5]),
            "the activation scale passes through untouched"
        );
        assert_eq!(
            (stats.folded_scales, stats.dequantized, stats.tensors),
            (1, 1, 2)
        );
        assert_inspect_matches(&bytes, &out, TargetDtype::F32);

        // BF16 and F16: the same product, rounded once, as `half` rounds it.
        let (st_bf16, _) = convert(&bytes, Dtype::BF16);
        let out_bf16 = tensors(&st_bf16);
        let want: Vec<u8> = expected
            .iter()
            .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
            .collect();
        assert_eq!(out_bf16["blk.0.attn_q.weight"], want);
        assert_inspect_matches(&bytes, &out_bf16, TargetDtype::BF16);

        let (st, _) = convert(&bytes, Dtype::F16);
        let out = tensors(&st);
        let want: Vec<u8> = expected
            .iter()
            .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
            .collect();
        assert_eq!(out["blk.0.attn_q.weight"], want);
        assert_inspect_matches(&bytes, &out, TargetDtype::F16);

        // `remember` runs the same hub, so it writes the same tensors.
        let remembered = anamnesis::parse_gguf_bytes(bytes)
            .unwrap()
            .remember_to_bytes(TargetDtype::BF16)
            .unwrap();
        assert_eq!(tensors(&remembered), out_bf16);
    }

    #[test]
    fn a_per_expert_scale_scales_each_expert_slab() {
        // Three experts of 64 × 2: one scale value per outermost slice.
        let weight = nvfp4(6); // 384 elements
        let scales = [1.0_f32, 0.5, 2.0];
        let bytes = gguf(&[
            Tensor {
                name: "blk.0.ffn_up_exps.weight",
                dims: vec![64, 2, 3],
                ggml_type: 40,
                data: weight.clone(),
            },
            Tensor {
                name: "blk.0.ffn_up_exps.scale",
                dims: vec![3],
                ggml_type: 0,
                data: f32s(&scales),
            },
        ]);
        let expected: Vec<u32> = raw(&weight, 384)
            .iter()
            .enumerate()
            .map(|(i, &v)| (v * scales[i / 128]).to_bits())
            .collect();
        let (st, stats) = convert(&bytes, Dtype::F32);
        let out = tensors(&st);
        assert_eq!(words(&out["blk.0.ffn_up_exps.weight"]), expected);
        assert_eq!(stats.folded_scales, 1);
        assert_eq!(out.len(), 1);
        assert_inspect_matches(&bytes, &out, TargetDtype::F32);
    }

    /// A `.scale` that does not fit the rule is left alone, and so is its
    /// weight: llama.cpp also uses `*.scale` for unrelated per-architecture
    /// scales, so anything short of an exact match is not ours to apply.
    #[test]
    fn scales_outside_the_rule_are_not_folded() {
        let case = |weight_name, dims: Vec<u64>, ggml_type, data, scale_dims, scale_type, scale| {
            vec![
                Tensor {
                    name: weight_name,
                    dims,
                    ggml_type,
                    data,
                },
                Tensor {
                    name: "a.scale",
                    dims: scale_dims,
                    ggml_type: scale_type,
                    data: scale,
                },
            ]
        };
        let cases = [
            (
                "F16 scale",
                case(
                    "a.weight",
                    vec![64, 4],
                    40,
                    nvfp4(4),
                    vec![1],
                    1,
                    vec![0x00, 0x3C],
                ),
            ),
            (
                "scale matches no axis",
                case(
                    "a.weight",
                    vec![64, 4],
                    40,
                    nvfp4(4),
                    vec![2],
                    0,
                    f32s(&[2.0, 2.0]),
                ),
            ),
            (
                "not an NVFP4 weight",
                case(
                    "a.weight",
                    vec![64, 4],
                    8,
                    vec![0u8; 8 * 34],
                    vec![1],
                    0,
                    f32s(&[2.0]),
                ),
            ),
            (
                "weight not named *.weight",
                case("a", vec![64, 4], 40, nvfp4(4), vec![1], 0, f32s(&[2.0])),
            ),
            (
                "expert slab not whole blocks",
                case(
                    "a.weight",
                    vec![32, 3, 2],
                    40,
                    nvfp4(3),
                    vec![2],
                    0,
                    f32s(&[2.0, 2.0]),
                ),
            ),
        ];
        for (label, tensors_in) in cases {
            let bytes = gguf(&tensors_in);
            let (st, stats) = convert(&bytes, Dtype::F32);
            let out = tensors(&st);
            assert_eq!(stats.folded_scales, 0, "{label}");
            assert_eq!(out.len(), 2, "{label}: weight and scale both written");
            assert_eq!(out["a.scale"].len(), tensors_in[1].data.len(), "{label}");
            if tensors_in[0].ggml_type == 40 {
                let n = usize::try_from(tensors_in[0].dims.iter().product::<u64>()).unwrap();
                let unscaled: Vec<u8> = raw(&tensors_in[0].data, n)
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect();
                assert_eq!(
                    out[tensors_in[0].name], unscaled,
                    "{label}: weight left unscaled"
                );
            }
            assert_inspect_matches(&bytes, &out, TargetDtype::F32);
        }
    }
}
