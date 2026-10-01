// SPDX-License-Identifier: MIT OR Apache-2.0

//! Which safetensors tensors count as quantisation companions (Phase 7.10).
//!
//! A companion (a weight's scale, zero-point, ...) is consumed by
//! dequantisation and never written. Until Phase 7.10 a tensor became one by
//! its **name** alone: anything ending in `_scale` was a scale. A plain model
//! with CLIP's or `SigLIP`'s `logit_scale`, a `layer_scale`, or a leftover
//! `input_scale` therefore lost those tensors on `remember` and `convert`, and
//! `inspect` under-reported the output to match. Now a scale is a companion only
//! when a quantised tensor exists for it.
//!
//! The same work found NVIDIA `ModelOpt` `NVFP4` checkpoints misread as
//! fine-grained `FP8` (with `bnb`) or as unquantised (without it, so `remember`
//! copied the packed bytes through as if they were weights). They are now
//! recognised under every feature set, reported, and refused.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::wildcard_enum_match_arm
)]

use std::collections::BTreeMap;
use std::fmt::Write as _;

use anamnesis::{AnamnesisError, ConvertOptions, ConvertTarget, QuantScheme, TargetDtype};

/// Writes a safetensors file byte for byte: `(name, dtype, shape, bytes)`.
fn safetensors(tensors: &[(&str, &str, &[usize], Vec<u8>)]) -> Vec<u8> {
    let mut header = String::from("{");
    let mut data = Vec::new();
    for (i, (name, dtype, shape, bytes)) in tensors.iter().enumerate() {
        let start = data.len();
        data.extend_from_slice(bytes);
        if i > 0 {
            header.push(',');
        }
        // EXPLICIT: `write!` into a `String` cannot fail.
        let _ = write!(
            header,
            "\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":{shape:?},\"data_offsets\":[{start},{}]}}",
            data.len()
        );
    }
    header.push('}');
    while header.len() % 8 != 0 {
        header.push(' ');
    }
    let mut out = u64::try_from(header.len()).unwrap().to_le_bytes().to_vec();
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&data);
    out
}

fn f32s(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
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

fn written_bytes(written: &BTreeMap<String, Vec<u8>>) -> u64 {
    written
        .values()
        .map(|d| u64::try_from(d.len()).unwrap())
        .sum()
}

#[test]
fn an_unquantised_model_keeps_its_scale_named_parameters() {
    let input = safetensors(&[
        (
            "text_projection.weight",
            "BF16",
            &[2, 2],
            vec![0x80, 0x3F, 0x00, 0x40, 0x40, 0x40, 0x80, 0x40],
        ),
        ("logit_scale", "F32", &[], f32s(&[4.6052])),
        (
            "vision.layer_scale",
            "F32",
            &[4],
            f32s(&[0.1, 0.2, 0.3, 0.4]),
        ),
        ("layer.0.mlp.input_scale", "F32", &[1], f32s(&[0.5])),
    ]);
    let model = anamnesis::parse_bytes(input.clone()).unwrap();
    assert_eq!(model.header.scheme, QuantScheme::Unquantized);

    // `remember` and `convert` both write all four, byte for byte.
    let remembered = tensors(&model.remember_to_bytes(TargetDtype::BF16).unwrap());
    let (converted, _) =
        anamnesis::convert_bytes(&input, ConvertTarget::Safetensors, &ConvertOptions::new())
            .unwrap();
    let converted = tensors(&converted);
    for written in [&remembered, &converted] {
        assert_eq!(written.len(), 4, "{:?}", written.keys());
        assert_eq!(written["logit_scale"], f32s(&[4.6052]));
        assert_eq!(written["vision.layer_scale"], f32s(&[0.1, 0.2, 0.3, 0.4]));
        assert_eq!(written["layer.0.mlp.input_scale"], f32s(&[0.5]));
    }

    // And `inspect` reports the size of what was written, not 56 B -> 8 B.
    let estimate = model.inspect().dequantized_size;
    assert_eq!(estimate, written_bytes(&remembered));
}

#[test]
fn real_companions_are_still_consumed() {
    // A per-tensor FP8 layer, its weight and activation scales, and a model
    // parameter that only looks like a scale.
    let input = safetensors(&[
        (
            "layer.0.mlp.weight",
            "F8_E4M3",
            &[2, 2],
            vec![0x38, 0x40, 0xB8, 0x00],
        ),
        ("layer.0.mlp.weight_scale", "F32", &[], f32s(&[2.0])),
        ("layer.0.mlp.input_scale", "F32", &[], f32s(&[0.25])),
        ("logit_scale", "F32", &[], f32s(&[4.6052])),
    ]);
    let model = anamnesis::parse_bytes(input).unwrap();
    assert_eq!(model.header.scheme, QuantScheme::PerTensorFp8);
    let written = tensors(&model.remember_to_bytes(TargetDtype::F32).unwrap());
    assert_eq!(
        written.keys().map(String::as_str).collect::<Vec<_>>(),
        ["layer.0.mlp.weight", "logit_scale"],
        "both scales of the quantised layer consumed, the model parameter kept"
    );
    // E4M3 0x38 = 1.0, 0x40 = 2.0, 0xB8 = -1.0, 0x00 = 0.0, times the scale 2.0.
    assert_eq!(written["layer.0.mlp.weight"], f32s(&[2.0, 4.0, -2.0, 0.0]));
    assert_eq!(
        model
            .inspect_with_options(
                &anamnesis::InspectOptions::new().with_output_dtype(TargetDtype::F32)
            )
            .dequantized_size,
        written_bytes(&written)
    );
}

#[test]
fn modelopt_nvfp4_is_recognised_and_refused() {
    // One layer laid out as NVIDIA ModelOpt exports NVFP4 (shapes scaled down
    // from nvidia/Llama-3.1-8B-Instruct-FP4: U8 [rows, cols/2], F8_E4M3
    // [rows, cols/16], F32 scalars).
    let input = safetensors(&[
        ("mlp.down_proj.weight", "U8", &[4, 32], (0..128).collect()),
        (
            "mlp.down_proj.weight_scale",
            "F8_E4M3",
            &[4, 4],
            vec![0x38; 16],
        ),
        ("mlp.down_proj.weight_scale_2", "F32", &[], f32s(&[0.01])),
        ("mlp.down_proj.input_scale", "F32", &[], f32s(&[0.5])),
        (
            "model.norm.weight",
            "BF16",
            &[2],
            vec![0x80, 0x3F, 0x80, 0x3F],
        ),
    ]);
    let header = anamnesis::parse_safetensors_header(&input).unwrap();
    assert_eq!(
        header.scheme,
        QuantScheme::Nvfp4,
        "not fine-grained FP8, not unquantised"
    );

    let model = anamnesis::parse_bytes(input.clone()).unwrap();
    let info = model.inspect();
    assert_eq!((info.quantized, info.scales, info.passthrough), (1, 3, 1));
    // Two values per packed byte: 128 bytes -> 256 values -> 512 B of BF16,
    // plus the 4 B norm.
    assert_eq!(info.dequantized_size, 256 * 2 + 4);

    let refused = |result: anamnesis::Result<_>| match result {
        Err(AnamnesisError::Unsupported { detail, .. }) => {
            assert!(detail.contains("ModelOpt NVFP4"), "{detail}");
        }
        Err(other) => panic!("expected Unsupported, got {other:?}"),
        Ok(()) => panic!("a ModelOpt NVFP4 checkpoint must not convert"),
    };
    refused(model.remember_to_bytes(TargetDtype::BF16).map(|_| ()));
    refused(
        anamnesis::convert_bytes(&input, ConvertTarget::Safetensors, &ConvertOptions::new())
            .map(|_| ()),
    );
}
