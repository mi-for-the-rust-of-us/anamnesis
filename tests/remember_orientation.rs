// SPDX-License-Identifier: MIT OR Apache-2.0

//! Output-orientation contract tests for `remember_to_bytes`.
//!
//! The dequantized safetensors that `remember` / `remember_to_bytes` emit
//! must be loadable by a **standard** consumer (candle, `transformers` as a
//! plain model), which expects every `nn.Linear` weight in
//! `[out_features, in_features]` orientation. The kernel-level
//! cross-validation (`tests/cross_validation_*.rs`) anchors **values** in the
//! canonical GEMM-native orientation and is structurally blind to the emitted
//! layout — the candle-mi dogfooding report
//! `docs/dogfooding-feedbacks/awq-gptq-dequant-transpose-orientation.md`
//! caught AWQ and GPTQ emitting `[in, out]` (the transpose) precisely because
//! nothing loaded the output through the public API.
//!
//! Every test here builds a synthetic model with **non-square** projection
//! dimensions (so orientation is unambiguous), routes it through
//! `parse → remember_to_bytes → safetensors::deserialize`, and asserts both
//! the emitted **shape** (`[out, in]`) and the **element mapping**
//! (`W_std[o][i] == W_native[i][o]` for the transposed schemes, identity for
//! the rest). Passthrough tensors must come through byte-identical — the
//! transpose applies to quantized projection weights only.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::similar_names
)]

use std::path::PathBuf;

use anamnesis::{TargetDtype, parse};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a minimal safetensors file in memory with the given tensors
/// (mirrors the `build_safetensors` helper in `src/model.rs` tests).
/// Only the feature-gated GPTQ/AWQ/BnB tests use it — the always-on FP8
/// test hand-assembles its header (`F8_E4M3` is not serializable through
/// the `safetensors` crate's writer).
#[cfg(any(feature = "gptq", feature = "awq", feature = "bnb"))]
fn build_safetensors(tensors: &[(&str, safetensors::Dtype, Vec<usize>, Vec<u8>)]) -> Vec<u8> {
    let views: Vec<(&str, safetensors::tensor::TensorView<'_>)> = tensors
        .iter()
        .map(|(name, dtype, shape, data)| {
            let view = safetensors::tensor::TensorView::new(*dtype, shape.clone(), data).unwrap();
            (*name, view)
        })
        .collect();
    safetensors::tensor::serialize(views, None).unwrap()
}

/// Write `bytes` to a uniquely named temp file and return its path.
fn write_temp_model(bytes: &[u8], tag: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("anamnesis_orient_{tag}.safetensors"));
    std::fs::write(&path, bytes).unwrap();
    path
}

/// Read one BF16 element (as `f32`) from raw little-endian BF16 bytes.
fn bf16_at(data: &[u8], idx: usize) -> f32 {
    let bits = u16::from_le_bytes([data[idx * 2], data[idx * 2 + 1]]);
    f32::from_bits(u32::from(bits) << 16)
}

/// `remember_to_bytes` on the model at `path`, deserialized for inspection.
fn remember_and_load(path: &PathBuf) -> Vec<u8> {
    let model = parse(path).unwrap();
    model.remember_to_bytes(TargetDtype::BF16).unwrap()
}

// ---------------------------------------------------------------------------
// GPTQ — emitted weight must be [out, in] (transposed from GEMM-native)
// ---------------------------------------------------------------------------

/// Synthetic GPTQ model: `in = 8`, `out = 16` (non-square), 4-bit,
/// `group_size = 8` (1 group), zeros stored 0 (actual = +1), scales 1.0.
/// `W_native[i][j] = ((3·i + j) mod 16) − 1` — non-symmetric so a transposed
/// element mapping cannot masquerade as the identity.
#[cfg(feature = "gptq")]
#[test]
fn gptq_remember_emits_out_in_orientation() {
    let in_features = 8usize;
    let out_features = 16usize;
    let pack_factor = 8usize; // 4-bit

    // qweight [in/pf, out] = [1, 16] I32: u32 for column j packs the 8 input
    // rows LSB-first (nibble for row i at bit 4·i) — the GPTQ convention.
    let mut qweight = Vec::new();
    for j in 0..out_features {
        let mut packed = 0u32;
        for i in 0..in_features {
            let nibble = ((3 * i + j) % 16) as u32;
            packed |= nibble << (4 * i);
        }
        qweight.extend_from_slice(&packed.to_le_bytes());
    }

    // qzeros [1, out/pf] = [1, 2] I32, all stored zeros = 0 (actual zero = 1).
    let qzeros = vec![0u8; 2 * 4];

    // scales [1, out] = [1, 16] F16, all 1.0.
    let scales: Vec<u8> = (0..out_features)
        .flat_map(|_| half::f16::from_f32(1.0).to_le_bytes())
        .collect();

    // 2-D BF16 passthrough (embed-tokens-like): must NOT be transposed.
    let embed_shape = vec![3usize, 8usize];
    let embed: Vec<u8> = (0..24u16)
        .flat_map(|v| {
            let bits = (f32::from(v).to_bits() >> 16) as u16;
            bits.to_le_bytes()
        })
        .collect();

    let file = build_safetensors(&[
        (
            "model.layers.0.self_attn.k_proj.qweight",
            safetensors::Dtype::I32,
            vec![in_features / pack_factor, out_features],
            qweight,
        ),
        (
            "model.layers.0.self_attn.k_proj.qzeros",
            safetensors::Dtype::I32,
            vec![1, out_features / pack_factor],
            qzeros,
        ),
        (
            "model.layers.0.self_attn.k_proj.scales",
            safetensors::Dtype::F16,
            vec![1, out_features],
            scales,
        ),
        (
            "model.embed_tokens.weight",
            safetensors::Dtype::BF16,
            embed_shape.clone(),
            embed.clone(),
        ),
    ]);
    let path = write_temp_model(&file, "gptq");
    let out_bytes = remember_and_load(&path);
    let parsed = safetensors::SafeTensors::deserialize(&out_bytes).unwrap();

    // Shape contract: standard nn.Linear orientation [out, in].
    let weight = parsed
        .tensor("model.layers.0.self_attn.k_proj.weight")
        .unwrap();
    assert_eq!(
        weight.shape(),
        &[out_features, in_features],
        "GPTQ remember output must be [out_features, in_features] \
         (standard nn.Linear orientation), got {:?}",
        weight.shape(),
    );

    // Element mapping: W_std[o][i] == W_native[i][o] = ((3·i + o) mod 16) − 1.
    let data = weight.data();
    for o in 0..out_features {
        for i in 0..in_features {
            let expected = ((3 * i + o) % 16) as f32 - 1.0;
            let actual = bf16_at(data, o * in_features + i);
            assert_eq!(
                actual, expected,
                "GPTQ W_std[{o}][{i}] should equal W_native[{i}][{o}]"
            );
        }
    }

    // Passthrough 2-D tensor: shape and bytes untouched.
    let pt = parsed.tensor("model.embed_tokens.weight").unwrap();
    assert_eq!(pt.shape(), embed_shape.as_slice());
    assert_eq!(pt.data(), embed.as_slice());

    std::fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------------
// AWQ — emitted weight must be [out, in] (transposed from GEMM-native)
// ---------------------------------------------------------------------------

/// Synthetic AWQ model: `in = 8`, `out = 16` (non-square), 4-bit,
/// `group_size = 8` (1 group), zeros 0 (no +1 in AWQ), scales 1.0.
/// `W_native[i][j] = (3·i + j) mod 16`, packed with the `AutoAWQ` GEMM
/// interleave (logical column offset `jo` at bit `4 × AWQ_REVERSE_ORDER[jo]`).
#[cfg(feature = "awq")]
#[test]
fn awq_remember_emits_out_in_orientation() {
    let in_features = 8usize;
    let out_features = 16usize;
    let pack_factor = 8usize; // 4-bit
    // AWQ_REVERSE_ORDER from awq/utils/packing_utils.py: logical column
    // offset jo is stored at bit position 4 × REV[jo].
    let rev = [0usize, 4, 1, 5, 2, 6, 3, 7];

    // qweight [in, out/pf] = [8, 2] I32, AutoAWQ interleave.
    let mut qweight = Vec::new();
    for i in 0..in_features {
        for c in 0..(out_features / pack_factor) {
            let mut packed = 0u32;
            for (jo, &pos) in rev.iter().enumerate() {
                let j = c * pack_factor + jo;
                let nibble = ((3 * i + j) % 16) as u32;
                packed |= nibble << (4 * pos);
            }
            qweight.extend_from_slice(&packed.to_le_bytes());
        }
    }

    // qzeros [1, out/pf] = [1, 2] I32, all zeros.
    let qzeros = vec![0u8; 2 * 4];

    // scales [1, out] = [1, 16] F16, all 1.0.
    let scales: Vec<u8> = (0..out_features)
        .flat_map(|_| half::f16::from_f32(1.0).to_le_bytes())
        .collect();

    let file = build_safetensors(&[
        (
            "model.layers.0.self_attn.k_proj.qweight",
            safetensors::Dtype::I32,
            vec![in_features, out_features / pack_factor],
            qweight,
        ),
        (
            "model.layers.0.self_attn.k_proj.qzeros",
            safetensors::Dtype::I32,
            vec![1, out_features / pack_factor],
            qzeros,
        ),
        (
            "model.layers.0.self_attn.k_proj.scales",
            safetensors::Dtype::F16,
            vec![1, out_features],
            scales,
        ),
    ]);
    let path = write_temp_model(&file, "awq");
    let out_bytes = remember_and_load(&path);
    let parsed = safetensors::SafeTensors::deserialize(&out_bytes).unwrap();

    let weight = parsed
        .tensor("model.layers.0.self_attn.k_proj.weight")
        .unwrap();
    assert_eq!(
        weight.shape(),
        &[out_features, in_features],
        "AWQ remember output must be [out_features, in_features] \
         (standard nn.Linear orientation), got {:?}",
        weight.shape(),
    );

    // Element mapping: W_std[o][i] == W_native[i][o] = (3·i + o) mod 16.
    let data = weight.data();
    for o in 0..out_features {
        for i in 0..in_features {
            let expected = ((3 * i + o) % 16) as f32;
            let actual = bf16_at(data, o * in_features + i);
            assert_eq!(
                actual, expected,
                "AWQ W_std[{o}][{i}] should equal W_native[{i}][{o}]"
            );
        }
    }

    std::fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------------
// BnB NF4 — already correct: shape comes from the quant_state blob [out, in]
// ---------------------------------------------------------------------------

/// Pin the already-correct `BnB4` orientation: the writer records the original
/// non-square 2-D shape in the `quant_state` blob and `remember` recovers it.
#[cfg(feature = "bnb")]
#[test]
fn bnb4_remember_preserves_out_in_shape() {
    use anamnesis::{BnbWriteInput, write_bnb_nf4_safetensors_bytes};

    // Non-square [6, 64] BF16 source (384 elements, multiple of 64).
    let rows = 6usize;
    let cols = 64usize;
    let bf16: Vec<u8> = (0..(rows * cols))
        .flat_map(|v| {
            let x = (v as f32) / 384.0 - 0.5;
            let bits = (x.to_bits() >> 16) as u16;
            bits.to_le_bytes()
        })
        .collect();
    let shape = [rows, cols];
    let inputs = vec![BnbWriteInput {
        name: "model.layers.0.mlp.gate_proj",
        shape: &shape,
        bf16_data: &bf16,
    }];
    let st_bytes = write_bnb_nf4_safetensors_bytes(&inputs).unwrap();
    let path = write_temp_model(&st_bytes, "bnb4");

    let out_bytes = remember_and_load(&path);
    let parsed = safetensors::SafeTensors::deserialize(&out_bytes).unwrap();
    let weight = parsed
        .tensor("model.layers.0.mlp.gate_proj.weight")
        .unwrap();
    assert_eq!(
        weight.shape(),
        &[rows, cols],
        "BnB4 remember output must recover the quant_state 2-D shape",
    );

    std::fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------------
// BnB INT8 — already correct: on-disk weight is [out, in], shape preserved
// ---------------------------------------------------------------------------

#[cfg(feature = "bnb")]
#[test]
fn bnb_int8_remember_preserves_out_in_shape_and_values() {
    let out_features = 4usize;
    let in_features = 10usize;

    // weight [out, in] I8: w[o][i] = o·10 + i (≤ 39, exact in BF16).
    let weight: Vec<u8> = (0..out_features)
        .flat_map(|o| (0..in_features).map(move |i| (o * 10 + i) as u8))
        .collect();
    // SCB [out] F32, all 127.0 → scale exactly 1.0.
    let scb: Vec<u8> = (0..out_features)
        .flat_map(|_| 127.0_f32.to_le_bytes())
        .collect();

    let file = build_safetensors(&[
        (
            "model.layers.0.mlp.down_proj.weight",
            safetensors::Dtype::I8,
            vec![out_features, in_features],
            weight,
        ),
        (
            "model.layers.0.mlp.down_proj.SCB",
            safetensors::Dtype::F32,
            vec![out_features],
            scb,
        ),
    ]);
    let path = write_temp_model(&file, "bnb_int8");
    let out_bytes = remember_and_load(&path);
    let parsed = safetensors::SafeTensors::deserialize(&out_bytes).unwrap();

    let weight = parsed
        .tensor("model.layers.0.mlp.down_proj.weight")
        .unwrap();
    assert_eq!(
        weight.shape(),
        &[out_features, in_features],
        "BnB INT8 remember output must preserve the on-disk [out, in] shape",
    );
    // Identity element mapping (no transpose for INT8).
    let data = weight.data();
    for o in 0..out_features {
        for i in 0..in_features {
            let expected = (o * 10 + i) as f32;
            let actual = bf16_at(data, o * in_features + i);
            assert_eq!(actual, expected, "BnB INT8 W[{o}][{i}] identity mapping");
        }
    }

    std::fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------------
// FP8 per-tensor — already correct: elementwise, shape preserved
// ---------------------------------------------------------------------------

/// Hand-built per-tensor FP8 fixture with a non-square [2, 4] weight
/// (the `safetensors` crate may not serialize `F8_E4M3`, so the header is
/// hand-assembled like `build_fp8_per_tensor_fixture` in `src/model.rs`).
#[test]
fn fp8_remember_preserves_out_in_shape_and_values() {
    let rows = 2usize;
    let cols = 4usize;
    // E4M3 values: 1.0=0x38, 1.5=0x3C, 2.0=0x40, 3.0=0x44, 4.0=0x48,
    // 6.0=0x4C, 8.0=0x50, 12.0=0x54 — all exact in BF16 after ×1.0 scale.
    let fp8_bytes: Vec<u8> = vec![0x38, 0x3C, 0x40, 0x44, 0x48, 0x4C, 0x50, 0x54];
    let expected: [f32; 8] = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0];
    let scale = 1.0_f32.to_le_bytes();

    let header = serde_json::json!({
        "layer.weight": {
            "dtype": "F8_E4M3",
            "shape": [rows, cols],
            "data_offsets": [0, 8],
        },
        "layer.weight_scale": {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [8, 12],
        },
    });
    let header_bytes = serde_json::to_vec(&header).unwrap();
    let mut file = Vec::new();
    file.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    file.extend_from_slice(&header_bytes);
    file.extend_from_slice(&fp8_bytes);
    file.extend_from_slice(&scale);

    let path = write_temp_model(&file, "fp8");
    let out_bytes = remember_and_load(&path);
    let parsed = safetensors::SafeTensors::deserialize(&out_bytes).unwrap();

    let weight = parsed.tensor("layer.weight").unwrap();
    assert_eq!(
        weight.shape(),
        &[rows, cols],
        "FP8 remember output must preserve the on-disk [out, in] shape",
    );
    let data = weight.data();
    for (idx, &exp) in expected.iter().enumerate() {
        assert_eq!(
            bf16_at(data, idx),
            exp,
            "FP8 element {idx} identity mapping"
        );
    }

    std::fs::remove_file(&path).ok();
}
