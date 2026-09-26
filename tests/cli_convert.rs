// SPDX-License-Identifier: MIT OR Apache-2.0

//! CLI smoke tests for `amn convert` — Phase 6 step 2.
//!
//! Spawns the built binary via `std::process::Command` to verify argument
//! parsing, dispatch routing, and end-to-end conversion for each
//! v0.6.0-available target. Pairs with the in-process library-level
//! round-trip suite at `tests/cross_validation_convert.rs` (step 3), which
//! exercises the same conversion paths without the CLI hop.
//!
//! Gated on `cli` because the binaries declare `required-features = ["cli"]`.
//! Each test is further gated on the format features its conversion needs, and
//! Cargo builds `amn` with the same feature set as this test crate, so a test
//! never runs against a binary that lacks the format it exercises.

#![cfg(feature = "cli")]
#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::same_item_push
)]

mod common;

use std::process::Command;

#[cfg(feature = "bnb")]
use common::bf16::bf16_bytes_from_f32_iter;
#[cfg(feature = "npz")]
use common::builders::build_npz_f32;
use common::builders::{build_safetensors_bf16, write_temp};

/// The `amn` binary. Cargo builds it for this test crate (with this crate's
/// feature set) and points `CARGO_BIN_EXE_amn` at it, so it can never be stale
/// or missing.
fn binary_path() -> &'static str {
    env!("CARGO_BIN_EXE_amn")
}

// ---------------------------------------------------------------------------
// Smoke tests
// ---------------------------------------------------------------------------

#[test]
#[cfg(feature = "npz")]
fn convert_npz_to_safetensors_smokes() {
    let f32_data: Vec<u8> = (0..4u32)
        .flat_map(|i| f32::from(i as u16).to_le_bytes())
        .collect();
    let other: Vec<u8> = (0..6u32)
        .flat_map(|i| (f32::from(i as u16) * 0.5).to_le_bytes())
        .collect();
    let npz_bytes = build_npz_f32(&[("w", &[2, 2], &f32_data), ("b", &[6], &other)]);
    let (_dir, in_path) = write_temp(&npz_bytes, "npz");
    let out_path = in_path.with_extension("safetensors");

    let output = Command::new(binary_path())
        .args([
            "convert",
            in_path.to_str().unwrap(),
            "--to",
            "safetensors",
            "-o",
            out_path.to_str().unwrap(),
        ])
        .output()
        .expect("run amn convert");
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "amn convert failed\nstdout: {stdout}\nstderr: {stderr}"
    );
    assert!(out_path.exists(), "output file missing");

    // Re-parse the safetensors output and confirm both tensors landed.
    let bytes = std::fs::read(&out_path).unwrap();
    let parsed = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let mut names: Vec<&str> = parsed.names();
    names.sort_unstable();
    assert_eq!(names, vec!["b", "w"]);
}

#[test]
#[cfg(feature = "gguf")]
fn convert_safetensors_bf16_to_gguf_smokes() {
    // 8 BF16 elements arranged as [2, 4]
    let bf16: Vec<u8> = (0..8u32)
        .flat_map(|i| {
            let v = i as f32;
            // CAST: u32 -> u16 truncation intentional
            let bits = (v.to_bits() >> 16) as u16;
            bits.to_le_bytes()
        })
        .collect();
    let st_bytes = build_safetensors_bf16(&[("w", &[2, 4], &bf16)]);
    let (_dir, in_path) = write_temp(&st_bytes, "safetensors");
    let out_path = in_path.with_file_name("out.gguf");

    let output = Command::new(binary_path())
        .args([
            "convert",
            in_path.to_str().unwrap(),
            "--to",
            "gguf",
            "-o",
            out_path.to_str().unwrap(),
        ])
        .output()
        .expect("run amn convert");
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "amn convert failed\nstdout: {stdout}\nstderr: {stderr}"
    );

    // Re-parse the GGUF output, expect 1 BF16 tensor.
    let parsed = anamnesis::parse_gguf(&out_path).unwrap();
    let info = parsed.inspect();
    assert_eq!(info.tensor_count, 1);
    let collected: Vec<_> = parsed.tensors().collect();
    assert_eq!(collected[0].name, "w");
    assert_eq!(collected[0].dtype, anamnesis::GgufType::BF16);
    // Safetensors shape [2, 4] is row-major, GGUF shape should be reversed [4, 2].
    assert_eq!(collected[0].shape, &[4, 2]);
    assert_eq!(collected[0].data.as_ref(), bf16.as_slice());
}

#[test]
#[cfg(feature = "bnb")]
fn convert_safetensors_bf16_to_bnb_nf4_smokes() {
    // 64 BF16 elements arranged as [64, 1] — exactly one NF4 block of 64.
    let bf16 = bf16_bytes_from_f32_iter((0..64).map(|i| (i as f32 - 31.5) / 32.0));
    let st_bytes = build_safetensors_bf16(&[("linear", &[64, 1], &bf16)]);
    let (_dir, in_path) = write_temp(&st_bytes, "safetensors");
    let out_path = in_path.with_file_name("out-bnb-nf4.safetensors");

    let output = Command::new(binary_path())
        .args([
            "convert",
            in_path.to_str().unwrap(),
            "--to",
            "bnb-nf4",
            "-o",
            out_path.to_str().unwrap(),
        ])
        .output()
        .expect("run amn convert");
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "amn convert failed\nstdout: {stdout}\nstderr: {stderr}"
    );

    // Re-parse — the safetensors should be classified as Bnb4.
    let model = anamnesis::parse(&out_path).unwrap();
    assert_eq!(model.header.scheme, anamnesis::QuantScheme::Bnb4);
    let names: Vec<&str> = model
        .header
        .tensors
        .iter()
        .map(|t| t.name.as_str())
        .collect();
    assert!(names.contains(&"linear.weight"));
    assert!(names.contains(&"linear.weight.absmax"));
    assert!(names.contains(&"linear.weight.quant_map"));
    assert!(names.contains(&"linear.weight.quant_state.bitsandbytes__nf4"));
}

#[test]
#[cfg(all(feature = "npz", feature = "bnb"))]
fn convert_npz_to_bnb_nf4_smokes() {
    // 64 F32 elements as [64, 1] — exactly one NF4 block. Before Phase 6.14 this
    // pair returned `Unsupported` (the NF4 target accepted only a plain-BF16
    // safetensors source); the BF16 hub now routes NPZ -> BF16 -> NF4 encoder.
    let f32_data: Vec<u8> = (0..64)
        .flat_map(|i| ((i as f32 - 31.5) / 32.0).to_le_bytes())
        .collect();
    let npz_bytes = build_npz_f32(&[("linear", &[64, 1], &f32_data)]);
    let (_dir, in_path) = write_temp(&npz_bytes, "npz");
    let out_path = in_path.with_file_name("out-npz-bnb-nf4.safetensors");

    let output = Command::new(binary_path())
        .args([
            "convert",
            in_path.to_str().unwrap(),
            "--to",
            "bnb-nf4",
            "-o",
            out_path.to_str().unwrap(),
        ])
        .output()
        .expect("run amn convert");
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "amn convert npz -> bnb-nf4 failed\nstdout: {stdout}\nstderr: {stderr}"
    );

    let model = anamnesis::parse(&out_path).unwrap();
    assert_eq!(model.header.scheme, anamnesis::QuantScheme::Bnb4);
    let names: Vec<&str> = model
        .header
        .tensors
        .iter()
        .map(|t| t.name.as_str())
        .collect();
    assert!(names.contains(&"linear.weight"));
    assert!(names.contains(&"linear.weight.absmax"));
}

/// `.pth` → `bnb-nf4` (Phase 6.14 Step 2) — previously `Unsupported`; the BF16
/// hub now routes it.
///
/// This fixture pins the **passthrough** half of the encoder contract: its three
/// F32 tensors are `[2, 1]`, `[2, 2]`, `[2, 2]` (2–4 elements), all far below the
/// 64-element NF4 block, so none is eligible and every tensor is written through
/// as `BF16` with no NF4 companions. The hub still performs the `F32` → `BF16`
/// conversion the encoder's input contract requires. The *quantising* half of the
/// contract is covered by the NPZ and GGUF cases above, which use 64-element
/// tensors.
#[test]
#[cfg(all(feature = "pth", feature = "bnb"))]
fn convert_pth_to_bnb_nf4_passes_ineligible_tensors_through() {
    let in_path = std::path::Path::new("tests/fixtures/pth_reference/algzoo_rnn_small.pth");
    assert!(in_path.exists(), "committed .pth fixture missing");
    let dir = tempfile::tempdir().expect("tempdir");
    let out_path = dir.path().join("out-pth-bnb-nf4.safetensors");

    let output = Command::new(binary_path())
        .args([
            "convert",
            in_path.to_str().unwrap(),
            "--to",
            "bnb-nf4",
            "-o",
            out_path.to_str().unwrap(),
        ])
        .output()
        .expect("run amn convert");
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "amn convert pth -> bnb-nf4 failed\nstdout: {stdout}\nstderr: {stderr}"
    );

    let model = anamnesis::parse(&out_path).unwrap();
    let names: Vec<&str> = model
        .header
        .tensors
        .iter()
        .map(|t| t.name.as_str())
        .collect();
    assert_eq!(names.len(), 3, "all three tensors survive: {names:?}");
    assert!(names.contains(&"rnn.weight_ih_l0"));
    assert!(names.contains(&"rnn.weight_hh_l0"));
    assert!(names.contains(&"linear.weight"));
    // Nothing was eligible, so no NF4 companion tensors were emitted.
    assert!(
        !names.iter().any(|n| n.ends_with(".absmax")),
        "no tensor is NF4-eligible, so none should gain companions: {names:?}"
    );
    // The hub converted F32 -> BF16 on the way to the encoder.
    assert!(
        model
            .header
            .tensors
            .iter()
            .all(|t| t.dtype == anamnesis::Dtype::BF16),
        "passthrough tensors should be BF16"
    );
}

/// `gguf` → `bnb-nf4` (Phase 6.14 Step 2): the hub dequantises/normalises the
/// GGUF source to BF16, then the NF4 encoder runs. GGUF stores dimensions
/// most-significant-first, so `[1, 64]` on disk is `[64, 1]` row-major — exactly
/// one NF4 block.
#[test]
#[cfg(all(feature = "gguf", feature = "bnb"))]
fn convert_gguf_to_bnb_nf4_smokes() {
    use std::collections::HashMap;

    let f32_bytes: Vec<u8> = (0..64)
        .flat_map(|i| ((i as f32 - 31.5) / 32.0).to_le_bytes())
        .collect();
    let shape = [1usize, 64];
    let tensors = [anamnesis::GgufWriteTensor {
        name: "linear",
        shape: &shape,
        dtype: anamnesis::GgufType::F32,
        data: &f32_bytes,
    }];

    let dir = tempfile::tempdir().expect("tempdir");
    let in_path = dir.path().join("in.gguf");
    anamnesis::write_gguf(&in_path, &tensors, &HashMap::new()).expect("write gguf fixture");
    let out_path = dir.path().join("out-gguf-bnb-nf4.safetensors");

    let output = Command::new(binary_path())
        .args([
            "convert",
            in_path.to_str().unwrap(),
            "--to",
            "bnb-nf4",
            "-o",
            out_path.to_str().unwrap(),
        ])
        .output()
        .expect("run amn convert");
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "amn convert gguf -> bnb-nf4 failed\nstdout: {stdout}\nstderr: {stderr}"
    );

    let model = anamnesis::parse(&out_path).unwrap();
    assert_eq!(model.header.scheme, anamnesis::QuantScheme::Bnb4);
    let names: Vec<&str> = model
        .header
        .tensors
        .iter()
        .map(|t| t.name.as_str())
        .collect();
    assert!(names.contains(&"linear.weight"));
    assert!(names.contains(&"linear.weight.absmax"));
}

/// `gguf` → `gguf` dequantise-in-place (Phase 6.14 Step 3) must **preserve the
/// source KV**. Re-emitting a GGUF with an empty metadata table would produce a
/// bare tensor container that no inference runtime can load, so the hub carries
/// the source's architecture / hyper-parameter keys into the output.
#[test]
#[cfg(feature = "gguf")]
fn convert_gguf_to_gguf_preserves_metadata() {
    use anamnesis::GgufMetadataValue;
    use std::collections::HashMap;

    let f32_bytes: Vec<u8> = (0..8).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let shape = [8usize];
    let tensors = [anamnesis::GgufWriteTensor {
        name: "w",
        shape: &shape,
        dtype: anamnesis::GgufType::F32,
        data: &f32_bytes,
    }];
    let mut meta: HashMap<String, GgufMetadataValue> = HashMap::new();
    meta.insert(
        "general.architecture".to_owned(),
        GgufMetadataValue::String("llama".to_owned()),
    );
    meta.insert("llama.block_count".to_owned(), GgufMetadataValue::U32(32));

    let dir = tempfile::tempdir().expect("tempdir");
    let in_path = dir.path().join("in.gguf");
    anamnesis::write_gguf(&in_path, &tensors, &meta).expect("write gguf fixture");
    let out_path = dir.path().join("out.gguf");

    let output = Command::new(binary_path())
        .args([
            "convert",
            in_path.to_str().unwrap(),
            "--to",
            "gguf",
            "-o",
            out_path.to_str().unwrap(),
        ])
        .output()
        .expect("run amn convert");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "amn convert gguf -> gguf failed\nstderr: {stderr}"
    );

    let reparsed = anamnesis::parse_gguf(&out_path).expect("re-parse converted gguf");
    let out_meta = reparsed.metadata();
    assert_eq!(
        out_meta.get("general.architecture"),
        Some(&GgufMetadataValue::String("llama".to_owned())),
        "source architecture KV must survive the round trip: {out_meta:?}"
    );
    assert_eq!(
        out_meta.get("llama.block_count"),
        Some(&GgufMetadataValue::U32(32)),
        "typed numeric KV must survive with its type intact: {out_meta:?}"
    );
}

/// A **quantised** safetensors → `gguf` used to fail with a "dequantise first"
/// error that forced the user to stage a temp file by hand. The hub now chains
/// the two hops internally (Phase 6.14 Step 3).
#[test]
#[cfg(feature = "gguf")]
fn convert_quantized_safetensors_to_gguf_auto_chains() {
    let in_path = std::path::Path::new("tests/fixtures/safetensors_reference/fp8.safetensors");
    assert!(in_path.exists(), "committed FP8 fixture missing");
    let dir = tempfile::tempdir().expect("tempdir");
    let out_path = dir.path().join("out.gguf");

    let output = Command::new(binary_path())
        .args([
            "convert",
            in_path.to_str().unwrap(),
            "--to",
            "gguf",
            "-o",
            out_path.to_str().unwrap(),
        ])
        .output()
        .expect("run amn convert");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "quantised safetensors -> gguf should auto-chain\nstdout: {stdout}\nstderr: {stderr}"
    );
    assert!(
        stdout.contains("dequantized to BF16"),
        "expected the chain to report a dequant step, got: {stdout}"
    );

    // The FP8 weight came out as BF16; nothing quantised remains.
    let reparsed = anamnesis::parse_gguf(&out_path).expect("re-parse converted gguf");
    assert!(
        reparsed.tensors().all(|t| !t.dtype.is_quantized()),
        "output GGUF must be scalar (dequantised)"
    );
}

/// `--gguf-metadata` / `--gguf-kv` (Phase 6.14 Step 4) and the precedence chain
/// they form with Step 3's inherited KV: **source KV → JSON file → `--gguf-kv`**,
/// each layer overriding the one before, with untouched source keys surviving.
#[test]
#[cfg(feature = "gguf")]
fn convert_gguf_metadata_flags_merge_over_source_kv() {
    use anamnesis::GgufMetadataValue;
    use std::collections::HashMap;

    let f32_bytes: Vec<u8> = (0..4).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let shape = [4usize];
    let tensors = [anamnesis::GgufWriteTensor {
        name: "w",
        shape: &shape,
        dtype: anamnesis::GgufType::F32,
        data: &f32_bytes,
    }];
    let mut source_meta: HashMap<String, GgufMetadataValue> = HashMap::new();
    source_meta.insert(
        "general.architecture".to_owned(),
        GgufMetadataValue::String("source-arch".to_owned()),
    );
    source_meta.insert(
        "general.name".to_owned(),
        GgufMetadataValue::String("keepme".to_owned()),
    );

    let dir = tempfile::tempdir().expect("tempdir");
    let in_path = dir.path().join("in.gguf");
    anamnesis::write_gguf(&in_path, &tensors, &source_meta).expect("write gguf fixture");

    let meta_path = dir.path().join("meta.json");
    std::fs::write(
        &meta_path,
        r#"{
             "general.architecture": "from-file",
             "llama.block_count": {"type": "u32", "value": 32},
             "tokenizer.ggml.token_type": {"type": "array", "item_type": "i32", "value": [1, 2]}
           }"#,
    )
    .expect("write metadata json");

    let out_path = dir.path().join("out.gguf");
    let output = Command::new(binary_path())
        .args([
            "convert",
            in_path.to_str().unwrap(),
            "--to",
            "gguf",
            "-o",
            out_path.to_str().unwrap(),
            "--gguf-metadata",
            meta_path.to_str().unwrap(),
            "--gguf-kv",
            "general.architecture=from-flag",
        ])
        .output()
        .expect("run amn convert");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "amn convert with metadata flags failed\nstderr: {stderr}"
    );

    let reparsed = anamnesis::parse_gguf(&out_path).expect("re-parse converted gguf");
    let meta = reparsed.metadata();

    // --gguf-kv beats the JSON file, which beats the inherited source value.
    assert_eq!(
        meta.get("general.architecture"),
        Some(&GgufMetadataValue::String("from-flag".to_owned())),
        "flag should win the three-way merge: {meta:?}"
    );
    // A source key nobody overrode survives (Step 3's inheritance).
    assert_eq!(
        meta.get("general.name"),
        Some(&GgufMetadataValue::String("keepme".to_owned())),
        "untouched source KV must survive: {meta:?}"
    );
    // Typed values from the file keep their declared width.
    assert_eq!(
        meta.get("llama.block_count"),
        Some(&GgufMetadataValue::U32(32))
    );
    assert_eq!(
        meta.get("tokenizer.ggml.token_type"),
        Some(&GgufMetadataValue::Array(Box::new(
            anamnesis::GgufMetadataArray::I32(vec![1, 2])
        ))),
        "the explicit item_type must survive as I32, not the inferred U32"
    );
}

/// A malformed `--gguf-kv` is rejected with a message naming the expected shape,
/// rather than being silently dropped.
#[test]
#[cfg(feature = "gguf")]
fn convert_rejects_malformed_gguf_kv() {
    let bf16: Vec<u8> = vec![0u8; 2];
    let st_bytes = build_safetensors_bf16(&[("w", &[1], &bf16)]);
    let (_dir, in_path) = write_temp(&st_bytes, "safetensors");

    let output = Command::new(binary_path())
        .args([
            "convert",
            in_path.to_str().unwrap(),
            "--to",
            "gguf",
            "--gguf-kv",
            "no-equals-sign",
        ])
        .output()
        .expect("run amn convert");
    assert!(
        !output.status.success(),
        "should reject malformed --gguf-kv"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("key=value"),
        "expected a `key=value` hint, got stderr: {stderr}"
    );
}

#[test]
fn convert_unknown_target_errors_cleanly() {
    let bf16: Vec<u8> = vec![0u8; 2];
    let st_bytes = build_safetensors_bf16(&[("w", &[1], &bf16)]);
    let (_dir, in_path) = write_temp(&st_bytes, "safetensors");

    let output = Command::new(binary_path())
        .args(["convert", in_path.to_str().unwrap(), "--to", "frobnicate"])
        .output()
        .expect("run amn convert");
    assert!(!output.status.success(), "should reject unknown target");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("supported convert targets"),
        "expected target-list error, got stderr: {stderr}"
    );
}
