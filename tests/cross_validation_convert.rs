// SPDX-License-Identifier: MIT OR Apache-2.0

//! Cross-format round-trip validation suite.
//!
//! Phase 6 Step 3 (`t1`–`t14`) exercises **every v0.6.0-available conversion
//! pair** through the low-level writers, both directions where the pipeline is
//! reversible. Phase 6.14 Step 5 (`t15`–`t20`) adds the **newly-wired cells**
//! (quantised-safetensors → gguf auto-chain, gguf → gguf, gguf/npz/`.pth` →
//! bnb-nf4) driven through the public [`convert`](anamnesis::convert) entry
//! point, pinning the BF16-hub dispatch, `ConvertStats` accounting, and
//! GGUF-KV inheritance a library consumer actually depends on. Each test
//! follows the same shape:
//!
//! 1. Build (or load) a deterministic fixture.
//! 2. Run the forward conversion.
//! 3. Parse the output back through the appropriate reader.
//! 4. Assert byte-exactness where the pipeline is lossless; assert the
//!    Phase-5 idempotency property where it isn't (BnB-NF4).
//! 5. Time both the forward and (where applicable) reverse legs, and
//!    print a comparison line against the Python equivalent if a sidecar
//!    `.timing.json` file is available (same shape as
//!    `tests/cross_validation_bnb_encode.rs`).
//!
//! All fixtures are built in-test (no external downloads). The single
//! exception is the BnB-NF4 byte-exact-against-Python check (#5) which
//! reuses an existing `tests/fixtures/bnb_reference/llama_1b_nf4.bin`
//! fixture as the Python encoder reference. That fixture is already
//! checked in for the Phase 5 cross-validation suite.

#![cfg(all(feature = "npz", feature = "pth", feature = "gguf", feature = "bnb"))]
#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::similar_names,
    clippy::wildcard_enum_match_arm,
    clippy::same_item_push,
    clippy::ignored_unit_patterns,
    clippy::items_after_statements,
    clippy::used_underscore_binding,
    clippy::redundant_locals,
    clippy::semicolon_if_nothing_returned
)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anamnesis::{
    BnbWriteInput, ConvertOptions, ConvertTarget, GgufMetadataValue, GgufType, GgufWriteTensor,
    NpzDtype, NpzTensor, QuantScheme, classify_inputs, convert, npz_to_safetensors_bytes, parse,
    parse_gguf, parse_pth, pth_to_safetensors_bytes, write_bnb_nf4_safetensors,
    write_bnb_nf4_safetensors_bytes, write_gguf,
};

// ===========================================================================
// Fixture builders (deterministic; pure functions of their inputs)
// ===========================================================================

/// Builds an in-memory BF16 safetensors file from a tensor list. Inputs:
/// (name, row-major shape, BF16 LE bytes). Tensors are emitted in the
/// `safetensors` crate's iteration order — for round-trip tests, that
/// ordering is what we compare against.
fn build_safetensors_bf16(tensors: &[(&str, &[usize], &[u8])]) -> Vec<u8> {
    let views: Vec<(&str, safetensors::tensor::TensorView<'_>)> = tensors
        .iter()
        .map(|(name, shape, data)| {
            let view = safetensors::tensor::TensorView::new(
                safetensors::Dtype::BF16,
                shape.to_vec(),
                data,
            )
            .unwrap();
            (*name, view)
        })
        .collect();
    safetensors::tensor::serialize(views, None).unwrap()
}

/// Builds an in-memory mixed-dtype safetensors file. Each tuple is
/// (name, dtype, shape, bytes).
fn build_safetensors_mixed(tensors: &[(&str, safetensors::Dtype, &[usize], &[u8])]) -> Vec<u8> {
    let views: Vec<(&str, safetensors::tensor::TensorView<'_>)> = tensors
        .iter()
        .map(|(name, dtype, shape, data)| {
            let view = safetensors::tensor::TensorView::new(*dtype, shape.to_vec(), data).unwrap();
            (*name, view)
        })
        .collect();
    safetensors::tensor::serialize(views, None).unwrap()
}

fn write_temp(bytes: &[u8], ext: &str) -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().expect("create temp dir");
    let path = dir.path().join(format!("fixture.{ext}"));
    std::fs::write(&path, bytes).unwrap();
    (dir, path)
}

fn bf16_bytes_from_f32_iter<I: IntoIterator<Item = f32>>(values: I) -> Vec<u8> {
    let mut out = Vec::new();
    for v in values {
        let bits = (v.to_bits() >> 16) as u16;
        out.extend_from_slice(&bits.to_le_bytes());
    }
    out
}

// ===========================================================================
// Timing helpers
// ===========================================================================

/// Times a closure, returns its result plus the elapsed wall time (µs).
fn timed<F: FnOnce() -> R, R>(label: &str, f: F) -> (R, f64) {
    let start = Instant::now();
    let result = f();
    let elapsed_us = start.elapsed().as_nanos() as f64 / 1000.0;
    eprintln!("  [{label}] rust={elapsed_us:.1} \u{00B5}s");
    (result, elapsed_us)
}

/// Python-timing sidecar payload.
///
/// Sidecar JSON shape:
/// ```json
/// {"path_label": "st_to_gguf", "py_seconds": 0.0234,
///  "py_library": "gguf 0.10.0", "shape": [4096, 4096]}
/// ```
struct PythonSidecar {
    elapsed_us: f64,
    shape: Vec<i64>,
    library: String,
}

fn read_python_sidecar(label: &str) -> Option<PythonSidecar> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("convert_reference")
        .join(format!("{label}.timing.json"));
    let bytes = std::fs::read(&p).ok()?;
    let v: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    let seconds = v.get("py_seconds")?.as_f64()?;
    let shape = v
        .get("shape")
        .and_then(serde_json::Value::as_array)
        .map(|arr| arr.iter().filter_map(serde_json::Value::as_i64).collect())
        .unwrap_or_default();
    let library = v
        .get("py_library")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown")
        .to_owned();
    Some(PythonSidecar {
        elapsed_us: seconds * 1_000_000.0,
        shape,
        library,
    })
}

/// Reports the rust-side wall time alongside the Python sidecar.
///
/// `rust_shape` is what the Rust test actually measured at — printed
/// next to Python's `shape` so the reader sees if the comparison is
/// apples-to-apples or overhead-vs-throughput. A pure-ratio line only
/// makes sense when the shapes match; otherwise the line is informative
/// but the bare multiplier is suppressed.
fn report_vs_python(label: &str, rust_us: f64, rust_shape: &[usize]) {
    match read_python_sidecar(label) {
        Some(sidecar) => {
            let shapes_match = sidecar.shape.len() == rust_shape.len()
                && sidecar
                    .shape
                    .iter()
                    .zip(rust_shape.iter())
                    .all(|(a, b)| i64::try_from(*b).is_ok_and(|c| c == *a));
            if shapes_match {
                let ratio = sidecar.elapsed_us / rust_us.max(f64::MIN_POSITIVE);
                eprintln!(
                    "  [{label}] rust={rust_us:.1} \u{00B5}s, python={:.1} \u{00B5}s \
                     ({ratio:.2}\u{00D7}, shape={:?}, {})",
                    sidecar.elapsed_us, sidecar.shape, sidecar.library
                );
            } else {
                eprintln!(
                    "  [{label}] rust={rust_us:.1} \u{00B5}s @ {rust_shape:?}, \
                     python={:.1} \u{00B5}s @ {:?} ({}) — \
                     shapes differ; size-matched ratio printed by t14_perf_vs_python",
                    sidecar.elapsed_us, sidecar.shape, sidecar.library
                );
            }
        }
        None => {
            eprintln!("  [{label}] rust={rust_us:.1} \u{00B5}s (no Python sidecar)");
        }
    }
}

// ===========================================================================
// Assertions with diagnostics
// ===========================================================================

fn assert_bytes_equal_with_diagnostic(left: &[u8], right: &[u8], label: &str) {
    assert_eq!(
        left.len(),
        right.len(),
        "{label}: byte count differs (left={}, right={})",
        left.len(),
        right.len()
    );
    let mut diffs = 0;
    for (i, (a, b)) in left.iter().zip(right.iter()).enumerate() {
        if a != b {
            if diffs < 5 {
                eprintln!("  {label}[{i}]: left=0x{a:02X}, right=0x{b:02X}");
            }
            diffs += 1;
        }
    }
    assert_eq!(diffs, 0, "{label}: {diffs} byte mismatches");
}

// ===========================================================================
// Helpers for building parsed-form inputs
// ===========================================================================

fn make_npz_tensor(name: &str, dtype: NpzDtype, shape: Vec<usize>, data: Vec<u8>) -> NpzTensor {
    NpzTensor {
        name: name.to_owned(),
        shape,
        dtype,
        data,
    }
}

// ===========================================================================
// TESTS — one #[test] per row of the conversion matrix
// ===========================================================================

// -- #1 NPZ -> safetensors --

#[test]
fn t1_npz_to_safetensors_bytes_exact() {
    eprintln!("--- t1_npz_to_safetensors_bytes_exact ---");
    let f32_data: Vec<u8> = (0..6u32).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let bias: Vec<u8> = (0..3u32)
        .flat_map(|i| (i as f32 * 0.25).to_le_bytes())
        .collect();
    let mut map: HashMap<String, NpzTensor> = HashMap::new();
    map.insert(
        "w".into(),
        make_npz_tensor("w", NpzDtype::F32, vec![2, 3], f32_data.clone()),
    );
    map.insert(
        "b".into(),
        make_npz_tensor("b", NpzDtype::F32, vec![3], bias.clone()),
    );

    let (st_bytes, _us) = timed("npz->st (forward)", || {
        npz_to_safetensors_bytes(&map).unwrap()
    });
    report_vs_python("npz_to_st", _us, &[2, 3]);

    let parsed = safetensors::SafeTensors::deserialize(&st_bytes).unwrap();
    let w = parsed.tensor("w").unwrap();
    assert_eq!(w.dtype(), safetensors::Dtype::F32);
    assert_eq!(w.shape(), &[2, 3]);
    assert_bytes_equal_with_diagnostic(w.data(), &f32_data, "t1.w");
    let b = parsed.tensor("b").unwrap();
    assert_eq!(b.dtype(), safetensors::Dtype::F32);
    assert_eq!(b.shape(), &[3]);
    assert_bytes_equal_with_diagnostic(b.data(), &bias, "t1.b");
}

// -- #2 PTH -> safetensors (already shipped at v0.3.1; we cover here so a regression
// in the convert pipeline can be detected).

#[test]
fn t2_pth_via_existing_fixture_to_safetensors_bytes_exact() {
    eprintln!("--- t2_pth_via_existing_fixture_to_safetensors_bytes_exact ---");
    // Use the AlgZoo rnn-small fixture shipped for Phase 3.5.
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("pth_reference")
        .join("algzoo_rnn_small.pth");
    if !fixture.exists() {
        eprintln!("  fixture not present; skipping");
        return;
    }
    let parsed = parse_pth(&fixture).unwrap();
    let pth_tensors = parsed.tensors().unwrap();

    let (st_bytes, _us) = timed("pth->st (forward)", || {
        pth_to_safetensors_bytes(&pth_tensors).unwrap()
    });
    // Shape varies with the AlgZoo fixture — sidecar always disagrees.
    report_vs_python("pth_to_st", _us, &[0]);

    let reread = safetensors::SafeTensors::deserialize(&st_bytes).unwrap();
    assert_eq!(reread.names().len(), pth_tensors.len());
    for t in &pth_tensors {
        let st_t = reread.tensor(&t.name).unwrap();
        assert_eq!(st_t.shape(), t.shape.as_slice(), "t2 shape `{}`", t.name);
        assert_bytes_equal_with_diagnostic(st_t.data(), t.data.as_ref(), &t.name);
    }
}

// -- #3 safetensors-BF16 -> GGUF (one-leg byte-exact)

#[test]
fn t3_st_bf16_to_gguf_bytes_exact() {
    eprintln!("--- t3_st_bf16_to_gguf_bytes_exact ---");
    // 8 BF16 values arranged as [2, 4] (row-major in safetensors).
    let bf16 = bf16_bytes_from_f32_iter((0..8).map(|i| i as f32 * 0.5 - 1.0));

    let st_bytes = build_safetensors_bf16(&[("w", &[2, 4], &bf16)]);
    let (_dir, st_path) = write_temp(&st_bytes, "safetensors");
    let gguf_path = st_path.with_file_name("out.gguf");

    let model = parse(&st_path).unwrap();
    let tensors_owned: Vec<(String, GgufType, Vec<usize>, Vec<u8>)> = model
        .header
        .tensors
        .iter()
        .map(|t| {
            let data_offset = model.header.header_size + 8;
            let start = data_offset + t.data_offsets.0;
            let end = data_offset + t.data_offsets.1;
            let raw = std::fs::read(&st_path).unwrap();
            let bytes = raw[start..end].to_vec();
            let mut msb_first = t.shape.clone();
            msb_first.reverse();
            (t.name.clone(), GgufType::BF16, msb_first, bytes)
        })
        .collect();
    let write_inputs: Vec<GgufWriteTensor<'_>> = tensors_owned
        .iter()
        .map(|(n, d, s, b)| GgufWriteTensor {
            name: n.as_str(),
            shape: s.as_slice(),
            dtype: *d,
            data: b.as_slice(),
        })
        .collect();

    let (_, us) = timed("st->gguf (forward)", || {
        write_gguf(&gguf_path, &write_inputs, &HashMap::new()).unwrap();
    });
    report_vs_python("st_to_gguf", us, &[4, 2]);

    let parsed_gguf = parse_gguf(&gguf_path).unwrap();
    let collected: Vec<_> = parsed_gguf.tensors().collect();
    assert_eq!(collected.len(), 1);
    let t = &collected[0];
    assert_eq!(t.name, "w");
    assert_eq!(t.dtype, GgufType::BF16);
    // GGUF shape is MSB-first; safetensors shape was [2, 4] row-major
    // → GGUF shape should be [4, 2].
    assert_eq!(t.shape, &[4, 2]);
    assert_bytes_equal_with_diagnostic(t.data.as_ref(), &bf16, "t3.w");
}

// -- #4 safetensors-BF16 -> GGUF -> safetensors-BF16 (full byte-exact loop)

#[test]
fn t4_st_bf16_to_gguf_to_st_byte_exact_loop() {
    eprintln!("--- t4_st_bf16_to_gguf_to_st_byte_exact_loop ---");
    let bf16 = bf16_bytes_from_f32_iter((0..12).map(|i| (i as f32 - 5.5) * 0.1));
    let st_bytes = build_safetensors_bf16(&[("w", &[3, 4], &bf16)]);
    let (_dir, st_path) = write_temp(&st_bytes, "safetensors");
    let gguf_path = st_path.with_file_name("ring.gguf");
    let st_path_2 = st_path.with_file_name("ring.safetensors");

    // Forward: ST -> GGUF (we build the GGUF write inputs by reading raw bytes).
    let raw = std::fs::read(&st_path).unwrap();
    let model = parse(&st_path).unwrap();
    let entry = &model.header.tensors[0];
    let data_offset = model.header.header_size + 8;
    let bytes =
        raw[data_offset + entry.data_offsets.0..data_offset + entry.data_offsets.1].to_vec();
    let mut msb_first = entry.shape.clone();
    msb_first.reverse();
    let tensors = vec![GgufWriteTensor {
        name: "w",
        shape: &msb_first,
        dtype: GgufType::BF16,
        data: &bytes,
    }];
    let (_, us_fwd) = timed("st->gguf (forward)", || {
        write_gguf(&gguf_path, &tensors, &HashMap::new()).unwrap();
    });

    // Reverse: GGUF -> ST. We re-serialise the GGUF tensor as a BF16
    // safetensors with the row-major shape recovered.
    let (_, us_rev) = timed("gguf->st (reverse)", || {
        let parsed = parse_gguf(&gguf_path).unwrap();
        let collected: Vec<_> = parsed.tensors().collect();
        assert_eq!(collected.len(), 1);
        let mut row_major = collected[0].shape.to_vec();
        row_major.reverse();
        let st =
            build_safetensors_bf16(&[(collected[0].name, &row_major, collected[0].data.as_ref())]);
        std::fs::write(&st_path_2, &st).unwrap();
    });
    eprintln!("  (loop: forward {us_fwd:.1} \u{00B5}s + reverse {us_rev:.1} \u{00B5}s)");

    // Final compare: the safetensors at the end of the loop must contain
    // a BF16 tensor with shape [3, 4] and bytes identical to the source.
    let final_bytes = std::fs::read(&st_path_2).unwrap();
    let parsed = safetensors::SafeTensors::deserialize(&final_bytes).unwrap();
    let t = parsed.tensor("w").unwrap();
    assert_eq!(t.dtype(), safetensors::Dtype::BF16);
    assert_eq!(t.shape(), &[3, 4]);
    assert_bytes_equal_with_diagnostic(t.data(), &bf16, "t4.w");
}

// -- #5 BnB-NF4 byte-exact against an existing Python (bitsandbytes) encoded
// fixture (Phase 5 cross-validation oracle).

#[test]
fn t5_bnb_nf4_byte_exact_vs_python_reference() {
    eprintln!("--- t5_bnb_nf4_byte_exact_vs_python_reference ---");
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("bnb_reference")
        .join("llama_1b_nf4.bin");
    if !fixture.exists() {
        eprintln!("  fixture not present; skipping");
        return;
    }
    let raw = std::fs::read(&fixture).unwrap();
    // Fixture header layout, container v2 (cross_validation_bnb_encode.rs):
    //   [4]AMNB magic, u32 version,
    //   u32 format_id, u32 total_elements, u32 block_size,
    //   u32 weight_len, u32 absmax_len, u32 quant_map_len,
    //   u32 nested_absmax_len, u32 nested_quant_map_len,
    //   u32 bf16_len, u32 f32_len,
    //   f32 nested_offset (0.0 for plain fixtures)
    let read_u32 = |off: usize| u32::from_le_bytes(raw[off..off + 4].try_into().unwrap());
    assert_eq!(
        &raw[..4],
        b"AMNB",
        "fixture is not a v2 `AMNB` container — regenerate with \
         tests/fixtures/bnb_reference/generate_bnb.py"
    );
    assert_eq!(read_u32(4), 2, "unsupported fixture container version");
    let format_id = read_u32(8);
    assert_eq!(format_id, 0, "expect plain (non-DQ) NF4 fixture");
    let total_elements = read_u32(12) as usize;
    let block_size = read_u32(16) as usize;
    let weight_len = read_u32(20) as usize;
    let absmax_len = read_u32(24) as usize;
    let quant_map_len = read_u32(28) as usize;
    let _expected_len = read_u32(40) as usize;
    let mut off = 52;
    let weight_data = raw[off..off + weight_len].to_vec();
    off += weight_len;
    let absmax_data = raw[off..off + absmax_len].to_vec();
    off += absmax_len;
    let quant_map_data = raw[off..off + quant_map_len].to_vec();

    // Decode to BF16 with the validated kernel, then re-encode through
    // the Phase-6 high-level writer. The resulting safetensors's
    // `weight` tensor bytes must equal the original Python-encoded
    // `weight_data`.
    let bf16 = anamnesis::remember::bnb::dequantize_bnb4_to_bf16(
        &weight_data,
        &absmax_data,
        &quant_map_data,
        total_elements,
        block_size,
    )
    .unwrap();

    // Recover the original shape: the fixture is 1-D `[total_elements]`
    // here. The BnB writer eligibility policy requires 2-D, so we
    // reshape to `[total_elements / block_size, block_size]` (a valid
    // 2-D shape whose product matches).
    let shape = [total_elements / block_size, block_size];
    let inputs = vec![BnbWriteInput {
        name: "weight",
        shape: &shape,
        bf16_data: &bf16,
    }];
    let (st_bytes, us) = timed("st->bnb-nf4 (forward)", || {
        write_bnb_nf4_safetensors_bytes(&inputs).unwrap()
    });
    // The bnb_reference fixture is 1-D total_elements; sidecar uses 2-D
    // so shapes always disagree by construction. t14 emits the
    // size-matched ratio.
    report_vs_python("st_to_bnb_nf4", us, &[0]);

    let parsed = safetensors::SafeTensors::deserialize(&st_bytes).unwrap();
    let weight = parsed.tensor("weight.weight").unwrap();
    assert_bytes_equal_with_diagnostic(weight.data(), &weight_data, "t5.weight");
    // Bonus: absmax should also be byte-exact since we re-derive it from
    // the BF16 the original decoder produced (Phase 5 round-trip
    // guarantee).
    let absmax = parsed.tensor("weight.weight.absmax").unwrap();
    assert_bytes_equal_with_diagnostic(absmax.data(), &absmax_data, "t5.absmax");
}

// -- #6 BnB-NF4 idempotency: encode(decode(encode(x))) == encode(x)

#[test]
fn t6_bnb_nf4_encode_is_idempotent() {
    eprintln!("--- t6_bnb_nf4_encode_is_idempotent ---");
    let bf16 = bf16_bytes_from_f32_iter((0..64).map(|i| (i as f32 - 31.5) / 32.0));
    let inputs = vec![BnbWriteInput {
        name: "w",
        shape: &[64, 1],
        bf16_data: &bf16,
    }];

    let st_bytes_1 = write_bnb_nf4_safetensors_bytes(&inputs).unwrap();
    let parsed1 = safetensors::SafeTensors::deserialize(&st_bytes_1).unwrap();
    let w1 = parsed1.tensor("w.weight").unwrap();
    let abs1 = parsed1.tensor("w.weight.absmax").unwrap();
    let qmap1 = parsed1.tensor("w.weight.quant_map").unwrap();

    // Decode back to BF16 and re-encode. The second encoding must equal
    // the first byte-for-byte.
    let bf16_round = anamnesis::remember::bnb::dequantize_bnb4_to_bf16(
        w1.data(),
        abs1.data(),
        qmap1.data(),
        64,
        64,
    )
    .unwrap();
    let inputs2 = vec![BnbWriteInput {
        name: "w",
        shape: &[64, 1],
        bf16_data: &bf16_round,
    }];
    let st_bytes_2 = write_bnb_nf4_safetensors_bytes(&inputs2).unwrap();
    let parsed2 = safetensors::SafeTensors::deserialize(&st_bytes_2).unwrap();
    let w2 = parsed2.tensor("w.weight").unwrap();
    assert_bytes_equal_with_diagnostic(w1.data(), w2.data(), "t6.weight");
}

// -- #7 NPZ -> safetensors -> GGUF multi-hop (F32 byte exactness)

#[test]
fn t7_npz_to_st_to_gguf_byte_exact() {
    eprintln!("--- t7_npz_to_st_to_gguf_byte_exact ---");
    let f32_data: Vec<u8> = (0..6u32).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let mut map: HashMap<String, NpzTensor> = HashMap::new();
    map.insert(
        "w".into(),
        make_npz_tensor("w", NpzDtype::F32, vec![2, 3], f32_data.clone()),
    );

    let (st_bytes, _) = timed("npz->st", || npz_to_safetensors_bytes(&map).unwrap());
    let (_dir, st_path) = write_temp(&st_bytes, "safetensors");
    let gguf_path = st_path.with_file_name("multi.gguf");

    let mut msb_first = vec![2usize, 3];
    msb_first.reverse();
    let tensors = vec![GgufWriteTensor {
        name: "w",
        shape: &msb_first,
        dtype: GgufType::F32,
        data: &f32_data,
    }];
    let (_, _) = timed("st->gguf", || {
        write_gguf(&gguf_path, &tensors, &HashMap::new()).unwrap()
    });

    let parsed = parse_gguf(&gguf_path).unwrap();
    let collected: Vec<_> = parsed.tensors().collect();
    assert_eq!(collected.len(), 1);
    assert_eq!(collected[0].dtype, GgufType::F32);
    assert_bytes_equal_with_diagnostic(collected[0].data.as_ref(), &f32_data, "t7.w");
}

// -- #8 Mixed-dtype safetensors -> GGUF (BF16 + F32 + I32)

#[test]
fn t8_mixed_dtype_st_to_gguf() {
    eprintln!("--- t8_mixed_dtype_st_to_gguf ---");
    let bf16 = bf16_bytes_from_f32_iter((0..4).map(|i| i as f32 * 0.25));
    let f32_data: Vec<u8> = (0..4u32).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let i32_data: Vec<u8> = (0..2i32).flat_map(i32::to_le_bytes).collect();
    let st_bytes = build_safetensors_mixed(&[
        ("a", safetensors::Dtype::BF16, &[4], &bf16),
        ("b", safetensors::Dtype::F32, &[4], &f32_data),
        ("c", safetensors::Dtype::I32, &[2], &i32_data),
    ]);
    let (_dir, st_path) = write_temp(&st_bytes, "safetensors");
    let gguf_path = st_path.with_file_name("mix.gguf");

    let inputs = vec![
        GgufWriteTensor {
            name: "a",
            shape: &[4],
            dtype: GgufType::BF16,
            data: &bf16,
        },
        GgufWriteTensor {
            name: "b",
            shape: &[4],
            dtype: GgufType::F32,
            data: &f32_data,
        },
        GgufWriteTensor {
            name: "c",
            shape: &[2],
            dtype: GgufType::I32,
            data: &i32_data,
        },
    ];
    write_gguf(&gguf_path, &inputs, &HashMap::new()).unwrap();

    let parsed = parse_gguf(&gguf_path).unwrap();
    let collected: Vec<_> = parsed.tensors().collect();
    assert_eq!(collected.len(), 3);
    // Tensors retain insertion order.
    assert_eq!(collected[0].name, "a");
    assert_eq!(collected[0].dtype, GgufType::BF16);
    assert_bytes_equal_with_diagnostic(collected[0].data.as_ref(), &bf16, "t8.a");
    assert_eq!(collected[1].name, "b");
    assert_eq!(collected[1].dtype, GgufType::F32);
    assert_bytes_equal_with_diagnostic(collected[1].data.as_ref(), &f32_data, "t8.b");
    assert_eq!(collected[2].name, "c");
    assert_eq!(collected[2].dtype, GgufType::I32);
    assert_bytes_equal_with_diagnostic(collected[2].data.as_ref(), &i32_data, "t8.c");
}

// -- #9 GGUF (unquantised) -> safetensors -> GGUF (full byte-exact loop)

#[test]
fn t9_gguf_to_st_to_gguf_byte_exact_loop() {
    eprintln!("--- t9_gguf_to_st_to_gguf_byte_exact_loop ---");
    let f32_data: Vec<u8> = (0..6u32)
        .flat_map(|i| (i as f32 * 1.5).to_le_bytes())
        .collect();
    let (_dir, gguf_path_a) = write_temp(&[], "gguf");
    let gguf_path_a = gguf_path_a; // for clarity
    let gguf_path_b = gguf_path_a.with_file_name("ring_b.gguf");
    let st_path = gguf_path_a.with_file_name("ring_mid.safetensors");

    // Write GGUF #1.
    let tensors = vec![GgufWriteTensor {
        name: "w",
        shape: &[3, 2],
        dtype: GgufType::F32,
        data: &f32_data,
    }];
    write_gguf(&gguf_path_a, &tensors, &HashMap::new()).unwrap();

    // GGUF #1 -> safetensors (manually mirror the CLI dispatch path).
    let parsed_a = parse_gguf(&gguf_path_a).unwrap();
    let collected_a: Vec<_> = parsed_a.tensors().collect();
    let mut row_major = collected_a[0].shape.to_vec();
    row_major.reverse();
    let st_bytes = build_safetensors_mixed(&[(
        collected_a[0].name,
        safetensors::Dtype::F32,
        &row_major,
        collected_a[0].data.as_ref(),
    )]);
    std::fs::write(&st_path, &st_bytes).unwrap();

    // Safetensors -> GGUF #2.
    let model = parse(&st_path).unwrap();
    let entry = &model.header.tensors[0];
    // Orientation pin (non-square): the GGUF header declares the shape
    // most-significant-first (`ne` order, [3, 2] here); the safetensors
    // side must carry the NumPy/torch row-major REVERSE ([2, 3]) — the
    // standard orientation a framework loads. Pins the reversal pair in
    // the GGUF→st and st→GGUF dispatch paths against regression.
    assert_eq!(collected_a[0].shape, &[3, 2], "GGUF-side shape (ne order)");
    assert_eq!(
        entry.shape,
        vec![2, 3],
        "safetensors-side shape must be the reverse of the GGUF ne order"
    );
    let raw = std::fs::read(&st_path).unwrap();
    let data_offset = model.header.header_size + 8;
    let mid_bytes =
        raw[data_offset + entry.data_offsets.0..data_offset + entry.data_offsets.1].to_vec();
    let mut msb = entry.shape.clone();
    msb.reverse();
    let tensors_b = vec![GgufWriteTensor {
        name: "w",
        shape: &msb,
        dtype: GgufType::F32,
        data: &mid_bytes,
    }];
    write_gguf(&gguf_path_b, &tensors_b, &HashMap::new()).unwrap();

    // Parse both GGUF files; the tensor data must match byte-for-byte.
    let parsed_b = parse_gguf(&gguf_path_b).unwrap();
    let collected_b: Vec<_> = parsed_b.tensors().collect();
    assert_eq!(collected_b.len(), 1);
    assert_eq!(collected_a[0].shape, collected_b[0].shape);
    assert_bytes_equal_with_diagnostic(
        collected_a[0].data.as_ref(),
        collected_b[0].data.as_ref(),
        "t9.w",
    );
}

// -- #10 Non-default alignment

#[test]
fn t10_non_default_gguf_alignment_8() {
    eprintln!("--- t10_non_default_gguf_alignment_8 ---");
    use anamnesis::GgufMetadataValue;
    let f32_data: Vec<u8> = (0..4u32).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let (_dir, _tmp) = write_temp(&[], "gguf");
    let gguf_path = _tmp.with_file_name("aligned.gguf");
    let mut metadata = HashMap::new();
    metadata.insert("general.alignment".to_owned(), GgufMetadataValue::U32(8));
    let tensors = vec![GgufWriteTensor {
        name: "w",
        shape: &[4],
        dtype: GgufType::F32,
        data: &f32_data,
    }];
    write_gguf(&gguf_path, &tensors, &metadata).unwrap();
    let parsed = parse_gguf(&gguf_path).unwrap();
    assert_eq!(parsed.alignment(), 8);
    for info in parsed.tensor_info() {
        assert_eq!(info.data_offset % 8, 0, "tensor `{}` misaligned", info.name);
    }
    let collected: Vec<_> = parsed.tensors().collect();
    assert_bytes_equal_with_diagnostic(collected[0].data.as_ref(), &f32_data, "t10.w");
}

// -- #11 Empty GGUF (metadata only, 0 tensors)

#[test]
fn t11_empty_gguf_metadata_only_roundtrip() {
    eprintln!("--- t11_empty_gguf_metadata_only_roundtrip ---");
    use anamnesis::GgufMetadataValue;
    let (_dir, gguf_path) = write_temp(&[], "gguf");
    let gguf_path = gguf_path.with_file_name("empty.gguf");
    let mut metadata = HashMap::new();
    metadata.insert(
        "general.architecture".into(),
        GgufMetadataValue::String("anamnesis-empty".into()),
    );
    metadata.insert(
        "general.name".into(),
        GgufMetadataValue::String("phase-6-test".into()),
    );
    write_gguf(&gguf_path, &[], &metadata).unwrap();

    let parsed = parse_gguf(&gguf_path).unwrap();
    assert!(parsed.is_empty());
    assert_eq!(
        parsed
            .metadata()
            .get("general.architecture")
            .and_then(GgufMetadataValue::as_string),
        Some("anamnesis-empty")
    );
    assert_eq!(
        parsed
            .metadata()
            .get("general.name")
            .and_then(GgufMetadataValue::as_string),
        Some("phase-6-test")
    );
}

// -- #12 Multi-dimensional tensor shape round-trip (4-D, 3-D, 1-D)

#[test]
fn t12_multidimensional_shapes_roundtrip() {
    eprintln!("--- t12_multidimensional_shapes_roundtrip ---");
    // 4-D F32: [2, 2, 2, 2] = 16 elements
    let d4: Vec<u8> = (0..16u32).flat_map(|i| (i as f32).to_le_bytes()).collect();
    // 3-D BF16: [2, 3, 4] = 24 elements
    let d3 = bf16_bytes_from_f32_iter((0..24).map(|i| i as f32 * 0.1));
    // 1-D I32: [5]
    let d1: Vec<u8> = (0..5i32).flat_map(i32::to_le_bytes).collect();

    let (_dir, st_path) = write_temp(&[], "safetensors");
    let gguf_path = st_path.with_file_name("multidim.gguf");

    let inputs = vec![
        GgufWriteTensor {
            name: "d4",
            shape: &[2, 2, 2, 2],
            dtype: GgufType::F32,
            data: &d4,
        },
        GgufWriteTensor {
            name: "d3",
            shape: &[2, 3, 4],
            dtype: GgufType::BF16,
            data: &d3,
        },
        GgufWriteTensor {
            name: "d1",
            shape: &[5],
            dtype: GgufType::I32,
            data: &d1,
        },
    ];
    write_gguf(&gguf_path, &inputs, &HashMap::new()).unwrap();

    let parsed = parse_gguf(&gguf_path).unwrap();
    let collected: Vec<_> = parsed.tensors().collect();
    assert_eq!(collected.len(), 3);
    assert_eq!(collected[0].shape, &[2, 2, 2, 2]);
    assert_eq!(collected[1].shape, &[2, 3, 4]);
    assert_eq!(collected[2].shape, &[5]);
    assert_bytes_equal_with_diagnostic(collected[0].data.as_ref(), &d4, "t12.d4");
    assert_bytes_equal_with_diagnostic(collected[1].data.as_ref(), &d3, "t12.d3");
    assert_bytes_equal_with_diagnostic(collected[2].data.as_ref(), &d1, "t12.d1");
}

// ===========================================================================
// BnB-NF4 classifier sanity test — ensures the eligibility heuristic
// agrees with the writer.
// ===========================================================================

#[test]
fn t13_classify_inputs_agrees_with_writer_emission() {
    eprintln!("--- t13_classify_inputs_agrees_with_writer_emission ---");
    let bf16_a = bf16_bytes_from_f32_iter((0..64).map(|i| i as f32 * 0.01));
    let bf16_b = bf16_bytes_from_f32_iter([1.0, 2.0, 3.0]);
    let inputs = vec![
        BnbWriteInput {
            name: "w",
            shape: &[64, 1],
            bf16_data: &bf16_a,
        },
        BnbWriteInput {
            name: "b",
            shape: &[3],
            bf16_data: &bf16_b,
        },
    ];
    let stats = classify_inputs(&inputs);
    assert_eq!(stats.quantized, 1);
    assert_eq!(stats.passthrough, 1);

    let (_dir, out_path) = write_temp(&[], "safetensors");
    let out_path = out_path.with_file_name("classify.safetensors");
    write_bnb_nf4_safetensors(&inputs, &out_path).unwrap();
    let model = parse(&out_path).unwrap();
    // The output should be Bnb4 (one quantised tensor present).
    assert_eq!(model.header.scheme, QuantScheme::Bnb4);
    // The 1-D passthrough must appear as `b` (BF16).
    let b_entry = model
        .header
        .tensors
        .iter()
        .find(|t| t.name == "b")
        .expect("passthrough `b` missing");
    assert_eq!(b_entry.dtype.to_string(), "BF16");
}

// ===========================================================================
// Phase 6.14 Step 5 — the newly-wired cells, driven through the public
// `convert()` API (not the low-level writers the tests above exercise).
//
// t1–t13 pin the *primitives*; these pin the *dispatch*: format detection,
// the BF16-hub reader/writer routing, `ConvertStats` accounting, GGUF-KV
// inheritance, and the auto-chain — all through the single entry point a
// library consumer (and the Phase 8 Python binding) actually calls. Cells
// covered: quantised-safetensors → gguf (auto-chain), gguf → gguf (KV +
// byte-exact scalar), gguf → bnb-nf4, npz → bnb-nf4, `.pth` → bnb-nf4.
// ===========================================================================

/// Writes a scalar `GGUF` fixture to a temp file, returning its dir + path.
/// The dir is returned so the caller keeps it alive for the file's lifetime.
fn write_scalar_gguf(
    tensors: &[GgufWriteTensor<'_>],
    metadata: &HashMap<String, GgufMetadataValue>,
) -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().expect("create temp dir");
    let path = dir.path().join("source.gguf");
    write_gguf(&path, tensors, metadata).unwrap();
    (dir, path)
}

// -- #15 quantised safetensors -> GGUF: the auto-chain (dequant-in, scalar-out).
// Not reversible (FP8 is lossy), so we assert the *routing*: the quantised
// weight is dequantised to BF16 on the way in, and the result is a loadable
// scalar GGUF.

#[test]
fn t15_convert_fp8_st_to_gguf_auto_chains_through_bf16() {
    eprintln!("--- t15_convert_fp8_st_to_gguf_auto_chains_through_bf16 ---");
    let input = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("safetensors_reference")
        .join("fp8.safetensors");
    if !input.exists() {
        eprintln!("  committed FP8 fixture missing; skipping");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("fp8-gguf.gguf");

    let stats = convert(&input, ConvertTarget::Gguf, &out, &ConvertOptions::new())
        .expect("fp8 safetensors -> gguf");
    // The FP8 weight is recovered to BF16 on the way in; the scalar norm
    // passes through. No temp file was staged by the caller.
    assert!(
        stats.dequantized >= 1,
        "the FP8 weight must dequantise through the hub: {stats:?}"
    );
    assert_eq!(stats.quantized, 0, "the gguf target quantises nothing");
    assert_eq!(
        stats.dequantized + stats.passthrough,
        stats.tensors,
        "ConvertStats must partition the written tensors: {stats:?}"
    );

    // The output is a real GGUF whose recovered weight is BF16.
    let parsed = parse_gguf(&out).unwrap();
    let weight = parsed
        .tensors()
        .find(|t| t.name == "model.layers.0.weight")
        .expect("recovered weight tensor missing from output GGUF");
    assert_eq!(
        weight.dtype,
        GgufType::BF16,
        "the auto-chain must land the quantised weight as BF16"
    );
}

// -- #16 GGUF -> GGUF: dequantise-in-place on a scalar source is a byte-exact
// passthrough, and the source metadata KV survives with types intact (the
// Step 3 regression: a KV-less re-emit is a bare tensor container no runtime
// can load).

#[test]
fn t16_convert_gguf_to_gguf_preserves_kv_and_bytes() {
    eprintln!("--- t16_convert_gguf_to_gguf_preserves_kv_and_bytes ---");
    let f32_data: Vec<u8> = (0..6u32)
        .flat_map(|i| (i as f32 * 1.5 - 2.0).to_le_bytes())
        .collect();
    let mut metadata = HashMap::new();
    metadata.insert(
        "general.architecture".to_owned(),
        GgufMetadataValue::String("anamnesis-test".to_owned()),
    );
    // A typed U32 key: its width must survive the round trip, not collapse to
    // a string.
    metadata.insert("test.block_count".to_owned(), GgufMetadataValue::U32(7));
    let tensors = vec![GgufWriteTensor {
        name: "w",
        shape: &[3, 2], // GGUF MSB-first
        dtype: GgufType::F32,
        data: &f32_data,
    }];
    let (_dir, src) = write_scalar_gguf(&tensors, &metadata);
    let out_dir = tempfile::tempdir().unwrap();
    let out = out_dir.path().join("gguf-gguf.gguf");

    let stats =
        convert(&src, ConvertTarget::Gguf, &out, &ConvertOptions::new()).expect("gguf -> gguf");
    assert_eq!(stats.dequantized, 0, "a scalar source dequantises nothing");
    assert_eq!(stats.passthrough, 1);
    assert_eq!(stats.tensors, 1);

    let parsed = parse_gguf(&out).unwrap();
    // Byte-exact: a scalar GGUF -> GGUF is losslessly reversible.
    let collected: Vec<_> = parsed.tensors().collect();
    assert_eq!(collected.len(), 1);
    assert_eq!(collected[0].name, "w");
    assert_eq!(collected[0].dtype, GgufType::F32);
    assert_bytes_equal_with_diagnostic(collected[0].data.as_ref(), &f32_data, "t16.w");

    // KV survives with types intact.
    assert_eq!(
        parsed
            .metadata()
            .get("general.architecture")
            .and_then(GgufMetadataValue::as_string),
        Some("anamnesis-test"),
        "source String KV must be inherited"
    );
    assert_eq!(
        parsed.metadata().get("test.block_count"),
        Some(&GgufMetadataValue::U32(7)),
        "source typed U32 KV must survive as U32, not collapse to String"
    );
}

// -- #17 GGUF -> GGUF with caller KV: `--gguf-metadata` merges OVER the
// inherited source KV (caller wins on a key collision).

#[test]
fn t17_convert_gguf_to_gguf_caller_kv_overrides_source() {
    eprintln!("--- t17_convert_gguf_to_gguf_caller_kv_overrides_source ---");
    let f32_data: Vec<u8> = (0..4u32).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let mut src_kv = HashMap::new();
    src_kv.insert(
        "general.name".to_owned(),
        GgufMetadataValue::String("from-source".to_owned()),
    );
    src_kv.insert(
        "general.architecture".to_owned(),
        GgufMetadataValue::String("llama".to_owned()),
    );
    let tensors = vec![GgufWriteTensor {
        name: "w",
        shape: &[4],
        dtype: GgufType::F32,
        data: &f32_data,
    }];
    let (_dir, src) = write_scalar_gguf(&tensors, &src_kv);
    let out_dir = tempfile::tempdir().unwrap();
    let out = out_dir.path().join("merged.gguf");

    // The caller overrides `general.name` and adds a new key; the untouched
    // `general.architecture` is inherited from the source.
    let mut caller_kv = HashMap::new();
    caller_kv.insert(
        "general.name".to_owned(),
        GgufMetadataValue::String("from-caller".to_owned()),
    );
    caller_kv.insert(
        "general.quantization_version".to_owned(),
        GgufMetadataValue::U32(2),
    );
    let opts = ConvertOptions::new().with_gguf_metadata(caller_kv);

    convert(&src, ConvertTarget::Gguf, &out, &opts).expect("gguf -> gguf with caller KV");

    let parsed = parse_gguf(&out).unwrap();
    assert_eq!(
        parsed
            .metadata()
            .get("general.name")
            .and_then(GgufMetadataValue::as_string),
        Some("from-caller"),
        "caller KV must win on a key collision"
    );
    assert_eq!(
        parsed
            .metadata()
            .get("general.architecture")
            .and_then(GgufMetadataValue::as_string),
        Some("llama"),
        "untouched source KV must still be inherited"
    );
    assert_eq!(
        parsed.metadata().get("general.quantization_version"),
        Some(&GgufMetadataValue::U32(2)),
        "caller-only key must be written"
    );
}

// -- #18 GGUF -> BnB-NF4: a 2-D one-block weight is encoded to NF4 (companions
// present), a 1-D bias passes through as BF16. Pins the encoder contract via
// the GGUF reader rather than the low-level writer (t5/t6).

#[test]
fn t18_convert_gguf_to_bnb_nf4_encodes_weight_passes_bias() {
    eprintln!("--- t18_convert_gguf_to_bnb_nf4_encodes_weight_passes_bias ---");
    // 64 BF16 weight values as [64, 1] row-major -> GGUF stores [1, 64].
    let w_bf16 = bf16_bytes_from_f32_iter((0..64).map(|i| (i as f32 - 31.5) / 32.0));
    let b_bf16 = bf16_bytes_from_f32_iter([0.5, -0.25, 1.0]);
    let tensors = vec![
        GgufWriteTensor {
            name: "w",
            shape: &[1, 64], // GGUF MSB-first for a row-major [64, 1]
            dtype: GgufType::BF16,
            data: &w_bf16,
        },
        GgufWriteTensor {
            name: "b",
            shape: &[3],
            dtype: GgufType::BF16,
            data: &b_bf16,
        },
    ];
    let (_dir, src) = write_scalar_gguf(&tensors, &HashMap::new());
    let out_dir = tempfile::tempdir().unwrap();
    let out = out_dir.path().join("gguf-bnb.safetensors");

    let stats = convert(&src, ConvertTarget::BnbNf4, &out, &ConvertOptions::new())
        .expect("gguf -> bnb-nf4");
    assert_eq!(
        stats.quantized, 1,
        "the 2-D one-block weight is NF4-encoded"
    );
    assert_eq!(stats.passthrough, 1, "the 1-D bias passes through as BF16");

    // The output must carry the NF4 companions for `w` and a bare BF16 `b`.
    let model = parse(&out).unwrap();
    assert_eq!(model.header.scheme, QuantScheme::Bnb4);
    let out_bytes = std::fs::read(&out).unwrap();
    let parsed = safetensors::SafeTensors::deserialize(&out_bytes).unwrap();
    assert!(
        parsed.tensor("w.weight").is_ok(),
        "NF4-packed weight missing"
    );
    assert!(
        parsed.tensor("w.weight.absmax").is_ok(),
        "NF4 absmax companion missing"
    );
    assert!(
        parsed.tensor("w.weight.quant_map").is_ok(),
        "NF4 quant_map companion missing"
    );
    let b = parsed.tensor("b").expect("passthrough bias missing");
    assert_eq!(b.dtype(), safetensors::Dtype::BF16);
}

// -- #19 NPZ -> BnB-NF4: end-to-end through the real gemma-scope fixture. Its
// tensors are 1-D 8-element spot slices, so every one passes through as BF16 —
// what we pin here is that the npz->bnb-nf4 cell routes at all (it returned
// `Unsupported` before Step 1) and yields a valid BnB safetensors.

#[test]
fn t19_convert_npz_to_bnb_nf4_routes_and_passes_through() {
    eprintln!("--- t19_convert_npz_to_bnb_nf4_routes_and_passes_through ---");
    let input = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("npz_reference")
        .join("gemma_scope_small.npz");
    if !input.exists() {
        eprintln!("  gemma-scope NPZ fixture missing; skipping");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("npz-bnb.safetensors");

    let stats = convert(&input, ConvertTarget::BnbNf4, &out, &ConvertOptions::new())
        .expect("npz -> bnb-nf4");
    assert!(stats.tensors > 0, "the fixture has tensors");
    assert_eq!(
        stats.quantized, 0,
        "1-D spot slices are all below the 2-D/one-block bar"
    );
    assert_eq!(
        stats.passthrough, stats.tensors,
        "every tensor passes through: {stats:?}"
    );

    // Every written tensor is plain BF16 (the passthrough contract).
    let out_bytes = std::fs::read(&out).unwrap();
    let parsed = safetensors::SafeTensors::deserialize(&out_bytes).unwrap();
    for name in parsed.names() {
        let t = parsed.tensor(name).unwrap();
        assert_eq!(
            t.dtype(),
            safetensors::Dtype::BF16,
            "passthrough tensor `{name}` must be BF16"
        );
    }
}

// -- #20 .pth -> BnB-NF4: the AlgZoo fixture's tensors are all below the
// one-block bar, so this is the passthrough half of the encoder contract
// reached through the `.pth` reader.

#[test]
fn t20_convert_pth_to_bnb_nf4_routes_and_passes_through() {
    eprintln!("--- t20_convert_pth_to_bnb_nf4_routes_and_passes_through ---");
    let input = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("pth_reference")
        .join("algzoo_rnn_small.pth");
    if !input.exists() {
        eprintln!("  AlgZoo .pth fixture missing; skipping");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("pth-bnb.safetensors");

    let stats = convert(&input, ConvertTarget::BnbNf4, &out, &ConvertOptions::new())
        .expect("pth -> bnb-nf4");
    assert!(stats.tensors > 0, "the fixture has tensors");
    assert_eq!(
        stats.quantized, 0,
        "AlgZoo tensors are below the one-block bar"
    );
    assert_eq!(
        stats.passthrough, stats.tensors,
        "every tensor passes through: {stats:?}"
    );

    let out_bytes = std::fs::read(&out).unwrap();
    let parsed = safetensors::SafeTensors::deserialize(&out_bytes).unwrap();
    for name in parsed.names() {
        let t = parsed.tensor(name).unwrap();
        assert_eq!(
            t.dtype(),
            safetensors::Dtype::BF16,
            "passthrough tensor `{name}` must be BF16"
        );
    }
}

// ===========================================================================
// Size-matched performance comparison vs Python (CPU only)
// ===========================================================================
//
// The other 13 tests use 4-to-64-byte synthetic fixtures to keep the
// correctness suite fast and deterministic. At those sizes Rust's
// per-call overhead dominates, so any "vs Python" ratio printed from
// them is misleading. This test runs each conversion on the **same
// 4096x4096 shape** the Python sidecar generator uses (~32 MiB BF16),
// so the elapsed times compare apples-to-apples.
//
// Gated `#[ignore]` so default `cargo test` stays fast; opt in with
// `cargo test --all-features --test cross_validation_convert -- --ignored --nocapture`.

const PERF_SHAPE: (usize, usize) = (4096, 4096);

fn synth_bf16_perf() -> Vec<u8> {
    let n = PERF_SHAPE.0 * PERF_SHAPE.1;
    let mut out = Vec::with_capacity(n * 2);
    for i in 0..n {
        let v = (i as f32 - (n as f32) * 0.5) / (n as f32);
        let bits = (v.to_bits() >> 16) as u16;
        out.extend_from_slice(&bits.to_le_bytes());
    }
    out
}

fn synth_f32_perf() -> Vec<u8> {
    let n = PERF_SHAPE.0 * PERF_SHAPE.1;
    let mut out = Vec::with_capacity(n * 4);
    for i in 0..n {
        let v = (i as f32 - (n as f32) * 0.5) / (n as f32);
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

#[test]
#[ignore = "perf comparison vs Python; run with --ignored"]
fn t14_perf_vs_python_size_matched() {
    eprintln!("--- t14_perf_vs_python_size_matched (shape={PERF_SHAPE:?}) ---");
    eprintln!(
        "  Each Rust measurement is reported against ALL applicable Python sidecars: \
         the ecosystem-default baseline (numpy / gguf-py / bitsandbytes) and, where \
         it differs, the PyTorch-CPU equivalent (npz_to_st_torch / st_to_gguf_torch)."
    );

    // 1. NPZ -> safetensors at 4096x4096 F32.
    let f32_buf = synth_f32_perf();
    let mut map: HashMap<String, NpzTensor> = HashMap::new();
    map.insert(
        "w".into(),
        make_npz_tensor(
            "w",
            NpzDtype::F32,
            vec![PERF_SHAPE.0, PERF_SHAPE.1],
            f32_buf.clone(),
        ),
    );
    let (_, npz_us) = timed("npz->st @ perf", || npz_to_safetensors_bytes(&map).unwrap());
    report_vs_python("npz_to_st", npz_us, &[PERF_SHAPE.0, PERF_SHAPE.1]);
    report_vs_python("npz_to_st_torch", npz_us, &[PERF_SHAPE.0, PERF_SHAPE.1]);

    // 2. safetensors-BF16 -> GGUF at 4096x4096 BF16.
    let bf16_buf = synth_bf16_perf();
    let (_dir, _st_path) = write_temp(&[], "safetensors");
    let gguf_path = _st_path.with_file_name("perf.gguf");
    let inputs = vec![GgufWriteTensor {
        name: "w",
        shape: &[PERF_SHAPE.1, PERF_SHAPE.0], // GGUF MSB-first
        dtype: GgufType::BF16,
        data: &bf16_buf,
    }];
    let (_, gguf_us) = timed("st->gguf @ perf", || {
        write_gguf(&gguf_path, &inputs, &HashMap::new()).unwrap();
    });
    report_vs_python("st_to_gguf", gguf_us, &[PERF_SHAPE.0, PERF_SHAPE.1]);
    report_vs_python("st_to_gguf_torch", gguf_us, &[PERF_SHAPE.0, PERF_SHAPE.1]);

    // 3. safetensors-BF16 -> BnB-NF4 at 4096x4096 BF16.
    let bnb_inputs = vec![BnbWriteInput {
        name: "w",
        shape: &[PERF_SHAPE.0, PERF_SHAPE.1],
        bf16_data: &bf16_buf,
    }];
    let (_, bnb_us) = timed("st->bnb-nf4 @ perf", || {
        write_bnb_nf4_safetensors_bytes(&bnb_inputs).unwrap()
    });
    report_vs_python("st_to_bnb_nf4", bnb_us, &[PERF_SHAPE.0, PERF_SHAPE.1]);

    // 4. PTH -> safetensors: synthesise an in-memory PthTensor list
    //    matching the perf shape. PthTensor is owned, so we can build it
    //    inline without a file round-trip.
    use std::borrow::Cow;
    let pth_tensors = vec![anamnesis::PthTensor {
        name: "w".into(),
        shape: vec![PERF_SHAPE.0, PERF_SHAPE.1],
        dtype: anamnesis::PthDtype::BF16,
        data: Cow::Borrowed(&bf16_buf),
    }];
    let (_, pth_us) = timed("pth->st @ perf", || {
        pth_to_safetensors_bytes(&pth_tensors).unwrap()
    });
    report_vs_python("pth_to_st", pth_us, &[PERF_SHAPE.0, PERF_SHAPE.1]);
}
