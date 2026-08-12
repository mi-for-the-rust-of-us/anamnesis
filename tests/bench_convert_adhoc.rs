// SPDX-License-Identifier: MIT OR Apache-2.0

//! Ad-hoc peak-heap measurements for the Phase 6.14 `convert` paths.
//!
//! Instruments `convert()` with `dhat` on three hub routes whose peak/
//! cumulative heap the Phase-6.14 copy-elimination pass targets:
//!
//! - **#2** `BF16` safetensors → `bnb-nf4` — the `to_bf16_bytes` full copy
//!   of an already-`BF16` hub.
//! - **#3** `gguf → gguf` with a large tokenizer KV — the source-KV clone
//!   in the `gguf` writer.
//! - **#4** `NPZ → safetensors` — the per-tensor `clone` in the `NPZ` reader
//!   instead of moving out of the owned map.
//!
//! All fixtures are synthetic (no external download). Each route runs inside
//! its own `dhat::Profiler` scope so the three measurements don't overlap
//! (dhat allows one profiler at a time), and fixtures are built *before* the
//! profiler starts so only `convert()`'s own allocations are counted.
//!
//! Run with (single-threaded so the sequential profiler scopes are safe):
//!
//! ```text
//! cargo test --release --features npz,gguf,bnb,pth \
//!   --test bench_convert_adhoc -- --ignored --nocapture
//! ```

#![cfg(all(feature = "npz", feature = "gguf", feature = "bnb"))]
#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::indexing_slicing,
    clippy::same_item_push
)]

use std::collections::HashMap;
use std::path::PathBuf;

use anamnesis::{
    ConvertOptions, ConvertTarget, GgufMetadataArray, GgufMetadataValue, GgufType, GgufWriteTensor,
    convert, write_gguf,
};

#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

// ---------------------------------------------------------------------------
// Synthetic fixture builders (built before the profiler; untracked)
// ---------------------------------------------------------------------------

fn synth_bf16(n_elements: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n_elements * 2);
    for i in 0..n_elements {
        let v = (i as f32 - (n_elements as f32) * 0.5) / (n_elements as f32);
        let bits = (v.to_bits() >> 16) as u16;
        out.extend_from_slice(&bits.to_le_bytes());
    }
    out
}

fn synth_f32(n_elements: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n_elements * 4);
    for i in 0..n_elements {
        let v = (i as f32 - (n_elements as f32) * 0.5) / (n_elements as f32);
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn build_safetensors_bf16(name: &str, shape: &[usize], data: &[u8]) -> Vec<u8> {
    let view = safetensors::tensor::TensorView::new(safetensors::Dtype::BF16, shape.to_vec(), data)
        .unwrap();
    safetensors::tensor::serialize([(name, view)], None).unwrap()
}

/// Minimal F32 `NPZ` (`ZIP` of `.npy`), mirroring `tests/cli_convert.rs`.
fn build_npz_f32(tensors: &[(&str, &[usize], &[u8])]) -> Vec<u8> {
    use std::io::Write;
    let mut zip = zip::ZipWriter::new(std::io::Cursor::new(Vec::<u8>::new()));
    let options: zip::write::SimpleFileOptions =
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
    for (name, shape, data) in tensors {
        zip.start_file(format!("{name}.npy"), options).unwrap();
        let shape_str: Vec<String> = shape.iter().map(usize::to_string).collect();
        let shape_tuple = if shape.len() == 1 {
            format!("({},)", shape_str[0])
        } else {
            format!("({})", shape_str.join(", "))
        };
        let dict = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': {shape_tuple}, }}");
        let mut header = dict.into_bytes();
        let header_total = 10 + header.len() + 1;
        let pad = (64 - header_total % 64) % 64;
        for _ in 0..pad {
            header.push(b' ');
        }
        header.push(b'\n');
        let header_len_u16 = u16::try_from(header.len()).unwrap();
        let mut entry_bytes: Vec<u8> = Vec::with_capacity(10 + header.len() + data.len());
        entry_bytes.extend_from_slice(&[0x93, b'N', b'U', b'M', b'P', b'Y', 1, 0]);
        entry_bytes.extend_from_slice(&header_len_u16.to_le_bytes());
        entry_bytes.extend_from_slice(&header);
        entry_bytes.extend_from_slice(data);
        zip.write_all(&entry_bytes).unwrap();
    }
    zip.finish().unwrap().into_inner()
}

fn write_temp(bytes: &[u8], ext: &str) -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join(format!("fixture.{ext}"));
    std::fs::write(&path, bytes).unwrap();
    (dir, path)
}

fn report(label: &str, stats: &dhat::HeapStats) {
    eprintln!(
        "  [{label}]\n    peak (max_bytes)   = {:>13} B  ({:>8.1} MiB)\n    \
         total (cum. alloc) = {:>13} B  ({:>8.1} MiB)\n    total blocks       = {}",
        stats.max_bytes,
        stats.max_bytes as f64 / (1024.0 * 1024.0),
        stats.total_bytes,
        stats.total_bytes as f64 / (1024.0 * 1024.0),
        stats.total_blocks,
    );
}

// ---------------------------------------------------------------------------
// The measurement
// ---------------------------------------------------------------------------

#[test]
#[ignore = "dhat peak-heap measurement; run with --release --ignored --nocapture"]
fn bench_convert_peak_heap() {
    eprintln!("\n=== Phase 6.14 convert peak-heap measurement ===");

    // --- #2: BF16 safetensors -> bnb-nf4 (8192 x 8192 = 128 MiB BF16 hub) ---
    {
        let shape = [8192usize, 8192];
        let bf16 = synth_bf16(shape[0] * shape[1]);
        let st = build_safetensors_bf16("model.weight", &shape, &bf16);
        let (_dir, in_path) = write_temp(&st, "safetensors");
        let out = in_path.with_file_name("out-bnb.safetensors");
        drop(bf16);
        drop(st);

        let profiler = dhat::Profiler::builder().testing().build();
        convert(
            &in_path,
            ConvertTarget::BnbNf4,
            &out,
            &ConvertOptions::new(),
        )
        .expect("bf16 st -> bnb-nf4");
        let stats = dhat::HeapStats::get();
        drop(profiler);
        report(
            "#2  BF16 safetensors -> bnb-nf4  (8192x8192, 128 MiB hub)",
            &stats,
        );
    }

    // --- #3: gguf -> gguf carrying a 256K-token tokenizer KV ---
    {
        const N_TOKENS: usize = 256 * 1024;
        let tokens: Vec<String> = (0..N_TOKENS).map(|i| format!("token_{i:08}")).collect();
        let mut metadata: HashMap<String, GgufMetadataValue> = HashMap::new();
        metadata.insert(
            "general.architecture".to_owned(),
            GgufMetadataValue::String("llama".to_owned()),
        );
        metadata.insert(
            "tokenizer.ggml.tokens".to_owned(),
            GgufMetadataValue::Array(Box::new(GgufMetadataArray::String(tokens))),
        );
        let f32_data = synth_f32(256);
        let src_tensors = [GgufWriteTensor {
            name: "blk.0.weight",
            shape: &[16, 16],
            dtype: GgufType::F32,
            data: &f32_data,
        }];
        let (_dir, in_path) = write_temp(&[], "gguf");
        let in_path = in_path.with_file_name("src.gguf");
        write_gguf(&in_path, &src_tensors, &metadata).unwrap();
        let out = in_path.with_file_name("out.gguf");
        drop(metadata);

        let profiler = dhat::Profiler::builder().testing().build();
        convert(&in_path, ConvertTarget::Gguf, &out, &ConvertOptions::new()).expect("gguf -> gguf");
        let stats = dhat::HeapStats::get();
        drop(profiler);
        report("#3  gguf -> gguf  (256K-token tokenizer KV)", &stats);
    }

    // --- #4: NPZ -> safetensors (2 x [4096 x 4096] F32 = 128 MiB) ---
    {
        let shape = [4096usize, 4096];
        let a = synth_f32(shape[0] * shape[1]);
        let b = synth_f32(shape[0] * shape[1]);
        let npz = build_npz_f32(&[("w0", &shape, &a), ("w1", &shape, &b)]);
        let (_dir, in_path) = write_temp(&npz, "npz");
        let out = in_path.with_file_name("out-npz.safetensors");
        drop(a);
        drop(b);
        drop(npz);

        let profiler = dhat::Profiler::builder().testing().build();
        convert(
            &in_path,
            ConvertTarget::Safetensors,
            &out,
            &ConvertOptions::new(),
        )
        .expect("npz -> safetensors");
        let stats = dhat::HeapStats::get();
        drop(profiler);
        report(
            "#4  NPZ -> safetensors  (2x 4096x4096 F32, 128 MiB)",
            &stats,
        );
    }

    eprintln!("=== end ===\n");
}
