// SPDX-License-Identifier: MIT OR Apache-2.0

//! Phase 7.2 — whole-model throughput benchmarks for the **threaded** paths.
//!
//! `dequant.rs` measures one kernel on one tensor; `parsing.rs` measures
//! header-only parses. Neither touches multi-threading, so until this file the
//! ~3–4× that Phase 7 (v0.7.0) shipped and the ~1.9× Phase 7.2 adds for `GGUF`
//! inputs had **no continuous regression guard at all** — only local, manual
//! best-of-5 runs.
//!
//! Each path is benchmarked at **1 thread and 4 threads** (the default budget).
//! Under CodSpeed's *walltime* instrument on macro runners
//! (`.github/workflows/codspeed.yml`) the pair is more informative than either
//! alone: a change that silently serialises the dispatch shows up as
//! `threads_4` converging on `threads_1`, which a single-budget benchmark would
//! report as a uniform slowdown indistinguishable from a slower kernel.
//!
//! # What is and is not in the measured region
//!
//! - `convert_gguf_to_safetensors` calls the public `convert()`, so it
//!   **includes the output file write** into a `TempDir`. That write is a large
//!   constant both budgets pay; it damps the visible ratio (locally ~1.25×
//!   end-to-end versus ~1.9× for the dequant stage alone — see
//!   `docs/perf-experiments.md` Experiment 12) but does not hide a regression in
//!   the part this bench exists to watch. The stage-isolated figure comes from
//!   the `#[ignore]`d `src/convert.rs::hub_scaling_bench`, which can reach the
//!   private `read_hub`; a bench target cannot.
//! - `remember_bf16_whole_model` calls `remember_to_bytes_with_options`, which
//!   returns a `Vec<u8>` and touches **no filesystem** — so it is the cleaner
//!   signal of the two, and the one that tracks Phase 7's headline number.
//!
//! # Fixtures
//!
//! Synthetic and self-contained: no dependency on the gitignored real models
//! under `tests/fixtures/gguf_reference/models/`, so this runs on a fresh clone
//! and in CI.
//!
//! The quantised `GGUF` fixture is written by a small hand-rolled serialiser
//! below rather than by the crate's own `write_gguf`, because that writer
//! deliberately rejects quantised dtypes (quantised emit is Phase 8.5). It is a
//! **deliberate duplicate** of the builder in
//! `src/convert.rs::quantized_gguf_tests`, following the precedent set in
//! `dequant.rs` ("duplicated locally so the bench has zero `tests/`
//! cross-dependency") — Cargo gives bench targets no way to share a module with
//! either `tests/` or a crate-private module.
//!
//! Run with:
//!
//! ```text
//! cargo bench --features gguf --bench convert
//! ```
//!
//! # Memory
//!
//! The `GGUF` fixture is ~10.6 `MiB` on disk and dequantises to ~22 `MiB` of
//! hub tensors; the safetensors fixture is ~16 `MiB` of `FP8` dequantising to
//! ~32 `MiB` of `BF16`. Peak resident stays under ~120 `MiB`.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    // The fixture builders push the same constant repeatedly (alignment
    // padding); the lint's `vec![x; n]` suggestion is correct but uglier here.
    clippy::same_item_push,
    // The module doc is prose (CodSpeed, Amdahl, run recipes). MSRV-1.88
    // clippy's `doc_markdown` allowlist lacks terms newer clippy accepts, so
    // allow it here rather than backticking English words.
    clippy::doc_markdown
)]

use std::path::PathBuf;

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};

use anamnesis::{ConvertOptions, ConvertTarget, RememberOptions, TargetDtype, convert};

/// The two budgets every group is measured at: the sequential baseline and the
/// library's default `min(cores, 4)`.
const BUDGETS: [usize; 2] = [1, 4];

/// `GGUF` default tensor-data alignment.
const ALIGNMENT: usize = 32;

// ---------------------------------------------------------------------------
// Deterministic synthesis
// ---------------------------------------------------------------------------

/// Knuth multiplicative hash on the index — the same filler `dequant.rs` uses,
/// so bit patterns are stable across runs and CodSpeed's comparison is not
/// perturbed by fixture churn.
fn fill_deterministic(buf: &mut [u8]) {
    for (i, b) in buf.iter_mut().enumerate() {
        *b = (i.wrapping_mul(2_654_435_761) & 0xFF) as u8;
    }
}

// ---------------------------------------------------------------------------
// Quantised GGUF fixture (hand-rolled; write_gguf rejects quantised dtypes)
// ---------------------------------------------------------------------------

/// `(name, GgufType discriminant, element count, bytes-on-disk)`.
struct GgufSpec {
    name: String,
    discriminant: u32,
    n_elements: usize,
    byte_len: usize,
}

/// Q8_0: 32-element blocks of 34 bytes (f16 scale + 32 int8).
fn q8_0(name: &str, n_elements: usize) -> GgufSpec {
    GgufSpec {
        name: name.to_owned(),
        discriminant: 8,
        n_elements,
        byte_len: (n_elements / 32) * 34,
    }
}

/// Q4_K: 256-element super-blocks of 144 bytes.
fn q4_k(name: &str, n_elements: usize) -> GgufSpec {
    GgufSpec {
        name: name.to_owned(),
        discriminant: 12,
        n_elements,
        byte_len: (n_elements / 256) * 144,
    }
}

/// F32 scalar passthrough.
fn f32_tensor(name: &str, n_elements: usize) -> GgufSpec {
    GgufSpec {
        name: name.to_owned(),
        discriminant: 0,
        n_elements,
        byte_len: n_elements * 4,
    }
}

/// Block bytes with a sane f16 scale, so the kernels do real arithmetic instead
/// of propagating `NaN` from a random exponent.
fn spec_data(spec: &GgufSpec) -> Vec<u8> {
    let mut buf = vec![0u8; spec.byte_len];
    fill_deterministic(&mut buf);
    match spec.discriminant {
        8 => {
            for block in buf.chunks_exact_mut(34) {
                block[0] = 0x00;
                block[1] = 0x3C; // f16 1.0
            }
        }
        12 => {
            for block in buf.chunks_exact_mut(144) {
                block[0] = 0x00;
                block[1] = 0x3C; // d    = f16 1.0
                block[2] = 0x00;
                block[3] = 0x38; // dmin = f16 0.5
            }
        }
        _ => {}
    }
    buf
}

/// A layer-shaped mix: skewed quantised weights (so the work-stealing partition
/// has something to balance) plus `F32` norms.
///
/// **Sized deliberately at ~10.6 `MiB` on disk.** The dispatch stays sequential
/// below `parallel::MIN_PARALLEL_BYTES` (4 `MiB`), so a smaller fixture would
/// run *both* budgets on the sequential path and the `threads_1` / `threads_4`
/// pair would report an identical time — a benchmark that looks healthy while
/// measuring nothing. An early draft of this file did exactly that at 2.64
/// `MiB`. Keep the total comfortably above the threshold if these numbers are
/// ever retuned.
fn gguf_specs() -> Vec<GgufSpec> {
    let mut specs = vec![
        q8_0("token_embd.weight", 4_194_304),
        q8_0("blk.0.attn_q.weight", 1_048_576),
        q8_0("blk.1.attn_q.weight", 524_288),
        q8_0("blk.2.attn_q.weight", 262_144),
        q4_k("blk.0.ffn_down.weight", 2_097_152),
        q4_k("blk.1.ffn_down.weight", 1_048_576),
        q4_k("blk.2.ffn_down.weight", 524_288),
    ];
    for i in 0..10 {
        specs.push(f32_tensor(&format!("blk.{i}.attn_norm.weight"), 65_536));
    }
    specs
}

fn push_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn push_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn push_string(buf: &mut Vec<u8>, s: &str) {
    push_u64(buf, s.len() as u64);
    buf.extend_from_slice(s.as_bytes());
}

fn pad_to_alignment(buf: &mut Vec<u8>) {
    while !buf.len().is_multiple_of(ALIGNMENT) {
        buf.push(0);
    }
}

/// Writes a `GGUF` v3 file carrying quantised tensor data.
fn build_gguf_fixture() -> (tempfile::TempDir, PathBuf) {
    let specs = gguf_specs();
    let mut buf = Vec::new();
    buf.extend_from_slice(b"GGUF");
    push_u32(&mut buf, 3);
    push_u64(&mut buf, specs.len() as u64);
    push_u64(&mut buf, 2);

    push_string(&mut buf, "general.architecture");
    push_u32(&mut buf, 8); // STRING
    push_string(&mut buf, "llama");
    push_string(&mut buf, "general.alignment");
    push_u32(&mut buf, 4); // UINT32
    push_u32(&mut buf, ALIGNMENT as u32);

    let mut relative = 0usize;
    let mut offsets = Vec::with_capacity(specs.len());
    for spec in &specs {
        offsets.push(relative);
        relative += spec.byte_len;
        while !relative.is_multiple_of(ALIGNMENT) {
            relative += 1;
        }
    }
    for (spec, &offset) in specs.iter().zip(offsets.iter()) {
        push_string(&mut buf, &spec.name);
        push_u32(&mut buf, 1); // 1-D
        push_u64(&mut buf, spec.n_elements as u64);
        push_u32(&mut buf, spec.discriminant);
        push_u64(&mut buf, offset as u64);
    }

    pad_to_alignment(&mut buf);
    for spec in &specs {
        buf.extend_from_slice(&spec_data(spec));
        pad_to_alignment(&mut buf);
    }

    let dir = tempfile::tempdir().expect("create temp dir");
    let path = dir.path().join("synth-quantized.gguf");
    std::fs::write(&path, &buf).expect("write gguf fixture");
    (dir, path)
}

// ---------------------------------------------------------------------------
// Quantised safetensors fixture (per-tensor FP8)
// ---------------------------------------------------------------------------

/// Number of `FP8` weights in the safetensors fixture. Skewed sizes again, and
/// enough tensors that a 4-worker pool has something to schedule.
const FP8_TENSORS: usize = 24;
/// Elements per `FP8` weight (`1 B` in, `2 B` `BF16` out).
const FP8_ELEMENTS: usize = 700_000;

/// Builds a per-tensor-`FP8` safetensors file (weight + `weight_scale` pairs),
/// the scheme `ParsedModel::dequantize_all` parallelises.
fn build_fp8_fixture() -> (tempfile::TempDir, PathBuf) {
    let mut header = serde_json::Map::new();
    let mut data: Vec<u8> = Vec::new();

    for i in 0..FP8_TENSORS {
        let w_off = data.len();
        let mut weight = vec![0u8; FP8_ELEMENTS];
        fill_deterministic(&mut weight);
        data.extend_from_slice(&weight);
        let mut w_info = serde_json::Map::new();
        w_info.insert("dtype".into(), "F8_E4M3".into());
        w_info.insert("shape".into(), serde_json::json!([FP8_ELEMENTS]));
        w_info.insert(
            "data_offsets".into(),
            serde_json::json!([w_off, data.len()]),
        );
        header.insert(format!("layer.{i}.weight"), w_info.into());

        let s_off = data.len();
        data.extend_from_slice(&(0.125_f32).to_le_bytes());
        let mut s_info = serde_json::Map::new();
        s_info.insert("dtype".into(), "F32".into());
        s_info.insert("shape".into(), serde_json::json!([1]));
        s_info.insert(
            "data_offsets".into(),
            serde_json::json!([s_off, data.len()]),
        );
        header.insert(format!("layer.{i}.weight_scale"), s_info.into());
    }

    let header_json = serde_json::to_string(&header).expect("serialize header");
    let mut file_bytes = Vec::new();
    file_bytes.extend_from_slice(&(header_json.len() as u64).to_le_bytes());
    file_bytes.extend_from_slice(header_json.as_bytes());
    file_bytes.extend_from_slice(&data);

    let dir = tempfile::tempdir().expect("create temp dir");
    let path = dir.path().join("synth-fp8.safetensors");
    std::fs::write(&path, &file_bytes).expect("write fp8 fixture");
    (dir, path)
}

// ---------------------------------------------------------------------------
// Benches
// ---------------------------------------------------------------------------

/// Quantised `GGUF` → safetensors through the public `convert()`. The Phase 7.2
/// path; includes the output write (see the module docs).
fn bench_convert_gguf_to_safetensors(c: &mut Criterion) {
    let (_dir, input) = build_gguf_fixture();
    let out_dir = tempfile::tempdir().expect("create temp dir");
    let input_bytes = std::fs::metadata(&input).expect("stat fixture").len();

    let mut group = c.benchmark_group("convert_gguf_to_safetensors");
    group.throughput(Throughput::Bytes(input_bytes));
    for threads in BUDGETS {
        let output = out_dir.path().join(format!("out-t{threads}.safetensors"));
        group.bench_function(format!("threads_{threads}"), |b| {
            b.iter(|| {
                let stats = convert(
                    black_box(&input),
                    ConvertTarget::Safetensors,
                    &output,
                    &ConvertOptions::new().with_threads(threads),
                )
                .expect("convert gguf -> safetensors");
                black_box(stats);
            });
        });
    }
    group.finish();
}

/// Whole-model `FP8` → `BF16` through `remember_to_bytes_with_options`. No file
/// I/O in the measured region, so this is the cleanest view of the Phase 7
/// threading win.
fn bench_remember_bf16_whole_model(c: &mut Criterion) {
    let (_dir, input) = build_fp8_fixture();
    let model = anamnesis::parse(&input).expect("parse fp8 fixture");
    let input_bytes = std::fs::metadata(&input).expect("stat fixture").len();

    let mut group = c.benchmark_group("remember_bf16_whole_model");
    group.throughput(Throughput::Bytes(input_bytes));
    for threads in BUDGETS {
        group.bench_function(format!("threads_{threads}"), |b| {
            b.iter(|| {
                let bytes = model
                    .remember_to_bytes_with_options(
                        TargetDtype::BF16,
                        RememberOptions::new().with_threads(black_box(threads)),
                    )
                    .expect("remember to bytes");
                black_box(bytes);
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_convert_gguf_to_safetensors,
    bench_remember_bf16_whole_model,
);
criterion_main!(benches);
