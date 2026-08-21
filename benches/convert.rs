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
//! - `remember_f32_whole_model` (Phase 7.4) is the same measurement at the
//!   wider output dtype. Both groups key `Throughput` on **input** bytes, so
//!   the `F32` series is expected to sit below the `BF16` one by roughly the
//!   cost of writing twice the output — that gap is the measurement, not a
//!   regression.
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

use anamnesis::{ConvertOptions, ConvertTarget, RememberOptions, TargetDtype};
// Only `bench_convert_gguf_to_safetensors` calls the path-to-path `convert()`;
// its in-memory sibling uses `anamnesis::convert_bytes` fully qualified. Gated
// with the group so the CI build (feature off) has no unused import.
#[cfg(feature = "bench-fileio")]
use anamnesis::convert;

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
            for block in buf.as_chunks_mut::<34>().0 {
                block[0] = 0x00;
                block[1] = 0x3C; // f16 1.0
            }
        }
        12 => {
            for block in buf.as_chunks_mut::<144>().0 {
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
///
/// # Not run in CI
///
/// Gated behind the `bench-fileio` feature, which
/// `.github/workflows/codspeed.yml` deliberately does not enable. This group
/// writes its output to a file, and on macro runners that measures CI storage
/// rather than this crate. See `Cargo.toml`'s `bench-fileio` entry for the two
/// measurements that settled it. Run it locally with
/// `cargo bench --all-features --bench convert`.
///
/// Its in-memory sibling `convert_bytes_gguf_to_safetensors` covers the same
/// conversion and **is** the CI guard for it: 90.70 ms to 19.63 ms across the
/// budgets, a real 4.62×, on the same runner where this group reports 1.00×.
#[cfg(feature = "bench-fileio")]
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
    // Phase 7.3: end-to-end cost of `F32` output, where the doubled write meets
    // the output stage that Experiment 12 measured at ~60 % of wall clock. This
    // is the number that answers "what does --out-dtype f32 cost me", as
    // distinct from the kernel-level arm in benches/dequant.rs.
    //
    // Added as a sibling id rather than a rename: renaming `threads_{n}` would
    // orphan its CodSpeed history, and that BF16 series is exactly the baseline
    // this phase must not regress. Records rather than gates.
    for threads in BUDGETS {
        let output = out_dir
            .path()
            .join(format!("out-f32-t{threads}.safetensors"));
        group.bench_function(format!("threads_{threads}_f32"), |b| {
            b.iter(|| {
                let stats = convert(
                    black_box(&input),
                    ConvertTarget::Safetensors,
                    &output,
                    &ConvertOptions::new()
                        .with_threads(threads)
                        .with_output_dtype(anamnesis::Dtype::F32),
                )
                .expect("convert gguf -> safetensors f32");
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

/// Whole-model `FP8` → `F32` through `remember_to_bytes_with_options`, the
/// Phase 7.4 counterpart of [`bench_remember_bf16_whole_model`].
///
/// **A new group, not a new id inside the `BF16` one, and never a rename.**
/// `convert`'s F32 arm could join its existing group because that group is
/// named for a conversion (`convert_gguf_to_safetensors`) rather than a width;
/// this one is named `remember_bf16_whole_model`, so an `f32` id inside it
/// would read as a contradiction. Either way the rule is the same: the `BF16`
/// series is the baseline this phase must not regress, and renaming its group
/// or its `threads_{n}` ids would orphan the CodSpeed history that makes the
/// comparison possible at all.
///
/// **What the pair is for.** `F32` writes twice the output bytes for identical
/// arithmetic — the kernels already compute in `f32`, and at this width the
/// narrowing step is simply absent. On a bandwidth-bound path the honest
/// expectation is therefore *not* parity but a widening-shaped cost, and the
/// two groups together are what turn that from an argument into a measurement.
/// A change that made `F32` mysteriously *cheap* would be as suspicious as one
/// that made it slow.
///
/// **And the measurement immediately corrected the expectation.** "A
/// widening-shaped cost" (~2×) holds only single-threaded. The first CodSpeed
/// walltime run of this pair, on macro runners:
///
/// | | `BF16` | `F32` | ratio |
/// |---|---:|---:|---:|
/// | `threads_1` | 78.17 ms | 165.44 ms | 2.12× |
/// | `threads_4` | 20.83 ms | 81.60 ms | **3.92×** |
///
/// `F32` scales to 4 threads at **2.03×** where `BF16` reaches **3.75×**:
/// writing twice the bytes saturates memory bandwidth sooner, so threading buys
/// roughly half as much. Quoting the ~2× output-byte ratio as the expected
/// *time* cost is therefore wrong above one thread, and the gap widens with the
/// thread budget rather than staying constant. This is precisely the effect the
/// `walltime` instrument exists to catch and a CPU-simulation instrument would
/// have missed.
///
/// `F16` gets no arm: it is the same 2 bytes per element as `BF16` and shares
/// its `write_scratch` shape, so it would track the `BF16` series rather than
/// add an axis. The width contrast is the one worth paying CI time for.
fn bench_remember_f32_whole_model(c: &mut Criterion) {
    let (_dir, input) = build_fp8_fixture();
    let model = anamnesis::parse(&input).expect("parse fp8 fixture");
    let input_bytes = std::fs::metadata(&input).expect("stat fixture").len();

    let mut group = c.benchmark_group("remember_f32_whole_model");
    // Throughput is keyed on **input** bytes, exactly as the BF16 group is, so
    // the two series are directly comparable. Keying it on output bytes would
    // silently normalise away the very doubling this arm exists to show.
    group.throughput(Throughput::Bytes(input_bytes));
    for threads in BUDGETS {
        group.bench_function(format!("threads_{threads}"), |b| {
            b.iter(|| {
                let bytes = model
                    .remember_to_bytes_with_options(
                        TargetDtype::F32,
                        RememberOptions::new().with_threads(black_box(threads)),
                    )
                    .expect("remember to bytes f32");
                black_box(bytes);
            });
        });
    }
    group.finish();
}

/// Whole-model `GGUF` → `BF16` through `ParsedGguf::remember_to_bytes_with_options`.
///
/// The guard for the v0.7.6 item-1 fix, and it exists because that defect had no
/// bench to trip. Until v0.7.6 the only whole-model `GGUF` `remember` lived
/// inside the CLI, where nothing benches; it was sequential, ignored
/// `--threads`, and was 1.24×–2.23× slower than `convert --to safetensors` on
/// the same file for byte-identical output. `convert_gguf_to_safetensors` was
/// fast throughout, so the benched path stayed green while the shipped verb was
/// slow. This group watches the library entry point both now run.
///
/// **Writes a file, unlike its `FP8` siblings, and that choice was measured
/// rather than assumed.** The obvious design was to match them and go through
/// `remember_to_bytes_with_options`, keeping file I/O out of the numerator. On
/// this path it fails: `safetensors::serialize` builds one contiguous output
/// buffer in a single thread, and that copy is large enough here to swamp the
/// threaded dequant. Measured on the bench fixture (best of 3, this host):
///
/// | form | 1 thread | 4 threads | ratio |
/// |---|---|---|---|
/// | `remember_to_bytes` | 14.71 ms | 14.50 ms | **1.01×** |
/// | `remember` (file) | 14.33 ms | 12.18 ms | **1.18×** |
///
/// **CI then refuted the choice that table justified, which is why this group
/// no longer runs there.** The 1.18× was measured on a developer desktop. On
/// CodSpeed macro runners the file form reports 167.45 ms at 1 thread against
/// 168.47 ms at 4 — 0.99×, so the scaling the file form was *chosen for* does
/// not survive the instrument that watches it. The open question this doc
/// carried for the first CodSpeed run has its answer, recorded rather than
/// left as a standing assumption.
///
/// A guard that reports 1.01× cannot do its job: if this path regressed to
/// sequential, `threads_4` would move by ~1.4 %, inside the noise. Through the
/// file form the same regression is a ~18 % jump, which CodSpeed will see. The
/// file form is also what `amn remember` actually runs.
///
/// The same probe on a real 132 MiB `Q6_K` model shows the same ordering
/// (`to_bytes` 1.10×, `to_file` 1.33×), so this is a property of the path, not
/// of the fixture.
///
/// **Open question for the first CodSpeed run**, in the shape Experiment 12
/// left for its sibling: `convert_gguf_to_safetensors` also writes a file, and
/// macro runners flattened it from 1.26–1.36× locally to 0.99× there. If this
/// group flattens the same way, the absolute `threads_4` time stops moving on a
/// sequential regression and the guard weakens — in which case say so here
/// rather than trusting it.
#[cfg(feature = "bench-fileio")]
fn bench_remember_gguf_whole_model(c: &mut Criterion) {
    let (_dir, input) = build_gguf_fixture();
    let parsed = anamnesis::parse_gguf(&input).expect("parse gguf fixture");
    let input_bytes = std::fs::metadata(&input).expect("stat fixture").len();

    let out_dir = tempfile::tempdir().expect("create temp dir");
    let mut group = c.benchmark_group("remember_gguf_whole_model");
    group.throughput(Throughput::Bytes(input_bytes));
    for threads in BUDGETS {
        let output = out_dir.path().join(format!("out-t{threads}.safetensors"));
        group.bench_function(format!("threads_{threads}"), |b| {
            b.iter(|| {
                parsed
                    .remember_with_options(
                        &output,
                        TargetDtype::BF16,
                        RememberOptions::new().with_threads(black_box(threads)),
                    )
                    .expect("remember gguf to file");
            });
        });
    }
    group.finish();
}

/// `convert_bytes` on a `GGUF` source: memory in, memory out, no filesystem in
/// the measured region.
///
/// Guards the in-memory pipeline no other group touches: the byte-form readers
/// behind `read_hub_from_bytes`, the magic-byte detector they dispatch on, and
/// the `Sink::Memory` arm of the safetensors writer. `convert` reaches none of
/// those, because it takes paths.
///
/// **It also settles a question, in the opposite direction to the one expected.**
/// Experiment 12 left open whether `convert`'s modest end-to-end threading
/// (1.26–1.36× locally, 0.99× on CodSpeed macro runners) hides a reader that
/// scales ~1.9×, with the output write as the Amdahl denominator. `convert_bytes`
/// removes the file write entirely, so it looked like the clean way to see the
/// reader alone. It is not. Measured on the bench fixture (best of 3, this host):
///
/// | path | 1 thread | 4 threads | ratio |
/// |---|---|---|---|
/// | `convert` (file) | 19.88 ms | 14.88 ms | **1.34×** |
/// | `convert_bytes` | 16.67 ms | 15.14 ms | **1.10×** |
///
/// Removing the write made scaling *worse*, because the in-memory verb trades
/// one serial cost for two: it copies the caller's `&[u8]` into an owned buffer
/// (each byte-form parser takes ownership) and then serialises into one
/// contiguous output buffer in a single thread, where the file path memory-maps
/// its input and streams its output per tensor. So `convert_bytes` cannot be
/// used to isolate the reader's scaling; it carries more serial work than the
/// path it was meant to strip down. Experiment 12's question stays open.
///
/// Read this group as a guard on the in-memory pipeline's **absolute** cost
/// rather than on its scaling. The threading of the stage it shares with
/// `convert` and `remember` is already guarded twice over by their groups; at
/// 1.10× a sequential regression here would show as a ~10 % jump on `threads_4`,
/// detectable but not the reason this exists.
fn bench_convert_bytes_gguf(c: &mut Criterion) {
    let (_dir, input) = build_gguf_fixture();
    let source = std::fs::read(&input).expect("read gguf fixture");
    // CAST: usize → u64, lossless widening; a bench fixture is far inside u64.
    #[allow(clippy::as_conversions)]
    let input_bytes = source.len() as u64;

    let mut group = c.benchmark_group("convert_bytes_gguf_to_safetensors");
    group.throughput(Throughput::Bytes(input_bytes));
    for threads in BUDGETS {
        group.bench_function(format!("threads_{threads}"), |b| {
            b.iter(|| {
                let (bytes, stats) = anamnesis::convert_bytes(
                    black_box(&source),
                    ConvertTarget::Safetensors,
                    &ConvertOptions::new().with_threads(black_box(threads)),
                )
                .expect("convert_bytes gguf -> safetensors");
                black_box((bytes, stats));
            });
        });
    }
    group.finish();
}

/// Stub for the feature-off build: `criterion_group!` names this function
/// unconditionally, so it has to exist in both configurations. Registering no
/// benchmarks is exactly the intent — see the real definition above.
#[cfg(not(feature = "bench-fileio"))]
fn bench_convert_gguf_to_safetensors(_c: &mut Criterion) {}

/// Stub for the feature-off build; see `bench_convert_gguf_to_safetensors`.
#[cfg(not(feature = "bench-fileio"))]
fn bench_remember_gguf_whole_model(_c: &mut Criterion) {}

criterion_group!(
    benches,
    bench_convert_gguf_to_safetensors,
    bench_remember_bf16_whole_model,
    bench_remember_f32_whole_model,
    bench_remember_gguf_whole_model,
    bench_convert_bytes_gguf,
);
criterion_main!(benches);
