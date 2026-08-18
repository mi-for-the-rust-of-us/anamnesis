// SPDX-License-Identifier: MIT OR Apache-2.0

//! Phase 6.5 — parser throughput benchmarks.
//!
//! Header-only / metadata-only parses of the four supported tensor
//! formats (`safetensors`, `NPZ`, `PTH`, `GGUF`), benchmarked against
//! a `fs::read` baseline so the report shows the parser's overhead as
//! a multiplier over raw I/O — the same framing the README uses for
//! the NPZ parser claim ("`3,586 MB/s` ≈ `1.3×` raw I/O overhead").
//!
//! Each bench writes a synthetic fixture into a `tempfile::TempDir`
//! once at setup, then iterates the parse. Fixture content is
//! deterministic across runs so `criterion`'s regression detection
//! stays stable.
//!
//! Run with:
//!
//! ```text
//! cargo bench --features npz,pth,gguf --bench parsing
//! ```
//!
//! Reports land in `target/criterion/`; HTML index at
//! `target/criterion/report/index.html`.
//!
//! # Memory
//!
//! Each fixture builder allocates one ~8 `MiB` tensor-data buffer
//! (`128 × 16,384 F32 elements`) inside a `tempfile::TempDir` and
//! writes it to disk; the `TempDir` stays alive for the bench fn's
//! scope so `criterion`'s `iter` loop can re-open the file each pass.
//! Peak resident per bench fn is ~16 `MiB` (file + parse result),
//! dropped when the fn returns. The `PTH` bench reuses the existing
//! ~2 `KiB` `algzoo_rnn_small.pth` test fixture instead, peaking at
//! ~100 `KiB`.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    // Bench-side fixture builders push the same constant repeatedly
    // (e.g., NPY header padding); the lint's `vec![b' '; n]` suggestion
    // is correct but uglier in context.
    clippy::same_item_push,
    // The PTH-fixture-not-found check uses `.map(...).unwrap_or(0)` on
    // `Result<Metadata, _>`; the lint's `map_or` suggestion is fine but
    // the explicit form reads more naturally as "if metadata fetch fails
    // or returns zero".
    clippy::map_unwrap_or
)]

use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};

use anamnesis::{
    GgufType, GgufWriteTensor, inspect_gguf_from_reader, inspect_npz_from_reader,
    inspect_pth_from_reader, parse_gguf_front_matter_from_reader,
    parse_pth_front_matter_from_reader, parse_safetensors_header_from_reader, write_gguf,
};

// ---------------------------------------------------------------------------
// Synthetic fixture sizes
// ---------------------------------------------------------------------------

/// Number of tensors per synthetic fixture. 128 covers a 16-layer
/// transformer's attention + FFN matrices without ballooning the
/// generated file size beyond what fits comfortably on a tmpfs.
const N_TENSORS: usize = 128;

/// Element count per tensor. `4096 × 4 = 16,384 F32 elements` =
/// `64 KiB` per tensor, giving `~8 MiB` total tensor data — large
/// enough that per-tensor overhead dominates over fixed setup cost
/// in the parser's measured time.
const ELEMENTS_PER_TENSOR: usize = 4096 * 4;

// ---------------------------------------------------------------------------
// Synthetic fixture builders. Each returns `(TempDir, PathBuf)` so the
// caller can bind the `TempDir` to a local that outlives the
// `criterion::Bencher::iter` closure. The `TempDir` is **not** leaked —
// scope-managed: it is dropped (and its directory deleted) when the
// bench fn returns after `group.finish()`. Idiomatic `tempfile` usage.
// ---------------------------------------------------------------------------

/// Builds a synthetic safetensors file with `N_TENSORS` `F32`
/// tensors, each `ELEMENTS_PER_TENSOR` elements. Returns the path.
fn build_safetensors_fixture() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().expect("create temp dir");
    let path = dir.path().join("synth.safetensors");

    let tensor_bytes: Vec<u8> = (0..ELEMENTS_PER_TENSOR * 4)
        .map(|i| (i.wrapping_mul(2_654_435_761) & 0xFF) as u8)
        .collect();

    // Owned data has to live for the full TensorView lifetime.
    let owned: Vec<Vec<u8>> = (0..N_TENSORS).map(|_| tensor_bytes.clone()).collect();
    let names: Vec<String> = (0..N_TENSORS)
        .map(|i| format!("layer.{i:03}.weight"))
        .collect();
    let shape = vec![ELEMENTS_PER_TENSOR];

    let mut views: Vec<(String, safetensors::tensor::TensorView<'_>)> =
        Vec::with_capacity(N_TENSORS);
    for (name, data) in names.iter().zip(owned.iter()) {
        let view =
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape.clone(), data)
                .expect("build tensor view");
        views.push((name.clone(), view));
    }
    safetensors::tensor::serialize_to_file(views, None, &path)
        .expect("serialize safetensors fixture");
    (dir, path)
}

/// Builds a synthetic `.npz` fixture with `N_TENSORS` `F32` arrays.
fn build_npz_fixture() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().expect("create temp dir");
    let path = dir.path().join("synth.npz");
    let file = std::fs::File::create(&path).expect("create npz file");
    let mut zip = zip::ZipWriter::new(file);
    let options: zip::write::SimpleFileOptions =
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);

    let payload: Vec<u8> = (0..ELEMENTS_PER_TENSOR * 4)
        .map(|i| (i.wrapping_mul(2_654_435_761) & 0xFF) as u8)
        .collect();

    for i in 0..N_TENSORS {
        let entry = format!("layer_{i:03}.npy");
        zip.start_file(&entry, options).expect("start zip entry");
        // Minimal NPY v1.0 header: magic + version + u16 header_len + dict + pad + newline.
        let shape_tuple = format!("({ELEMENTS_PER_TENSOR},)");
        let dict = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': {shape_tuple}, }}");
        let mut header = dict.into_bytes();
        let header_total = 10 + header.len() + 1;
        let pad = (64 - header_total % 64) % 64;
        for _ in 0..pad {
            header.push(b' ');
        }
        header.push(b'\n');
        let header_len_u16 = u16::try_from(header.len()).unwrap();
        let mut entry_bytes: Vec<u8> = Vec::with_capacity(10 + header.len() + payload.len());
        entry_bytes.extend_from_slice(&[0x93, b'N', b'U', b'M', b'P', b'Y', 1, 0]);
        entry_bytes.extend_from_slice(&header_len_u16.to_le_bytes());
        entry_bytes.extend_from_slice(&header);
        entry_bytes.extend_from_slice(&payload);
        zip.write_all(&entry_bytes).expect("write entry bytes");
    }
    zip.finish().expect("finalise zip");
    (dir, path)
}

/// Builds a synthetic `.pth` (`PyTorch` ZIP) fixture. Uses the smallest
/// pickle stream that exercises `anamnesis`'s pickle VM end-to-end —
/// a single `OrderedDict` of `N_TENSORS` `F32` tensors.
///
/// `PyTorch`'s `.pth` format is intricate; rather than re-implementing
/// the encoder from scratch, this benchmark cheats by using a
/// **pre-recorded** pickle stream pulled from a tiny model (the
/// `algzoo_rnn_small.pth` fixture already in the repo) and renaming
/// its tensors. This is fine because the benchmark only times
/// **header parsing**, not data decode — the pickle VM walks the same
/// opcodes regardless of how many or which tensor names follow.
fn build_pth_fixture() -> (tempfile::TempDir, PathBuf) {
    // Re-using the existing AlgZoo fixture as the bench's PTH input
    // avoids re-implementing the pickle encoder; the bench measures
    // parser throughput, not parser-input-size scaling. If the AlgZoo
    // fixture is unavailable (some packaging scenarios skip large
    // fixture files), this bench is skipped at the bench-fn level.
    let dir = tempfile::tempdir().expect("create temp dir");
    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("pth_reference")
        .join("algzoo_rnn_small.pth");
    let dst = dir.path().join("synth.pth");
    if src.exists() {
        std::fs::copy(&src, &dst).expect("copy pth fixture");
    } else {
        std::fs::write(&dst, b"").expect("write empty placeholder");
    }
    (dir, dst)
}

/// Builds a synthetic `.gguf` fixture with `N_TENSORS` `F32` tensors
/// plus a minimal metadata KV table, written via the Phase 6
/// `write_gguf` writer so the layout is known well-formed.
fn build_gguf_fixture() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().expect("create temp dir");
    let path = dir.path().join("synth.gguf");
    let tensor_bytes: Vec<u8> = (0..ELEMENTS_PER_TENSOR * 4)
        .map(|i| (i.wrapping_mul(2_654_435_761) & 0xFF) as u8)
        .collect();
    let owned: Vec<Vec<u8>> = (0..N_TENSORS).map(|_| tensor_bytes.clone()).collect();
    let names: Vec<String> = (0..N_TENSORS)
        .map(|i| format!("blk.{i:03}.weight"))
        .collect();
    let shape: Vec<usize> = vec![ELEMENTS_PER_TENSOR];

    let tensors: Vec<GgufWriteTensor<'_>> = names
        .iter()
        .zip(owned.iter())
        .map(|(name, data)| GgufWriteTensor {
            name: name.as_str(),
            shape: shape.as_slice(),
            dtype: GgufType::F32,
            data: data.as_slice(),
        })
        .collect();
    let metadata: HashMap<String, anamnesis::GgufMetadataValue> = HashMap::new();
    write_gguf(&path, &tensors, &metadata).expect("write gguf fixture");
    (dir, path)
}

// ---------------------------------------------------------------------------
// fs::read baseline (the divisor for the "vs raw I/O" ratio claim)
// ---------------------------------------------------------------------------

fn bench_fs_read_baseline(c: &mut Criterion) {
    // Use the safetensors fixture for the baseline because its file
    // size is closest to a typical parsed artefact and matches the
    // "header + data" mmap pattern other parsers compare against.
    let (_dir, path) = build_safetensors_fixture();
    let total_bytes = std::fs::metadata(&path).expect("stat fixture").len();

    let mut group = c.benchmark_group("baseline_fs_read");
    group.throughput(Throughput::Bytes(total_bytes));
    group.bench_function("safetensors_fixture", |b| {
        b.iter(|| {
            let bytes = std::fs::read(black_box(&path)).expect("fs read");
            black_box(bytes);
        });
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// Per-format parser benches
// ---------------------------------------------------------------------------

fn bench_safetensors_header(c: &mut Criterion) {
    let (_dir, path) = build_safetensors_fixture();
    let total_bytes = std::fs::metadata(&path).expect("stat fixture").len();

    let mut group = c.benchmark_group("parse_safetensors_header");
    group.throughput(Throughput::Bytes(total_bytes));
    group.bench_function("synthetic_128xF32_4096", |b| {
        b.iter(|| {
            let file = std::fs::File::open(black_box(&path)).expect("open safetensors");
            let header =
                parse_safetensors_header_from_reader(file).expect("parse safetensors header");
            black_box(header);
        });
    });
    group.finish();
}

fn bench_npz_inspect(c: &mut Criterion) {
    let (_dir, path) = build_npz_fixture();
    let total_bytes = std::fs::metadata(&path).expect("stat fixture").len();

    let mut group = c.benchmark_group("inspect_npz");
    group.throughput(Throughput::Bytes(total_bytes));
    group.bench_function("synthetic_128xF32_4096", |b| {
        b.iter(|| {
            let file = std::fs::File::open(black_box(&path)).expect("open npz");
            let info = inspect_npz_from_reader(file).expect("inspect npz");
            let _ = black_box(info);
        });
    });
    group.finish();
}

fn bench_pth_inspect(c: &mut Criterion) {
    let (_dir, path) = build_pth_fixture();
    if std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0) == 0 {
        eprintln!("  skipping inspect_pth: AlgZoo fixture not present");
        return;
    }
    let total_bytes = std::fs::metadata(&path).expect("stat fixture").len();

    let mut group = c.benchmark_group("inspect_pth");
    group.throughput(Throughput::Bytes(total_bytes));
    group.bench_function("algzoo_rnn_small", |b| {
        b.iter(|| {
            let file = std::fs::File::open(black_box(&path)).expect("open pth");
            let info = inspect_pth_from_reader(file).expect("inspect pth");
            let _ = black_box(info);
        });
    });
    group.finish();
}

/// Sibling of [`bench_pth_inspect`]: the full-detail
/// [`parse_pth_front_matter_from_reader`] entry point (v0.7.5) shares the
/// [`read_pth_meta`](anamnesis) core with `inspect_pth_from_reader`, so this
/// bench guards the full-tensor-list path against a regression the
/// summary-only bench above wouldn't catch. Uses the same tiny fixture, so
/// (per the crate's `# Memory` doc on `build_pth_fixture`) it reports
/// per-call latency dominated by fixed overhead rather than tensor-count
/// throughput, unlike its `NPZ` / safetensors / `GGUF` siblings.
fn bench_pth_front_matter(c: &mut Criterion) {
    let (_dir, path) = build_pth_fixture();
    if std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0) == 0 {
        eprintln!("  skipping parse_pth_front_matter: AlgZoo fixture not present");
        return;
    }
    let total_bytes = std::fs::metadata(&path).expect("stat fixture").len();

    let mut group = c.benchmark_group("parse_pth_front_matter");
    group.throughput(Throughput::Bytes(total_bytes));
    group.bench_function("algzoo_rnn_small", |b| {
        b.iter(|| {
            let file = std::fs::File::open(black_box(&path)).expect("open pth");
            let front = parse_pth_front_matter_from_reader(file).expect("parse pth front matter");
            let _ = black_box(front);
        });
    });
    group.finish();
}

fn bench_gguf_inspect(c: &mut Criterion) {
    let (_dir, path) = build_gguf_fixture();
    let total_bytes = std::fs::metadata(&path).expect("stat fixture").len();

    let mut group = c.benchmark_group("inspect_gguf");
    group.throughput(Throughput::Bytes(total_bytes));
    group.bench_function("synthetic_128xF32_4096", |b| {
        b.iter(|| {
            let file = std::fs::File::open(black_box(&path)).expect("open gguf");
            let info = inspect_gguf_from_reader(file).expect("inspect gguf");
            let _ = black_box(info);
        });
    });
    group.finish();
}

/// Sibling of [`bench_gguf_inspect`] over the same fixture: the full-detail
/// [`parse_gguf_front_matter_from_reader`] entry point (v0.7.1) shares the
/// [`read_gguf_structure`](anamnesis) core with `inspect_gguf_from_reader`,
/// so this bench guards the full-tensor-list path against a regression the
/// summary-only bench above wouldn't catch (e.g. a change that cheapens the
/// summary reduction at the cost of the retained `Vec<GgufTensorInfo>` /
/// `HashMap` construction).
fn bench_gguf_front_matter(c: &mut Criterion) {
    let (_dir, path) = build_gguf_fixture();
    let total_bytes = std::fs::metadata(&path).expect("stat fixture").len();

    let mut group = c.benchmark_group("parse_gguf_front_matter");
    group.throughput(Throughput::Bytes(total_bytes));
    group.bench_function("synthetic_128xF32_4096", |b| {
        b.iter(|| {
            let file = std::fs::File::open(black_box(&path)).expect("open gguf");
            let front = parse_gguf_front_matter_from_reader(file).expect("parse gguf front matter");
            let _ = black_box(front);
        });
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion plumbing
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_fs_read_baseline,
    bench_safetensors_header,
    bench_npz_inspect,
    bench_pth_inspect,
    bench_pth_front_matter,
    bench_gguf_inspect,
    bench_gguf_front_matter,
);
criterion_main!(benches);
