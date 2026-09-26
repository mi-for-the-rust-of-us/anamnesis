// SPDX-License-Identifier: MIT OR Apache-2.0

//! Ad-hoc `.pth` parsing benchmark + phase profiling on torchvision models.
//!
//! Not part of CI: every test is gated `#[ignore]` and needs three torchvision
//! checkpoints (45 MB, 98 MB and 330 MB) that are not committed. Run with:
//!
//! ```text
//! cargo test --release --features pth --test bench_pth_adhoc -- --nocapture --ignored
//! ```
//!
//! ## Fetching the fixtures
//!
//! `.gitignore` keeps `tests/fixtures/pth_benchmark/` out of the repository.
//! Populate it once from the repository root (in `pwsh`; needs `torch` and
//! `torchvision`), saving each model's pretrained `state_dict` under the file
//! name the tests look for:
//!
//! ```text
//! New-Item -ItemType Directory -Force tests/fixtures/pth_benchmark
//! python -c "import torch, torchvision.models as m; [torch.save(getattr(m, n)(weights='DEFAULT').state_dict(), f'tests/fixtures/pth_benchmark/{n}.pth') for n in ('resnet18', 'resnet50', 'vit_b_16')]"
//! ```
//!
//! `torch.save` writes the ZIP-based format (`PyTorch` 1.6 and later) that
//! `parse_pth` reads. A missing file is reported as `SKIP` rather than failing,
//! so the harness still runs with any subset present.

#![cfg(feature = "pth")]
#![allow(
    clippy::unwrap_used,
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::wildcard_enum_match_arm,
    unsafe_code
)]

use std::path::PathBuf;
use std::time::Instant;

fn bench_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("pth_benchmark")
}

fn bench_file(label: &str, filename: &str, iters: u64) {
    let path = bench_dir().join(filename);
    if !path.exists() {
        println!("  SKIP {label} (not found)");
        return;
    }
    let size_mb = std::fs::metadata(&path).unwrap().len() as f64 / 1024.0 / 1024.0;

    // Warmup
    let parsed = anamnesis::parse_pth(&path).unwrap();
    let tensors = parsed.tensors().unwrap();
    let n_tensors = tensors.len();
    let n_params: usize = tensors
        .iter()
        .map(|t| t.shape.iter().copied().product::<usize>())
        .sum();
    drop(tensors);
    drop(parsed);

    // Benchmark parse + tensors() together (the full pipeline).
    let start = Instant::now();
    for _ in 0..iters {
        let p = anamnesis::parse_pth(&path).unwrap();
        let t = p.tensors().unwrap();
        std::hint::black_box(&t);
    }
    let elapsed = start.elapsed();
    let ms = elapsed.as_secs_f64() * 1000.0 / iters as f64;
    let throughput = size_mb / (ms / 1000.0);

    println!(
        "  {label}: {ms:.1} ms  ({n_tensors} tensors, {n_params} params, {throughput:.0} MB/s)"
    );
}

/// Profile individual phases using mmap (matching what `parse_pth` does).
fn profile_phases(label: &str, filename: &str, iters: u64) {
    let path = bench_dir().join(filename);
    if !path.exists() {
        println!("  SKIP {label} profile (not found)");
        return;
    }
    let size_mb = std::fs::metadata(&path).unwrap().len() as f64 / 1024.0 / 1024.0;

    // Phase 1: mmap only
    let start = Instant::now();
    for _ in 0..iters {
        let file = std::fs::File::open(&path).unwrap();
        let mmap = unsafe { memmap2::Mmap::map(&file) }.unwrap();
        std::hint::black_box(&mmap);
    }
    let mmap_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    // No separate ZIP-directory phase. This used to time `zip::ZipArchive::new`,
    // but `parse_pth` has read the container with the vendored
    // `src/parse/zip.rs` reader since v0.6.7, so that number described a crate
    // production no longer runs. The vendored reader is `pub(crate)`, so its
    // cost is folded into the "container + pickle + index + copy" line below.

    // Phase 2: full parse_pth
    let start = Instant::now();
    for _ in 0..iters {
        let _ = anamnesis::parse_pth(&path).unwrap();
    }
    let total_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    // Phase 3: memcpy baseline (simulates the unavoidable tensor copy cost)
    let file = std::fs::File::open(&path).unwrap();
    let mmap = unsafe { memmap2::Mmap::map(&file) }.unwrap();
    let start = Instant::now();
    for _ in 0..iters {
        let copy = mmap[..].to_vec();
        std::hint::black_box(&copy);
    }
    let memcpy_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    // Phase 4: fs::read for comparison
    let start = Instant::now();
    for _ in 0..iters {
        let raw = std::fs::read(&path).unwrap();
        std::hint::black_box(&raw);
    }
    let fsread_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    let rest_ms = total_ms - mmap_ms;

    println!("  {label} ({size_mb:.0} MB) phase breakdown:");
    println!("    mmap:             {mmap_ms:6.1} ms");
    println!(
        "    full parse_pth:   {total_ms:6.1} ms  (container+pickle+index+copy = {rest_ms:.1} ms)"
    );
    println!("    memcpy baseline:  {memcpy_ms:6.1} ms  (mmap[..].to_vec)");
    println!("    fs::read (ref):   {fsread_ms:6.1} ms");
    println!();
}

#[test]
#[ignore = "requires local torchvision checkpoints under tests/fixtures/pth_benchmark/"]
fn bench_pth_resnet18() {
    println!();
    bench_file("resnet18 (45 MB)", "resnet18.pth", 20);
}

#[test]
#[ignore = "requires local torchvision checkpoints under tests/fixtures/pth_benchmark/"]
fn bench_pth_resnet50() {
    println!();
    bench_file("resnet50 (98 MB)", "resnet50.pth", 10);
}

#[test]
#[ignore = "requires local torchvision checkpoints under tests/fixtures/pth_benchmark/"]
fn bench_pth_vit_b_16() {
    println!();
    bench_file("vit_b_16 (330 MB)", "vit_b_16.pth", 5);
}

#[test]
#[ignore = "requires local torchvision checkpoints under tests/fixtures/pth_benchmark/"]
fn profile_pth_phases() {
    println!("\n  === Phase Profiling (mmap path) ===\n");
    profile_phases("resnet18", "resnet18.pth", 20);
    profile_phases("resnet50", "resnet50.pth", 10);
    profile_phases("vit_b_16", "vit_b_16.pth", 5);
}
