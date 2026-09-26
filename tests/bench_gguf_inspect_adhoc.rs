// SPDX-License-Identifier: MIT OR Apache-2.0

//! Ad-hoc benchmark comparing `parse_gguf(path).inspect()` (mmap-backed)
//! against `inspect_gguf_from_reader(File::open(path)?)` (reader-generic
//! over a `std::fs::File`) on the locally-downloaded GGUF fixtures.
//!
//! Not part of CI — requires the model directory at
//! `tests/fixtures/gguf_reference/models/` (gitignored, populated by
//! `generate_gguf.py`).
//!
//! Run with:
//!
//! ```text
//! cargo test --release --features gguf --test bench_gguf_inspect_adhoc \
//!     bench_gguf_inspect_paths -- --nocapture --ignored
//! ```
//!
//! With `target-cpu=native` for the perf-claim measurement protocol
//! (see `CLAUDE.md` § Performance Changes):
//!
//! ```text
//! $env:RUSTFLAGS = "-C target-cpu=native"
//! cargo test --release --features gguf --test bench_gguf_inspect_adhoc \
//!     bench_gguf_inspect_paths -- --nocapture --ignored
//! ```
//!
//! Prints best-of-5 median per file with min/max range, plus the median
//! reader/mmap ratio across all files.

#![cfg(feature = "gguf")]
#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::indexing_slicing
)]

mod common;

use std::path::PathBuf;

use anamnesis::{inspect_gguf_from_reader, parse_gguf};

use common::bench::{min_median_max, time_loop};

/// Number of timed iterations per file per substrate.
///
/// 5 is the project's standard sample size for perf-claim measurements
/// (see `CLAUDE.md` § Performance Changes — best-of-5 median).
const ITERATIONS: usize = 5;

/// One additional warm-up iteration before timing begins, to amortise
/// the cold-cache cost of the first read on a freshly-opened file.
const WARMUP_ITERATIONS: usize = 1;

#[test]
#[ignore = "needs locally-downloaded GGUF models under tests/fixtures/gguf_reference/models/"]
fn bench_gguf_inspect_paths() {
    let models_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("gguf_reference")
        .join("models");

    if !models_dir.exists() {
        eprintln!(
            "SKIP: {} does not exist (download via generate_gguf.py)",
            models_dir.display()
        );
        return;
    }

    let mut entries: Vec<_> = std::fs::read_dir(&models_dir)
        .expect("read models dir")
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("gguf"))
        .collect();
    entries.sort();
    assert!(
        !entries.is_empty(),
        "no .gguf files in {}",
        models_dir.display()
    );

    eprintln!(
        "\nGGUF inspect benchmark — {} files, best-of-{} median (warmup={})",
        entries.len(),
        ITERATIONS,
        WARMUP_ITERATIONS,
    );
    eprintln!(
        "{:<60}  {:>9}  {:>21}  {:>21}  {:>7}",
        "file", "size", "mmap min/med/max (\u{b5}s)", "reader min/med/max (\u{b5}s)", "ratio"
    );
    eprintln!("{}", "=".repeat(132));

    let mut ratios: Vec<f64> = Vec::with_capacity(entries.len());

    for path in &entries {
        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("?");
        let file_size = std::fs::metadata(path).ok().map_or(0, |m| m.len());

        // mmap-backed path: parse_gguf(path).inspect()
        let mut mmap_samples = time_loop(WARMUP_ITERATIONS, ITERATIONS, || {
            parse_gguf(path).expect("parse_gguf").inspect()
        });
        let (mmap_lo, mmap_mid, mmap_hi) = min_median_max(&mut mmap_samples);

        // Reader-generic path: inspect_gguf_from_reader(File::open(path)?)
        let mut reader_samples = time_loop(WARMUP_ITERATIONS, ITERATIONS, || {
            let f = std::fs::File::open(path).expect("open");
            inspect_gguf_from_reader(f).expect("inspect_gguf_from_reader")
        });
        let (reader_lo, reader_mid, reader_hi) = min_median_max(&mut reader_samples);

        let ratio = reader_mid.as_secs_f64() / mmap_mid.as_secs_f64();
        ratios.push(ratio);

        eprintln!(
            "{name:<60}  {size_mib:>6.1} MiB  {mmap_lo_us:>6.1}/{mmap_mid_us:>6.1}/{mmap_hi_us:>6.1}  {reader_lo_us:>6.0}/{reader_mid_us:>6.0}/{reader_hi_us:>6.0}  {ratio:>6.1}\u{d7}",
            size_mib = file_size as f64 / (1024.0 * 1024.0),
            mmap_lo_us = mmap_lo.as_secs_f64() * 1e6,
            mmap_mid_us = mmap_mid.as_secs_f64() * 1e6,
            mmap_hi_us = mmap_hi.as_secs_f64() * 1e6,
            reader_lo_us = reader_lo.as_secs_f64() * 1e6,
            reader_mid_us = reader_mid.as_secs_f64() * 1e6,
            reader_hi_us = reader_hi.as_secs_f64() * 1e6,
        );
    }

    eprintln!("{}", "=".repeat(132));
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = ratios.len();
    let r_min = ratios[0];
    let r_med = ratios[n / 2];
    let r_max = ratios[n - 1];
    let r_mean = ratios.iter().sum::<f64>() / n as f64;
    eprintln!(
        "reader / mmap ratio across {n} files: min={r_min:.1}\u{d7}  median={r_med:.1}\u{d7}  mean={r_mean:.1}\u{d7}  max={r_max:.1}\u{d7}",
    );
}
