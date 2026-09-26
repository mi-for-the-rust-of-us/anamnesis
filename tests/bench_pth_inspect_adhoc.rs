// SPDX-License-Identifier: MIT OR Apache-2.0

//! Ad-hoc benchmarks comparing `parse_pth(path).inspect()` (mmap-backed)
//! against `inspect_pth_from_reader(File::open(path)?)` (reader-generic
//! over a `std::fs::File`) on `.pth` fixtures.
//!
//! Two `#[ignore]`-gated tests are provided:
//!
//! - `bench_pth_inspect_paths` — runs on the 3 in-tree `AlgZoo` fixtures
//!   under `tests/fixtures/pth_reference/`. Prints per-file min/median/max
//!   and the overall reader/mmap ratio. Use this as the regression-
//!   detection harness for `inspect_pth_from_reader`.
//!
//! - `bench_pth_inspect_algzoo_sweep` — sweeps every `.pth` file under
//!   the directory pointed to by `ANAMNESIS_ALGZOO_DIR` (typically the
//!   external `algzoo_weights/` corpus of ~7 000 files used by
//!   `candle-mi`'s `stoicheia` module). Per-file output is suppressed; a
//!   summary table reports min / median / mean / max of the per-file
//!   medians for each substrate. This is the broader-population
//!   validation that the parity claim in the rustdoc holds across the
//!   full corpus, not just the 3 in-tree fixtures.
//!
//! Both are gated behind `#[ignore]` so they do not run on every
//! `cargo test`. Run with:
//!
//! ```text
//! cargo test --release --features pth --test bench_pth_inspect_adhoc \
//!     bench_pth_inspect_paths -- --nocapture --ignored
//! ```
//!
//! With `target-cpu=native` for the perf-claim measurement protocol
//! (see `CLAUDE.md` § Performance Changes):
//!
//! ```text
//! $env:RUSTFLAGS = "-C target-cpu=native"
//! cargo test --release --features pth --test bench_pth_inspect_adhoc \
//!     bench_pth_inspect_paths -- --nocapture --ignored
//! ```
//!
//! For the `AlgZoo` sweep, additionally set `ANAMNESIS_ALGZOO_DIR`:
//!
//! ```text
//! $env:RUSTFLAGS = "-C target-cpu=native"
//! $env:ANAMNESIS_ALGZOO_DIR = "C:/Users/Eric JACOPIN/Documents/Data/algzoo_weights"
//! cargo test --release --features pth --test bench_pth_inspect_adhoc \
//!     bench_pth_inspect_algzoo_sweep -- --nocapture --ignored
//! ```

#![cfg(feature = "pth")]
#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::indexing_slicing,
    // Single-letter pairs like `(mm_min, mm_p25, ...)` trip the
    // similar-names lint even though the prefix disambiguates them. Suppressing
    // crate-wide for this test file (which is statistics-heavy by design).
    clippy::similar_names
)]

/// Per-family aggregation buckets: family name → (mmap medians, reader medians,
/// reader/mmap ratios), one record per `.pth` file in that family.
type FamilyBuckets = std::collections::BTreeMap<String, (Vec<f64>, Vec<f64>, Vec<f64>)>;

mod common;

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anamnesis::{inspect_pth_from_reader, parse_pth};

use common::bench::{min_median_max, time_loop};

/// Number of timed iterations per file per substrate.
///
/// 5 is the project's standard sample size for perf-claim measurements
/// (see `CLAUDE.md` § Performance Changes — best-of-5 median).
const ITERATIONS: usize = 5;

/// One additional warm-up iteration before timing begins, to amortise
/// the cold-cache cost of the first read on a freshly-opened file.
const WARMUP_ITERATIONS: usize = 1;

/// Environment variable selecting the external `.pth` corpus to sweep.
///
/// Set to the path of `algzoo_weights/` (or any directory containing
/// `.pth` files) before invoking `bench_pth_inspect_algzoo_sweep`. When
/// unset or invalid, the sweep test prints a SKIP message and returns.
const ALGZOO_DIR_ENV: &str = "ANAMNESIS_ALGZOO_DIR";

/// Per-file timing record produced by [`measure_file`].
struct FileTiming {
    file_size: u64,
    mmap_min: Duration,
    mmap_med: Duration,
    mmap_max: Duration,
    reader_min: Duration,
    reader_med: Duration,
    reader_max: Duration,
}

impl FileTiming {
    fn ratio(&self) -> f64 {
        self.reader_med.as_secs_f64() / self.mmap_med.as_secs_f64()
    }
}

/// Times one `.pth` file end-to-end across both substrates.
fn measure_file(path: &Path) -> FileTiming {
    let file_size = std::fs::metadata(path).ok().map_or(0, |m| m.len());

    let mut mmap_samples = time_loop(WARMUP_ITERATIONS, ITERATIONS, || {
        parse_pth(path).expect("parse_pth").inspect()
    });
    let (mmap_min, mmap_med, mmap_max) = min_median_max(&mut mmap_samples);

    let mut reader_samples = time_loop(WARMUP_ITERATIONS, ITERATIONS, || {
        let f = std::fs::File::open(path).expect("open");
        inspect_pth_from_reader(f).expect("inspect_pth_from_reader")
    });
    let (reader_min, reader_med, reader_max) = min_median_max(&mut reader_samples);

    FileTiming {
        file_size,
        mmap_min,
        mmap_med,
        mmap_max,
        reader_min,
        reader_med,
        reader_max,
    }
}

/// Returns `(min, p25, median, p75, mean, max)` of an unordered sample of
/// `f64` values (e.g., per-file median durations or per-file ratios).
fn distribution(samples: &mut [f64]) -> (f64, f64, f64, f64, f64, f64) {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = samples.len();
    let p25 = samples[n / 4];
    let p75 = samples[(n * 3) / 4];
    let med = samples[n / 2];
    let lo = samples[0];
    let hi = samples[n - 1];
    let mean = samples.iter().sum::<f64>() / n as f64;
    (lo, p25, med, p75, mean, hi)
}

#[test]
#[ignore = "ad-hoc perf benchmark; run on demand with --release --ignored"]
fn bench_pth_inspect_paths() {
    let fixture_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("pth_reference");

    let fixtures = [
        "algzoo_rnn_small.pth",
        "algzoo_transformer_small.pth",
        "algzoo_rnn_blog.pth",
    ];

    eprintln!(
        "\nPTH inspect benchmark — {} files, best-of-{} median (warmup={})",
        fixtures.len(),
        ITERATIONS,
        WARMUP_ITERATIONS,
    );
    eprintln!(
        "{:<40}  {:>7}  {:>21}  {:>21}  {:>7}",
        "file", "size", "mmap min/med/max (\u{b5}s)", "reader min/med/max (\u{b5}s)", "ratio"
    );
    eprintln!("{}", "=".repeat(112));

    let mut ratios: Vec<f64> = Vec::with_capacity(fixtures.len());

    for name in fixtures {
        let path = fixture_dir.join(name);
        let t = measure_file(&path);
        ratios.push(t.ratio());

        eprintln!(
            "{name:<40}  {size_kib:>5.1} KiB  {mmap_lo_us:>6.1}/{mmap_mid_us:>6.1}/{mmap_hi_us:>6.1}  {reader_lo_us:>6.1}/{reader_mid_us:>6.1}/{reader_hi_us:>6.1}  {ratio:>6.2}\u{d7}",
            size_kib = t.file_size as f64 / 1024.0,
            mmap_lo_us = t.mmap_min.as_secs_f64() * 1e6,
            mmap_mid_us = t.mmap_med.as_secs_f64() * 1e6,
            mmap_hi_us = t.mmap_max.as_secs_f64() * 1e6,
            reader_lo_us = t.reader_min.as_secs_f64() * 1e6,
            reader_mid_us = t.reader_med.as_secs_f64() * 1e6,
            reader_hi_us = t.reader_max.as_secs_f64() * 1e6,
            ratio = t.ratio(),
        );
    }

    eprintln!("{}", "=".repeat(112));
    let (r_min, _, r_med, _, r_mean, r_max) = distribution(&mut ratios);
    eprintln!(
        "reader / mmap ratio across {n} files: min={r_min:.2}\u{d7}  median={r_med:.2}\u{d7}  mean={r_mean:.2}\u{d7}  max={r_max:.2}\u{d7}",
        n = ratios.len(),
    );
}

/// Sweeps every `.pth` file under `$ANAMNESIS_ALGZOO_DIR`, reports
/// aggregate min / median / mean / max of per-file medians for both
/// substrates. Per-file output is suppressed; only the summary table is
/// printed so the output stays terminal-friendly across ~7 000 files.
///
/// Use this to validate that the rustdoc's parity claim
/// (`inspect_pth_from_reader` at ~1.5× of `parse_pth(path).inspect()`
/// on KiB-scale fixtures) holds across the full corpus, not just the 3
/// in-tree fixtures.
#[test]
#[ignore = "needs ANAMNESIS_ALGZOO_DIR pointing at an external .pth corpus"]
fn bench_pth_inspect_algzoo_sweep() {
    let Ok(env_value) = std::env::var(ALGZOO_DIR_ENV) else {
        eprintln!(
            "SKIP: {ALGZOO_DIR_ENV} not set. Example:\n  $env:{ALGZOO_DIR_ENV} = \
             \"C:/Users/Eric JACOPIN/Documents/Data/algzoo_weights\""
        );
        return;
    };
    let dir = PathBuf::from(env_value);
    if !dir.is_dir() {
        eprintln!("SKIP: {} is not a directory", dir.display());
        return;
    }

    let mut entries: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("read_dir")
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("pth"))
        .collect();
    entries.sort();

    if entries.is_empty() {
        eprintln!("SKIP: no .pth files under {}", dir.display());
        return;
    }

    eprintln!(
        "\nPTH inspect AlgZoo sweep — {} files in {}",
        entries.len(),
        dir.display(),
    );
    eprintln!("  best-of-{ITERATIONS} median per file, {WARMUP_ITERATIONS} warm-up iteration");

    // Group files by task-family prefix so we can report per-family
    // stats alongside the global distribution. We strip the trailing
    // `_<digits>_<digits>_..._<digits>` hyperparameter suffix by finding
    // the first `_<digit>` underscore-digit boundary and cutting there.
    // The four `AlgZoo` task families end up as `2nd_argmax`,
    // `argmedian`, `longest_cycle`, `median` — note that a stem like
    // `2nd_argmax_16_10_0_0` starts with a digit but the family name
    // does too, so we cannot key on "first digit"; the underscore-digit
    // boundary is what separates task name from hyperparameters.
    let family_of = |p: &Path| -> String {
        let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("?");
        let bytes = stem.as_bytes();
        let mut cut = stem.len();
        for i in 0..bytes.len().saturating_sub(1) {
            if bytes[i] == b'_' && bytes[i + 1].is_ascii_digit() {
                cut = i;
                break;
            }
        }
        stem.get(..cut).unwrap_or(stem).to_owned()
    };

    let mut all_mmap: Vec<f64> = Vec::with_capacity(entries.len());
    let mut all_reader: Vec<f64> = Vec::with_capacity(entries.len());
    let mut all_ratios: Vec<f64> = Vec::with_capacity(entries.len());
    let mut all_sizes: Vec<u64> = Vec::with_capacity(entries.len());

    // Per-family buckets: family name → (mmap medians, reader medians, ratios)
    let mut family_buckets: FamilyBuckets = std::collections::BTreeMap::new();

    let sweep_start = Instant::now();
    for path in &entries {
        let t = measure_file(path);
        all_sizes.push(t.file_size);
        let mmap_us = t.mmap_med.as_secs_f64() * 1e6;
        let reader_us = t.reader_med.as_secs_f64() * 1e6;
        all_mmap.push(mmap_us);
        all_reader.push(reader_us);
        all_ratios.push(t.ratio());

        let family = family_of(path);
        let entry = family_buckets.entry(family).or_default();
        entry.0.push(mmap_us);
        entry.1.push(reader_us);
        entry.2.push(t.ratio());
    }
    let sweep_elapsed = sweep_start.elapsed();

    let total_bytes: u64 = all_sizes.iter().sum();

    // Global distributions.
    let (mm_min, mm_p25, mm_med, mm_p75, mm_mean, mm_max) = distribution(&mut all_mmap);
    let (rd_min, rd_p25, rd_med, rd_p75, rd_mean, rd_max) = distribution(&mut all_reader);
    let (rt_min, rt_p25, rt_med, rt_p75, rt_mean, rt_max) = distribution(&mut all_ratios);

    eprintln!(
        "\n  bench wall-clock: {:.1}s  ({:.1} files/s)",
        sweep_elapsed.as_secs_f64(),
        entries.len() as f64 / sweep_elapsed.as_secs_f64(),
    );
    eprintln!(
        "  total bytes inspected: {:.1} MiB",
        total_bytes as f64 / (1024.0 * 1024.0),
    );

    eprintln!(
        "\n  {:<14}  {:>9}  {:>9}  {:>9}  {:>9}  {:>9}  {:>9}",
        "metric (\u{b5}s)", "min", "p25", "median", "p75", "mean", "max"
    );
    eprintln!("  {}", "-".repeat(82));
    eprintln!(
        "  {:<14}  {mm_min:>9.1}  {mm_p25:>9.1}  {mm_med:>9.1}  {mm_p75:>9.1}  {mm_mean:>9.1}  {mm_max:>9.1}",
        "mmap path",
    );
    eprintln!(
        "  {:<14}  {rd_min:>9.1}  {rd_p25:>9.1}  {rd_med:>9.1}  {rd_p75:>9.1}  {rd_mean:>9.1}  {rd_max:>9.1}",
        "reader path",
    );
    eprintln!(
        "  {:<14}  {rt_min:>9.2}  {rt_p25:>9.2}  {rt_med:>9.2}  {rt_p75:>9.2}  {rt_mean:>9.2}  {rt_max:>9.2}",
        "reader/mmap",
    );

    // Per-family breakdown — sanity-check that no family is an outlier
    // (e.g., parser path is family-independent; the only variable is
    // tensor count and shape).
    if family_buckets.len() > 1 {
        eprintln!(
            "\n  {:<18}  {:>6}  {:>11}  {:>11}  {:>10}",
            "family", "count", "mmap med (\u{b5}s)", "reader med (\u{b5}s)", "ratio med"
        );
        eprintln!("  {}", "-".repeat(72));
        for (family, (mut mmap_vec, mut reader_vec, mut ratio_vec)) in family_buckets {
            let mmap_med = {
                mmap_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
                mmap_vec[mmap_vec.len() / 2]
            };
            let reader_med = {
                reader_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
                reader_vec[reader_vec.len() / 2]
            };
            let ratio_med = {
                ratio_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
                ratio_vec[ratio_vec.len() / 2]
            };
            eprintln!(
                "  {family:<18}  {count:>6}  {mmap_med:>9.1}  {reader_med:>11.1}  {ratio_med:>9.2}\u{d7}",
                count = mmap_vec.len(),
            );
        }
    }
}
