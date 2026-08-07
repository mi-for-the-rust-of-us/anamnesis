// SPDX-License-Identifier: MIT OR Apache-2.0

//! Phase 7.2 ad-hoc scaling measurements for the parallelised `GGUF`-input
//! `convert` path, plus the calibration behind `parallel::MIN_PARALLEL_BYTES`.
//!
//! Not part of CI — every test is gated `#[ignore]`. This harness answers the
//! two questions Phase 7.2 has to answer with data rather than argument, per
//! `CLAUDE.md` § Performance Changes:
//!
//! 1. **What does an idle scoped thread pool cost?** `bench_scoped_pool_overhead`
//!    times `std::thread::scope` spawn + join with workers that do nothing. That
//!    fixed cost, set against the dequant paths' measured throughput, is what
//!    `MIN_PARALLEL_BYTES` has to clear — the named size threshold
//!    `CONVENTIONS.md` "When Parallelizing Work" rule 3 requires.
//! 2. **Does the `GGUF` reader actually scale?** `bench_gguf_convert_scaling`
//!    converts a **real** quantised `GGUF` to safetensors at
//!    `threads ∈ {1, 2, 4, 8, 16}`. `threads = 1` is the honest *before* number:
//!    until Phase 7.2, `convert::read_gguf` was unconditionally sequential, so
//!    the 1-thread median is exactly what v0.7.1 shipped.
//!
//! Determinism is asserted **before** any timing (the Experiment 11 discipline):
//! if the 1-thread and 8-thread outputs are not byte-identical, the numbers below
//! are meaningless and the test fails instead of reporting them.
//!
//! ## Fixtures
//!
//! The real models live in `tests/fixtures/gguf_reference/models/`, which is
//! **gitignored** (~13 `GiB`; download recipe in
//! `tests/fixtures/gguf_reference/generate_gguf.py`). Every fixture-dependent
//! test **skips with a message** when the file is absent, so the harness stays
//! runnable on a fresh clone.
//!
//! ## Running
//!
//! ```text
//! $env:RUSTFLAGS = "-C target-cpu=native"
//! cargo test --release --features gguf --test bench_gguf_convert_adhoc -- --ignored --nocapture
//! $env:RUSTFLAGS = $null
//! ```

#![cfg(feature = "gguf")]
#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::indexing_slicing,
    // Dev-only harness: the module doc is prose (PowerShell recipes, run
    // commands) and MSRV-1.88 clippy's `doc_markdown` allowlist lacks terms
    // newer clippy accepts.
    clippy::doc_markdown
)]

use std::path::{Path, PathBuf};
use std::time::Instant;

use anamnesis::{convert, ConvertOptions, ConvertTarget};

/// Best-of-N sample count. `CLAUDE.md` mandates best-of-5 for a perf-claim
/// commit; this harness runs the same 5 so the reported median is the one the
/// commit message quotes.
const SAMPLES: usize = 5;

/// Thread budgets swept by the scaling bench. Includes 1 (the v0.7.1 sequential
/// baseline) and 16 (the reference machine's core count) to bracket the knee.
const BUDGETS: [usize; 5] = [1, 2, 4, 8, 16];

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// Primary fixture: a 135 M-parameter `Q4_K_M` model — small enough to run five
/// samples per budget in seconds, large enough that the per-tensor work dwarfs
/// the pool cost.
const PRIMARY_MODEL: &str = "SmolLM2-135M-Instruct-Q4_K_M.gguf";

/// Secondary fixture: a 1.1 B-parameter `Q5_0` model, to confirm the knee does
/// not move with model size.
const LARGE_MODEL: &str = "tinyllama-1.1b-chat-v1.0.Q5_0.gguf";

fn model_path(file_name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("gguf_reference")
        .join("models")
        .join(file_name)
}

// ---------------------------------------------------------------------------
// Timing helpers
// ---------------------------------------------------------------------------

/// Median + range of an ascending-sorted `&[f64]`, formatted for stderr. Same
/// shape as the sibling `bench_pass2_adhoc` harness so the two are comparable.
fn fmt_stats(samples: &[f64]) -> String {
    let median = samples[samples.len() / 2];
    let min = samples[0];
    let max = samples[samples.len() - 1];
    format!("median {median:.2} ms (min {min:.2}, max {max:.2})")
}

/// Best-of-N timing helper: one warm-up call, then `SAMPLES` timed calls,
/// returning the ascending-sorted millisecond samples.
fn time_best_of_n<F>(mut f: F) -> Vec<f64>
where
    F: FnMut(),
{
    f();

    let mut samples: Vec<f64> = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let start = Instant::now();
        f();
        samples.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples
}

// ---------------------------------------------------------------------------
// 1. Scoped-pool overhead — the MIN_PARALLEL_BYTES calibration
// ---------------------------------------------------------------------------

/// Times `std::thread::scope` spawn + join with workers that immediately
/// return, for a range of worker counts. This is the fixed cost every parallel
/// dispatch pays before doing any useful work — the quantity
/// `parallel::MIN_PARALLEL_BYTES` exists to amortise.
///
/// Reported in **microseconds**: at these sizes milliseconds lose all
/// resolution.
#[test]
#[ignore = "ad-hoc measurement; run explicitly with --ignored"]
fn bench_scoped_pool_overhead() {
    eprintln!("\n=== scoped-thread-pool spawn+join overhead (empty workers) ===");

    for n_workers in [2usize, 4, 8, 16] {
        // 200 pool cycles per sample so the per-cycle figure is resolvable
        // against the clock's granularity.
        const CYCLES: usize = 200;
        let samples = time_best_of_n(|| {
            for _ in 0..CYCLES {
                std::thread::scope(|scope| {
                    let handles: Vec<_> = (0..n_workers).map(|_| scope.spawn(|| {})).collect();
                    for handle in handles {
                        handle.join().expect("worker join");
                    }
                });
            }
        });
        let per_cycle_us = (samples[samples.len() / 2] * 1000.0) / CYCLES as f64;
        eprintln!(
            "{n_workers:>2} workers: {per_cycle_us:>7.1} us / pool cycle   [{CYCLES} cycles/sample, {}]",
            fmt_stats(&samples)
        );
    }

    eprintln!(
        "\nCalibration: MIN_PARALLEL_BYTES must be large enough that the pool cost above\n\
         is a small fraction of the dequant time. At ~1-2 GB/s of input throughput,\n\
         N MiB of input takes roughly N/1.5 ms."
    );
}

// ---------------------------------------------------------------------------
// 2. GGUF -> safetensors convert scaling
// ---------------------------------------------------------------------------

/// Converts `model` to safetensors at each budget in [`BUDGETS`], after
/// asserting that the 1-thread and 8-thread outputs are byte-identical.
fn scaling_for(model: &str) {
    let input = model_path(model);
    if !input.exists() {
        eprintln!("SKIP {model}: fixture absent (gitignored; see generate_gguf.py)");
        return;
    }
    let input_bytes = std::fs::metadata(&input).expect("stat fixture").len();
    let dir = tempfile::tempdir().expect("temp dir");

    eprintln!("\n=== {model} -> safetensors ({:.1} MiB input) ===", {
        input_bytes as f64 / (1024.0 * 1024.0)
    });

    // Determinism gate: thread count is a performance knob, never a correctness
    // variable. Run this BEFORE reporting any timing.
    let out_seq = dir.path().join("determinism-t1.safetensors");
    let out_par = dir.path().join("determinism-t8.safetensors");
    convert(
        &input,
        ConvertTarget::Safetensors,
        &out_seq,
        &ConvertOptions::new().with_threads(1),
    )
    .expect("convert at 1 thread");
    convert(
        &input,
        ConvertTarget::Safetensors,
        &out_par,
        &ConvertOptions::new().with_threads(8),
    )
    .expect("convert at 8 threads");
    let seq_bytes = std::fs::read(&out_seq).expect("read 1-thread output");
    let par_bytes = std::fs::read(&out_par).expect("read 8-thread output");
    assert_eq!(
        seq_bytes, par_bytes,
        "{model}: 1-thread and 8-thread outputs must be byte-identical"
    );
    eprintln!(
        "determinism: 1 thread == 8 threads ({} bytes) OK",
        seq_bytes.len()
    );
    drop(seq_bytes);
    drop(par_bytes);

    let mut baseline_median = 0.0_f64;
    for threads in BUDGETS {
        let output = dir.path().join(format!("scaling-t{threads}.safetensors"));
        let samples = time_best_of_n(|| {
            convert(
                &input,
                ConvertTarget::Safetensors,
                &output,
                &ConvertOptions::new().with_threads(threads),
            )
            .expect("convert");
        });
        let median = samples[samples.len() / 2];
        if threads == 1 {
            baseline_median = median;
        }
        let speedup = baseline_median / median;
        eprintln!(
            "{threads:>2} threads: {}  -> {speedup:.2}x vs 1 thread",
            fmt_stats(&samples)
        );
        std::fs::remove_file(&output).ok();
    }
}

/// The primary scaling measurement. `threads = 1` is the v0.7.1 *before*
/// number (the reader was unconditionally sequential); every other row is the
/// Phase 7.2 *after*.
#[test]
#[ignore = "ad-hoc measurement; run explicitly with --ignored"]
fn bench_gguf_convert_scaling() {
    scaling_for(PRIMARY_MODEL);
}

/// Same sweep on a ~7x larger model, to confirm the knee is a property of the
/// memory system rather than of the fixture.
#[test]
#[ignore = "ad-hoc measurement; run explicitly with --ignored"]
fn bench_gguf_convert_scaling_large() {
    scaling_for(LARGE_MODEL);
}
