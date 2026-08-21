// SPDX-License-Identifier: MIT OR Apache-2.0
//! Paired A/B benchmarks: **the instrument that decides whether a change is faster.**
//!
//! # Why this exists beside `dequant.rs`
//!
//! `dequant.rs` is a *pointwise* harness. It measures one binary; a verdict
//! needs two separate runs, and the difference between them carries every
//! environmental drift that occurred in between. That design is defeated by the
//! effect `CONVENTIONS.md` § *Benchmark evidence* records, where whole-program
//! code layout moved an **untouched** kernel by up to 23 % locally and 4.16 % on
//! CI. A difference smaller than that cannot be read out of two pointwise runs,
//! however many samples each takes.
//!
//! Tango measures the *pair*: both versions are loaded together and interleaved
//! sample by sample, so clock, thermal, cache and load drift apply to both sides
//! at once and cancel in the difference. This is the interleaved best-of-N
//! discipline `CLAUDE.md` § *Performance Changes* already prescribes by hand,
//! run automatically and at a granularity no human alternating two binaries can
//! reach.
//!
//! **It runs on x86-64, on the maintainer's own machine** — the architecture the
//! crate is developed and released on, and the one `CodSpeed` cannot measure at
//! all, its walltime instrument being `aarch64`-only with no self-hosted option.
//!
//! # What it does not fix
//!
//! Pairing removes *noise*, not *bias*. Two different binaries have two
//! different code layouts, and the measured difference still contains that.
//! Tango turns a noisy question into a reproducible one; it does not separate
//! "the algorithm got faster" from "the code moved". Nothing practical does.
//!
//! # Measured floor: about 2 %
//!
//! Comparing this harness against an export of **byte-identical source**, which
//! is the only honest way to calibrate an instrument:
//!
//! | arm | delta on identical code |
//! |---|---|
//! | `awq_int4_f32` | **+2.20 %** * |
//! | `awq_int4_f16` | +1.74 % * |
//! | `gptq_int4_f32` | +1.06 % * |
//! | `awq_int4_bf16`, `gptq_int4_bf16`, `gptq_int4_f16` | <= +0.58 % |
//!
//! Note the stars. **A `*` means "significant given the sampling", not "real".**
//! Three arms earned one while nothing had changed. That is the residue of
//! compile-to-compile layout differences, and no amount of extra sampling
//! removes it.
//!
//! So: treat anything under ~2.5 % as no signal whatever the star says, treat
//! 5 % as the threshold worth acting on, and treat the ~45 % this harness
//! measured for Phase 7.7's `AWQ` migration as fact. For comparison, the
//! pointwise criterion setup this replaces has a floor of 10-23 % locally, and
//! CI walltime about 4 %.
//!
//! # Workflow
//!
//! ```text
//! cargo install cargo-export                     # once
//!
//! # 1. on the baseline commit, export the compiled harness
//! cargo export target/benchmarks -- bench --bench=ab --features gptq,awq,bnb,gguf
//!
//! # 2. switch to the candidate code, then compare against it
//! cargo bench --bench=ab --features gptq,awq,bnb,gguf -- compare target/benchmarks/ab
//! ```
//!
//! A `*` on a row marks a statistically significant difference, and the process
//! exits non-zero when a significant regression is found, so this is usable as a
//! gate rather than only as a report.
//!
//! # Coverage, and how to extend it
//!
//! `AWQ` and `GPTQ`, at all three [`OutputElement`](anamnesis::OutputElement)
//! widths. That pairing is deliberate rather than arbitrary: the two kernels are
//! structurally identical apart from `AWQ`'s `AWQ_ORDER` scatter, they received
//! the *same* source change in Phase 7.7, and they disagreed about it — `GPTQ`
//! gained 9.87 % while `AWQ` lost ~45 %. Keeping both means every future run
//! carries its own control.
//!
//! Adding a kernel is mechanical: build its fixture as `dequant.rs` does, then
//! add one `benchmark_fn` per width. Each arm costs roughly ten seconds.

// A bench is its own crate and inherits none of `src/lib.rs`'s inner attributes,
// so the lint policy is restated here. Same set `dequant.rs` carries, and for the
// same reason: a fixture that cannot be built is a broken benchmark, and
// panicking on it is the correct response rather than a defect to be handled.
#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]

use std::hint::black_box;

use anamnesis::{
    Dtype, F16Out, F32Out, dequantize_awq, dequantize_awq_to_bf16, dequantize_gptq,
    dequantize_gptq_to_bf16,
};
use tango_bench::{IntoBenchmarks, benchmark_fn, tango_benchmarks};

/// Rows for the synthetic layer fixture, matching `dequant.rs` so both
/// harnesses measure the same shape.
const LAYER_ROWS: usize = 4096;
/// Cols for the synthetic layer fixture.
const LAYER_COLS: usize = 11008;
/// Quantised bit width for both kernels below.
const BITS: u8 = 4;
/// Group size for both kernels below.
const GROUP_SIZE: usize = 128;

/// Knuth multiplicative hash on the index, the same filler `dequant.rs` uses, so
/// bit patterns are stable across runs and the pair is not perturbed by fixture
/// churn.
fn synth_bytes(len: usize) -> Vec<u8> {
    let mut v = vec![0u8; len];
    for (i, b) in v.iter_mut().enumerate() {
        // CAST: usize -> u64 widening for the hash, then -> u8 by deliberate
        // truncation to a byte pattern. Neither is a value-carrying conversion.
        #[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
        {
            *b = ((i as u64).wrapping_mul(2_654_435_761) >> 16) as u8;
        }
    }
    v
}

/// `BF16` `0.5` little-endian, repeated. Non-zero so dequant output is
/// non-trivial, exactly as `dequant.rs` builds it.
fn bf16_half_scales(n_elements: usize) -> Vec<u8> {
    let mut scales = vec![0u8; n_elements * 2];
    for pair in scales.as_chunks_mut::<2>().0 {
        *pair = [0x00, 0x3F];
    }
    scales
}

/// `(qweight, scales, qzeros)` for `AWQ`: `qweight` is
/// `[in_features, out_features / pack_factor]`, column-packed.
fn awq_fixture() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let pack_factor = 32 / BITS as usize;
    let num_groups = LAYER_ROWS / GROUP_SIZE;
    (
        synth_bytes(LAYER_ROWS * (LAYER_COLS / pack_factor) * 4),
        bf16_half_scales(num_groups * LAYER_COLS),
        synth_bytes(num_groups * (LAYER_COLS / pack_factor) * 4),
    )
}

/// `(qweight, scales, qzeros)` for `GPTQ`: `qweight` is
/// `[in_features / pack_factor, out_features]`, row-packed. The opposite
/// packing to `AWQ`, which is why the two kernels' pass 1 differs.
fn gptq_fixture() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let pack_factor = 32 / BITS as usize;
    let num_groups = LAYER_ROWS / GROUP_SIZE;
    (
        synth_bytes((LAYER_ROWS / pack_factor) * LAYER_COLS * 4),
        bf16_half_scales(num_groups * LAYER_COLS),
        synth_bytes(num_groups * (LAYER_COLS / pack_factor) * 4),
    )
}

/// One arm per output width. Phase 7.7 item 1a established that the three
/// [`OutputElement`](anamnesis::OutputElement) monomorphisations are three
/// separate codegen outcomes, and that a suite measuring one of them measures
/// one of them.
fn dequant_benchmarks() -> impl IntoBenchmarks {
    let awq = awq_fixture();
    let (awq1, awq2, awq3) = (awq.clone(), awq.clone(), awq);
    let gptq = gptq_fixture();
    let (gptq1, gptq2, gptq3) = (gptq.clone(), gptq.clone(), gptq);

    [
        benchmark_fn("awq_int4_bf16", move |b| {
            let (qw, sc, qz) = awq1.clone();
            b.iter(move || {
                black_box(
                    dequantize_awq_to_bf16(
                        black_box(&qw),
                        black_box(&sc),
                        black_box(&qz),
                        LAYER_ROWS,
                        LAYER_COLS,
                        GROUP_SIZE,
                        BITS,
                        Dtype::BF16,
                    )
                    .expect("awq bf16"),
                )
            })
        }),
        benchmark_fn("awq_int4_f32", move |b| {
            let (qw, sc, qz) = awq2.clone();
            b.iter(move || {
                black_box(
                    dequantize_awq::<F32Out>(
                        black_box(&qw),
                        black_box(&sc),
                        black_box(&qz),
                        LAYER_ROWS,
                        LAYER_COLS,
                        GROUP_SIZE,
                        BITS,
                        Dtype::BF16,
                    )
                    .expect("awq f32"),
                )
            })
        }),
        benchmark_fn("awq_int4_f16", move |b| {
            let (qw, sc, qz) = awq3.clone();
            b.iter(move || {
                black_box(
                    dequantize_awq::<F16Out>(
                        black_box(&qw),
                        black_box(&sc),
                        black_box(&qz),
                        LAYER_ROWS,
                        LAYER_COLS,
                        GROUP_SIZE,
                        BITS,
                        Dtype::BF16,
                    )
                    .expect("awq f16"),
                )
            })
        }),
        benchmark_fn("gptq_int4_bf16", move |b| {
            let (qw, sc, qz) = gptq1.clone();
            b.iter(move || {
                black_box(
                    dequantize_gptq_to_bf16(
                        black_box(&qw),
                        black_box(&sc),
                        black_box(&qz),
                        None,
                        LAYER_ROWS,
                        LAYER_COLS,
                        GROUP_SIZE,
                        BITS,
                        Dtype::BF16,
                    )
                    .expect("gptq bf16"),
                )
            })
        }),
        benchmark_fn("gptq_int4_f32", move |b| {
            let (qw, sc, qz) = gptq2.clone();
            b.iter(move || {
                black_box(
                    dequantize_gptq::<F32Out>(
                        black_box(&qw),
                        black_box(&sc),
                        black_box(&qz),
                        None,
                        LAYER_ROWS,
                        LAYER_COLS,
                        GROUP_SIZE,
                        BITS,
                        Dtype::BF16,
                    )
                    .expect("gptq f32"),
                )
            })
        }),
        benchmark_fn("gptq_int4_f16", move |b| {
            let (qw, sc, qz) = gptq3.clone();
            b.iter(move || {
                black_box(
                    dequantize_gptq::<F16Out>(
                        black_box(&qw),
                        black_box(&sc),
                        black_box(&qz),
                        None,
                        LAYER_ROWS,
                        LAYER_COLS,
                        GROUP_SIZE,
                        BITS,
                        Dtype::BF16,
                    )
                    .expect("gptq f16"),
                )
            })
        }),
    ]
}

// `tango_benchmarks!` generates `fn main` itself in 0.8; the older
// `tango_main!()` is deprecated and no longer needed.
tango_benchmarks!(dequant_benchmarks());
