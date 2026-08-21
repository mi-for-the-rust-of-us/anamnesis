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
//! # What its numbers are NOT
//!
//! **Do not quote the absolute millisecond figures this harness prints.** It
//! samples adaptively to resolve the *difference between two binaries*, which
//! is what makes it sensitive; the absolute magnitude of either side is not a
//! stable statistic and is not what it optimises for. Measured here: three
//! consecutive invocations on unchanged code reported `bnb_int8` at `F16` as
//! **75.9, 149.2 and 75.8 ms**, while every paired delta in those same runs
//! stayed within +/-2 %.
//!
//! So: **this harness answers "is A faster than B", not "how long does A
//! take".** For an absolute magnitude — including a ratio between two output
//! widths, which is a *within-binary* comparison with no layout term between
//! the arms — use `benches/dequant.rs` under criterion, whose 100-sample median
//! is the right statistic for the question.
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
//! All seven dequant families `dequant.rs` covers, at all three
//! [`OutputElement`](anamnesis::OutputElement) widths: **21 arms**.
//!
//! The `AWQ` / `GPTQ` pairing is the load-bearing one. Those two kernels are
//! structurally identical apart from `AWQ`'s `AWQ_ORDER` scatter, they received
//! the *same* source change in Phase 7.7, and they disagreed about it — `GPTQ`
//! gained 9.87 % while `AWQ` lost ~45 %. The rest are here so that **every run
//! carries its own controls**: a kernel nobody touched that moves as much as the
//! one under test is the signal that the reading is layout, not algorithm.
//!
//! Adding a kernel is mechanical: build its fixture as `dequant.rs` does, then
//! add one `benchmark_fn` per width.
//!
//! # Running a subset
//!
//! A full pass is ~21 arms. When iterating on one kernel, filter — but **keep at
//! least one untouched family in the filter as a control**:
//!
//! ```text
//! cargo bench --bench=ab --features gptq,awq,bnb,gguf -- \
//!     compare target/benchmarks/ab --filter '{awq,gptq}_*' --noise-threshold 2.5
//! ```
//!
//! `--noise-threshold` defaults to **1 %**, which is below this harness's
//! measured floor and will star differences that are not real. Pass **2.5**,
//! for the reason tabulated above.

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
    clippy::cast_precision_loss,
    // Each fixture is cloned once per output width, giving deliberately parallel
    // binding names (`nf4a`/`nf4b`/`nf4c`). The similarity is the point: they are
    // the same fixture, and naming them apart would imply a difference that does
    // not exist. `tango`'s closures are `move`, so one shared binding will not do.
    clippy::similar_names
)]

use std::hint::black_box;

use anamnesis::{
    Dtype, F16Out, F32Out, GgufType, dequantize_awq, dequantize_awq_to_bf16, dequantize_bnb_int8,
    dequantize_bnb_int8_to_bf16, dequantize_bnb4, dequantize_bnb4_to_bf16, dequantize_fp8,
    dequantize_fp8_to_bf16, dequantize_gguf, dequantize_gguf_to_bf16, dequantize_gptq,
    dequantize_gptq_to_bf16, dequantize_per_tensor_fp8, dequantize_per_tensor_fp8_to_bf16,
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
/// `BnB` `NF4` block size.
const BNB_BLOCK: usize = 64;
/// Canonical `NF4` codebook, the same constant as `anamnesis::NF4_CODEBOOK`,
/// duplicated so the bench has no dependency on a non-public item.
const NF4_CODEBOOK: [f32; 16] = [
    -1.0,
    -0.696_192_8,
    -0.525_073_05,
    -0.394_917_5,
    -0.284_441_38,
    -0.184_773_43,
    -0.091_050_036,
    0.0,
    0.079_580_3,
    0.160_930_2,
    0.246_112_3,
    0.337_915_24,
    0.440_709_83,
    0.562_617,
    0.722_956_84,
    1.0,
];

/// Knuth multiplicative hash on the index, identical to
/// `dequant.rs::fill_deterministic`, so bit patterns are stable across runs, the
/// pair is not perturbed by fixture churn, and absolute times from the two
/// harnesses may be compared.
fn synth_bytes(len: usize) -> Vec<u8> {
    let mut v = vec![0u8; len];
    for (i, b) in v.iter_mut().enumerate() {
        // CAST: usize -> u8 by deliberate truncation to a byte pattern, not a
        // value-carrying conversion.
        //
        // **Byte-for-byte the expression `dequant.rs::fill_deterministic` uses**,
        // and that matters: the two harnesses are quoted side by side (see the
        // `F16Out` cost table), so a different filler would give different
        // quantised values, different codebook indices and different denormal
        // counts, making their absolute times incomparable while looking as
        // though they compared.
        #[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
        {
            *b = (i.wrapping_mul(2_654_435_761) & 0xFF) as u8;
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

/// `(weight, scale_data)` for fine-grained `FP8`: `128x128` block scales, `BF16`
/// `1.0` (`0x3F80`) so the arithmetic is real without perturbing magnitudes.
fn fp8_fine_fixture() -> (Vec<u8>, Vec<u8>) {
    let blocks = (LAYER_ROWS / 128) * (LAYER_COLS / 128);
    (
        synth_bytes(LAYER_ROWS * LAYER_COLS),
        [0x80u8, 0x3F].repeat(blocks),
    )
}

/// `(weight, absmax, quant_map)` for `BnB` `NF4` at `block_size = 64`. `absmax`
/// is all `1.0`, so the dequantised output is the codebook itself.
fn bnb4_fixture() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let total = LAYER_ROWS * LAYER_COLS;
    let absmax = (0..total / BNB_BLOCK)
        .flat_map(|_| 1.0_f32.to_le_bytes())
        .collect();
    let quant_map = NF4_CODEBOOK.iter().flat_map(|v| v.to_le_bytes()).collect();
    (synth_bytes(total / 2), absmax, quant_map)
}

/// `(weight, scb)` for `BnB` `INT8`: per-row `f32` absmax, all `1.0`. Note the
/// shape is `[out_features, in_features]`, the opposite of `GPTQ`/`AWQ` above.
fn bnb_int8_fixture() -> (Vec<u8>, Vec<u8>) {
    let scb = (0..LAYER_COLS)
        .flat_map(|_| 1.0_f32.to_le_bytes())
        .collect();
    (synth_bytes(LAYER_COLS * LAYER_ROWS), scb)
}

/// `Q4_K` raw blocks: 256 elements per 144-byte super-block.
fn gguf_q4k_fixture() -> Vec<u8> {
    synth_bytes((LAYER_ROWS * LAYER_COLS / 256) * 144)
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
    let fp8f = fp8_fine_fixture();
    let (fp8a, fp8b, fp8c) = (fp8f.clone(), fp8f.clone(), fp8f);
    let fp8t = synth_bytes(LAYER_ROWS * LAYER_COLS);
    let (fp8t1, fp8t2, fp8t3) = (fp8t.clone(), fp8t.clone(), fp8t);
    let nf4 = bnb4_fixture();
    let (nf4a, nf4b, nf4c) = (nf4.clone(), nf4.clone(), nf4);
    let i8 = bnb_int8_fixture();
    let (i8a, i8b, i8c) = (i8.clone(), i8.clone(), i8);
    let gg = gguf_q4k_fixture();
    let (gg1, gg2, gg3) = (gg.clone(), gg.clone(), gg);

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
        benchmark_fn("fp8_fine_bf16", move |b| {
            let (w, s) = fp8a.clone();
            b.iter(move || {
                black_box(
                    dequantize_fp8_to_bf16(
                        black_box(&w),
                        black_box(&s),
                        LAYER_ROWS,
                        LAYER_COLS,
                        Dtype::BF16,
                    )
                    .expect("fp8 fine bf16"),
                )
            })
        }),
        benchmark_fn("fp8_fine_f32", move |b| {
            let (w, s) = fp8b.clone();
            b.iter(move || {
                black_box(
                    dequantize_fp8::<F32Out>(
                        black_box(&w),
                        black_box(&s),
                        LAYER_ROWS,
                        LAYER_COLS,
                        Dtype::BF16,
                    )
                    .expect("fp8 fine f32"),
                )
            })
        }),
        benchmark_fn("fp8_fine_f16", move |b| {
            let (w, s) = fp8c.clone();
            b.iter(move || {
                black_box(
                    dequantize_fp8::<F16Out>(
                        black_box(&w),
                        black_box(&s),
                        LAYER_ROWS,
                        LAYER_COLS,
                        Dtype::BF16,
                    )
                    .expect("fp8 fine f16"),
                )
            })
        }),
        benchmark_fn("fp8_tensor_bf16", move |b| {
            let w = fp8t1.clone();
            b.iter(move || {
                black_box(
                    dequantize_per_tensor_fp8_to_bf16(black_box(&w), black_box(0.125_f32))
                        .expect("fp8 tensor bf16"),
                )
            })
        }),
        benchmark_fn("fp8_tensor_f32", move |b| {
            let w = fp8t2.clone();
            b.iter(move || {
                black_box(
                    dequantize_per_tensor_fp8::<F32Out>(black_box(&w), black_box(0.125_f32))
                        .expect("fp8 tensor f32"),
                )
            })
        }),
        benchmark_fn("fp8_tensor_f16", move |b| {
            let w = fp8t3.clone();
            b.iter(move || {
                black_box(
                    dequantize_per_tensor_fp8::<F16Out>(black_box(&w), black_box(0.125_f32))
                        .expect("fp8 tensor f16"),
                )
            })
        }),
        benchmark_fn("bnb_nf4_bf16", move |b| {
            let (w, a, q) = nf4a.clone();
            b.iter(move || {
                black_box(
                    dequantize_bnb4_to_bf16(
                        black_box(&w),
                        black_box(&a),
                        black_box(&q),
                        LAYER_ROWS * LAYER_COLS,
                        BNB_BLOCK,
                    )
                    .expect("bnb nf4 bf16"),
                )
            })
        }),
        benchmark_fn("bnb_nf4_f32", move |b| {
            let (w, a, q) = nf4b.clone();
            b.iter(move || {
                black_box(
                    dequantize_bnb4::<F32Out>(
                        black_box(&w),
                        black_box(&a),
                        black_box(&q),
                        LAYER_ROWS * LAYER_COLS,
                        BNB_BLOCK,
                    )
                    .expect("bnb nf4 f32"),
                )
            })
        }),
        benchmark_fn("bnb_nf4_f16", move |b| {
            let (w, a, q) = nf4c.clone();
            b.iter(move || {
                black_box(
                    dequantize_bnb4::<F16Out>(
                        black_box(&w),
                        black_box(&a),
                        black_box(&q),
                        LAYER_ROWS * LAYER_COLS,
                        BNB_BLOCK,
                    )
                    .expect("bnb nf4 f16"),
                )
            })
        }),
        benchmark_fn("bnb_int8_bf16", move |b| {
            let (w, s) = i8a.clone();
            b.iter(move || {
                black_box(
                    dequantize_bnb_int8_to_bf16(
                        black_box(&w),
                        black_box(&s),
                        LAYER_COLS,
                        LAYER_ROWS,
                    )
                    .expect("bnb int8 bf16"),
                )
            })
        }),
        benchmark_fn("bnb_int8_f32", move |b| {
            let (w, s) = i8b.clone();
            b.iter(move || {
                black_box(
                    dequantize_bnb_int8::<F32Out>(
                        black_box(&w),
                        black_box(&s),
                        LAYER_COLS,
                        LAYER_ROWS,
                    )
                    .expect("bnb int8 f32"),
                )
            })
        }),
        benchmark_fn("bnb_int8_f16", move |b| {
            let (w, s) = i8c.clone();
            b.iter(move || {
                black_box(
                    dequantize_bnb_int8::<F16Out>(
                        black_box(&w),
                        black_box(&s),
                        LAYER_COLS,
                        LAYER_ROWS,
                    )
                    .expect("bnb int8 f16"),
                )
            })
        }),
        benchmark_fn("gguf_q4k_bf16", move |b| {
            let r = gg1.clone();
            b.iter(move || {
                black_box(
                    dequantize_gguf_to_bf16(black_box(&r), GgufType::Q4_K, LAYER_ROWS * LAYER_COLS)
                        .expect("gguf q4k bf16"),
                )
            })
        }),
        benchmark_fn("gguf_q4k_f32", move |b| {
            let r = gg2.clone();
            b.iter(move || {
                black_box(
                    dequantize_gguf::<F32Out>(
                        black_box(&r),
                        GgufType::Q4_K,
                        LAYER_ROWS * LAYER_COLS,
                    )
                    .expect("gguf q4k f32"),
                )
            })
        }),
        benchmark_fn("gguf_q4k_f16", move |b| {
            let r = gg3.clone();
            b.iter(move || {
                black_box(
                    dequantize_gguf::<F16Out>(
                        black_box(&r),
                        GgufType::Q4_K,
                        LAYER_ROWS * LAYER_COLS,
                    )
                    .expect("gguf q4k f16"),
                )
            })
        }),
    ]
}

// `tango_benchmarks!` generates `fn main` itself in 0.8; the older
// `tango_main!()` is deprecated and no longer needed.
tango_benchmarks!(dequant_benchmarks());
