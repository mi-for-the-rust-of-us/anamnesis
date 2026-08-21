// SPDX-License-Identifier: MIT OR Apache-2.0

//! Ad-hoc benchmarks for the dequantization kernels.
//!
//! Not part of CI — gated `#[ignore]`. Run with:
//!
//! ```text
//! cargo test --release --features gguf --test bench_dequant_adhoc \
//!     -- --nocapture --ignored
//! ```
//!
//! The synthetic fixtures use byte patterns that exercise the dequant
//! pipelines at realistic layer sizes; actual byte values do not affect
//! timing because the kernels have no data-dependent branches.
//!
//! ## What the GGUF benches measure
//!
//! `bench_gguf_q8_0` and `bench_gguf_q4_0` run the same kernel logic two
//! ways and compare:
//!
//! - **NEW** (current `dequantize_gguf_to_bf16`) — `Vec::with_capacity` +
//!   per-block `extend_from_slice`, the v0.4.0 refactor pattern.
//! - **OLD** (`dequantize_via_indexed_sink`) — bench-local replay of the
//!   pre-refactor pattern: pre-allocate `vec![0u8; out_byte_len]`, drive
//!   the public streaming API `dequantize_gguf_blocks_to_bf16` with a
//!   sink that tracks an offset and writes via indexed slice.
//!
//! Both call the SAME underlying scalar kernels via the same public
//! streaming API; only the output-buffer strategy differs. This is the
//! cleanest possible side-by-side test of the v0.4.0 CHANGELOG claim
//! that `Vec::with_capacity` + `extend_from_slice` saves ~10–15 % over
//! `vec![0u8; n]`.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::indexing_slicing
)]

use std::time::Instant;

use anamnesis::dequantize_per_tensor_fp8_to_bf16;
#[cfg(feature = "gguf")]
use anamnesis::{GgufType, dequantize_gguf_blocks_to_bf16, dequantize_gguf_to_bf16};

/// Median + range of an ascending-sorted `&[f64]`, formatted for stderr.
fn fmt_stats(samples: &[f64]) -> String {
    let median = samples[samples.len() / 2];
    let min = samples[0];
    let max = samples[samples.len() - 1];
    format!("median {median:.2} ms (min {min:.2}, max {max:.2})")
}

/// Best-of-5 timing helper. Calls `f()` 5 times after a 2-iteration
/// warmup, returning the sorted millisecond samples. The closure
/// returns a "live" byte to defeat dead-code elimination in the
/// optimiser.
fn time_best_of_5<F>(mut f: F) -> Vec<f64>
where
    F: FnMut() -> u8,
{
    // Warmup
    let _ = f();
    let _ = f();

    let mut samples: Vec<f64> = Vec::with_capacity(5);
    let mut anti_dce: u64 = 0;
    for _ in 0..5 {
        let start = Instant::now();
        anti_dce = anti_dce.wrapping_add(u64::from(f()));
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        samples.push(ms);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    eprintln!("(anti-DCE accumulator: {anti_dce})");
    samples
}

// ---------------------------------------------------------------------------
// FP8 per-tensor (always-on)
// ---------------------------------------------------------------------------

/// Best-of-5 release-mode median for `dequantize_per_tensor_fp8_to_bf16`
/// on a 4096 × 11008 layer (~45M FP8 elements, ~90 MB FP8 input,
/// ~180 MB BF16 output). Sized like a real Llama-class FFN layer.
#[test]
#[ignore = "ad-hoc benchmark; run with --release --ignored --nocapture"]
fn bench_fp8_per_tensor() {
    const ROWS: usize = 4096;
    const COLS: usize = 11008;
    const N: usize = ROWS * COLS;
    let weight: Vec<u8> = (0..N)
        .map(|i| ((i as u64 * 0x9E37_79B9) >> 24) as u8)
        .collect();
    let scale: f32 = 0.5;

    eprintln!(
        "\n=== bench_fp8_per_tensor ({ROWS} × {COLS} = {} elements, {} MB → {} MB BF16) ===",
        N,
        N / 1_000_000,
        (N * 2) / 1_000_000,
    );

    let samples = time_best_of_5(|| {
        let out = dequantize_per_tensor_fp8_to_bf16(&weight, scale).unwrap();
        out[out.len() - 1]
    });

    eprintln!("samples (ms): {samples:?}");
    eprintln!("{}", fmt_stats(&samples));
    eprintln!(
        "throughput: {:.0} MB/s (BF16 output)",
        ((N * 2) as f64 / 1_000_000.0) / (samples[2] / 1000.0)
    );
}

// ---------------------------------------------------------------------------
// Phase 7.4: the BF16 default must not regress when the four families gain
// their loop-fission split
// ---------------------------------------------------------------------------

/// Best-of-5 release-mode median for **every `remember` kernel family at
/// `BF16`**, on one comparable ~45M-element fixture each.
///
/// This exists for one job: v0.7.4 splits `FP8` / `GPTQ` / `AWQ` / `BnB` into
/// an arithmetic pass plus `OutputElement::write_scratch`, which adds a pass
/// over an L1-resident scratch on the default path. That is a plausible
/// slowdown, and `CONVENTIONS.md` will not let a `// VECTORIZED: confirmed`
/// annotation stand on asm evidence alone — it also wants a measurement
/// showing the kernel is at least as fast as the previous baseline.
///
/// **Deliberately written against the `*_to_bf16` names only**, never the new
/// generic entry points, so the identical test compiles and runs on the parent
/// commit. That is what makes it a before/after comparison rather than an
/// absolute number:
///
/// ```text
/// cargo test --release --all-features --test bench_dequant_adhoc \
///     bench_bf16_all_families -- --nocapture --ignored     # after
/// git stash && git checkout HEAD~1 && <same command>       # before
/// ```
#[test]
#[ignore = "ad-hoc benchmark; run with --release --ignored --nocapture"]
fn bench_bf16_all_families() {
    eprintln!("\n=== bench_bf16_all_families (BF16 default path, best-of-5 median) ===");

    // -- FP8 fine-grained at two shapes, one scale per 128×128 block ---------
    // Two shapes because the v0.7.4 split proved strongly shape-dependent: the
    // column count decides how many 128-wide blocks a row holds, and therefore
    // how often the per-block prologue (`load_scale` + slicing) runs relative
    // to the element work.
    for (rows, cols, label) in [
        (4096_usize, 11008_usize, "fp8_fg_4096x11008"),
        (4096, 4096, "fp8_fg_4096x4096 "),
    ] {
        let weight: Vec<u8> = (0..rows * cols)
            .map(|i| ((i as u64 * 0x9E37_79B9) >> 24) as u8)
            .collect();
        let scale_blocks = rows.div_ceil(128) * cols.div_ceil(128);
        let scales: Vec<u8> = (0..scale_blocks)
            .flat_map(|i| (0.5_f32 + (i % 8) as f32 * 0.01).to_le_bytes())
            .collect();
        let samples = time_best_of_5(|| {
            let out = anamnesis::dequantize_fp8_to_bf16(
                &weight,
                &scales,
                rows,
                cols,
                anamnesis::Dtype::F32,
            )
            .unwrap();
            out[out.len() - 1]
        });
        // Normalised per megaelement so the two shapes compare directly
        // despite differing element counts.
        let per_melem = samples[2] / ((rows * cols) as f64 / 1.0e6);
        eprintln!("{label} {}  [{per_melem:.3} ms/Melem]", fmt_stats(&samples));
    }

    // -- BnB INT8: 4096 × 11008, one SCB per row -----------------------------
    #[cfg(feature = "bnb")]
    {
        const OUT_F: usize = 4096;
        const IN_F: usize = 11008;
        let weight: Vec<u8> = (0..OUT_F * IN_F)
            .map(|i| ((i as u64 * 0x9E37_79B9) >> 24) as u8)
            .collect();
        let scb: Vec<u8> = (0..OUT_F)
            .flat_map(|i| (1.0_f32 + (i % 16) as f32).to_le_bytes())
            .collect();
        let samples = time_best_of_5(|| {
            let out = anamnesis::dequantize_bnb_int8_to_bf16(&weight, &scb, OUT_F, IN_F).unwrap();
            out[out.len() - 1]
        });
        eprintln!("bnb_int8          {}", fmt_stats(&samples));
    }

    // -- BnB NF4: 45M elements, block_size 64 --------------------------------
    #[cfg(feature = "bnb")]
    {
        const N: usize = 4096 * 11008;
        const BLOCK: usize = 64;
        let weight: Vec<u8> = (0..N / 2)
            .map(|i| ((i as u64 * 0x9E37_79B9) >> 24) as u8)
            .collect();
        let absmax: Vec<u8> = (0..N / BLOCK)
            .flat_map(|i| (0.5_f32 + (i % 8) as f32 * 0.01).to_le_bytes())
            .collect();
        let quant_map: Vec<u8> = (0..16)
            .flat_map(|i| ((i as f32 - 8.0) / 8.0).to_le_bytes())
            .collect();
        let samples = time_best_of_5(|| {
            let out =
                anamnesis::dequantize_bnb4_to_bf16(&weight, &absmax, &quant_map, N, BLOCK).unwrap();
            out[out.len() - 1]
        });
        eprintln!("bnb_nf4           {}", fmt_stats(&samples));
    }

    // -- GPTQ INT4: 4096 in × 11008 out, group_size 128 ----------------------
    #[cfg(feature = "gptq")]
    {
        const IN_F: usize = 4096;
        const OUT_F: usize = 11008;
        const GROUP: usize = 128;
        let packed_rows = IN_F / 8;
        let num_groups = IN_F / GROUP;
        let qweight: Vec<u8> = (0..packed_rows * OUT_F * 4)
            .map(|i| ((i as u64 * 0x9E37_79B9) >> 24) as u8)
            .collect();
        let scales: Vec<u8> = (0..num_groups * OUT_F)
            .flat_map(|i| (0.01_f32 + (i % 8) as f32 * 0.001).to_le_bytes())
            .collect();
        let qzeros = vec![0x77u8; num_groups * (OUT_F / 8) * 4];
        let samples = time_best_of_5(|| {
            let out = anamnesis::dequantize_gptq_to_bf16(
                &qweight,
                &scales,
                &qzeros,
                None,
                IN_F,
                OUT_F,
                GROUP,
                4,
                anamnesis::Dtype::F32,
            )
            .unwrap();
            out[out.len() - 1]
        });
        eprintln!("gptq_int4         {}", fmt_stats(&samples));
    }

    // -- AWQ INT4: 4096 in × 11008 out, group_size 128 -----------------------
    #[cfg(feature = "awq")]
    {
        const IN_F: usize = 4096;
        const OUT_F: usize = 11008;
        const GROUP: usize = 128;
        let packed_cols = OUT_F / 8;
        let num_groups = IN_F / GROUP;
        let qweight: Vec<u8> = (0..IN_F * packed_cols * 4)
            .map(|i| ((i as u64 * 0x9E37_79B9) >> 24) as u8)
            .collect();
        let scales: Vec<u8> = (0..num_groups * OUT_F)
            .flat_map(|i| (0.01_f32 + (i % 8) as f32 * 0.001).to_le_bytes())
            .collect();
        let qzeros = vec![0x77u8; num_groups * packed_cols * 4];
        let samples = time_best_of_5(|| {
            let out = anamnesis::dequantize_awq_to_bf16(
                &qweight,
                &scales,
                &qzeros,
                IN_F,
                OUT_F,
                GROUP,
                4,
                anamnesis::Dtype::F32,
            )
            .unwrap();
            out[out.len() - 1]
        });
        eprintln!("awq_int4          {}", fmt_stats(&samples));
    }
}

/// Whole-model `remember` at 1 and 4 threads, the number a user actually feels.
///
/// `bench_bf16_all_families` isolates one kernel on one thread, which is the
/// right level to attribute a codegen change but the wrong level to size its
/// consequences. This bench closes that gap two ways:
///
/// - **Per-tensor parallelism is on**, through `parallel::map_indexed`, exactly
///   as a real `remember` call has it. Threading cannot remove a per-element
///   cost (it scales both sides equally), but it does change how much of the
///   wall clock the kernels account for.
/// - **The serialization and allocation stages are included**, and they dilute
///   the kernel ratio. Phase 7.3 saw `F32` cost 1.79× at kernel level but only
///   1.54–1.61× end to end for exactly this reason.
///
/// Uses only `remember_to_bytes_with_options` + `RememberOptions`, both present
/// before v0.7.4, so the identical test runs on the parent commit's binary.
///
/// Diagnostic for the open discrepancy in
/// `docs/phase-7.4-bf16-perf-discrepancy.md`: puts the isolated kernel and the
/// whole-model path **in one process on one fixture**, so the internal
/// contradiction (end to end faster *per element* than the kernel it contains)
/// can be resolved without any before/after comparison at all.
///
/// Reports, in order:
///
/// 1. the `QuantScheme` the header actually classified to, ruling out the
///    possibility that the two benches run different kernels;
/// 2. the kernel alone, called `TENSORS` times exactly as `dequantize_all`
///    calls it, so the element count matches the whole-model figure;
/// 3. the same again but writing into **one reused output buffer**, which
///    isolates per-iteration allocation and first-touch page-fault cost;
/// 4. the whole-model path at 1 and 4 threads.
///
/// If (2) exceeds (4), the isolated bench is not measuring what its name says.
/// If (3) is far below (2), the gap is allocation, not kernel.
#[test]
#[ignore = "ad-hoc diagnostic; run with --release --ignored --nocapture"]
fn diag_kernel_vs_whole_model_same_process() {
    use anamnesis::{Dtype, RememberOptions, TargetDtype};

    const TENSORS: usize = 4;
    const ROWS: usize = 4096;
    const COLS: usize = 4096;
    const BLOCK: usize = 128;
    const ELEMS: usize = ROWS * COLS;
    let melem = (TENSORS * ELEMS) as f64 / 1.0e6;

    // Same bytes the whole-model fixture uses, built once.
    let weight: Vec<u8> = (0..ELEMS)
        .map(|i| ((i as u64 * 0x9E37_79B9) >> 24) as u8)
        .collect();
    let scale_rows = ROWS.div_ceil(BLOCK);
    let scale_cols = COLS.div_ceil(BLOCK);
    let scales: Vec<u8> = (0..scale_rows * scale_cols)
        .flat_map(|k| (0.125_f32 + (k % 8) as f32 * 0.01).to_le_bytes())
        .collect();

    let mut header = serde_json::Map::new();
    let mut data: Vec<u8> = Vec::new();
    for i in 0..TENSORS {
        let w_off = data.len();
        data.extend_from_slice(&weight);
        header.insert(
            format!("layer.{i}.weight"),
            serde_json::json!({
                "dtype": "F8_E4M3",
                "shape": [ROWS, COLS],
                "data_offsets": [w_off, data.len()],
            }),
        );
        let s_off = data.len();
        data.extend_from_slice(&scales);
        header.insert(
            format!("layer.{i}.weight_scale"),
            serde_json::json!({
                "dtype": "F32",
                "shape": [scale_rows, scale_cols],
                "data_offsets": [s_off, data.len()],
            }),
        );
    }
    let header_json = serde_json::to_string(&header).unwrap();
    let mut file_bytes = Vec::new();
    file_bytes.extend_from_slice(&(header_json.len() as u64).to_le_bytes());
    file_bytes.extend_from_slice(header_json.as_bytes());
    file_bytes.extend_from_slice(&data);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("diag-fp8.safetensors");
    std::fs::write(&path, &file_bytes).unwrap();
    let model = anamnesis::parse(&path).unwrap();

    eprintln!("\n=== diag_kernel_vs_whole_model_same_process ===");
    eprintln!(
        "fixture: {TENSORS} x {ROWS}x{COLS} FP8 = {melem:.1} Melem, scheme = {:?}",
        model.header.scheme
    );
    // The whole-model path only dequantises tensors the classifier tagged
    // `Quantized`. If it tagged them `Passthrough`, `remember` copies bytes and
    // never calls a kernel, which would make every whole-model figure here a
    // measurement of `memcpy`.
    eprintln!(
        "roles: quantized={} scales={} passthrough={}",
        model.header.quantized_count(),
        model.header.scale_count(),
        model.header.passthrough_count()
    );
    let out_len = model
        .remember_to_bytes(TargetDtype::BF16)
        .map_or(0, |b| b.len());
    eprintln!(
        "remember output = {out_len} B; BF16 of all weights would be {} B",
        TENSORS * ELEMS * 2
    );

    // (2) kernel alone, TENSORS calls per iteration, fresh output each call.
    let s = time_best_of_5(|| {
        let mut last = 0u8;
        for _ in 0..TENSORS {
            let out = anamnesis::dequantize_fp8_to_bf16(&weight, &scales, ROWS, COLS, Dtype::F32)
                .unwrap();
            last = out[out.len() - 1];
        }
        last
    });
    eprintln!(
        "kernel_fresh_alloc    {}  [{:.3} ms/Melem]",
        fmt_stats(&s),
        s[2] / melem
    );

    // (2b) the same work, but through the **generic** entry point rather than
    // the non-generic `*_to_bf16` wrapper. `dequantize_fp8::<Bf16Out>` is
    // instantiated in *this* crate, so LLVM can inline and specialise it the
    // way it does for `remember`'s lib-internal call. The wrapper is an opaque
    // cross-crate call by comparison. If this arm is much faster than (2), then
    // arm (2) never measured what `remember` executes.
    let s_generic = time_best_of_5(|| {
        let mut last = 0u8;
        for _ in 0..TENSORS {
            let out = anamnesis::dequantize_fp8::<anamnesis::Bf16Out>(
                &weight,
                &scales,
                ROWS,
                COLS,
                Dtype::F32,
            )
            .unwrap();
            last = out[out.len() - 1];
        }
        last
    });
    eprintln!(
        "kernel_generic_inline {}  [{:.3} ms/Melem]",
        fmt_stats(&s_generic),
        s_generic[2] / melem
    );

    // (2c) four *distinct* input buffers, matching the whole-model path, which
    // reads four separate mmap regions. Arms (2) and (2b) reuse one 16.7 MB
    // buffer four times; if that reuse creates a cache-conflict pattern against
    // the 33.5 MB output, this arm will be much faster and the reuse is the
    // artefact.
    let weights: Vec<Vec<u8>> = (0..TENSORS).map(|_| weight.clone()).collect();
    let s_distinct = time_best_of_5(|| {
        let mut last = 0u8;
        for w in &weights {
            let out =
                anamnesis::dequantize_fp8_to_bf16(w, &scales, ROWS, COLS, Dtype::F32).unwrap();
            last = out[out.len() - 1];
        }
        last
    });
    eprintln!(
        "kernel_distinct_bufs  {}  [{:.3} ms/Melem]",
        fmt_stats(&s_distinct),
        s_distinct[2] / melem
    );

    // (2d) one call only, to check the per-tensor cost is linear rather than
    // degrading across repeated 33.5 MB allocate/write/free cycles.
    let s_one = time_best_of_5(|| {
        let out =
            anamnesis::dequantize_fp8_to_bf16(&weight, &scales, ROWS, COLS, Dtype::F32).unwrap();
        out[out.len() - 1]
    });
    eprintln!(
        "kernel_single_call    {}  [{:.3} ms/Melem]",
        fmt_stats(&s_one),
        s_one[2] / (ELEMS as f64 / 1.0e6)
    );

    // (3) the allocation control: same total bytes, no kernel work at all.
    let s_alloc = time_best_of_5(|| {
        let mut last = 0u8;
        for _ in 0..TENSORS {
            let out = vec![0u8; ELEMS * 2];
            last = out[out.len() - 1];
        }
        last
    });
    eprintln!(
        "alloc_only_control    {}  [{:.3} ms/Melem]",
        fmt_stats(&s_alloc),
        s_alloc[2] / melem
    );

    // (3b) the same whole-model work, but with the input in an **owned heap
    // buffer** instead of an mmap (`parse_bytes` vs `parse`). That is the last
    // structural difference between the isolated arms, which read a `Vec`, and
    // the whole-model arm, which reads a memory map.
    {
        let owned = anamnesis::parse_bytes(file_bytes.clone()).unwrap();
        let s_owned = time_best_of_5(|| {
            let bytes = owned
                .remember_to_bytes_with_options(
                    TargetDtype::BF16,
                    RememberOptions::new().with_threads(1),
                )
                .unwrap();
            bytes[bytes.len() - 1]
        });
        eprintln!(
            "whole_model_owned_t1  {}  [{:.3} ms/Melem]",
            fmt_stats(&s_owned),
            s_owned[2] / melem
        );
    }

    // (4) whole model.
    for threads in [1usize, 4] {
        let s = time_best_of_5(|| {
            let bytes = model
                .remember_to_bytes_with_options(
                    TargetDtype::BF16,
                    RememberOptions::new().with_threads(threads),
                )
                .unwrap();
            bytes[bytes.len() - 1]
        });
        eprintln!(
            "whole_model_t{threads:<9} {}  [{:.3} ms/Melem]",
            fmt_stats(&s),
            s[2] / melem
        );
    }
}

#[test]
#[ignore = "ad-hoc benchmark; run with --release --ignored --nocapture"]
fn bench_remember_whole_model_threaded() {
    use anamnesis::{RememberOptions, TargetDtype};

    // 4 layers of 4096 × 4096 FP8, each with its own 32 × 32 block-scale grid.
    //
    // Two deliberate choices, both learned the hard way:
    //
    // - **Fine-grained, not per-tensor.** That is the kernel
    //   `bench_bf16_all_families` measures in isolation, so the two numbers are
    //   directly comparable.
    // - **Realistically sized tensors.** A first draft used 24 × 1024 × 1024,
    //   whose 2 MB of output per tensor stays cache-resident, and it reported
    //   the v0.7.4 kernels **1.78× faster** — the exact opposite of the
    //   isolated bench's 1.17× slower, from the same two binaries. Real
    //   attention and FFN weights are 4096 × 4096 and larger, i.e. tens of MB
    //   of output per tensor and firmly DRAM-bound, which is the regime where
    //   the extra pass actually costs something. Sizing the fixture below that
    //   knee measures a regime no real model is in.
    const TENSORS: usize = 4;
    const ROWS: usize = 4096;
    const COLS: usize = 4096;
    const BLOCK: usize = 128;

    let mut header = serde_json::Map::new();
    let mut data: Vec<u8> = Vec::new();
    for i in 0..TENSORS {
        let w_off = data.len();
        data.extend((0..ROWS * COLS).map(|k| ((k as u64 * 0x9E37_79B9) >> 24) as u8));
        header.insert(
            format!("layer.{i}.weight"),
            serde_json::json!({
                "dtype": "F8_E4M3",
                "shape": [ROWS, COLS],
                "data_offsets": [w_off, data.len()],
            }),
        );
        let s_off = data.len();
        let scale_rows = ROWS.div_ceil(BLOCK);
        let scale_cols = COLS.div_ceil(BLOCK);
        for k in 0..scale_rows * scale_cols {
            data.extend_from_slice(&(0.125_f32 + (k % 8) as f32 * 0.01).to_le_bytes());
        }
        header.insert(
            format!("layer.{i}.weight_scale"),
            serde_json::json!({
                "dtype": "F32",
                "shape": [scale_rows, scale_cols],
                "data_offsets": [s_off, data.len()],
            }),
        );
    }
    let header_json = serde_json::to_string(&header).unwrap();
    let mut file_bytes = Vec::new();
    file_bytes.extend_from_slice(&(header_json.len() as u64).to_le_bytes());
    file_bytes.extend_from_slice(header_json.as_bytes());
    file_bytes.extend_from_slice(&data);

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("synth-fp8.safetensors");
    std::fs::write(&path, &file_bytes).unwrap();
    let model = anamnesis::parse(&path).unwrap();

    eprintln!(
        "\n=== bench_remember_whole_model_threaded ({TENSORS} tensors × {ROWS}×{COLS} fine-grained FP8) ==="
    );
    for threads in [1usize, 4] {
        let samples = time_best_of_5(|| {
            let bytes = model
                .remember_to_bytes_with_options(
                    TargetDtype::BF16,
                    RememberOptions::new().with_threads(threads),
                )
                .unwrap();
            bytes[bytes.len() - 1]
        });
        eprintln!("remember_bf16_t{threads:<2}      {}", fmt_stats(&samples));
    }
}

// ---------------------------------------------------------------------------
// GGUF: NEW (Vec::with_capacity + extend_from_slice) vs OLD
// (vec![0u8; n] + sink-with-offset)
// ---------------------------------------------------------------------------

/// Bench-local replay of the pre-v0.4.0-refactor pattern: pre-allocate a
/// zero-initialised `Vec<u8>` of the exact output size and have the
/// streaming API write into it via indexed `copy_from_slice` with an
/// offset cursor. Same kernel logic as `dequantize_gguf_to_bf16`, only
/// the output-buffer strategy differs.
#[cfg(feature = "gguf")]
fn dequantize_via_indexed_sink(
    data: &[u8],
    dtype: GgufType,
    n_elements: usize,
) -> anamnesis::Result<Vec<u8>> {
    let out_byte_len = n_elements
        .checked_mul(2)
        .expect("output size overflow in bench fixture");
    let mut out = vec![0u8; out_byte_len];
    let mut offset = 0usize;
    dequantize_gguf_blocks_to_bf16(data, dtype, n_elements, |block_out| {
        out[offset..offset + block_out.len()].copy_from_slice(block_out);
        offset += block_out.len();
        Ok(())
    })?;
    Ok(out)
}

/// Synthesizes `n_blocks` of `Q8_0`-formatted bytes (34 bytes per
/// 32-element block: `f16 d` + `i8 qs[32]`). Byte values are arbitrary
/// — the kernel has no data-dependent branches, so timing is identical
/// to a real model's bytes. Using a non-zero `d` ensures the runtime
/// `d × qs[j]` multiplications are not optimised away.
#[cfg(feature = "gguf")]
fn build_q8_0_buffer(n_blocks: usize) -> Vec<u8> {
    const BLOCK_BYTES: usize = 34;
    let mut buf = vec![0u8; n_blocks * BLOCK_BYTES];
    // Set d = f16(1.0) = 0x3C00 in every block (stored LE in bytes 0..2).
    // Keep qs[32] = 0..0 (irrelevant for timing).
    for block in buf.as_chunks_mut::<BLOCK_BYTES>().0 {
        block[0] = 0x00;
        block[1] = 0x3C;
    }
    buf
}

/// Synthesizes `n_blocks` of `Q4_0`-formatted bytes (18 bytes per
/// 32-element block: `f16 d` + 16 bytes of packed nibbles).
#[cfg(feature = "gguf")]
fn build_q4_0_buffer(n_blocks: usize) -> Vec<u8> {
    const BLOCK_BYTES: usize = 18;
    let mut buf = vec![0u8; n_blocks * BLOCK_BYTES];
    for block in buf.as_chunks_mut::<BLOCK_BYTES>().0 {
        block[0] = 0x00;
        block[1] = 0x3C;
    }
    buf
}

/// Runs the NEW vs OLD comparison for a single `(dtype, n_elements)`
/// configuration and prints a one-line result row. Returns the signed
/// percent delta of NEW vs OLD median.
#[cfg(feature = "gguf")]
fn run_gguf_one(label: &str, data: &[u8], dtype: GgufType, n_elements: usize) -> f64 {
    let samples_new = time_best_of_5(|| {
        let out = dequantize_gguf_to_bf16(data, dtype, n_elements).unwrap();
        out[out.len() - 1]
    });
    let samples_old = time_best_of_5(|| {
        let out = dequantize_via_indexed_sink(data, dtype, n_elements).unwrap();
        out[out.len() - 1]
    });

    let median_new = samples_new[2];
    let median_old = samples_old[2];
    let delta_pct = (median_new - median_old) / median_old * 100.0;
    eprintln!(
        "{label:<20}  NEW {median_new:>7.2} ms (range {:.2}-{:.2})  \
         OLD {median_old:>7.2} ms (range {:.2}-{:.2})  Δ {delta_pct:+.1}%",
        samples_new[0], samples_new[4], samples_old[0], samples_old[4],
    );
    delta_pct
}

/// Sweeps `dequantize_gguf_to_bf16` (NEW) vs the `vec![0u8; n]`
/// indexed-sink alternative (OLD) across four output sizes spanning the
/// L3-resident → deeply-DRAM-bound regime, on both `Q8_0` and `Q4_0`.
///
/// Sizes (output BF16 bytes):
/// - **2 MB** (1M elements) — output fits comfortably in L3 on most CPUs.
/// - **16 MB** (8M elements) — output spills to DRAM on smaller L3 caches.
/// - **90 MB** (45M elements) — original single-size measurement (Llama FFN scale).
/// - **200 MB** (100M elements) — solidly DRAM-bound, tests memory-pressure regime.
///
/// If the directional finding (`Q8_0` NEW slower / `Q4_0` NEW faster) is
/// real, it should hold across all four sizes. If it flips at some
/// size, the bottleneck is cache-resident vs DRAM-bound and the
/// measurement at any single size was misleading.
#[cfg(feature = "gguf")]
#[test]
#[ignore = "ad-hoc benchmark; run with --release --features gguf --ignored --nocapture"]
fn bench_gguf_size_sweep() {
    // (label, n_elements). Each must be a multiple of 32 (Q4_0/Q8_0 block size).
    const SIZES: &[(&str, usize)] = &[
        ("1M (2 MB BF16)", 1_048_576),
        ("8M (16 MB BF16)", 8 * 1_048_576),
        ("45M (90 MB BF16)", 4096 * 11008),
        ("100M (200 MB BF16)", 100 * 1_048_576),
    ];

    eprintln!(
        "\n=== bench_gguf_size_sweep — NEW (current Vec::with_capacity + extend_from_slice) \
         vs OLD (vec![0u8; n] + indexed sink) ===\n"
    );

    eprintln!("--- Q8_0 ---");
    let mut q8_deltas: Vec<f64> = Vec::with_capacity(SIZES.len());
    for &(label, n) in SIZES {
        let data = build_q8_0_buffer(n / 32);
        let delta = run_gguf_one(label, &data, GgufType::Q8_0, n);
        q8_deltas.push(delta);
    }

    eprintln!("\n--- Q4_0 ---");
    let mut q4_deltas: Vec<f64> = Vec::with_capacity(SIZES.len());
    for &(label, n) in SIZES {
        let data = build_q4_0_buffer(n / 32);
        let delta = run_gguf_one(label, &data, GgufType::Q4_0, n);
        q4_deltas.push(delta);
    }

    eprintln!(
        "\n--- Summary: NEW vs OLD median deltas across sizes ---\n\
         Q8_0 deltas: {q8_deltas:+.1?}\n\
         Q4_0 deltas: {q4_deltas:+.1?}"
    );
    eprintln!(
        "\nDirectional finding holds if all Q8_0 deltas have the same \
         sign and all Q4_0 deltas have the same (opposite) sign."
    );
}
