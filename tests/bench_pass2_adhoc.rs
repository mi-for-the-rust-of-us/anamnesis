// SPDX-License-Identifier: MIT OR Apache-2.0

//! Phase 7 Stage-0 ad-hoc benchmarks for the shared pass-2 `f32 -> BF16`
//! writer and the kernels that feed it.
//!
//! Not part of CI — every test is gated `#[ignore]`. The point of this
//! harness is to answer, *before* any SIMD `unsafe` is written, the two
//! questions Phase 7 hinges on:
//!
//! 1. **Is the shared writer bandwidth-bound?** `bench_pure_writer` runs the
//!    exact loop that [`crate::remember::gguf::write_scratch_to_bf16`] runs
//!    (replicated here because that helper is `pub(crate)`), and
//!    `bench_memcpy_ceiling` measures the practical streaming ceiling on this
//!    machine. If the writer's GB/s approaches the memcpy ceiling it is
//!    bandwidth-bound and explicit AVX2 cannot help it; if it sits well below,
//!    there is compute headroom for SIMD to recover.
//! 2. **What is each kernel's scalar baseline?** The FP8 (1:1, closest to the
//!    pure writer) and GGUF `Q8_0` / `Q4_0` (light unpack) end-to-end benches
//!    establish the scalar medians the SIMD paths must beat.
//!
//! ## Running the two baselines
//!
//! The SSE2-vs-native comparison is driven by recompiling this same binary
//! under two `target-cpu` settings — no `cfg` in the file. PowerShell:
//!
//! ```text
//! # native (local reference; the CLAUDE.md perf-gate baseline)
//! $env:RUSTFLAGS = "-C target-cpu=native"
//! cargo test --release --features gguf --test bench_pass2_adhoc -- --ignored --nocapture
//!
//! # SSE2 (the default-PyPI-wheel baseline; the Phase 8 wheel-relevant number)
//! $env:RUSTFLAGS = "-C target-cpu=x86-64"
//! cargo test --release --features gguf --test bench_pass2_adhoc -- --ignored --nocapture
//!
//! $env:RUSTFLAGS = $null
//! ```
//!
//! Byte values in the synthetic fixtures do not affect timing — every kernel
//! here is branch-free, so a fixed pseudo-random fill times identically to a
//! real model's bytes while defeating constant-folding.

// This is a measurement prototype (Phase 7 SIMD exhaustion): it hand-writes an
// AVX2 kernel to prove/disprove the ROADMAP's "SIMD the pass-2 writer" thesis
// before any product `unsafe` is committed. `unsafe` is allowed here (the crate
// lint denies it library-wide) with `// SAFETY:` on the one intrinsic block.
#![allow(unsafe_code)]
#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::indexing_slicing,
    // The `*const f32 -> *const __m256i` cast feeds `_mm256_loadu_si256`, an
    // UNALIGNED load — the higher alignment the lint warns about is never required.
    clippy::cast_ptr_alignment,
    // This dev-only harness's module doc is prose (PowerShell commands, run
    // recipes). MSRV-1.88 clippy's `doc_markdown` allowlist lacks terms newer
    // clippy accepts (e.g. "PowerShell"), so allow it here rather than backtick
    // every prose word to satisfy the oldest toolchain.
    clippy::doc_markdown
)]

use std::time::Instant;

use anamnesis::dequantize_per_tensor_fp8_to_bf16;
#[cfg(feature = "gguf")]
use anamnesis::{GgufType, dequantize_gguf_to_bf16};

// ---------------------------------------------------------------------------
// Timing helpers (mirrors tests/bench_dequant_adhoc.rs)
// ---------------------------------------------------------------------------

/// Median + range of an ascending-sorted `&[f64]`, formatted for stderr.
fn fmt_stats(samples: &[f64]) -> String {
    let median = samples[samples.len() / 2];
    let min = samples[0];
    let max = samples[samples.len() - 1];
    format!("median {median:.3} ms (min {min:.3}, max {max:.3})")
}

/// Best-of-N timing helper. Calls `f()` `N` times after a 3-iteration warmup,
/// returning the sorted millisecond samples. The closure returns a "live"
/// byte to defeat dead-code elimination in the optimiser. `N = 9` (up from the
/// sibling harness's 5) because this runs on an interactive machine where a few
/// samples get perturbed by background load — a larger `N` makes the **min**
/// (the least-perturbed sample, the honest hardware ceiling) reliable.
const SAMPLES: usize = 9;

fn time_best_of_n<F>(mut f: F) -> Vec<f64>
where
    F: FnMut() -> u8,
{
    for _ in 0..3 {
        let _ = f();
    }

    let mut samples: Vec<f64> = Vec::with_capacity(SAMPLES);
    let mut anti_dce: u64 = 0;
    for _ in 0..SAMPLES {
        let start = Instant::now();
        anti_dce = anti_dce.wrapping_add(u64::from(f()));
        samples.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    eprintln!("(anti-DCE accumulator: {anti_dce})");
    samples
}

/// Prints median + range, plus a **min-time** (best-case) GB/s figure.
/// `bytes_streamed` is the read + write traffic per iteration; the min-time
/// throughput is the honest ceiling metric — least perturbed by background
/// load — and is directly comparable to the memcpy ceiling for the roofline
/// classification. Median is reported too for wall-clock context.
fn report(label: &str, samples: &[f64], bytes_streamed: usize) {
    let min_ms = samples[0];
    let median_ms = samples[samples.len() / 2];
    let gbps_min = (bytes_streamed as f64 / 1e9) / (min_ms / 1000.0);
    let gbps_median = (bytes_streamed as f64 / 1e9) / (median_ms / 1000.0);
    eprintln!("{label}: {}", fmt_stats(samples));
    eprintln!("{label}: {gbps_min:.1} GB/s (min-time ceiling) / {gbps_median:.1} GB/s (median)");
}

// ---------------------------------------------------------------------------
// Scalar oracle replica — kept byte-identical to
// `remember::fp8::f32_bits_to_bf16_bits` (the `pub(crate)` original is not
// reachable from an integration test). Round-to-nearest-even via the
// `0x7FFF + lsb` bias. This replica is ALSO the golden-vector oracle the
// Stage-2 AVX2/NEON intrinsics will be checked against.
// ---------------------------------------------------------------------------

fn f32_bits_to_bf16_bits(bits: u32) -> u16 {
    let lsb = (bits >> 16) & 1;
    let rounding_bias = 0x7FFF_u32 + lsb;
    (bits.wrapping_add(rounding_bias) >> 16) as u16
}

/// The exact loop shape of `write_scratch_to_bf16`: contiguous f32 read,
/// branch-free convert, contiguous 2-byte write, distinct in/out slices.
fn scalar_write_bf16(scratch: &[f32], out: &mut [u8]) {
    for (&val, out_pair) in scratch.iter().zip(out.as_chunks_mut::<2>().0) {
        let bf16 = f32_bits_to_bf16_bits(val.to_bits());
        out_pair.copy_from_slice(&bf16.to_le_bytes());
    }
}

/// Explicit AVX2 `f32x8 → BF16x8` — the ROADMAP's Phase-7 centerpiece,
/// hand-written to test whether it beats the (already auto-vectorized) scalar
/// path. Lane-for-lane identical to `f32_bits_to_bf16_bits`: round-to-nearest-
/// even via the `0x7FFF + lsb` bias, then take the upper 16 bits and pack to u16.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_write_bf16(scratch: &[f32], out: &mut [u8]) {
    use std::arch::x86_64::{
        __m256i, _mm_storeu_si128, _mm256_add_epi32, _mm256_and_si256, _mm256_castsi256_si128,
        _mm256_loadu_si256, _mm256_packus_epi32, _mm256_permute4x64_epi64, _mm256_set1_epi32,
        _mm256_srli_epi32,
    };
    // SAFETY: caller guarantees AVX2 is available (checked via
    // is_x86_feature_detected! before dispatch). That is the function-wide
    // precondition; the per-operation bounds arguments sit at each `unsafe`
    // block below, which Edition 2024 requires to be written out rather than
    // inherited from the `unsafe fn` signature (`unsafe_op_in_unsafe_fn`).
    let n = scratch.len().min(out.len() / 2);
    let chunks = n / 8;
    let bias_base = _mm256_set1_epi32(0x7FFF);
    let one = _mm256_set1_epi32(1);
    for c in 0..chunks {
        // SAFETY: `c < chunks` and `chunks * 8 <= n <= scratch.len()`, so the
        // 8-lane read at element offset `c * 8` lies wholly inside `scratch`.
        // The load is unaligned (`loadu`), so no alignment precondition applies.
        let v = unsafe { _mm256_loadu_si256(scratch.as_ptr().add(c * 8).cast::<__m256i>()) };
        let lsb = _mm256_and_si256(_mm256_srli_epi32::<16>(v), one);
        let bias = _mm256_add_epi32(bias_base, lsb);
        let rounded = _mm256_srli_epi32::<16>(_mm256_add_epi32(v, bias));
        // Pack 8×u32 (each ≤ 0xFFFF, so packus does not saturate) → u16.
        let packed = _mm256_packus_epi32(rounded, rounded);
        // packus is per-128-bit-lane; permute u64 lanes [0,2] into the low 128
        // to make the 8 wanted u16 contiguous.
        let perm = _mm256_permute4x64_epi64::<0b11_01_10_00>(packed);
        // SAFETY: `n <= out.len() / 2`, so `chunks * 16 <= 2n <= out.len()` and
        // the 16-byte write at byte offset `c * 16` lies wholly inside `out`.
        // The store is unaligned (`storeu`). `out` and `scratch` are distinct
        // slices, so the write cannot alias the load above.
        unsafe {
            _mm_storeu_si128(
                out.as_mut_ptr().add(c * 16).cast(),
                _mm256_castsi256_si128(perm),
            );
        }
    }
    for i in chunks * 8..n {
        let bf16 = f32_bits_to_bf16_bits(scratch[i].to_bits());
        out[2 * i] = bf16 as u8;
        out[2 * i + 1] = (bf16 >> 8) as u8;
    }
}

/// Bit-exactness gate: the AVX2 writer must be 0-ULP identical to the scalar
/// oracle over a large varied sample (incl. the 8-lane tail) before any timing
/// is trusted. Runs only where AVX2 is present.
#[cfg(target_arch = "x86_64")]
#[test]
#[ignore = "ad-hoc benchmark; run with --release --ignored --nocapture"]
fn avx2_writer_is_bit_exact() {
    if !std::arch::is_x86_feature_detected!("avx2") {
        eprintln!("AVX2 not available — skipping bit-exactness check");
        return;
    }
    // 100003 is prime → exercises the non-multiple-of-8 scalar tail.
    let scratch = build_f32_scratch(100_003);
    let mut out_scalar = vec![0u8; scratch.len() * 2];
    let mut out_avx2 = vec![0u8; scratch.len() * 2];
    scalar_write_bf16(&scratch, &mut out_scalar);
    // SAFETY: guarded by the is_x86_feature_detected! check above.
    unsafe { avx2_write_bf16(&scratch, &mut out_avx2) };
    assert_eq!(
        out_scalar, out_avx2,
        "AVX2 writer diverged from scalar oracle"
    );
    eprintln!(
        "avx2_writer_is_bit_exact: 0 ULP over {} elements ✓",
        scratch.len()
    );
}

// ---------------------------------------------------------------------------
// Roofline: pure writer vs memcpy ceiling
// ---------------------------------------------------------------------------

/// Number of f32 elements (~45M) sized like a Llama-class FFN layer:
/// ~180 MB f32 scratch in, ~90 MB BF16 out — safely DRAM-bound.
const WRITER_N: usize = 4096 * 11008;

/// A varied f32 fill: non-trivial exponents/mantissas so the RNE bias path is
/// exercised and nothing constant-folds.
fn build_f32_scratch(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let bits = ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 32) as u32;
            // Keep it finite: clear the top exponent bits so we never make inf/NaN,
            // which would be uninteresting (and identical timing anyway).
            f32::from_bits(bits & 0x7F7F_FFFF)
        })
        .collect()
}

/// The pure pass-2 writer in isolation — the exact code the SIMD helper
/// replaces. Its GB/s vs the memcpy ceiling classifies the writer as
/// bandwidth-bound (SIMD cannot help) or compute-bound (headroom exists).
#[test]
#[ignore = "ad-hoc benchmark; run with --release --ignored --nocapture"]
fn bench_pure_writer() {
    let scratch = build_f32_scratch(WRITER_N);
    let mut out = vec![0u8; WRITER_N * 2];
    eprintln!(
        "\n=== bench_pure_writer ({WRITER_N} elems, {} MB f32 -> {} MB BF16) ===",
        WRITER_N * 4 / 1_000_000,
        WRITER_N * 2 / 1_000_000,
    );
    let samples = time_best_of_n(|| {
        scalar_write_bf16(&scratch, &mut out);
        out[out.len() - 1]
    });
    // Traffic = read 4 B/elem (f32) + write 2 B/elem (BF16) = 6 B/elem.
    report("pure_writer", &samples, WRITER_N * 6);
}

/// SIMD exhaustion: scalar vs explicit-AVX2 writer, both timed in the SAME job
/// on the SAME buffer (within-job A/B → the ratio is robust to runner-CPU
/// variance, the CI-friendly design). If the AVX2 ratio is ~1.0 the writer is
/// confirmed bandwidth-bound and the ROADMAP's "SIMD the pass-2 writer for
/// 4–8×" thesis is disproven with a real intrinsic implementation.
#[cfg(target_arch = "x86_64")]
#[test]
#[ignore = "ad-hoc benchmark; run with --release --ignored --nocapture"]
fn bench_writer_scalar_vs_avx2() {
    if !std::arch::is_x86_feature_detected!("avx2") {
        eprintln!("AVX2 not available — skipping");
        return;
    }
    let scratch = build_f32_scratch(WRITER_N);
    let mut out = vec![0u8; WRITER_N * 2];
    eprintln!("\n=== bench_writer_scalar_vs_avx2 ({WRITER_N} elems) ===");
    let scalar = time_best_of_n(|| {
        scalar_write_bf16(&scratch, &mut out);
        out[out.len() - 1]
    });
    let avx2 = time_best_of_n(|| {
        // SAFETY: guarded by the is_x86_feature_detected! check above.
        unsafe { avx2_write_bf16(&scratch, &mut out) };
        out[out.len() - 1]
    });
    report("writer_scalar", &scalar, WRITER_N * 6);
    report("writer_avx2  ", &avx2, WRITER_N * 6);
    let ratio = scalar[0] / avx2[0];
    eprintln!("writer_avx2 speedup vs scalar (min-time): {ratio:.2}× (>1 = AVX2 faster)");
}

/// The streaming ceiling on this machine: a large `copy_from_slice`. Any
/// kernel whose GB/s approaches this number is bandwidth-bound.
#[test]
#[ignore = "ad-hoc benchmark; run with --release --ignored --nocapture"]
fn bench_memcpy_ceiling() {
    // Match the pure_writer read+write footprint (~270 MB touched). A varied
    // fill plus `black_box` on both operands stops the optimiser from proving
    // the copy is dead (a constant `src` folds `dst[last]` to a constant and
    // elides the whole memcpy — which is what happened before this guard).
    let bytes = WRITER_N * 3;
    let src: Vec<u8> = (0..bytes).map(|i| (i & 0xFF) as u8).collect();
    let mut dst = vec![0u8; bytes];
    eprintln!(
        "\n=== bench_memcpy_ceiling ({} MB copy) ===",
        bytes / 1_000_000
    );
    let samples = time_best_of_n(|| {
        let s = std::hint::black_box(src.as_slice());
        dst.copy_from_slice(s);
        let d = std::hint::black_box(dst.as_slice());
        d[d.len() - 1]
    });
    // Traffic = read `bytes` + write `bytes`.
    report("memcpy_ceiling", &samples, bytes * 2);
}

// ---------------------------------------------------------------------------
// End-to-end per-kernel scalar baselines
// ---------------------------------------------------------------------------

/// FP8 per-tensor — the 1:1 kernel closest to the pure writer (one input byte
/// -> one BF16, single scale multiply). ~45M elements.
#[test]
#[ignore = "ad-hoc benchmark; run with --release --ignored --nocapture"]
fn bench_fp8_per_tensor() {
    const N: usize = WRITER_N;
    let weight: Vec<u8> = (0..N)
        .map(|i| ((i as u64).wrapping_mul(0x9E37_79B9) >> 24) as u8)
        .collect();
    let scale: f32 = 0.5;
    eprintln!(
        "\n=== bench_fp8_per_tensor ({N} elems, {} MB FP8 -> {} MB BF16) ===",
        N / 1_000_000,
        N * 2 / 1_000_000,
    );
    let samples = time_best_of_n(|| {
        let out = dequantize_per_tensor_fp8_to_bf16(&weight, scale).unwrap();
        out[out.len() - 1]
    });
    // Traffic = read 1 B/elem (FP8) + write 2 B/elem (BF16) = 3 B/elem.
    report("fp8_per_tensor", &samples, N * 3);
}

/// Multi-threading prototype (Experiment 10 follow-up): does splitting a real
/// dequant across `std::thread::scope` threads scale? Per-tensor FP8 is
/// element-independent (uniform scale), so chunking the input at any byte
/// boundary and concatenating is byte-identical to the sequential result — the
/// determinism invariant `CONVENTIONS.md` requires. Reports speedup vs 1 thread
/// at 1/2/4/8/16 threads. FP8 is compute-bound (Experiment 10), so this is the
/// near-linear case; contrast with the bandwidth-bound writer below.
#[test]
#[ignore = "ad-hoc benchmark; run with --release --ignored --nocapture"]
fn bench_parallel_fp8_scaling() {
    const N: usize = WRITER_N; // ~45 MB FP8 input
    let weight: Vec<u8> = (0..N)
        .map(|i| ((i as u64).wrapping_mul(0x9E37_79B9) >> 24) as u8)
        .collect();
    let scale: f32 = 0.5;

    // Determinism gate: parallel (8-way) concat must equal the sequential bytes.
    let seq = dequantize_per_tensor_fp8_to_bf16(&weight, scale).unwrap();
    let chunk8 = weight.len().div_ceil(8);
    let par8: Vec<u8> = std::thread::scope(|s| {
        let handles: Vec<_> = weight
            .chunks(chunk8)
            .map(|c| s.spawn(move || dequantize_per_tensor_fp8_to_bf16(c, scale).unwrap()))
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect()
    });
    assert_eq!(seq, par8, "parallel FP8 output diverged from sequential");
    eprintln!("\n=== bench_parallel_fp8_scaling (determinism: 8-way == sequential ✓) ===");

    let mut baseline = 0.0_f64;
    for threads in [1usize, 2, 4, 8, 16] {
        let chunk = weight.len().div_ceil(threads);
        let samples = time_best_of_n(|| {
            std::thread::scope(|s| {
                let handles: Vec<_> = weight
                    .chunks(chunk)
                    .map(|c| s.spawn(move || dequantize_per_tensor_fp8_to_bf16(c, scale).unwrap()))
                    .collect();
                let mut acc = 0u8;
                for h in handles {
                    let v = h.join().unwrap();
                    acc ^= v[v.len() - 1];
                }
                acc
            })
        });
        let median = samples[samples.len() / 2];
        if threads == 1 {
            baseline = median;
        }
        eprintln!(
            "  threads={threads:2}: median {median:6.2} ms  speedup {:.2}×  ({:.0} MB/s BF16)",
            baseline / median,
            (N * 2) as f64 / 1_000_000.0 / (median / 1000.0),
        );
    }
}

/// Branchless E4M3 → f32-bits, replicating `remember::fp8::e4m3_to_f32_bits`
/// (the `pub(crate)` original is unreachable here). Subnormal table built at
/// runtime: subnormal value = `mant × 2^-9`.
fn e4m3_to_f32_bits_replica(byte: u8, sub_table: &[u32; 8]) -> u32 {
    let b = u32::from(byte);
    let sign = b >> 7;
    let exp = (b >> 3) & 0xF;
    let mant = b & 0x7;
    let normal_bits = (sign << 31) | ((exp + 120) << 23) | (mant << 20);
    let sub_bits = sub_table[mant as usize] | (sign << 31);
    let sub_flag = exp.wrapping_sub(1) >> 31;
    let sub_mask = 0u32.wrapping_sub(sub_flag);
    let selected = (sub_bits & sub_mask) | (normal_bits & !sub_mask);
    let nan_flag = ((b & 0x7F) ^ 0x7F).wrapping_sub(1) >> 31;
    let nan_mask = 0u32.wrapping_sub(nan_flag);
    let nan_bits = (sign << 31) | 0x7FC0_0000;
    (nan_bits & nan_mask) | (selected & !nan_mask)
}

/// Per-tensor FP8 dequant writing into a caller-provided disjoint `out` slice —
/// the `CONVENTIONS.md` disjoint-output-region pattern (no per-thread alloc, no
/// zero-fill double-write). This is what a real threaded implementation would do.
fn fp8_dequant_into(input: &[u8], scale: f32, out: &mut [u8], sub_table: &[u32; 8]) {
    for (&b, pair) in input.iter().zip(out.as_chunks_mut::<2>().0) {
        let scaled = f32::from_bits(e4m3_to_f32_bits_replica(b, sub_table)) * scale;
        let bf16 = f32_bits_to_bf16_bits(scaled.to_bits());
        pair.copy_from_slice(&bf16.to_le_bytes());
    }
}

/// The HONEST threading-scaling measurement: one pre-allocated output buffer,
/// each thread writes a disjoint `split_at_mut` slice — no per-thread allocation,
/// no zero-fill. Contrast `bench_parallel_fp8_scaling` (Vec-per-thread) to see the
/// allocation artifact this eliminates.
#[test]
#[ignore = "ad-hoc benchmark; run with --release --ignored --nocapture"]
fn bench_parallel_fp8_disjoint_slices() {
    const N: usize = WRITER_N;
    let weight: Vec<u8> = (0..N)
        .map(|i| ((i as u64).wrapping_mul(0x9E37_79B9) >> 24) as u8)
        .collect();
    let scale: f32 = 0.5;
    let sub_table: [u32; 8] = core::array::from_fn(|m| (m as f32 / 512.0).to_bits());
    let mut out = vec![0u8; N * 2];

    eprintln!("\n=== bench_parallel_fp8_disjoint_slices (pre-allocated output, split_at_mut) ===");
    let mut baseline = 0.0_f64;
    for threads in [1usize, 2, 4, 8, 16] {
        // Rows-of-work: split output into `threads` disjoint 2-byte-aligned slices,
        // pair each with its input chunk.
        let in_chunk = weight.len().div_ceil(threads);
        let out_chunk = in_chunk * 2;
        let samples = time_best_of_n(|| {
            std::thread::scope(|s| {
                let ins = weight.chunks(in_chunk);
                let outs = out.chunks_mut(out_chunk);
                for (ci, co) in ins.zip(outs) {
                    // PARALLEL: disjoint output slice per thread; each reads shared-
                    // immutable input chunk + sub_table, writes only its own slice.
                    s.spawn(|| fp8_dequant_into(ci, scale, co, &sub_table));
                }
            });
            out[out.len() - 1]
        });
        let median = samples[samples.len() / 2];
        if threads == 1 {
            baseline = median;
        }
        eprintln!(
            "  threads={threads:2}: median {median:6.2} ms  speedup {:.2}×  ({:.0} MB/s BF16)",
            baseline / median,
            (N * 2) as f64 / 1_000_000.0 / (median / 1000.0),
        );
    }
}

/// Synthesizes `n_blocks` of `Q8_0` bytes (34 B/block: `f16 d` + `i8 qs[32]`).
/// `d = f16(1.0)` so the `d * qs` multiplies are not folded away.
#[cfg(feature = "gguf")]
fn build_q8_0_buffer(n_blocks: usize) -> Vec<u8> {
    const BLOCK_BYTES: usize = 34;
    let mut buf = vec![0u8; n_blocks * BLOCK_BYTES];
    for block in buf.as_chunks_mut::<BLOCK_BYTES>().0 {
        block[0] = 0x00;
        block[1] = 0x3C;
    }
    buf
}

/// Synthesizes `n_blocks` of `Q4_0` bytes (18 B/block: `f16 d` + 16 packed nibbles).
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

/// GGUF `Q8_0` — bandwidth-bound (no bit unpacking), Experiment 3 flagged it as
/// the kernel most sensitive to output-write overhead.
#[cfg(feature = "gguf")]
#[test]
#[ignore = "ad-hoc benchmark; run with --release --features gguf --ignored --nocapture"]
fn bench_gguf_q8_0() {
    const N_BLOCKS: usize = WRITER_N / 32;
    const N: usize = N_BLOCKS * 32;
    let data = build_q8_0_buffer(N_BLOCKS);
    eprintln!(
        "\n=== bench_gguf_q8_0 ({N} elems -> {} MB BF16) ===",
        N * 2 / 1_000_000
    );
    let samples = time_best_of_n(|| {
        let out = dequantize_gguf_to_bf16(&data, GgufType::Q8_0, N).unwrap();
        out[out.len() - 1]
    });
    // Traffic ~ read 1 B/elem (i8) + write 2 B/elem = 3 B/elem (scale is amortised).
    report("gguf_q8_0", &samples, N * 3);
}

/// GGUF `Q4_0` — packed-nibble unpack gives it more compute per output byte
/// than `Q8_0`; the contrast with `Q8_0` is the compute-vs-bandwidth signal.
#[cfg(feature = "gguf")]
#[test]
#[ignore = "ad-hoc benchmark; run with --release --features gguf --ignored --nocapture"]
fn bench_gguf_q4_0() {
    const N_BLOCKS: usize = WRITER_N / 32;
    const N: usize = N_BLOCKS * 32;
    let data = build_q4_0_buffer(N_BLOCKS);
    eprintln!(
        "\n=== bench_gguf_q4_0 ({N} elems -> {} MB BF16) ===",
        N * 2 / 1_000_000
    );
    let samples = time_best_of_n(|| {
        let out = dequantize_gguf_to_bf16(&data, GgufType::Q4_0, N).unwrap();
        out[out.len() - 1]
    });
    // Traffic ~ read 0.5 B/elem (packed nibble) + write 2 B/elem = 2.5 B/elem.
    report("gguf_q4_0", &samples, N * 5 / 2);
}

/// A `type_size`-correct synthetic block buffer for any GGUF quant. The kernels
/// are branch-free, so a bounded, varied, non-degenerate fill (finite `f16`
/// scales, moderate grid indices) times identically to a real model's bytes.
#[cfg(feature = "gguf")]
fn build_gguf_buffer(dtype: GgufType, n_blocks: usize) -> Vec<u8> {
    let ts = dtype.type_size().expect("known GGUF type size");
    (0..n_blocks * ts)
        .map(|i| ((((i as u64).wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as u8) & 0x3F) | 0x08)
        .collect()
}

/// Threading scaling of a compute-heavy, gather-bound GGUF kernel (`IQ3_S` — one
/// of the slowest, per the sweep). Compute-dominated kernels have low memory
/// traffic per unit work, so they should scale *past* the ~4-thread bandwidth
/// plateau that caps memory-heavy FP8. Uses the Vec-returning public API
/// (per-thread alloc → a slight *under*-statement, so the true scaling is at
/// least this). Chunks are whole-super-block aligned so each thread's slice is
/// an independent, valid GGUF payload.
#[cfg(feature = "gguf")]
#[test]
#[ignore = "ad-hoc benchmark; run with --release --features gguf --ignored --nocapture"]
fn bench_parallel_iq3s_scaling() {
    let dt = GgufType::IQ3_S;
    let bs = dt.block_size();
    let ts = dt.type_size().unwrap();
    // ~8.4M elements = 32768 super-blocks.
    let total_blocks = (8192 * 1024) / bs;
    let data = build_gguf_buffer(dt, total_blocks);
    eprintln!("\n=== bench_parallel_iq3s_scaling (compute/gather-bound; public Vec API) ===");
    let mut baseline = 0.0_f64;
    for threads in [1usize, 2, 4, 8, 16] {
        let blocks_per = total_blocks.div_ceil(threads);
        let samples = time_best_of_n(|| {
            std::thread::scope(|s| {
                let mut acc = 0u8;
                let handles: Vec<_> = (0..threads)
                    .filter_map(|t| {
                        let b0 = (t * blocks_per).min(total_blocks);
                        let b1 = ((t + 1) * blocks_per).min(total_blocks);
                        if b0 == b1 {
                            return None;
                        }
                        let chunk = &data[b0 * ts..b1 * ts];
                        let n = (b1 - b0) * bs;
                        Some(s.spawn(move || dequantize_gguf_to_bf16(chunk, dt, n).unwrap()))
                    })
                    .collect();
                for h in handles {
                    let v = h.join().unwrap();
                    acc ^= v[v.len() - 1];
                }
                acc
            })
        });
        let median = samples[samples.len() / 2];
        if threads == 1 {
            baseline = median;
        }
        eprintln!(
            "  threads={threads:2}: median {median:6.2} ms  speedup {:.2}×",
            baseline / median
        );
    }
}

/// Cross-kernel ranking sweep at a FIXED element count (so median time ranks
/// kernels directly). Covers the compute-heavy K-quant and IQ families that the
/// roofline never measured — the prime "where else is the time going?" suspects.
/// Reports BF16-output MB/s (same output size for every kernel → comparable).
#[cfg(feature = "gguf")]
#[test]
#[ignore = "ad-hoc benchmark; run with --release --features gguf --ignored --nocapture"]
fn bench_gguf_kernel_sweep() {
    // ~8.4M elements, a multiple of 256 (the K-quant/IQ super-block size).
    const TARGET: usize = 8192 * 1024;
    let kernels = [
        ("Q4_0 ", GgufType::Q4_0),
        ("Q8_0 ", GgufType::Q8_0),
        ("Q2_K ", GgufType::Q2_K),
        ("Q4_K ", GgufType::Q4_K),
        ("Q6_K ", GgufType::Q6_K),
        ("IQ4_XS", GgufType::IQ4_XS),
        ("IQ2_XS", GgufType::IQ2_XS),
        ("IQ3_S ", GgufType::IQ3_S),
        ("IQ1_S ", GgufType::IQ1_S),
        ("TQ2_0 ", GgufType::TQ2_0),
        ("MXFP4 ", GgufType::MXFP4),
    ];
    eprintln!("\n=== bench_gguf_kernel_sweep ({TARGET} elems each; higher MB/s = faster) ===");
    for (name, dt) in kernels {
        let bs = dt.block_size();
        let n_blocks = TARGET / bs;
        let n = n_blocks * bs;
        let data = build_gguf_buffer(dt, n_blocks);
        let samples = time_best_of_n(|| {
            let out = dequantize_gguf_to_bf16(&data, dt, n).unwrap();
            out[out.len() - 1]
        });
        // Same output size (2n) for every kernel → BF16-output MB/s ranks them.
        report(&format!("gguf_{name}"), &samples, n * 2);
    }
}
