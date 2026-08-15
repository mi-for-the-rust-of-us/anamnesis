// SPDX-License-Identifier: MIT OR Apache-2.0

//! Peak-heap regression assertions for `AWQ` dequantisation.
//!
//! Mirrors [`peak_heap_gptq`](peak_heap_gptq.rs). `AWQ` shares
//! `GPTQ`'s asymptotic claim — Phase 6.5 of the ROADMAP carries the
//! same wording on [`dequantize_awq_to_bf16`](anamnesis::dequantize_awq_to_bf16):
//!
//! > "lazy precomputation keeps peak heap within `output_size +
//! > O(out_features)`, not `output_size + O(num_groups × out_features)`"
//!
//! `AWQ`'s kernel uses the same scratch-buffer shape as `GPTQ`:
//! three `Vec<f32>[out_features]` arrays for unpacked weights,
//! zero-points, and scales — see `src/remember/awq.rs` lines 263–265.
//! The only structural difference is the `qweight` layout: `AWQ` is
//! column-packed (`[in_features, out_features / pack_factor]`) where
//! `GPTQ` is row-packed (`[in_features / pack_factor, out_features]`),
//! but neither layout changes the scratch claim.
//!
//! Both assertions sit behind `#[ignore]`. Run with:
//!
//! ```text
//! cargo test --release --features awq --test peak_heap_awq \
//!   -- --ignored --nocapture
//! ```
//!
//! # Memory
//!
//! Same as [`peak_heap_gptq`](peak_heap_gptq.rs): output bytes
//! dominate; scratch is `3 × out_features × 4` bytes per the kernel
//! contract. The layer-size variant peaks at ~90 `MiB` resident; the
//! small variant peaks at ~2 `MiB`.

#![cfg(feature = "awq")]
#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::wildcard_enum_match_arm
)]

#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

// ---------------------------------------------------------------------------
// dhat serialisation
// ---------------------------------------------------------------------------

/// Serialises the `dhat` profiler across this binary's tests.
///
/// `dhat` installs a global allocator wrapper and permits **one** live
/// `Profiler` per process, but `cargo test` runs test functions on parallel
/// threads by default. Without this guard a second test entering
/// `Profiler::builder().build()` while the first is still live either panics
/// ("optional dhat: only one Profiler can be running at a time") or, worse,
/// silently attributes one test's allocations to another's peak.
///
/// That is not hypothetical. Before v0.7.4 this file held two tests and passed
/// by luck; adding per-dtype cases made it fail with a reported scratch of
/// `137 x out_features x 4` against a true `3 x`. `peak_heap_gguf.rs` had the
/// same latent bug from v0.7.3, where it panicked outright under the default
/// thread count.
///
/// Held for the profiler's whole lifetime: declare this **before** the
/// `Profiler`, so the profiler (declared later) drops first.
fn dhat_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

use anamnesis::{Bf16Out, Dtype, F16Out, F32Out, OutputElement, dequantize_awq};

// ---------------------------------------------------------------------------
// Synthesis (deterministic, by index — mirrors peak_heap_gptq.rs)
// ---------------------------------------------------------------------------

fn fill_deterministic(buf: &mut [u8]) {
    for (i, b) in buf.iter_mut().enumerate() {
        *b = (i.wrapping_mul(2_654_435_761) & 0xFF) as u8;
    }
}

fn synth_bytes(n: usize) -> Vec<u8> {
    let mut v = vec![0u8; n];
    fill_deterministic(&mut v);
    v
}

/// Builds `(qweight, scales, qzeros)` for an `AWQ` `INT4` fixture.
///
/// Shapes:
/// - `qweight`: `[in_features, out_features / 8]` u32 entries
///   (4-bit packed, 8 nibbles per u32, **column-packed** — the
///   key difference vs `GPTQ`).
/// - `scales`: `[num_groups, out_features]` `BF16` LE.
/// - `qzeros`: `[num_groups, out_features / 8]` u32 entries.
///
/// `bits = 4`, `pack_factor = 8`. As with the `GPTQ` test, the
/// assertion only cares about the `O(out_features)` scratch claim,
/// so the actual data values are irrelevant.
fn synth_awq_fixture(
    in_features: usize,
    out_features: usize,
    group_size: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let pack_factor: usize = 8;
    // AWQ column-packed: in_features rows × (out_features / pack_factor) cols × 4 bytes per u32
    let qweight = synth_bytes(in_features * (out_features / pack_factor) * 4);

    let num_groups = in_features / group_size;
    // BF16 LE = 2 bytes per element. `0x3F00` = BF16 0.5 (so dequant
    // produces well-defined non-zero output instead of NaN-soup that
    // an arbitrary bit pattern in `scales` could yield).
    let mut scales = vec![0u8; num_groups * out_features * 2];
    for pair in scales.chunks_exact_mut(2) {
        pair[0] = 0x00;
        pair[1] = 0x3F;
    }

    let qzeros = synth_bytes(num_groups * (out_features / pack_factor) * 4);

    (qweight, scales, qzeros)
}

// ---------------------------------------------------------------------------
// Decomposition assertion (identical shape to peak_heap_gptq.rs)
// ---------------------------------------------------------------------------

/// Computes the ceiling `output_size + K × out_features × 4` and
/// asserts the dhat-observed peak stays below it. `K = 5` covers the
/// three `Vec<f32>` scratch buffers (3 × 4 bytes/element = 12 bytes
/// per element) plus a 2× headroom for allocator overhead.
///
/// On failure, the message decomposes the actual peak so the reader
/// can spot whether the regression is in scratch (still `O(out_features)`
/// but bigger) or in eager-precomputation (now scaling with
/// `num_groups × out_features`).
const K_AWQ_SCRATCH: usize = 5;

fn assert_peak_heap_within(out_features: usize, output_bytes: usize, max_bytes: usize) {
    let expected_ceiling = output_bytes + K_AWQ_SCRATCH * out_features * 4;
    assert!(
        max_bytes <= expected_ceiling,
        "AWQ peak heap {} bytes exceeded ceiling {} bytes \
         (output={} bytes, out_features={}, K=5 × 4-byte scratch slack). \
         Excess of {} bytes suggests {} × out_features regression — \
         likely eager precomputation has crept in.",
        max_bytes,
        expected_ceiling,
        output_bytes,
        out_features,
        max_bytes.saturating_sub(output_bytes),
        max_bytes.saturating_sub(output_bytes) / (out_features * 4),
    );
}

// ---------------------------------------------------------------------------
// Per-dtype driver (v0.7.4)
// ---------------------------------------------------------------------------

/// Runs one `AWQ` dequant at output width `E` under `dhat` and asserts both
/// halves of the contract: the output really is `E::BYTES` per element, and the
/// peak stays within `output + O(out_features)`.
///
/// Parameterised over `E` since v0.7.4, because the `# Memory` claim moved from
/// a fixed `n × 2` to `n × E::BYTES`. Asserting only the `BF16` case would let
/// an `F32` request silently break the ceiling that the `# Memory` section
/// promises `src/lib.rs` verifies to the byte.
fn check_awq<E: OutputElement>(
    label: &str,
    in_features: usize,
    out_features: usize,
    group_size: usize,
) {
    // Held for the whole body, not just the profiler: `dhat` counts every
    // allocation in the process, so a concurrent test synthesising its own
    // fixture would inflate this one's peak even with the profiler serialised.
    let _dhat_guard = dhat_lock();
    let (qweight, scales, qzeros) = synth_awq_fixture(in_features, out_features, group_size);
    let output_bytes = in_features * out_features * E::BYTES;

    let _profiler = dhat::Profiler::builder().testing().build();
    let out = dequantize_awq::<E>(
        &qweight,
        &scales,
        &qzeros,
        in_features,
        out_features,
        group_size,
        4,
        Dtype::BF16,
    )
    .expect("awq dequant");
    let stats = dhat::HeapStats::get();

    assert_eq!(
        out.len(),
        output_bytes,
        "{label}: output width must follow E::BYTES"
    );
    eprintln!(
        "AWQ {label}: peak={} B, output={} B ({} B/elem), scratch={} B (= {} × out_features × 4)",
        stats.max_bytes,
        output_bytes,
        E::BYTES,
        stats.max_bytes.saturating_sub(output_bytes),
        stats.max_bytes.saturating_sub(output_bytes) / (out_features * 4),
    );
    assert_peak_heap_within(out_features, output_bytes, stats.max_bytes);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
#[ignore = "dhat peak-heap assertion; run with --ignored --nocapture"]
fn peak_heap_awq_small_1m_elements() {
    // 1024 × 1024 = 1_048_576 elements; output ≈ 2 MiB at BF16; scratch =
    // 3 × 1024 × 4 = 12 KiB, independent of the output width.
    check_awq::<Bf16Out>("small_1m_bf16", 1024, 1024, 128);
}

#[test]
#[ignore = "dhat peak-heap assertion; run with --ignored --nocapture"]
fn peak_heap_awq_small_1m_elements_f32() {
    // Same fixture at 4 bytes per element: output doubles, scratch must not.
    check_awq::<F32Out>("small_1m_f32", 1024, 1024, 128);
}

#[test]
#[ignore = "dhat peak-heap assertion; run with --ignored --nocapture"]
fn peak_heap_awq_small_1m_elements_f16() {
    check_awq::<F16Out>("small_1m_f16", 1024, 1024, 128);
}

#[test]
#[ignore = "dhat peak-heap assertion; run with --ignored --nocapture"]
fn peak_heap_awq_layer_45m_elements_f32() {
    // 4096 × 11008 ≈ 45M elements; output ≈ 180 MiB at F32.
    check_awq::<F32Out>("layer_45m_f32", 4096, 11008, 128);
}

#[test]
#[ignore = "dhat peak-heap assertion; run with --ignored --nocapture"]
fn peak_heap_awq_layer_45m_elements() {
    // Held for the whole body, not just the profiler: `dhat` counts every
    // allocation in the process, so a concurrent test synthesising its own
    // fixture would inflate this one's peak even with the profiler serialised.
    let _dhat_guard = dhat_lock();
    // 4096 × 11008 ≈ 45M elements; output ≈ 90 MiB; scratch =
    // 3 × 11008 × 4 = 130 KiB.
    let in_features: usize = 4096;
    let out_features: usize = 11008;
    let group_size: usize = 128;

    let (qweight, scales, qzeros) = synth_awq_fixture(in_features, out_features, group_size);
    let output_bytes = in_features * out_features * 2;

    let _profiler = dhat::Profiler::builder().testing().build();
    let _out = dequantize_awq::<Bf16Out>(
        &qweight,
        &scales,
        &qzeros,
        in_features,
        out_features,
        group_size,
        4,
        Dtype::BF16,
    )
    .expect("awq dequant");
    let stats = dhat::HeapStats::get();

    eprintln!(
        "AWQ layer (45M): peak={} B, output={} B, scratch={} B (= {} × out_features × 4)",
        stats.max_bytes,
        output_bytes,
        stats.max_bytes.saturating_sub(output_bytes),
        stats.max_bytes.saturating_sub(output_bytes) / (out_features * 4),
    );
    assert_peak_heap_within(out_features, output_bytes, stats.max_bytes);
}
