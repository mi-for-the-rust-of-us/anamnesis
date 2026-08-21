// SPDX-License-Identifier: MIT OR Apache-2.0

//! Peak-heap regression assertions for `GPTQ` dequantisation.
//!
//! Phase 6.5 of the ROADMAP carries this claim on
//! [`dequantize_gptq_to_bf16`](anamnesis::dequantize_gptq_to_bf16):
//!
//! > "lazy precomputation keeps peak heap within `output_size +
//! > O(out_features)`, not `output_size + O(num_groups × out_features)`"
//!
//! The kernel honours that today by allocating three `Vec<f32>` scratch
//! buffers each of size `out_features` (for unpacked weights, zeros,
//! and scales) and refilling them lazily only when the cached group
//! changes — see `src/remember/gptq.rs` lines 343–407. The assertion
//! below detects regressions in two directions:
//!
//! 1. **Eager precomputation drift** — if a future refactor builds
//!    per-group state up front (turning `O(out_features)` into
//!    `O(num_groups × out_features)`), the peak heap blows past the
//!    ceiling.
//! 2. **Forgot-to-reuse drift** — if scratch buffers are reallocated
//!    per-row instead of reused, the steady-state heap is unchanged
//!    but the peak briefly doubles. Slack `K` accounts for this.
//!
//! Both assertions sit behind `#[ignore]` so default `cargo test` runs
//! skip them — `dhat::Alloc` wraps the global allocator for the entire
//! binary, so the test binary boots a tiny bit slower than other test
//! binaries, but no heap-tracking happens until a `dhat::Profiler` is
//! created. Run with:
//!
//! ```text
//! cargo test --release --features gptq --test peak_heap_gptq \
//!   -- --ignored --nocapture
//! ```
//!
//! # Memory
//!
//! Each test allocates `output_size` bytes (`in_features × out_features
//! × 2`) for the `BF16` output, plus the kernel-side scratch the
//! assertion is calibrated to detect. The layer-size variant peaks at
//! ~90 `MiB` resident; the small-fixture variant peaks at ~2 `MiB`.

#![cfg(feature = "gptq")]
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

use anamnesis::{Dtype, dequantize_gptq_to_bf16};

// ---------------------------------------------------------------------------
// Synthesis (deterministic, by index)
// ---------------------------------------------------------------------------

/// Fills a byte buffer with a deterministic Knuth-multiplicative-hash
/// pattern. Reused across fixture sizes so reruns produce the same
/// peak numbers.
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

/// Builds `(qweight, scales, qzeros)` for a `GPTQ INT4` fixture with
/// `g_idx = None` (no activation reordering — the simpler path).
///
/// Shapes:
/// - `qweight`: `[in_features / 8, out_features]` u32 entries (4-bit
///   packed, 8 nibbles per u32, row-packed).
/// - `scales`: `[num_groups, out_features]` `BF16` LE.
/// - `qzeros`: `[num_groups, out_features / 8]` u32 entries.
///
/// `bits = 4` and `pack_factor = 8` are hard-coded since the assertion
/// here cares only about the `O(out_features)` scratch invariant, not
/// per-bit-width tuning.
fn synth_gptq_fixture(
    in_features: usize,
    out_features: usize,
    group_size: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let pack_factor: usize = 8;
    let packed_rows = in_features / pack_factor;
    let qweight = synth_bytes(packed_rows * out_features * 4);

    let num_groups = in_features / group_size;
    // BF16 LE = 2 bytes per element. `0x3F00` = BF16 0.5 (so dequant
    // produces well-defined non-zero output instead of NaN-soup that
    // an arbitrary bit pattern in `scales` could yield).
    let mut scales = vec![0u8; num_groups * out_features * 2];
    for pair in scales.as_chunks_mut::<2>().0 {
        pair[0] = 0x00;
        pair[1] = 0x3F;
    }

    let qzeros = synth_bytes(num_groups * (out_features / pack_factor) * 4);

    (qweight, scales, qzeros)
}

// ---------------------------------------------------------------------------
// Decomposition assertion
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
const K_GPTQ_SCRATCH: usize = 5;

fn assert_peak_heap_within(out_features: usize, output_bytes: usize, max_bytes: usize) {
    let expected_ceiling = output_bytes + K_GPTQ_SCRATCH * out_features * 4;
    assert!(
        max_bytes <= expected_ceiling,
        "GPTQ peak heap {} bytes exceeded ceiling {} bytes \
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
// Tests
// ---------------------------------------------------------------------------

#[test]
#[ignore = "dhat peak-heap assertion; run with --ignored --nocapture"]
fn peak_heap_gptq_small_1m_elements() {
    // Held for the whole body, not just the profiler: `dhat` counts every
    // allocation in the process, so a concurrent test synthesising its own
    // fixture would inflate this one's peak even with the profiler serialised.
    let _dhat_guard = dhat_lock();
    // 1024 × 1024 = 1_048_576 elements; output ≈ 2 MiB; scratch =
    // 3 × 1024 × 4 = 12 KiB.
    let in_features: usize = 1024;
    let out_features: usize = 1024;
    let group_size: usize = 128;

    let (qweight, scales, qzeros) = synth_gptq_fixture(in_features, out_features, group_size);
    let output_bytes = in_features * out_features * 2;

    let _profiler = dhat::Profiler::builder().testing().build();
    let _out = dequantize_gptq_to_bf16(
        &qweight,
        &scales,
        &qzeros,
        None,
        in_features,
        out_features,
        group_size,
        4,
        Dtype::BF16,
    )
    .expect("gptq dequant");
    let stats = dhat::HeapStats::get();

    eprintln!(
        "GPTQ small (1M): peak={} B, output={} B, scratch={} B (= {} × out_features × 4)",
        stats.max_bytes,
        output_bytes,
        stats.max_bytes.saturating_sub(output_bytes),
        stats.max_bytes.saturating_sub(output_bytes) / (out_features * 4),
    );
    assert_peak_heap_within(out_features, output_bytes, stats.max_bytes);
}

#[test]
#[ignore = "dhat peak-heap assertion; run with --ignored --nocapture"]
fn peak_heap_gptq_layer_45m_elements() {
    // Held for the whole body, not just the profiler: `dhat` counts every
    // allocation in the process, so a concurrent test synthesising its own
    // fixture would inflate this one's peak even with the profiler serialised.
    let _dhat_guard = dhat_lock();
    // 4096 × 11008 ≈ 45M elements; output ≈ 90 MiB; scratch =
    // 3 × 11008 × 4 = 130 KiB.
    let in_features: usize = 4096;
    let out_features: usize = 11008;
    let group_size: usize = 128;

    let (qweight, scales, qzeros) = synth_gptq_fixture(in_features, out_features, group_size);
    let output_bytes = in_features * out_features * 2;

    let _profiler = dhat::Profiler::builder().testing().build();
    let _out = dequantize_gptq_to_bf16(
        &qweight,
        &scales,
        &qzeros,
        None,
        in_features,
        out_features,
        group_size,
        4,
        Dtype::BF16,
    )
    .expect("gptq dequant");
    let stats = dhat::HeapStats::get();

    eprintln!(
        "GPTQ layer (45M): peak={} B, output={} B, scratch={} B (= {} × out_features × 4)",
        stats.max_bytes,
        output_bytes,
        stats.max_bytes.saturating_sub(output_bytes),
        stats.max_bytes.saturating_sub(output_bytes) / (out_features * 4),
    );
    assert_peak_heap_within(out_features, output_bytes, stats.max_bytes);
}
