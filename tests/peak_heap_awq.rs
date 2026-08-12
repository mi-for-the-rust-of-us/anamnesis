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

use anamnesis::{Dtype, dequantize_awq_to_bf16};

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
// Tests
// ---------------------------------------------------------------------------

#[test]
#[ignore = "dhat peak-heap assertion; run with --ignored --nocapture"]
fn peak_heap_awq_small_1m_elements() {
    // 1024 × 1024 = 1_048_576 elements; output ≈ 2 MiB; scratch =
    // 3 × 1024 × 4 = 12 KiB.
    let in_features: usize = 1024;
    let out_features: usize = 1024;
    let group_size: usize = 128;

    let (qweight, scales, qzeros) = synth_awq_fixture(in_features, out_features, group_size);
    let output_bytes = in_features * out_features * 2;

    let _profiler = dhat::Profiler::builder().testing().build();
    let _out = dequantize_awq_to_bf16(
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
        "AWQ small (1M): peak={} B, output={} B, scratch={} B (= {} × out_features × 4)",
        stats.max_bytes,
        output_bytes,
        stats.max_bytes.saturating_sub(output_bytes),
        stats.max_bytes.saturating_sub(output_bytes) / (out_features * 4),
    );
    assert_peak_heap_within(out_features, output_bytes, stats.max_bytes);
}

#[test]
#[ignore = "dhat peak-heap assertion; run with --ignored --nocapture"]
fn peak_heap_awq_layer_45m_elements() {
    // 4096 × 11008 ≈ 45M elements; output ≈ 90 MiB; scratch =
    // 3 × 11008 × 4 = 130 KiB.
    let in_features: usize = 4096;
    let out_features: usize = 11008;
    let group_size: usize = 128;

    let (qweight, scales, qzeros) = synth_awq_fixture(in_features, out_features, group_size);
    let output_bytes = in_features * out_features * 2;

    let _profiler = dhat::Profiler::builder().testing().build();
    let _out = dequantize_awq_to_bf16(
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
