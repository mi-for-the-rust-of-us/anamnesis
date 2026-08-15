// SPDX-License-Identifier: MIT OR Apache-2.0

//! Peak-heap regression assertion for `BnB` double-quant dequantisation.
//!
//! Phase 6.5 of the ROADMAP carries this claim on
//! [`dequantize_bnb4_double_quant_to_bf16`]:
//!
//! > "verify no intermediate byte-serialization allocation"
//!
//! The kernel honours that today by computing the recovered absmax
//! values directly as `f32` (`Vec<f32>[num_blocks]`) and passing the
//! `&[f32]` slice into the core dequant loop without ever serialising
//! them to a `Vec<u8>` first — see `src/remember/bnb.rs` lines
//! 386–426. The assertion below detects a regression in two
//! directions:
//!
//! 1. **Byte-serialization drift** — if a future refactor introduces
//!    a `Vec<u8>[num_blocks × 4]` intermediate (e.g., to "share the
//!    `&[u8]` decode path" with the plain `NF4` kernel), the peak
//!    heap gains an extra `num_blocks × 4` bytes that the assertion
//!    catches.
//! 2. **Eager-output drift** — if a future refactor accidentally
//!    allocates two output buffers (e.g., one intermediate and one
//!    final), the peak doubles.
//!
//! Assertion sits behind `#[ignore]`. Run with:
//!
//! ```text
//! cargo test --release --features bnb --test peak_heap_bnb_dq \
//!   -- --ignored --nocapture
//! ```
//!
//! [`dequantize_bnb4_double_quant_to_bf16`]: anamnesis::remember::bnb::dequantize_bnb4_double_quant_to_bf16
//!
//! # Memory
//!
//! Each test allocates `output_size = total_elements × 2` bytes for
//! the `BF16` output, plus the kernel-side `Vec<f32>[num_blocks]`
//! recovered absmax + a `Vec<f32>[block_size]` block scratch. The
//! layer-size variant peaks at ~90 `MiB` resident; the small variant
//! peaks at ~2 `MiB`.

#![cfg(feature = "bnb")]
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

use anamnesis::remember::bnb::dequantize_bnb4_double_quant_to_bf16;

// ---------------------------------------------------------------------------
// Constants (bitsandbytes defaults)
// ---------------------------------------------------------------------------

/// `bitsandbytes` `NF4` block size — 64 elements per absmax block.
const BLOCK_SIZE: usize = 64;

/// `bitsandbytes` double-quant nested block size — 256 absmax bytes
/// per nested-absmax scale.
const NESTED_BLOCK_SIZE: usize = 256;

// ---------------------------------------------------------------------------
// Deterministic synthesis
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

/// Canonical `NF4` codebook (16 entries × `F32` LE = 64 bytes). Same
/// values as `anamnesis::NF4_CODEBOOK`.
fn nf4_codebook_bytes() -> Vec<u8> {
    let codebook: [f32; 16] = [
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
    codebook.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Synthesises a `256`-entry `F32` LE table for the nested codebook.
/// Real `bitsandbytes` nested codebooks are sorted quantiles of the
/// `absmax` distribution; for the heap-peak assertion the bit
/// pattern is irrelevant.
fn nested_codebook_bytes() -> Vec<u8> {
    let mut out = Vec::with_capacity(1024);
    for i in 0..256 {
        // CAST: usize → f32, indices 0..256 fit exactly in f32 mantissa
        let value = i as f32 / 255.0;
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

/// Container for the five byte buffers a `BnB-NF4` double-quant
/// kernel call consumes. Factored into a struct so the `clippy::type_complexity`
/// lint stays satisfied (a 5-tuple return type exceeds the threshold).
struct BnbDqFixture {
    weight: Vec<u8>,
    absmax: Vec<u8>,
    quant_map: Vec<u8>,
    nested_absmax: Vec<u8>,
    nested_quant_map: Vec<u8>,
}

/// Builds a `BnB-NF4` double-quant fixture of `total_elements` elements.
fn synth_bnb_dq_fixture(total_elements: usize) -> BnbDqFixture {
    // Packed 4-bit nibbles: 2 elements per byte.
    let weight = synth_bytes(total_elements / 2);
    // Quantised absmax: one byte per block, indexes into the nested codebook.
    let num_blocks = total_elements / BLOCK_SIZE;
    let absmax = synth_bytes(num_blocks);
    let quant_map = nf4_codebook_bytes();
    // Nested absmax: f32 per nested-block (ceil-div).
    let num_nested_blocks = num_blocks.div_ceil(NESTED_BLOCK_SIZE);
    let nested_absmax: Vec<u8> = (0..num_nested_blocks)
        .flat_map(|_| 1.0_f32.to_le_bytes())
        .collect();
    let nested_quant_map = nested_codebook_bytes();
    BnbDqFixture {
        weight,
        absmax,
        quant_map,
        nested_absmax,
        nested_quant_map,
    }
}

// ---------------------------------------------------------------------------
// Decomposition assertion
// ---------------------------------------------------------------------------

/// `dhat::HeapStats::max_bytes` reports the sum of live allocation
/// sizes at peak — exact, no allocator-internal bookkeeping. On the
/// reference machine the observed scratch matches the documented
/// `Vec<f32>[num_blocks] + Vec<f32>[block_size]` claim **to the
/// byte**; this allows a tight noise-tolerance assertion that catches
/// a single intermediate `Vec<u8>[num_blocks × 4]` regression (the
/// most-likely byte-serialization drift).
///
/// `4 KiB` cross-platform allocator noise tolerance. Tighter than
/// `tests/peak_heap_gptq.rs`'s `K = 5` ratio because the claim under
/// test here ("no intermediate byte-serialization") is precisely a
/// `1×` `scratch_bytes` regression, which a ratio ≥ 2 cannot detect.
const ALLOC_NOISE_TOLERANCE_BYTES: usize = 4 * 1024;

fn assert_peak_heap_within(total_elements: usize, output_bytes: usize, max_bytes: usize) {
    let num_blocks = total_elements / BLOCK_SIZE;
    let expected_scratch = num_blocks * 4 + BLOCK_SIZE * 4;
    let expected_total = output_bytes + expected_scratch;
    let observed_excess = max_bytes.saturating_sub(expected_total);
    assert!(
        observed_excess <= ALLOC_NOISE_TOLERANCE_BYTES,
        "BnB-DQ peak heap {max_bytes} bytes exceeded expected {expected_total} bytes \
         by {observed_excess} bytes (tolerance {ALLOC_NOISE_TOLERANCE_BYTES} bytes). \
         Expected: output={output_bytes} bytes + scratch={expected_scratch} bytes \
         (num_blocks × 4 + block_size × 4). \
         Excess > tolerance suggests an intermediate byte-serialization \
         regression — most likely a Vec<u8>[num_blocks × 4] crept in."
    );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
#[ignore = "dhat peak-heap assertion; run with --ignored --nocapture"]
fn peak_heap_bnb_dq_small_1m_elements() {
    // Held for the whole body, not just the profiler: `dhat` counts every
    // allocation in the process, so a concurrent test synthesising its own
    // fixture would inflate this one's peak even with the profiler serialised.
    let _dhat_guard = dhat_lock();
    let total_elements: usize = 1024 * 1024; // = 1_048_576
    let fixture = synth_bnb_dq_fixture(total_elements);
    let output_bytes = total_elements * 2;

    let _profiler = dhat::Profiler::builder().testing().build();
    let _out = dequantize_bnb4_double_quant_to_bf16(
        &fixture.weight,
        &fixture.absmax,
        &fixture.quant_map,
        &fixture.nested_absmax,
        &fixture.nested_quant_map,
        0.0, // nested_offset: synthetic fixture, never offset-compressed
        total_elements,
        BLOCK_SIZE,
        NESTED_BLOCK_SIZE,
    )
    .expect("bnb dq dequant");
    let stats = dhat::HeapStats::get();

    let num_blocks = total_elements / BLOCK_SIZE;
    eprintln!(
        "BnB-DQ small (1M): peak={} B, output={} B, absmax+scratch={} B \
         (= num_blocks × 4 + block_size × 4 with num_blocks={})",
        stats.max_bytes,
        output_bytes,
        stats.max_bytes.saturating_sub(output_bytes),
        num_blocks,
    );
    assert_peak_heap_within(total_elements, output_bytes, stats.max_bytes);
}

#[test]
#[ignore = "dhat peak-heap assertion; run with --ignored --nocapture"]
fn peak_heap_bnb_dq_layer_45m_elements() {
    // Held for the whole body, not just the profiler: `dhat` counts every
    // allocation in the process, so a concurrent test synthesising its own
    // fixture would inflate this one's peak even with the profiler serialised.
    let _dhat_guard = dhat_lock();
    let total_elements: usize = 4096 * 11008; // = 45_088_768
    let fixture = synth_bnb_dq_fixture(total_elements);
    let output_bytes = total_elements * 2;

    let _profiler = dhat::Profiler::builder().testing().build();
    let _out = dequantize_bnb4_double_quant_to_bf16(
        &fixture.weight,
        &fixture.absmax,
        &fixture.quant_map,
        &fixture.nested_absmax,
        &fixture.nested_quant_map,
        0.0, // nested_offset: synthetic fixture, never offset-compressed
        total_elements,
        BLOCK_SIZE,
        NESTED_BLOCK_SIZE,
    )
    .expect("bnb dq dequant");
    let stats = dhat::HeapStats::get();

    let num_blocks = total_elements / BLOCK_SIZE;
    eprintln!(
        "BnB-DQ layer (45M): peak={} B, output={} B, absmax+scratch={} B \
         (= num_blocks × 4 + block_size × 4 with num_blocks={})",
        stats.max_bytes,
        output_bytes,
        stats.max_bytes.saturating_sub(output_bytes),
        num_blocks,
    );
    assert_peak_heap_within(total_elements, output_bytes, stats.max_bytes);
}
