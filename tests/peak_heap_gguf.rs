// SPDX-License-Identifier: MIT OR Apache-2.0

//! Peak-heap regression assertions for `GGUF` dequantisation, **per output
//! dtype**.
//!
//! Sibling of [`peak_heap_gptq`](peak_heap_gptq.rs),
//! [`peak_heap_awq`](peak_heap_awq.rs) and
//! [`peak_heap_bnb_dq`](peak_heap_bnb_dq.rs). `GGUF` was the one dequant family
//! with no `dhat` coverage at all, which mattered once Phase 7.3 made its output
//! width a caller-chosen parameter: the `# Memory` claim on
//! `dequantize_gguf` is stated in terms of `E::BYTES`, so it has to be
//! verified per output type rather than once for `BF16` and assumed to
//! generalise.
//!
//! # The claim under test
//!
//! `dequantize_gguf::<E>` documents peak heap as the output buffer **and
//! nothing else**:
//!
//! > Allocates a single `Vec<u8>` of length `n_elements × E::BYTES` \[…\] Peak
//! > heap is the output buffer itself.
//!
//! That is a stronger claim than the `GPTQ` / `AWQ` families make. They allow
//! `output_size + O(out_features)` of scratch; `GGUF` allows **no heap scratch
//! at all**, because both kernel runners keep their `[f32; QK]` scratch and
//! their block output buffer on the *stack*. So the ceiling here is the output
//! size plus a small fixed allowance, not a term that grows with the tensor.
//!
//! The streaming entry point `dequantize_gguf_blocks::<E>` claims even less:
//! stack only, no heap allocation in its frame at any output width. Both are
//! asserted below.
//!
//! All assertions sit behind `#[ignore]`. Run with:
//!
//! ```text
//! cargo test --release --features gguf --test peak_heap_gguf \
//!   -- --ignored --nocapture
//! ```
//!
//! # Memory
//!
//! The layer-sized fixture is 4096 × 11008 ≈ 45 M elements of `Q4_K`, which is
//! ~25 `MiB` of input. Peak resident is dominated by the output: ~90 `MiB` at
//! `BF16` and `F16`, ~180 `MiB` at `F32`. The streaming test's peak is the
//! input fixture alone.

#![cfg(feature = "gguf")]
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

use anamnesis::{Bf16Out, F16Out, F32Out, GgufType, OutputElement, dequantize_gguf};

// ---------------------------------------------------------------------------
// Synthesis (deterministic, by index — mirrors the sibling harnesses)
// ---------------------------------------------------------------------------

fn fill_deterministic(buf: &mut [u8]) {
    for (i, b) in buf.iter_mut().enumerate() {
        *b = (i.wrapping_mul(2_654_435_761) & 0xFF) as u8;
    }
}

/// Builds `Q4_K` block bytes for `n_elements`, with sane `f16` scales.
///
/// `Q4_K` is 256-element super-blocks of 144 bytes. The scale and min are
/// written as well-defined `f16` values (1.0 and 0.5) rather than left as
/// arbitrary bit patterns, so the kernel does real arithmetic instead of
/// propagating `NaN`. The peak-heap claim does not depend on the values, but a
/// `NaN`-soup fixture makes any failure much harder to read.
fn synth_q4_k(n_elements: usize) -> Vec<u8> {
    assert!(n_elements.is_multiple_of(256), "Q4_K needs whole blocks");
    let mut buf = vec![0u8; (n_elements / 256) * 144];
    fill_deterministic(&mut buf);
    for block in buf.chunks_exact_mut(144) {
        block[0] = 0x00;
        block[1] = 0x3C; // d    = f16 1.0
        block[2] = 0x00;
        block[3] = 0x38; // dmin = f16 0.5
    }
    buf
}

// ---------------------------------------------------------------------------
// Assertion
// ---------------------------------------------------------------------------

/// Fixed heap allowance above the output buffer, in bytes.
///
/// `GGUF` allocates exactly one `Vec` and keeps every scratch buffer on the
/// stack, so unlike the `GPTQ` / `AWQ` harnesses there is **no per-tensor
/// scratch term** here. The slack covers allocator bookkeeping only, and is
/// deliberately a constant rather than a multiple of any tensor dimension: if a
/// future change introduces heap scratch that scales with the tensor, this
/// assertion must fail rather than absorb it.
const FIXED_SLACK_BYTES: usize = 64 * 1024;

fn assert_peak_is_output_plus_constant<E: OutputElement>(
    label: &str,
    n_elements: usize,
    max_bytes: usize,
) {
    let output_bytes = n_elements * E::BYTES;
    let ceiling = output_bytes + FIXED_SLACK_BYTES;
    let overhead = max_bytes.saturating_sub(output_bytes);
    eprintln!(
        "{label}: peak={max_bytes} B, output={output_bytes} B ({} B/elem), \
         overhead={overhead} B ({:.4} B/elem)",
        E::BYTES,
        overhead as f64 / n_elements as f64,
    );
    assert!(
        max_bytes <= ceiling,
        "{label}: peak heap {max_bytes} B exceeded ceiling {ceiling} B \
         (output={output_bytes} B + {FIXED_SLACK_BYTES} B fixed slack). \
         Overhead of {overhead} B is {:.4} B per element, which is not a \
         constant — heap scratch that scales with the tensor has crept in, \
         and the `# Memory` claim on `dequantize_gguf` no longer holds.",
        overhead as f64 / n_elements as f64,
    );
}

/// Runs the owned-`Vec` entry point under `dhat` and asserts the ceiling.
fn check_owned<E: OutputElement>(label: &str, n_elements: usize) {
    let raw = synth_q4_k(n_elements);

    let _profiler = dhat::Profiler::builder().testing().build();
    let out = dequantize_gguf::<E>(&raw, GgufType::Q4_K, n_elements).expect("gguf dequant");
    let stats = dhat::HeapStats::get();

    assert_eq!(
        out.len(),
        n_elements * E::BYTES,
        "{label}: output width must follow E::BYTES"
    );
    assert_peak_is_output_plus_constant::<E>(label, n_elements, stats.max_bytes);
}

// ---------------------------------------------------------------------------
// Tests — owned Vec, one per output dtype
// ---------------------------------------------------------------------------

/// 4096 × 11008 ≈ 45 M elements, matching the sibling harnesses' layer size.
const LAYER_ELEMENTS: usize = 4096 * 11008;

/// 1024 × 1024 elements, the small variant.
const SMALL_ELEMENTS: usize = 1024 * 1024;

#[test]
#[ignore = "dhat peak-heap assertion; run with --ignored --nocapture"]
fn peak_heap_gguf_bf16_layer() {
    check_owned::<Bf16Out>("GGUF Q4_K layer, BF16", LAYER_ELEMENTS);
}

#[test]
#[ignore = "dhat peak-heap assertion; run with --ignored --nocapture"]
fn peak_heap_gguf_f16_layer() {
    check_owned::<F16Out>("GGUF Q4_K layer, F16", LAYER_ELEMENTS);
}

#[test]
#[ignore = "dhat peak-heap assertion; run with --ignored --nocapture"]
fn peak_heap_gguf_f32_layer() {
    // The one that matters most for this phase: F32 doubles the output, and the
    // claim is that it doubles *exactly*, with no extra scratch appearing.
    check_owned::<F32Out>("GGUF Q4_K layer, F32", LAYER_ELEMENTS);
}

#[test]
#[ignore = "dhat peak-heap assertion; run with --ignored --nocapture"]
fn peak_heap_gguf_bf16_small() {
    check_owned::<Bf16Out>("GGUF Q4_K small, BF16", SMALL_ELEMENTS);
}

#[test]
#[ignore = "dhat peak-heap assertion; run with --ignored --nocapture"]
fn peak_heap_gguf_f32_small() {
    check_owned::<F32Out>("GGUF Q4_K small, F32", SMALL_ELEMENTS);
}

// ---------------------------------------------------------------------------
// Test — streaming entry point allocates nothing
// ---------------------------------------------------------------------------

/// The streaming variant's `# Memory` section claims **stack only**: no heap
/// allocation in its frame, at any output width.
///
/// Asserted by profiling a sink that accumulates a byte count rather than the
/// bytes, so the only heap in the measured region would be the function's own.
/// The ceiling is a small constant independent of `n_elements`, which is the
/// property that distinguishes "streams" from "buffers internally".
#[test]
#[ignore = "dhat peak-heap assertion; run with --ignored --nocapture"]
fn peak_heap_gguf_streaming_allocates_nothing() {
    let n_elements = LAYER_ELEMENTS;
    let raw = synth_q4_k(n_elements);

    for (label, bytes) in [("BF16", Bf16Out::BYTES), ("F32", F32Out::BYTES)] {
        let mut total = 0usize;
        let _profiler = dhat::Profiler::builder().testing().build();
        if bytes == Bf16Out::BYTES {
            anamnesis::dequantize_gguf_blocks::<Bf16Out, _>(
                &raw,
                GgufType::Q4_K,
                n_elements,
                |block| {
                    total += block.len();
                    Ok(())
                },
            )
            .expect("streaming dequant");
        } else {
            anamnesis::dequantize_gguf_blocks::<F32Out, _>(
                &raw,
                GgufType::Q4_K,
                n_elements,
                |block| {
                    total += block.len();
                    Ok(())
                },
            )
            .expect("streaming dequant");
        }
        let stats = dhat::HeapStats::get();

        assert_eq!(
            total,
            n_elements * bytes,
            "{label}: sink saw the wrong total byte count"
        );
        eprintln!(
            "GGUF streaming, {label}: peak={} B for {n_elements} elements \
             ({} B of output streamed)",
            stats.max_bytes, total
        );
        assert!(
            stats.max_bytes <= FIXED_SLACK_BYTES,
            "GGUF streaming, {label}: peak heap {} B exceeds the {FIXED_SLACK_BYTES} B \
             fixed allowance. The streaming entry point claims stack-only \
             operation, so its peak must not scale with the tensor at all.",
            stats.max_bytes,
        );
    }
}
