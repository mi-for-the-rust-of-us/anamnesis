// SPDX-License-Identifier: MIT OR Apache-2.0

//! Timing and synthetic-fixture helpers for the `bench_*_adhoc.rs` harnesses.
//!
//! Only helpers whose copies were identical live here. The harnesses whose
//! variants differ keep them local: `bench_pass2_adhoc.rs` prints three
//! decimals in its `fmt_stats` and runs an anti-dead-code accumulator in its
//! `time_best_of_n`, and those differences are part of what it measures.

use std::time::{Duration, Instant};

/// Median + range of an ascending-sorted `&[f64]`, formatted for stderr.
pub fn fmt_stats(samples: &[f64]) -> String {
    let median = samples[samples.len() / 2];
    let min = samples[0];
    let max = samples[samples.len() - 1];
    format!("median {median:.2} ms (min {min:.2}, max {max:.2})")
}

/// Returns `(min, median, max)` of a slice of `Duration`s. Sorts the input.
pub fn min_median_max(samples: &mut [Duration]) -> (Duration, Duration, Duration) {
    samples.sort_unstable();
    let lo = samples[0];
    let hi = samples[samples.len() - 1];
    let mid = samples[samples.len() / 2];
    (lo, mid, hi)
}

/// Times one closure `iterations` times after `warmup` untimed calls that
/// warm the file cache.
pub fn time_loop<T, F: FnMut() -> T>(warmup: usize, iterations: usize, mut f: F) -> Vec<Duration> {
    for _ in 0..warmup {
        let _ = f();
    }
    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let t = Instant::now();
        let _ = f();
        samples.push(t.elapsed());
    }
    samples
}

/// Synthesizes `n_blocks` of `Q8_0`-formatted bytes (34 bytes per
/// 32-element block: `f16 d` + `i8 qs[32]`). Byte values are arbitrary:
/// the kernel has no data-dependent branches, so timing is identical
/// to a real model's bytes. Using a non-zero `d` ensures the runtime
/// `d × qs[j]` multiplications are not optimised away.
pub fn build_q8_0_buffer(n_blocks: usize) -> Vec<u8> {
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
pub fn build_q4_0_buffer(n_blocks: usize) -> Vec<u8> {
    const BLOCK_BYTES: usize = 18;
    let mut buf = vec![0u8; n_blocks * BLOCK_BYTES];
    for block in buf.as_chunks_mut::<BLOCK_BYTES>().0 {
        block[0] = 0x00;
        block[1] = 0x3C;
    }
    buf
}
