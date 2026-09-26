// SPDX-License-Identifier: MIT OR Apache-2.0

//! Helpers for the `peak_heap_*.rs` binaries.
//!
//! Those binaries install `dhat::Alloc` as their global allocator and measure
//! the heap inside a `dhat::Profiler` scope. Nothing here allocates except
//! [`synth_bytes`], which every caller runs **before** the profiler starts, so
//! moving these helpers out of the binaries leaves every measured region
//! unchanged.

/// Serialises the `dhat` profiler across one binary's tests.
///
/// `dhat` installs a global allocator wrapper and permits **one** live
/// `Profiler` per process, but `cargo test` runs test functions on parallel
/// threads by default. Without this guard a second test entering
/// `Profiler::builder().build()` while the first is still live either panics
/// ("optional dhat: only one Profiler can be running at a time") or, worse,
/// silently attributes one test's allocations to another's peak.
///
/// That is not hypothetical. Before v0.7.4 `peak_heap_awq.rs` held two tests
/// and passed by luck; adding per-dtype cases made it fail with a reported
/// scratch of `137 x out_features x 4` against a true `3 x`.
/// `peak_heap_gguf.rs` had the same latent bug from v0.7.3, where it panicked
/// outright under the default thread count.
///
/// The `static` lives inside this function, and each `tests/*.rs` binary
/// compiles its own copy of this module, so every binary still gets its own
/// lock (binaries are separate processes; there is nothing to share).
///
/// Held for the profiler's whole lifetime: declare this **before** the
/// `Profiler`, so the profiler (declared later) drops first.
pub fn dhat_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Fills a byte buffer with a deterministic Knuth-multiplicative-hash
/// pattern. Reused across fixture sizes so reruns produce the same
/// peak numbers.
pub fn fill_deterministic(buf: &mut [u8]) {
    for (i, b) in buf.iter_mut().enumerate() {
        // CAST: usize → u8, masked to the low byte first
        *b = (i.wrapping_mul(2_654_435_761) & 0xFF) as u8;
    }
}

/// Allocates `n` bytes filled by [`fill_deterministic`].
pub fn synth_bytes(n: usize) -> Vec<u8> {
    let mut v = vec![0u8; n];
    fill_deterministic(&mut v);
    v
}
