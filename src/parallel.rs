// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared per-item parallel dispatch for the compute-heavy transform paths.
//!
//! Phase 7 (v0.7.0) parallelised `ParsedModel::dequantize_all` with an inline
//! `std::thread::scope` block that split the tensor list into contiguous,
//! **equal-count** chunks. Phase 7.2 brings the `GGUF` reader
//! (`convert::read_gguf`) under the same dispatch, and equal-count chunking does
//! not transfer: `GGUF` tensor sizes are far more skewed than a safetensors
//! shard's — a single `token_embd.weight` can outweigh two hundred norm tensors
//! — so a static split by *count* leaves one worker holding most of the *bytes*
//! and the pool idles waiting for it.
//!
//! This module replaces the inline block with one helper, [`map_indexed`], used
//! by both paths, so the crate has a single site to reason about (and a single
//! `// PARALLEL:` annotation to keep honest).
//!
//! # Design
//!
//! - **Partition — an atomic cursor, not a static split.** Each worker claims
//!   the next unclaimed item with a single `fetch_add`, so a worker that draws a
//!   300 `MiB` embedding table simply claims fewer items than one drawing 1 `KiB`
//!   norms. The pool self-balances without knowing item sizes up front.
//! - **The atomic is scheduling state, never output state.** One `fetch_add` per
//!   *item* — never inside a kernel loop — and it is the only shared mutable
//!   datum. Every `f` call allocates and writes only its own result, so the
//!   outputs stay strictly disjoint, exactly as `CONVENTIONS.md` "When
//!   Parallelizing Work" requires.
//! - **Determinism.** Results are index-tagged and sorted before returning, so
//!   [`map_indexed`] yields items in input order for **any** thread count and
//!   **any** steal order. Thread count is a performance knob, never a
//!   correctness variable.
//! - **Deterministic errors too** — see [`map_indexed`]'s "Error selection"
//!   section: a failing input reports the *same* error sequentially and in
//!   parallel.
//!
//! The thread budget itself is resolved by
//! [`resolve_thread_budget`](crate::model::resolve_thread_budget) — caller-owned
//! and hardware-bounded, **never** derived from a file-declared quantity.

// Only the parallel half raises errors of its own (a panicked worker); the
// sequential path merely propagates whatever `f` returned.
#[cfg(feature = "parallel")]
use crate::AnamnesisError;

/// Total input-byte floor below which [`map_indexed`] stays sequential.
///
/// `CONVENTIONS.md` "When Parallelizing Work" rule 3 requires a **named**
/// size threshold: below it, spawn/join cost dominates and the sequential path
/// must run instead. v0.7.0's inline dispatch only checked the item *count*
/// (`> 1`), which let an eight-tensor toy model pay four thread spawns to
/// dequantise a few `KiB`.
///
/// **Calibration** — measured, not estimated (`docs/perf-experiments.md`
/// Experiment 12; 5950X, Windows 11, `target-cpu=native`, best-of-5):
///
/// - Spawning **and** joining a 4-worker `std::thread::scope` pool costs
///   **236 µs** (2 workers 119 µs, 8 workers 329 µs, 16 workers 655 µs).
///   Windows thread creation is expensive — this is ~2× what a Linux figure
///   would suggest, which is precisely why the threshold is measured per
///   platform-of-record rather than assumed.
/// - The `GGUF` hub read sustains **~1.05 `GB/s`** of input on one thread and
///   scales **~1.9×** at four, so the parallel path breaks even at roughly
///   **0.5 `MiB`** of input (`work × (1 − 1/1.9) > 236 µs`).
///
/// 4 `MiB` is the break-even with an ~8× margin: the pool costs under ~6 % of
/// the work there, and the dispatch still nets ~1.7× at exactly the threshold
/// while a mis-set value can only cost a few hundred microseconds. The
/// threshold counts **input** bytes (the quantised blocks read), not output
/// bytes, because that is the quantity both callers already know without a
/// second pass.
pub(crate) const MIN_PARALLEL_BYTES: u64 = 4 * 1024 * 1024;

/// Maps `f` over `items`, in parallel when it is worth it, returning the results
/// **in `items` order regardless of thread count**.
///
/// `f` receives `(index, &item)` and must be a pure function of its arguments
/// plus shared-**immutable** captured state: it allocates and writes only the
/// value it returns. `on_result` fires on the **calling** thread only, once per
/// successfully produced result, so a caller can drive a progress bar without
/// its `FnMut` ever crossing a thread boundary.
///
/// Runs **sequentially** — a plain in-order loop, no threads spawned — when any
/// of the following holds: the `parallel` Cargo feature is off, `threads <= 1`,
/// there is at most one item, or `work_bytes` is below
/// [`MIN_PARALLEL_BYTES`]. Otherwise `min(threads, items.len())` scoped workers
/// draw items from a shared atomic cursor until the list is exhausted.
///
/// # Errors
///
/// Propagates whatever `f` returns, and returns [`AnamnesisError::Parse`] if a
/// worker thread panics (which the crate's `panic`-denying lint floor makes
/// unreachable in practice — it is a fail-closed backstop, not an expected path).
///
/// ## Error selection is deterministic
///
/// The sequential path stops at the **first** failing item, so it reports the
/// error of the lowest failing index. The parallel path reproduces that exactly:
/// each worker stops claiming new items after its own first failure and reports
/// `(index, error)`, and the caller keeps the failure with the smallest index.
///
/// That yields the true minimum, not merely a minimum of what happened to run.
/// Let `m` be the lowest failing index. The cursor is monotone, so `m` is only
/// handed out after every index below it has been handed out — and every index
/// below `m` succeeds, so no worker can have stopped before `m` is claimed.
/// Whichever worker claims `m` has therefore seen no prior failure, processes
/// `m`, and reports it. A malformed input thus produces the same error message
/// at 1 thread and at 16.
///
/// # Memory
///
/// Allocates one `Vec` of `items.len()` results plus, on the parallel path, one
/// index tag per result (`usize`) and a per-worker staging `Vec` that is drained
/// into the shared collection at join time. No copy of the *input* is made — the
/// workers read `items` through a shared reference.
pub(crate) fn map_indexed<T, R, F, P>(
    items: &[T],
    threads: usize,
    work_bytes: u64,
    f: F,
    mut on_result: P,
) -> crate::Result<Vec<R>>
where
    T: Sync,
    R: Send,
    F: Fn(usize, &T) -> crate::Result<R> + Sync,
    P: FnMut(&R),
{
    // The three gates, stated once so both builds read the same rule: a budget
    // above one, more than one item, and enough work to amortise the pool.
    let worth_spawning = threads > 1 && items.len() > 1 && work_bytes >= MIN_PARALLEL_BYTES;

    #[cfg(feature = "parallel")]
    if worth_spawning {
        return map_indexed_parallel(items, threads, &f, &mut on_result);
    }

    // With the `parallel` feature off the verdict is computed and discarded —
    // the dispatch is unconditionally sequential.
    #[cfg(not(feature = "parallel"))]
    let _ = worth_spawning;

    let mut out = Vec::with_capacity(items.len());
    for (idx, item) in items.iter().enumerate() {
        let result = f(idx, item)?;
        on_result(&result);
        out.push(result);
    }
    Ok(out)
}

/// The parallel half of [`map_indexed`], split out so the sequential build
/// compiles without any of the threading machinery.
///
/// `f` and `on_result` arrive by reference because the scope's worker closures
/// capture the former (shared) while the join loop drives the latter (unique) on
/// the calling thread.
#[cfg(feature = "parallel")]
fn map_indexed_parallel<T, R, F, P>(
    items: &[T],
    threads: usize,
    f: &F,
    on_result: &mut P,
) -> crate::Result<Vec<R>>
where
    T: Sync,
    R: Send,
    F: Fn(usize, &T) -> crate::Result<R> + Sync,
    P: FnMut(&R),
{
    use std::sync::atomic::{AtomicUsize, Ordering};

    let n_workers = threads.min(items.len());
    let cursor = AtomicUsize::new(0);

    let mut collected: Vec<(usize, R)> = Vec::with_capacity(items.len());
    // The lowest-indexed failure seen across all workers; see the "Error
    // selection is deterministic" section on `map_indexed`.
    let mut failure: Option<(usize, AnamnesisError)> = None;
    let mut panicked = false;

    // PARALLEL: `items` is claimed one entry at a time through a shared
    // `AtomicUsize` cursor — a self-balancing partition, since a worker that
    // draws a large tensor simply claims fewer entries. The cursor is the only
    // shared mutable state and is touched once per *item*, never inside a
    // kernel; `f` reads shared-immutable input and writes only the value it
    // returns, so the output regions are disjoint by construction and there is
    // no reduction. Results carry their input index and are sorted below, so the
    // returned order — and therefore every byte the callers serialise — is
    // identical for any thread count and any steal order. `on_result` runs only
    // on this thread.
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..n_workers)
            .map(|_| {
                let cursor = &cursor;
                scope.spawn(move || {
                    let mut local: Vec<(usize, R)> = Vec::new();
                    loop {
                        let idx = cursor.fetch_add(1, Ordering::Relaxed);
                        let Some(item) = items.get(idx) else { break };
                        match f(idx, item) {
                            Ok(result) => local.push((idx, result)),
                            // Stop claiming after this worker's own first
                            // failure; the remaining entries fall to the other
                            // workers, which keeps the minimum-index argument
                            // above valid.
                            Err(err) => return Err((idx, err)),
                        }
                    }
                    Ok(local)
                })
            })
            .collect();

        // Join in spawn order on the calling thread. Every handle is joined even
        // once a failure is known: the deterministic-error rule needs the
        // *minimum* failing index, not the first one observed.
        for handle in handles {
            match handle.join() {
                Ok(Ok(local)) => {
                    for (idx, result) in local {
                        on_result(&result);
                        collected.push((idx, result));
                    }
                }
                Ok(Err((idx, err))) => {
                    if failure.as_ref().is_none_or(|&(seen, _)| idx < seen) {
                        failure = Some((idx, err));
                    }
                }
                Err(_) => panicked = true,
            }
        }
    });

    if panicked {
        return Err(AnamnesisError::Parse {
            reason: "parallel dequant worker thread panicked".into(),
        });
    }
    if let Some((_, err)) = failure {
        return Err(err);
    }

    // Determinism: reassemble in input order regardless of how the cursor
    // happened to distribute the work.
    collected.sort_by_key(|&(idx, _)| idx);
    Ok(collected.into_iter().map(|(_, result)| result).collect())
}

#[cfg(test)]
#[allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing
)]
mod tests {
    use super::{map_indexed, MIN_PARALLEL_BYTES};
    use crate::AnamnesisError;

    /// A work-byte figure comfortably above [`MIN_PARALLEL_BYTES`], so the
    /// parallel path is actually taken when the feature is on.
    const BIG: u64 = MIN_PARALLEL_BYTES * 2;

    /// Order is an input-order property, not a completion-order one: the same
    /// sequence must come back for every thread budget.
    #[test]
    fn results_are_in_input_order_for_every_budget() {
        let items: Vec<usize> = (0..97).collect();
        let baseline =
            map_indexed(&items, 1, BIG, |_, &v| Ok(v * 3), |_| {}).expect("sequential map");
        assert_eq!(baseline, items.iter().map(|v| v * 3).collect::<Vec<_>>());

        for threads in [1usize, 2, 4, 8, 16] {
            let out =
                map_indexed(&items, threads, BIG, |_, &v| Ok(v * 3), |_| {}).expect("parallel map");
            assert_eq!(out, baseline, "order must not depend on thread count");
        }
    }

    /// The index handed to `f` is the item's position, not a per-worker counter.
    #[test]
    fn closure_receives_the_input_index() {
        let items: Vec<usize> = (0..64).map(|i| i * 10).collect();
        let out = map_indexed(&items, 8, BIG, |idx, &v| Ok((idx, v)), |_| {}).expect("map");
        for (expected, &(idx, v)) in out.iter().enumerate() {
            assert_eq!(idx, expected);
            assert_eq!(v, expected * 10);
        }
    }

    /// `on_result` fires exactly once per produced result, on the calling
    /// thread — the property that lets callers drive a progress bar with a
    /// plain `FnMut`.
    #[test]
    fn on_result_fires_once_per_item() {
        let items: Vec<usize> = (0..50).collect();
        for threads in [1usize, 4] {
            let mut seen = 0usize;
            let out = map_indexed(&items, threads, BIG, |_, &v| Ok(v), |_| seen += 1)
                .expect("map with callback");
            assert_eq!(seen, out.len());
            assert_eq!(seen, items.len());
        }
    }

    /// A malformed input must produce the *same* error at 1 thread and at 16 —
    /// the lowest failing index wins, never whichever worker happened to fail
    /// first.
    #[test]
    fn error_selection_is_deterministic_across_budgets() {
        let items: Vec<usize> = (0..128).collect();
        // Three failing entries; index 37 is the lowest.
        let failing = |_: usize, &v: &usize| -> crate::Result<usize> {
            if v == 37 || v == 80 || v == 127 {
                Err(AnamnesisError::Parse {
                    reason: format!("item {v} rejected"),
                })
            } else {
                Ok(v)
            }
        };

        for threads in [1usize, 2, 4, 8, 16] {
            let err =
                map_indexed(&items, threads, BIG, failing, |_| {}).expect_err("the map must fail");
            assert_eq!(
                err.to_string(),
                AnamnesisError::Parse {
                    reason: "item 37 rejected".into(),
                }
                .to_string(),
                "the lowest failing index must win at {threads} threads"
            );
        }
    }

    /// Empty and single-item inputs take the sequential path and must still
    /// behave.
    #[test]
    fn degenerate_inputs() {
        let empty: Vec<usize> = Vec::new();
        let out = map_indexed(&empty, 8, BIG, |_, &v| Ok(v), |_| {}).expect("empty map");
        assert!(out.is_empty());

        let one = vec![42usize];
        let out = map_indexed(&one, 8, BIG, |_, &v| Ok(v + 1), |_| {}).expect("single map");
        assert_eq!(out, vec![43]);
    }

    /// Work below [`MIN_PARALLEL_BYTES`] must not spawn: the result is
    /// identical either way, so the observable contract is just that the output
    /// still matches the sequential baseline.
    #[test]
    fn below_the_size_threshold_still_maps_correctly() {
        let items: Vec<usize> = (0..40).collect();
        let out = map_indexed(&items, 8, 1024, |_, &v| Ok(v * 2), |_| {}).expect("small map");
        assert_eq!(out, items.iter().map(|v| v * 2).collect::<Vec<_>>());
    }
}
