// SPDX-License-Identifier: MIT OR Apache-2.0

//! Cooperative cancellation for the long-running transform paths.
//!
//! # Why a token rather than a callback
//!
//! `remember` and `convert` already accept a progress callback, and the obvious
//! way to add cancellation would be to let that callback say *stop*. It does not
//! work here, and the reason is worth stating so it is not re-proposed: on the
//! parallel path the progress hook fires on the **calling** thread as each
//! worker joins, which is deliberate (an `FnMut` closing over a progress bar
//! never crosses a thread boundary) and which means the hook is not called at
//! all while work is in flight. A callback-based stop could only take effect
//! after the work it was meant to stop had finished.
//!
//! A [`CancelToken`] is read by the workers themselves, once per tensor at the
//! point the scheduling cursor hands one out. That is the same
//! once-per-item, never-inside-a-kernel discipline `CONVENTIONS.md` already
//! sanctions for the cursor, and it is why cancellation is prompt even though
//! progress is coarse.
//!
//! # The `PyO3` shape this exists for
//!
//! [Phase 8](https://github.com/mi-for-the-rust-of-us/anamnesis/blob/main/ROADMAP.md)
//! releases the `GIL` around a large `parse` / `remember` / `convert`. A
//! released `GIL` means `KeyboardInterrupt` is not delivered until Rust
//! returns, so without something like this a notebook user cannot `Ctrl-C` a
//! multi-minute conversion and a web worker cannot honour a request timeout.
//! The binding runs the work on a spawned thread, polls
//! `Python::check_signals()` on the main thread, and calls
//! [`CancelToken::cancel`] when a signal arrives.
//!
//! The token is `Clone + Send + Sync`, so the same handle can be held by the
//! caller, the workers, and a watchdog thread at once.
//!
//! # What cancellation guarantees
//!
//! A cancelled run returns [`AnamnesisError::Cancelled`](crate::AnamnesisError::Cancelled)
//! and **writes no output file**: every path builds its complete result in
//! memory before serialising, so the check happens strictly before any byte
//! reaches the filesystem. A cancelled run therefore leaves nothing to clean
//! up. It also allocates no output file to truncate — this is not a "delete
//! the partial file" story, it is a "never create one" story.
//!
//! Cancellation is **cooperative and not instantaneous**: a worker already
//! inside a tensor finishes that tensor. The bound is one tensor's
//! dequantisation, not the whole model.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// A shared, clonable flag that asks an in-flight `remember` or `convert` to
/// stop at the next tensor boundary.
///
/// Cheap to clone (one `Arc` bump) and cheap to poll (one relaxed atomic load).
/// Attach one to [`RememberOptions`](crate::RememberOptions) or
/// [`ConvertOptions`](crate::ConvertOptions), keep a clone, and call
/// [`cancel`](Self::cancel) from any thread.
///
/// ```rust
/// use anamnesis::{CancelToken, RememberOptions};
///
/// let token = CancelToken::new();
/// let opts = RememberOptions::new().with_cancel(token.clone());
///
/// // From a signal handler, a watchdog thread, or a timeout:
/// token.cancel();
/// assert!(token.is_cancelled());
/// # let _ = opts;
/// ```
///
/// Cancellation is one-way: there is no `reset`. A token that has been
/// cancelled stays cancelled, so a handle cannot be reused for a second run and
/// silently fail to stop it.
#[derive(Debug, Clone, Default)]
pub struct CancelToken {
    /// Set once by [`cancel`](CancelToken::cancel); read once per work item by
    /// the dispatch. `Relaxed` ordering is sufficient: the flag carries no data
    /// and publishes nothing, so there is nothing for an acquire/release pair
    /// to order against. The only cost of a late observation is one more tensor
    /// of work.
    flag: Arc<AtomicBool>,
}

impl CancelToken {
    /// Returns a fresh, uncancelled token.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Requests cancellation. Idempotent, and callable from any thread.
    ///
    /// Returns immediately: it sets a flag, it does not wait for the work to
    /// stop. The cancelled call returns
    /// [`AnamnesisError::Cancelled`](crate::AnamnesisError::Cancelled) once the
    /// in-flight tensors finish.
    pub fn cancel(&self) {
        self.flag.store(true, Ordering::Relaxed);
    }

    /// Returns `true` once [`cancel`](Self::cancel) has been called.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::Relaxed)
    }
}

/// Returns `Err(Cancelled)` if `cancel` is present and set, `Ok(())` otherwise.
///
/// The single poll site's shape, so the sequential and parallel dispatches
/// cannot disagree about what a cancelled run reports.
///
/// # Errors
///
/// Returns [`AnamnesisError::Cancelled`](crate::AnamnesisError::Cancelled) when
/// the token is set.
pub(crate) fn check(cancel: Option<&CancelToken>) -> crate::Result<()> {
    if cancel.is_some_and(CancelToken::is_cancelled) {
        return Err(crate::AnamnesisError::Cancelled);
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::{CancelToken, check};

    #[test]
    fn a_fresh_token_is_not_cancelled() {
        let token = CancelToken::new();
        assert!(!token.is_cancelled());
        assert!(check(Some(&token)).is_ok());
        assert!(check(None).is_ok());
    }

    #[test]
    fn cancel_is_visible_through_every_clone() {
        let token = CancelToken::new();
        let clone = token.clone();
        clone.cancel();
        assert!(
            token.is_cancelled(),
            "a clone shares the flag, it does not copy it"
        );
        assert!(matches!(
            check(Some(&token)),
            Err(crate::AnamnesisError::Cancelled)
        ));
    }

    #[test]
    fn cancel_is_idempotent() {
        let token = CancelToken::new();
        token.cancel();
        token.cancel();
        assert!(token.is_cancelled());
    }

    /// The token must cross a thread boundary, which is the whole point.
    #[test]
    fn cancel_crosses_threads() {
        let token = CancelToken::new();
        let worker = token.clone();
        std::thread::scope(|scope| {
            scope.spawn(move || worker.cancel());
        });
        assert!(token.is_cancelled());
    }
}
