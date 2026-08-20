// SPDX-License-Identifier: MIT OR Apache-2.0

//! Caller-configurable resource limits for the parsers.
//!
//! Every anamnesis parser already enforces a permanent, per-format constant
//! cap on each header-declared allocation (`NPZ_MAX_ARRAY_BYTES`,
//! `MAX_PKL_SIZE`, the `GGUF` `MAX_*` family, `MAX_SAFETENSORS_HEADER_BYTES`,
//! …). Those caps are tuned for server / GPU hosts and cannot be relaxed.
//! [`ParseLimits`] layers a *second*, caller-supplied ceiling on top of them:
//! a memory-constrained edge board or a per-slot `MLaaS` worker passes a
//! [`ParseLimits`] tightened to its own budget, and the parser rejects an
//! over-budget declaration fail-fast with [`AnamnesisError::LimitExceeded`](crate::AnamnesisError::LimitExceeded)
//! **before** it allocates.
//!
//! The ceiling is **tighten-only**: where a built-in per-format constant exists
//! — the per-item allocation caps and the `GGUF` count caps — the effective
//! limit is `min(format_constant, parse_limit)`, so a caller can only make the
//! parser stricter, never weaker than the floor. The cumulative-aggregate and
//! decompression-ratio axes have no built-in floor (anamnesis imposes no
//! inherent aggregate or ratio cap) and are pure caller-supplied bounds.
//! [`ParseLimits::default`] is *unbounded* on every axis ([`u64::MAX`] sentinel),
//! so the default leaves only the per-format constants in force — behaviour
//! identical to a build with no [`ParseLimits`] at all.

/// Caller-supplied resource budget threaded through the parser entry points.
///
/// Construct with [`ParseLimits::default`] (unbounded — today's behaviour) and
/// tighten individual axes with the `with_*` builders:
///
/// ```
/// use anamnesis::ParseLimits;
///
/// // Reject any single allocation over 256 MiB, a cumulative parse-time heap
/// // over 1 GiB, a file declaring more than 4096 tensors / arrays / KV
/// // entries, or a compressed entry that inflates more than 1000×.
/// let limits = ParseLimits::default()
///     .with_max_single_alloc(256 * 1024 * 1024)
///     .with_max_total_bytes(1024 * 1024 * 1024)
///     .with_max_item_count(4096)
///     .with_max_decompression_ratio(1000);
/// ```
///
/// Pass it to the `parse_*_with_limits` entry points. Each axis is enforced
/// fail-fast — the size and count axes *before* the corresponding allocation,
/// the decompression-ratio axis from archive metadata *before* reading — so an
/// over-budget or zip-bomb file returns an error without first committing the
/// memory.
// Deliberately not `Copy`: `ParseLimits` is a configuration/budget value passed
// by `&ParseLimits` everywhere (and borrowed by the `.pth` pickle VM), so a
// `Copy` derive would trip clippy's `trivially_copy_pass_by_ref` on the 16-byte
// struct. `Clone` covers the rare by-value need.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
// The shared `max_` prefix is intentional — every field is a caller-set maximum
// (`max_single_alloc_bytes`, `max_total_bytes`, `max_item_count`); the prefix is
// what makes the budget axes read uniformly.
#[allow(clippy::struct_field_names)]
pub struct ParseLimits {
    /// Upper bound, in bytes, on any single header-declared buffer a parser
    /// allocates eagerly: an `NPZ` array and `NPY` header, a `.pth` `data.pkl`
    /// entry and each pickle string/bytes payload, a `GGUF` variable-length
    /// (string) read and scalar metadata array, the safetensors header, and the
    /// `ZIP` central-directory read on the streaming (`.npz` / `.pth`-reader)
    /// path (the mmap `.pth` path reads it zero-copy, so it is not charged).
    /// [`u64::MAX`] means unbounded — only the per-format constant cap applies.
    ///
    /// Not covered (by design): memory-mapped tensor bodies, which are never
    /// heap-allocated at parse time (their declared total is what `inspect_*`
    /// reports for the host's inspect-before-parse policy gate).
    max_single_alloc_bytes: u64,

    /// Upper bound, in bytes, on the *cumulative* parse-time heap a file may
    /// drive — the running sum of every eager allocation
    /// [`Self::max_single_alloc_bytes`] gates. Closes the many-small-items blow-up: a file declaring thousands
    /// of buffers each just under the single-allocation cap is rejected once
    /// their total crosses this budget, before the host `OOM`s. Since Phase
    /// 6.11 the running sum also includes the `.pth` pickle VM's working set —
    /// every value it pushes plus the deep size of each memo clone — so a
    /// crafted pickle that amplifies a small opcode stream into multi-GiB heap
    /// is bounded here too (and, independently of this caller budget, by the
    /// VM's permanent `MAX_PICKLE_WORKING_SET` floor). [`u64::MAX`] means
    /// unbounded.
    max_total_bytes: u64,

    /// Upper bound on the total number of declared items in a file — `GGUF`
    /// tensors and metadata KV entries, or `ZIP` central-directory entries (the
    /// `.npz` and `.pth` containers, via the vendored reader). [`u64::MAX`]
    /// means unbounded — only the per-format constant cap applies.
    max_item_count: u64,

    /// Upper bound on a compressed archive entry's uncompressed-to-compressed
    /// expansion ratio — the zip-bomb cap for `DEFLATE` `NPZ` entries. A few-KB
    /// entry that *honestly* declares a gigabyte-scale uncompressed size passes
    /// every byte-size check yet is a `1 000 000:1` amplification no real file
    /// produces; this rejects it from the archive metadata before allocating.
    /// `STORED` entries report equal sizes (ratio `1`) and always pass.
    /// [`u64::MAX`] means unbounded. Applies to `NPZ` only (the sole `DEFLATE`
    /// path; `.pth` is `STORED`-only, `GGUF` / safetensors are not zipped).
    max_decompression_ratio: u64,
}

impl ParseLimits {
    /// Returns an unbounded budget: every axis is [`u64::MAX`], so only the
    /// permanent per-format constant caps apply. Identical to
    /// [`ParseLimits::default`], but usable in `const` context.
    #[must_use]
    pub const fn unbounded() -> Self {
        Self {
            max_single_alloc_bytes: u64::MAX,
            max_total_bytes: u64::MAX,
            max_item_count: u64::MAX,
            max_decompression_ratio: u64::MAX,
        }
    }

    /// Sets the maximum single-allocation budget, in bytes, and returns the
    /// updated value. See [`ParseLimits::max_single_alloc_bytes`].
    #[must_use]
    pub const fn with_max_single_alloc(mut self, bytes: u64) -> Self {
        self.max_single_alloc_bytes = bytes;
        self
    }

    /// Sets the maximum cumulative parse-time heap budget, in bytes, and
    /// returns the updated value. See [`ParseLimits::max_total_bytes`].
    #[must_use]
    pub const fn with_max_total_bytes(mut self, bytes: u64) -> Self {
        self.max_total_bytes = bytes;
        self
    }

    /// Sets the maximum declared-item-count budget and returns the updated
    /// value. See [`ParseLimits::max_item_count`].
    #[must_use]
    pub const fn with_max_item_count(mut self, count: u64) -> Self {
        self.max_item_count = count;
        self
    }

    /// Sets the maximum decompression (expansion) ratio for compressed archive
    /// entries and returns the updated value. See
    /// [`ParseLimits::max_decompression_ratio`].
    #[must_use]
    pub const fn with_max_decompression_ratio(mut self, ratio: u64) -> Self {
        self.max_decompression_ratio = ratio;
        self
    }

    /// Returns the maximum single-allocation budget, in bytes ([`u64::MAX`] if
    /// unbounded).
    #[must_use]
    pub const fn max_single_alloc_bytes(&self) -> u64 {
        self.max_single_alloc_bytes
    }

    /// Returns the maximum cumulative parse-time heap budget, in bytes
    /// ([`u64::MAX`] if unbounded).
    #[must_use]
    pub const fn max_total_bytes(&self) -> u64 {
        self.max_total_bytes
    }

    /// Returns the maximum declared-item-count budget ([`u64::MAX`] if
    /// unbounded).
    #[must_use]
    pub const fn max_item_count(&self) -> u64 {
        self.max_item_count
    }

    /// Returns the maximum decompression (expansion) ratio for compressed
    /// archive entries ([`u64::MAX`] if unbounded).
    #[must_use]
    pub const fn max_decompression_ratio(&self) -> u64 {
        self.max_decompression_ratio
    }

    /// Rejects a single allocation of `requested` bytes if it exceeds the
    /// caller's [`ParseLimits::max_single_alloc_bytes`] budget. Called at every
    /// site that allocates a header-declared buffer, immediately after the
    /// permanent per-format constant check.
    ///
    /// `context` names the offending region for the error message (e.g.
    /// `` "NPZ array `weight`" ``).
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::LimitExceeded`](crate::AnamnesisError::LimitExceeded)
    /// if `requested` exceeds the configured maximum single allocation.
    pub(crate) fn check_alloc(&self, requested: u64, context: &str) -> crate::Result<()> {
        if requested > self.max_single_alloc_bytes {
            return Err(crate::AnamnesisError::LimitExceeded {
                limit: "max_single_alloc_bytes",
                message: format!(
                    "requested allocation {requested} bytes exceeds caller \
                     ParseLimits max_single_alloc {} ({context})",
                    self.max_single_alloc_bytes
                ),
            });
        }
        Ok(())
    }

    /// Rejects a declared item count if it exceeds the caller's
    /// [`ParseLimits::max_item_count`] budget. Called at every site that reads
    /// a file-declared count of tensors / arrays / KV entries, immediately
    /// after the permanent per-format constant check.
    ///
    /// `context` names the counted population for the error message (e.g.
    /// `"GGUF tensor count"`).
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::LimitExceeded`](crate::AnamnesisError::LimitExceeded)
    /// if `count` exceeds the configured maximum item count.
    // Callers: the `gguf` parse path (tensor / KV counts) and the vendored ZIP
    // central-directory reader (`crate::parse::zip`, compiled under `npz`/`pth`,
    // for the entry count). With none of `npz`/`pth`/`gguf` enabled this helper
    // has no caller, which is correct (not dead code in the public sense).
    // `check_alloc` always has a caller via the always-on safetensors path, so
    // it needs no such guard.
    #[cfg_attr(
        not(any(feature = "npz", feature = "pth", feature = "gguf")),
        allow(dead_code)
    )]
    pub(crate) fn check_item_count(&self, count: u64, context: &str) -> crate::Result<()> {
        if count > self.max_item_count {
            return Err(crate::AnamnesisError::LimitExceeded {
                limit: "max_item_count",
                message: format!(
                    "declared item count {count} exceeds caller ParseLimits \
                     max_item_count {} ({context})",
                    self.max_item_count
                ),
            });
        }
        Ok(())
    }

    /// Rejects a compressed archive entry whose uncompressed-to-compressed
    /// expansion ratio exceeds the caller's
    /// [`ParseLimits::max_decompression_ratio`] cap — the zip-bomb guard,
    /// checked from archive metadata **before** the entry is read. A `STORED`
    /// entry has `uncompressed == compressed` (ratio `1`) and always passes.
    ///
    /// `context` names the offending entry for the error message (e.g. the
    /// array name).
    ///
    /// The bound is `uncompressed <= max_ratio * compressed`, evaluated with a
    /// `checked_mul` so a `compressed == 0` entry is rejected only when it
    /// declares a non-zero `uncompressed` (an empty entry passes), and an
    /// allowed-bound overflow (a huge ratio cap) is treated as within bound.
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::LimitExceeded`](crate::AnamnesisError::LimitExceeded)
    /// if the declared expansion ratio exceeds the configured maximum.
    // Only the `npz` parse path reads `DEFLATE` (compressed) archive entries;
    // with that feature disabled this helper has no caller.
    #[cfg_attr(not(feature = "npz"), allow(dead_code))]
    pub(crate) fn check_decompression_ratio(
        &self,
        uncompressed: u64,
        compressed: u64,
        context: &str,
    ) -> crate::Result<()> {
        if self.max_decompression_ratio == u64::MAX {
            return Ok(());
        }
        // Reject when `uncompressed > max_ratio * compressed`. A `checked_mul`
        // overflow means the allowed bound exceeds `u64` (≥ any `uncompressed`)
        // → within bound; `compressed == 0` yields `allowed == 0` → only a
        // non-empty entry is rejected.
        match self.max_decompression_ratio.checked_mul(compressed) {
            Some(allowed) if uncompressed > allowed => Err(crate::AnamnesisError::LimitExceeded {
                limit: "max_decompression_ratio",
                message: format!(
                    "decompression ratio (uncompressed {uncompressed} / compressed \
                     {compressed}) exceeds caller ParseLimits max_decompression_ratio \
                     {} ({context})",
                    self.max_decompression_ratio
                ),
            }),
            _ => Ok(()),
        }
    }

    /// Reads an entire reader into an owned `Vec<u8>`, bounded by
    /// [`ParseLimits::max_single_alloc_bytes`] so a hostile or unbounded stream
    /// cannot exhaust memory before the parser sees it. The read is capped at
    /// `max_single_alloc_bytes + 1` bytes; reaching that cap trips
    /// [`check_alloc`](Self::check_alloc) and the read is rejected with a clean
    /// `Err` instead of growing without bound. This is the copy-based
    /// untrusted-input path's counterpart to the mmap path's lazy paging — both
    /// route the size check through `check_alloc` so the bound cannot drift.
    ///
    /// `context` names the source for the error message.
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::Io`](crate::AnamnesisError::Io) if the
    /// underlying read fails.
    /// Returns
    /// [`AnamnesisError::LimitExceeded`](crate::AnamnesisError::LimitExceeded)
    /// (`max_single_alloc_bytes`) if the bytes read exceed the caller's budget —
    /// this is `check_alloc`'s variant.
    /// Returns [`AnamnesisError::Parse`](crate::AnamnesisError::Parse) only in the
    /// degenerate case that the read length overflows `u64`.
    ///
    /// # Memory
    ///
    /// Allocates one `Vec<u8>` holding the whole stream (at most
    /// `max_single_alloc_bytes + 1` bytes). With `ParseLimits::default()` the
    /// bound is `u64::MAX`, i.e. effectively the file size — matching the mmap
    /// path's no-inherent-limit behaviour.
    // Used by the copy-based `parse_*_from_reader` entry points (always-on
    // safetensors + the `gguf`/`pth` features); never dead in the public sense.
    pub(crate) fn read_to_vec_bounded<R: std::io::Read>(
        &self,
        reader: R,
        context: &str,
    ) -> crate::Result<Vec<u8>> {
        use std::io::Read as _;
        let cap = self.max_single_alloc_bytes.saturating_add(1);
        let mut buf = Vec::new();
        reader
            .take(cap)
            .read_to_end(&mut buf)
            .map_err(crate::AnamnesisError::Io)?;
        let read_len = u64::try_from(buf.len()).map_err(|_| crate::AnamnesisError::Parse {
            reason: format!("{context}: read length overflows u64"),
        })?;
        self.check_alloc(read_len, context)?;
        Ok(buf)
    }
}

impl Default for ParseLimits {
    /// Returns an unbounded budget — every axis [`u64::MAX`] — so the default
    /// leaves only the permanent per-format constant caps in force. Parsing
    /// with `ParseLimits::default()` is byte-for-byte identical to parsing
    /// through the limit-free entry points.
    fn default() -> Self {
        Self::unbounded()
    }
}

/// Running byte accountant for a single parse, enforcing both the per-item
/// [`ParseLimits::max_single_alloc_bytes`] ceiling and the cumulative
/// [`ParseLimits::max_total_bytes`] aggregate budget.
///
/// One `Budget` is created per parse from the caller's [`ParseLimits`] and
/// threaded through the eager-allocation sites: each site `charge`s the bytes
/// it is about to allocate, fail-fast *before* the allocation. The aggregate
/// closes the many-small-items blow-up the per-item cap misses.
///
/// Owns a (cheap) [`ParseLimits`] clone rather than borrowing, so the per-parse
/// reader/VM structs can own a `Budget` field without a lifetime parameter.
pub(crate) struct Budget {
    /// The caller's limits (the immutable ceilings).
    limits: ParseLimits,
    /// Cumulative bytes charged so far this parse.
    total_bytes: u64,
}

impl Budget {
    /// Creates a fresh accountant for one parse, with a zero running total.
    pub(crate) fn new(limits: &ParseLimits) -> Self {
        Self {
            limits: limits.clone(),
            total_bytes: 0,
        }
    }

    /// An unbounded accountant, for tests that exercise a parser helper without
    /// a budget in the picture.
    ///
    /// `#[cfg(test)]` since v0.7.6. It used to be reachable from production
    /// code as *"convenience for the inspect paths (not yet `ParseLimits`-aware)"*,
    /// and that parenthesis was the whole defect Phase 7.6 item 7 closed: the
    /// inspect paths are `ParseLimits`-aware now, so nothing outside a test
    /// wants an accountant that cannot say no.
    #[cfg(test)]
    pub(crate) fn unbounded() -> Self {
        Self::new(&ParseLimits::unbounded())
    }

    /// Charges `bytes` against both the per-item single-allocation ceiling and
    /// the cumulative aggregate budget, **before** the allocation. Enforces the
    /// per-item cap first (rejecting an oversized single item before it can
    /// inflate the running total), then adds to and checks the aggregate.
    ///
    /// `context` names the offending region for the error message.
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::LimitExceeded`](crate::AnamnesisError::LimitExceeded)
    /// if `bytes` exceeds [`ParseLimits::max_single_alloc_bytes`] or the new total
    /// exceeds [`ParseLimits::max_total_bytes`]; returns
    /// [`AnamnesisError::Parse`](crate::AnamnesisError::Parse) if the running
    /// total overflows `u64`.
    pub(crate) fn charge_alloc(&mut self, bytes: u64, context: &str) -> crate::Result<()> {
        // Per-item ceiling first (Step 1), so a single oversized item is
        // rejected on its own terms before it touches the running total.
        self.limits.check_alloc(bytes, context)?;
        let new_total =
            self.total_bytes
                .checked_add(bytes)
                .ok_or_else(|| crate::AnamnesisError::Parse {
                    reason: format!("aggregate byte total overflow charging {bytes} ({context})"),
                })?;
        if new_total > self.limits.max_total_bytes {
            return Err(crate::AnamnesisError::LimitExceeded {
                limit: "max_total_bytes",
                message: format!(
                    "cumulative declared bytes {new_total} exceeds caller ParseLimits \
                     max_total_bytes {} ({context})",
                    self.limits.max_total_bytes
                ),
            });
        }
        self.total_bytes = new_total;
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::{Budget, ParseLimits};

    #[test]
    fn default_is_unbounded() {
        let limits = ParseLimits::default();
        assert_eq!(limits.max_single_alloc_bytes(), u64::MAX);
        assert_eq!(limits.max_total_bytes(), u64::MAX);
        assert_eq!(limits.max_item_count(), u64::MAX);
        assert_eq!(limits.max_decompression_ratio(), u64::MAX);
        assert_eq!(limits, ParseLimits::unbounded());
    }

    #[test]
    fn check_decompression_ratio_boundary() {
        // At the cap passes; one over fails. ratio cap 100, compressed 10 →
        // allowed 1000.
        let limits = ParseLimits::default().with_max_decompression_ratio(100);
        assert!(limits.check_decompression_ratio(1000, 10, "ctx").is_ok());
        let err = limits
            .check_decompression_ratio(1001, 10, "ctx")
            .unwrap_err();
        assert!(
            matches!(err, crate::AnamnesisError::LimitExceeded { limit, .. } if limit == "max_decompression_ratio"),
            "expected ratio error, got: {err}"
        );

        // `compressed == 0`: empty entry passes, non-empty is rejected.
        assert!(limits.check_decompression_ratio(0, 0, "ctx").is_ok());
        assert!(limits.check_decompression_ratio(1, 0, "ctx").is_err());

        // A huge ratio cap whose `allowed` bound overflows `u64` is within
        // bound (never a panic).
        let huge = ParseLimits::default().with_max_decompression_ratio(u64::MAX / 2);
        assert!(huge.check_decompression_ratio(10, 4, "ctx").is_ok());

        // Unbounded (default) never fires.
        assert!(
            ParseLimits::default()
                .check_decompression_ratio(u64::MAX, 1, "ctx")
                .is_ok()
        );
    }

    #[test]
    fn builders_set_only_their_axis() {
        let limits = ParseLimits::default().with_max_single_alloc(1024);
        assert_eq!(limits.max_single_alloc_bytes(), 1024);
        assert_eq!(limits.max_total_bytes(), u64::MAX);
        assert_eq!(limits.max_item_count(), u64::MAX);
        assert_eq!(limits.max_decompression_ratio(), u64::MAX);

        let limits = ParseLimits::default().with_max_total_bytes(4096);
        assert_eq!(limits.max_total_bytes(), 4096);
        assert_eq!(limits.max_single_alloc_bytes(), u64::MAX);
        assert_eq!(limits.max_item_count(), u64::MAX);
        assert_eq!(limits.max_decompression_ratio(), u64::MAX);

        let limits = ParseLimits::default().with_max_item_count(8);
        assert_eq!(limits.max_item_count(), 8);
        assert_eq!(limits.max_single_alloc_bytes(), u64::MAX);
        assert_eq!(limits.max_total_bytes(), u64::MAX);
        assert_eq!(limits.max_decompression_ratio(), u64::MAX);

        let limits = ParseLimits::default().with_max_decompression_ratio(1000);
        assert_eq!(limits.max_decompression_ratio(), 1000);
        assert_eq!(limits.max_single_alloc_bytes(), u64::MAX);
        assert_eq!(limits.max_total_bytes(), u64::MAX);
        assert_eq!(limits.max_item_count(), u64::MAX);
    }

    #[test]
    fn budget_aggregate_catches_what_per_item_misses() {
        // Each 100-byte charge is well under the 1000-byte single-alloc cap,
        // but their sum must not cross the 250-byte aggregate budget.
        let limits = ParseLimits::default()
            .with_max_single_alloc(1000)
            .with_max_total_bytes(250);
        let mut budget = Budget::new(&limits);
        assert!(budget.charge_alloc(100, "a").is_ok()); // total 100
        assert!(budget.charge_alloc(100, "b").is_ok()); // total 200
        // Third charge would reach 300 > 250 — rejected by the aggregate even
        // though 100 passes the per-item cap.
        let err = budget.charge_alloc(100, "c").unwrap_err();
        assert!(
            matches!(err, crate::AnamnesisError::LimitExceeded { limit, .. } if limit == "max_total_bytes"),
            "expected aggregate error, got: {err}"
        );
    }

    #[test]
    fn budget_per_item_cap_still_applies() {
        let limits = ParseLimits::default()
            .with_max_single_alloc(50)
            .with_max_total_bytes(u64::MAX);
        let mut budget = Budget::new(&limits);
        let err = budget.charge_alloc(51, "x").unwrap_err();
        assert!(
            matches!(err, crate::AnamnesisError::LimitExceeded { limit, .. } if limit == "max_single_alloc_bytes"),
            "expected single-alloc error, got: {err}"
        );
    }

    #[test]
    fn budget_unbounded_charges_until_overflow() {
        let mut budget = Budget::unbounded();
        assert!(budget.charge_alloc(u64::MAX, "x").is_ok());
        // A second non-zero charge overflows the running total → clean error,
        // never a panic.
        let err = budget.charge_alloc(1, "y").unwrap_err();
        assert!(
            matches!(err, crate::AnamnesisError::Parse { ref reason } if reason.contains("overflow")),
            "expected overflow error, got: {err}"
        );
    }

    #[test]
    fn check_alloc_boundary() {
        let limits = ParseLimits::default().with_max_single_alloc(1024);
        // At the cap passes; one over fails.
        assert!(limits.check_alloc(1024, "ctx").is_ok());
        assert!(limits.check_alloc(1025, "ctx").is_err());
        // Unbounded never fires.
        assert!(ParseLimits::default().check_alloc(u64::MAX, "ctx").is_ok());
    }

    #[test]
    fn check_item_count_boundary() {
        let limits = ParseLimits::default().with_max_item_count(4);
        assert!(limits.check_item_count(4, "ctx").is_ok());
        assert!(limits.check_item_count(5, "ctx").is_err());
        assert!(
            ParseLimits::default()
                .check_item_count(u64::MAX, "ctx")
                .is_ok()
        );
    }
}
