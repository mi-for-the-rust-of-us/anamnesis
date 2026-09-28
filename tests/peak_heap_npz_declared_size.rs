// SPDX-License-Identifier: MIT OR Apache-2.0

//! A `DEFLATE` `NPZ` entry that declares gigabytes must not allocate them.
//!
//! Phase 7.9, audit finding M-6: `read_array_data` sized its buffer from the
//! entry's declared uncompressed size, so a few hundred bytes claiming 3 GB
//! committed 3 GB before the read failed. `STORED` entries are now held to
//! equal sizes by the `ZIP` reader; `DEFLATE` arrays grow as inflated bytes
//! arrive. Only a heap measurement can tell that growth from the old eager
//! allocation (both end in the same error), so this binary runs the lying
//! archive under `dhat` and bounds the peak. It takes milliseconds, so unlike
//! the other `peak_heap_*` binaries it is not `#[ignore]`d.

#![cfg(feature = "npz")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod common;

use common::builders::npz_with_declared_size;
use common::heap::dhat_lock;

#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

#[test]
fn a_deflate_size_lie_allocates_what_arrives_not_what_is_declared() {
    let bytes = npz_with_declared_size(
        "{'descr': '|u1', 'fortran_order': False, 'shape': (3000000000,), }",
        &[0u8; 16],
        true,
        Some(0xF000_0000),
    );
    let _dhat_guard = dhat_lock();
    let _profiler = dhat::Profiler::builder().testing().build();
    let result = anamnesis::parse_npz_bytes(bytes);
    let peak = dhat::HeapStats::get().max_bytes;
    assert!(result.is_err());
    assert!(
        peak < 1 << 20,
        "peak heap {peak} bytes: the declared 3 GB was allocated up front"
    );
}
