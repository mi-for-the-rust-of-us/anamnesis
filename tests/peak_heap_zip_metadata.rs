// SPDX-License-Identifier: MIT OR Apache-2.0

//! Phase 6.12 metadata-reduction measurement: the vendored ZIP
//! central-directory reader vs the `zip` crate's `ZipArchive::new`.
//!
//! `ZipArchive::new` eagerly materialises the whole central directory into a
//! fat per-entry `ZipFileData` record; anamnesis needs only a
//! `name → (offset, size)` index. This binary parses a 50 000-tiny-entry
//! `.pth` archive both ways under `dhat` and reports the resident and peak
//! metadata heap, so the reduction is reproducible and recorded.
//!
//! Measured on the dev machine (release, 50 001 entries): resident metadata
//! drops from **337 B/entry** (`zip` crate, 16.9 MB) to **41 B/entry**
//! (vendored `parse_pth`, 2.1 MB) — an **8.07× resident reduction** (3.12× on
//! peak). The 41 B/entry matches the Phase 6.8 analysis's ~40 B/entry target
//! (achieved by indexing entries in a sorted, `shrink_to_fit`-trimmed
//! `Vec<(Box<str>, usize, usize)>` rather than a `HashMap<String, …>` — no
//! hash-table power-of-two bucket slack, no `String` capacity word). That
//! analysis projected ~12×, but on these short entry names the `zip` crate
//! measures 337 B/entry (not the estimated ~500), so ~8× is the real ceiling
//! and the vendored index now sits at it.
//!
//! Both measurements go through what each library actually exposes: the
//! vendored reader via the public `parse_pth` (the `.pth` mmap path, with a
//! trivial empty-`state_dict` pickle so the pickle VM contributes ~nothing and
//! the container index dominates), and the `zip` crate via `ZipArchive::new`
//! (the call `parse_pth` used before Phase 6.12). `dhat` tracks the global
//! allocator, so the memory-mapped file body is **not** counted — only the
//! container metadata heap, which is the quantity under comparison.
//!
//! Behind `#[ignore]` so default `cargo test` skips it. Run with:
//!
//! ```text
//! cargo test --release --features pth --test peak_heap_zip_metadata \
//!   -- --ignored --nocapture
//! ```

#![cfg(feature = "pth")]
#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]

use std::io::Write;

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

/// Builds a `.pth` archive with `n` tiny STORED `archive/data/{i}` entries plus
/// a valid empty-`state_dict` `archive/data.pkl` (`PROTO 2`, `EMPTY_DICT`,
/// `STOP`). The tensor-data entries are unreferenced by the pickle, so
/// `parse_pth` indexes all `n + 1` entries but extracts zero tensors.
fn build_many_entry_pth(n: usize) -> tempfile::NamedTempFile {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    {
        let file = std::fs::File::create(tmp.path()).unwrap();
        let mut writer = zip::ZipWriter::new(file);
        let opts = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored);
        writer.start_file("archive/data.pkl", opts).unwrap();
        writer.write_all(&[0x80, 0x02, b'}', b'.']).unwrap();
        for i in 0..n {
            writer
                .start_file(format!("archive/data/{i}"), opts)
                .unwrap();
            writer.write_all(&[0u8]).unwrap();
        }
        writer.finish().unwrap();
    }
    tmp
}

/// Resident (`curr_bytes`) and peak (`max_bytes`) heap captured around one
/// container parse.
struct HeapSample {
    resident: usize,
    peak: usize,
}

#[test]
#[ignore = "dhat metadata-reduction measurement; run with --ignored --nocapture"]
fn zip_container_metadata_reduction() {
    const N: usize = 50_000;

    // Held for the whole body, not just the profiler: `dhat` counts every
    // allocation in the process, so a concurrent test synthesising its own
    // fixture would inflate this one's peak even with the profiler serialised.
    let _dhat_guard = dhat_lock();
    let entries = N + 1;
    let tmp = build_many_entry_pth(N);

    // Vendored reader via the public `parse_pth` mmap path. Hold the result
    // across `HeapStats::get()` so `curr_bytes` reflects the resident index.
    let vendored = {
        let profiler = dhat::Profiler::builder().testing().build();
        let parsed = anamnesis::parse_pth(tmp.path()).expect("parse_pth");
        assert!(parsed.is_empty(), "empty state_dict should yield 0 tensors");
        let stats = dhat::HeapStats::get();
        drop(parsed);
        drop(profiler);
        HeapSample {
            resident: stats.curr_bytes,
            peak: stats.max_bytes,
        }
    };

    // The `zip` crate's `ZipArchive::new` — the path replaced in Phase 6.12.
    let zip_crate = {
        let profiler = dhat::Profiler::builder().testing().build();
        let file = std::fs::File::open(tmp.path()).unwrap();
        let archive = zip::ZipArchive::new(file).expect("ZipArchive::new");
        assert!(archive.len() >= N, "archive should hold all entries");
        let stats = dhat::HeapStats::get();
        drop(archive);
        drop(profiler);
        HeapSample {
            resident: stats.curr_bytes,
            peak: stats.max_bytes,
        }
    };

    eprintln!("ZIP container metadata over {entries} entries (50k tiny STORED + data.pkl):");
    eprintln!(
        "  vendored (parse_pth)      resident={} B ({} B/entry)  peak={} B",
        vendored.resident,
        vendored.resident / entries,
        vendored.peak,
    );
    eprintln!(
        "  zip::ZipArchive::new      resident={} B ({} B/entry)  peak={} B",
        zip_crate.resident,
        zip_crate.resident / entries,
        zip_crate.peak,
    );
    eprintln!(
        "  reduction: resident {:.2}x, peak {:.2}x",
        zip_crate.resident as f64 / vendored.resident.max(1) as f64,
        zip_crate.peak as f64 / vendored.peak.max(1) as f64,
    );

    // Regression guard: the vendored reader's resident metadata must stay well
    // below the `zip` crate's fat per-entry records.
    assert!(
        vendored.resident < zip_crate.resident,
        "expected a resident-metadata reduction, got vendored {} B >= zip {} B",
        vendored.resident,
        zip_crate.resident,
    );
}
