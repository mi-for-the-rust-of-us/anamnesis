// SPDX-License-Identifier: MIT OR Apache-2.0

//! The anamnesis half of the `GGUF` speed table in `docs/validation.md`.
//!
//! Every `GGUF` cross-validation fixture (65 536 elements each, in
//! `tests/fixtures/gguf_reference/`) is dequantised to `BF16` with the
//! protocol `tests/fixtures/gguf_reference/time_gguf_py.py` applies to
//! `gguf-py`: 20 warm-up calls, then 200 timed calls in one process, min and
//! median reported. Run the two one after the other, never together, on an
//! idle machine, five times each, alternating; the table publishes each
//! side's **best-of-5 median** (the lowest of the five per-pass medians), the
//! rule `CLAUDE.md` sets for absolute numbers on real fixtures:
//!
//! ```text
//! $env:RUSTFLAGS = "-C target-cpu=native"
//! cargo test --release --features gguf --test bench_gguf_table_adhoc -- --ignored --nocapture
//! $env:RUSTFLAGS = $null
//! python tests/fixtures/gguf_reference/time_gguf_py.py
//! ```
//!
//! Only the 65 536-element fixtures are timed, the size every table row is
//! measured at. Phase 7.11's exhaustive `F8_E4M3` fixtures (1061 and 1281
//! elements, every code) are cross-validation inputs, not speed-table rows.
//!
//! Phase 7.10 wrote this after finding the table's figures unreproducible: they
//! came from single calls timed inside the cross-validation tests, which run in
//! parallel threads by default, so a kernel could be timed while its neighbours
//! competed for the same cores.
//!
//! Phase 7.11 moved the table to best-of-5. On 2026-11-07 the first few
//! fixtures of a process ran about 2× slow in two passes of five while every
//! other row was stable, in both the per-fixture order and with the inputs
//! moved 2 KiB off page alignment; a whole-set warm-up pass did not remove it.
//! The paired harness showed no code regression on those kernels, so it is an
//! effect of the process, not of the kernel, and one pass can land in it.

#![cfg(feature = "gguf")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]

mod common;

use anamnesis::{GgufType, dequantize_gguf_to_bf16};

use common::bench::{min_median_max, time_loop};
use common::fixture::read_u32_le;
use common::gguf::gguf_type_from_disc;

const WARMUP: usize = 20;
const ITERATIONS: usize = 200;
/// The element count of every speed-table fixture.
const TABLE_ELEMENTS: usize = 65_536;

#[test]
#[ignore = "ad-hoc benchmark; run with --release --features gguf --ignored --nocapture"]
fn bench_gguf_fixture_table() {
    let dir =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gguf_reference");
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "bin"))
        .collect();
    paths.sort();

    eprintln!("\nfixture\tkernel\tmin_us\tmedian_us\t({WARMUP} warm-up, {ITERATIONS} timed calls)");
    for path in paths {
        let data = std::fs::read(&path).unwrap();
        if data.get(..4) != Some(b"AMNG".as_slice()) {
            continue;
        }
        let dtype: GgufType = gguf_type_from_disc(read_u32_le(&data, 8));
        let n_elements = usize::try_from(read_u32_le(&data, 12)).unwrap();
        if n_elements != TABLE_ELEMENTS {
            continue;
        }
        let raw_len = usize::try_from(read_u32_le(&data, 16)).unwrap();
        let raw = &data[28..28 + raw_len];

        let mut samples = time_loop(WARMUP, ITERATIONS, || {
            dequantize_gguf_to_bf16(raw, dtype, n_elements).unwrap()
        });
        let (min, median, _) = min_median_max(&mut samples);
        let name = path.file_stem().unwrap().to_string_lossy();
        eprintln!(
            "{name}\t{dtype}\t{:.1}\t{:.1}",
            min.as_secs_f64() * 1e6,
            median.as_secs_f64() * 1e6
        );
    }
}
