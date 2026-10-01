// SPDX-License-Identifier: MIT OR Apache-2.0

//! The anamnesis half of the `GGUF` speed table in `docs/validation.md`.
//!
//! Every `GGUF` cross-validation fixture (65 536 elements each, in
//! `tests/fixtures/gguf_reference/`) is dequantised to `BF16` with the
//! protocol `tests/fixtures/gguf_reference/time_gguf_py.py` applies to
//! `gguf-py`: 20 warm-up calls, then 200 timed calls in one process, min and
//! median reported. Run the two one after the other, never together, on an
//! idle machine:
//!
//! ```text
//! $env:RUSTFLAGS = "-C target-cpu=native"
//! cargo test --release --features gguf --test bench_gguf_table_adhoc -- --ignored --nocapture
//! $env:RUSTFLAGS = $null
//! python tests/fixtures/gguf_reference/time_gguf_py.py
//! ```
//!
//! Phase 7.10 wrote this after finding the table's figures unreproducible: they
//! came from single calls timed inside the cross-validation tests, which run in
//! parallel threads by default, so a kernel could be timed while its neighbours
//! competed for the same cores.

#![cfg(feature = "gguf")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]

mod common;

use anamnesis::{GgufType, dequantize_gguf_to_bf16};

use common::bench::{min_median_max, time_loop};
use common::fixture::read_u32_le;
use common::gguf::gguf_type_from_disc;

const WARMUP: usize = 20;
const ITERATIONS: usize = 200;

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
