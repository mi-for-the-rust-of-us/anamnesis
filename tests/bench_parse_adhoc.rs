// SPDX-License-Identifier: MIT OR Apache-2.0

//! Ad-hoc benchmark for `anamnesis::parse` + `ParsedModel::inspect` on a
//! large local safetensors file.
//!
//! Not part of CI — `#[ignore]`-gated and requires a multi-GB
//! safetensors file in the user's `HuggingFace` cache. Run with:
//!
//! ```text
//! cargo test --release --test bench_parse_adhoc bench_parse_safetensors_large \
//!     -- --nocapture --ignored
//! ```
//!
//! ## What it measures
//!
//! `parse()` reads the file into a buffer, then parses the safetensors
//! header. `inspect()` derives summary information from the header
//! (zero further I/O). For an inspect-only workflow the *amount* of data
//! actually examined is the header (~1 MiB on a multi-GB shard), so any
//! eager whole-file read in `parse()` is wasted work.
//!
//! This bench captures the wall-time impact of that wasted work on a
//! ~11 GiB single-file safetensors model. Audit finding #2 predicted
//! that switching the always-on `parse()` from `std::fs::read` to a
//! memory-mapped buffer (`memmap2::Mmap`) would dramatically improve
//! `parse()` + `inspect()` wall-time for large files because mmap setup
//! is constant-time vs `fs::read`'s linear cost in file size.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::indexing_slicing
)]

mod common;

use std::path::PathBuf;
use std::time::Instant;

use common::bench::fmt_stats;

/// Relative path under the `HuggingFace` cache to a large safetensors
/// model file (~11 GiB). Chosen because (a) it is a single-file shard
/// (no multi-file orchestration needed) and (b) it is large enough that
/// `fs::read` on it is dominated by I/O cost, making any difference
/// from a mmap-based path obvious.
const STARCODER2_3B_RELATIVE: &str = ".cache/huggingface/hub/\
    models--bigcode--starcoder2-3b/snapshots/\
    733247c55e3f73af49ce8e9c7949bf14af205928/model.safetensors";

fn fixture_path() -> Option<PathBuf> {
    let home = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME"))?;
    let path = PathBuf::from(home).join(STARCODER2_3B_RELATIVE);
    path.exists().then_some(path)
}

#[test]
#[ignore = "requires local 11+ GB starcoder2-3b safetensors file"]
fn bench_parse_safetensors_large() {
    let Some(path) = fixture_path() else {
        eprintln!("SKIP: starcoder2-3b not found under $HOME/{STARCODER2_3B_RELATIVE}");
        return;
    };

    let bytes = path.metadata().unwrap().len();
    eprintln!(
        "\n=== bench_parse_safetensors_large ({} MiB safetensors) ===",
        bytes / 1_048_576,
    );

    // Warm up the OS file cache. Two passes: first cold, second hot.
    let _ = anamnesis::parse(&path).unwrap();
    let _ = anamnesis::parse(&path).unwrap();

    // Best-of-5 — parse() alone.
    let mut parse_samples: Vec<f64> = Vec::with_capacity(5);
    let mut tensors_seen: usize = 0;
    for _ in 0..5 {
        let start = Instant::now();
        let model = anamnesis::parse(&path).unwrap();
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        // Defeat dead-code elimination by reading from the result.
        tensors_seen = model.inspect().quantized + model.inspect().passthrough;
        parse_samples.push(ms);
    }
    parse_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

    eprintln!("\n-- parse() alone --");
    eprintln!("samples (ms): {parse_samples:?}");
    eprintln!("{}", fmt_stats(&parse_samples));
    eprintln!("(saw {tensors_seen} tensors via inspect)");

    // Best-of-5 — parse() followed by inspect(). Inspect is supposed to
    // be near-free vs the parse cost; this measurement makes sure that
    // remains true (i.e. mmap doesn't accidentally make inspect lazy
    // about touching the header).
    let mut parse_inspect_samples: Vec<f64> = Vec::with_capacity(5);
    for _ in 0..5 {
        let start = Instant::now();
        let model = anamnesis::parse(&path).unwrap();
        let info = model.inspect();
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        // Defeat DCE
        let _ = std::hint::black_box(info);
        parse_inspect_samples.push(ms);
    }
    parse_inspect_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

    eprintln!("\n-- parse() + inspect() --");
    eprintln!("samples (ms): {parse_inspect_samples:?}");
    eprintln!("{}", fmt_stats(&parse_inspect_samples));

    // For the AFTER (mmap) variant we expect parse() median to drop
    // by orders of magnitude (from seconds to microseconds) because
    // mmap setup is constant-time. inspect() should remain a tiny
    // fixed cost dominated by header parsing. The DELTA on
    // parse()+inspect() vs parse() alone tells us the inspect overhead
    // — should be sub-millisecond regardless of file size.
    let parse_median = parse_samples[2];
    let parse_inspect_median = parse_inspect_samples[2];
    eprintln!(
        "\ninspect() overhead median: {:.2} ms",
        parse_inspect_median - parse_median
    );
}
