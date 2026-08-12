// SPDX-License-Identifier: MIT OR Apache-2.0

//! Ad-hoc benchmark for `parse_npz` on the real Gemma Scope file.
//! Not part of CI — requires the 302 MB file on disk.
//!
//! Run with: `cargo test --release --features npz bench_npz_real -- --nocapture --ignored`

#![cfg(feature = "npz")]
#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::as_conversions,
    clippy::cast_precision_loss
)]

use std::path::PathBuf;
use std::time::Instant;

/// Relative path under the `HuggingFace` cache to the Gemma Scope SAE weights.
const GEMMA_SCOPE_RELATIVE: &str = ".cache/huggingface/hub/\
    models--google--gemma-scope-2b-pt-res/snapshots/\
    fd571b47c1c64851e9b1989792367b9babb4af63/\
    layer_0/width_16k/average_l0_105/params.npz";

/// Resolve the Gemma Scope file path from the user's home directory.
fn gemma_scope_path() -> Option<PathBuf> {
    let home = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME"))?;
    let path = PathBuf::from(home).join(GEMMA_SCOPE_RELATIVE);
    if path.exists() { Some(path) } else { None }
}

#[test]
#[ignore = "requires local 302 MB Gemma Scope file"]
fn bench_npz_real() {
    let Some(path) = gemma_scope_path() else {
        eprintln!("SKIP: Gemma Scope file not found under $HOME/{GEMMA_SCOPE_RELATIVE}");
        return;
    };

    // Warm up filesystem cache
    let _ = anamnesis::parse_npz(&path);

    // Timed run
    let start = Instant::now();
    let tensors = anamnesis::parse_npz(&path).unwrap();
    let elapsed = start.elapsed();

    let total_bytes: usize = tensors.values().map(|t| t.data.len()).sum();
    let throughput_mb = total_bytes as f64 / elapsed.as_secs_f64() / 1_000_000.0;

    eprintln!("\n=== parse_npz benchmark (Gemma Scope 2B SAE, 302 MB) ===");
    eprintln!(
        "Total: {:.1} ms, {} MB, {throughput_mb:.0} MB/s",
        elapsed.as_secs_f64() * 1000.0,
        total_bytes / 1_000_000
    );
    for (name, t) in &tensors {
        eprintln!(
            "  {name}: {:?} {}, {} MB",
            t.shape,
            t.dtype,
            t.data.len() / 1_000_000
        );
    }

    // Baseline: raw file read (no ZIP, no parsing — just I/O throughput)
    let start2 = Instant::now();
    let raw = std::fs::read(&path).unwrap();
    let elapsed2 = start2.elapsed();
    let throughput2 = raw.len() as f64 / elapsed2.as_secs_f64() / 1_000_000.0;
    eprintln!(
        "\nRaw fs::read: {:.1} ms, {} MB, {throughput2:.0} MB/s",
        elapsed2.as_secs_f64() * 1000.0,
        raw.len() / 1_000_000
    );

    eprintln!(
        "parse_npz overhead vs raw I/O: {:.1}×",
        elapsed.as_secs_f64() / elapsed2.as_secs_f64()
    );
}
