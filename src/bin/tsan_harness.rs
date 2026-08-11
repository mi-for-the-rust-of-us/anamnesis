// SPDX-License-Identifier: MIT OR Apache-2.0

//! ThreadSanitizer harness for the parallel dequant dispatch.
//!
//! **Why this is a binary and not a test.** The race detector needs an
//! *instrumented standard library* — the Rust Unstable Book is explicit that
//! ThreadSanitizer "require\[s\] all code to be instrumented since otherwise
//! \[it\] generate\[s\] false positives". Instrumenting `std` means `-Zbuild-std`,
//! and `-Zbuild-std` is unusable with `cargo test`
//! (<https://github.com/rust-lang/cargo/issues/13146>, still open): the build
//! fails to link with `error[E0152]: duplicate lang item in crate `core``,
//! because `cargo test` also compiles dev-dependencies, several of which
//! (`cfg-if` via the unconditional `flate2` dev-dep, and friends) are crates
//! `std` itself depends on.
//!
//! `cargo run --bin` does **not** build dev-dependencies, so this target sheds
//! that collision surface and can be built the way the Unstable Book's own
//! sanitizer example is: `cargo run -Zbuild-std --target <triple>`.
//!
//! Running the checks here rather than through `cargo test` also avoids two
//! false positives that are otherwise unavoidable without an instrumented
//! `std`, both of which this project hit:
//!
//! - libtest's own result channel racing (`std::sync::mpmc`),
//!   <https://github.com/rust-lang/rust/issues/39608>.
//! - `std::thread::scope`'s `Arc` teardown appearing to race
//!   (`free` versus `atomic_sub`),
//!   <https://github.com/rust-lang/rust/issues/101206>. This one is fatal for
//!   this crate specifically: scoped threads *are* the mechanism under test, so
//!   without instrumented `std` the detector fires on the very thing it is
//!   meant to be checking.
//!
//! **What it exercises.** The same dispatch `remember` and `convert` use:
//! `ParsedModel::dequantize_all` → `parallel::map_indexed` → a scoped worker
//! pool drawing from the atomic cursor. The fixture is sized above
//! `parallel::MIN_PARALLEL_BYTES` on purpose — below that floor the dispatch
//! runs sequentially and the harness would spawn no threads at all, checking
//! nothing.
//!
//! Exits non-zero on a determinism mismatch; ThreadSanitizer itself exits 66 if
//! it finds a race.
//!
//! ```text
//! cargo +nightly run -Zbuild-std --target x86_64-unknown-linux-gnu --bin tsan-harness
//! ```

// A dev-only harness: it asserts and reports rather than returning Results into
// a library caller, so the crate's no-panic lint floor is relaxed here as it is
// in the other dev-only harnesses under `tests/`.
#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::indexing_slicing,
    // The module docs above are prose about toolchain behaviour, quoting the
    // Unstable Book and naming upstream issues. Backticking "ThreadSanitizer"
    // and the bracketed quotation would hurt readability more than it helps.
    clippy::doc_markdown
)]

use anamnesis::{parse_bytes, RememberOptions, TargetDtype};

/// Number of `FP8` weights in the fixture. A prime count so the atomic cursor
/// distributes work unevenly across every budget tried below, rather than
/// dividing cleanly and hiding an ordering bug.
const TENSORS: usize = 17;

/// Elements per weight. `17 × 262_144 ≈ 4.25 MiB` of quantised input, which
/// clears the 4 `MiB` `MIN_PARALLEL_BYTES` floor so the parallel path is
/// actually taken.
const ELEMENTS: usize = 262_144;

/// Thread budgets to cross-check. 1 is the sequential baseline.
const BUDGETS: [usize; 4] = [1, 2, 4, 8];

/// Builds a per-tensor-`FP8` safetensors image in memory: one `F8_E4M3` weight
/// plus its `F32` scale per layer, which is the scheme
/// `ParsedModel::dequantize_all` parallelises.
fn build_fp8_fixture() -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut data: Vec<u8> = Vec::new();

    for i in 0..TENSORS {
        let w_off = data.len();
        data.extend((0..ELEMENTS).map(|j| (j.wrapping_mul(2_654_435_761) & 0xFF) as u8));
        let mut w = serde_json::Map::new();
        w.insert("dtype".into(), "F8_E4M3".into());
        w.insert("shape".into(), serde_json::json!([ELEMENTS]));
        w.insert(
            "data_offsets".into(),
            serde_json::json!([w_off, data.len()]),
        );
        header.insert(format!("layer.{i}.weight"), w.into());

        let s_off = data.len();
        data.extend_from_slice(&0.125_f32.to_le_bytes());
        let mut s = serde_json::Map::new();
        s.insert("dtype".into(), "F32".into());
        s.insert("shape".into(), serde_json::json!([1]));
        s.insert(
            "data_offsets".into(),
            serde_json::json!([s_off, data.len()]),
        );
        header.insert(format!("layer.{i}.weight_scale"), s.into());
    }

    let header_json = serde_json::to_string(&header).expect("serialize header");
    let mut out = Vec::with_capacity(8 + header_json.len() + data.len());
    out.extend_from_slice(&(header_json.len() as u64).to_le_bytes());
    out.extend_from_slice(header_json.as_bytes());
    out.extend_from_slice(&data);
    out
}

fn main() {
    let bytes = build_fp8_fixture();
    println!(
        "tsan-harness: fixture {} tensors, {:.2} MiB quantised input",
        TENSORS,
        (TENSORS * ELEMENTS) as f64 / (1024.0 * 1024.0)
    );

    let model = parse_bytes(bytes).expect("parse synthetic FP8 fixture");

    let baseline = model
        .remember_to_bytes_with_options(TargetDtype::BF16, RememberOptions::new().with_threads(1))
        .expect("sequential dequant");
    println!("tsan-harness: baseline {} bytes", baseline.len());

    let mut failures = 0usize;
    for threads in BUDGETS {
        // Each iteration spawns a fresh scoped pool over the atomic cursor.
        let out = model
            .remember_to_bytes_with_options(
                TargetDtype::BF16,
                RememberOptions::new().with_threads(threads),
            )
            .expect("threaded dequant");
        if out == baseline {
            println!("tsan-harness: {threads:>2} threads -> byte-identical OK");
        } else {
            eprintln!(
                "tsan-harness: {threads:>2} threads -> MISMATCH ({} vs {} bytes)",
                out.len(),
                baseline.len()
            );
            failures += 1;
        }
    }

    // The hardware-resolved default must agree too.
    let default_out = model
        .remember_to_bytes(TargetDtype::BF16)
        .expect("default-budget dequant");
    if default_out == baseline {
        println!("tsan-harness: default budget -> byte-identical OK");
    } else {
        eprintln!("tsan-harness: default budget -> MISMATCH");
        failures += 1;
    }

    if failures > 0 {
        eprintln!("tsan-harness: {failures} determinism failure(s)");
        std::process::exit(1);
    }
    println!("tsan-harness: all budgets byte-identical");
}
