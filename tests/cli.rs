// SPDX-License-Identifier: MIT OR Apache-2.0

//! CLI integration tests for the `anamnesis` / `amn` binary.
//!
//! These tests build and invoke the binary via `std::process::Command` to
//! verify argument parsing, subcommand routing, and output format. They
//! complement the library-level tests in `cross_validation.rs`.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing
)]

use std::process::Command;

/// Path to the built binary (cargo sets this via the test harness).
///
/// Panics with a diagnostic message if the binary is missing or stale
/// (version mismatch with `Cargo.toml`). Fix: run `cargo build` before
/// `cargo test`.
fn binary_path() -> std::path::PathBuf {
    // `cargo test` builds binaries into target/debug/
    let mut path = std::env::current_exe()
        .expect("cannot determine test executable path")
        .parent()
        .expect("no parent directory")
        .parent()
        .expect("no grandparent directory")
        .to_path_buf();
    path.push(if cfg!(windows) {
        "anamnesis.exe"
    } else {
        "anamnesis"
    });

    // Guard: binary must exist
    assert!(
        path.exists(),
        "CLI binary not found at {}. Run `cargo build --features cli` before `cargo test`.",
        path.display()
    );

    // Guard: binary version must match Cargo.toml
    let output = Command::new(&path)
        .arg("--version")
        .output()
        .unwrap_or_else(|e| panic!("cannot run {}: {e}", path.display()));
    let stdout = String::from_utf8_lossy(&output.stdout);
    let expected = env!("CARGO_PKG_VERSION");
    assert!(
        stdout.contains(expected),
        "STALE BINARY: {} reports `{stdout}` \
         but Cargo.toml has v{expected}. Run `cargo build --features cli` before `cargo test`.",
        path.display()
    );

    path
}

/// Build a minimal safetensors file in a temp directory for testing.
///
/// Contains:
/// - 1 FP8 weight tensor (2×2, all 1.0 in E4M3)
/// - 1 F32 scale tensor (scalar [1], value 2.0)
/// - 1 BF16 passthrough tensor (norm, 1 element)
fn create_test_fixture() -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().expect("failed to create temp dir");
    let path = dir.path().join("test-fp8.safetensors");

    let fp8_data = vec![0x38u8; 4]; // 2×2 of 1.0 in E4M3
    let scale_data = 2.0_f32.to_le_bytes().to_vec();
    let norm_data = vec![0x80, 0x3F]; // BF16 1.0

    let mut header_map = serde_json::Map::new();

    let mut w_info = serde_json::Map::new();
    w_info.insert("dtype".into(), "F8_E4M3".into());
    w_info.insert("shape".into(), serde_json::json!([2, 2]));
    w_info.insert("data_offsets".into(), serde_json::json!([0, 4]));
    header_map.insert("layer.weight".into(), w_info.into());

    let mut s_info = serde_json::Map::new();
    s_info.insert("dtype".into(), "F32".into());
    s_info.insert("shape".into(), serde_json::json!([1]));
    s_info.insert("data_offsets".into(), serde_json::json!([4, 8]));
    header_map.insert("layer.weight_scale".into(), s_info.into());

    let mut n_info = serde_json::Map::new();
    n_info.insert("dtype".into(), "BF16".into());
    n_info.insert("shape".into(), serde_json::json!([1]));
    n_info.insert("data_offsets".into(), serde_json::json!([8, 10]));
    header_map.insert("norm.weight".into(), n_info.into());

    let header_json = serde_json::to_string(&header_map).unwrap();
    let header_bytes = header_json.as_bytes();

    // CAST: usize → u64, header length fits in u64
    #[allow(clippy::as_conversions)]
    let header_len = header_bytes.len() as u64;
    let mut file_bytes = Vec::new();
    file_bytes.extend_from_slice(&header_len.to_le_bytes());
    file_bytes.extend_from_slice(header_bytes);
    file_bytes.extend_from_slice(&fp8_data);
    file_bytes.extend_from_slice(&scale_data);
    file_bytes.extend_from_slice(&norm_data);

    std::fs::write(&path, &file_bytes).unwrap();
    (dir, path)
}

// ---------------------------------------------------------------------------
// Parse subcommand
// ---------------------------------------------------------------------------

#[test]
fn cli_parse_subcommand() {
    let (_dir, fixture) = create_test_fixture();

    let output = Command::new(binary_path())
        .args(["parse", fixture.to_str().unwrap()])
        .output()
        .expect("failed to run binary");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "parse failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(stdout.contains("3 tensors parsed"), "stdout: {stdout}");
    assert!(stdout.contains("quantized"), "stdout: {stdout}");
    assert!(stdout.contains("passthrough"), "stdout: {stdout}");
}

// ---------------------------------------------------------------------------
// Inspect subcommand
// ---------------------------------------------------------------------------

#[test]
fn cli_inspect_subcommand() {
    let (_dir, fixture) = create_test_fixture();

    let output = Command::new(binary_path())
        .args(["inspect", fixture.to_str().unwrap()])
        .output()
        .expect("failed to run binary");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "inspect failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(stdout.contains("Format:"), "stdout: {stdout}");
    assert!(stdout.contains("FP8"), "stdout: {stdout}");
    assert!(stdout.contains("Passthrough:"), "stdout: {stdout}");
}

#[test]
fn cli_info_alias() {
    let (_dir, fixture) = create_test_fixture();

    let output = Command::new(binary_path())
        .args(["info", fixture.to_str().unwrap()])
        .output()
        .expect("failed to run binary");

    assert!(
        output.status.success(),
        "info alias failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

// ---------------------------------------------------------------------------
// Remember subcommand
// ---------------------------------------------------------------------------

#[test]
fn cli_remember_subcommand() {
    let (dir, fixture) = create_test_fixture();
    let output_path = dir.path().join("test-bf16.safetensors");

    let output = Command::new(binary_path())
        .args([
            "remember",
            fixture.to_str().unwrap(),
            "--output",
            output_path.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run binary");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "remember failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(stdout.contains("Parsing..."), "stdout: {stdout}");
    assert!(stdout.contains("Output:"), "stdout: {stdout}");
    assert!(output_path.exists(), "output file not created");
}

#[test]
fn cli_dequantize_alias() {
    let (dir, fixture) = create_test_fixture();
    let output_path = dir.path().join("test-bf16.safetensors");

    let output = Command::new(binary_path())
        .args([
            "dequantize",
            fixture.to_str().unwrap(),
            "--output",
            output_path.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run binary");

    assert!(
        output.status.success(),
        "dequantize alias failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output_path.exists(), "output file not created");
}

/// `remember --to f32|f16` must reach the file, not just be accepted.
///
/// Asserts the *payload width* rather than the file size: a safetensors header
/// is JSON whose length varies with the dtype string, so comparing file sizes
/// would be off by a handful of bytes for reasons that have nothing to do with
/// the tensor data. Reading the header back and checking the declared dtype is
/// what actually pins the contract.
#[test]
fn cli_remember_honours_every_output_dtype() {
    for (flag, want_dtype, want_elem_bytes) in [
        ("bf16", "BF16", 2_usize),
        ("f32", "F32", 4),
        ("f16", "F16", 2),
    ] {
        let (dir, fixture) = create_test_fixture();
        let output_path = dir.path().join(format!("out-{flag}.safetensors"));

        let output = Command::new(binary_path())
            .args([
                "remember",
                fixture.to_str().unwrap(),
                "--to",
                flag,
                "--output",
                output_path.to_str().unwrap(),
            ])
            .output()
            .expect("failed to run binary");

        assert!(
            output.status.success(),
            "remember --to {flag} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output_path.exists(), "--to {flag}: no output file");

        // Re-parse and confirm the dequantised tensor really carries the
        // requested width, header and payload agreeing.
        let model = anamnesis::parse(&output_path).expect("re-parse output");
        let quantised: Vec<_> = model
            .header
            .tensors
            .iter()
            .filter(|t| t.dtype.to_string() == want_dtype)
            .collect();
        assert!(
            !quantised.is_empty(),
            "--to {flag}: no tensor declared {want_dtype}"
        );
        for t in quantised {
            let elements: usize = t.shape.iter().product();
            assert_eq!(
                t.byte_len(),
                elements * want_elem_bytes,
                "--to {flag}: tensor `{}` payload is not {want_elem_bytes} B/element",
                t.name
            );
        }
    }
}

/// The size the summary line reports must be the size of the file it just
/// wrote, at the width the caller asked for.
///
/// Regression guard for a defect found by running the documented command rather
/// than by reading the code: `run_remember_safetensors` built its `InspectInfo`
/// with `From<&SafetensorsHeader>`, which is hard-wired to the `BF16` default,
/// so `--to f32` printed the `BF16` estimate. The file was correct; only the
/// number beside it was wrong, which is the more insidious of the two.
///
/// The expected figures are arithmetic, not golden: the fixture holds 4
/// dequantised `FP8` elements plus a 1-element `BF16` passthrough norm, so the
/// estimate is `4 × E::BYTES + 2`. The `F32` line must therefore differ from
/// the `BF16` one, which is precisely what the bug prevented.
#[test]
fn cli_remember_reports_the_size_of_the_dtype_it_wrote() {
    for (flag, want) in [("bf16", "10 B"), ("f32", "18 B"), ("f16", "10 B")] {
        let (dir, fixture) = create_test_fixture();
        let output_path = dir.path().join(format!("sized-{flag}.safetensors"));

        let output = Command::new(binary_path())
            .args([
                "remember",
                fixture.to_str().unwrap(),
                "--to",
                flag,
                "--output",
                output_path.to_str().unwrap(),
            ])
            .output()
            .expect("failed to run binary");
        assert!(output.status.success(), "remember --to {flag} failed");

        let stdout = String::from_utf8_lossy(&output.stdout);
        let summary = stdout
            .lines()
            .find(|l| l.starts_with("Output:"))
            .unwrap_or_else(|| panic!("--to {flag}: no `Output:` line in\n{stdout}"));
        assert!(
            summary.contains(&format!("({want})")),
            "--to {flag}: summary must report {want}, got `{summary}`"
        );
    }
}

/// The dtype must also reach the derived filename, since `derive_output_path`
/// builds the suffix from `TargetDtype`'s `Display`. A silent `-bf16` suffix on
/// an `F32` file would be exactly the failure v0.7.3's design note warned about.
#[test]
fn cli_remember_derives_the_dtype_suffix() {
    for (flag, want_suffix) in [("bf16", "-bf16"), ("f32", "-f32"), ("f16", "-f16")] {
        let (dir, fixture) = create_test_fixture();
        let output = Command::new(binary_path())
            .args(["remember", fixture.to_str().unwrap(), "--to", flag])
            .current_dir(dir.path())
            .output()
            .expect("failed to run binary");
        assert!(
            output.status.success(),
            "remember --to {flag} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains(want_suffix),
            "--to {flag}: expected `{want_suffix}` in the derived path; stdout: {stdout}"
        );
    }
}

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

#[test]
fn cli_nonexistent_file() {
    let output = Command::new(binary_path())
        .args(["parse", "/tmp/nonexistent_anamnesis_cli_test.safetensors"])
        .output()
        .expect("failed to run binary");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("error:"), "stderr: {stderr}");
}

#[test]
fn cli_unsupported_target_dtype() {
    let (_dir, fixture) = create_test_fixture();

    let output = Command::new(binary_path())
        .args(["remember", fixture.to_str().unwrap(), "--to", "int8"])
        .output()
        .expect("failed to run binary");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("error:"), "stderr: {stderr}");
}

#[test]
fn cli_no_subcommand_shows_help() {
    let output = Command::new(binary_path())
        .output()
        .expect("failed to run binary");

    // clap exits with error when no subcommand is given
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("Usage") || stderr.contains("usage"),
        "stderr: {stderr}"
    );
}

#[test]
fn cli_version_flag() {
    let output = Command::new(binary_path())
        .args(["--version"])
        .output()
        .expect("failed to run binary");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("anamnesis"), "stdout: {stdout}");
    assert!(
        stdout.contains(env!("CARGO_PKG_VERSION")),
        "stdout: {stdout}"
    );
}

/// `amn remember` accepts an `NPZ`, which it refused until v0.7.6 while
/// `amn convert --to safetensors` on the same file did exactly the same job.
///
/// The verb has to mean one thing across all four formats, or a Python binding
/// inherits four meanings. `--to` is accepted and vacuous here, resolved the
/// same way the `.pth` arm resolves it: nothing in an `NPZ` is quantised, so
/// there is nothing to narrow or widen, and refusing a dtype on a file that is
/// already that dtype would be hostile.
#[cfg(feature = "npz")]
#[test]
fn cli_remember_accepts_npz_and_matches_convert() {
    let dir = tempfile::tempdir().expect("tempdir");
    let via_remember = dir.path().join("remember.safetensors");
    let via_convert = dir.path().join("convert.safetensors");

    let out = Command::new(binary_path())
        .args([
            "remember",
            "tests/fixtures/npz_reference/gemma_scope_small.npz",
            "-o",
            via_remember.to_str().expect("path"),
        ])
        .output()
        .expect("run amn remember");
    assert!(
        out.status.success(),
        "remember on an NPZ failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let out = Command::new(binary_path())
        .args([
            "convert",
            "tests/fixtures/npz_reference/gemma_scope_small.npz",
            "--to",
            "safetensors",
            "-o",
            via_convert.to_str().expect("path"),
        ])
        .output()
        .expect("run amn convert");
    assert!(
        out.status.success(),
        "convert on an NPZ failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    assert_eq!(
        std::fs::read(&via_remember).expect("read remember output"),
        std::fs::read(&via_convert).expect("read convert output"),
        "remember and convert must produce the same file for an NPZ input"
    );
}

/// The vacuous dtype is accepted, and a value that is not a dtype at all is
/// still rejected.
#[cfg(feature = "npz")]
#[test]
fn cli_remember_npz_accepts_a_vacuous_dtype_but_not_nonsense() {
    let dir = tempfile::tempdir().expect("tempdir");
    for flag in ["bf16", "f32", "f16", "safetensors"] {
        let out_path = dir.path().join(format!("{flag}.safetensors"));
        let out = Command::new(binary_path())
            .args([
                "remember",
                "tests/fixtures/npz_reference/gemma_scope_small.npz",
                "--to",
                flag,
                "-o",
                out_path.to_str().expect("path"),
            ])
            .output()
            .expect("run amn remember");
        assert!(
            out.status.success(),
            "remember --to {flag} on an NPZ failed"
        );
    }

    let out = Command::new(binary_path())
        .args([
            "remember",
            "tests/fixtures/npz_reference/gemma_scope_small.npz",
            "--to",
            "int8",
        ])
        .output()
        .expect("run amn remember");
    assert!(
        !out.status.success(),
        "a value that is not an output dtype must still be rejected"
    );
}
