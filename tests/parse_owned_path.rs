// SPDX-License-Identifier: MIT OR Apache-2.0

//! Parity + safety tests for the copy-based (`no-mmap`) parse paths added in
//! Phase 6.13 Step 1.
//!
//! Each format is parsed three ways — path/mmap, `parse_*_bytes` (owned), and
//! `parse_*_from_reader` (owned) — and the results must be identical, proving
//! the `Backing` plumbing reads the same bytes regardless of origin. Plus:
//! malformed bytes yield a clean `Err` (never a panic), and a tightened
//! `ParseLimits` rejects an oversized read.
//!
//! `NPZ` joined at v0.7.6 (Phase 7.6 item 3). Until then it had no owned entry
//! points at all, so the contract this file exists to pin had one format
//! missing — the format whose Python-side callers most often hold bytes rather
//! than a path.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::wildcard_enum_match_arm
)]

use std::fs;

use anamnesis::ParseLimits;

// ---------------------------------------------------------------------------
// safetensors (always-on)
// ---------------------------------------------------------------------------

const ST_FP8: &str = "tests/fixtures/safetensors_reference/fp8.safetensors";

#[test]
fn safetensors_owned_paths_match_mmap() {
    use anamnesis::{TargetDtype, parse, parse_bytes, parse_from_reader};

    let bytes = fs::read(ST_FP8).expect("read fp8 fixture");

    let m_path = parse(ST_FP8).expect("path parse");
    let m_bytes = parse_bytes(bytes).expect("bytes parse");
    let m_reader = parse_from_reader(fs::File::open(ST_FP8).expect("open")).expect("reader parse");

    // Header parity.
    let header = format!("{:?}", m_path.inspect());
    assert_eq!(
        header,
        format!("{:?}", m_bytes.inspect()),
        "bytes inspect differs"
    );
    assert_eq!(
        header,
        format!("{:?}", m_reader.inspect()),
        "reader inspect differs"
    );

    // Full tensor-data parity, through dequantisation to BF16.
    let d_path = m_path
        .remember_to_bytes(TargetDtype::BF16)
        .expect("remember path");
    let d_bytes = m_bytes
        .remember_to_bytes(TargetDtype::BF16)
        .expect("remember bytes");
    let d_reader = m_reader
        .remember_to_bytes(TargetDtype::BF16)
        .expect("remember reader");
    assert_eq!(d_path, d_bytes, "bytes-path dequant differs from mmap");
    assert_eq!(d_path, d_reader, "reader-path dequant differs from mmap");
    assert!(!d_path.is_empty());
}

#[test]
fn safetensors_malformed_bytes_is_clean_err() {
    use anamnesis::parse_bytes;
    // The first 8 bytes decode to an absurd header length → rejected by the cap,
    // not a panic.
    assert!(parse_bytes(b"not a real safetensors file, only bytes".to_vec()).is_err());
    // Empty input: header length prefix cannot be read.
    assert!(parse_bytes(Vec::new()).is_err());
}

#[test]
fn safetensors_reader_respects_max_single_alloc() {
    use anamnesis::parse_from_reader_with_limits;
    // The fixture is 96 bytes; an 8-byte ceiling must reject the read.
    let limits = ParseLimits::default().with_max_single_alloc(8);
    let Err(err) = parse_from_reader_with_limits(fs::File::open(ST_FP8).expect("open"), &limits)
    else {
        panic!("oversized read should be rejected, not OOM");
    };
    assert!(
        matches!(err, anamnesis::AnamnesisError::LimitExceeded { limit, .. } if limit == "max_single_alloc_bytes"),
        "expected LimitExceeded(max_single_alloc_bytes), got: {err}"
    );
}

// ---------------------------------------------------------------------------
// PyTorch .pth
// ---------------------------------------------------------------------------

#[cfg(feature = "pth")]
mod pth {
    use super::{ParseLimits, fs};
    use anamnesis::{ParsedPth, parse_pth, parse_pth_bytes, parse_pth_from_reader};

    const PTH: &str = "tests/fixtures/pth_reference/algzoo_rnn_small.pth";

    /// `(name, data)` for every tensor, sorted — the parity fingerprint.
    fn fingerprint(p: &ParsedPth) -> Vec<(String, Vec<u8>)> {
        let mut v: Vec<(String, Vec<u8>)> = p
            .tensors()
            .expect("tensors")
            .into_iter()
            .map(|t| (t.name, t.data.into_owned()))
            .collect();
        v.sort();
        v
    }

    #[test]
    fn pth_owned_paths_match_mmap() {
        let bytes = fs::read(PTH).expect("read pth fixture");

        let p_path = parse_pth(PTH).expect("path parse");
        let p_bytes = parse_pth_bytes(bytes).expect("bytes parse");
        let p_reader =
            parse_pth_from_reader(fs::File::open(PTH).expect("open")).expect("reader parse");

        let header = format!("{:?}", p_path.inspect());
        assert_eq!(header, format!("{:?}", p_bytes.inspect()));
        assert_eq!(header, format!("{:?}", p_reader.inspect()));

        let fp = fingerprint(&p_path);
        assert!(!fp.is_empty(), "fixture should have tensors");
        assert_eq!(
            fp,
            fingerprint(&p_bytes),
            "bytes-path tensors differ from mmap"
        );
        assert_eq!(
            fp,
            fingerprint(&p_reader),
            "reader-path tensors differ from mmap"
        );
    }

    #[test]
    fn pth_malformed_bytes_is_clean_err() {
        assert!(parse_pth_bytes(b"garbage, not a zip archive".to_vec()).is_err());
        assert!(parse_pth_bytes(Vec::new()).is_err());
    }

    #[test]
    fn pth_reader_respects_max_single_alloc() {
        use anamnesis::parse_pth_from_reader_with_limits;
        let limits = ParseLimits::default().with_max_single_alloc(8);
        let Err(err) =
            parse_pth_from_reader_with_limits(fs::File::open(PTH).expect("open"), &limits)
        else {
            panic!("oversized read should be rejected");
        };
        assert!(
            matches!(err, anamnesis::AnamnesisError::LimitExceeded { limit, .. } if limit == "max_single_alloc_bytes"),
            "expected LimitExceeded(max_single_alloc_bytes), got: {err}"
        );
    }
}

// ---------------------------------------------------------------------------
// GGUF
// ---------------------------------------------------------------------------
//
// No GGUF fixture is committed (real models are large / local-only), so the
// parity test synthesises a minimal valid GGUF in memory via the crate's own
// writer (`write_gguf_to_writer`). It is therefore self-contained and **always
// runs in CI** — unlike an `exists()`-guarded fixture, which would silently
// no-op the parity check on a fresh checkout.

#[cfg(feature = "gguf")]
mod gguf {
    use super::fs;
    use std::collections::HashMap;
    use std::io::Cursor;

    use anamnesis::{
        GgufType, GgufWriteTensor, ParsedGguf, parse_gguf, parse_gguf_bytes,
        parse_gguf_from_reader, write_gguf_to_writer,
    };

    /// `(name, data)` for every tensor, sorted — the parity fingerprint.
    fn fingerprint(p: &ParsedGguf) -> Vec<(String, Vec<u8>)> {
        let mut v: Vec<(String, Vec<u8>)> = p
            .tensors()
            .map(|t| (t.name.to_string(), t.data.into_owned()))
            .collect();
        v.sort();
        v
    }

    /// A minimal valid GGUF (two tensors of distinct dtypes), produced by the
    /// crate's own encoder so the parity test needs no on-disk fixture. The byte
    /// values are arbitrary — parity only compares byte-for-byte across paths.
    fn synthesize_gguf() -> Vec<u8> {
        let f32_bytes: Vec<u8> = (0u32..6).flat_map(u32::to_le_bytes).collect();
        let i32_bytes: Vec<u8> = (0i32..4).flat_map(i32::to_le_bytes).collect();
        let f32_shape = [2usize, 3];
        let i32_shape = [4usize];
        let tensors = [
            GgufWriteTensor {
                name: "w.f32",
                shape: &f32_shape,
                dtype: GgufType::F32,
                data: &f32_bytes,
            },
            GgufWriteTensor {
                name: "w.i32",
                shape: &i32_shape,
                dtype: GgufType::I32,
                data: &i32_bytes,
            },
        ];
        let mut cursor = Cursor::new(Vec::new());
        write_gguf_to_writer(&mut cursor, &tensors, &HashMap::new()).expect("write gguf");
        cursor.into_inner()
    }

    #[test]
    fn gguf_owned_paths_match_mmap() {
        let bytes = synthesize_gguf();

        // The path/mmap entry point needs a file on disk.
        let tmp = tempfile::NamedTempFile::new().expect("temp file");
        fs::write(tmp.path(), &bytes).expect("stage gguf");

        let p_path = parse_gguf(tmp.path()).expect("path parse");
        let p_bytes = parse_gguf_bytes(bytes.clone()).expect("bytes parse");
        let p_reader = parse_gguf_from_reader(Cursor::new(bytes)).expect("reader parse");

        let header = format!("{:?}", p_path.inspect());
        assert_eq!(header, format!("{:?}", p_bytes.inspect()));
        assert_eq!(header, format!("{:?}", p_reader.inspect()));

        let fp = fingerprint(&p_path);
        assert!(!fp.is_empty(), "synthesised gguf should have tensors");
        assert_eq!(
            fp,
            fingerprint(&p_bytes),
            "bytes-path tensors differ from mmap"
        );
        assert_eq!(
            fp,
            fingerprint(&p_reader),
            "reader-path tensors differ from mmap"
        );
    }

    #[test]
    fn gguf_malformed_bytes_is_clean_err() {
        assert!(parse_gguf_bytes(b"XXXX not a gguf file".to_vec()).is_err());
        assert!(parse_gguf_bytes(Vec::new()).is_err());
    }
}

// ---------------------------------------------------------------------------
// NPZ (feature-gated)
// ---------------------------------------------------------------------------

#[cfg(feature = "npz")]
mod npz {
    use super::{ParseLimits, fs};
    use anamnesis::{
        NpzTensor, parse_npz, parse_npz_bytes, parse_npz_from_reader,
        parse_npz_from_reader_with_limits,
    };
    use std::collections::HashMap;

    const NPZ_FIXTURE: &str = "tests/fixtures/npz_reference/gemma_scope_small.npz";

    /// Sorted `(name, shape, dtype, data)` so two maps compare independently of
    /// `HashMap` iteration order.
    fn fingerprint(map: &HashMap<String, NpzTensor>) -> Vec<(String, Vec<usize>, String, Vec<u8>)> {
        let mut out: Vec<_> = map
            .values()
            .map(|t| {
                (
                    t.name.clone(),
                    t.shape.clone(),
                    t.dtype.to_string(),
                    t.data.clone(),
                )
            })
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    /// The three `NPZ` entry points must return identical arrays.
    ///
    /// `NPZ` had no owned paths at all before v0.7.6, which is why this suite
    /// covered three formats and not four: the copy-based untrusted-input
    /// contract Phase 6.13 documents simply had no `NPZ` instance.
    #[test]
    fn npz_owned_paths_match_path() {
        let bytes = fs::read(NPZ_FIXTURE).expect("read npz fixture");

        let m_path = parse_npz(NPZ_FIXTURE).expect("path parse");
        let m_bytes = parse_npz_bytes(bytes).expect("bytes parse");
        let m_reader = parse_npz_from_reader(fs::File::open(NPZ_FIXTURE).expect("open"))
            .expect("reader parse");

        let fp = fingerprint(&m_path);
        assert!(!fp.is_empty(), "the fixture should hold arrays");
        assert_eq!(fp, fingerprint(&m_bytes), "bytes path differs from path");
        assert_eq!(fp, fingerprint(&m_reader), "reader path differs from path");
    }

    #[test]
    fn npz_malformed_bytes_is_clean_err() {
        assert!(parse_npz_bytes(b"XXXX not a zip archive".to_vec()).is_err());
        assert!(parse_npz_bytes(Vec::new()).is_err());
    }

    /// A tightened budget rejects the streamed read before the archive is
    /// buffered, the same way the safetensors and `.pth` reader paths do.
    #[test]
    fn npz_reader_honours_a_tight_budget() {
        let limits = ParseLimits::default().with_max_single_alloc(16);
        let err =
            parse_npz_from_reader_with_limits(fs::File::open(NPZ_FIXTURE).expect("open"), &limits)
                .expect_err("a 16-byte budget cannot admit the archive");
        assert!(
            matches!(err, anamnesis::AnamnesisError::LimitExceeded { .. }),
            "expected LimitExceeded, got {err:?}"
        );
    }
}
