// SPDX-License-Identifier: MIT OR Apache-2.0

//! `NPZ`/`NPY` archive parsing — zero-copy bulk read for near-I/O-speed extraction.
//!
//! This module implements a lean `NPY` header parser and bulk data reader that
//! bypasses per-element deserialization entirely. For little-endian data on
//! little-endian machines (>99% of ML files on x86/ARM), the raw bytes in the
//! `NPY` file ARE the correct in-memory representation — no per-element
//! processing is needed.
//!
//! The `ZIP` container is parsed by the vendored, read-only central-directory
//! reader (`crate::parse::zip`, Phase 6.12); `DEFLATE` entries are inflated by
//! `flate2` / `miniz_oxide`. `NPZ` archives typically use the `STORE` method
//! (no compression) for large arrays, making the `ZIP` layer a pure
//! passthrough.
//!
//! # Performance
//!
//! On a 302 MB `NPZ` file (Gemma Scope 2B SAE, 5 `F32` arrays):
//! - Raw `fs::read`: ~64 ms (I/O baseline)
//! - This parser: near I/O baseline (bulk `read_exact`, zero per-element work)
//! - Previous `npyz`-backed parser: ~1,500 ms (per-element deserialization)

use std::collections::HashMap;
use std::fmt;
use std::io::{Read, Seek};
use std::path::Path;

use crate::ParseLimits;
use crate::error::AnamnesisError;
use crate::limits::Budget;
use crate::parse::utils::{PREALLOC_SOFT_CAP, byteswap_inplace};

// ---------------------------------------------------------------------------
// NPY magic
// ---------------------------------------------------------------------------

/// `NPY` magic bytes: `\x93NUMPY`.
const NPY_MAGIC: &[u8; 6] = b"\x93NUMPY";

/// Upper bound on an `NPY` header's declared length, in bytes (1 MiB).
///
/// `NPY` v1 stores `header_len` as a `u16` (≤64 KiB — already safe by datatype),
/// but v2/v3 store it as a `u32`, which an adversarial file can set up to 4 GiB
/// to drive a `vec![0u8; header_len]` allocation. Real `NPY` headers are <1 KiB;
/// 1 MiB is generous head-room for any plausible legitimate file while rejecting
/// the 4 GiB declared-header window before allocating.
const NPY_MAX_HEADER_BYTES: usize = 1 << 20;

/// Upper bound on a single `NPY`/`NPZ` array's raw byte length (8 GiB).
///
/// The element-count and byte-count `checked_mul`s in [`read_array_data`]
/// already reject the `usize`-overflow case, but rejection there only happens
/// at the arithmetic-overflow boundary. This cap makes rejection deterministic
/// well below overflow: 8 GiB matches the largest real tensors on consumer GPUs
/// while rejecting absurd declared shapes. Mirrors the bounded-array-element
/// pattern recommended in `candle #3533`. A `u64` so the literal compiles on
/// 32-bit targets, where `1usize << 33` would overflow.
const NPZ_MAX_ARRAY_BYTES: u64 = 1 << 33;

// ---------------------------------------------------------------------------
// NpzDtype
// ---------------------------------------------------------------------------

/// Element data type for tensors parsed from `NPZ`/`NPY` archives.
///
/// Includes `BF16` which `NumPy` cannot represent natively. When a `BF16`
/// tensor is detected (stored as void/`V2` by `JAX`), anamnesis reads the
/// raw bytes and interprets them as `half::bf16`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum NpzDtype {
    /// Boolean (1 byte per element).
    Bool,
    /// Unsigned 8-bit integer.
    U8,
    /// Signed 8-bit integer.
    I8,
    /// Unsigned 16-bit integer.
    U16,
    /// Signed 16-bit integer.
    I16,
    /// Unsigned 32-bit integer.
    U32,
    /// Signed 32-bit integer.
    I32,
    /// Unsigned 64-bit integer.
    U64,
    /// Signed 64-bit integer.
    I64,
    /// 16-bit IEEE 754 half-precision (`f2` in `NumPy`).
    F16,
    /// 16-bit brain floating point (`BF16`).
    ///
    /// `NumPy` has no native `BF16` dtype. `JAX` stores `BF16` as void (`V2`)
    /// with `bfloat16` metadata. This variant represents that interpretation.
    BF16,
    /// 32-bit IEEE 754 single-precision.
    F32,
    /// 64-bit IEEE 754 double-precision.
    F64,
}

impl NpzDtype {
    /// Returns the number of bytes per element for this dtype.
    #[must_use]
    pub const fn byte_size(self) -> usize {
        match self {
            Self::Bool | Self::U8 | Self::I8 => 1,
            Self::U16 | Self::I16 | Self::F16 | Self::BF16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 | Self::I64 | Self::F64 => 8,
        }
    }
}

/// Displays the canonical uppercase name (e.g., `"F32"`, `"BF16"`, `"BOOL"`).
///
/// This is the string used in inspection output and cross-validation tests.
impl fmt::Display for NpzDtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Bool => "BOOL",
            Self::U8 => "U8",
            Self::I8 => "I8",
            Self::U16 => "U16",
            Self::I16 => "I16",
            Self::U32 => "U32",
            Self::I32 => "I32",
            Self::U64 => "U64",
            Self::I64 => "I64",
            Self::F16 => "F16",
            Self::BF16 => "BF16",
            Self::F32 => "F32",
            Self::F64 => "F64",
        };
        f.write_str(s)
    }
}

// ---------------------------------------------------------------------------
// NpzTensor
// ---------------------------------------------------------------------------

/// A single tensor extracted from an `NPZ` archive.
///
/// Contains the tensor name, shape, dtype, and raw byte data in little-endian,
/// row-major (C) order. Framework consumers can interpret `data` directly
/// according to `dtype` and `shape`.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct NpzTensor {
    /// Tensor name as stored in the archive (without `.npy` extension).
    /// Matches the `HashMap` key returned by [`parse_npz`].
    pub name: String,
    /// Tensor dimensions (e.g., `[16384, 2304]`).
    pub shape: Vec<usize>,
    /// Element data type (e.g., `F32`, `BF16`).
    pub dtype: NpzDtype,
    /// Raw bytes in row-major (C) order, little-endian.
    /// Length equals `product(shape) × dtype.byte_size()`.
    pub data: Vec<u8>,
}

impl NpzTensor {
    /// Builds a tensor from its parts.
    ///
    /// The construction path for callers that *synthesise* `NPZ` tensors to
    /// feed `npz_to_safetensors` (or the `_bytes` form), rather than obtaining
    /// them from [`parse_npz`]. It exists because the struct is
    /// `#[non_exhaustive]` since v0.7.6: a literal cannot be written from
    /// outside the crate, and this constructor absorbs any field added later
    /// without breaking the callers that build one.
    ///
    /// The arguments are moved, not validated: `data` is trusted to be
    /// `product(shape) × dtype.byte_size()` bytes in little-endian, row-major
    /// (C) order, exactly as [`parse_npz`] produces it. A mismatch surfaces
    /// downstream, where the safetensors writer checks it.
    #[must_use]
    pub const fn new(name: String, shape: Vec<usize>, dtype: NpzDtype, data: Vec<u8>) -> Self {
        Self {
            name,
            shape,
            dtype,
            data,
        }
    }
}

// ---------------------------------------------------------------------------
// NPY header parsing
// ---------------------------------------------------------------------------

/// Parsed `NPY` header: dtype, endianness, memory order, and shape.
struct NpyHeader {
    /// Parsed element data type.
    dtype: NpzDtype,
    /// `true` if the data is stored big-endian (descr prefix `>`).
    big_endian: bool,
    /// `true` if the data is in Fortran (column-major) order.
    fortran_order: bool,
    /// Array shape (e.g., `[16384, 2304]`).
    shape: Vec<usize>,
}

/// Parses the `NPY` header from a reader, consuming the header bytes.
///
/// Supports `NPY` format versions 1.0, 2.0, and 3.0. The declared header
/// length is bounded by both `NPY_MAX_HEADER_BYTES` and the caller's `budget`
/// (per-item single-allocation cap + cumulative aggregate) before the header
/// buffer is allocated.
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if the magic bytes, version, or header
/// dict are malformed, or the declared header length exceeds the cap or the
/// `budget`.
fn parse_npy_header(reader: &mut impl Read, budget: &mut Budget) -> crate::Result<NpyHeader> {
    // Read magic (6 bytes) + major (1) + minor (1) = 8 bytes.
    let mut preamble = [0u8; 8];
    reader
        .read_exact(&mut preamble)
        .map_err(|e| AnamnesisError::Parse {
            reason: format!("NPY preamble read failed: {e}"),
        })?;

    // INDEX: preamble is exactly 8 bytes, slicing [..6] is safe
    #[allow(clippy::indexing_slicing)]
    if &preamble[..6] != NPY_MAGIC {
        return Err(AnamnesisError::Parse {
            reason: "invalid NPY magic bytes".into(),
        });
    }

    // INDEX: preamble[6] is safe (8-byte array)
    // EXPLICIT: preamble[7] (minor version) is read but unused — the NPY spec
    // defines only minor version 0 for all major versions (1, 2, 3).
    #[allow(clippy::indexing_slicing)]
    let major = preamble[6];

    // Read header length (version-dependent).
    let header_len: usize = match major {
        1 => {
            let mut buf = [0u8; 2];
            reader
                .read_exact(&mut buf)
                .map_err(|e| AnamnesisError::Parse {
                    reason: format!("NPY v1 header length read failed: {e}"),
                })?;
            usize::from(u16::from_le_bytes(buf))
        }
        2 | 3 => {
            let mut buf = [0u8; 4];
            reader
                .read_exact(&mut buf)
                .map_err(|e| AnamnesisError::Parse {
                    reason: format!("NPY v{major} header length read failed: {e}"),
                })?;
            // CAST: u32 → usize, NPY headers are always small
            #[allow(clippy::as_conversions)]
            let len = u32::from_le_bytes(buf) as usize;
            len
        }
        _ => {
            return Err(AnamnesisError::Unsupported {
                format: "NPY".into(),
                detail: format!("unsupported NPY version {major}"),
            });
        }
    };

    // Reject an adversarial declared header length before allocating. The
    // v2/v3 `u32` length can reach 4 GiB; legitimate headers are <1 KiB.
    if header_len > NPY_MAX_HEADER_BYTES {
        return Err(AnamnesisError::LimitExceeded {
            limit: "NPY_MAX_HEADER_BYTES",
            message: format!(
                "NPY header length {header_len} bytes exceeds the \
                 {NPY_MAX_HEADER_BYTES}-byte cap"
            ),
        });
    }
    // Caller-supplied ceiling (per-item single-alloc + cumulative aggregate),
    // layered on top of the permanent cap above.
    // CAST: usize → u64, lossless widening on all supported targets
    #[allow(clippy::as_conversions)]
    budget.charge_alloc(header_len as u64, "NPY header")?;

    // Read header string.
    let mut header_buf = vec![0u8; header_len];
    reader
        .read_exact(&mut header_buf)
        .map_err(|e| AnamnesisError::Parse {
            reason: format!("NPY header data read failed: {e}"),
        })?;

    let header_str = std::str::from_utf8(&header_buf).map_err(|e| AnamnesisError::Parse {
        reason: format!("NPY header is not valid UTF-8: {e}"),
    })?;

    // Parse the Python dict literal.
    let (dtype, big_endian) = extract_descr(header_str)?;
    let fortran_order = extract_fortran_order(header_str);
    let shape = extract_shape(header_str)?;

    Ok(NpyHeader {
        dtype,
        big_endian,
        fortran_order,
        shape,
    })
}

/// Extracts the `descr` field from the `NPY` header dict and maps it to
/// `(NpzDtype, is_big_endian)`.
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if the descr field is missing.
/// Returns [`AnamnesisError::Unsupported`] if the dtype string is not recognized.
fn extract_descr(header: &str) -> crate::Result<(NpzDtype, bool)> {
    // Find 'descr': then extract the quoted value after it.
    let descr_start = header.find("'descr'").or_else(|| header.find("\"descr\""));
    let descr_start = descr_start.ok_or_else(|| AnamnesisError::Parse {
        reason: "NPY header missing 'descr' field".into(),
    })?;

    // Skip past 'descr': to find the value.
    let after_key = header
        .get(descr_start..)
        .and_then(|s| s.find(':').map(|i| descr_start + i + 1))
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "NPY header 'descr' field has no value".into(),
        })?;

    let value_str = header
        .get(after_key..)
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "NPY header truncated after 'descr'".into(),
        })?;

    // Extract the string between quotes. Detect the quote character from the
    // first quote found in the value portion (not the entire header tail),
    // so mixed-quote headers like {'descr': "<f4", 'other': ...} work.
    let trimmed = value_str.trim_start();
    let quote_char = match trimmed.as_bytes().first() {
        Some(b'\'') => '\'',
        Some(b'"') => '"',
        _ => {
            return Err(AnamnesisError::Parse {
                reason: "NPY header 'descr' value not quoted".into(),
            });
        }
    };
    let inner = trimmed.get(1..).ok_or_else(|| AnamnesisError::Parse {
        reason: "NPY header 'descr' value truncated after opening quote".into(),
    })?;
    let closing = inner
        .find(quote_char)
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "NPY header 'descr' value missing closing quote".into(),
        })?;
    let descr = inner.get(..closing).ok_or_else(|| AnamnesisError::Parse {
        reason: "NPY header 'descr' extraction failed".into(),
    })?;

    parse_descr(descr)
}

/// Maps a `NumPy` dtype descriptor string (e.g., `<f4`, `>u2`, `|V2`) to
/// `(NpzDtype, is_big_endian)`.
///
/// # Errors
///
/// Returns [`AnamnesisError::Unsupported`] if the descriptor is not recognized.
fn parse_descr(descr: &str) -> crate::Result<(NpzDtype, bool)> {
    // First character is endianness: '<' = LE, '>' = BE, '|' = N/A, '=' = native.
    // BORROW: explicit .as_bytes() to inspect endianness prefix byte
    let bytes = descr.as_bytes();
    if bytes.len() < 2 {
        return Err(AnamnesisError::Unsupported {
            format: "NPY".into(),
            detail: format!("dtype descriptor too short: '{descr}'"),
        });
    }

    // INDEX: bytes.len() >= 2 guarantees byte 0 exists; the endianness prefix in
    // any valid descriptor is ASCII (`<`, `>`, `|`, `=`), so byte 0 is the whole
    // first character (a non-ASCII first byte simply won't match `>` below).
    #[allow(clippy::indexing_slicing)]
    let endian_char = bytes[0];
    // `descr.get(1..)` — NOT `&descr[1..]`: a `str` slice panics when byte index
    // 1 is not a UTF-8 char boundary (a descriptor whose first character is a
    // multi-byte codepoint). `.get` yields `None` there → a clean error, never a
    // panic on attacker-controlled `NPY` header bytes.
    let type_str = descr.get(1..).ok_or_else(|| AnamnesisError::Unsupported {
        format: "NPY".into(),
        detail: format!("dtype descriptor is not ASCII: '{descr}'"),
    })?;

    // EXPLICIT: '=' (native endian) is treated as little-endian. All modern ML
    // platforms (x86-64, ARM64) are LE; a BE-native machine would need '>' explicitly.
    let big_endian = endian_char == b'>';

    let dtype = match type_str {
        // Boolean
        "b1" => NpzDtype::Bool,
        // Unsigned integers
        "u1" => NpzDtype::U8,
        "u2" => NpzDtype::U16,
        "u4" => NpzDtype::U32,
        "u8" => NpzDtype::U64,
        // Signed integers
        "i1" => NpzDtype::I8,
        "i2" => NpzDtype::I16,
        "i4" => NpzDtype::I32,
        "i8" => NpzDtype::I64,
        // Floats
        "f2" => NpzDtype::F16,
        "f4" => NpzDtype::F32,
        "f8" => NpzDtype::F64,
        // Void (BF16 via JAX convention)
        "V2" => NpzDtype::BF16,
        _ => {
            return Err(AnamnesisError::Unsupported {
                format: "NPY".into(),
                detail: format!("unsupported dtype descriptor '{descr}'"),
            });
        }
    };

    Ok((dtype, big_endian))
}

/// Extracts the `fortran_order` field from the `NPY` header dict.
///
/// Returns `false` if the field is not found or has any value other than
/// `True` (defaults to C-order).
// EXPLICIT: returns false (C-order) for missing or malformed fortran_order
// fields. The NPY spec mandates this field, but defaulting to C-order is
// the safe choice — Fortran-order is rejected by parse_npz anyway.
fn extract_fortran_order(header: &str) -> bool {
    // Look for 'fortran_order': True
    header
        .find("'fortran_order'")
        .or_else(|| header.find("\"fortran_order\""))
        .is_some_and(|pos| {
            header
                .get(pos..)
                .and_then(|s| s.find(':').and_then(|i| s.get(i + 1..)))
                .is_some_and(|val| val.trim_start().starts_with("True"))
        })
}

/// Extracts the `shape` field from the `NPY` header dict and parses it as
/// a tuple of `usize` dimensions.
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if the shape field is missing or
/// contains invalid dimension values.
fn extract_shape(header: &str) -> crate::Result<Vec<usize>> {
    let shape_start = header.find("'shape'").or_else(|| header.find("\"shape\""));
    let shape_start = shape_start.ok_or_else(|| AnamnesisError::Parse {
        reason: "NPY header missing 'shape' field".into(),
    })?;

    // Find the opening paren after 'shape':
    let after_key = header
        .get(shape_start..)
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "NPY header truncated at 'shape'".into(),
        })?;
    let paren_open = after_key.find('(').ok_or_else(|| AnamnesisError::Parse {
        reason: "NPY header 'shape' value missing opening paren".into(),
    })?;
    let inner_start = after_key
        .get(paren_open + 1..)
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "NPY header 'shape' truncated after paren".into(),
        })?;
    let paren_close = inner_start.find(')').ok_or_else(|| AnamnesisError::Parse {
        reason: "NPY header 'shape' value missing closing paren".into(),
    })?;
    let inner = inner_start
        .get(..paren_close)
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "NPY header 'shape' extraction failed".into(),
        })?;

    inner
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| {
            s.trim()
                .parse::<usize>()
                .map_err(|e| AnamnesisError::Parse {
                    reason: format!("NPY shape dimension parse error: {e}"),
                })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Bulk data extraction
// ---------------------------------------------------------------------------

/// Reads array data as raw little-endian bytes in one bulk `read_exact` call.
///
/// For little-endian data on a little-endian machine, the raw bytes are the
/// correct in-memory representation — zero per-element processing. For
/// big-endian data, a byte-swap pass is applied in-place after the bulk read.
///
/// `entry_size` is the ZIP entry's declared uncompressed size. The
/// shape-derived `data_bytes` is rejected if it exceeds `entry_size`
/// **before** any allocation: an entry cannot hold (or decompress to) more
/// than it declares, so a small entry claiming an enormous shape — or a
/// `DEFLATE` entry whose declared shape would balloon the allocation — fails
/// fast instead of driving a multi-`GiB` `vec!`. This mirrors the
/// `data.len() == n_blocks × type_size` cross-check the `GGUF` dequant path
/// performs, and complements the absolute `NPZ_MAX_ARRAY_BYTES` cap.
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if the element count or byte count
/// overflows `usize`, if `data_bytes` exceeds the entry's declared size, the
/// `NPZ_MAX_ARRAY_BYTES` cap, or the caller's `budget` (per-item
/// single-allocation cap + cumulative aggregate), or if the read fails.
fn read_array_data(
    reader: &mut impl Read,
    header: &NpyHeader,
    entry_size: u64,
    budget: &mut Budget,
) -> crate::Result<Vec<u8>> {
    let n_elements: usize = header
        .shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "element count overflow".into(),
        })?;

    let data_bytes = n_elements
        .checked_mul(header.dtype.byte_size())
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "data byte count overflow".into(),
        })?;

    // CAST: usize → u64, lossless widening on all supported targets
    #[allow(clippy::as_conversions)]
    let data_bytes_u64 = data_bytes as u64;
    // Reject before allocating: the declared shape cannot require more bytes
    // than the ZIP entry holds (bounds the alloc to the entry's honest size
    // and any DEFLATE expansion to it).
    if data_bytes_u64 > entry_size {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "NPY array size {data_bytes} bytes exceeds the declared ZIP \
                 entry size {entry_size} bytes"
            ),
        });
    }
    if data_bytes_u64 > NPZ_MAX_ARRAY_BYTES {
        return Err(AnamnesisError::LimitExceeded {
            limit: "NPZ_MAX_ARRAY_BYTES",
            message: format!(
                "NPY array size {data_bytes} bytes exceeds the \
                 {NPZ_MAX_ARRAY_BYTES}-byte cap"
            ),
        });
    }
    // Caller-supplied ceiling (per-item single-alloc + cumulative aggregate),
    // layered on top of the permanent caps above. The aggregate is what bounds
    // `parse_npz`'s peak heap (every array is held in the returned map at once).
    budget.charge_alloc(data_bytes_u64, "NPZ array data")?;

    let mut buf = vec![0u8; data_bytes];
    reader
        .read_exact(&mut buf)
        .map_err(|e| AnamnesisError::Parse {
            reason: format!("array data read failed ({data_bytes} bytes): {e}"),
        })?;

    // Byte-swap for big-endian data with multi-byte elements.
    if header.big_endian && header.dtype.byte_size() > 1 {
        byteswap_inplace(&mut buf, header.dtype.byte_size());
    }

    Ok(buf)
}

/// Opens a [`Read`] over an `NPZ` entry's bytes, inflating `DEFLATE` entries on
/// the fly while leaving `STORED` entries as a direct byte window.
///
/// The codec lives here, in the `.npz` consumer — the vendored container reader
/// (`crate::parse::zip`) stays codec-free and hands back the raw (possibly
/// compressed) entry bytes.
///
/// # Errors
///
/// Returns [`AnamnesisError::Unsupported`] if the entry uses a compression
/// method other than `STORED` or `DEFLATE`, or [`AnamnesisError::Parse`] /
/// [`AnamnesisError::Io`] if the entry's local header / data offset cannot be
/// resolved.
fn open_npz_entry_reader<'a, R: Read + Seek>(
    src: &'a mut crate::parse::zip::ReaderSource<R>,
    entry: &crate::parse::zip::ZipEntry,
    name: &str,
) -> crate::Result<Box<dyn Read + 'a>> {
    let raw = src.entry_data_reader(entry)?;
    // TRAIT_OBJECT: STORED and DEFLATE entry readers have different concrete
    // types; a boxed `dyn Read` unifies them for the NPY header/array parser
    // without duplicating the parse logic per compression method.
    match entry.method {
        crate::parse::zip::Compression::Stored => Ok(Box::new(raw)),
        crate::parse::zip::Compression::Deflate => {
            Ok(Box::new(flate2::read::DeflateDecoder::new(raw)))
        }
        crate::parse::zip::Compression::Unsupported(method) => Err(AnamnesisError::Unsupported {
            format: "NPZ".into(),
            detail: format!("array '{name}' uses unsupported ZIP compression method {method}"),
        }),
    }
}

// ---------------------------------------------------------------------------
// NpzTensorInfo / NpzInspectInfo (lightweight, header-only)
// ---------------------------------------------------------------------------

/// Lightweight per-tensor metadata from an `NPZ` archive.
///
/// Produced by [`inspect_npz`]. Contains only `NPY` header information —
/// no tensor data is read from the archive.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct NpzTensorInfo {
    /// Tensor name (without `.npy` extension).
    pub name: String,
    /// Tensor dimensions (e.g., `[16384, 2304]`).
    pub shape: Vec<usize>,
    /// Element data type (e.g., `F32`, `BF16`).
    pub dtype: NpzDtype,
    /// Total byte length (`product(shape) * dtype.byte_size()`).
    pub byte_len: usize,
}

/// Summary information about an `NPZ` archive, derived from headers only.
///
/// Produced by [`inspect_npz`]. No tensor data is loaded — peak memory
/// is proportional to the number of tensors (metadata only), not the
/// file size.
#[derive(Debug, Clone)]
#[non_exhaustive]
#[must_use]
pub struct NpzInspectInfo {
    /// Per-tensor metadata.
    pub tensors: Vec<NpzTensorInfo>,
    /// Total size of all tensor data in bytes.
    pub total_bytes: u64,
    /// Distinct dtypes found (in order of first occurrence).
    pub dtypes: Vec<NpzDtype>,
    /// Estimated tensor-data size in bytes after conversion, at
    /// [`output_dtype`](Self::output_dtype).
    ///
    /// **Always equal to [`total_bytes`](Self::total_bytes)**, and that is the
    /// answer rather than a placeholder: `NPZ` arrays are already full
    /// precision, so a conversion copies every array through in its **source**
    /// dtype. Present so a caller reading
    /// [`InspectSummary`](crate::InspectSummary) gets the same question
    /// answered for every format instead of having to know which formats
    /// expand.
    pub dequantized_size: u64,
    /// The output dtype [`dequantized_size`](Self::dequantized_size) assumes.
    ///
    /// Recorded, but **vacuous** for this format rather than wrong: nothing is
    /// dequantised, so the figure is the same at every width — the same sense
    /// in which `amn convert weights.npz --out-dtype f32` is accepted and
    /// widens nothing.
    pub output_dtype: crate::TargetDtype,
}

impl crate::inspect::sealed::Sealed for NpzInspectInfo {}

impl crate::InspectSummary for NpzInspectInfo {
    fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    fn current_size(&self) -> u64 {
        self.total_bytes
    }

    fn dequantized_size(&self) -> u64 {
        self.dequantized_size
    }

    fn output_dtype(&self) -> crate::TargetDtype {
        self.output_dtype
    }
}

impl fmt::Display for NpzInspectInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Format:      NPZ archive")?;
        write!(f, "\nTensors:     {}", self.tensors.len())?;
        write!(
            f,
            "\nTotal size:  {}",
            crate::inspect::format_bytes(self.total_bytes)
        )?;
        let dtype_list: String = self
            .dtypes
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        write!(f, "\nDtypes:      {dtype_list}")?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Inspects an `NPZ` archive on disk, returning metadata for all arrays
/// without reading tensor data.
///
/// Reads only `NPY` headers (~128 bytes per array) — no bulk data
/// extraction. For a 300 MB file this uses kilobytes of memory instead
/// of 300 MB.
///
/// This is a thin convenience wrapper that opens `path` as a [`std::fs::File`]
/// and delegates to [`inspect_npz_from_reader`]. Callers that need to inspect
/// an `NPZ` from any other `Read + Seek` substrate (in-memory `Cursor`,
/// HTTP-range-backed adapter, custom transport) should call
/// [`inspect_npz_from_reader`] directly.
///
/// # Errors
///
/// Returns [`AnamnesisError::Io`] if the file cannot be opened or read.
///
/// Returns [`AnamnesisError::Parse`] if the `ZIP` archive is malformed or
/// an `NPY` header is invalid.
///
/// Returns [`AnamnesisError::LimitExceeded`] if a `ZIP` entry count / name
/// length or an `NPY` header size exceeds a permanent cap (always-on).
///
/// Returns [`AnamnesisError::Unsupported`] if an array uses Fortran order
/// or an unsupported dtype.
///
/// # Memory
///
/// Allocates only per-tensor metadata (name, shape, dtype). No tensor
/// data is read or allocated. Peak memory ≈ kilobytes for typical models.
///
/// **Note:** If a shape's element count overflows `usize`, `byte_len`
/// saturates to `usize::MAX` and `total_bytes` saturates to `u64::MAX`.
/// This differs from `parse_npz`, which returns `Err` on the same overflow.
/// The distinction is intentional: `inspect_npz` is best-effort metadata
/// extraction, while `parse_npz` must validate before allocating buffers.
pub fn inspect_npz(path: impl AsRef<Path>) -> crate::Result<NpzInspectInfo> {
    inspect_npz_with_options(path, &crate::InspectOptions::new())
}

/// Inspects an `NPZ` archive on disk under a caller-supplied
/// [`InspectOptions`](crate::InspectOptions).
///
/// The path-based twin of [`inspect_npz_from_reader_with_options`], and the
/// `NPZ` counterpart of `parse_npz_with_limits`: the parse could be bounded
/// before v0.7.6, the inspect that is supposed to *precede* it could not.
///
/// # Errors
///
/// Returns [`AnamnesisError::Io`] if the file cannot be opened or read, and
/// otherwise as [`inspect_npz_from_reader_with_options`].
///
/// # Memory
///
/// As [`inspect_npz_from_reader_with_options`]: `O(n_arrays)` metadata, no
/// array data.
pub fn inspect_npz_with_options(
    path: impl AsRef<Path>,
    options: &crate::InspectOptions,
) -> crate::Result<NpzInspectInfo> {
    let file = std::fs::File::open(path.as_ref())?;
    inspect_npz_from_reader_with_options(file, options)
}

/// Inspects an `NPZ` archive from any `Read + Seek` source, returning
/// metadata for all arrays without reading tensor data.
///
/// This is the cheap first half of the **inspect-before-parse** policy gate:
/// the returned [`NpzInspectInfo`] reports `total_bytes` and `tensors.len()`,
/// which a host checks against its own budget before calling the authoritative
/// [`parse_npz_with_limits`] under a matched [`ParseLimits`].
///
/// This is the reader-generic core of [`inspect_npz`]: the path-based
/// variant is a two-line wrapper that opens a file and delegates here. By
/// accepting any `Read + Seek` substrate, callers can supply alternative
/// I/O backings — in-memory cursors (`std::io::Cursor`), shared-buffer
/// adapters, or HTTP-range-backed transports that lazily fetch only the
/// bytes they need.
///
/// # Range-read access pattern
///
/// `NPZ` is a `ZIP` archive whose central directory lives at the *end* of
/// the file. An HTTP-range-backed adapter typically only needs three
/// logical fetches to satisfy this function:
///
/// 1. **End-of-file scan for the EOCD record** (~64 KiB worst case) — the
///    vendored reader seeks to the file's end and scans backwards for the
///    end-of-central-directory signature.
/// 2. **One read for the central directory** (a few KiB for typical ML
///    `NPZ` archives, since each `STORE`d entry produces one fixed-size
///    record plus its UTF-8 name) — the vendored reader seeks to the offset
///    recorded in the EOCD and reads the full directory.
/// 3. **One read per entry for the local file header + `NPY` header**
///    (~512 B per `.npy` entry) — for each entry, the vendored reader seeks
///    to the local file header offset, then this function reads the
///    `NPY` preamble + dict header (~128 B in practice).
///
/// A naive adapter that issues one HTTP range request per `read`/`seek`
/// call will still work but may be inefficient. Adapters that prefetch
/// and cache the EOCD region and the central directory on first access
/// amortise the round trips effectively to two HTTP requests plus one
/// per entry. For a typical 5-array Gemma Scope `params.npz`, that is
/// ~7 small range requests covering well under 100 KiB instead of the
/// full ~300 MiB download.
///
/// Anamnesis does not ship an HTTP transport itself — the network layer
/// belongs in downstream crates (e.g., `hf-fm`'s safetensors range-reader
/// extended to `NPZ`). This function defines the I/O contract such an
/// adapter must satisfy.
///
/// # Errors
///
/// Returns [`AnamnesisError::Io`] if a `read` or `seek` on the supplied
/// reader fails.
///
/// Returns [`AnamnesisError::Parse`] if the `ZIP` archive is malformed or
/// an `NPY` header is invalid.
///
/// Returns [`AnamnesisError::LimitExceeded`] if a `ZIP` entry count / name
/// length or an `NPY` header size exceeds a permanent cap (always-on).
///
/// Returns [`AnamnesisError::Unsupported`] if an array uses Fortran order
/// or an unsupported dtype.
///
/// # Memory
///
/// Allocates only per-tensor metadata (name, shape, dtype). No tensor
/// data is read or allocated. Peak memory is proportional to the number
/// of entries (a few hundred bytes per tensor plus the transient `NPY`
/// header buffer during parsing), independent of the archive's
/// data-segment size.
///
/// **Saturation note:** if a shape's element count overflows `usize`,
/// `byte_len` saturates to `usize::MAX` and `total_bytes` saturates to
/// `u64::MAX`. Behaviour matches [`inspect_npz`] and differs from
/// `parse_npz`, which returns `Err` on the same overflow.
pub fn inspect_npz_from_reader<R: Read + Seek>(reader: R) -> crate::Result<NpzInspectInfo> {
    inspect_npz_from_reader_with_options(reader, &crate::InspectOptions::new())
}

/// Inspects an `NPZ` archive from any `Read + Seek` source, under a
/// caller-supplied [`InspectOptions`](crate::InspectOptions).
///
/// The two-knob form of [`inspect_npz_from_reader`], new in v0.7.6.
/// `output_dtype` is recorded and vacuous here (`NPZ` arrays are already full
/// precision), so the knob that does work is **`limits`**, and it closes the
/// sharpest instance of the gap in the family: `NPZ` had no bounded inspect at
/// all, not even the front-matter alternative `GGUF` and `.pth` offer. The walk
/// is `O(entries)` in both I/O and allocation, and the permanent
/// `ZIP_MAX_ENTRIES` floor is `1 << 20`, so a hostile archive could force a
/// million-entry central directory, a million `NPY` header reads, and a
/// `Vec<NpzTensorInfo>` no caller could cap — on the very call
/// [`README.md`](https://github.com/mi-for-the-rust-of-us/anamnesis#parsing-untrusted-input)
/// tells a multi-tenant host to make **first**. `limits` now bounds the
/// central-directory read, the item count, and every `NPY` header, cumulatively
/// as well as individually.
///
/// # Errors
///
/// As [`inspect_npz_from_reader`], plus [`AnamnesisError::LimitExceeded`] when
/// a declared count or allocation exceeds `options.limits` rather than only
/// when it exceeds a permanent cap.
///
/// # Memory
///
/// `O(n_arrays)` metadata — names, shapes and dtypes — plus one `NPY` header at
/// a time. No array data is read.
pub fn inspect_npz_from_reader_with_options<R: Read + Seek>(
    reader: R,
    options: &crate::InspectOptions,
) -> crate::Result<NpzInspectInfo> {
    let mut src = crate::parse::zip::ReaderSource::new(reader)?;
    let entries = crate::parse::zip::read_central_directory(&mut src, &options.limits)?;
    // One accountant for the whole walk, so the running total bounds the header
    // set rather than each header in isolation.
    let mut budget = Budget::new(&options.limits);

    // Clamp the pre-allocation hint: the central-directory entry count is
    // attacker-influenced (a many-empty-entries zip), so trusting it for
    // `with_capacity` would commit ~1–2× the file size eagerly. The Vec still
    // grows as entries are pushed. Mirrors the `GGUF` parser's
    // `PREALLOC_SOFT_CAP` clamp.
    let mut tensors = Vec::with_capacity(entries.len().min(PREALLOC_SOFT_CAP));
    let mut total_bytes: u64 = 0;
    let mut dtypes: Vec<NpzDtype> = Vec::new();

    for entry in &entries {
        // Strip .npy suffix; skip non-.npy entries (e.g., __MACOSX/).
        let name = match entry.name.strip_suffix(".npy") {
            // BORROW: .to_owned() converts &str slice to owned String
            Some(n) => n.to_owned(),
            None => continue,
        };

        // The header read is charged to the caller's budget. Before v0.7.6 this
        // was `Budget::unbounded()` by design, which left the permanent
        // `NPY_MAX_HEADER_BYTES` cap as the only bound on the call a host is
        // told to make *first* on an untrusted archive — the one call it could
        // not tighten. One accountant for the whole walk, so the cumulative
        // total bounds the headers as a set and not merely one at a time.
        let mut entry_reader = open_npz_entry_reader(&mut src, entry, &name)?;
        let header = parse_npy_header(&mut entry_reader, &mut budget)?;

        if header.fortran_order {
            return Err(AnamnesisError::Unsupported {
                format: "NPZ".into(),
                detail: format!(
                    "fortran-order arrays not supported (array '{name}'). \
                     ML frameworks save C-order by default"
                ),
            });
        }

        let n_elements: usize = header
            .shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .unwrap_or(usize::MAX);
        let byte_len = n_elements.saturating_mul(header.dtype.byte_size());

        // CAST: usize → u64, byte lengths fit in u64
        #[allow(clippy::as_conversions)]
        {
            total_bytes = total_bytes.saturating_add(byte_len as u64);
        }

        if !dtypes.contains(&header.dtype) {
            dtypes.push(header.dtype);
        }

        tensors.push(NpzTensorInfo {
            name,
            shape: header.shape,
            dtype: header.dtype,
            byte_len,
        });
    }

    Ok(NpzInspectInfo {
        tensors,
        total_bytes,
        dtypes,
        // Not a copy made out of symmetry: `NPZ` arrays are passed through in
        // their source dtype, so the post-conversion size *is* the stored size.
        dequantized_size: total_bytes,
        output_dtype: options.output_dtype,
    })
}

/// Parses an `NPZ` archive, returning all arrays as a name-to-tensor map.
///
/// Implements a lean `NPY` header parser with bulk data extraction. For
/// little-endian data on a little-endian machine (the common case for ML
/// weight files), the raw bytes are returned directly — zero per-element
/// deserialization.
///
/// # Errors
///
/// Returns [`AnamnesisError::Io`] if the file cannot be opened or read.
///
/// Returns [`AnamnesisError::Parse`] if the `ZIP` archive is malformed, an
/// `NPY` header is invalid, or array data is truncated.
///
/// Returns [`AnamnesisError::LimitExceeded`] if a `ZIP` entry count / name
/// length, an `NPY` header size, or a declared array size exceeds a permanent
/// cap (all always-on, reachable even at default limits).
///
/// Returns [`AnamnesisError::Unsupported`] if an array uses Fortran order
/// or an unsupported dtype.
///
/// # Memory
///
/// Allocates one `Vec<u8>` per array (the raw data). Peak memory equals the
/// sum of all arrays. No intermediate typed buffers — data goes directly
/// from the `ZIP` entry to the output `Vec<u8>`.
pub fn parse_npz(path: impl AsRef<Path>) -> crate::Result<HashMap<String, NpzTensor>> {
    parse_npz_with_limits(path, &ParseLimits::default())
}

/// Parses an `NPZ` archive under a caller-supplied [`ParseLimits`] budget.
///
/// Identical to [`parse_npz`] but enforces the caller's [`ParseLimits`] —
/// fail-fast, before each allocation — over the declared archive entry count
/// (the item-count cap), each compressed entry's expansion ratio (the
/// decompression-ratio cap, rejecting a `DEFLATE` zip bomb from metadata before
/// reading), and every `NPY` header and array allocation (the per-allocation
/// cap **and** the cumulative-byte aggregate, which bounds `parse_npz`'s peak
/// heap since every array is held in the returned map at once). The built-in
/// per-format caps still apply; `limits` can only tighten them. [`parse_npz`]
/// is the `ParseLimits::default()` (unbounded) special case.
///
/// # Errors
///
/// Returns [`AnamnesisError::Io`] if the file cannot be opened or read.
///
/// Returns [`AnamnesisError::LimitExceeded`] if a declared count, allocation,
/// or decompression ratio exceeds `limits` or a permanent `NPZ` cap.
/// Returns [`AnamnesisError::Parse`] if the `ZIP` archive is malformed, an
/// `NPY` header is invalid, or array data is truncated.
///
/// Returns [`AnamnesisError::Unsupported`] if an array uses Fortran order
/// or an unsupported dtype.
///
/// # Memory
///
/// Allocates one `Vec<u8>` per array (the raw data). Peak memory equals the
/// sum of all arrays. No intermediate typed buffers — data goes directly
/// from the `ZIP` entry to the output `Vec<u8>`.
pub fn parse_npz_with_limits(
    path: impl AsRef<Path>,
    limits: &ParseLimits,
) -> crate::Result<HashMap<String, NpzTensor>> {
    let file = std::fs::File::open(path.as_ref())?;
    parse_npz_from_zip_reader(file, limits)
}

/// Parses `NPZ` bytes already held in memory, returning the arrays as a
/// name-to-tensor map.
///
/// The `NPZ` member of the copy-based, mmap-free family every other format has
/// had since [Phase 6.13](https://github.com/mi-for-the-rust-of-us/anamnesis/blob/main/ROADMAP.md)
/// — `parse_bytes` for safetensors, `parse_gguf_bytes`, `parse_pth_bytes`. Its
/// absence was not a design decision but an omission, and it meant the
/// untrusted-input contract those siblings document had no `NPZ` instance: a
/// caller holding bytes had to write a temporary file first.
///
/// [`parse_npz_bytes`] is the [`ParseLimits::default`] (unbounded) special case
/// of [`parse_npz_bytes_with_limits`].
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if the `ZIP` archive is malformed, an
/// `NPY` header is invalid, or array data is truncated.
/// Returns [`AnamnesisError::LimitExceeded`] if a declared count, allocation,
/// or decompression ratio exceeds a permanent `NPZ` cap.
/// Returns [`AnamnesisError::Unsupported`] if an array uses an unsupported
/// dtype.
/// Returns [`AnamnesisError::Io`] only from the in-memory cursor, which does
/// not fail in practice.
///
/// # Memory
///
/// Holds `bytes` for the duration of the parse **and** allocates one `Vec<u8>`
/// per array, so peak is the archive plus the decompressed arrays. Contrast
/// [`parse_npz`], which streams from a file and peaks at the arrays alone.
pub fn parse_npz_bytes(bytes: Vec<u8>) -> crate::Result<HashMap<String, NpzTensor>> {
    parse_npz_bytes_with_limits(bytes, &ParseLimits::default())
}

/// Parses owned `NPZ` bytes under a caller-supplied [`ParseLimits`] budget.
///
/// Rejects an input larger than [`ParseLimits::max_single_alloc_bytes`] before
/// parsing, then enforces every applicable ceiling exactly as
/// [`parse_npz_with_limits`] does.
///
/// # Errors
///
/// Returns [`AnamnesisError::LimitExceeded`] if `bytes` exceeds `limits`, or if
/// a declared count, allocation or decompression ratio does. Otherwise as
/// [`parse_npz_bytes`].
///
/// # Memory
///
/// As [`parse_npz_bytes`].
pub fn parse_npz_bytes_with_limits(
    bytes: Vec<u8>,
    limits: &ParseLimits,
) -> crate::Result<HashMap<String, NpzTensor>> {
    // CAST: usize → u64, lossless widening on all supported targets.
    #[allow(clippy::as_conversions)]
    let len = bytes.len() as u64;
    limits.check_alloc(len, "NPZ bytes")?;
    parse_npz_from_zip_reader(std::io::Cursor::new(bytes), limits)
}

/// Parses an `NPZ` archive from any reader, returning the arrays as a
/// name-to-tensor map — the copy-based, mmap-free path for untrusted streams.
///
/// The whole stream is read into an owned buffer (bounded by [`ParseLimits`])
/// and parsed from memory, so a truncated or hostile stream is a clean `Err`.
/// Takes `Read` alone, not `Read + Seek`, matching `parse_pth_from_reader` and
/// `parse_gguf_from_reader`: the seeks the `ZIP` container needs happen over the
/// owned buffer, so an HTTP body or a pipe works without a seekable adapter.
/// (`inspect_npz_from_reader` does require `Seek`, because it deliberately
/// avoids buffering the archive at all.)
///
/// # Errors
///
/// Returns [`AnamnesisError::Io`] if the reader fails.
/// Returns [`AnamnesisError::LimitExceeded`] if the bytes read exceed `limits`.
/// Otherwise as [`parse_npz_bytes`].
///
/// # Memory
///
/// Reads the stream into an owned `Vec<u8>` of at most
/// `max_single_alloc_bytes + 1` bytes, then as [`parse_npz_bytes`].
pub fn parse_npz_from_reader<R: Read>(reader: R) -> crate::Result<HashMap<String, NpzTensor>> {
    parse_npz_from_reader_with_limits(reader, &ParseLimits::default())
}

/// Parses an `NPZ` archive from any reader under a caller-supplied
/// [`ParseLimits`] budget — the bounded, mmap-free path for untrusted input.
///
/// # Errors
///
/// As [`parse_npz_from_reader`].
///
/// # Memory
///
/// As [`parse_npz_from_reader`].
pub fn parse_npz_from_reader_with_limits<R: Read>(
    reader: R,
    limits: &ParseLimits,
) -> crate::Result<HashMap<String, NpzTensor>> {
    let bytes = limits.read_to_vec_bounded(reader, "NPZ archive")?;
    parse_npz_bytes_with_limits(bytes, limits)
}

/// The single parse body shared by the path, bytes and reader entry points, so
/// the three cannot drift on limit enforcement or `NPY` interpretation.
fn parse_npz_from_zip_reader<R: Read + Seek>(
    reader: R,
    limits: &ParseLimits,
) -> crate::Result<HashMap<String, NpzTensor>> {
    let mut src = crate::parse::zip::ReaderSource::new(reader)?;
    // The reader enforces `limits` (item-count + the central-directory single
    // allocation) fail-fast, before materialising the entry vector — so a
    // tight-budget caller bounds the container metadata, not only the
    // downstream array data.
    let entries = crate::parse::zip::read_central_directory(&mut src, limits)?;

    // One accountant for the whole archive: `parse_npz` holds every array in
    // the returned map simultaneously, so the running total bounds peak heap.
    let mut budget = Budget::new(limits);
    // Clamp the pre-allocation hint (see `inspect_npz_from_reader`): a
    // many-empty-entries zip would otherwise drive a ~1–2× file-size eager
    // `HashMap` allocation. The map grows as entries are inserted.
    let mut result = HashMap::with_capacity(entries.len().min(PREALLOC_SOFT_CAP));

    for entry in &entries {
        // Strip .npy suffix; skip non-.npy entries (e.g., __MACOSX/).
        let name = match entry.name.strip_suffix(".npy") {
            // BORROW: .to_owned() converts &str slice to owned String for HashMap key
            Some(n) => n.to_owned(),
            None => continue,
        };

        // Zip-bomb guard: reject an entry whose declared uncompressed size is an
        // absurd multiple of its compressed size, from archive metadata, before
        // reading or allocating anything. STORED entries have ratio 1 and pass.
        limits.check_decompression_ratio(entry.uncompressed_size, entry.compressed_size, &name)?;

        // The uncompressed (declared) size is the cross-check bound passed to
        // `read_array_data`: an entry cannot decode to more than it declares.
        let entry_size = entry.uncompressed_size;
        let mut entry_reader = open_npz_entry_reader(&mut src, entry, &name)?;
        let header = parse_npy_header(&mut entry_reader, &mut budget)?;

        if header.fortran_order {
            return Err(AnamnesisError::Unsupported {
                format: "NPZ".into(),
                detail: format!(
                    "fortran-order arrays not supported (array '{name}'). \
                     ML frameworks save C-order by default"
                ),
            });
        }

        let data = read_array_data(&mut entry_reader, &header, entry_size, &mut budget)?;

        result.insert(
            name.clone(),
            NpzTensor {
                name,
                shape: header.shape,
                dtype: header.dtype,
                data,
            },
        );
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(
    clippy::panic,
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::float_cmp
)]
mod tests {
    use std::io::Write;

    use super::*;

    // -- NpzDtype::byte_size -------------------------------------------------

    #[test]
    fn byte_size_1() {
        assert_eq!(NpzDtype::Bool.byte_size(), 1);
        assert_eq!(NpzDtype::U8.byte_size(), 1);
        assert_eq!(NpzDtype::I8.byte_size(), 1);
    }

    #[test]
    fn byte_size_2() {
        assert_eq!(NpzDtype::U16.byte_size(), 2);
        assert_eq!(NpzDtype::I16.byte_size(), 2);
        assert_eq!(NpzDtype::F16.byte_size(), 2);
        assert_eq!(NpzDtype::BF16.byte_size(), 2);
    }

    #[test]
    fn byte_size_4() {
        assert_eq!(NpzDtype::U32.byte_size(), 4);
        assert_eq!(NpzDtype::I32.byte_size(), 4);
        assert_eq!(NpzDtype::F32.byte_size(), 4);
    }

    #[test]
    fn byte_size_8() {
        assert_eq!(NpzDtype::U64.byte_size(), 8);
        assert_eq!(NpzDtype::I64.byte_size(), 8);
        assert_eq!(NpzDtype::F64.byte_size(), 8);
    }

    // -- NpzDtype Display ----------------------------------------------------

    #[test]
    fn display() {
        assert_eq!(NpzDtype::F32.to_string(), "F32");
        assert_eq!(NpzDtype::BF16.to_string(), "BF16");
        assert_eq!(NpzDtype::I64.to_string(), "I64");
        assert_eq!(NpzDtype::Bool.to_string(), "BOOL");
    }

    // -- parse_descr ---------------------------------------------------------

    #[test]
    fn parse_descr_float_types() {
        assert_eq!(parse_descr("<f2").unwrap(), (NpzDtype::F16, false));
        assert_eq!(parse_descr("<f4").unwrap(), (NpzDtype::F32, false));
        assert_eq!(parse_descr("<f8").unwrap(), (NpzDtype::F64, false));
        assert_eq!(parse_descr(">f4").unwrap(), (NpzDtype::F32, true));
    }

    #[test]
    fn parse_descr_int_types() {
        assert_eq!(parse_descr("|i1").unwrap(), (NpzDtype::I8, false));
        assert_eq!(parse_descr("<i2").unwrap(), (NpzDtype::I16, false));
        assert_eq!(parse_descr("<i4").unwrap(), (NpzDtype::I32, false));
        assert_eq!(parse_descr("<i8").unwrap(), (NpzDtype::I64, false));
        assert_eq!(parse_descr(">i4").unwrap(), (NpzDtype::I32, true));
    }

    #[test]
    fn parse_descr_uint_types() {
        assert_eq!(parse_descr("|u1").unwrap(), (NpzDtype::U8, false));
        assert_eq!(parse_descr("<u2").unwrap(), (NpzDtype::U16, false));
        assert_eq!(parse_descr("<u4").unwrap(), (NpzDtype::U32, false));
        assert_eq!(parse_descr("<u8").unwrap(), (NpzDtype::U64, false));
    }

    #[test]
    fn parse_descr_bool() {
        assert_eq!(parse_descr("|b1").unwrap(), (NpzDtype::Bool, false));
    }

    #[test]
    fn parse_descr_bf16_void() {
        assert_eq!(parse_descr("|V2").unwrap(), (NpzDtype::BF16, false));
    }

    #[test]
    fn parse_descr_unsupported() {
        assert!(parse_descr("<c8").is_err()); // complex
        assert!(parse_descr("<U4").is_err()); // unicode string
        assert!(parse_descr("x").is_err()); // too short
    }

    // Regression (fuzz_npz_limits crash, 2026-06-15): a descriptor whose first
    // character is a multi-byte UTF-8 codepoint must yield a clean error, never
    // panic on the `&descr[1..]` str-slice char boundary. `bytes.len() >= 2`
    // here, but byte index 1 lands inside the first character.
    #[test]
    fn parse_descr_multibyte_first_char_is_clean_error() {
        assert!(parse_descr("\u{359}f4").is_err()); // U+0359 is 2 bytes (CD 99)
        assert!(parse_descr("é4").is_err()); // 'é' is 2 bytes (C3 A9)
        assert!(parse_descr("\u{1F600}").is_err()); // 4-byte emoji
    }

    // -- extract_descr -------------------------------------------------------

    #[test]
    fn extract_descr_from_header() {
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }";
        let (dtype, be) = extract_descr(header).unwrap();
        assert_eq!(dtype, NpzDtype::F32);
        assert!(!be);
    }

    #[test]
    fn extract_descr_double_quotes() {
        let header = "{\"descr\": \"<i4\", \"fortran_order\": False, \"shape\": (10,), }";
        let (dtype, _) = extract_descr(header).unwrap();
        assert_eq!(dtype, NpzDtype::I32);
    }

    #[test]
    fn extract_descr_mixed_quotes() {
        // Double-quoted descr value followed by single-quoted keys.
        // Previously broken: the quote-char detection scanned the entire
        // header tail, picking up the wrong quote character.
        let header = "{'descr': \"<f4\", 'fortran_order': False, 'shape': (2, 3), }";
        let (dtype, be) = extract_descr(header).unwrap();
        assert_eq!(dtype, NpzDtype::F32);
        assert!(!be);
    }

    // -- extract_fortran_order ------------------------------------------------

    #[test]
    fn fortran_order_false() {
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }";
        assert!(!extract_fortran_order(header));
    }

    #[test]
    fn fortran_order_true() {
        let header = "{'descr': '<f4', 'fortran_order': True, 'shape': (2, 3), }";
        assert!(extract_fortran_order(header));
    }

    #[test]
    fn fortran_order_missing() {
        let header = "{'descr': '<f4', 'shape': (2, 3), }";
        assert!(!extract_fortran_order(header));
    }

    // -- extract_shape -------------------------------------------------------

    #[test]
    fn shape_scalar() {
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (), }";
        let shape = extract_shape(header).unwrap();
        assert!(shape.is_empty());
    }

    #[test]
    fn shape_1d() {
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (16384,), }";
        let shape = extract_shape(header).unwrap();
        assert_eq!(shape, vec![16384]);
    }

    #[test]
    fn shape_2d() {
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (2304, 16384), }";
        let shape = extract_shape(header).unwrap();
        assert_eq!(shape, vec![2304, 16384]);
    }

    #[test]
    fn shape_3d() {
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3, 4), }";
        let shape = extract_shape(header).unwrap();
        assert_eq!(shape, vec![2, 3, 4]);
    }

    // -- NPY header roundtrip ------------------------------------------------

    /// Build a minimal NPY v1 file with the given header and data bytes.
    fn make_npy_v1(header_str: &str, data: &[u8]) -> Vec<u8> {
        let header_bytes = header_str.as_bytes();
        // Pad header to 64-byte alignment (magic=6 + version=2 + len=2 = 10).
        let total_before_pad = 10 + header_bytes.len();
        let padding = (64 - (total_before_pad % 64)) % 64;
        let padded_len = header_bytes.len() + padding;

        let mut npy = Vec::new();
        npy.extend_from_slice(NPY_MAGIC);
        npy.push(1); // major
        npy.push(0); // minor
        npy.extend_from_slice(&(padded_len as u16).to_le_bytes());
        npy.extend_from_slice(header_bytes);
        // Pad with spaces, ending with newline.
        if padding > 0 {
            npy.extend(std::iter::repeat_n(b' ', padding - 1));
            npy.push(b'\n');
        }
        npy.extend_from_slice(data);
        npy
    }

    #[test]
    fn roundtrip_f32_npy_v1() {
        let values: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let mut data = Vec::new();
        for v in &values {
            data.extend_from_slice(&v.to_le_bytes());
        }

        let npy = make_npy_v1(
            "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 2), }",
            &data,
        );

        let mut reader = std::io::Cursor::new(&npy);
        let header = parse_npy_header(&mut reader, &mut Budget::unbounded()).unwrap();
        assert_eq!(header.dtype, NpzDtype::F32);
        assert!(!header.big_endian);
        assert!(!header.fortran_order);
        assert_eq!(header.shape, vec![2, 2]);

        // u64::MAX entry size: this test exercises the read/byteswap path, not
        // the entry-size guard (covered by its own test below).
        let result =
            read_array_data(&mut reader, &header, u64::MAX, &mut Budget::unbounded()).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn roundtrip_f32_big_endian() {
        // Big-endian f32: 1.0 = 0x3F800000 → bytes [3F, 80, 00, 00]
        let data_be: Vec<u8> = vec![0x3F, 0x80, 0x00, 0x00];

        let npy = make_npy_v1(
            "{'descr': '>f4', 'fortran_order': False, 'shape': (1,), }",
            &data_be,
        );

        let mut reader = std::io::Cursor::new(&npy);
        let header = parse_npy_header(&mut reader, &mut Budget::unbounded()).unwrap();
        assert!(header.big_endian);

        let result =
            read_array_data(&mut reader, &header, u64::MAX, &mut Budget::unbounded()).unwrap();
        // After byteswap: [00, 00, 80, 3F] = 1.0 in LE
        assert_eq!(result, vec![0x00, 0x00, 0x80, 0x3F]);
        let val = f32::from_le_bytes([result[0], result[1], result[2], result[3]]);
        assert_eq!(val, 1.0);
    }

    #[test]
    fn npy_v2_header() {
        let header_str = "{'descr': '<f8', 'fortran_order': False, 'shape': (1,), }";
        let header_bytes = header_str.as_bytes();
        let total_before_pad = 12 + header_bytes.len();
        let padding = (64 - (total_before_pad % 64)) % 64;
        let padded_len = header_bytes.len() + padding;

        let mut npy = Vec::new();
        npy.extend_from_slice(NPY_MAGIC);
        npy.push(2); // major
        npy.push(0); // minor
        npy.extend_from_slice(&(padded_len as u32).to_le_bytes());
        npy.extend_from_slice(header_bytes);
        if padding > 0 {
            npy.extend(std::iter::repeat_n(b' ', padding - 1));
            npy.push(b'\n');
        }
        // One f64 value
        npy.extend_from_slice(&42.5_f64.to_le_bytes());

        let mut reader = std::io::Cursor::new(&npy);
        let header = parse_npy_header(&mut reader, &mut Budget::unbounded()).unwrap();
        assert_eq!(header.dtype, NpzDtype::F64);
        assert_eq!(header.shape, vec![1]);

        let result =
            read_array_data(&mut reader, &header, u64::MAX, &mut Budget::unbounded()).unwrap();
        assert_eq!(result, 42.5_f64.to_le_bytes());
    }

    #[test]
    fn invalid_magic_rejected() {
        let data = b"NOT_NUMPY_DATA_AT_ALL";
        let mut reader = std::io::Cursor::new(data);
        assert!(parse_npy_header(&mut reader, &mut Budget::unbounded()).is_err());
    }

    #[test]
    fn fortran_order_rejected_in_parse_npz() {
        // We can't easily test parse_npz with Fortran order without creating
        // a real NPZ file, but we can verify the extraction logic.
        let header = "{'descr': '<f4', 'fortran_order': True, 'shape': (2, 3), }";
        assert!(extract_fortran_order(header));
    }

    // -- Gap tests (review findings G32–G36) ---------------------------------

    // G32: Fortran-order rejection through parse_npz end-to-end
    #[test]
    fn fortran_order_rejected_end_to_end() {
        // Build a minimal NPZ containing a single Fortran-order NPY entry.
        let tmp = tempfile::NamedTempFile::new().unwrap();
        {
            let file = std::fs::File::create(tmp.path()).unwrap();
            let mut zip = zip::ZipWriter::new(file);
            let options = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            zip.start_file("arr.npy", options).unwrap();

            // Build NPY v1 with fortran_order: True
            let header_str = "{'descr': '<f4', 'fortran_order': True, 'shape': (2, 2), }";
            let npy = make_npy_v1(header_str, &[0u8; 16]); // 4 f32 zeros
            zip.write_all(&npy).unwrap();
            zip.finish().unwrap();
        }
        let err = parse_npz(tmp.path()).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("Fortran-order") || msg.contains("fortran"),
            "expected Fortran-order error, got: {msg}"
        );
    }

    // G33: Empty NPZ archive
    #[test]
    fn empty_npz_archive() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        {
            let file = std::fs::File::create(tmp.path()).unwrap();
            let zip = zip::ZipWriter::new(file);
            zip.finish().unwrap();
        }
        let result = parse_npz(tmp.path()).unwrap();
        assert!(result.is_empty(), "empty NPZ should return empty map");

        // NN4: also verify inspect_npz handles empty archives
        let info = inspect_npz(tmp.path()).unwrap();
        assert!(info.tensors.is_empty());
        assert_eq!(info.total_bytes, 0);
    }

    // G35: Native-endian '=' prefix in parse_descr
    #[test]
    fn parse_descr_native_endian() {
        // '=' means native endian — treated as LE on all modern platforms
        let (dtype, be) = parse_descr("=f4").unwrap();
        assert_eq!(dtype, NpzDtype::F32);
        assert!(!be, "'=' should not be treated as big-endian");
    }

    // G34: Big-endian array through parse_npz end-to-end
    #[test]
    fn big_endian_through_parse_npz() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        {
            let file = std::fs::File::create(tmp.path()).unwrap();
            let mut zip = zip::ZipWriter::new(file);
            let options = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            zip.start_file("val.npy", options).unwrap();

            // NPY with big-endian f32: 1.0 = [3F, 80, 00, 00] BE
            let npy = make_npy_v1(
                "{'descr': '>f4', 'fortran_order': False, 'shape': (1,), }",
                &[0x3F, 0x80, 0x00, 0x00],
            );
            zip.write_all(&npy).unwrap();
            zip.finish().unwrap();
        }
        let tensors = parse_npz(tmp.path()).unwrap();
        let t = tensors.get("val").expect("val not found");
        assert_eq!(t.dtype, NpzDtype::F32);
        // After byteswap: [00, 00, 80, 3F] = 1.0 LE
        assert_eq!(t.data, vec![0x00, 0x00, 0x80, 0x3F]);
    }

    /// Phase 6.8 Step 1: a tightened `ParseLimits` rejects an `NPY` header, an
    /// array's byte size, or the entry count that `ParseLimits::default()`
    /// (unbounded) accepts; the default is byte-identical to the limit-free
    /// `parse_npz`. The array (4000 bytes) is made far larger than the small
    /// `NPY` header so each single-allocation gate can be isolated.
    #[test]
    fn parse_npz_respects_parse_limits() {
        // One little-endian f32 array `val`, shape (1000,) → 4000 bytes.
        let data = vec![0u8; 4000];
        let tmp = tempfile::NamedTempFile::new().unwrap();
        {
            let file = std::fs::File::create(tmp.path()).unwrap();
            let mut zip = zip::ZipWriter::new(file);
            let options = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            zip.start_file("val.npy", options).unwrap();
            let npy = make_npy_v1(
                "{'descr': '<f4', 'fortran_order': False, 'shape': (1000,), }",
                &data,
            );
            zip.write_all(&npy).unwrap();
            zip.finish().unwrap();
        }

        // Default (unbounded) parses, and equals the limit-free entry point.
        let baseline = parse_npz(tmp.path()).unwrap();
        let with_default = parse_npz_with_limits(tmp.path(), &ParseLimits::default()).unwrap();
        assert_eq!(with_default.get("val").unwrap().data.len(), 4000);
        assert_eq!(
            baseline.get("val").unwrap().data,
            with_default.get("val").unwrap().data
        );

        // Array single-allocation ceiling: the small NPY header (< 3999 bytes)
        // passes, but the 4000-byte array is rejected at 3999 and accepted at
        // 4000.
        let err = parse_npz_with_limits(
            tmp.path(),
            &ParseLimits::default().with_max_single_alloc(3999),
        )
        .unwrap_err();
        assert!(
            matches!(err, AnamnesisError::LimitExceeded { limit, .. } if limit == "max_single_alloc_bytes"),
            "expected array single-alloc limit error, got: {err}"
        );
        assert!(
            parse_npz_with_limits(
                tmp.path(),
                &ParseLimits::default().with_max_single_alloc(4000)
            )
            .is_ok()
        );

        // Container single-allocation ceiling: since Phase 6.12 the
        // central-directory read on the streaming (reader) path is charged to
        // `max_single_alloc` too, so a 1-byte budget rejects the container
        // fail-fast — before any NPY header or array is reached.
        let err =
            parse_npz_with_limits(tmp.path(), &ParseLimits::default().with_max_single_alloc(1))
                .unwrap_err();
        assert!(
            matches!(err, AnamnesisError::LimitExceeded { limit, .. } if limit == "max_single_alloc_bytes"),
            "expected container/header single-alloc limit error, got: {err}"
        );

        // Item-count ceiling: 1 entry rejected at 0, accepted at 1.
        let err = parse_npz_with_limits(tmp.path(), &ParseLimits::default().with_max_item_count(0))
            .unwrap_err();
        assert!(
            matches!(err, AnamnesisError::LimitExceeded { limit, .. } if limit == "max_item_count"),
            "expected item-count limit error, got: {err}"
        );
        assert!(
            parse_npz_with_limits(tmp.path(), &ParseLimits::default().with_max_item_count(1))
                .is_ok()
        );
    }

    /// Phase 6.8 Step 2: the aggregate `max_total_bytes` rejects a file whose
    /// arrays each pass `max_single_alloc` but whose cumulative heap (the peak
    /// `parse_npz` holds, since every array lands in the returned map) crosses
    /// the budget — the many-small-items blow-up the per-item cap misses.
    #[test]
    fn parse_npz_aggregate_budget() {
        // Three little-endian f32 arrays, shape (1000,) → 4000 bytes each.
        let data = vec![0u8; 4000];
        let tmp = tempfile::NamedTempFile::new().unwrap();
        {
            let file = std::fs::File::create(tmp.path()).unwrap();
            let mut zip = zip::ZipWriter::new(file);
            let options = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            for name in ["a.npy", "b.npy", "c.npy"] {
                zip.start_file(name, options).unwrap();
                let npy = make_npy_v1(
                    "{'descr': '<f4', 'fortran_order': False, 'shape': (1000,), }",
                    &data,
                );
                zip.write_all(&npy).unwrap();
            }
            zip.finish().unwrap();
        }

        // Each 4000-byte array clears a 4096-byte single-allocation cap, so with
        // no aggregate budget the file parses — isolating that the per-item cap
        // alone does NOT reject the 12000-byte total.
        let per_item_only = ParseLimits::default().with_max_single_alloc(4096);
        assert!(parse_npz_with_limits(tmp.path(), &per_item_only).is_ok());

        // Same per-item cap, but an 8000-byte aggregate budget < 3×4000 → the
        // cumulative charge crosses it and the parse is rejected.
        let with_aggregate = per_item_only.with_max_total_bytes(8000);
        let err = parse_npz_with_limits(tmp.path(), &with_aggregate).unwrap_err();
        assert!(
            matches!(err, AnamnesisError::LimitExceeded { limit, .. } if limit == "max_total_bytes"),
            "expected aggregate limit error, got: {err}"
        );

        // A generous aggregate budget parses, and the default is unchanged.
        assert!(
            parse_npz_with_limits(
                tmp.path(),
                &ParseLimits::default().with_max_total_bytes(1 << 20)
            )
            .is_ok()
        );
        assert!(parse_npz_with_limits(tmp.path(), &ParseLimits::default()).is_ok());
    }

    /// Phase 6.8 Step 3: the decompression-ratio cap rejects a `DEFLATE` entry
    /// whose declared uncompressed size is an absurd multiple of its compressed
    /// size (a zip bomb), from archive metadata, while leaving `STORED` entries
    /// (ratio 1) and the unbounded default untouched.
    #[test]
    fn parse_npz_decompression_ratio_cap() {
        // A `DEFLATE` `.npy` of 40000 zero bytes compresses to a few dozen
        // bytes → a > 100:1 expansion ratio.
        let zeros = vec![0u8; 40_000];
        let tmp = tempfile::NamedTempFile::new().unwrap();
        {
            let file = std::fs::File::create(tmp.path()).unwrap();
            let mut zip = zip::ZipWriter::new(file);
            let options = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated);
            zip.start_file("z.npy", options).unwrap();
            let npy = make_npy_v1(
                "{'descr': '<u1', 'fortran_order': False, 'shape': (40000,), }",
                &zeros,
            );
            zip.write_all(&npy).unwrap();
            zip.finish().unwrap();
        }

        // A 2:1 cap rejects the high-ratio entry from metadata, before reading.
        let err = parse_npz_with_limits(
            tmp.path(),
            &ParseLimits::default().with_max_decompression_ratio(2),
        )
        .unwrap_err();
        assert!(
            matches!(err, AnamnesisError::LimitExceeded { limit, .. } if limit == "max_decompression_ratio"),
            "expected ratio limit error, got: {err}"
        );

        // A generous cap and the unbounded default both parse it.
        assert!(
            parse_npz_with_limits(
                tmp.path(),
                &ParseLimits::default().with_max_decompression_ratio(100_000)
            )
            .is_ok()
        );
        assert!(parse_npz_with_limits(tmp.path(), &ParseLimits::default()).is_ok());

        // A STORED entry (ratio 1) is never falsely rejected, even at ratio 1.
        let stored = tempfile::NamedTempFile::new().unwrap();
        {
            let file = std::fs::File::create(stored.path()).unwrap();
            let mut zip = zip::ZipWriter::new(file);
            let options = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            zip.start_file("s.npy", options).unwrap();
            let npy = make_npy_v1(
                "{'descr': '<u1', 'fortran_order': False, 'shape': (16,), }",
                &[0u8; 16],
            );
            zip.write_all(&npy).unwrap();
            zip.finish().unwrap();
        }
        assert!(
            parse_npz_with_limits(
                stored.path(),
                &ParseLimits::default().with_max_decompression_ratio(1)
            )
            .is_ok()
        );
    }

    /// Phase 6.12: a `DEFLATE` `.npy` entry inflates correctly through the
    /// vendored reader's `BoundedReader` + `flate2` decoder — the bytes parsed
    /// out must equal the original (not just `is_ok()`).
    #[test]
    fn parse_npz_deflate_roundtrip_values() {
        let values: [f32; 6] = [1.5, -2.25, 3.0, 0.0, 42.5, -100.0];
        let mut raw = Vec::new();
        for v in &values {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        let tmp = tempfile::NamedTempFile::new().unwrap();
        {
            let file = std::fs::File::create(tmp.path()).unwrap();
            let mut zip = zip::ZipWriter::new(file);
            let options = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated);
            zip.start_file("w.npy", options).unwrap();
            let npy = make_npy_v1(
                "{'descr': '<f4', 'fortran_order': False, 'shape': (6,), }",
                &raw,
            );
            zip.write_all(&npy).unwrap();
            zip.finish().unwrap();
        }

        let tensors = parse_npz(tmp.path()).unwrap();
        let t = tensors.get("w").expect("w not found");
        assert_eq!(t.dtype, NpzDtype::F32);
        assert_eq!(t.shape, vec![6]);
        assert_eq!(
            t.data, raw,
            "inflated DEFLATE bytes must round-trip exactly"
        );

        // inspect_npz reports the same shape/dtype over the DEFLATE entry too.
        let info = inspect_npz(tmp.path()).unwrap();
        assert_eq!(info.tensors.len(), 1);
        assert_eq!(info.tensors[0].shape, vec![6]);
        assert_eq!(info.tensors[0].dtype, NpzDtype::F32);
    }

    /// The `PREALLOC_SOFT_CAP` clamp on the entry-count pre-allocation hint is
    /// only a hint: an archive with more than `PREALLOC_SOFT_CAP` (256) entries
    /// must still parse every entry (the `HashMap` grows past the clamp).
    #[test]
    fn parse_npz_more_entries_than_prealloc_cap() {
        const N: usize = 300; // > PREALLOC_SOFT_CAP (256)
        let tmp = tempfile::NamedTempFile::new().unwrap();
        {
            let file = std::fs::File::create(tmp.path()).unwrap();
            let mut zip = zip::ZipWriter::new(file);
            let options = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            for i in 0..N {
                zip.start_file(format!("a{i}.npy"), options).unwrap();
                let npy = make_npy_v1(
                    "{'descr': '<u1', 'fortran_order': False, 'shape': (1,), }",
                    &[0u8],
                );
                zip.write_all(&npy).unwrap();
            }
            zip.finish().unwrap();
        }
        let tensors = parse_npz(tmp.path()).unwrap();
        assert_eq!(
            tensors.len(),
            N,
            "every entry must round-trip past the clamp"
        );
    }

    /// Phase 7.6 item 7: the inspect the host is told to run **first** is now
    /// bounded by the host's own budget.
    ///
    /// Until v0.7.6 the summary inspects were "intentionally limit-free", so the
    /// only bound on this call was the permanent `ZIP_MAX_ENTRIES` floor of
    /// `1 << 20` — a hostile archive could force a million-entry walk and a
    /// `Vec<NpzTensorInfo>` no caller could cap, on the call `README.md`
    /// recommends making before anything else. That inverts `ParseLimits`, whose
    /// whole premise is that a caller can always tighten.
    #[test]
    fn inspect_npz_honours_the_caller_budget() {
        let data = vec![0u8; 4000];
        let tmp = tempfile::NamedTempFile::new().unwrap();
        {
            let file = std::fs::File::create(tmp.path()).unwrap();
            let mut zip = zip::ZipWriter::new(file);
            let options = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            for name in ["a.npy", "b.npy", "c.npy"] {
                zip.start_file(name, options).unwrap();
                let npy = make_npy_v1(
                    "{'descr': '<f4', 'fortran_order': False, 'shape': (1000,), }",
                    &data,
                );
                zip.write_all(&npy).unwrap();
            }
            zip.finish().unwrap();
        }

        // Unbounded: three arrays, as stored.
        let info = inspect_npz(tmp.path()).unwrap();
        assert_eq!(info.tensors.len(), 3);

        // An item-count budget below the declared entry count rejects before the
        // entry vector is materialised.
        let tight =
            crate::InspectOptions::new().with_limits(ParseLimits::default().with_max_item_count(2));
        let err = inspect_npz_with_options(tmp.path(), &tight).unwrap_err();
        assert!(
            matches!(err, AnamnesisError::LimitExceeded { .. }),
            "an item-count budget must bound the inspect walk, got {err:?}"
        );

        // A per-allocation budget below one `NPY` header rejects the header read.
        let tiny = crate::InspectOptions::new()
            .with_limits(ParseLimits::default().with_max_single_alloc(8));
        let err = inspect_npz_with_options(tmp.path(), &tiny).unwrap_err();
        assert!(
            matches!(err, AnamnesisError::LimitExceeded { .. }),
            "a per-allocation budget must bound the header reads, got {err:?}"
        );

        // The same budget through the reader-generic entry point.
        let bytes = std::fs::read(tmp.path()).unwrap();
        let err =
            inspect_npz_from_reader_with_options(std::io::Cursor::new(&bytes), &tight).unwrap_err();
        assert!(
            matches!(err, AnamnesisError::LimitExceeded { .. }),
            "the reader form must enforce the same budget, got {err:?}"
        );

        // And a generous budget still succeeds, so the gate is not simply stuck.
        let roomy = crate::InspectOptions::new()
            .with_limits(ParseLimits::default().with_max_item_count(100));
        assert_eq!(
            inspect_npz_with_options(tmp.path(), &roomy)
                .unwrap()
                .tensors
                .len(),
            3
        );
    }

    /// Phase 6.8 Step 5: the documented inspect-before-parse policy gate.
    /// `inspect_npz`'s reported `total_bytes` lets a host predict an
    /// over-budget file *without parsing*; `parse_npz_with_limits` then enforces
    /// the same budget authoritatively. This ties the inspect report to the
    /// parse enforcement.
    #[test]
    fn inspect_npz_total_predicts_parse_limits_gate() {
        // One f32 array, shape (1000,) → 4000 declared bytes.
        let data = vec![0u8; 4000];
        let tmp = tempfile::NamedTempFile::new().unwrap();
        {
            let file = std::fs::File::create(tmp.path()).unwrap();
            let mut zip = zip::ZipWriter::new(file);
            let options = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            zip.start_file("w.npy", options).unwrap();
            let npy = make_npy_v1(
                "{'descr': '<f4', 'fortran_order': False, 'shape': (1000,), }",
                &data,
            );
            zip.write_all(&npy).unwrap();
            zip.finish().unwrap();
        }

        // Step 1: inspect (cheap, header-only) reports the declared total.
        let info = inspect_npz(tmp.path()).unwrap();
        assert_eq!(info.total_bytes, 4000);
        assert_eq!(info.tensors.len(), 1);

        // Step 2: a host whose budget is below the inspected total rejects early
        // — and `parse_npz_with_limits` under that same budget agrees (Err). The
        // parse charges the `NPY` header too, so the inspected total is a
        // lower bound: a budget at/below it always rejects.
        let policy_budget = info.total_bytes - 1;
        assert!(info.total_bytes > policy_budget); // the host's cheap pre-check fires
        assert!(
            parse_npz_with_limits(
                tmp.path(),
                &ParseLimits::default().with_max_total_bytes(policy_budget)
            )
            .is_err()
        );

        // Step 3: a generous budget (> inspected total + header overhead) parses.
        assert!(
            parse_npz_with_limits(
                tmp.path(),
                &ParseLimits::default().with_max_total_bytes(info.total_bytes * 2)
            )
            .is_ok()
        );
    }

    // G36: inspect_npz overflow — large shape values saturate gracefully
    #[test]
    fn inspect_npz_overflow_saturates() {
        // Build NPZ with an array whose shape would overflow usize when
        // multiplied. inspect_npz should saturate rather than panic.
        let tmp = tempfile::NamedTempFile::new().unwrap();
        {
            let file = std::fs::File::create(tmp.path()).unwrap();
            let mut zip = zip::ZipWriter::new(file);
            let options = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            zip.start_file("huge.npy", options).unwrap();

            // Shape with dimensions that overflow when multiplied:
            // (usize::MAX, 2) → usize::MAX * 2 overflows
            // We encode this as NPY v1 header with huge shape.
            // But we can't actually store usize::MAX elements — the shape
            // in the header can claim anything. inspect_npz only reads
            // the header, not the data, so the file can be tiny.
            let shape_str = format!(
                "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, 2), }}",
                usize::MAX / 2 + 1
            );
            let npy = make_npy_v1(&shape_str, &[]); // no actual data
            zip.write_all(&npy).unwrap();
            zip.finish().unwrap();
        }

        let info = inspect_npz(tmp.path()).unwrap();
        assert_eq!(info.tensors.len(), 1);
        // Element count overflows → unwrap_or(usize::MAX) → saturating_mul
        // The byte_len should be usize::MAX (saturated)
        assert_eq!(info.tensors[0].byte_len, usize::MAX);
    }

    // -- Phase 4.7: reader-generic inspection --------------------------------

    /// Build a minimal in-memory NPZ archive containing a single STORED .npy
    /// entry with the given header and (zero-filled) data bytes. Returns the
    /// raw archive bytes — callers can wrap them in a Cursor to test the
    /// reader-generic API.
    fn make_in_memory_npz(arr_name: &str, header_str: &str, data: &[u8]) -> Vec<u8> {
        let mut buf: Vec<u8> = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut buf);
            let mut zip = zip::ZipWriter::new(cursor);
            let options = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            let entry_name = format!("{arr_name}.npy");
            zip.start_file(&entry_name, options).unwrap();
            let npy = make_npy_v1(header_str, data);
            zip.write_all(&npy).unwrap();
            zip.finish().unwrap();
        }
        buf
    }

    /// `inspect_npz_from_reader` over an in-memory `Cursor` returns the same
    /// `NpzInspectInfo` as `inspect_npz` over the same archive on disk.
    /// Locks the contract that the reader-generic and path-based APIs are
    /// substrate-equivalent — the substrate (file vs. cursor) cannot change
    /// the metadata. This is what downstream HTTP-range adapters rely on.
    #[test]
    fn inspect_from_reader_matches_path() {
        // Build a multi-array NPZ in memory: one F32 [2, 3] and one I64 [4].
        let mut buf: Vec<u8> = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut buf);
            let mut zip = zip::ZipWriter::new(cursor);
            let options = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);

            // Array 1: F32 [2, 3] = 24 bytes
            zip.start_file("weights.npy", options).unwrap();
            let npy1 = make_npy_v1(
                "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }",
                &[0u8; 24],
            );
            zip.write_all(&npy1).unwrap();

            // Array 2: I64 [4] = 32 bytes
            zip.start_file("indices.npy", options).unwrap();
            let npy2 = make_npy_v1(
                "{'descr': '<i8', 'fortran_order': False, 'shape': (4,), }",
                &[0u8; 32],
            );
            zip.write_all(&npy2).unwrap();

            zip.finish().unwrap();
        }

        // Write the same bytes to a temp file for the path-based comparison.
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), &buf).unwrap();

        let path_info = inspect_npz(tmp.path()).unwrap();
        let reader_info = inspect_npz_from_reader(std::io::Cursor::new(&buf)).unwrap();

        // Substrate-equivalence: every field of NpzInspectInfo matches.
        assert_eq!(path_info.tensors.len(), reader_info.tensors.len());
        assert_eq!(path_info.total_bytes, reader_info.total_bytes);
        assert_eq!(path_info.dtypes, reader_info.dtypes);
        for (a, b) in path_info.tensors.iter().zip(reader_info.tensors.iter()) {
            assert_eq!(a.name, b.name);
            assert_eq!(a.shape, b.shape);
            assert_eq!(a.dtype, b.dtype);
            assert_eq!(a.byte_len, b.byte_len);
        }

        // And spot-check the actual values to make sure we are not just
        // comparing two equal-but-wrong outputs.
        assert_eq!(reader_info.tensors.len(), 2);
        assert_eq!(reader_info.total_bytes, 24 + 32);
    }

    /// `inspect_npz_from_reader` returns `Ok` with no entries when handed a
    /// well-formed empty ZIP archive (i.e., the archive parses but contains
    /// no `.npy` payloads). Mirrors the existing `empty_npz_archive` test
    /// for the path-based variant.
    #[test]
    fn inspect_from_reader_empty_archive() {
        let mut buf: Vec<u8> = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut buf);
            let zip = zip::ZipWriter::new(cursor);
            zip.finish().unwrap();
        }
        let info = inspect_npz_from_reader(std::io::Cursor::new(&buf)).unwrap();
        assert!(info.tensors.is_empty());
        assert_eq!(info.total_bytes, 0);
        assert!(info.dtypes.is_empty());
    }

    /// `inspect_npz_from_reader` propagates Fortran-order rejection through
    /// the same code path as `inspect_npz`. Confirms the refactor did not
    /// silently lose the unsupported-format guard.
    #[test]
    fn inspect_from_reader_rejects_fortran_order() {
        let buf = make_in_memory_npz(
            "arr",
            "{'descr': '<f4', 'fortran_order': True, 'shape': (2, 2), }",
            &[0u8; 16],
        );
        let err = inspect_npz_from_reader(std::io::Cursor::new(&buf)).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("Fortran-order") || msg.contains("fortran"),
            "expected Fortran-order error, got: {msg}"
        );
    }

    // -- DoS hardening (Phase 6.6) -------------------------------------------

    /// A v2 `NPY` header declaring `header_len = u32::MAX` is rejected before
    /// the `vec![0u8; header_len]` allocation (Phase 6.6 Step 2). Mirrors the
    /// candle #3533 "tiny malicious header → expect `Err`" pattern: the input
    /// is 12 bytes yet would otherwise drive a 4 GiB allocation.
    #[test]
    fn header_len_cap_rejects_oversized_v2_header() {
        let mut npy = Vec::new();
        npy.extend_from_slice(NPY_MAGIC);
        npy.push(2); // major 2 → u32 header length
        npy.push(0); // minor
        npy.extend_from_slice(&u32::MAX.to_le_bytes());

        let mut reader = std::io::Cursor::new(&npy);
        let Err(err) = parse_npy_header(&mut reader, &mut Budget::unbounded()) else {
            panic!("expected error for oversized header length");
        };
        let msg = err.to_string();
        assert!(
            msg.contains("header length") && msg.contains("cap"),
            "expected header-length cap error, got: {msg}"
        );
    }

    /// A header just at the cap is accepted past the length gate (it then fails
    /// later on the truncated body), confirming the bound is `>` not `>=`.
    #[test]
    fn header_len_cap_accepts_boundary() {
        // header_len == NPY_MAX_HEADER_BYTES must pass the gate; we supply no
        // body, so the subsequent read_exact fails — but NOT with a cap error.
        let mut npy = Vec::new();
        npy.extend_from_slice(NPY_MAGIC);
        npy.push(2);
        npy.push(0);
        // CAST: usize → u32, NPY_MAX_HEADER_BYTES (1 MiB) fits in u32
        npy.extend_from_slice(&(NPY_MAX_HEADER_BYTES as u32).to_le_bytes());

        let mut reader = std::io::Cursor::new(&npy);
        let Err(err) = parse_npy_header(&mut reader, &mut Budget::unbounded()) else {
            panic!("expected truncated-body error past the cap gate");
        };
        let msg = err.to_string();
        assert!(
            !msg.contains("cap"),
            "boundary value must pass the cap gate, got cap error: {msg}"
        );
    }

    /// An array whose declared shape product × element size exceeds the
    /// `NPZ_MAX_ARRAY_BYTES` cap is rejected before `vec![0u8; data_bytes]`
    /// (Phase 6.6 Step 3). On 64-bit the cap fires; on 32-bit the existing
    /// `checked_mul` overflow guard fires first — both reject without
    /// allocating or panicking.
    #[test]
    fn array_bytes_cap_rejects_oversized_shape() {
        let header = NpyHeader {
            dtype: NpzDtype::F32,
            big_endian: false,
            fortran_order: false,
            // 3e9 elements × 4 bytes = 12 GiB > 8 GiB cap (and overflows a
            // 32-bit usize at the byte-count multiply).
            shape: vec![3_000_000_000usize],
        };
        // u64::MAX entry size so the absolute cap (not the entry-size guard)
        // is what rejects this shape.
        let mut empty = std::io::Cursor::new(Vec::new());
        let result = read_array_data(&mut empty, &header, u64::MAX, &mut Budget::unbounded());
        assert!(
            result.is_err(),
            "oversized declared array must be rejected, got Ok"
        );
    }

    /// A declared shape requiring more bytes than the ZIP entry holds is
    /// rejected before any allocation (Phase 6.7 Step 1) — the entry-size
    /// cross-check, mirroring the `GGUF` dequant path. Here a 16-byte entry is
    /// handed a header declaring 1000 `F32` elements (4000 bytes).
    #[test]
    fn array_bytes_rejected_above_entry_size() {
        let header = NpyHeader {
            dtype: NpzDtype::F32,
            big_endian: false,
            fortran_order: false,
            shape: vec![1000],
        };
        let mut empty = std::io::Cursor::new(Vec::new());
        let Err(err) = read_array_data(&mut empty, &header, 16, &mut Budget::unbounded()) else {
            panic!("over-declared shape vs entry size must be rejected");
        };
        let msg = err.to_string();
        assert!(
            msg.contains("declared ZIP entry size"),
            "expected entry-size rejection, got: {msg}"
        );
    }
}
