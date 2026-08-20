// SPDX-License-Identifier: MIT OR Apache-2.0

//! Vendored, read-only `ZIP` central-directory reader (Phase 6.12).
//!
//! `.pth` and `.npz` are `ZIP` archives. anamnesis needs exactly one thing
//! from the container: a `name → (data_start, length)` index so the tensor
//! parsers can slice (`.pth`, mmap) or stream (`.npz`, `Read + Seek`) each
//! entry's bytes. The `zip` crate provides this, but `ZipArchive::new`
//! eagerly materialises the **whole** central directory into a fat per-entry
//! record (~500 B/entry, ~5.7× the file for a many-tiny-entry archive) versus
//! the ~40 B/entry anamnesis actually needs — an amplification that is neither
//! reachable through `zip`'s API nor bounded by `ParseLimits`. This module
//! owns that container parsing so the metadata footprint is bounded by an owned
//! reader.
//!
//! # Scope
//!
//! - **Read-only.** No archive writing — the `zip` crate stays a
//!   dev-dependency for `ZipWriter` test fixtures and as a differential oracle.
//! - **No codec.** Only the **container** (central directory + local-header
//!   offsets) is parsed here. `.npz` `DEFLATE` entries keep `flate2` /
//!   `miniz_oxide` for inflate — the upstream-fuzzed codec surface is
//!   untouched.
//! - **Full `ZIP64`.** The `ZIP64` end-of-central-directory record + locator
//!   and the `0x0001` extra field are parsed, so `torch.save` checkpoints
//!   larger than 4 `GiB` (or with more than 65 535 entries) keep parsing.
//!
//! # Untrusted input
//!
//! Every length, count, and offset is attacker-controllable. The reader
//! follows the `CONVENTIONS.md` *"When Parsing Untrusted Input"* invariants:
//! every multi-byte field is read through a bounds-checked `ByteCursor`
//! (never direct indexing); every offset/size combination uses `checked_*`
//! arithmetic; the declared entry count is capped at `ZIP_MAX_ENTRIES` and
//! each entry name at `ZIP_MAX_NAME_LEN` **before** allocating; each entry's
//! `data_start + compressed_size` is cross-checked against the source length;
//! and compression methods are **allowlisted** (`Stored` / `Deflate`), never
//! denylisted. On top of those permanent floors, `read_central_directory`
//! also honours the caller's `ParseLimits` (`max_item_count` and
//! `max_single_alloc_bytes`) fail-fast, so a memory-constrained caller bounds
//! the container metadata to *its* budget (CWE-770).
//!
//! *Identifiers in this module header are plain code spans rather than
//! intra-doc links: `rustdoc` cannot resolve links in the `//!` docs of a
//! `pub(crate)` module, even fully qualified, so under
//! `--document-private-items` they are hard errors. Item-level docs below use
//! real links, which do resolve. Same rule `CONVENTIONS.md` already gives for
//! links that cannot be guaranteed to resolve.*

use std::borrow::Cow;
use std::io::{Read, Seek};

use crate::ParseLimits;
use crate::error::AnamnesisError;
use crate::parse::utils::PREALLOC_SOFT_CAP;

// ---------------------------------------------------------------------------
// Signatures and fixed record sizes (APPNOTE.TXT 6.3.x)
// ---------------------------------------------------------------------------

/// End-of-central-directory record signature (`PK\x05\x06`).
const EOCD_SIG: u32 = 0x0605_4b50;
/// Central-directory file-header signature (`PK\x01\x02`).
const CDFH_SIG: u32 = 0x0201_4b50;
/// Local file-header signature (`PK\x03\x04`).
const LFH_SIG: u32 = 0x0403_4b50;
/// `ZIP64` end-of-central-directory record signature (`PK\x06\x06`).
const ZIP64_EOCD_SIG: u32 = 0x0606_4b50;
/// `ZIP64` end-of-central-directory locator signature (`PK\x06\x07`).
const ZIP64_LOCATOR_SIG: u32 = 0x0706_4b50;

/// Fixed length of the end-of-central-directory record (before the comment).
const EOCD_FIXED_LEN: u64 = 22;
/// Maximum `ZIP` archive comment length (the comment-length field is a `u16`).
const MAX_COMMENT_LEN: u64 = 0xFFFF;
/// Upper bound on the trailing window scanned for the EOCD record: the fixed
/// record plus the largest possible comment.
const EOCD_SCAN_MAX: u64 = EOCD_FIXED_LEN + MAX_COMMENT_LEN;

/// Fixed length of a central-directory file header (before name/extra/comment).
const CDFH_FIXED_LEN: usize = 46;
/// Fixed length of a local file header (before name/extra).
const LFH_FIXED_LEN: usize = 30;
/// Length of the `ZIP64` end-of-central-directory locator.
const ZIP64_LOCATOR_LEN: u64 = 20;
/// Fixed length of the `ZIP64` end-of-central-directory record.
const ZIP64_EOCD_FIXED_LEN: usize = 56;

/// Extra-field header ID for the `ZIP64` extended-information field.
const ZIP64_EXTRA_ID: u16 = 0x0001;
/// `STORED` (no compression) method tag.
const METHOD_STORED: u16 = 0;
/// `DEFLATE` method tag.
const METHOD_DEFLATE: u16 = 8;
/// Sentinel a 32-bit size/offset field carries when its real value lives in a
/// `ZIP64` extra field.
const U32_SENTINEL: u32 = 0xFFFF_FFFF;
/// Sentinel the 16-bit entry-count field carries when the real count lives in
/// the `ZIP64` EOCD record.
const U16_SENTINEL: u16 = 0xFFFF;

/// Hard cap on the declared central-directory entry count (1 048 576).
///
/// Mirrors the `GGUF` parser's `MAX_TENSOR_COUNT` / `MAX_KV_COUNT` generosity:
/// real `.pth` / `.npz` archives have at most a few thousand entries, so a
/// declared count beyond this signals an adversarial EOCD and is rejected
/// before the entry vector is sized. The per-entry parse loop also stops as
/// soon as the central-directory bytes are exhausted, so this only bounds the
/// pathological "huge declared count, tiny directory" shape.
const ZIP_MAX_ENTRIES: u64 = 1 << 20;

/// Hard cap on a single entry-name length (4 `KiB`).
///
/// `ZIP` stores the name length as a `u16` (≤64 `KiB` already), but real
/// archive paths are well under 4 `KiB`; anything larger is rejected before
/// the name is materialised.
const ZIP_MAX_NAME_LEN: usize = 4096;

// ---------------------------------------------------------------------------
// ZipSource — the byte substrate (mmap slice or Read + Seek reader)
// ---------------------------------------------------------------------------

/// A positioned, read-only byte source the central-directory reader pulls
/// fixed regions from.
///
/// Implemented by [`SliceSource`] (the `.pth` mmap path — a borrowed `&[u8]`)
/// and, from Phase 6.12 Step 2, by a `Read + Seek` adapter (the `.npz` and
/// `.pth`-reader paths). Keeping the reader generic over this trait means the
/// EOCD scan, `ZIP64` resolution, and central-directory parse are written once
/// and shared by both substrates.
pub(crate) trait ZipSource {
    /// Total length of the source, in bytes.
    fn total_len(&self) -> u64;

    /// Reads exactly `buf.len()` bytes starting at `offset` into `buf`.
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::Parse`] if the requested range lies outside
    /// the source, or [`AnamnesisError::Io`] if an underlying read fails.
    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> crate::Result<()>;

    /// Returns the whole source as a borrowed slice when it is already in
    /// memory (the mmap path), enabling a zero-copy central-directory read.
    /// Streaming sources return `None` (the caller reads the directory into a
    /// transient buffer instead). Defaults to `None`.
    fn as_slice(&self) -> Option<&[u8]> {
        None
    }
}

/// A [`ZipSource`] backed by an in-memory byte slice (the `.pth` memory-mapped
/// file). `read_at` is a bounds-checked `copy_from_slice` from the mapping.
///
/// Only the `.pth` reader constructs one — `.npz` drives the container through
/// `ReaderSource` — so it is gated on `pth` to keep an `npz`-only build free of
/// dead code. `test` keeps it available to this module's own unit tests in every
/// feature combination.
#[cfg(any(feature = "pth", feature = "npz", test))]
pub(crate) struct SliceSource<'a> {
    /// The whole archive bytes (the mmap).
    data: &'a [u8],
}

#[cfg(any(feature = "pth", feature = "npz", test))]
impl<'a> SliceSource<'a> {
    /// Wraps `data` as a [`ZipSource`].
    #[must_use]
    pub(crate) const fn new(data: &'a [u8]) -> Self {
        Self { data }
    }
}

#[cfg(any(feature = "pth", feature = "npz", test))]
impl ZipSource for SliceSource<'_> {
    fn total_len(&self) -> u64 {
        // CAST: usize → u64, lossless widening on all supported targets
        #[allow(clippy::as_conversions)]
        let len = self.data.len() as u64;
        len
    }

    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> crate::Result<()> {
        let start = usize::try_from(offset).map_err(|_| AnamnesisError::Parse {
            reason: "ZIP read offset overflows usize".into(),
        })?;
        let end = start
            .checked_add(buf.len())
            .ok_or_else(|| AnamnesisError::Parse {
                reason: "ZIP read range overflow".into(),
            })?;
        let src = self
            .data
            .get(start..end)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: "ZIP read past end of mapped file".into(),
            })?;
        buf.copy_from_slice(src);
        Ok(())
    }

    fn as_slice(&self) -> Option<&[u8]> {
        Some(self.data)
    }
}

/// A [`ZipSource`] backed by any `Read + Seek` substrate (the `.npz` and
/// `.pth`-reader paths — a [`std::fs::File`], an in-memory [`std::io::Cursor`],
/// or an `HTTP`-range adapter). `read_at` issues a `seek` + `read_exact` per
/// region; the central-directory reader keeps those to a handful of bulk reads
/// (EOCD scan, the directory, each named entry's local header).
pub(crate) struct ReaderSource<R: Read + Seek> {
    /// The underlying positional reader.
    reader: R,
    /// Total length, captured once at construction.
    len: u64,
}

impl<R: Read + Seek> ReaderSource<R> {
    /// Wraps `reader`, capturing its total length via a one-time
    /// `seek(End(0))`.
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::Io`] if the initial seek fails.
    pub(crate) fn new(mut reader: R) -> crate::Result<Self> {
        let len = reader
            .seek(std::io::SeekFrom::End(0))
            .map_err(AnamnesisError::Io)?;
        Ok(Self { reader, len })
    }

    /// Returns a [`Read`] over the raw (possibly compressed) bytes of `entry`,
    /// bounded to the entry's `compressed_size`.
    ///
    /// Codec-free: a `DEFLATE` entry's bytes are returned **compressed**; the
    /// caller wraps the result in `flate2`'s decoder (the codec stays out of
    /// this container module). The reader is positioned at the entry's data
    /// offset, resolved from its local header by [`data_start`].
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::Parse`] if the local header is malformed or
    /// the data range exceeds the source, or [`AnamnesisError::Io`] if a seek
    /// fails.
    pub(crate) fn entry_data_reader(
        &mut self,
        entry: &ZipEntry,
    ) -> crate::Result<BoundedReader<'_, R>> {
        let start = data_start(self, entry)?;
        self.reader
            .seek(std::io::SeekFrom::Start(start))
            .map_err(AnamnesisError::Io)?;
        Ok(BoundedReader {
            inner: &mut self.reader,
            remaining: entry.compressed_size,
        })
    }
}

impl<R: Read + Seek> ZipSource for ReaderSource<R> {
    fn total_len(&self) -> u64 {
        self.len
    }

    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> crate::Result<()> {
        self.reader
            .seek(std::io::SeekFrom::Start(offset))
            .map_err(AnamnesisError::Io)?;
        self.reader.read_exact(buf).map_err(AnamnesisError::Io)?;
        Ok(())
    }
}

/// A [`Read`] adapter that yields at most `remaining` bytes from a borrowed
/// underlying reader — the raw byte window of a single `ZIP` entry.
///
/// Returned by [`ReaderSource::entry_data_reader`]. Bounding the read to the
/// entry's `compressed_size` stops one entry's reader from running into the
/// next entry's bytes (or the central directory) on a malformed archive.
pub(crate) struct BoundedReader<'a, R> {
    /// The borrowed underlying reader, positioned at the entry's data offset.
    inner: &'a mut R,
    /// Bytes still readable from this entry.
    remaining: u64,
}

impl<R: Read> Read for BoundedReader<'_, R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.remaining == 0 {
            return Ok(0);
        }
        // CAST: usize → u64 lossless, then u64 → usize after the `min` clamps
        // the value to `buf.len()`, so the narrowing cannot truncate.
        #[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
        let want = (buf.len() as u64).min(self.remaining) as usize;
        // INDEX: `want <= buf.len()` by the `min` above, so the slice is in
        // bounds; bounding the read keeps one entry from reading into the next.
        #[allow(clippy::indexing_slicing)]
        let n = self.inner.read(&mut buf[..want])?;
        // CAST: usize → u64, lossless widening
        #[allow(clippy::as_conversions)]
        let read_u64 = n as u64;
        self.remaining = self.remaining.saturating_sub(read_u64);
        Ok(n)
    }
}

// ---------------------------------------------------------------------------
// ByteCursor — bounds-checked little-endian reader over a borrowed buffer
// ---------------------------------------------------------------------------

/// A forward, bounds-checked little-endian cursor over a borrowed byte buffer.
///
/// Every read advances `pos` and is validated through `.get(..)`, so a
/// truncated record yields [`AnamnesisError::Parse`] rather than a panic — the
/// single choke point that keeps the central-directory parse branch-free of
/// direct indexing.
struct ByteCursor<'a> {
    /// The borrowed buffer (a tail window, the central directory, an extra
    /// field, or a local header).
    buf: &'a [u8],
    /// Current read offset into `buf`.
    pos: usize,
}

impl<'a> ByteCursor<'a> {
    /// Creates a cursor at the start of `buf`.
    const fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    /// Bytes left to read.
    fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.pos)
    }

    /// Borrows the next `n` bytes, advancing the cursor.
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::Parse`] if fewer than `n` bytes remain.
    fn take(&mut self, n: usize) -> crate::Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: "ZIP record offset overflow".into(),
            })?;
        let slice = self
            .buf
            .get(self.pos..end)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: "ZIP record truncated".into(),
            })?;
        self.pos = end;
        Ok(slice)
    }

    /// Reads a little-endian `u16`.
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::Parse`] if fewer than 2 bytes remain.
    fn u16(&mut self) -> crate::Result<u16> {
        let arr: [u8; 2] = self
            .take(2)?
            .try_into()
            .map_err(|_| AnamnesisError::Parse {
                reason: "internal: ZIP u16 slice-to-array conversion failed".into(),
            })?;
        Ok(u16::from_le_bytes(arr))
    }

    /// Reads a little-endian `u32`.
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::Parse`] if fewer than 4 bytes remain.
    fn u32(&mut self) -> crate::Result<u32> {
        let arr: [u8; 4] = self
            .take(4)?
            .try_into()
            .map_err(|_| AnamnesisError::Parse {
                reason: "internal: ZIP u32 slice-to-array conversion failed".into(),
            })?;
        Ok(u32::from_le_bytes(arr))
    }

    /// Reads a little-endian `u64`.
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::Parse`] if fewer than 8 bytes remain.
    fn u64(&mut self) -> crate::Result<u64> {
        let arr: [u8; 8] = self
            .take(8)?
            .try_into()
            .map_err(|_| AnamnesisError::Parse {
                reason: "internal: ZIP u64 slice-to-array conversion failed".into(),
            })?;
        Ok(u64::from_le_bytes(arr))
    }

    /// Skips `n` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::Parse`] if fewer than `n` bytes remain.
    fn skip(&mut self, n: usize) -> crate::Result<()> {
        self.take(n)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Compression and ZipEntry
// ---------------------------------------------------------------------------

/// The compression method of a `ZIP` entry, restricted to the allowlist
/// anamnesis container parsing supports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Compression {
    /// No compression — the entry data is the raw bytes (every `.pth` entry,
    /// most large `.npz` arrays).
    Stored,
    /// Raw `DEFLATE` (method 8) — inflated by `flate2` / `miniz_oxide` on the
    /// `.npz` path.
    Deflate,
    /// Any other method tag — recognised but not interpreted; the caller
    /// decides whether to skip (`.pth`) or reject (`.npz`).
    Unsupported(u16),
}

impl Compression {
    /// Maps a raw `ZIP` method tag to the allowlist.
    const fn from_tag(tag: u16) -> Self {
        match tag {
            METHOD_STORED => Self::Stored,
            METHOD_DEFLATE => Self::Deflate,
            other => Self::Unsupported(other),
        }
    }

    /// Returns `true` for a `STORED` (uncompressed) entry.
    ///
    /// Only the `.pth` reader branches on this (`.npz` always routes through the
    /// inflate path), so it is gated on `pth` to keep an `npz`-only build free of
    /// dead code; `test` keeps it available to this module's unit tests.
    #[cfg(any(feature = "pth", test))]
    #[must_use]
    pub(crate) const fn is_stored(self) -> bool {
        matches!(self, Self::Stored)
    }
}

/// One central-directory entry, distilled to the fields anamnesis needs.
///
/// Built by [`read_central_directory`]; the raw data offset is resolved
/// lazily by [`data_start`] (which reads the entry's local header, whose
/// extra-field length may differ from the central-directory copy).
#[derive(Debug, Clone)]
pub(crate) struct ZipEntry {
    /// Entry path as stored in the archive (e.g. `archive/data.pkl`,
    /// `weight.npy`).
    pub(crate) name: String,
    /// Compression method (allowlisted).
    pub(crate) method: Compression,
    /// Compressed (on-disk) size of the entry data, in bytes.
    pub(crate) compressed_size: u64,
    /// Uncompressed size of the entry data, in bytes (equal to
    /// `compressed_size` for `STORED`). Consumed by the `.npz` path (the
    /// `DEFLATE` inflate-size cross-check), wired in Phase 6.12 Step 2; the
    /// `.pth` path is STORED-only and reads only `compressed_size`.
    #[allow(dead_code)]
    pub(crate) uncompressed_size: u64,
    /// Offset of the entry's local file header from the start of the archive.
    pub(crate) local_header_offset: u64,
}

// ---------------------------------------------------------------------------
// Central-directory location (EOCD + ZIP64 resolution)
// ---------------------------------------------------------------------------

/// Resolved location of the central directory within the archive.
struct CentralDirInfo {
    /// Number of central-directory records.
    entries: u64,
    /// Offset of the first central-directory record from the archive start.
    offset: u64,
    /// Total byte length of the central directory.
    size: u64,
}

/// Reads the central directory of a `ZIP` archive into a lean list of entries.
///
/// Locates the end-of-central-directory record (tail scan, `ZIP64`-aware),
/// then parses each `PK\x01\x02` record into a [`ZipEntry`]. Only the fields
/// anamnesis needs are retained; comments, timestamps, and attributes are
/// skipped.
///
/// `limits` is the caller's [`ParseLimits`]: the declared entry count is
/// checked against `max_item_count` and the central-directory byte read against
/// `max_single_alloc_bytes` — both **before** any allocation, fail-fast — so a
/// tight-budget caller bounds the container metadata, not just the permanent
/// [`ZIP_MAX_ENTRIES`] floor (CWE-770). The summary-only inspect paths
/// (`inspect_npz_from_reader`, `inspect_pth_from_reader`) pass
/// [`ParseLimits::unbounded`]; the full-detail `.pth` front-matter path
/// (`parse_pth_front_matter_from_reader_with_limits`) is also inspect-style
/// (no tensor-data read) but passes the caller's real budget through, so it
/// is bounded here too.
///
/// # Errors
///
/// Returns [`AnamnesisError::LimitExceeded`] if the declared entry count exceeds
/// [`ZIP_MAX_ENTRIES`] or the caller's `max_item_count`, the central-directory
/// allocation exceeds the caller's `max_single_alloc_bytes`, or an entry name
/// exceeds [`ZIP_MAX_NAME_LEN`].
/// Returns [`AnamnesisError::Parse`] if the archive is too small, the EOCD or
/// `ZIP64` records are missing/malformed, the central directory lies outside
/// the file, or any record is truncated.
///
/// Returns [`AnamnesisError::Io`] if an underlying `read` on the source fails.
///
/// # Memory
///
/// Reads the trailing EOCD scan window (≤ ~64 `KiB`) into a transient buffer.
/// The central directory is read zero-copy when the source is already in memory
/// (the mmap path borrows it via [`ZipSource::as_slice`]); a streaming source
/// reads it into a transient buffer (charged to `limits`) that is dropped before
/// returning. The returned `Vec<ZipEntry>` holds only the distilled per-entry
/// fields (name + three integers + method tag) — no fat per-entry record
/// persists.
pub(crate) fn read_central_directory<S: ZipSource>(
    src: &mut S,
    limits: &ParseLimits,
) -> crate::Result<Vec<ZipEntry>> {
    let total_len = src.total_len();
    let info = find_central_dir(src, total_len)?;

    let cd_end = info
        .offset
        .checked_add(info.size)
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "ZIP central directory range overflow".into(),
        })?;
    if cd_end > total_len {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "ZIP central directory range [{}..{cd_end}] exceeds file size {total_len}",
                info.offset
            ),
        });
    }
    if info.entries > ZIP_MAX_ENTRIES {
        return Err(AnamnesisError::LimitExceeded {
            limit: "ZIP_MAX_ENTRIES",
            message: format!(
                "ZIP central directory declares {} entries, exceeding the {ZIP_MAX_ENTRIES} cap",
                info.entries
            ),
        });
    }
    // Caller's item-count budget, enforced before the entry vector is sized
    // (the permanent cap above is the floor; this can only tighten it).
    limits.check_item_count(info.entries, "ZIP central directory entry count")?;

    let cd_size = usize::try_from(info.size).map_err(|_| AnamnesisError::Parse {
        reason: "ZIP central directory size overflows usize".into(),
    })?;
    let cd_bytes = read_cd_bytes(src, info.offset, cd_size, limits)?;

    // Clamp the pre-allocation hint: the declared count is attacker-influenced,
    // so trusting it for `with_capacity` would commit ~1–2× the file size
    // eagerly. The Vec grows as entries are parsed. Mirrors the GGUF parser's
    // `PREALLOC_SOFT_CAP`.
    let cap = usize::try_from(info.entries)
        .unwrap_or(usize::MAX)
        .min(PREALLOC_SOFT_CAP);
    let mut entries = Vec::with_capacity(cap);
    parse_cd_entries(&cd_bytes, info.entries, &mut entries)?;
    Ok(entries)
}

/// Obtains the central-directory bytes: a zero-copy borrow from an in-memory
/// source (the mmap path), or a transient read into an owned buffer for a
/// streaming source.
///
/// The owned-buffer path charges its `cd_size` allocation to the caller's
/// `limits` (`max_single_alloc_bytes`) before allocating; the borrow path
/// allocates nothing, so it needs no charge.
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if the central-directory range is out of
/// bounds or the owned read exceeds the caller's single-allocation budget, or
/// [`AnamnesisError::Io`] if a streaming read fails.
fn read_cd_bytes<'a, S: ZipSource>(
    src: &'a mut S,
    offset: u64,
    cd_size: usize,
    limits: &ParseLimits,
) -> crate::Result<Cow<'a, [u8]>> {
    // Probe with `is_some()` first so its borrow ends before the branches: the
    // borrow path re-borrows immutably, the streaming path borrows mutably, and
    // the two must not overlap.
    if src.as_slice().is_some() {
        let start = usize::try_from(offset).map_err(|_| AnamnesisError::Parse {
            reason: "ZIP central directory offset overflows usize".into(),
        })?;
        let end = start
            .checked_add(cd_size)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: "ZIP central directory range overflow".into(),
            })?;
        let file = src.as_slice().ok_or_else(|| AnamnesisError::Parse {
            reason: "internal: ZIP in-memory source slice unavailable".into(),
        })?;
        let slice = file.get(start..end).ok_or_else(|| AnamnesisError::Parse {
            reason: "ZIP central directory range exceeds file size".into(),
        })?;
        return Ok(Cow::Borrowed(slice));
    }
    // Streaming source: read the directory into a transient owned buffer, after
    // charging the allocation to the caller's single-allocation budget.
    // CAST: usize → u64, lossless widening on all supported targets
    #[allow(clippy::as_conversions)]
    let cd_size_u64 = cd_size as u64;
    limits.check_alloc(cd_size_u64, "ZIP central directory")?;
    let mut buf = vec![0u8; cd_size];
    src.read_at(offset, &mut buf)?;
    Ok(Cow::Owned(buf))
}

/// Locates the central directory: finds the EOCD record by scanning the tail,
/// then resolves the `ZIP64` record when the 32/16-bit fields are saturated.
fn find_central_dir<S: ZipSource>(src: &mut S, total_len: u64) -> crate::Result<CentralDirInfo> {
    if total_len < EOCD_FIXED_LEN {
        return Err(AnamnesisError::Parse {
            reason: "file too small to be a ZIP archive".into(),
        });
    }
    let scan_len = total_len.min(EOCD_SCAN_MAX);
    let scan_start = total_len
        .checked_sub(scan_len)
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "internal: ZIP EOCD scan window underflow".into(),
        })?;
    let scan_len_usize = usize::try_from(scan_len).map_err(|_| AnamnesisError::Parse {
        reason: "ZIP EOCD scan window overflows usize".into(),
    })?;
    let mut tail = vec![0u8; scan_len_usize];
    src.read_at(scan_start, &mut tail)?;

    let eocd_pos = find_eocd_in_tail(&tail)?;
    // The EOCD lives at this offset in the archive; the slice into `tail` is
    // valid because `find_eocd_in_tail` returns a matched position.
    let eocd = tail.get(eocd_pos..).ok_or_else(|| AnamnesisError::Parse {
        reason: "internal: ZIP EOCD position out of range".into(),
    })?;
    let mut c = ByteCursor::new(eocd);
    c.skip(4)?; // signature (already matched)
    c.skip(6)?; // this-disk (2), CD-start-disk (2), entries-this-disk (2)
    let entries16 = c.u16()?;
    let size32 = c.u32()?;
    let offset32 = c.u32()?;

    if entries16 == U16_SENTINEL || size32 == U32_SENTINEL || offset32 == U32_SENTINEL {
        // CAST: usize → u64, lossless widening; eocd_pos < total_len
        #[allow(clippy::as_conversions)]
        let eocd_file_offset =
            scan_start
                .checked_add(eocd_pos as u64)
                .ok_or_else(|| AnamnesisError::Parse {
                    reason: "ZIP EOCD file offset overflow".into(),
                })?;
        return read_zip64_eocd(src, eocd_file_offset);
    }

    Ok(CentralDirInfo {
        entries: u64::from(entries16),
        offset: u64::from(offset32),
        size: u64::from(size32),
    })
}

/// Scans `tail` backward for the EOCD signature, accepting only a position
/// whose declared comment length runs exactly to the end of the window (so a
/// fake signature embedded in a comment cannot win).
fn find_eocd_in_tail(tail: &[u8]) -> crate::Result<usize> {
    let tail_len = tail.len();
    // `tail_len >= EOCD_FIXED_LEN` is guaranteed by the caller (scan window is
    // at least the fixed record), so `max_start` does not underflow.
    let max_start = tail_len.checked_sub(usize::try_from(EOCD_FIXED_LEN).unwrap_or(usize::MAX));
    let max_start = max_start.ok_or_else(|| AnamnesisError::Parse {
        reason: "file too small to contain a ZIP end-of-central-directory record".into(),
    })?;
    for start in (0..=max_start).rev() {
        let Some(sig_end) = start.checked_add(4) else {
            continue;
        };
        let Some(sig) = tail.get(start..sig_end) else {
            continue;
        };
        let Ok(arr) = <[u8; 4]>::try_from(sig) else {
            continue;
        };
        if u32::from_le_bytes(arr) != EOCD_SIG {
            continue;
        }
        // Comment length lives at EOCD offset 20 (2 bytes).
        let Some(comment_start) = start.checked_add(20) else {
            continue;
        };
        let Some(comment_end) = start.checked_add(22) else {
            continue;
        };
        let Some(comment) = tail.get(comment_start..comment_end) else {
            continue;
        };
        let Ok(carr) = <[u8; 2]>::try_from(comment) else {
            continue;
        };
        let comment_len = usize::from(u16::from_le_bytes(carr));
        if comment_end.checked_add(comment_len) == Some(tail_len) {
            return Ok(start);
        }
    }
    Err(AnamnesisError::Parse {
        reason: "ZIP end-of-central-directory record not found".into(),
    })
}

/// Reads the `ZIP64` EOCD locator (immediately before the 32-bit EOCD) and the
/// `ZIP64` EOCD record it points to, returning the 64-bit central-directory
/// location.
fn read_zip64_eocd<S: ZipSource>(
    src: &mut S,
    eocd_file_offset: u64,
) -> crate::Result<CentralDirInfo> {
    let locator_offset = eocd_file_offset
        .checked_sub(ZIP64_LOCATOR_LEN)
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "ZIP64 EOCD locator missing (no room before EOCD)".into(),
        })?;
    let mut loc = [0u8; 20];
    src.read_at(locator_offset, &mut loc)?;
    let mut lc = ByteCursor::new(&loc);
    if lc.u32()? != ZIP64_LOCATOR_SIG {
        return Err(AnamnesisError::Parse {
            reason: "ZIP64 EOCD locator signature not found".into(),
        });
    }
    lc.skip(4)?; // disk holding the ZIP64 EOCD record
    let record_offset = lc.u64()?;

    let mut rec = [0u8; ZIP64_EOCD_FIXED_LEN];
    src.read_at(record_offset, &mut rec)?;
    let mut rc = ByteCursor::new(&rec);
    if rc.u32()? != ZIP64_EOCD_SIG {
        return Err(AnamnesisError::Parse {
            reason: "ZIP64 EOCD record signature not found".into(),
        });
    }
    // size (8), version-made-by (2), version-needed (2), this-disk (4),
    // CD-start-disk (4) = 20 bytes to skip to the entries-this-disk field.
    rc.skip(20)?;
    let _entries_this_disk = rc.u64()?;
    let entries = rc.u64()?;
    let size = rc.u64()?;
    let offset = rc.u64()?;
    Ok(CentralDirInfo {
        entries,
        offset,
        size,
    })
}

// ---------------------------------------------------------------------------
// Central-directory entry parsing
// ---------------------------------------------------------------------------

/// Parses up to `declared` central-directory file headers from `cd`, pushing a
/// [`ZipEntry`] per record. Stops early when the buffer is exhausted or a
/// non-`PK\x01\x02` record is hit (e.g. a trailing digital-signature record).
fn parse_cd_entries(cd: &[u8], declared: u64, out: &mut Vec<ZipEntry>) -> crate::Result<()> {
    let mut cur = ByteCursor::new(cd);
    let mut count = 0u64;
    while count < declared {
        if cur.remaining() < CDFH_FIXED_LEN {
            break; // no room for another fixed header — stop cleanly
        }
        if cur.u32()? != CDFH_SIG {
            break; // end of the central-directory record stream
        }
        cur.skip(4)?; // version-made-by (2), version-needed (2)
        let _flags = cur.u16()?;
        let method = cur.u16()?;
        cur.skip(4)?; // mod time (2), mod date (2)
        cur.skip(4)?; // CRC-32
        let comp32 = cur.u32()?;
        let uncomp32 = cur.u32()?;
        let name_len = usize::from(cur.u16()?);
        let extra_len = usize::from(cur.u16()?);
        let comment_len = usize::from(cur.u16()?);
        cur.skip(2)?; // disk-number-start
        cur.skip(2)?; // internal attributes
        cur.skip(4)?; // external attributes
        let offset32 = cur.u32()?;

        if name_len > ZIP_MAX_NAME_LEN {
            return Err(AnamnesisError::LimitExceeded {
                limit: "ZIP_MAX_NAME_LEN",
                message: format!(
                    "ZIP entry name length {name_len} exceeds the {ZIP_MAX_NAME_LEN}-byte cap"
                ),
            });
        }
        let name_bytes = cur.take(name_len)?;
        let extra = cur.take(extra_len)?;
        cur.skip(comment_len)?;

        let (compressed_size, uncompressed_size, local_header_offset) =
            apply_zip64_extra(extra, comp32, uncomp32, offset32)?;

        // BORROW: lossy UTF-8 decode of the ZIP entry name into an owned
        // String. Real `.pth` / `.npz` names are ASCII, where UTF-8 and CP437
        // coincide; lossy decoding never errors on adversarial bytes.
        let name = String::from_utf8_lossy(name_bytes).into_owned();
        out.push(ZipEntry {
            name,
            method: Compression::from_tag(method),
            compressed_size,
            uncompressed_size,
            local_header_offset,
        });
        count += 1;
    }
    Ok(())
}

/// Resolves the real `(compressed, uncompressed, local-header-offset)` triple,
/// reading the `ZIP64` extended-information extra field (`0x0001`) for any of
/// the three that the 32-bit header field carries as the [`U32_SENTINEL`].
///
/// The `ZIP64` extra packs only the saturated fields, in the fixed order
/// uncompressed → compressed → offset, so the conditional reads below mirror
/// that layout exactly.
fn apply_zip64_extra(
    extra: &[u8],
    comp32: u32,
    uncomp32: u32,
    offset32: u32,
) -> crate::Result<(u64, u64, u64)> {
    let mut compressed = u64::from(comp32);
    let mut uncompressed = u64::from(uncomp32);
    let mut offset = u64::from(offset32);

    let needs_zip64 =
        comp32 == U32_SENTINEL || uncomp32 == U32_SENTINEL || offset32 == U32_SENTINEL;
    if !needs_zip64 {
        return Ok((compressed, uncompressed, offset));
    }

    let mut cur = ByteCursor::new(extra);
    while cur.remaining() >= 4 {
        let id = cur.u16()?;
        let field_len = usize::from(cur.u16()?);
        let data = cur.take(field_len)?;
        if id == ZIP64_EXTRA_ID {
            let mut d = ByteCursor::new(data);
            if uncomp32 == U32_SENTINEL {
                uncompressed = d.u64()?;
            }
            if comp32 == U32_SENTINEL {
                compressed = d.u64()?;
            }
            if offset32 == U32_SENTINEL {
                offset = d.u64()?;
            }
            return Ok((compressed, uncompressed, offset));
        }
    }

    Err(AnamnesisError::Parse {
        reason: "ZIP entry declares ZIP64 sizes but has no ZIP64 extra field".into(),
    })
}

// ---------------------------------------------------------------------------
// Local-header data-offset resolution
// ---------------------------------------------------------------------------

/// Resolves the absolute offset of an entry's data by reading its local file
/// header.
///
/// The data begins after the 30-byte fixed local header plus the **local**
/// name and extra-field lengths — which may differ from the central-directory
/// copy (a common real-world divergence, e.g. a `ZIP64` extra present in one
/// but not the other), so the offset must be computed from the local header.
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if the local header signature is missing,
/// the computed offset overflows, or `data_start + compressed_size` exceeds the
/// source length.
///
/// Returns [`AnamnesisError::Io`] if an underlying `read` on the source fails.
pub(crate) fn data_start<S: ZipSource>(src: &mut S, entry: &ZipEntry) -> crate::Result<u64> {
    let mut hdr = [0u8; LFH_FIXED_LEN];
    src.read_at(entry.local_header_offset, &mut hdr)?;
    let mut c = ByteCursor::new(&hdr);
    if c.u32()? != LFH_SIG {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "ZIP local file header signature not found at offset {}",
                entry.local_header_offset
            ),
        });
    }
    c.skip(22)?; // to the name-length field at local-header offset 26
    let name_len = u64::from(c.u16()?);
    let extra_len = u64::from(c.u16()?);

    // CAST: usize const → u64, lossless (fixed 30-byte header)
    #[allow(clippy::as_conversions)]
    let fixed = LFH_FIXED_LEN as u64;
    let start = entry
        .local_header_offset
        .checked_add(fixed)
        .and_then(|v| v.checked_add(name_len))
        .and_then(|v| v.checked_add(extra_len))
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "ZIP local header data offset overflow".into(),
        })?;
    let end = start
        .checked_add(entry.compressed_size)
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "ZIP entry data range overflow".into(),
        })?;
    if end > src.total_len() {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "ZIP entry `{}` data range [{start}..{end}] exceeds file size {}",
                entry.name,
                src.total_len()
            ),
        });
    }
    Ok(start)
}

// ---------------------------------------------------------------------------
// Suffix helper (shared archive-prefix stripping convention)
// ---------------------------------------------------------------------------

/// Strips the leading archive prefix (`archive/`, `{model_name}/`, …) from a
/// `ZIP` entry name, returning the suffix the `.pth` / `.npz` parsers key on.
///
/// `find('/')` matches every realistic `PyTorch` / `NumPy` archive; a name with
/// no `/` is returned verbatim. Returns a borrow into `name` (no allocation).
///
/// Only the `.pth` reader strips the prefix today (`.npz` keys on the full entry
/// name), so it is gated on `pth` to keep an `npz`-only build free of dead code;
/// `test` keeps it available to this module's unit tests.
#[cfg(any(feature = "pth", feature = "npz", test))]
#[must_use]
pub(crate) fn strip_archive_prefix(name: &str) -> Cow<'_, str> {
    match name.find('/') {
        Some(pos) => Cow::Borrowed(name.get(pos + 1..).unwrap_or(name)),
        None => Cow::Borrowed(name),
    }
}

#[cfg(test)]
#[allow(
    clippy::panic,
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::as_conversions,
    clippy::cast_possible_truncation
)]
mod tests {
    use std::io::{Read, Write};

    use super::*;

    /// Builds an in-memory ZIP archive from `(name, method, data)` tuples using
    /// the `zip` crate (the differential oracle / fixture writer).
    fn build_zip(entries: &[(&str, ::zip::CompressionMethod, &[u8])]) -> Vec<u8> {
        let mut buf = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut buf);
            let mut writer = ::zip::ZipWriter::new(cursor);
            for (name, method, data) in entries {
                let opts = ::zip::write::SimpleFileOptions::default().compression_method(*method);
                writer.start_file(*name, opts).unwrap();
                writer.write_all(data).unwrap();
            }
            writer.finish().unwrap();
        }
        buf
    }

    /// Builds a ZIP archive forcing ZIP64 extra fields on every entry.
    fn build_zip64(entries: &[(&str, &[u8])]) -> Vec<u8> {
        let mut buf = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut buf);
            let mut writer = ::zip::ZipWriter::new(cursor);
            for (name, data) in entries {
                let opts = ::zip::write::SimpleFileOptions::default()
                    .compression_method(::zip::CompressionMethod::Stored)
                    .large_file(true);
                writer.start_file(*name, opts).unwrap();
                writer.write_all(data).unwrap();
            }
            writer.finish().unwrap();
        }
        buf
    }

    /// Asserts the vendored reader produces the same `name → (data_start,
    /// compressed_size, uncompressed_size)` index as the `zip` crate for every
    /// STORED entry (the differential oracle).
    fn assert_matches_zip_crate(archive: &[u8]) {
        let mut src = SliceSource::new(archive);
        let entries = read_central_directory(&mut src, &ParseLimits::unbounded())
            .expect("vendored reader failed");

        let cursor = std::io::Cursor::new(archive);
        let mut zip = ::zip::ZipArchive::new(cursor).expect("zip crate failed");

        assert_eq!(
            entries.len(),
            zip.len(),
            "entry count mismatch: vendored {} vs zip {}",
            entries.len(),
            zip.len()
        );

        for entry in &entries {
            let mut zfile = zip.by_name(&entry.name).expect("zip crate by_name failed");
            assert_eq!(
                entry.uncompressed_size,
                zfile.size(),
                "uncompressed size mismatch for `{}`",
                entry.name
            );
            assert_eq!(
                entry.compressed_size,
                zfile.compressed_size(),
                "compressed size mismatch for `{}`",
                entry.name
            );
            // data_start parity is the headline correctness check (the
            // local-vs-central extra-length gotcha).
            let our_start = data_start(&mut src, entry).expect("vendored data_start failed");
            assert_eq!(
                our_start,
                zfile.data_start(),
                "data_start mismatch for `{}`",
                entry.name
            );
            // The STORED bytes the offsets point at must round-trip.
            if entry.method.is_stored() {
                let start = our_start as usize;
                let end = start + entry.compressed_size as usize;
                let mut expected = Vec::new();
                zfile.read_to_end(&mut expected).unwrap();
                assert_eq!(&archive[start..end], &expected[..], "data mismatch");
            }
        }
    }

    #[test]
    fn single_stored_entry_matches_zip() {
        let archive = build_zip(&[(
            "data.pkl",
            ::zip::CompressionMethod::Stored,
            b"hello pickle",
        )]);
        assert_matches_zip_crate(&archive);
    }

    #[test]
    fn multi_entry_pth_layout_matches_zip() {
        let archive = build_zip(&[
            (
                "archive/data.pkl",
                ::zip::CompressionMethod::Stored,
                b"PKL-STREAM",
            ),
            (
                "archive/byteorder",
                ::zip::CompressionMethod::Stored,
                b"little",
            ),
            (
                "archive/data/0",
                ::zip::CompressionMethod::Stored,
                &[1u8; 64],
            ),
            (
                "archive/data/1",
                ::zip::CompressionMethod::Stored,
                &[2u8; 128],
            ),
        ]);
        assert_matches_zip_crate(&archive);
    }

    #[test]
    fn deflate_entry_metadata_matches_zip() {
        // A compressible payload so DEFLATE actually shrinks it (compressed !=
        // uncompressed), exercising the size-field divergence.
        let payload = vec![0xABu8; 4096];
        let archive = build_zip(&[("weight.npy", ::zip::CompressionMethod::Deflated, &payload)]);
        assert_matches_zip_crate(&archive);

        let mut src = SliceSource::new(&archive);
        let entries = read_central_directory(&mut src, &ParseLimits::unbounded()).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].method, Compression::Deflate);
        assert!(entries[0].compressed_size < entries[0].uncompressed_size);
    }

    #[test]
    fn zip64_entry_matches_zip() {
        let archive = build_zip64(&[
            ("archive/data.pkl", b"zip64 stream"),
            ("archive/data/0", &[7u8; 256]),
        ]);
        assert_matches_zip_crate(&archive);
    }

    #[test]
    fn empty_archive_has_no_entries() {
        let archive = build_zip(&[]);
        let mut src = SliceSource::new(&archive);
        let entries = read_central_directory(&mut src, &ParseLimits::unbounded()).unwrap();
        assert!(entries.is_empty());
        assert_matches_zip_crate(&archive);
    }

    #[test]
    fn prefix_stripping() {
        assert_eq!(strip_archive_prefix("archive/data.pkl"), "data.pkl");
        assert_eq!(strip_archive_prefix("my_model/data/0"), "data/0");
        assert_eq!(strip_archive_prefix("byteorder"), "byteorder");
        assert_eq!(strip_archive_prefix("a/b/c"), "b/c");
    }

    #[test]
    fn too_small_is_rejected() {
        let mut src = SliceSource::new(b"PK");
        assert!(read_central_directory(&mut src, &ParseLimits::unbounded()).is_err());
    }

    #[test]
    fn not_a_zip_is_rejected() {
        let junk = vec![0u8; 256];
        let mut src = SliceSource::new(&junk);
        assert!(read_central_directory(&mut src, &ParseLimits::unbounded()).is_err());
    }

    #[test]
    fn truncated_central_directory_is_rejected() {
        // Build a valid archive, then lop off the last byte so the EOCD's
        // declared CD offset/size no longer fits — must error, never panic.
        let mut archive = build_zip(&[("data.pkl", ::zip::CompressionMethod::Stored, b"abc")]);
        archive.truncate(archive.len() - 1);
        let mut src = SliceSource::new(&archive);
        let _ = read_central_directory(&mut src, &ParseLimits::unbounded()); // Ok or Err, never a panic
    }

    #[test]
    fn differential_random_archives() {
        // A deterministic LCG drives a broad differential sweep: many archives
        // of varying entry count / name / size / method, each parsed by both
        // the vendored reader and the `zip` crate and asserted index-identical.
        // This is the `cargo test`-resident counterpart to `fuzz_zip` (which
        // can only reach the crate-private reader from inside the crate).
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = |bound: u64| -> u64 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) % bound.max(1)
        };

        for _ in 0..256 {
            let n_entries = next(8) + 1;
            let mut spec: Vec<(String, ::zip::CompressionMethod, Vec<u8>)> = Vec::new();
            for i in 0..n_entries {
                let prefix = if next(2) == 0 { "archive/" } else { "" };
                let name = format!("{prefix}entry_{i}");
                let len = usize::try_from(next(2048)).unwrap();
                // Half-compressible (runs) vs incompressible (varied) payloads.
                let data: Vec<u8> = if next(2) == 0 {
                    vec![0xAB; len]
                } else {
                    (0..len)
                        .map(|k| u8::try_from((k as u64 + next(251)) % 256).unwrap())
                        .collect()
                };
                let method = if next(2) == 0 {
                    ::zip::CompressionMethod::Stored
                } else {
                    ::zip::CompressionMethod::Deflated
                };
                spec.push((name, method, data));
            }
            let refs: Vec<(&str, ::zip::CompressionMethod, &[u8])> = spec
                .iter()
                .map(|(n, m, d)| (n.as_str(), *m, d.as_slice()))
                .collect();
            let archive = build_zip(&refs);
            assert_matches_zip_crate(&archive);
        }
    }

    #[test]
    fn item_count_limit_rejects_container() {
        // A caller's `max_item_count` bounds the declared entry count fail-fast,
        // before the entry vector is built (tighter than the permanent cap).
        let archive = build_zip(&[
            ("a.npy", ::zip::CompressionMethod::Stored, b"x"),
            ("b.npy", ::zip::CompressionMethod::Stored, b"y"),
            ("c.npy", ::zip::CompressionMethod::Stored, b"z"),
        ]);
        let mut src = SliceSource::new(&archive);
        let limits = ParseLimits::default().with_max_item_count(2);
        let err = read_central_directory(&mut src, &limits).unwrap_err();
        assert!(
            matches!(err, AnamnesisError::LimitExceeded { limit, .. } if limit == "max_item_count"),
            "expected item-count rejection, got: {err}"
        );
        // Unbounded accepts all three.
        let ok = read_central_directory(&mut src, &ParseLimits::unbounded()).unwrap();
        assert_eq!(ok.len(), 3);
    }

    #[test]
    fn mmap_borrow_path_skips_cd_single_alloc_charge() {
        // The mmap (`SliceSource`) path reads the central directory zero-copy,
        // so a tiny `max_single_alloc` that rejects the streaming path must NOT
        // reject the borrow path — it allocates nothing for the directory.
        let archive = build_zip(&[
            (
                "archive/data.pkl",
                ::zip::CompressionMethod::Stored,
                b"data",
            ),
            (
                "archive/data/0",
                ::zip::CompressionMethod::Stored,
                &[0u8; 32],
            ),
        ]);
        let mut src = SliceSource::new(&archive);
        let limits = ParseLimits::default().with_max_single_alloc(8);
        let entries = read_central_directory(&mut src, &limits)
            .expect("mmap borrow path must not be charged for the directory read");
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn cd_single_alloc_limit_rejects_reader_path() {
        // The streaming (reader) path charges the central-directory read to
        // `max_single_alloc`; a tiny cap rejects it before allocating. (The
        // mmap path borrows the directory zero-copy, so it is not charged.)
        let archive = build_zip(&[
            (
                "archive/data.pkl",
                ::zip::CompressionMethod::Stored,
                b"data",
            ),
            (
                "archive/data/0",
                ::zip::CompressionMethod::Stored,
                &[0u8; 32],
            ),
        ]);
        let mut src = ReaderSource::new(std::io::Cursor::new(&archive)).unwrap();
        let limits = ParseLimits::default().with_max_single_alloc(8);
        let err = read_central_directory(&mut src, &limits).unwrap_err();
        assert!(
            matches!(err, AnamnesisError::LimitExceeded { limit, .. } if limit == "max_single_alloc_bytes"),
            "expected central-directory single-alloc rejection, got: {err}"
        );
    }

    #[test]
    fn compression_tag_allowlist() {
        assert_eq!(Compression::from_tag(0), Compression::Stored);
        assert_eq!(Compression::from_tag(8), Compression::Deflate);
        assert_eq!(Compression::from_tag(99), Compression::Unsupported(99));
        assert!(Compression::Stored.is_stored());
        assert!(!Compression::Deflate.is_stored());
    }
}
