// SPDX-License-Identifier: MIT OR Apache-2.0

//! `PyTorch` `.pth` `state_dict` parsing — minimal pickle VM with security boundary.
//!
//! Since `PyTorch` 1.6 (July 2020), `torch.save()` produces a ZIP archive
//! containing a pickle stream (`data.pkl`) that describes the `state_dict`
//! structure, plus raw tensor data files (`data/0`, `data/1`, ...).
//!
//! This module implements a minimal pickle interpreter (~36 opcodes) that
//! reconstructs the `state_dict` structure, then extracts tensor metadata
//! (name, shape, dtype) and raw data. An explicit `GLOBAL` allowlist rejects
//! any callable not related to `PyTorch` tensor reconstruction — equivalent
//! to `weights_only=True` but stricter.
//!
//! # Security
//!
//! Unlike Python's `pickle.load()`, this parser **never executes arbitrary
//! code**. Only allowlisted `GLOBAL` references (`torch._utils`,
//! `torch.*Storage`, `collections.OrderedDict`) are accepted. Unrecognized
//! globals produce `AnamnesisError::Parse`.
//!
//! The parser is also hardened against unguarded-allocation denial of service
//! (the [`candle #3533`](https://github.com/huggingface/candle/issues/3533)
//! class). The declared `data.pkl` size is capped at `MAX_PKL_SIZE` on both
//! the mmap and reader entry points before the pickle VM runs, and each
//! 4-byte-length pickle payload (`BINUNICODE`, `BINSTRING`, `BINBYTES`) is
//! capped at `MAX_PICKLE_PAYLOAD` before its clone. A malformed or malicious
//! archive fails fast with `AnamnesisError::Parse` instead of driving a
//! multi-GiB allocation.

use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt;
use std::io::{Read, Seek};
use std::path::Path;

use crate::backing::Backing;
use crate::error::AnamnesisError;
use crate::limits::Budget;
use crate::parse::safetensors::Dtype;
use crate::parse::utils::{PREALLOC_SOFT_CAP, byteswap_inplace};
// `ZipSource` is in scope for the reader-path `total_len` / `read_at` calls on
// the vendored `ReaderSource`; the trait methods are otherwise unreachable.
use crate::ParseLimits;
use crate::parse::zip::ZipSource;

/// Maximum declared size for a `.pth` archive's `data.pkl` entry that the
/// parsers will materialise or interpret.
///
/// Enforced on both entry points via [`enforce_pkl_size_cap`]: the mmap-based
/// [`parse_pth`] checks it before slicing `data.pkl` out of the mapped file
/// and running the pickle VM over it, and the reader-based
/// [`inspect_pth_from_reader`] checks it before reading the entry into memory.
///
/// Real `data.pkl` is typically <100 KiB even on torchvision-class 300 MB
/// models; 100 MiB is roughly three orders of magnitude above realistic.
/// The cap defends against an adversarial central directory that claims a
/// multi-GiB pickle: we reject before allocating or interpreting, so neither
/// `Vec::with_capacity` nor the pickle VM can be coaxed into multi-GiB work on
/// attacker-controlled input.
const MAX_PKL_SIZE: u64 = 100 * 1024 * 1024;

/// Permanent ceiling on the pickle VM's cumulative working-set heap (512 MiB).
///
/// `MAX_PKL_SIZE` bounds the opcode *stream*, not the heap each opcode
/// produces: a small crafted pickle can drive the value stack, memo clones,
/// and nested values to multi-GiB heap (`N`-floods, `BINGET` replay of a
/// memoised subtree). This cap charges every value the VM allocates — pushed
/// values, container backings, and the deep size of each memo clone — to a
/// running total and rejects before crossing it. A non-amplifying pickle's
/// VM heap is `O(data.pkl size)`, so at 5× the 100 MiB `MAX_PKL_SIZE` disk
/// cap this is roughly three orders of magnitude above any realistic
/// `state_dict` and invisible to honest files. Always-on (enforced even under
/// `ParseLimits::default()`); the caller's `max_total_bytes` tightens it.
const MAX_PICKLE_WORKING_SET: u64 = 512 * 1024 * 1024;

/// Permanent ceiling on the structural nesting depth of any pickle value the
/// VM constructs (256).
///
/// `BINPERSID` / `REDUCE` / `BUILD` each wrap their operand one `Box` deeper
/// for a single opcode byte, so an unbounded `Q`-chain builds an
/// arbitrarily deep value whose derived recursive `Drop` overflows the call
/// stack (a `panic = "abort"` process kill). Capping construction depth means
/// a value deeper than this is never built, which keeps `Drop` — and every
/// recursive walk (`Clone`, deep-size accounting, the post-execution
/// extraction traversal) — bounded to at most this many frames. Real torch
/// `state_dict` values nest ~6 deep; the post-execution
/// [`MAX_PICKLE_NESTING`] cap is 32; 256 is generous headroom and far below
/// any stack-overflow threshold. Always-on, independent of [`ParseLimits`].
const MAX_PICKLE_VM_DEPTH: u32 = 256;

/// Maximum declared size for a `.pth` archive's `byteorder` entry.
///
/// The entry contains literally `"little"` or `"big"` (≤6 bytes). 64 B is
/// generous head-room; anything beyond that signals a malformed or
/// adversarial archive.
const MAX_BYTEORDER_SIZE: u64 = 64;

/// Rejects a `data.pkl` ZIP entry whose declared size exceeds [`MAX_PKL_SIZE`].
///
/// Shared by the mmap path ([`parse_pth`]) and the reader path
/// ([`inspect_pth_from_reader`]) so the bound is identical by construction:
/// any future change to the cap or its wording applies to both entry points.
///
/// # Errors
///
/// Returns [`AnamnesisError::LimitExceeded`] if `declared` exceeds
/// [`MAX_PKL_SIZE`] or the caller's `ParseLimits` single-allocation budget.
fn enforce_pkl_size_cap(
    declared: u64,
    entry_name: &str,
    limits: &ParseLimits,
) -> crate::Result<()> {
    if declared > MAX_PKL_SIZE {
        return Err(AnamnesisError::LimitExceeded {
            limit: "MAX_PKL_SIZE",
            message: format!(
                "ZIP entry `{entry_name}`: declared size {declared} bytes exceeds \
                 the {MAX_PKL_SIZE}-byte cap"
            ),
        });
    }
    // Caller-supplied ceiling, layered on top of the permanent cap above.
    limits.check_alloc(declared, "PTH data.pkl entry")?;
    Ok(())
}

// ---------------------------------------------------------------------------
// PthDtype
// ---------------------------------------------------------------------------

/// Element data type derived from `PyTorch` storage class names.
///
/// Maps `torch.FloatStorage` to `F32`, `torch.HalfStorage` to `F16`, etc.
/// Covers all storage types emitted by `torch.save()` for `state_dict` files.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum PthDtype {
    /// 16-bit IEEE 754 half-precision (`torch.HalfStorage`).
    F16,
    /// 16-bit brain floating point (`torch.BFloat16Storage`).
    BF16,
    /// 32-bit IEEE 754 single-precision (`torch.FloatStorage`).
    F32,
    /// 64-bit IEEE 754 double-precision (`torch.DoubleStorage`).
    F64,
    /// Unsigned 8-bit integer (`torch.ByteStorage`).
    U8,
    /// Signed 8-bit integer (`torch.CharStorage`).
    I8,
    /// Signed 16-bit integer (`torch.ShortStorage`).
    I16,
    /// Signed 32-bit integer (`torch.IntStorage`).
    I32,
    /// Signed 64-bit integer (`torch.LongStorage`).
    I64,
    /// Boolean (`torch.BoolStorage`).
    Bool,
}

impl PthDtype {
    /// Returns the number of bytes per element for this dtype.
    #[must_use]
    pub const fn byte_size(self) -> usize {
        match self {
            Self::Bool | Self::U8 | Self::I8 => 1,
            Self::F16 | Self::BF16 | Self::I16 => 2,
            Self::F32 | Self::I32 => 4,
            Self::F64 | Self::I64 => 8,
        }
    }

    /// Converts to the anamnesis `Dtype` used for safetensors output.
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::Unsupported`] if no `safetensors` equivalent
    /// exists (currently all variants map successfully).
    pub fn to_dtype(self) -> crate::Result<Dtype> {
        match self {
            Self::F16 => Ok(Dtype::F16),
            Self::BF16 => Ok(Dtype::BF16),
            Self::F32 => Ok(Dtype::F32),
            Self::F64 => Ok(Dtype::F64),
            Self::U8 => Ok(Dtype::U8),
            Self::I8 => Ok(Dtype::I8),
            Self::I16 => Ok(Dtype::I16),
            Self::I32 => Ok(Dtype::I32),
            Self::I64 => Ok(Dtype::I64),
            Self::Bool => Ok(Dtype::Bool),
        }
    }

    /// Converts directly to `safetensors::Dtype`, skipping the intermediate
    /// anamnesis `Dtype`. Used by `pth_to_safetensors` for efficiency.
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::Unsupported`] if no `safetensors` equivalent
    /// exists (currently all variants map successfully).
    pub fn to_safetensors_dtype(self) -> crate::Result<safetensors::Dtype> {
        match self {
            Self::F16 => Ok(safetensors::Dtype::F16),
            Self::BF16 => Ok(safetensors::Dtype::BF16),
            Self::F32 => Ok(safetensors::Dtype::F32),
            Self::F64 => Ok(safetensors::Dtype::F64),
            Self::U8 => Ok(safetensors::Dtype::U8),
            Self::I8 => Ok(safetensors::Dtype::I8),
            Self::I16 => Ok(safetensors::Dtype::I16),
            Self::I32 => Ok(safetensors::Dtype::I32),
            Self::I64 => Ok(safetensors::Dtype::I64),
            Self::Bool => Ok(safetensors::Dtype::BOOL),
        }
    }

    /// Parses a `PyTorch` storage class name into a `PthDtype`.
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::Parse`] if the storage name is not recognized.
    fn from_storage_class(module: &str, name: &str) -> crate::Result<Self> {
        if module != "torch" {
            return Err(AnamnesisError::Parse {
                reason: format!("unknown storage module `{module}.{name}`"),
            });
        }
        match name {
            "FloatStorage" => Ok(Self::F32),
            "DoubleStorage" => Ok(Self::F64),
            "HalfStorage" => Ok(Self::F16),
            "BFloat16Storage" => Ok(Self::BF16),
            "LongStorage" => Ok(Self::I64),
            "IntStorage" => Ok(Self::I32),
            "ShortStorage" => Ok(Self::I16),
            "CharStorage" => Ok(Self::I8),
            "ByteStorage" => Ok(Self::U8),
            "BoolStorage" => Ok(Self::Bool),
            _ => Err(AnamnesisError::Parse {
                reason: format!("unknown storage class `torch.{name}`"),
            }),
        }
    }
}

impl fmt::Display for PthDtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::F16 => "F16",
            Self::BF16 => "BF16",
            Self::F32 => "F32",
            Self::F64 => "F64",
            Self::U8 => "U8",
            Self::I8 => "I8",
            Self::I16 => "I16",
            Self::I32 => "I32",
            Self::I64 => "I64",
            Self::Bool => "BOOL",
        };
        f.write_str(s)
    }
}

// ---------------------------------------------------------------------------
// PthTensor
// ---------------------------------------------------------------------------

/// A single tensor view from a parsed `.pth` file.
///
/// Borrows raw data from the memory-mapped file (zero-copy for contiguous
/// little-endian tensors) or owns a copy (non-contiguous / big-endian).
///
/// **FFI / Python boundary:** `name` and `shape` are owned, but `data` borrows
/// the owning [`ParsedPth`] for `'a` in the zero-copy case. A binding that hands
/// the bytes to another runtime (e.g. a `NumPy` array) must call
/// `data.into_owned()` first, so the array never aliases bytes the owning
/// [`ParsedPth`] can drop. See `docs/python-interop.md` (ownership contract).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PthTensor<'a> {
    /// Tensor name (`state_dict` key, e.g. `"linear.weight"`).
    pub name: String,
    /// Tensor shape (e.g. `[16, 10]` for a 16-by-10 matrix).
    pub shape: Vec<usize>,
    /// Element data type.
    pub dtype: PthDtype,
    /// Raw bytes in row-major, native-endian order.
    ///
    /// `Cow::Borrowed` when the tensor data is contiguous and little-endian
    /// (zero-copy slice from the mmap). `Cow::Owned` when a layout
    /// transformation was required (non-contiguous strides or
    /// big-endian byte-swap).
    pub data: Cow<'a, [u8]>,
}

impl<'a> PthTensor<'a> {
    /// Builds a tensor view from its parts.
    ///
    /// The construction path for callers that *synthesise* `.pth` tensors to
    /// feed `pth_to_safetensors` (or the `_bytes` form), rather than obtaining
    /// them from [`ParsedPth::tensors`]. It exists because the struct is
    /// `#[non_exhaustive]` since v0.7.6: a literal cannot be written from
    /// outside the crate, and this constructor absorbs any field added later
    /// without breaking the callers that build one.
    ///
    /// The arguments are moved, not validated: `data` is trusted to be
    /// `product(shape) × dtype.byte_size()` bytes in row-major, native-endian
    /// order, exactly as [`ParsedPth::tensors`] produces it. Pass
    /// `Cow::Owned` to hand over a buffer, `Cow::Borrowed` to lend one for
    /// `'a`.
    #[must_use]
    pub const fn new(
        name: String,
        shape: Vec<usize>,
        dtype: PthDtype,
        data: Cow<'a, [u8]>,
    ) -> Self {
        Self {
            name,
            shape,
            dtype,
            data,
        }
    }
}

/// Tensor metadata extracted from the pickle stream (no data).
#[derive(Debug)]
struct TensorMeta {
    name: String,
    shape: Vec<usize>,
    dtype: PthDtype,
    /// Data file index in the ZIP archive (e.g., `"0"` → `data/0`).
    storage_key: String,
    /// Byte offset into the storage file.
    storage_offset: usize,
    strides: Vec<usize>,
}

/// A parsed `.pth` file — owns the memory-mapped data and provides
/// zero-copy tensor access.
///
/// Created by [`parse_pth`] (memory-mapped) or by [`parse_pth_bytes`] /
/// [`parse_pth_from_reader`] (owned copy). Call
/// [`tensors()`](ParsedPth::tensors) to get `PthTensor` views that borrow
/// directly from the backing bytes.
#[derive(Debug)]
pub struct ParsedPth {
    /// File bytes — a memory map (path-based [`parse_pth`]) or an owned
    /// `Vec<u8>` (copy-based [`parse_pth_bytes`] / [`parse_pth_from_reader`]).
    buffer: Backing,
    /// Per-tensor metadata (name, shape, dtype, storage location).
    meta: Vec<TensorMeta>,
    /// ZIP entry index: suffix → `(data_start, data_len)` in the mmap.
    entry_index: EntryIndex,
    /// Whether the file uses big-endian storage.
    big_endian: bool,
}

/// Compact, sorted `suffix → (data_start, data_len)` index over a `.pth`
/// archive's STORED entries.
///
/// A sorted `Vec` queried by binary search, rather than a `HashMap`: for a
/// many-tiny-entry archive the hash table's power-of-two bucket array (≈31%
/// slack at 50 000 entries) and `String`'s 8-byte capacity word per key
/// dominate the resident metadata. The `Vec` is exact-sized and keys are
/// `Box<str>` (no capacity word), roughly halving resident bytes/entry. The
/// handful of lookups per parse (`data.pkl`, `byteorder`, one per tensor) are
/// `O(log n)` — negligible.
#[derive(Debug)]
struct EntryIndex {
    /// `(suffix, data_start, data_len)` triples, sorted ascending by `suffix`.
    entries: Vec<(Box<str>, usize, usize)>,
}

impl EntryIndex {
    /// Looks up an entry suffix, returning its `(data_start, data_len)`.
    #[must_use]
    fn get(&self, name: &str) -> Option<(usize, usize)> {
        match self
            .entries
            .binary_search_by(|(key, _, _)| key.as_ref().cmp(name))
        {
            // INDEX: `binary_search_by` returns an in-bounds index on `Ok`.
            #[allow(clippy::indexing_slicing)]
            Ok(idx) => {
                let (_, start, len) = self.entries[idx];
                Some((start, len))
            }
            Err(_) => None,
        }
    }
}

impl ParsedPth {
    /// Returns tensor views borrowing directly from the mmap.
    ///
    /// For contiguous little-endian tensors (>99% of real files), the
    /// data is a zero-copy `&[u8]` slice from the mmap — no heap
    /// allocation. Non-contiguous or big-endian tensors get an owned copy.
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::Parse`] if a storage entry is missing
    /// or a tensor's byte range exceeds the storage.
    ///
    /// # Memory
    ///
    /// For contiguous little-endian tensors, data is zero-copy (`Cow::Borrowed`
    /// from the mmap) — no per-tensor allocation. Non-contiguous or big-endian
    /// tensors allocate an owned `Vec<u8>` of `n_elements × dtype.byte_size()`
    /// bytes. Peak memory: the mmap (file-sized) plus one owned copy per
    /// non-contiguous tensor. The `Vec<PthTensor>` itself is lightweight
    /// (metadata + `Cow` pointers).
    pub fn tensors(&self) -> crate::Result<Vec<PthTensor<'_>>> {
        let mut tensors = Vec::with_capacity(self.meta.len());
        for m in &self.meta {
            let storage_suffix = format!("data/{}", m.storage_key);
            let (storage_start, storage_len) = self
                .entry_index
                .get(storage_suffix.as_str())
                .ok_or_else(|| AnamnesisError::Parse {
                    reason: format!("ZIP entry `{storage_suffix}` not found"),
                })?;
            let storage = self
                .buffer
                .get(storage_start..storage_start + storage_len)
                .ok_or_else(|| AnamnesisError::Parse {
                    reason: format!("storage `{}`: backing slice out of bounds", m.storage_key),
                })?;

            let elem_size = m.dtype.byte_size();
            let data: Cow<'_, [u8]> = if is_contiguous(&m.shape, &m.strides) && !self.big_endian {
                // Zero-copy: borrow directly from the mmap.
                let n_elements: usize = m
                    .shape
                    .iter()
                    .try_fold(1usize, |acc, &d| acc.checked_mul(d))
                    .ok_or_else(|| AnamnesisError::Parse {
                        reason: format!("tensor `{}`: element count overflow", m.name),
                    })?;
                let n_bytes =
                    n_elements
                        .checked_mul(elem_size)
                        .ok_or_else(|| AnamnesisError::Parse {
                            reason: format!("tensor `{}`: byte count overflow", m.name),
                        })?;
                let end =
                    m.storage_offset
                        .checked_add(n_bytes)
                        .ok_or_else(|| AnamnesisError::Parse {
                            reason: format!("tensor `{}`: storage end offset overflow", m.name),
                        })?;
                Cow::Borrowed(storage.get(m.storage_offset..end).ok_or_else(|| {
                    AnamnesisError::Parse {
                        reason: format!(
                            "tensor `{}`: storage read out of bounds \
                             ([{}..{}], storage len = {})",
                            m.name,
                            m.storage_offset,
                            end,
                            storage.len()
                        ),
                    }
                })?)
            } else if is_contiguous(&m.shape, &m.strides) {
                // Contiguous but big-endian: copy + byte-swap.
                let n_elements: usize = m
                    .shape
                    .iter()
                    .try_fold(1usize, |acc, &d| acc.checked_mul(d))
                    .ok_or_else(|| AnamnesisError::Parse {
                        reason: format!("tensor `{}`: element count overflow", m.name),
                    })?;
                let n_bytes =
                    n_elements
                        .checked_mul(elem_size)
                        .ok_or_else(|| AnamnesisError::Parse {
                            reason: format!("tensor `{}`: byte count overflow", m.name),
                        })?;
                let end =
                    m.storage_offset
                        .checked_add(n_bytes)
                        .ok_or_else(|| AnamnesisError::Parse {
                            reason: format!("tensor `{}`: storage end offset overflow", m.name),
                        })?;
                let mut buf = storage
                    .get(m.storage_offset..end)
                    .ok_or_else(|| AnamnesisError::Parse {
                        reason: format!("tensor `{}`: storage read out of bounds", m.name),
                    })?
                    .to_vec();
                byteswap_inplace(&mut buf, elem_size);
                Cow::Owned(buf)
            } else {
                // Non-contiguous: copy to contiguous layout.
                let mut buf =
                    copy_to_contiguous(storage, m.storage_offset, &m.shape, &m.strides, elem_size)?;
                if self.big_endian && elem_size > 1 {
                    byteswap_inplace(&mut buf, elem_size);
                }
                Cow::Owned(buf)
            };

            tensors.push(PthTensor {
                name: m.name.clone(),
                shape: m.shape.clone(),
                dtype: m.dtype,
                data,
            });
        }
        Ok(tensors)
    }

    /// Returns the number of tensors.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.meta.len()
    }

    /// Returns `true` if the file contained no tensors.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.meta.is_empty()
    }

    /// Returns inspection info derived from the parsed metadata.
    ///
    /// No I/O — purely computed from the tensor metadata extracted during
    /// [`parse_pth`].
    pub fn inspect(&self) -> PthInspectInfo {
        build_pth_inspect_info(&self.meta, self.big_endian)
    }

    /// Converts the parsed `.pth` tensors to a safetensors file.
    ///
    /// Equivalent to calling [`tensors()`](Self::tensors) followed by
    /// `pth_to_safetensors` — but as a single convenience method.
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::Io`] if the output file cannot be written.
    /// Returns [`AnamnesisError::Parse`] if tensor extraction or
    /// serialization fails.
    pub fn to_safetensors(&self, output: impl AsRef<std::path::Path>) -> crate::Result<()> {
        let tensors = self.tensors()?;
        crate::remember::pth::pth_to_safetensors(&tensors, output)
    }

    /// Converts the parsed `.pth` tensors to an in-memory safetensors byte
    /// buffer.
    ///
    /// Equivalent to calling [`tensors()`](Self::tensors) followed by
    /// [`pth_to_safetensors_bytes`](crate::remember::pth::pth_to_safetensors_bytes)
    /// — but as a single convenience method. The returned bytes can be
    /// passed directly to `VarBuilder::from_buffered_safetensors`.
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::Parse`] if tensor extraction or
    /// serialization fails.
    ///
    /// # Memory
    ///
    /// Materialises all tensors (zero-copy for contiguous LE data), then
    /// serialises into a single `Vec<u8>`. Peak heap ≈ mmap + output buffer.
    pub fn to_safetensors_bytes(&self) -> crate::Result<Vec<u8>> {
        let tensors = self.tensors()?;
        crate::remember::pth::pth_to_safetensors_bytes(&tensors)
    }

    /// Returns lightweight per-tensor metadata (name, shape, dtype, byte
    /// length) without materializing tensor data.
    ///
    /// Use this for display-only paths (e.g., `amn parse`) where the raw
    /// bytes are not needed. Avoids the per-tensor entry-index lookup and
    /// bounds checking that [`tensors()`](Self::tensors) performs.
    #[must_use]
    pub fn tensor_info(&self) -> Vec<PthTensorInfo> {
        build_pth_tensor_info(&self.meta)
    }
}

/// Maps parsed pickle metadata to the lightweight [`PthTensorInfo`] shape.
///
/// Shared by [`ParsedPth::tensor_info`] (mmap-backed path) and
/// [`parse_pth_front_matter_from_reader_with_limits`] (reader-generic path)
/// so there is exactly one place that turns a [`TensorMeta`] into a
/// [`PthTensorInfo`] — any future change to the byte-length computation
/// applies to both entry points identically.
///
/// Uses the same never-short-circuiting `u64` `saturating_mul` fold as
/// [`build_pth_inspect_info`], narrowed to `usize` at the end, rather than a
/// `checked_mul` `try_fold` that stops at the first overflowing dimension:
/// a shape with a zero dimension *after* an earlier overflow (e.g.
/// `[2^33, 2^33, 0]`) has zero elements regardless of how large the earlier
/// dimensions are, and a short-circuiting fold would never see the trailing
/// zero, reporting `usize::MAX` instead of `0`. Consistency with
/// `build_pth_inspect_info` matters beyond this function's own correctness:
/// [`build_front_matter_inspect_info`] sums this function's `byte_len`
/// output, so a divergence here would make
/// `PthFrontMatter::inspect().total_bytes` disagree with
/// `inspect_pth_from_reader(...).total_bytes` on the same adversarial file.
fn build_pth_tensor_info(meta: &[TensorMeta]) -> Vec<PthTensorInfo> {
    meta.iter()
        .map(|m| {
            // CAST: usize → u64, shape dims and dtype byte size fit in u64
            #[allow(clippy::as_conversions)]
            let n_elements: u64 = m
                .shape
                .iter()
                .copied()
                .fold(1u64, |acc, d| acc.saturating_mul(d as u64));
            // CAST: usize → u64, byte size of a single element is ≤ 8 → fits
            #[allow(clippy::as_conversions)]
            let byte_size = m.dtype.byte_size() as u64;
            let byte_len_u64 = n_elements.saturating_mul(byte_size);
            // Saturating narrow: `usize::MAX` on a 32-bit target where
            // `byte_len_u64` exceeds `usize::MAX`; exact on 64-bit targets.
            let byte_len = usize::try_from(byte_len_u64).unwrap_or(usize::MAX);
            PthTensorInfo {
                name: m.name.clone(),
                shape: m.shape.clone(),
                dtype: m.dtype,
                byte_len,
            }
        })
        .collect()
}

/// Lightweight per-tensor metadata from a parsed `.pth` file.
///
/// Produced by [`ParsedPth::tensor_info`]. Contains only metadata —
/// no data access, no mmap slicing.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PthTensorInfo {
    /// Tensor name (`state_dict` key).
    pub name: String,
    /// Tensor shape.
    pub shape: Vec<usize>,
    /// Element data type.
    pub dtype: PthDtype,
    /// Total byte length (`product(shape) * dtype.byte_size()`).
    pub byte_len: usize,
}

/// Summary information about a parsed `.pth` file.
///
/// Produced by [`ParsedPth::inspect`]. No I/O — derived from metadata.
#[derive(Debug, Clone)]
#[non_exhaustive]
#[must_use]
pub struct PthInspectInfo {
    /// Number of tensors in the `state_dict`.
    pub tensor_count: usize,
    /// Total size of raw tensor data in bytes.
    pub total_bytes: u64,
    /// Distinct dtypes found (in order of first occurrence).
    pub dtypes: Vec<PthDtype>,
    /// Whether the file uses big-endian storage.
    pub big_endian: bool,
}

impl fmt::Display for PthInspectInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Format:      PyTorch state_dict (.pth)")?;
        write!(f, "\nTensors:     {}", self.tensor_count)?;
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
        let endian = if self.big_endian {
            "big-endian"
        } else {
            "little-endian"
        };
        write!(f, "\nByte order:  {endian}")?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Pickle VM (internal)
// ---------------------------------------------------------------------------

/// Value on the pickle VM stack.
///
/// Not all Python types are represented — only the subset produced by
/// `torch.save()` for `state_dict` files.
// Fields like `Bool(bool)` and `Bytes(Vec<u8>)` are populated by the pickle
// VM when interpreting `state_dict` streams but are not destructured during
// tensor extraction. They must remain for correct pickle interpretation.
#[derive(Debug, Clone)]
#[allow(dead_code)]
// EXHAUSTIVE: private enum — crate owns and matches all variants; wildcard
// arms are used in extraction code where most variants are irrelevant.
#[allow(clippy::wildcard_enum_match_arm)]
enum PickleValue {
    None,
    Bool(bool),
    Int(i64),
    String(String),
    Bytes(Vec<u8>),
    Tuple(Vec<PickleValue>),
    List(Vec<PickleValue>),
    /// Key-value pairs preserving insertion order (matches Python `OrderedDict`).
    Dict(Vec<(PickleValue, PickleValue)>),
    /// A `GLOBAL` reference: `module.name`.
    Global {
        module: String,
        name: String,
    },
    /// A persistent ID referencing tensor data in the ZIP archive.
    PersistentId(Box<PickleValue>),
    /// Result of `REDUCE(callable, args)`.
    Reduced {
        callable: Box<PickleValue>,
        args: Box<PickleValue>,
    },
    /// Result of `BUILD(obj, state)` — sets `obj.__dict__` from `state`.
    Built {
        obj: Box<PickleValue>,
        state: Box<PickleValue>,
    },
}

/// Checks whether a `GLOBAL` reference is in the security allowlist.
///
/// Only `PyTorch` tensor-reconstruction callables, storage classes, and
/// `collections.OrderedDict` are permitted.
fn is_allowed_global(module: &str, name: &str) -> bool {
    matches!(
        (module, name),
        (
            "torch._utils",
            "_rebuild_tensor_v2" | "_rebuild_parameter" | "_rebuild_parameter_with_state"
        ) | (
            "torch",
            "FloatStorage"
                | "DoubleStorage"
                | "HalfStorage"
                | "BFloat16Storage"
                | "LongStorage"
                | "IntStorage"
                | "ShortStorage"
                | "CharStorage"
                | "ByteStorage"
                | "BoolStorage"
        ) | ("collections", "OrderedDict")
            | ("torch.nn.parameter", "Parameter")
    )
}

/// Returns `true` if this is a `REDUCE(OrderedDict, ())` call that should
/// produce an empty `Dict` instead of a `Reduced` node.
fn is_ordered_dict_constructor(callable: &PickleValue, args: &PickleValue) -> bool {
    if let PickleValue::Global { module, name } = callable
        && module == "collections"
        && name == "OrderedDict"
        && let PickleValue::Tuple(items) = args
    {
        return items.is_empty();
    }
    false
}

/// Per-opcode soft cap on a single pickle string/bytes payload (64 MiB).
///
/// The 4-byte-length opcodes (`BINUNICODE`, `BINSTRING`, `BINBYTES`) declare a
/// `u32` length, and each materialises an owned `String`/`Vec<u8>` clone of
/// that length. [`MAX_PKL_SIZE`] already bounds the whole stream, but a single
/// 99 MiB string inside an otherwise-legitimate 100 MiB pickle would still
/// allocate in one shot. Capping each payload at 64 MiB — far above any
/// realistic `state_dict` key, storage-class name, or `byteorder` string —
/// hardens against that without affecting real files. Defence in depth.
const MAX_PICKLE_PAYLOAD: usize = 64 * 1024 * 1024;

/// Rejects a single pickle string/bytes payload whose declared length exceeds
/// the permanent [`MAX_PICKLE_PAYLOAD`] per-item cap, **before** the payload is
/// read or allocated.
///
/// This is the pre-allocation guard only. The cumulative charge (the caller's
/// `max_total_bytes` aggregate and the permanent [`MAX_PICKLE_WORKING_SET`]
/// floor) happens once, when the resulting value is pushed through
/// [`PickleVm::push_value`], so there is a single working-set accounting point.
///
/// # Errors
///
/// Returns [`AnamnesisError::LimitExceeded`] if `len` exceeds [`MAX_PICKLE_PAYLOAD`].
fn enforce_pickle_payload_cap(len: usize, opcode: &str) -> crate::Result<()> {
    if len > MAX_PICKLE_PAYLOAD {
        return Err(AnamnesisError::LimitExceeded {
            limit: "MAX_PICKLE_PAYLOAD",
            message: format!(
                "pickle {opcode} length {len} bytes exceeds the \
                 {MAX_PICKLE_PAYLOAD}-byte cap"
            ),
        });
    }
    Ok(())
}

/// Per-stack-slot bookkeeping, maintained in lockstep with [`PickleVm::stack`]
/// so the VM can bound nesting depth in `O(1)` per opcode (a per-push deep
/// walk would be `O(n²)` on a `BINPERSID` chain — a CPU-DoS in the guard
/// itself).
#[derive(Clone, Copy)]
struct SlotMeta {
    /// Structural nesting depth of the value in this slot (leaf = 1).
    depth: u32,
}

/// Minimal pickle VM state.
struct PickleVm<'a> {
    /// Raw pickle bytes.
    data: &'a [u8],
    /// Current read position.
    pos: usize,
    /// Value stack.
    stack: Vec<PickleValue>,
    /// Per-slot metadata, kept length-synced with [`PickleVm::stack`]. The
    /// invariant `stack.len() == meta_stack.len()` is preserved by routing
    /// every growth through [`PickleVm::push_value`] and every shrink through
    /// [`PickleVm::pop`] / [`PickleVm::pop_mark`].
    meta_stack: Vec<SlotMeta>,
    /// Mark stack (positions in the value stack).
    mark_stack: Vec<usize>,
    /// Memo table (protocol 2+ `BINPUT`/`BINGET`, protocol 4+ `MEMOIZE`).
    /// Stores each value with its [`SlotMeta`] so `BINGET` restores depth in
    /// `O(1)` without re-walking the cloned subtree.
    memo: HashMap<u32, (PickleValue, SlotMeta)>,
    /// Auto-incrementing memo key for `MEMOIZE` opcode.
    next_memo_id: u32,
    /// Cumulative heap charged to the VM working set this parse (bytes),
    /// bounded by [`MAX_PICKLE_WORKING_SET`] and the caller's
    /// [`ParseLimits::max_total_bytes`]. Monotonic — a conservative
    /// over-approximation of live heap (peak ≤ cumulative).
    working_set: u64,
    /// Caller-supplied allocation budget — the per-item single-allocation cap
    /// and the cumulative parse-time-heap aggregate, charged on every value
    /// the VM allocates (pushed values and memo clones).
    budget: Budget,
}

impl<'a> PickleVm<'a> {
    fn new(data: &'a [u8], limits: &ParseLimits) -> Self {
        Self {
            data,
            pos: 0,
            stack: Vec::new(),
            meta_stack: Vec::new(),
            mark_stack: Vec::new(),
            memo: HashMap::new(),
            next_memo_id: 0,
            working_set: 0,
            budget: Budget::new(limits),
        }
    }

    /// Shallow heap footprint of a single value node (bytes): the enum slot
    /// plus the immediately-owned `String` / `Bytes` payload, but **not** its
    /// children (which are charged when they are themselves pushed).
    fn shallow_size(v: &PickleValue) -> u64 {
        // CAST: usize → u64, lossless widening on all supported targets
        #[allow(clippy::as_conversions)]
        let base = std::mem::size_of::<PickleValue>() as u64;
        // EXHAUSTIVE: only String/Bytes carry an immediate owned payload; every
        // other variant's heap is its children, charged when they are pushed.
        #[allow(clippy::wildcard_enum_match_arm)]
        let payload = match v {
            PickleValue::String(s) => s.len(),
            PickleValue::Bytes(b) => b.len(),
            _ => 0,
        };
        // CAST: usize → u64, lossless widening
        #[allow(clippy::as_conversions)]
        let payload = payload as u64;
        base.saturating_add(payload)
    }

    /// Deep heap footprint of a value and all its children (bytes). Recursive,
    /// but bounded to [`MAX_PICKLE_VM_DEPTH`] frames because no value deeper
    /// than that is ever constructed. Used only on memo-clone opcodes, where
    /// the whole subtree is freshly allocated.
    fn deep_size(v: &PickleValue) -> u64 {
        let mut total = Self::shallow_size(v);
        // EXHAUSTIVE: only the nested variants add children; the `_` arm covers
        // every leaf variant (no children). The recursion is depth-bounded by
        // MAX_PICKLE_VM_DEPTH, so it cannot overflow the call stack.
        #[allow(clippy::wildcard_enum_match_arm)]
        match v {
            PickleValue::Tuple(items) | PickleValue::List(items) => {
                for item in items {
                    total = total.saturating_add(Self::deep_size(item));
                }
            }
            PickleValue::Dict(pairs) => {
                for (k, val) in pairs {
                    total = total
                        .saturating_add(Self::deep_size(k))
                        .saturating_add(Self::deep_size(val));
                }
            }
            PickleValue::PersistentId(inner) => {
                total = total.saturating_add(Self::deep_size(inner));
            }
            PickleValue::Reduced { callable, args } => {
                total = total
                    .saturating_add(Self::deep_size(callable))
                    .saturating_add(Self::deep_size(args));
            }
            PickleValue::Built { obj, state } => {
                total = total
                    .saturating_add(Self::deep_size(obj))
                    .saturating_add(Self::deep_size(state));
            }
            _ => {}
        }
        total
    }

    /// Charges `bytes` to the VM working set, enforcing both the permanent
    /// [`MAX_PICKLE_WORKING_SET`] floor and the caller's
    /// [`ParseLimits::max_total_bytes`] aggregate.
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::LimitExceeded`] if the running total would
    /// exceed the permanent working-set floor, or if [`Budget::charge_alloc`]
    /// rejects the caller's aggregate; [`AnamnesisError::Parse`] if the running
    /// total overflows `u64`.
    fn charge(&mut self, bytes: u64, context: &str) -> crate::Result<()> {
        self.working_set =
            self.working_set
                .checked_add(bytes)
                .ok_or_else(|| AnamnesisError::Parse {
                    reason: "pickle working-set byte counter overflow".into(),
                })?;
        if self.working_set > MAX_PICKLE_WORKING_SET {
            return Err(AnamnesisError::LimitExceeded {
                limit: "MAX_PICKLE_WORKING_SET",
                message: format!(
                    "pickle VM working set exceeds the {MAX_PICKLE_WORKING_SET}-byte cap \
                     ({context})"
                ),
            });
        }
        self.budget.charge_alloc(bytes, context)
    }

    /// Pushes a value and its depth onto the stack, the **only** way the value
    /// stack grows. Charges the value's shallow heap cost and rejects a depth
    /// past [`MAX_PICKLE_VM_DEPTH`], so vectors (a) flat-stack and (c) nesting
    /// are bounded at the single choke point.
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::LimitExceeded`] if `depth` exceeds
    /// [`MAX_PICKLE_VM_DEPTH`] or the working-set charge is rejected.
    fn push_value(&mut self, v: PickleValue, depth: u32) -> crate::Result<()> {
        if depth > MAX_PICKLE_VM_DEPTH {
            return Err(AnamnesisError::LimitExceeded {
                limit: "MAX_PICKLE_VM_DEPTH",
                message: format!(
                    "pickle value nesting depth {depth} exceeds the \
                     {MAX_PICKLE_VM_DEPTH} cap"
                ),
            });
        }
        self.charge(Self::shallow_size(&v), "PTH pickle value")?;
        self.stack.push(v);
        self.meta_stack.push(SlotMeta { depth });
        Ok(())
    }

    /// Pushes a leaf value (depth 1). Convenience over [`PickleVm::push_value`]
    /// for the constant- and payload-sized opcodes.
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::Parse`] if the working-set charge is rejected.
    fn push_leaf(&mut self, v: PickleValue) -> crate::Result<()> {
        self.push_value(v, 1)
    }

    /// Raises the top slot's depth to at least `child_depth` after an in-place
    /// container mutation (`APPEND(S)` / `SETITEM(S)` grow the top `List` /
    /// `Dict` without a fresh push), re-enforcing [`MAX_PICKLE_VM_DEPTH`].
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::LimitExceeded`] if the updated depth exceeds
    /// [`MAX_PICKLE_VM_DEPTH`].
    fn bump_top_depth(&mut self, child_depth: u32) -> crate::Result<()> {
        if let Some(meta) = self.meta_stack.last_mut() {
            meta.depth = meta.depth.max(child_depth);
            if meta.depth > MAX_PICKLE_VM_DEPTH {
                return Err(AnamnesisError::LimitExceeded {
                    limit: "MAX_PICKLE_VM_DEPTH",
                    message: format!(
                        "pickle value nesting depth {} exceeds the \
                         {MAX_PICKLE_VM_DEPTH} cap",
                        meta.depth
                    ),
                });
            }
        }
        Ok(())
    }

    // -- byte reading helpers ------------------------------------------------

    fn read_u8(&mut self) -> crate::Result<u8> {
        let b = self
            .data
            .get(self.pos)
            .copied()
            .ok_or_else(|| AnamnesisError::Parse {
                reason: "unexpected end of pickle stream".into(),
            })?;
        self.pos += 1;
        Ok(b)
    }

    fn read_u16_le(&mut self) -> crate::Result<u16> {
        let bytes: [u8; 2] = self.read_fixed()?;
        Ok(u16::from_le_bytes(bytes))
    }

    fn read_i32_le(&mut self) -> crate::Result<i32> {
        let bytes: [u8; 4] = self.read_fixed()?;
        Ok(i32::from_le_bytes(bytes))
    }

    fn read_u32_le(&mut self) -> crate::Result<u32> {
        let bytes: [u8; 4] = self.read_fixed()?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn read_u64_le(&mut self) -> crate::Result<u64> {
        let bytes: [u8; 8] = self.read_fixed()?;
        Ok(u64::from_le_bytes(bytes))
    }

    /// Reads exactly `N` bytes from the pickle stream into a fixed-size array.
    fn read_fixed<const N: usize>(&mut self) -> crate::Result<[u8; N]> {
        let hi = self
            .pos
            .checked_add(N)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: "pickle offset overflow".into(),
            })?;
        let slice = self
            .data
            .get(self.pos..hi)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: "unexpected end of pickle stream".into(),
            })?;
        self.pos = hi;
        // try_into: slice length == N guaranteed by .get(pos..pos+N)
        let arr: [u8; N] = slice.try_into().map_err(|_| AnamnesisError::Parse {
            reason: "internal: slice-to-array conversion failed".into(),
        })?;
        Ok(arr)
    }

    fn read_bytes(&mut self, n: usize) -> crate::Result<&'a [u8]> {
        let hi = self
            .pos
            .checked_add(n)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: "pickle offset overflow".into(),
            })?;
        let slice = self
            .data
            .get(self.pos..hi)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: "unexpected end of pickle stream".into(),
            })?;
        self.pos = hi;
        Ok(slice)
    }

    /// Reads a newline-terminated string (for text-mode `GLOBAL` opcode).
    fn read_line(&mut self) -> crate::Result<&'a str> {
        let start = self.pos;
        loop {
            let b = self.read_u8()?;
            if b == b'\n' {
                let line =
                    self.data
                        .get(start..self.pos - 1)
                        .ok_or_else(|| AnamnesisError::Parse {
                            reason: "pickle line read out of bounds".into(),
                        })?;
                return std::str::from_utf8(line).map_err(|e| AnamnesisError::Parse {
                    reason: format!("non-UTF-8 pickle string: {e}"),
                });
            }
        }
    }

    // -- stack helpers -------------------------------------------------------

    /// Pops the top value and its [`SlotMeta`], keeping the value and metadata
    /// stacks length-synced.
    fn pop(&mut self) -> crate::Result<(PickleValue, SlotMeta)> {
        let v = self.stack.pop().ok_or_else(|| AnamnesisError::Parse {
            reason: "pickle stack underflow".into(),
        })?;
        // meta_stack.len() == stack.len() (every push goes through push_value),
        // so a successful stack.pop() guarantees a matching meta; the ok_or_else
        // is a defensive desync guard, never expected to fire.
        let meta = self.meta_stack.pop().ok_or_else(|| AnamnesisError::Parse {
            reason: "pickle metadata stack desync".into(),
        })?;
        Ok((v, meta))
    }

    /// Pops everything above the latest mark as parallel value + metadata
    /// vectors (same length).
    fn pop_mark(&mut self) -> crate::Result<(Vec<PickleValue>, Vec<SlotMeta>)> {
        let mark_pos = self.mark_stack.pop().ok_or_else(|| AnamnesisError::Parse {
            reason: "pickle mark stack underflow".into(),
        })?;
        if mark_pos > self.stack.len() {
            return Err(AnamnesisError::Parse {
                reason: "pickle mark position out of range".into(),
            });
        }
        let items = self.stack.split_off(mark_pos);
        let metas = self.meta_stack.split_off(mark_pos);
        Ok((items, metas))
    }

    /// Maximum `depth` over a slice of [`SlotMeta`] (0 for an empty slice).
    fn max_depth(metas: &[SlotMeta]) -> u32 {
        metas.iter().map(|m| m.depth).max().unwrap_or(0)
    }

    /// Clones the top value (and its [`SlotMeta`]) for memoisation, charging
    /// the clone's deep heap size **before** the clone so an over-budget memo
    /// is rejected without allocating the duplicate. Bounds vector (b) on the
    /// `BINPUT` / `MEMOIZE` side.
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::Parse`] on empty stack or if the charge is
    /// rejected.
    fn memoize_top(&mut self, opcode: &str) -> crate::Result<(PickleValue, SlotMeta)> {
        let bytes = match self.stack.last() {
            Some(v) => Self::deep_size(v),
            None => {
                return Err(AnamnesisError::Parse {
                    reason: format!("{opcode}: empty stack"),
                });
            }
        };
        self.charge(bytes, "PTH pickle memo clone")?;
        // The immutable borrow above has ended; charge() does not touch the
        // stack, so the top is still present. `.last().cloned()` avoids both
        // indexing and unwrap.
        let val = self
            .stack
            .last()
            .cloned()
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!("{opcode}: empty stack"),
            })?;
        let meta = self
            .meta_stack
            .last()
            .copied()
            .ok_or_else(|| AnamnesisError::Parse {
                reason: "pickle metadata stack desync".into(),
            })?;
        Ok((val, meta))
    }

    /// Clones a memoised value for `BINGET` replay, charging its children's
    /// deep heap **before** the clone (the root's shallow size is charged when
    /// [`PickleVm::push_value`] re-pushes it, so the total is the exact deep
    /// size). Bounds vector (b) on the `BINGET` side.
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::Parse`] if the key is absent or the charge is
    /// rejected.
    fn memo_get_clone(&mut self, key: u32, opcode: &str) -> crate::Result<(PickleValue, u32)> {
        let (children, depth) = match self.memo.get(&key) {
            Some((v, m)) => (
                Self::deep_size(v).saturating_sub(Self::shallow_size(v)),
                m.depth,
            ),
            None => {
                return Err(AnamnesisError::Parse {
                    reason: format!("{opcode}: memo key {key} not found"),
                });
            }
        };
        self.charge(children, "PTH pickle memo replay")?;
        let val = match self.memo.get(&key) {
            Some((v, _)) => v.clone(),
            None => {
                return Err(AnamnesisError::Parse {
                    reason: format!("{opcode}: memo key {key} not found"),
                });
            }
        };
        Ok((val, depth))
    }

    // -- main execution loop -------------------------------------------------

    /// Executes the pickle stream and returns the top-of-stack value.
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::Parse`] on malformed opcodes, stack
    /// underflow, non-allowlisted globals, or unrecognized opcodes.
    fn execute(&mut self) -> crate::Result<PickleValue> {
        loop {
            let opcode = self.read_u8()?;
            match opcode {
                // PROTO — protocol version (skip the version byte)
                0x80 => {
                    let _version = self.read_u8()?;
                }
                // FRAME — protocol 4+ frame marker (skip 8-byte length)
                0x95 => {
                    let _frame_len = self.read_u64_le()?;
                }
                // STOP — end of pickle, return top of stack
                b'.' => return self.pop().map(|(v, _)| v),

                // -- push constants ------------------------------------------
                b'N' => self.push_leaf(PickleValue::None)?,
                // NEWTRUE
                0x88 => self.push_leaf(PickleValue::Bool(true))?,
                // NEWFALSE
                0x89 => self.push_leaf(PickleValue::Bool(false))?,

                // -- integers ------------------------------------------------

                // BININT (4-byte signed)
                b'J' => {
                    let v = self.read_i32_le()?;
                    // CAST: i32 → i64, lossless widening
                    self.push_leaf(PickleValue::Int(i64::from(v)))?;
                }
                // BININT1 (1-byte unsigned)
                b'K' => {
                    let v = self.read_u8()?;
                    // CAST: u8 → i64, lossless widening
                    self.push_leaf(PickleValue::Int(i64::from(v)))?;
                }
                // BININT2 (2-byte unsigned)
                b'M' => {
                    let v = self.read_u16_le()?;
                    // CAST: u16 → i64, lossless widening
                    self.push_leaf(PickleValue::Int(i64::from(v)))?;
                }
                // LONG1 (arbitrary-precision int, 1-byte size prefix)
                0x8a => {
                    let n = self.read_u8()?;
                    // CAST: u8 → usize, lossless on all platforms
                    let bytes = self.read_bytes(usize::from(n))?;
                    let val = long1_to_i64(bytes)?;
                    self.push_leaf(PickleValue::Int(val))?;
                }

                // -- strings -------------------------------------------------

                // SHORT_BINUNICODE (1-byte length prefix)
                0x8c => {
                    let n = self.read_u8()?;
                    // CAST: u8 → usize, lossless
                    let bytes = self.read_bytes(usize::from(n))?;
                    let s = std::str::from_utf8(bytes).map_err(|e| AnamnesisError::Parse {
                        reason: format!("non-UTF-8 pickle string: {e}"),
                    })?;
                    // BORROW: .to_owned() converts &str (borrowed from pickle stream) to owned String
                    self.push_leaf(PickleValue::String(s.to_owned()))?;
                }
                // BINUNICODE (4-byte length prefix)
                b'X' => {
                    let n = self.read_u32_le()?;
                    let len = usize::try_from(n).map_err(|_| AnamnesisError::Parse {
                        reason: "BINUNICODE length overflow".into(),
                    })?;
                    enforce_pickle_payload_cap(len, "BINUNICODE")?;
                    let bytes = self.read_bytes(len)?;
                    let s = std::str::from_utf8(bytes).map_err(|e| AnamnesisError::Parse {
                        reason: format!("non-UTF-8 pickle string: {e}"),
                    })?;
                    // BORROW: .to_owned() converts &str (borrowed from pickle stream) to owned String
                    self.push_leaf(PickleValue::String(s.to_owned()))?;
                }
                // SHORT_BINSTRING (1-byte length, protocol 2 — bytes, not str)
                b'U' => {
                    let n = self.read_u8()?;
                    // CAST: u8 → usize, lossless
                    let bytes = self.read_bytes(usize::from(n))?;
                    // Protocol 2 SHORT_BINSTRING is used for ASCII identifiers;
                    // treat as UTF-8 string if valid, otherwise keep as bytes.
                    match std::str::from_utf8(bytes) {
                        // BORROW: .to_owned() converts &str to owned String
                        Ok(s) => self.push_leaf(PickleValue::String(s.to_owned()))?,
                        // BORROW: .to_vec() converts &[u8] to owned Vec<u8>
                        Err(_) => self.push_leaf(PickleValue::Bytes(bytes.to_vec()))?,
                    }
                }
                // BINSTRING (4-byte length, protocol 2)
                b'T' => {
                    let n = self.read_i32_le()?;
                    if n < 0 {
                        return Err(AnamnesisError::Parse {
                            reason: "negative BINSTRING length".into(),
                        });
                    }
                    let len = usize::try_from(n).map_err(|_| AnamnesisError::Parse {
                        reason: "BINSTRING length overflow".into(),
                    })?;
                    enforce_pickle_payload_cap(len, "BINSTRING")?;
                    let bytes = self.read_bytes(len)?;
                    match std::str::from_utf8(bytes) {
                        // BORROW: .to_owned() converts &str to owned String
                        Ok(s) => self.push_leaf(PickleValue::String(s.to_owned()))?,
                        // BORROW: .to_vec() converts &[u8] to owned Vec<u8>
                        Err(_) => self.push_leaf(PickleValue::Bytes(bytes.to_vec()))?,
                    }
                }

                // -- bytes ---------------------------------------------------

                // BINBYTES (4-byte length, protocol 3+)
                b'B' => {
                    let n = self.read_u32_le()?;
                    let len = usize::try_from(n).map_err(|_| AnamnesisError::Parse {
                        reason: "BINBYTES length overflow".into(),
                    })?;
                    enforce_pickle_payload_cap(len, "BINBYTES")?;
                    let bytes = self.read_bytes(len)?;
                    // BORROW: .to_vec() converts &[u8] (borrowed from pickle stream) to owned Vec<u8>
                    self.push_leaf(PickleValue::Bytes(bytes.to_vec()))?;
                }
                // SHORT_BINBYTES (1-byte length, protocol 3+)
                b'C' => {
                    let n = self.read_u8()?;
                    // CAST: u8 → usize, lossless
                    let bytes = self.read_bytes(usize::from(n))?;
                    // BORROW: .to_vec() converts &[u8] (borrowed from pickle stream) to owned Vec<u8>
                    self.push_leaf(PickleValue::Bytes(bytes.to_vec()))?;
                }

                // -- containers ----------------------------------------------

                // EMPTY_DICT
                b'}' => self.push_leaf(PickleValue::Dict(Vec::new()))?,
                // EMPTY_LIST
                b']' => self.push_leaf(PickleValue::List(Vec::new()))?,
                // EMPTY_TUPLE
                b')' => self.push_leaf(PickleValue::Tuple(Vec::new()))?,
                // MARK
                b'(' => self.mark_stack.push(self.stack.len()),

                // TUPLE (pop to mark → tuple)
                b't' => {
                    let (items, metas) = self.pop_mark()?;
                    let depth = Self::max_depth(&metas).saturating_add(1);
                    self.push_value(PickleValue::Tuple(items), depth)?;
                }
                // TUPLE1
                0x85 => {
                    let (a, ma) = self.pop()?;
                    self.push_value(PickleValue::Tuple(vec![a]), ma.depth.saturating_add(1))?;
                }
                // TUPLE2
                0x86 => {
                    let (b, mb) = self.pop()?;
                    let (a, ma) = self.pop()?;
                    let depth = ma.depth.max(mb.depth).saturating_add(1);
                    self.push_value(PickleValue::Tuple(vec![a, b]), depth)?;
                }
                // TUPLE3
                0x87 => {
                    let (c, mc) = self.pop()?;
                    let (b, mb) = self.pop()?;
                    let (a, ma) = self.pop()?;
                    let depth = ma.depth.max(mb.depth).max(mc.depth).saturating_add(1);
                    self.push_value(PickleValue::Tuple(vec![a, b, c]), depth)?;
                }

                // SETITEMS (pop mark → alternating key/value pairs → dict)
                b'u' => {
                    let (items, metas) = self.pop_mark()?;
                    if items.len() % 2 != 0 {
                        return Err(AnamnesisError::Parse {
                            reason: "SETITEMS: odd number of items on stack".into(),
                        });
                    }
                    let child_depth = Self::max_depth(&metas).saturating_add(1);
                    let dict = self.stack.last_mut().ok_or_else(|| AnamnesisError::Parse {
                        reason: "SETITEMS: empty stack (no dict)".into(),
                    })?;
                    if let PickleValue::Dict(ref mut pairs) = *dict {
                        let mut iter = items.into_iter();
                        while let Some(key) = iter.next() {
                            // EXPLICIT: the odd-length check above guarantees
                            // a value exists for every key.
                            let val = iter.next().ok_or_else(|| AnamnesisError::Parse {
                                reason: "SETITEMS: missing value for key".into(),
                            })?;
                            pairs.push((key, val));
                        }
                    } else {
                        return Err(AnamnesisError::Parse {
                            reason: "SETITEMS: top of stack is not a dict".into(),
                        });
                    }
                    self.bump_top_depth(child_depth)?;
                }
                // SETITEM (pop value, key → dict)
                b's' => {
                    let (value, mv) = self.pop()?;
                    let (key, mk) = self.pop()?;
                    let child_depth = mk.depth.max(mv.depth).saturating_add(1);
                    let dict = self.stack.last_mut().ok_or_else(|| AnamnesisError::Parse {
                        reason: "SETITEM: empty stack (no dict)".into(),
                    })?;
                    if let PickleValue::Dict(ref mut pairs) = *dict {
                        pairs.push((key, value));
                    } else {
                        return Err(AnamnesisError::Parse {
                            reason: "SETITEM: top of stack is not a dict".into(),
                        });
                    }
                    self.bump_top_depth(child_depth)?;
                }
                // APPEND
                b'a' => {
                    let (item, mi) = self.pop()?;
                    let child_depth = mi.depth.saturating_add(1);
                    let list = self.stack.last_mut().ok_or_else(|| AnamnesisError::Parse {
                        reason: "APPEND: empty stack (no list)".into(),
                    })?;
                    if let PickleValue::List(ref mut items) = *list {
                        items.push(item);
                    } else {
                        return Err(AnamnesisError::Parse {
                            reason: "APPEND: top of stack is not a list".into(),
                        });
                    }
                    self.bump_top_depth(child_depth)?;
                }
                // APPENDS
                b'e' => {
                    let (new_items, metas) = self.pop_mark()?;
                    let child_depth = Self::max_depth(&metas).saturating_add(1);
                    let list = self.stack.last_mut().ok_or_else(|| AnamnesisError::Parse {
                        reason: "APPENDS: empty stack (no list)".into(),
                    })?;
                    if let PickleValue::List(ref mut items) = *list {
                        items.extend(new_items);
                    } else {
                        return Err(AnamnesisError::Parse {
                            reason: "APPENDS: top of stack is not a list".into(),
                        });
                    }
                    self.bump_top_depth(child_depth)?;
                }

                // -- object construction -------------------------------------

                // GLOBAL (text mode: "module\nname\n")
                b'c' => {
                    // BORROW: .to_owned() converts &str (borrowed from pickle stream) to owned String
                    let module = self.read_line()?.to_owned();
                    // BORROW: .to_owned() converts &str (borrowed from pickle stream) to owned String
                    let name = self.read_line()?.to_owned();
                    if !is_allowed_global(&module, &name) {
                        return Err(AnamnesisError::DisallowedGlobal { module, name });
                    }
                    self.push_leaf(PickleValue::Global { module, name })?;
                }
                // STACK_GLOBAL (protocol 4+: pop name, pop module from stack)
                0x93 => {
                    let (name_val, _) = self.pop()?;
                    let (module_val, _) = self.pop()?;
                    let (module, name) = match (&module_val, &name_val) {
                        (PickleValue::String(m), PickleValue::String(n)) => {
                            (m.as_str(), n.as_str())
                        }
                        _ => {
                            return Err(AnamnesisError::Parse {
                                reason: "STACK_GLOBAL: module/name are not strings".into(),
                            });
                        }
                    };
                    if !is_allowed_global(module, name) {
                        return Err(AnamnesisError::DisallowedGlobal {
                            module: module.to_owned(),
                            name: name.to_owned(),
                        });
                    }
                    self.push_leaf(PickleValue::Global {
                        // BORROW: .to_owned() converts &str to owned String
                        module: module.to_owned(),
                        // BORROW: .to_owned() converts &str to owned String
                        name: name.to_owned(),
                    })?;
                }
                // REDUCE (pop args, pop callable → Reduced)
                // NEWOBJ (pop args, pop cls → Reduced) — same semantics
                b'R' | 0x81 => {
                    let (args, m_args) = self.pop()?;
                    let (callable, m_callable) = self.pop()?;
                    // Semantic interpretation: REDUCE(OrderedDict, ()) → empty Dict.
                    // Python actually calls OrderedDict() here, producing a real dict
                    // that SETITEMS will later populate. Without this, SETITEMS fails.
                    if is_ordered_dict_constructor(&callable, &args) {
                        self.push_leaf(PickleValue::Dict(Vec::new()))?;
                    } else {
                        let depth = m_args.depth.max(m_callable.depth).saturating_add(1);
                        self.push_value(
                            PickleValue::Reduced {
                                callable: Box::new(callable),
                                args: Box::new(args),
                            },
                            depth,
                        )?;
                    }
                }
                // BUILD (pop state, peek obj → Built)
                b'b' => {
                    let (state, m_state) = self.pop()?;
                    let (obj, m_obj) = self.pop()?;
                    let depth = m_obj.depth.max(m_state.depth).saturating_add(1);
                    self.push_value(
                        PickleValue::Built {
                            obj: Box::new(obj),
                            state: Box::new(state),
                        },
                        depth,
                    )?;
                }
                // BINPERSID (pop id → PersistentId)
                b'Q' => {
                    let (pid, m_pid) = self.pop()?;
                    self.push_value(
                        PickleValue::PersistentId(Box::new(pid)),
                        m_pid.depth.saturating_add(1),
                    )?;
                }

                // -- memo operations -----------------------------------------

                // BINPUT (1-byte key)
                b'q' => {
                    let key = self.read_u8()?;
                    let entry = self.memoize_top("BINPUT")?;
                    // CAST: u8 → u32, lossless
                    self.memo.insert(u32::from(key), entry);
                }
                // LONG_BINPUT (4-byte key)
                b'r' => {
                    let key = self.read_u32_le()?;
                    let entry = self.memoize_top("LONG_BINPUT")?;
                    self.memo.insert(key, entry);
                }
                // BINGET (1-byte key)
                b'h' => {
                    let key = self.read_u8()?;
                    // CAST: u8 → u32, lossless
                    let (val, depth) = self.memo_get_clone(u32::from(key), "BINGET")?;
                    self.push_value(val, depth)?;
                }
                // LONG_BINGET (4-byte key)
                b'j' => {
                    let key = self.read_u32_le()?;
                    let (val, depth) = self.memo_get_clone(key, "LONG_BINGET")?;
                    self.push_value(val, depth)?;
                }
                // MEMOIZE (protocol 4+: auto-assigns next key)
                0x94 => {
                    let entry = self.memoize_top("MEMOIZE")?;
                    self.memo.insert(self.next_memo_id, entry);
                    self.next_memo_id =
                        self.next_memo_id
                            .checked_add(1)
                            .ok_or_else(|| AnamnesisError::Parse {
                                reason: "pickle memo table overflow (>2^32 MEMOIZE opcodes)".into(),
                            })?;
                }

                _ => {
                    return Err(AnamnesisError::Parse {
                        reason: format!("unsupported pickle opcode 0x{opcode:02x}"),
                    });
                }
            }
        }
    }
}

/// Converts a pickle `LONG1` byte sequence to `i64`.
///
/// Pickle's `LONG1` stores an arbitrary-precision signed integer in
/// little-endian two's-complement. We support values that fit in `i64`.
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if the value exceeds `i64` range.
fn long1_to_i64(bytes: &[u8]) -> crate::Result<i64> {
    if bytes.is_empty() {
        return Ok(0);
    }
    if bytes.len() > 8 {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "LONG1 value too large ({} bytes, max 8 for i64)",
                bytes.len()
            ),
        });
    }
    // Two's complement: if the high bit of the last byte is set, the
    // number is negative. Pad with 0xFF for negative, 0x00 for positive.
    let last = bytes.last().copied().ok_or_else(|| AnamnesisError::Parse {
        reason: "LONG1 empty bytes".into(),
    })?;
    let pad = if last & 0x80 != 0 { 0xFF } else { 0x00 };
    let mut buf = [pad; 8];
    // bytes.len() ≤ 8 checked above; .get_mut returns Some
    let dest = buf
        .get_mut(..bytes.len())
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "LONG1 internal: slice bounds exceeded".into(),
        })?;
    dest.copy_from_slice(bytes);
    Ok(i64::from_le_bytes(buf))
}

// ---------------------------------------------------------------------------
// Tensor extraction
// ---------------------------------------------------------------------------

/// Intermediate representation of a tensor reference parsed from the pickle.
struct TensorRef {
    name: String,
    /// Data file index in the ZIP archive (e.g., `"0"` → `archive/data/0`).
    storage_key: String,
    dtype: PthDtype,
    /// Byte offset into the storage file.
    storage_offset: usize,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

/// Attempts to extract an `i64` from a `PickleValue::Int`.
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if the value is not an `Int`.
fn as_i64(val: &PickleValue) -> crate::Result<i64> {
    if let PickleValue::Int(v) = val {
        Ok(*v)
    } else {
        Err(AnamnesisError::Parse {
            reason: format!("expected int, got {val:?}"),
        })
    }
}

/// Attempts to extract a `usize` from a `PickleValue::Int`.
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if the value is not an `Int` or
/// does not fit in `usize`.
fn as_usize(val: &PickleValue) -> crate::Result<usize> {
    let v = as_i64(val)?;
    usize::try_from(v).map_err(|_| AnamnesisError::Parse {
        reason: format!("integer {v} does not fit in usize"),
    })
}

/// Attempts to extract a `&str` from a `PickleValue::String`.
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if the value is not a `String`.
fn as_str(val: &PickleValue) -> crate::Result<&str> {
    if let PickleValue::String(s) = val {
        Ok(s.as_str())
    } else {
        Err(AnamnesisError::Parse {
            reason: format!("expected string, got {val:?}"),
        })
    }
}

/// Converts a `PickleValue::Tuple` into a `Vec<usize>` (for shape/strides).
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if the value is not a `Tuple` or
/// any element is not a non-negative integer fitting in `usize`.
fn tuple_to_usize_vec(val: &PickleValue) -> crate::Result<Vec<usize>> {
    if let PickleValue::Tuple(items) = val {
        items.iter().map(as_usize).collect()
    } else {
        Err(AnamnesisError::Parse {
            reason: format!("expected tuple, got {val:?}"),
        })
    }
}

/// Parses a `_rebuild_tensor_v2` call's arguments into a `TensorRef`.
///
/// Expected args tuple:
/// `(PersistentId(storage_info), offset, shape, strides, requires_grad, metadata)`
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if `args` is not a 6-element `Tuple`,
/// if the storage info is malformed, or if shape/strides/offset values
/// cannot be extracted as valid dimensions.
// EXHAUSTIVE: PickleValue is private; wildcards catch irrelevant variants
#[allow(clippy::wildcard_enum_match_arm)]
fn parse_rebuild_args(name: &str, args: &PickleValue) -> crate::Result<TensorRef> {
    let PickleValue::Tuple(items) = args else {
        return Err(AnamnesisError::Parse {
            reason: format!("tensor `{name}`: expected tuple args for _rebuild_tensor_v2"),
        });
    };

    if items.len() < 4 {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "tensor `{name}`: _rebuild_tensor_v2 needs ≥4 args, got {}",
                items.len()
            ),
        });
    }

    // Item 0: PersistentId wrapping a tuple of (tag, storage_type, key, device, n_elements)
    let persistent_id = items.first().ok_or_else(|| AnamnesisError::Parse {
        reason: format!("tensor `{name}`: missing args[0]"),
    })?;
    let storage_tuple = match persistent_id {
        PickleValue::PersistentId(inner) => match inner.as_ref() {
            PickleValue::Tuple(t) => t,
            other => {
                return Err(AnamnesisError::Parse {
                    reason: format!(
                        "tensor `{name}`: PersistentId payload is not a tuple: {other:?}"
                    ),
                });
            }
        },
        other => {
            return Err(AnamnesisError::Parse {
                reason: format!("tensor `{name}`: expected PersistentId, got {other:?}"),
            });
        }
    };

    if storage_tuple.len() < 5 {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "tensor `{name}`: storage tuple needs ≥5 items, got {}",
                storage_tuple.len()
            ),
        });
    }

    // storage_tuple[1] is the storage type Global, [2] is the data file key
    let st1 = storage_tuple.get(1).ok_or_else(|| AnamnesisError::Parse {
        reason: format!("tensor `{name}`: missing storage_tuple[1]"),
    })?;
    let dtype = match st1 {
        PickleValue::Global { module, name: cls } => PthDtype::from_storage_class(module, cls)?,
        other => {
            return Err(AnamnesisError::Parse {
                reason: format!("tensor `{name}`: expected storage Global, got {other:?}"),
            });
        }
    };
    let st2 = storage_tuple.get(2).ok_or_else(|| AnamnesisError::Parse {
        reason: format!("tensor `{name}`: missing storage_tuple[2]"),
    })?;
    // BORROW: owned copy needed — TensorMeta outlives the PickleValue borrow
    let storage_key = as_str(st2)?.to_owned();

    // items[1]: storage offset (in elements, not bytes)
    let it1 = items.get(1).ok_or_else(|| AnamnesisError::Parse {
        reason: format!("tensor `{name}`: missing args[1]"),
    })?;
    let storage_offset_elements = as_usize(it1)?;
    let storage_offset = storage_offset_elements
        .checked_mul(dtype.byte_size())
        .ok_or_else(|| AnamnesisError::Parse {
            reason: format!("tensor `{name}`: storage offset overflow"),
        })?;

    // items[2], items[3]: shape and strides
    let it2 = items.get(2).ok_or_else(|| AnamnesisError::Parse {
        reason: format!("tensor `{name}`: missing args[2]"),
    })?;
    let it3 = items.get(3).ok_or_else(|| AnamnesisError::Parse {
        reason: format!("tensor `{name}`: missing args[3]"),
    })?;
    let shape = tuple_to_usize_vec(it2)?;
    let strides = tuple_to_usize_vec(it3)?;

    if shape.len() != strides.len() {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "tensor `{name}`: shape ndim {} != strides ndim {}",
                shape.len(),
                strides.len()
            ),
        });
    }

    Ok(TensorRef {
        // BORROW: owned copy — TensorRef outlives the PickleValue borrow
        name: name.to_owned(),
        storage_key,
        dtype,
        storage_offset,
        shape,
        strides,
    })
}

/// Maximum nesting depth for recursive pickle value extraction.
///
/// Real `.pth` files have at most 2–3 levels of `Built`/`Reduced`
/// wrapping:
/// - **Level 0**: `Dict` (the `state_dict` itself)
/// - **Level 1**: `Built { Dict, metadata }` (`OrderedDict` + `__dict__`)
/// - **Level 2**: `Reduced { _rebuild_parameter, ... }` wrapping a tensor
///
/// 32 is generous — it prevents stack overflow from adversarial pickles
/// with deeply nested `BUILD` opcodes while accepting any realistic file.
const MAX_PICKLE_NESTING: u32 = 32;

// EXHAUSTIVE: PickleValue is private; wildcards catch irrelevant variants
#[allow(clippy::wildcard_enum_match_arm)]
/// Unwraps nested pickle structures to find the inner `_rebuild_tensor_v2` call.
///
/// Handles:
/// - Direct: `Reduced { _rebuild_tensor_v2, args }`
/// - Parameter-wrapped: `Reduced { _rebuild_parameter, Tuple([Reduced { _rebuild_tensor_v2, args }, ...]) }`
/// - `Built`-wrapped: `Built { obj, state }` → recurse into `obj`
///
/// `depth` guards against stack overflow from adversarial pickle nesting.
fn unwrap_to_rebuild(val: &PickleValue, depth: u32) -> Option<(&PickleValue, &PickleValue)> {
    if depth > MAX_PICKLE_NESTING {
        return None;
    }
    match val {
        PickleValue::Reduced { callable, args, .. } => {
            if let PickleValue::Global { module, name } = callable.as_ref() {
                if module == "torch._utils" && name == "_rebuild_tensor_v2" {
                    return Some((callable, args));
                }
                // _rebuild_parameter wraps _rebuild_tensor_v2 as first arg:
                // REDUCE(_rebuild_parameter, TUPLE(REDUCE(_rebuild_tensor_v2, ...), ...))
                if module == "torch._utils"
                    && (name == "_rebuild_parameter" || name == "_rebuild_parameter_with_state")
                    && let PickleValue::Tuple(items) = args.as_ref()
                    && let Some(first) = items.first()
                {
                    return unwrap_to_rebuild(first, depth + 1);
                }
            }
            None
        }
        PickleValue::Built { obj, .. } => unwrap_to_rebuild(obj, depth + 1),
        _ => None,
    }
}

/// Extracts the dict of key→value pairs from the top-level pickle value.
///
/// Handles both a raw `Dict` and a `Reduced { OrderedDict, ... }`.
/// `depth` guards against stack overflow from adversarial pickle nesting.
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if the root value is not a `Dict`,
/// `Built`, or recognized `Reduced`, or if nesting exceeds
/// [`MAX_PICKLE_NESTING`].
// EXHAUSTIVE: PickleValue is private; wildcards catch irrelevant variants
#[allow(clippy::wildcard_enum_match_arm)]
fn extract_dict_pairs(
    root: &PickleValue,
    depth: u32,
) -> crate::Result<&[(PickleValue, PickleValue)]> {
    if depth > MAX_PICKLE_NESTING {
        return Err(AnamnesisError::Parse {
            reason: "pickle nesting limit exceeded in extract_dict_pairs".into(),
        });
    }
    match root {
        PickleValue::Dict(pairs) => Ok(pairs),
        PickleValue::Reduced { callable, args: _ } => {
            // REDUCE(OrderedDict, ()) is converted to Dict by
            // is_ordered_dict_constructor() in execute(). If a Reduced
            // {OrderedDict} reaches here, it indicates a bug in the VM
            // or an unrecognized opcode sequence. Returning Ok(&[]) would
            // silently lose all tensors — always error instead.
            if let PickleValue::Global { module, name } = callable.as_ref()
                && module == "collections"
                && name == "OrderedDict"
            {
                return Err(AnamnesisError::Parse {
                    reason: "OrderedDict arrived as Reduced (expected Dict \
                             after REDUCE rewrite); possible pickle VM bug"
                        .into(),
                });
            }
            Err(AnamnesisError::Parse {
                reason: format!("top-level pickle value is not a dict: {root:?}"),
            })
        }
        PickleValue::Built { obj, state: _ } => {
            // BUILD(obj, state) sets obj.__dict__ = state. The tensor data
            // lives in obj (the OrderedDict), not in state (which is the
            // __dict__ containing _metadata). Always recurse into obj.
            extract_dict_pairs(obj, depth + 1)
        }
        _ => Err(AnamnesisError::Parse {
            reason: format!("top-level pickle value is not a dict or OrderedDict: {root:?}"),
        }),
    }
}

/// Computes the expected strides for a contiguous (row-major) tensor.
fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let ndim = shape.len();
    let mut strides = vec![1usize; ndim];
    // Walk right-to-left, accumulating the product.
    // EXPLICIT: manual indexing used because each stride depends on the
    // previously computed stride[i+1]; an iterator chain cannot express this.
    for i in (0..ndim.saturating_sub(1)).rev() {
        // .get() returns Some because i < ndim-1 ⇒ i+1 < ndim
        if let (Some(&prev), Some(&dim)) = (strides.get(i + 1), shape.get(i + 1))
            && let Some(s) = strides.get_mut(i)
        {
            // saturating_mul: overflow on astronomic shapes produces
            // usize::MAX, causing is_contiguous() to return false
            // and copy_to_contiguous() to fail with a checked error
            // — safe but intentional degradation.
            *s = prev.saturating_mul(dim);
        }
    }
    strides
}

/// Returns `true` if the tensor's strides match contiguous (row-major) layout.
fn is_contiguous(shape: &[usize], strides: &[usize]) -> bool {
    if shape.len() != strides.len() {
        return false;
    }
    let expected = contiguous_strides(shape);
    strides == expected
}

/// Copies a non-contiguous tensor to contiguous (row-major) layout.
///
/// This is the slow path for tensors with non-standard strides (e.g.,
/// transposed views). Rare in `state_dict` files but must be handled for
/// correctness.
///
/// Uses the two-level bounds pattern from CONVENTIONS.md: validate the
/// maximum reachable source byte offset **once** before the loop, then
/// use plain arithmetic inside. The output buffer is pre-allocated to
/// the exact size, so destination writes are also bounds-safe.
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if shape and strides have different
/// lengths, element count or byte count overflows `usize`, if the maximum
/// stride offset overflows, or if the source data range exceeds the
/// storage slice.
fn copy_to_contiguous(
    storage: &[u8],
    offset: usize,
    shape: &[usize],
    strides: &[usize],
    elem_size: usize,
) -> crate::Result<Vec<u8>> {
    if shape.len() != strides.len() {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "shape ndim {} != strides ndim {}",
                shape.len(),
                strides.len()
            ),
        });
    }

    let n_elements: usize = shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "element count overflow".into(),
        })?;
    let out_bytes = n_elements
        .checked_mul(elem_size)
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "output size overflow".into(),
        })?;

    // -- Pre-validation: compute max reachable source byte offset -----------
    //
    // The maximum element offset (in elements) is sum(stride[i] * (shape[i]-1))
    // across all dimensions. The maximum byte offset is:
    //   offset + max_elem_offset * elem_size + elem_size
    // If this fits within storage, every inner-loop access is in bounds.
    let max_elem_offset: usize = shape
        .iter()
        .zip(strides.iter())
        .try_fold(0usize, |acc, (&dim, &stride)| {
            dim.checked_sub(1)
                .and_then(|d| d.checked_mul(stride))
                .and_then(|ds| acc.checked_add(ds))
        })
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "max stride offset overflow".into(),
        })?;
    let max_src_end = offset
        .checked_add(max_elem_offset.checked_mul(elem_size).ok_or_else(|| {
            AnamnesisError::Parse {
                reason: "max source byte offset overflow".into(),
            }
        })?)
        .and_then(|b| b.checked_add(elem_size))
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "max source end offset overflow".into(),
        })?;
    if max_src_end > storage.len() {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "non-contiguous tensor: max source byte [{max_src_end}] \
                 exceeds storage len {}",
                storage.len()
            ),
        });
    }

    // -- Inner loop: plain arithmetic, no per-element bounds checks ----------
    let mut out = vec![0u8; out_bytes];
    let ndim = shape.len();
    let mut coords = vec![0usize; ndim];

    // VECTORIZED: scalar fallback — per-element coordinate tracking (coords[]
    // update) introduces cross-iteration state that prevents auto-vectorization.
    // Non-contiguous tensors are rare in practice (<0.1% of state_dict files).
    for flat_idx in 0..n_elements {
        // Compute source element offset from strides and coordinates.
        // All values are bounded by the pre-validation above.
        let src_elem_offset: usize = coords
            .iter()
            .zip(strides.iter())
            .map(|(&c, &s)| c * s)
            .sum();
        let src_byte = offset + src_elem_offset * elem_size;
        let dst_byte = flat_idx * elem_size;

        // INDEX: src_byte + elem_size ≤ max_src_end ≤ storage.len(),
        // validated before the loop. dst_byte + elem_size ≤ out_bytes = out.len().
        #[allow(clippy::indexing_slicing)]
        out[dst_byte..dst_byte + elem_size]
            .copy_from_slice(&storage[src_byte..src_byte + elem_size]);

        // Increment coordinates (rightmost dimension first).
        // EXPLICIT: manual coordinate increment; iterator-based
        // multi-index generation would allocate per-element.
        for d in (0..ndim).rev() {
            if let (Some(c), Some(&s)) = (coords.get_mut(d), shape.get(d)) {
                *c += 1;
                if *c < s {
                    break;
                }
                *c = 0;
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Parses a `PyTorch` `.pth` `state_dict` file.
///
/// Returns a [`ParsedPth`] that owns the memory-mapped file and provides
/// zero-copy access via [`ParsedPth::tensors()`]. Tensor order matches the
/// `OrderedDict` insertion order from the original Python `state_dict`.
///
/// # Supported Formats
///
/// Only modern `.pth` files (`PyTorch` ≥ 1.6, ZIP-based) are supported.
/// Legacy raw-pickle files (pre-1.6) are rejected with
/// [`AnamnesisError::Unsupported`].
///
/// # Security
///
/// The pickle interpreter uses an explicit `GLOBAL` allowlist. Non-`PyTorch`
/// callables (e.g., `os.system`, `subprocess.Popen`) are rejected with
/// [`AnamnesisError::Parse`], preventing arbitrary code execution.
///
/// Against unguarded-allocation denial of service, the declared `data.pkl`
/// size is capped at `MAX_PKL_SIZE` before the entry is sliced from the mmap
/// and interpreted (shared with [`inspect_pth_from_reader`] via
/// `enforce_pkl_size_cap`), and every 4-byte-length pickle payload is bounded
/// by `MAX_PICKLE_PAYLOAD`. Oversized declarations return
/// [`AnamnesisError::Parse`] rather than allocating.
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if the file is not a valid `PyTorch`
/// ZIP archive or uses unsupported pickle opcodes.
///
/// Returns [`AnamnesisError::DisallowedGlobal`] for a pickle `GLOBAL` outside
/// the `torch.*` allowlist, and [`AnamnesisError::LimitExceeded`] for a
/// permanent-cap rejection (`data.pkl` size, pickle payload / working-set /
/// nesting depth). Both are always-on, so reachable even at the default
/// (unbounded) limits this wrapper passes.
///
/// Returns [`AnamnesisError::Unsupported`] for legacy (pre-1.6) `.pth`
/// files that are raw pickle without ZIP wrapping.
///
/// Returns [`AnamnesisError::Io`] if the file cannot be read.
///
/// # Memory
///
/// Memory-maps the file with `memmap2` (page prefaulting). Tensor data
/// is **not** copied during parsing — [`ParsedPth::tensors()`] slices
/// directly from the mmap (zero-copy for contiguous little-endian tensors,
/// which is >99% of real files). Peak memory during parsing ≈ file size
/// (mapped, not heap-allocated) + ~1 KB metadata per tensor. The mmap is
/// released when the returned `ParsedPth` is dropped.
pub fn parse_pth(path: impl AsRef<Path>) -> crate::Result<ParsedPth> {
    parse_pth_with_limits(path, &ParseLimits::default())
}

/// Parses a `.pth` archive under a caller-supplied [`ParseLimits`] budget.
///
/// Identical to [`parse_pth`] but enforces the caller's [`ParseLimits`] —
/// fail-fast, before allocation — over the `ZIP` container metadata (the
/// declared central-directory entry count, via `max_item_count`), the declared
/// `data.pkl` entry size (the per-allocation cap), and each 4-byte-length
/// pickle payload (per-allocation **and** the cumulative-byte aggregate, since
/// each payload is an owned clone). The built-in `ZIP_MAX_ENTRIES` /
/// `MAX_PKL_SIZE` / `MAX_PICKLE_PAYLOAD` caps still apply; `limits` can only
/// tighten them. [`parse_pth`] is the `ParseLimits::default()` (unbounded)
/// special case.
///
/// The pickle VM's **working set** — the value stack, memoised clones, and
/// nested values — is governed (Phase 6.11): every value the VM allocates is
/// charged through the same `Budget` the payloads use, so the cumulative-byte
/// aggregate covers the whole heap the VM builds, not just the string/bytes
/// payloads. Two always-on permanent floors back this independently of the
/// caller's limits: `MAX_PICKLE_WORKING_SET` (total VM heap) bounds
/// value-stack and memo-replay amplification (`N`-floods, `BINGET` replay of a
/// large memoised subtree), and `MAX_PICKLE_VM_DEPTH` (construction nesting
/// depth) prevents an over-deep value whose recursive `Drop` would overflow
/// the stack. The `MAX_PKL_SIZE` opcode-stream cap bounds the *stream*; these
/// caps bound the *heap each opcode produces* — which the stream cap alone
/// does not.
///
/// # Errors
///
/// Returns [`AnamnesisError::LimitExceeded`] if a declared `ZIP` entry count,
/// `data.pkl` size, or pickle payload / working-set / nesting depth exceeds
/// `limits` or a permanent cap.
/// Returns [`AnamnesisError::DisallowedGlobal`] if the pickle references a
/// `GLOBAL` outside the `torch.*` allowlist.
/// Returns [`AnamnesisError::Parse`] if the file is not a valid `PyTorch` ZIP
/// archive or uses unsupported pickle opcodes.
///
/// Returns [`AnamnesisError::Unsupported`] for legacy (pre-1.6) `.pth`
/// files that are raw pickle without ZIP wrapping.
///
/// Returns [`AnamnesisError::Io`] if the file cannot be read.
///
/// # Memory
///
/// Memory-maps the file with `memmap2` (page prefaulting). Tensor data
/// is **not** copied during parsing — [`ParsedPth::tensors()`] slices
/// directly from the mmap. Peak memory during parsing ≈ file size (mapped,
/// not heap-allocated) + ~1 KB metadata per tensor. The mmap is released
/// when the returned `ParsedPth` is dropped.
#[allow(unsafe_code)]
pub fn parse_pth_with_limits(
    path: impl AsRef<Path>,
    limits: &ParseLimits,
) -> crate::Result<ParsedPth> {
    let file = std::fs::File::open(path.as_ref())?;
    // SAFETY: memmap2::Mmap requires unsafe because the OS could modify the
    // mapped region if another process writes to the file concurrently.
    // Model files are read-only artifacts — this is the standard assumption
    // for all tensor format parsers (same as safetensors crate's mmap path).
    // Untrusted callers that cannot make that assumption use `parse_pth_bytes`
    // / `parse_pth_from_reader` instead (no mmap, no `SIGBUS`).
    let raw =
        unsafe { memmap2::MmapOptions::new().populate().map(&file) }.map_err(AnamnesisError::Io)?;
    parsed_pth_from_backing(Backing::Mmap(raw), limits)
}

/// Builds a [`ParsedPth`] from an already-acquired byte backing — the single
/// construction site shared by the mmap path ([`parse_pth_with_limits`]) and
/// the copy-based paths ([`parse_pth_bytes_with_limits`] /
/// [`parse_pth_from_reader_with_limits`]). All parsing runs over the backing as
/// a `&[u8]`, so the logic cannot drift between paths; the backing is retained
/// so [`ParsedPth::tensors`] can borrow tensor data from it.
fn parsed_pth_from_backing(buffer: Backing, limits: &ParseLimits) -> crate::Result<ParsedPth> {
    let raw: &[u8] = &buffer;

    // Legacy format detection: check ZIP magic before attempting to parse.
    let magic = raw.get(..4).ok_or_else(|| AnamnesisError::Parse {
        reason: "file too small to be a .pth archive".into(),
    })?;
    if magic.first() == Some(&0x80) && magic.get(1).is_some_and(|&b| b <= 0x05) {
        return Err(AnamnesisError::Unsupported {
            format: "pth".into(),
            detail: "legacy .pth format (pre-PyTorch 1.6) is not supported; \
                     re-save with torch.save()"
                .into(),
        });
    }
    if magic != b"PK\x03\x04" {
        return Err(AnamnesisError::Parse {
            reason: "file is not a ZIP archive (missing PK\\x03\\x04 magic)".into(),
        });
    }

    // 1. Pre-index all ZIP entry names → (data_start, size) for O(1) lookup,
    //    over the vendored central-directory reader (Phase 6.12). This replaces
    //    the O(n) find_entry_name scanning per tensor. `limits` bounds the
    //    container metadata fail-fast.
    let entry_index = build_entry_index(raw, limits)?;

    // 2. Read byte order (default to little-endian).
    let big_endian = match entry_index.get("byteorder") {
        Some((start, len)) => {
            let bytes = raw
                .get(start..start + len)
                .ok_or_else(|| AnamnesisError::Parse {
                    reason: "byteorder entry out of bounds".into(),
                })?;
            let text = std::str::from_utf8(bytes).map_err(|e| AnamnesisError::Parse {
                reason: format!("byteorder entry is not UTF-8: {e}"),
            })?;
            match text.trim() {
                "little" => false,
                "big" => true,
                other => {
                    return Err(AnamnesisError::Parse {
                        reason: format!(
                            "unknown byte order `{other}` (expected `little` or `big`)"
                        ),
                    });
                }
            }
        }
        None => false, // default: little-endian
    };

    // 3. Read and execute the pickle stream, then extract tensor metadata.
    //    Data is NOT copied here — `tensors()` will borrow from the backing.
    let (pkl_start, pkl_len) =
        entry_index
            .get("data.pkl")
            .ok_or_else(|| AnamnesisError::Parse {
                reason: "ZIP entry `data.pkl` not found".into(),
            })?;
    // Bound the pickle stream before slicing/interpreting it. `build_entry_index`
    // already guarantees `pkl_start + pkl_len <= raw.len()`, so this fires only
    // for files genuinely larger than the cap — closing the multi-GiB-`data.pkl`
    // case the reader path already guards against.
    let pkl_len_u64 = u64::try_from(pkl_len).map_err(|_| AnamnesisError::Parse {
        reason: format!("ZIP entry `data.pkl`: declared size {pkl_len} overflows u64"),
    })?;
    enforce_pkl_size_cap(pkl_len_u64, "data.pkl", limits)?;
    let pkl_end = pkl_start
        .checked_add(pkl_len)
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "data.pkl slice end offset overflow".into(),
        })?;
    let pkl_data = raw
        .get(pkl_start..pkl_end)
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "data.pkl slice out of bounds".into(),
        })?;
    let meta = interpret_pickle_to_meta(pkl_data, limits)?;

    Ok(ParsedPth {
        buffer,
        meta,
        entry_index,
        big_endian,
    })
}

/// Parses `.pth` bytes already held in memory, returning a [`ParsedPth`] that
/// **owns** them — the copy-based, mmap-free path.
///
/// **Recommended for untrusted input**: no mmap, so a truncated or hostile
/// source is a clean `Err`, never a `SIGBUS`. [`parse_pth_bytes`] is the
/// [`ParseLimits::default`] (unbounded) special case of
/// [`parse_pth_bytes_with_limits`].
///
/// # Errors
///
/// Returns the same errors as [`parse_pth_with_limits`]:
/// [`AnamnesisError::Parse`] (invalid ZIP / unsupported pickle opcodes),
/// [`AnamnesisError::DisallowedGlobal`] (a `GLOBAL` outside the `torch.*`
/// allowlist), [`AnamnesisError::LimitExceeded`] (a `data.pkl` size or pickle
/// payload / working-set / nesting depth exceeding a permanent cap — always-on,
/// so reachable even at the default limits this wrapper passes), and
/// [`AnamnesisError::Unsupported`] (legacy raw-pickle format).
///
/// # Memory
///
/// Takes ownership of `bytes` (no copy); peak heap is the input size plus the
/// per-tensor metadata.
pub fn parse_pth_bytes(bytes: Vec<u8>) -> crate::Result<ParsedPth> {
    parse_pth_bytes_with_limits(bytes, &ParseLimits::default())
}

/// Parses owned `.pth` bytes under a caller-supplied [`ParseLimits`] budget —
/// the bounded, mmap-free path for untrusted input.
///
/// Rejects an input larger than [`ParseLimits::max_single_alloc_bytes`] before
/// parsing, then enforces every applicable `limits` ceiling exactly as
/// [`parse_pth_with_limits`] does.
///
/// # Errors
///
/// Returns [`AnamnesisError::LimitExceeded`] if `bytes` exceeds `limits`.
/// Returns [`AnamnesisError::Parse`] on the malformed-input conditions of
/// [`parse_pth_with_limits`]; [`AnamnesisError::DisallowedGlobal`] for a pickle
/// `GLOBAL` outside the `torch.*` allowlist.
/// Returns [`AnamnesisError::Unsupported`] for a legacy (pre-1.6) `.pth` file.
///
/// # Memory
///
/// Takes ownership of `bytes` (no copy); peak heap is the input size plus the
/// per-tensor metadata.
pub fn parse_pth_bytes_with_limits(
    bytes: Vec<u8>,
    limits: &ParseLimits,
) -> crate::Result<ParsedPth> {
    let len = u64::try_from(bytes.len()).map_err(|_| AnamnesisError::Parse {
        reason: "pth bytes: length overflows u64".into(),
    })?;
    limits.check_alloc(len, "pth bytes")?;
    parsed_pth_from_backing(Backing::Owned(bytes), limits)
}

/// Parses a `.pth` artefact from any reader, returning a [`ParsedPth`] that
/// **owns** the bytes — the copy-based, mmap-free path.
///
/// **Recommended for untrusted streamed input.** [`parse_pth_from_reader`] is
/// the [`ParseLimits::default`] (unbounded) special case of
/// [`parse_pth_from_reader_with_limits`].
///
/// # Errors
///
/// Returns [`AnamnesisError::Io`] if the reader fails;
/// [`AnamnesisError::Parse`] on malformed input;
/// [`AnamnesisError::DisallowedGlobal`] for a pickle `GLOBAL` outside the
/// `torch.*` allowlist; [`AnamnesisError::LimitExceeded`] for a permanent-cap
/// rejection (both always-on, so reachable even at default limits); or
/// [`AnamnesisError::Unsupported`] for a legacy `.pth` file.
///
/// # Memory
///
/// Reads the whole stream into an owned `Vec<u8>`; peak heap is the artefact
/// size plus the per-tensor metadata.
pub fn parse_pth_from_reader<R: Read>(reader: R) -> crate::Result<ParsedPth> {
    parse_pth_from_reader_with_limits(reader, &ParseLimits::default())
}

/// Parses a `.pth` artefact from any reader under a caller-supplied
/// [`ParseLimits`] budget — the bounded, mmap-free path for untrusted input.
///
/// The read is bounded by [`ParseLimits::max_single_alloc_bytes`] so an
/// unbounded or hostile stream cannot exhaust memory.
///
/// # Errors
///
/// Returns [`AnamnesisError::Io`] if the reader fails.
/// Returns [`AnamnesisError::LimitExceeded`] if the bytes read exceed `limits`.
/// Returns [`AnamnesisError::Parse`] on the malformed-input conditions of
/// [`parse_pth_with_limits`]; [`AnamnesisError::DisallowedGlobal`] for a pickle
/// `GLOBAL` outside the `torch.*` allowlist.
/// Returns [`AnamnesisError::Unsupported`] for a legacy `.pth` file.
///
/// # Memory
///
/// Reads the stream into an owned `Vec<u8>` of at most
/// `max_single_alloc_bytes + 1` bytes; peak heap is the artefact size plus the
/// per-tensor metadata.
pub fn parse_pth_from_reader_with_limits<R: Read>(
    reader: R,
    limits: &ParseLimits,
) -> crate::Result<ParsedPth> {
    let bytes = limits.read_to_vec_bounded(reader, "pth file")?;
    parse_pth_bytes_with_limits(bytes, limits)
}

/// Runs the pickle VM on `pkl_data` and extracts the per-tensor metadata
/// records (name, shape, dtype, storage key, storage offset, strides) that
/// both [`parse_pth`] and [`inspect_pth_from_reader`] need.
///
/// Sharing this step by construction keeps the security boundary identical
/// across the two entry points: every callable that reaches [`PickleVm`]
/// goes through the same [`is_allowed_global`] allowlist regardless of the
/// substrate the bytes came from. Any future tightening of the pickle
/// interpreter automatically applies to both paths.
///
/// `pkl_data` is borrowed; the returned [`TensorMeta`] records own their
/// strings and shape vectors so they outlive the input buffer (the
/// reader-generic entry point hands us an owned `Vec<u8>` that goes out of
/// scope before we return).
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] for any malformed opcode, stack
/// underflow, disallowed `GLOBAL`, or invalid `_rebuild_tensor_v2` argument
/// shape — same error surface as the pickle VM and tensor-reference
/// extractors return on their own.
fn interpret_pickle_to_meta(
    pkl_data: &[u8],
    limits: &ParseLimits,
) -> crate::Result<Vec<TensorMeta>> {
    let mut vm = PickleVm::new(pkl_data, limits);
    let root = vm.execute()?;

    let dict_pairs = extract_dict_pairs(&root, 0)?;
    let mut meta = Vec::new();
    for (key, value) in dict_pairs {
        let name = as_str(key)?;
        if let Some((_callable, args)) = unwrap_to_rebuild(value, 0) {
            let tref = parse_rebuild_args(name, args)?;
            meta.push(TensorMeta {
                name: tref.name,
                shape: tref.shape,
                dtype: tref.dtype,
                storage_key: tref.storage_key,
                storage_offset: tref.storage_offset,
                strides: tref.strides,
            });
        }
    }
    Ok(meta)
}

/// Builds a [`PthInspectInfo`] from the parsed per-tensor metadata.
///
/// Shared by [`ParsedPth::inspect`] (mmap-backed path) and
/// [`inspect_pth_from_reader`] (reader-generic path) so the two entry
/// points are guaranteed substrate-equivalent — every field of the
/// resulting `PthInspectInfo` is computed by the same code regardless of
/// which entry point produced the underlying [`TensorMeta`] records.
fn build_pth_inspect_info(meta: &[TensorMeta], big_endian: bool) -> PthInspectInfo {
    let mut total_bytes: u64 = 0;
    let mut dtypes: Vec<PthDtype> = Vec::new();
    for m in meta {
        // CAST: usize → u64, element counts and byte sizes fit in u64
        #[allow(clippy::as_conversions)]
        let n_elements: u64 = m
            .shape
            .iter()
            .copied()
            .fold(1u64, |acc, d| acc.saturating_mul(d as u64));
        // CAST: usize → u64, byte size of a single element is ≤ 8 → fits
        #[allow(clippy::as_conversions)]
        let byte_size = m.dtype.byte_size() as u64;
        total_bytes = total_bytes.saturating_add(n_elements.saturating_mul(byte_size));
        if !dtypes.contains(&m.dtype) {
            dtypes.push(m.dtype);
        }
    }
    PthInspectInfo {
        tensor_count: meta.len(),
        total_bytes,
        dtypes,
        big_endian,
    }
}

/// Inspects a `PyTorch` `.pth` archive from any `Read + Seek` source,
/// returning tensor-count / total-bytes / dtype / byte-order summary without
/// materialising any of the tensor-data files inside the archive.
///
/// This is the cheap first half of the **inspect-before-parse** policy gate:
/// the returned [`PthInspectInfo`] reports `total_bytes` and `tensor_count`,
/// which a host checks against its own budget before calling the authoritative
/// [`parse_pth_with_limits`] under a matched [`ParseLimits`].
///
/// This is the reader-generic core of `.pth` inspection: callers supply any
/// positional substrate (in-memory [`std::io::Cursor`], an `HTTP`-range-
/// backed adapter, a custom transport, …) and receive the same
/// [`PthInspectInfo`] as [`ParsedPth::inspect`]. The path-based [`parse_pth`]
/// remains the right entry point for callers that need the full
/// [`ParsedPth`] (with zero-copy tensor data via `memmap2::Mmap`);
/// `inspect_pth_from_reader` exists for the inspect-only case where
/// materialising the tensor-data files is wasteful — typically the use case
/// is browsing a remote shard before deciding whether to download it.
///
/// # Range-read access pattern
///
/// `.pth` is a ZIP archive: a sequence of local-file headers followed by a
/// central directory at the end of the file. The bytes
/// `inspect_pth_from_reader` actually touches on the substrate are:
///
/// 1. **`seek(SeekFrom::End(0))` once** to discover the total content
///    length and the EOCD scan window. An `HTTP`-range adapter that already
///    knows the `Content-Length` of the artefact answers this without a
///    fetch.
/// 2. **A short read of the local-file header magic at offset 0** —
///    4 bytes, used to separate *"legacy pre-1.6 raw pickle"* from
///    *"not a valid ZIP"* before the vendored reader walks the directory.
/// 3. **The end-of-central-directory (EOCD) scan** the vendored reader
///    performs — up to ~64 KiB read near the end of the file.
/// 4. **The central directory** — typically a few KiB, also near the end of
///    the file.
/// 5. **One bulk read of `data.pkl`** — the vendored reader resolves the
///    entry's local-file header (~30 B), then this function reads the entry
///    data (typically <100 KiB even on torchvision-class 300 MB models).
/// 6. **Optional one bulk read of `byteorder`** — usually 6 bytes
///    (`"little"`), 3 bytes (`"big"`), or absent.
///
/// The tensor-data files (`data/0`, `data/1`, …) inside the archive are
/// **never touched**. Total transfer for an `HTTP`-range adapter inspecting
/// a 300 MB torchvision `.pth`: well under 100 KiB instead of 300 MB.
///
/// Why `Read + Seek` (and not just `Read`): the ZIP format reads the central
/// directory at the end of the file, then seeks back to each local-file
/// header to read entry payloads. The vendored reader requires `Read + Seek`
/// for that reason, and `inspect_pth_from_reader` inherits the constraint
/// verbatim.
///
/// Anamnesis itself does not ship an `HTTP` transport; the network layer
/// belongs in downstream crates (e.g., `hf-fm`'s `HttpRangeReader`
/// adapter). This function defines the I/O contract such an adapter must
/// satisfy.
///
/// # Why metadata-only (no `parse_pth_from_reader`)
///
/// The local-file [`parse_pth(path)`](parse_pth) returns a [`ParsedPth`]
/// with `Cow::Borrowed` zero-copy slices into the underlying mmap. A
/// reader-based equivalent that preserved that contract would need the
/// reader to outlive every borrowed slice — workable for some
/// `Read + Seek` substrates but awkward for `HTTP`-range adapters where
/// each tensor read is a fresh fetch. Constraining the reader-based entry
/// point to **inspect-only** sidesteps that lifetime complexity for the
/// v0.11 use case (browsing tensor metadata before downloading) while
/// leaving room for a future `parse_pth_from_reader` if a streaming-data
/// use case develops.
///
/// # Performance
///
/// Unlike the GGUF reader-generic path, this function does **not** wrap the
/// caller-supplied reader in [`std::io::BufReader`]. The asymmetry is
/// deliberate: GGUF's parser issues many small `read_exact` calls (4–8 B
/// per typed primitive) which collapse one syscall per primitive on a raw
/// [`std::fs::File`] substrate; `inspect_pth_from_reader` only ever issues
/// bulk reads (the vendored reader's ZIP central-directory scan followed by
/// one `read_to_end` per named entry). Adding a `BufReader`
/// would only insert one extra memcpy per buffer-fill without saving any
/// syscalls. `HTTP`-range adapters that prefetch the EOCD + central
/// directory + `data.pkl` regions on first access amortise away every
/// round trip.
///
/// Per-file timings — best-of-5 release-mode median, `target-cpu=native`,
/// measured by
/// [`tests/bench_pth_inspect_adhoc.rs`](../tests/bench_pth_inspect_adhoc.rs)
/// against the matching `torch.load(weights_only=True)` baseline at
/// [`tests/fixtures/pth_reference/bench_python_inspect.py`](../tests/fixtures/pth_reference/bench_python_inspect.py)
/// (`PyTorch` `2.10.0+cu130`).
///
/// **Broad-sample baseline — 6 960-file `AlgZoo` sweep (the full
/// `algzoo_weights/` corpus from `candle-mi` v0.1.9, 22.6 MiB total,
/// median file size 2.5 KiB, 4 task families).** Per-file medians (µs):
///
/// | Substrate | min | p25 | median | p75 | mean | max |
/// |---|---:|---:|---:|---:|---:|---:|
/// | mmap path        | 117.4 | 122.3 | **124.0** | 128.4 | 127.7 | 415.1 |
/// | reader path      | 160.0 | 165.9 | **168.7** | 173.1 | 177.9 | 612.5 |
/// | `torch.load`     | 489.3 | 500.6 | **504.3** | 512.7 | 559.2 | 951.3 |
///
/// Median speedups across all 6 960 files: the mmap path is **4.07×
/// faster than `torch.load`**, the reader path is **2.99× faster than
/// `torch.load`**, and the reader path takes **1.36× the time** of the
/// mmap path (a structurally fixed ~45 µs gap, with p25=1.34× and
/// p75=1.39× — a tight distribution).
///
/// **In-tree baseline — 3-fixture spot check.** Same machine, same
/// methodology, the three `.pth` files checked into the repo:
///
/// | Fixture | `torch.load` median | mmap median | reader median |
/// |---|---:|---:|---:|
/// | `algzoo_rnn_small.pth` (2.0 KiB) | 532.7 µs | 134.4 µs | 220.1 µs |
/// | `algzoo_transformer_small.pth` (3.5 KiB) | 858.7 µs | 154.0 µs | 236.1 µs |
/// | `algzoo_rnn_blog.pth` (3.3 KiB) | 530.6 µs | 133.1 µs | 151.5 µs |
///
/// The reader speedup vs `torch.load` is a **lower bound**: scaling to a
/// torchvision-class 300 MB `.pth`, `torch.load`'s time grows linearly
/// with total `data/N` size while `inspect_pth_from_reader`'s time stays
/// bounded by `data.pkl` size (tens of KiB), so the ratio grows by
/// orders of magnitude on large models. See
/// [`docs/perf-experiments.md`](../docs/perf-experiments.md) Experiment 6
/// for the per-family breakdown, the `longest_cycle` outlier analysis,
/// and the full method.
///
/// # Errors
///
/// Returns [`AnamnesisError::Io`] if a `read` or `seek` on the supplied
/// reader fails.
///
/// Returns [`AnamnesisError::LimitExceeded`] if the `data.pkl` declared size
/// exceeds the 100 MiB cap (`MAX_PKL_SIZE`, defensive against adversarial
/// central directories), the `byteorder` entry declared size exceeds its
/// `MAX_BYTEORDER_SIZE` cap, or the declared entry count / central-directory
/// size exceeds the `ZIP_MAX_ENTRIES` cap or the caller's limits.
///
/// Returns [`AnamnesisError::DisallowedGlobal`] if the pickle references a
/// `GLOBAL` outside the `torch.*` allowlist.
///
/// Returns [`AnamnesisError::Parse`] if the file is shorter than 4 bytes,
/// the local-file header magic is not `PK\x03\x04`, the central directory is
/// malformed, the `data.pkl` entry is missing, the `byteorder` bytes are not
/// UTF-8 or not `"little"`/`"big"`, the pickle VM rejects the opcode stream, or
/// any `_rebuild_tensor_v2` call has malformed arguments.
///
/// Returns [`AnamnesisError::Unsupported`] for legacy (pre-`PyTorch` 1.6)
/// `.pth` files that begin with a raw pickle byte (`0x80` followed by a
/// protocol byte ≤ `0x05`) rather than the ZIP magic.
///
/// # Source context
///
/// Errors describe the **format-level problem**, not the source identity.
/// The function is reader-agnostic — the source could be a file, an
/// in-memory `Cursor`, or an `HTTP`-range adapter. Callers that have a
/// source name (filename, URL, etc.) should wrap the returned error with
/// that context. This matches anamnesis's existing convention
/// (`parse_safetensors_header_from_reader`, `inspect_npz_from_reader`, and
/// `inspect_gguf_from_reader` all return source-agnostic errors).
///
/// # Memory
///
/// Allocates only metadata structures: the materialised `data.pkl` bytes
/// (typically <100 KiB; capped at 100 MiB before allocation), the
/// pickle-VM stack and memo table (O(`n_tensors`)), and the per-tensor
/// `Vec<TensorMeta>` (O(`n_tensors`)). The materialised `data.pkl` buffer
/// is dropped before the function returns. No tensor-data files inside the
/// archive are read or allocated for. Peak heap is `O(pkl_size + n_tensors)`,
/// independent of the file's total size — a torchvision 300 MB `.pth`
/// inspects with ~150 KiB peak heap.
pub fn inspect_pth_from_reader<R: Read + Seek>(reader: R) -> crate::Result<PthInspectInfo> {
    // The inspect path is intentionally limit-free: it reports the totals a host
    // checks against its policy (the inspect-before-parse gate), then calls
    // `parse_pth_with_limits` for enforcement. Use the unbounded default.
    let (big_endian, meta) = read_pth_meta(reader, &ParseLimits::default())?;
    Ok(build_pth_inspect_info(&meta, big_endian))
}

/// Reads and interprets a `.pth` archive's `data.pkl`, returning the byte
/// order and the full per-tensor metadata.
///
/// Composes [`read_pth_archive_for_inspect`] (I/O) and
/// [`interpret_pickle_to_meta`] (pickle VM) so [`inspect_pth_from_reader`]
/// and [`parse_pth_front_matter_from_reader_with_limits`] share this step
/// by construction — the same shape as [`interpret_pickle_to_meta`]'s own
/// sharing rationale. `limits` bounds both the ZIP central-directory walk
/// and the `data.pkl` size cap; `inspect_pth_from_reader` passes
/// [`ParseLimits::default`] (identical to [`ParseLimits::unbounded`], so
/// this preserves that function's existing limit-free behaviour exactly).
///
/// # Errors
///
/// Returns [`AnamnesisError::Io`] / [`AnamnesisError::Parse`] /
/// [`AnamnesisError::LimitExceeded`] / [`AnamnesisError::Unsupported`] /
/// [`AnamnesisError::DisallowedGlobal`] under the same conditions
/// documented on [`inspect_pth_from_reader`].
fn read_pth_meta<R: Read + Seek>(
    reader: R,
    limits: &ParseLimits,
) -> crate::Result<(bool, Vec<TensorMeta>)> {
    let (big_endian, pkl_bytes) = read_pth_archive_for_inspect(reader, limits)?;
    let meta = interpret_pickle_to_meta(&pkl_bytes, limits)?;
    Ok((big_endian, meta))
}

/// I/O step of [`inspect_pth_from_reader`] and
/// [`parse_pth_front_matter_from_reader_with_limits`] (via [`read_pth_meta`]):
/// separates legacy-format detection, ZIP-archive open, byte-order
/// resolution, and `data.pkl` materialisation from the format-agnostic
/// pickle interpretation.
///
/// `limits` bounds the ZIP central-directory walk and the `data.pkl` size
/// cap (on top of the permanent `MAX_PKL_SIZE` / `ZIP_MAX_ENTRIES` floors,
/// which always apply regardless of `limits`).
///
/// Returns `(big_endian, pkl_bytes)`. Splitting the I/O step keeps
/// [`inspect_pth_from_reader`] readable as three short calls (read bytes,
/// interpret, summarise).
///
/// # Errors
///
/// Returns [`AnamnesisError::Io`] if reading or seeking fails.
///
/// Returns [`AnamnesisError::Parse`] / [`AnamnesisError::LimitExceeded`] /
/// [`AnamnesisError::Unsupported`] under the same conditions documented on
/// [`inspect_pth_from_reader`].
fn read_pth_archive_for_inspect<R: Read + Seek>(
    reader: R,
    limits: &ParseLimits,
) -> crate::Result<(bool, Vec<u8>)> {
    let mut src = crate::parse::zip::ReaderSource::new(reader)?;
    let total_len = src.total_len();
    if total_len < 4 {
        return Err(AnamnesisError::Parse {
            reason: "file too small to be a .pth archive".into(),
        });
    }
    // Probe the local-file header magic to keep the *"legacy pre-1.6 raw
    // pickle"* diagnostic distinct from the generic *"not a ZIP"* diagnostic —
    // same precedent as the mmap-backed `parse_pth`.
    let mut magic = [0u8; 4];
    src.read_at(0, &mut magic)?;
    // INDEX: `magic` is a fixed 4-byte array; [0]/[1] are always in bounds.
    #[allow(clippy::indexing_slicing)]
    if magic[0] == 0x80 && magic[1] <= 0x05 {
        return Err(AnamnesisError::Unsupported {
            format: "pth".into(),
            detail: "legacy .pth format (pre-PyTorch 1.6) is not supported; \
                     re-save with torch.save()"
                .into(),
        });
    }
    if magic != *b"PK\x03\x04" {
        return Err(AnamnesisError::Parse {
            reason: "file is not a ZIP archive (missing PK\\x03\\x04 magic)".into(),
        });
    }

    // Walk the central directory once over the vendored reader to locate
    // `data.pkl` and (optional) `byteorder` by suffix — same suffix-stripping
    // convention as `build_entry_index`, so both newer-style `archive/data.pkl`
    // and older-style `{model_name}/data.pkl` archives are accepted. Bounded
    // by the caller's `limits` (the permanent ZIP_MAX_ENTRIES floor always
    // applies inside the reader regardless).
    let entries = crate::parse::zip::read_central_directory(&mut src, limits)?;
    let mut pkl_entry: Option<&crate::parse::zip::ZipEntry> = None;
    let mut byteorder_entry: Option<&crate::parse::zip::ZipEntry> = None;
    for entry in &entries {
        let suffix = crate::parse::zip::strip_archive_prefix(&entry.name);
        if suffix == "data.pkl" && pkl_entry.is_none() {
            pkl_entry = Some(entry);
        } else if suffix == "byteorder" && byteorder_entry.is_none() {
            byteorder_entry = Some(entry);
        }
    }
    let pkl_entry = pkl_entry.ok_or_else(|| AnamnesisError::Parse {
        reason: "ZIP entry `data.pkl` not found".into(),
    })?;

    // Bounded by the caller's `limits`, layered on top of the permanent
    // MAX_PKL_SIZE cap enforced inside `enforce_pkl_size_cap` itself.
    enforce_pkl_size_cap(pkl_entry.uncompressed_size, "data.pkl", limits)?;

    let big_endian = match byteorder_entry {
        Some(entry) => {
            if entry.uncompressed_size > MAX_BYTEORDER_SIZE {
                return Err(AnamnesisError::LimitExceeded {
                    limit: "MAX_BYTEORDER_SIZE",
                    message: format!(
                        "ZIP entry `{}`: declared size {} bytes exceeds the \
                         {MAX_BYTEORDER_SIZE}-byte byteorder cap",
                        entry.name, entry.uncompressed_size
                    ),
                });
            }
            let buf = read_pth_entry_bytes(&mut src, entry)?;
            let text = std::str::from_utf8(&buf).map_err(|e| AnamnesisError::Parse {
                reason: format!("byteorder entry is not UTF-8: {e}"),
            })?;
            match text.trim() {
                "little" => false,
                "big" => true,
                other => {
                    return Err(AnamnesisError::Parse {
                        reason: format!(
                            "unknown byte order `{other}` (expected `little` or `big`)"
                        ),
                    });
                }
            }
        }
        None => false, // default: little-endian
    };

    let pkl_bytes = read_pth_entry_bytes(&mut src, pkl_entry)?;
    Ok((big_endian, pkl_bytes))
}

/// Full front matter of a parsed `.pth` file — every tensor's name, shape,
/// dtype, and byte length, plus the archive's byte order — read from any
/// `Read + Seek` source without touching the tensor-data files.
///
/// The full-detail counterpart to [`PthInspectInfo`]: where that type
/// reports aggregate statistics for a cheap inspect-before-parse policy
/// gate, `PthFrontMatter` carries the same per-tensor list that
/// [`ParsedPth::tensor_info`] exposes for the mmap-backed path. Produced by
/// [`parse_pth_front_matter_from_reader`] / `_with_limits`.
#[derive(Debug, Clone)]
#[non_exhaustive]
#[must_use]
pub struct PthFrontMatter {
    /// Per-tensor metadata (name, shape, dtype, byte length) in
    /// `state_dict` order.
    pub tensors: Vec<PthTensorInfo>,
    /// Whether the file uses big-endian storage.
    pub big_endian: bool,
}

impl PthFrontMatter {
    /// Reduces this front matter to the aggregate [`PthInspectInfo`]
    /// summary. No I/O.
    pub fn inspect(&self) -> PthInspectInfo {
        build_front_matter_inspect_info(&self.tensors, self.big_endian)
    }
}

/// Parses full `.pth` front matter from any `Read + Seek` source — the
/// complete per-tensor list, without materialising any tensor-data files.
///
/// The full-detail counterpart to [`inspect_pth_from_reader`]. Both are the
/// [`ParseLimits::default`] (unbounded) special case of their `_with_limits`
/// form, so the two are **limits-equivalent** — only the permanent `.pth`
/// caps (`MAX_PKL_SIZE`, `MAX_BYTEORDER_SIZE`, `MAX_PICKLE_WORKING_SET`,
/// `MAX_PICKLE_VM_DEPTH`, `ZIP_MAX_ENTRIES`) bound this call. What differs
/// is what survives the parse: the aggregate summary, or every tensor name
/// and shape handed to the caller.
///
/// **Prefer [`parse_pth_front_matter_from_reader_with_limits`] for
/// untrusted input.** Because this entry point returns every parsed tensor
/// name and shape instead of reducing them to counts, its exposure at a
/// given budget is strictly larger than [`inspect_pth_from_reader`]'s, and
/// an explicit [`ParseLimits`] is the only thing that bounds it below the
/// permanent caps. This mirrors [`parse_pth_from_reader`] /
/// [`parse_pth_from_reader_with_limits`].
///
/// # Errors
///
/// Returns [`AnamnesisError::Io`] / [`AnamnesisError::Parse`] /
/// [`AnamnesisError::LimitExceeded`] / [`AnamnesisError::Unsupported`] /
/// [`AnamnesisError::DisallowedGlobal`] under the same conditions
/// documented on [`inspect_pth_from_reader`].
///
/// # Memory
///
/// Same footprint class as [`inspect_pth_from_reader`]:
/// `O(pkl_size + n_tensors)`, independent of the file's total size. The
/// retained [`PthFrontMatter::tensors`] additionally clones each tensor's
/// name `String` and shape `Vec<usize>` (proportional to `n_tensors`, not
/// to file size) rather than discarding them into aggregate counters.
pub fn parse_pth_front_matter_from_reader<R: Read + Seek>(
    reader: R,
) -> crate::Result<PthFrontMatter> {
    parse_pth_front_matter_from_reader_with_limits(reader, &ParseLimits::default())
}

/// Parses full `.pth` front matter from any `Read + Seek` source under a
/// caller-supplied [`ParseLimits`] budget — the bounded, reader-generic
/// full-detail path.
///
/// # Errors
///
/// Returns the same error conditions as
/// [`parse_pth_front_matter_from_reader`], with `limits` additionally
/// bounding the ZIP central-directory walk and the `data.pkl` size cap.
///
/// # Memory
///
/// Same footprint as [`parse_pth_front_matter_from_reader`].
pub fn parse_pth_front_matter_from_reader_with_limits<R: Read + Seek>(
    reader: R,
    limits: &ParseLimits,
) -> crate::Result<PthFrontMatter> {
    let (big_endian, meta) = read_pth_meta(reader, limits)?;
    Ok(PthFrontMatter {
        tensors: build_pth_tensor_info(&meta),
        big_endian,
    })
}

/// Builds a [`PthInspectInfo`] from parsed front-matter tensor info.
///
/// Kept deliberately separate from [`build_pth_inspect_info`] (which
/// reduces from [`TensorMeta`] instead): [`ParsedPth`] keeps only the lean
/// `TensorMeta` on the struct and builds the heavier [`PthTensorInfo`]
/// (which clones every tensor name and shape) only on demand via
/// [`ParsedPth::tensor_info`]. Routing [`ParsedPth::inspect`] and
/// [`inspect_pth_from_reader`] through this reduction instead would force
/// that heavier allocation onto both, for a summary that doesn't need it.
fn build_front_matter_inspect_info(tensors: &[PthTensorInfo], big_endian: bool) -> PthInspectInfo {
    let mut total_bytes: u64 = 0;
    let mut dtypes: Vec<PthDtype> = Vec::new();
    for t in tensors {
        // CAST: usize → u64, byte lengths fit in u64
        #[allow(clippy::as_conversions)]
        let byte_len = t.byte_len as u64;
        total_bytes = total_bytes.saturating_add(byte_len);
        if !dtypes.contains(&t.dtype) {
            dtypes.push(t.dtype);
        }
    }
    PthInspectInfo {
        tensor_count: tensors.len(),
        total_bytes,
        dtypes,
        big_endian,
    }
}

/// Reads a `.pth` ZIP entry's contents into a `Vec`, inflating `DEFLATE`
/// entries on the fly (the codec stays in this consumer; the vendored container
/// reader returns the raw bytes).
///
/// The read is bounded to the entry's declared `uncompressed_size`, which the
/// caller has already capped (`MAX_PKL_SIZE` / `MAX_BYTEORDER_SIZE`). This is a
/// **zip-bomb guard**: an unbounded `read_to_end` on a `DEFLATE` decoder would
/// expand the entire compressed stream regardless of the declared size, so a
/// few-KiB entry could inflate to gigabytes. `Read::take` caps both the bytes
/// buffered and the inflation work; the `Vec` grows as bytes arrive, so a tiny
/// file that merely *declares* a 100 `MiB` size cannot drive a 100 `MiB` eager
/// allocation either. (A pathological `STORED` entry whose
/// `uncompressed_size < compressed_size` is truncated at the declared size; the
/// pickle VM then rejects the short stream — a clean error, never unsound.)
///
/// # Errors
///
/// Returns [`AnamnesisError::Unsupported`] if the entry uses a compression
/// method other than `STORED` or `DEFLATE`, [`AnamnesisError::Parse`] if the
/// local header is malformed, or [`AnamnesisError::Io`] if a read fails.
fn read_pth_entry_bytes<R: Read + Seek>(
    src: &mut crate::parse::zip::ReaderSource<R>,
    entry: &crate::parse::zip::ZipEntry,
) -> crate::Result<Vec<u8>> {
    // Already capped by the caller (`enforce_pkl_size_cap` / the byteorder cap).
    let limit = entry.uncompressed_size;
    let raw = src.entry_data_reader(entry)?;
    // TRAIT_OBJECT: STORED vs DEFLATE entry readers differ in concrete type; a
    // boxed `dyn Read` unifies them for the bounded full-entry read.
    let reader: Box<dyn Read + '_> = match entry.method {
        crate::parse::zip::Compression::Stored => Box::new(raw),
        crate::parse::zip::Compression::Deflate => Box::new(flate2::read::DeflateDecoder::new(raw)),
        crate::parse::zip::Compression::Unsupported(method) => {
            return Err(AnamnesisError::Unsupported {
                format: "pth".into(),
                detail: format!(
                    "ZIP entry `{}` uses unsupported compression method {method}",
                    entry.name
                ),
            });
        }
    };
    // Grows as bytes arrive, bounded by `take` — never an eager declared-size
    // allocation, never an unbounded inflation.
    let mut buf = Vec::new();
    reader
        .take(limit)
        .read_to_end(&mut buf)
        .map_err(AnamnesisError::Io)?;
    Ok(buf)
}

/// Builds an O(1) index of ZIP entry suffix → `(data_start, data_len)` in `raw`,
/// over the vendored central-directory reader ([`crate::parse::zip`]).
///
/// Only indexes STORED entries (uncompressed). The suffix is the part after
/// the archive prefix (e.g., `"data.pkl"`, `"data/0"`, `"byteorder"`).
///
/// `limits` bounds the container metadata fail-fast (entry count +
/// central-directory allocation) for the `parse_pth_with_limits` path; the
/// permanent caps inside the reader are the floor.
///
/// # Errors
///
/// Returns [`AnamnesisError::LimitExceeded`] if the declared entry count or
/// central-directory size exceeds `limits` or the `ZIP_MAX_ENTRIES` cap.
/// Returns [`AnamnesisError::Parse`] if the central directory is malformed, a
/// local-header data offset cannot be resolved, `data_start` or `size`
/// overflows `usize`, or an entry's byte range exceeds the file size.
fn build_entry_index(raw: &[u8], limits: &ParseLimits) -> crate::Result<EntryIndex> {
    let mut src = crate::parse::zip::SliceSource::new(raw);
    let entries = crate::parse::zip::read_central_directory(&mut src, limits)?;

    // Clamp the pre-allocation hint: a many-entries zip would otherwise drive
    // an eager `with_capacity` ~proportional to the file size. The Vec grows as
    // entries are pushed. Mirrors the `GGUF` parser's `PREALLOC_SOFT_CAP`.
    let mut index: Vec<(Box<str>, usize, usize)> =
        Vec::with_capacity(entries.len().min(PREALLOC_SOFT_CAP));

    for entry in &entries {
        // EXPLICIT: PyTorch ZIP archives use STORED (no compression)
        // exclusively. Compressed entries indicate a non-PyTorch ZIP or
        // manual re-compression — skip them rather than error, since they
        // are metadata entries (e.g., .format_version) that the parser
        // does not need. Tensor data entries are always STORED.
        if !entry.method.is_stored() {
            continue;
        }

        // `data_start` reads the entry's local header and cross-checks
        // `data_start + compressed_size <= raw.len()`, so the result is a
        // validated in-bounds offset.
        let data_start = crate::parse::zip::data_start(&mut src, entry)?;
        let data_start = usize::try_from(data_start).map_err(|_| AnamnesisError::Parse {
            reason: format!("ZIP entry `{}`: data_start overflows usize", entry.name),
        })?;
        let data_len =
            usize::try_from(entry.compressed_size).map_err(|_| AnamnesisError::Parse {
                reason: format!("ZIP entry `{}`: size overflows usize", entry.name),
            })?;

        // Strip the archive prefix to get the suffix key.
        // "archive/data.pkl" → "data.pkl", "my_model/data/0" → "data/0".
        let suffix = crate::parse::zip::strip_archive_prefix(&entry.name);
        if !suffix.is_empty() {
            // BORROW: `Box::from(&str)` copies the borrowed suffix into an
            // exact-sized owned key (no `String` capacity slack), which must
            // outlive the `entries` vector.
            index.push((Box::from(suffix.as_ref()), data_start, data_len));
        }
    }

    // Sort for binary-search lookup. On a duplicate suffix (pathological —
    // real `.pth` entry names are unique) keep the last-pushed, matching the
    // prior `HashMap` insert semantics: a stable sort leaves equal keys in push
    // order, so reversing puts the last-pushed first, `dedup_by` keeps that
    // first-of-run, and reversing again restores ascending order.
    index.sort_by(|a, b| a.0.cmp(&b.0));
    index.reverse();
    index.dedup_by(|a, b| a.0 == b.0);
    index.reverse();
    // Reclaim the `push`-growth capacity slack (a `Vec` over-allocates up to ~2×
    // while growing): the index is now immutable, so trim it to exact length —
    // this is what keeps resident bytes/entry near the theoretical floor.
    index.shrink_to_fit();

    Ok(EntryIndex { entries: index })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(
    clippy::panic,
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::as_conversions,
    clippy::wildcard_enum_match_arm
)]
mod tests {
    use std::io::Write;

    use super::*;

    /// Unbounded limits for VM/helper tests that predate `ParseLimits`. A
    /// `static` (not a temporary) so `&UNBOUNDED_LIMITS` outlives the borrowing
    /// `PickleVm`. Equivalent to the old no-limits behaviour.
    static UNBOUNDED_LIMITS: ParseLimits = ParseLimits::unbounded();

    // -- PthDtype ------------------------------------------------------------

    #[test]
    fn dtype_byte_sizes() {
        assert_eq!(PthDtype::Bool.byte_size(), 1);
        assert_eq!(PthDtype::U8.byte_size(), 1);
        assert_eq!(PthDtype::I8.byte_size(), 1);
        assert_eq!(PthDtype::F16.byte_size(), 2);
        assert_eq!(PthDtype::BF16.byte_size(), 2);
        assert_eq!(PthDtype::I16.byte_size(), 2);
        assert_eq!(PthDtype::F32.byte_size(), 4);
        assert_eq!(PthDtype::I32.byte_size(), 4);
        assert_eq!(PthDtype::F64.byte_size(), 8);
        assert_eq!(PthDtype::I64.byte_size(), 8);
    }

    #[test]
    fn dtype_display() {
        assert_eq!(PthDtype::F32.to_string(), "F32");
        assert_eq!(PthDtype::BF16.to_string(), "BF16");
        assert_eq!(PthDtype::Bool.to_string(), "BOOL");
    }

    #[test]
    fn dtype_to_dtype_roundtrip() {
        assert_eq!(PthDtype::F32.to_dtype().unwrap(), Dtype::F32);
        assert_eq!(PthDtype::F16.to_dtype().unwrap(), Dtype::F16);
        assert_eq!(PthDtype::BF16.to_dtype().unwrap(), Dtype::BF16);
        assert_eq!(PthDtype::I64.to_dtype().unwrap(), Dtype::I64);
        assert_eq!(PthDtype::Bool.to_dtype().unwrap(), Dtype::Bool);
    }

    #[test]
    fn dtype_from_storage_class() {
        assert_eq!(
            PthDtype::from_storage_class("torch", "FloatStorage").unwrap(),
            PthDtype::F32
        );
        assert_eq!(
            PthDtype::from_storage_class("torch", "BFloat16Storage").unwrap(),
            PthDtype::BF16
        );
        assert!(PthDtype::from_storage_class("torch", "UnknownStorage").is_err());
        assert!(PthDtype::from_storage_class("numpy", "FloatStorage").is_err());
    }

    // -- LONG1 conversion ----------------------------------------------------

    #[test]
    fn long1_zero() {
        assert_eq!(long1_to_i64(&[]).unwrap(), 0);
    }

    #[test]
    fn long1_positive() {
        // 255 = 0xFF as unsigned, but in two's complement with sign bit clear
        // we need two bytes: 0xFF, 0x00
        assert_eq!(long1_to_i64(&[0xFF, 0x00]).unwrap(), 255);
        assert_eq!(long1_to_i64(&[0x01]).unwrap(), 1);
        assert_eq!(long1_to_i64(&[0x80, 0x00]).unwrap(), 128);
    }

    #[test]
    fn long1_negative() {
        // -1 in two's complement = 0xFF
        assert_eq!(long1_to_i64(&[0xFF]).unwrap(), -1);
        // -128 = 0x80
        assert_eq!(long1_to_i64(&[0x80]).unwrap(), -128);
    }

    #[test]
    fn long1_too_large() {
        let big = vec![0x01; 9]; // 9 bytes > 8
        assert!(long1_to_i64(&big).is_err());
    }

    // -- Contiguous strides --------------------------------------------------

    #[test]
    fn contiguous_strides_2d() {
        assert_eq!(contiguous_strides(&[3, 4]), vec![4, 1]);
        assert_eq!(contiguous_strides(&[16, 10]), vec![10, 1]);
    }

    #[test]
    fn contiguous_strides_1d() {
        assert_eq!(contiguous_strides(&[5]), vec![1]);
    }

    #[test]
    fn contiguous_strides_scalar() {
        assert_eq!(contiguous_strides(&[]), Vec::<usize>::new());
    }

    #[test]
    fn is_contiguous_true() {
        assert!(is_contiguous(&[3, 4], &[4, 1]));
        assert!(is_contiguous(&[5], &[1]));
    }

    #[test]
    fn is_contiguous_transposed() {
        // A transposed 3×4 matrix would have strides [1, 3]
        assert!(!is_contiguous(&[3, 4], &[1, 3]));
    }

    // -- Pickle VM -----------------------------------------------------------

    #[test]
    fn vm_simple_int() {
        // PROTO 2, BININT1 42, STOP
        let pkl = &[0x80, 0x02, b'K', 42, b'.'];
        let mut vm = PickleVm::new(pkl, &UNBOUNDED_LIMITS);
        let result = vm.execute().unwrap();
        assert!(matches!(result, PickleValue::Int(42)));
    }

    #[test]
    fn vm_string() {
        // PROTO 2, SHORT_BINUNICODE "hi" (len=2), STOP
        let pkl = &[0x80, 0x02, 0x8c, 0x02, b'h', b'i', b'.'];
        let mut vm = PickleVm::new(pkl, &UNBOUNDED_LIMITS);
        let result = vm.execute().unwrap();
        assert!(matches!(result, PickleValue::String(ref s) if s == "hi"));
    }

    #[test]
    fn vm_tuple2() {
        // PROTO 2, BININT1 1, BININT1 2, TUPLE2, STOP
        let pkl = &[0x80, 0x02, b'K', 1, b'K', 2, 0x86, b'.'];
        let mut vm = PickleVm::new(pkl, &UNBOUNDED_LIMITS);
        let result = vm.execute().unwrap();
        if let PickleValue::Tuple(items) = result {
            assert_eq!(items.len(), 2);
        } else {
            panic!("expected Tuple");
        }
    }

    #[test]
    fn vm_empty_dict() {
        // PROTO 2, EMPTY_DICT, STOP
        let pkl = &[0x80, 0x02, b'}', b'.'];
        let mut vm = PickleVm::new(pkl, &UNBOUNDED_LIMITS);
        let result = vm.execute().unwrap();
        assert!(matches!(result, PickleValue::Dict(ref d) if d.is_empty()));
    }

    #[test]
    fn vm_dict_with_setitem() {
        // PROTO 2, EMPTY_DICT, SHORT_BINUNICODE "k" (len=1), BININT1 7, SETITEM, STOP
        let pkl = &[0x80, 0x02, b'}', 0x8c, 1, b'k', b'K', 7, b's', b'.'];
        let mut vm = PickleVm::new(pkl, &UNBOUNDED_LIMITS);
        let result = vm.execute().unwrap();
        if let PickleValue::Dict(pairs) = result {
            assert_eq!(pairs.len(), 1);
        } else {
            panic!("expected Dict");
        }
    }

    #[test]
    fn vm_memo_roundtrip() {
        // PROTO 2, BININT1 99, BINPUT 0, POP (not implemented — use different approach)
        // Instead: BININT1 99, BINPUT 0, BININT1 0, BINGET 0, TUPLE2, STOP
        // This stores 99 in memo[0], pushes 0, gets memo[0] (=99), makes tuple (0, 99)
        let pkl = &[
            0x80, 0x02, // PROTO 2
            b'K', 99, // BININT1 99
            b'q', 0, // BINPUT 0
            b'K', 0, // BININT1 0
            b'h', 0,    // BINGET 0
            0x86, // TUPLE2
            b'.', // STOP
        ];
        let mut vm = PickleVm::new(pkl, &UNBOUNDED_LIMITS);
        let result = vm.execute().unwrap();
        if let PickleValue::Tuple(items) = result {
            assert_eq!(items.len(), 2);
            assert!(matches!(&items[0], PickleValue::Int(0)));
            assert!(matches!(&items[1], PickleValue::Int(99)));
        } else {
            panic!("expected Tuple");
        }
    }

    #[test]
    fn vm_rejects_disallowed_global() {
        // PROTO 2, GLOBAL "os\nsystem\n", STOP
        let pkl = b"\x80\x02cos\nsystem\n.";
        let mut vm = PickleVm::new(pkl, &UNBOUNDED_LIMITS);
        let err = vm.execute().unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("disallowed pickle global"), "got: {msg}");
        assert!(msg.contains("os.system"), "got: {msg}");
    }

    #[test]
    fn vm_rejects_unknown_opcode() {
        // PROTO 2, unknown opcode 0xFF, STOP
        let pkl = &[0x80, 0x02, 0xFF, b'.'];
        let mut vm = PickleVm::new(pkl, &UNBOUNDED_LIMITS);
        let err = vm.execute().unwrap_err();
        assert!(err.to_string().contains("unsupported pickle opcode 0xff"));
    }

    #[test]
    fn vm_allows_torch_global() {
        // PROTO 2, GLOBAL "torch._utils\n_rebuild_tensor_v2\n", STOP
        let pkl = b"\x80\x02ctorch._utils\n_rebuild_tensor_v2\n.";
        let mut vm = PickleVm::new(pkl, &UNBOUNDED_LIMITS);
        let result = vm.execute().unwrap();
        assert!(matches!(
            result,
            PickleValue::Global { ref module, ref name }
            if module == "torch._utils" && name == "_rebuild_tensor_v2"
        ));
    }

    // -- Legacy detection ----------------------------------------------------

    #[test]
    fn reject_legacy_pth() {
        // Fake legacy pickle: starts with PROTO 2 but no ZIP magic
        let data = vec![0x80, 0x02, 0x00, 0x00, 0x00];
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), &data).unwrap();
        let err = parse_pth(tmp.path()).unwrap_err();
        assert!(err.to_string().contains("legacy .pth format"));
    }

    #[test]
    fn reject_non_zip() {
        // Random bytes, not ZIP, not pickle
        let data = vec![0x00, 0x01, 0x02, 0x03, 0x04];
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), &data).unwrap();
        let err = parse_pth(tmp.path()).unwrap_err();
        assert!(err.to_string().contains("not a ZIP archive"));
    }

    #[test]
    fn reject_too_small() {
        let data = vec![0x50, 0x4B]; // Just "PK" — too short
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), &data).unwrap();
        let err = parse_pth(tmp.path()).unwrap_err();
        assert!(err.to_string().contains("too small"));
    }

    // -- Gap tests (review findings G1–G31) ----------------------------------

    // G1: FRAME opcode (0x95) — skips 8-byte frame length, no-op
    #[test]
    fn vm_frame_opcode() {
        // PROTO 4, FRAME (8 bytes length), BININT1 42, STOP
        let pkl: &[u8] = &[
            0x80, 0x04, // PROTO 4
            0x95, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // FRAME(2)
            b'K', 42,   // BININT1
            b'.', // STOP
        ];
        let mut vm = PickleVm::new(pkl, &UNBOUNDED_LIMITS);
        let result = vm.execute().unwrap();
        assert!(matches!(result, PickleValue::Int(42)));
    }

    // G2: NONE opcode
    #[test]
    fn vm_none() {
        let pkl: &[u8] = &[0x80, 0x02, b'N', b'.'];
        let mut vm = PickleVm::new(pkl, &UNBOUNDED_LIMITS);
        let result = vm.execute().unwrap();
        assert!(matches!(result, PickleValue::None));
    }

    // G3: NEWTRUE / NEWFALSE opcodes
    #[test]
    fn vm_newtrue_newfalse() {
        // PROTO 2, NEWTRUE, NEWFALSE, TUPLE2, STOP
        let pkl: &[u8] = &[0x80, 0x02, 0x88, 0x89, 0x86, b'.'];
        let mut vm = PickleVm::new(pkl, &UNBOUNDED_LIMITS);
        let result = vm.execute().unwrap();
        if let PickleValue::Tuple(items) = result {
            assert!(matches!(items[0], PickleValue::Bool(true)));
            assert!(matches!(items[1], PickleValue::Bool(false)));
        } else {
            panic!("expected Tuple");
        }
    }

    // G4: BININT (4-byte signed)
    #[test]
    fn vm_binint() {
        // PROTO 2, BININT 0x01020304 (little-endian = 67305985), STOP
        let pkl: &[u8] = &[0x80, 0x02, b'J', 0x04, 0x03, 0x02, 0x01, b'.'];
        let mut vm = PickleVm::new(pkl, &UNBOUNDED_LIMITS);
        let result = vm.execute().unwrap();
        assert!(matches!(result, PickleValue::Int(0x0102_0304)));
    }

    // G4b: BININT negative
    #[test]
    fn vm_binint_negative() {
        // PROTO 2, BININT -1 (0xFFFFFFFF LE), STOP
        let pkl: &[u8] = &[0x80, 0x02, b'J', 0xFF, 0xFF, 0xFF, 0xFF, b'.'];
        let mut vm = PickleVm::new(pkl, &UNBOUNDED_LIMITS);
        let result = vm.execute().unwrap();
        assert!(matches!(result, PickleValue::Int(-1)));
    }

    // G5: BININT2 (2-byte unsigned)
    #[test]
    fn vm_binint2() {
        // PROTO 2, BININT2 0x0100 (= 256 LE), STOP
        let pkl: &[u8] = &[0x80, 0x02, b'M', 0x00, 0x01, b'.'];
        let mut vm = PickleVm::new(pkl, &UNBOUNDED_LIMITS);
        let result = vm.execute().unwrap();
        assert!(matches!(result, PickleValue::Int(256)));
    }

    // G6: BINUNICODE (4-byte length)
    #[test]
    fn vm_binunicode() {
        // PROTO 2, BINUNICODE "abc" (length=3 LE), STOP
        let pkl: &[u8] = &[
            0x80, 0x02, b'X', 0x03, 0x00, 0x00, 0x00, b'a', b'b', b'c', b'.',
        ];
        let mut vm = PickleVm::new(pkl, &UNBOUNDED_LIMITS);
        let result = vm.execute().unwrap();
        if let PickleValue::String(s) = result {
            assert_eq!(s, "abc");
        } else {
            panic!("expected String, got {result:?}");
        }
    }

    /// Phase 6.8 Step 1: a tightened `ParseLimits` rejects a `data.pkl` entry
    /// size and a pickle payload that the unbounded default accepts.
    #[test]
    fn pth_respects_parse_limits() {
        // `data.pkl` entry-size threading via the shared helper: at the cap
        // passes, one over fails.
        let tight = ParseLimits::default().with_max_single_alloc(16);
        assert!(enforce_pkl_size_cap(16, "data.pkl", &tight).is_ok());
        let err = enforce_pkl_size_cap(17, "data.pkl", &tight).unwrap_err();
        assert!(
            matches!(err, AnamnesisError::LimitExceeded { limit, .. } if limit == "max_single_alloc_bytes"),
            "expected pkl-size limit error, got: {err}"
        );

        // Pickle-payload threading via the VM: BINUNICODE "abc" (3 bytes).
        let pkl: &[u8] = &[
            0x80, 0x02, b'X', 0x03, 0x00, 0x00, 0x00, b'a', b'b', b'c', b'.',
        ];
        // Default (unbounded) executes.
        assert!(PickleVm::new(pkl, &UNBOUNDED_LIMITS).execute().is_ok());
        // A 1-byte single-allocation ceiling rejects the 3-byte payload.
        let tight = ParseLimits::default().with_max_single_alloc(1);
        let err = PickleVm::new(pkl, &tight).execute().unwrap_err();
        assert!(
            matches!(err, AnamnesisError::LimitExceeded { limit, .. } if limit == "max_single_alloc_bytes"),
            "expected pickle-payload limit error, got: {err}"
        );
    }

    /// Phase 6.8 Step 2 + Phase 6.11: the aggregate `max_total_bytes` rejects a
    /// pickle whose individual payloads each pass `max_single_alloc` but whose
    /// cumulative working-set heap crosses the budget. Since Phase 6.11 the
    /// charge is the full per-value working set — the enum slot
    /// (`size_of::<PickleValue>()`) plus the owned `String`/`Bytes` payload —
    /// not the raw payload alone, so two 3-byte strings cost
    /// `2 × (size_of::<PickleValue>() + 3)`.
    #[test]
    fn pth_aggregate_budget() {
        // PROTO 2, BINUNICODE "abc" (3), BINUNICODE "def" (3), STOP.
        let pkl: &[u8] = &[
            0x80, 0x02, b'X', 0x03, 0x00, 0x00, 0x00, b'a', b'b', b'c', b'X', 0x03, 0x00, 0x00,
            0x00, b'd', b'e', b'f', b'.',
        ];
        // CAST: usize → u64, lossless widening
        let value_cost = std::mem::size_of::<PickleValue>() as u64 + 3;
        // A budget one byte short of both values rejects on the second push.
        let tight = ParseLimits::default().with_max_total_bytes(2 * value_cost - 1);
        let err = PickleVm::new(pkl, &tight).execute().unwrap_err();
        assert!(
            matches!(err, AnamnesisError::LimitExceeded { limit, .. } if limit == "max_total_bytes"),
            "expected aggregate limit error, got: {err}"
        );
        // A budget covering both values exactly fits.
        let fits = ParseLimits::default().with_max_total_bytes(2 * value_cost);
        assert!(PickleVm::new(pkl, &fits).execute().is_ok());
    }

    // G7: SHORT_BINSTRING
    #[test]
    fn vm_short_binstring() {
        // PROTO 2, SHORT_BINSTRING "xy" (length=2), STOP
        let pkl: &[u8] = &[0x80, 0x02, b'U', 0x02, b'x', b'y', b'.'];
        let mut vm = PickleVm::new(pkl, &UNBOUNDED_LIMITS);
        let result = vm.execute().unwrap();
        if let PickleValue::String(s) = result {
            assert_eq!(s, "xy");
        } else {
            panic!("expected String, got {result:?}");
        }
    }

    // G8: SHORT_BINBYTES
    #[test]
    fn vm_short_binbytes() {
        // PROTO 2, SHORT_BINBYTES [0xDE, 0xAD] (length=2), STOP
        let pkl: &[u8] = &[0x80, 0x02, b'C', 0x02, 0xDE, 0xAD, b'.'];
        let mut vm = PickleVm::new(pkl, &UNBOUNDED_LIMITS);
        let result = vm.execute().unwrap();
        if let PickleValue::Bytes(b) = result {
            assert_eq!(b, vec![0xDE, 0xAD]);
        } else {
            panic!("expected Bytes, got {result:?}");
        }
    }

    // G9: EMPTY_LIST and EMPTY_TUPLE
    #[test]
    fn vm_empty_list() {
        let pkl: &[u8] = &[0x80, 0x02, b']', b'.'];
        let mut vm = PickleVm::new(pkl, &UNBOUNDED_LIMITS);
        let result = vm.execute().unwrap();
        if let PickleValue::List(items) = result {
            assert!(items.is_empty());
        } else {
            panic!("expected List, got {result:?}");
        }
    }

    #[test]
    fn vm_empty_tuple() {
        let pkl: &[u8] = &[0x80, 0x02, b')', b'.'];
        let mut vm = PickleVm::new(pkl, &UNBOUNDED_LIMITS);
        let result = vm.execute().unwrap();
        if let PickleValue::Tuple(items) = result {
            assert!(items.is_empty());
        } else {
            panic!("expected Tuple, got {result:?}");
        }
    }

    // G10: TUPLE1 and TUPLE3
    #[test]
    fn vm_tuple1() {
        // PROTO 2, BININT1 7, TUPLE1, STOP
        let pkl: &[u8] = &[0x80, 0x02, b'K', 7, 0x85, b'.'];
        let mut vm = PickleVm::new(pkl, &UNBOUNDED_LIMITS);
        let result = vm.execute().unwrap();
        if let PickleValue::Tuple(items) = result {
            assert_eq!(items.len(), 1);
            assert!(matches!(items[0], PickleValue::Int(7)));
        } else {
            panic!("expected Tuple, got {result:?}");
        }
    }

    #[test]
    fn vm_tuple3() {
        // PROTO 2, BININT1 1, BININT1 2, BININT1 3, TUPLE3, STOP
        let pkl: &[u8] = &[0x80, 0x02, b'K', 1, b'K', 2, b'K', 3, 0x87, b'.'];
        let mut vm = PickleVm::new(pkl, &UNBOUNDED_LIMITS);
        let result = vm.execute().unwrap();
        if let PickleValue::Tuple(items) = result {
            assert_eq!(items.len(), 3);
            assert!(matches!(items[0], PickleValue::Int(1)));
            assert!(matches!(items[1], PickleValue::Int(2)));
            assert!(matches!(items[2], PickleValue::Int(3)));
        } else {
            panic!("expected Tuple, got {result:?}");
        }
    }

    // G11: SETITEMS
    #[test]
    fn vm_setitems() {
        // PROTO 2, EMPTY_DICT, MARK, SHORT_BINUNICODE "a", BININT1 1,
        //          SHORT_BINUNICODE "b", BININT1 2, SETITEMS, STOP
        let pkl: &[u8] = &[
            0x80, 0x02, b'}', b'(', 0x8C, 0x01, b'a', b'K', 1, 0x8C, 0x01, b'b', b'K', 2, b'u',
            b'.',
        ];
        let mut vm = PickleVm::new(pkl, &UNBOUNDED_LIMITS);
        let result = vm.execute().unwrap();
        if let PickleValue::Dict(pairs) = result {
            assert_eq!(pairs.len(), 2);
        } else {
            panic!("expected Dict, got {result:?}");
        }
    }

    // G12: APPEND and APPENDS
    #[test]
    fn vm_append() {
        // PROTO 2, EMPTY_LIST, BININT1 42, APPEND, STOP
        let pkl: &[u8] = &[0x80, 0x02, b']', b'K', 42, b'a', b'.'];
        let mut vm = PickleVm::new(pkl, &UNBOUNDED_LIMITS);
        let result = vm.execute().unwrap();
        if let PickleValue::List(items) = result {
            assert_eq!(items.len(), 1);
            assert!(matches!(items[0], PickleValue::Int(42)));
        } else {
            panic!("expected List, got {result:?}");
        }
    }

    #[test]
    fn vm_appends() {
        // PROTO 2, EMPTY_LIST, MARK, BININT1 1, BININT1 2, APPENDS, STOP
        let pkl: &[u8] = &[0x80, 0x02, b']', b'(', b'K', 1, b'K', 2, b'e', b'.'];
        let mut vm = PickleVm::new(pkl, &UNBOUNDED_LIMITS);
        let result = vm.execute().unwrap();
        if let PickleValue::List(items) = result {
            assert_eq!(items.len(), 2);
            assert!(matches!(items[0], PickleValue::Int(1)));
            assert!(matches!(items[1], PickleValue::Int(2)));
        } else {
            panic!("expected List, got {result:?}");
        }
    }

    // G17: LONG_BINPUT / LONG_BINGET (4-byte memo keys)
    #[test]
    fn vm_long_memo_roundtrip() {
        // PROTO 2, BININT1 77, LONG_BINPUT key=1, BININT1 0,
        //          LONG_BINGET key=1, TUPLE2, STOP
        let pkl: &[u8] = &[
            0x80, 0x02, b'K', 77, b'r', 0x01, 0x00, 0x00, 0x00, // LONG_BINPUT(1)
            b'K', 0, b'j', 0x01, 0x00, 0x00, 0x00, // LONG_BINGET(1)
            0x86, b'.', // TUPLE2, STOP
        ];
        let mut vm = PickleVm::new(pkl, &UNBOUNDED_LIMITS);
        let result = vm.execute().unwrap();
        if let PickleValue::Tuple(items) = result {
            assert_eq!(items.len(), 2);
            assert!(matches!(items[0], PickleValue::Int(0)));
            assert!(matches!(items[1], PickleValue::Int(77)));
        } else {
            panic!("expected Tuple, got {result:?}");
        }
    }

    // G18: MEMOIZE (proto 4+)
    #[test]
    fn vm_memoize() {
        // PROTO 4, BININT1 99, MEMOIZE, BININT1 0, BINGET key=0, TUPLE2, STOP
        let pkl: &[u8] = &[
            0x80, 0x04, b'K', 99, 0x94, // MEMOIZE (auto-assigns key 0)
            b'K', 0, b'h', 0x00, // BINGET(0)
            0x86, b'.', // TUPLE2, STOP
        ];
        let mut vm = PickleVm::new(pkl, &UNBOUNDED_LIMITS);
        let result = vm.execute().unwrap();
        if let PickleValue::Tuple(items) = result {
            assert_eq!(items.len(), 2);
            assert!(matches!(items[0], PickleValue::Int(0)));
            assert!(matches!(items[1], PickleValue::Int(99)));
        } else {
            panic!("expected Tuple, got {result:?}");
        }
    }

    // G20: LONG1 with exactly 8 bytes (negative value, two's complement)
    #[test]
    fn long1_8byte_negative() {
        // -1 in 8-byte two's complement: all 0xFF
        let result = long1_to_i64(&[0xFF; 8]).unwrap();
        assert_eq!(result, -1);
    }

    #[test]
    fn long1_8byte_max_positive() {
        // i64::MAX = 0x7FFF_FFFF_FFFF_FFFF LE = [0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0x7F]
        let result = long1_to_i64(&[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x7F]).unwrap();
        assert_eq!(result, i64::MAX);
    }

    #[test]
    fn long1_8byte_min_negative() {
        // i64::MIN = 0x8000_0000_0000_0000 LE = [0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x80]
        let result = long1_to_i64(&[0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80]).unwrap();
        assert_eq!(result, i64::MIN);
    }

    // G25: MAX_PICKLE_NESTING enforcement
    #[test]
    fn unwrap_to_rebuild_rejects_deep_nesting() {
        // Pass depth = 33 (> MAX_PICKLE_NESTING) to trigger the guard on entry.
        // The guard fires immediately without actual recursion — it checks
        // depth before inspecting the value.
        let leaf = PickleValue::Int(0);
        let result = unwrap_to_rebuild(&leaf, MAX_PICKLE_NESTING + 1);
        assert!(result.is_none(), "should reject nesting beyond limit");
    }

    // -- Phase 6.11: pickle-VM working-set governance ------------------------

    /// Vector (c): an over-deep `BINPERSID` (`Q`) chain is rejected at
    /// construction by the permanent [`MAX_PICKLE_VM_DEPTH`] floor — under
    /// *unbounded* caller limits — so a value whose recursive `Drop` would
    /// overflow the stack never forms. The input is a few hundred bytes; no
    /// large allocation occurs.
    #[test]
    fn vm_rejects_overdeep_persid_chain() {
        // PROTO 2, BININT1 0, then one BINPERSID per byte (each +1 depth).
        let mut pkl = vec![0x80, 0x02, b'K', 0x00];
        pkl.resize(pkl.len() + MAX_PICKLE_VM_DEPTH as usize + 8, b'Q');
        pkl.push(b'.'); // STOP (unreached — the cap trips mid-stream)
        let err = PickleVm::new(&pkl, &UNBOUNDED_LIMITS)
            .execute()
            .unwrap_err();
        assert!(
            matches!(err, AnamnesisError::LimitExceeded { limit, .. } if limit == "MAX_PICKLE_VM_DEPTH"),
            "expected depth-cap rejection, got: {err}"
        );
    }

    /// Vector (a): a flood of `NONE` opcodes is now charged per value, so a
    /// tight `max_total_bytes` rejects it after a handful of pushes — the
    /// value-stack heap is budgeted, not free.
    #[test]
    fn vm_charges_none_flood() {
        let mut pkl = vec![0x80, 0x02]; // PROTO 2
        pkl.resize(pkl.len() + 64, b'N'); // 64 × NONE
        pkl.push(b'.');
        // Each NONE charges size_of::<PickleValue>() (dozens of bytes); a
        // 200-byte aggregate budget cannot hold 64 of them.
        let tight = ParseLimits::default().with_max_total_bytes(200);
        let err = PickleVm::new(&pkl, &tight).execute().unwrap_err();
        assert!(
            matches!(err, AnamnesisError::LimitExceeded { limit, .. } if limit == "max_total_bytes"),
            "expected working-set budget rejection, got: {err}"
        );
    }

    /// Vector (b): memoised-subtree replay is charged the clone's deep heap, so
    /// repeated `BINGET` of a list is bounded by `max_total_bytes` (the
    /// construction fits; the clones do not).
    #[test]
    fn vm_charges_memo_replay() {
        // PROTO 2, EMPTY_LIST, MARK, BININT1 1/2/3, APPENDS, BINPUT 0,
        // then BINGET 0 × 16, STOP.
        let mut pkl = vec![
            0x80, 0x02, b']', b'(', b'K', 1, b'K', 2, b'K', 3, b'e', b'q', 0x00,
        ];
        for _ in 0..16 {
            pkl.push(b'h'); // BINGET
            pkl.push(0x00); // memo key 0
        }
        pkl.push(b'.');
        let tight = ParseLimits::default().with_max_total_bytes(1500);
        let err = PickleVm::new(&pkl, &tight).execute().unwrap_err();
        assert!(
            matches!(err, AnamnesisError::LimitExceeded { limit, .. } if limit == "max_total_bytes"),
            "expected memo-replay budget rejection, got: {err}"
        );
    }

    /// The permanent [`MAX_PICKLE_WORKING_SET`] floor is enforced even under
    /// unbounded caller limits — exercised on the accounting helper directly so
    /// no real multi-hundred-MB allocation is needed.
    #[test]
    fn vm_permanent_working_set_floor() {
        let mut vm = PickleVm::new(&[], &UNBOUNDED_LIMITS);
        assert!(vm.charge(MAX_PICKLE_WORKING_SET, "test").is_ok());
        let err = vm.charge(1, "test").unwrap_err();
        assert!(
            matches!(err, AnamnesisError::LimitExceeded { limit, .. } if limit == "MAX_PICKLE_WORKING_SET"),
            "expected permanent working-set floor rejection, got: {err}"
        );
    }

    // G28: copy_to_contiguous with transposed 2D tensor
    #[test]
    fn copy_to_contiguous_transposed_2x3() {
        // 2×3 F32 tensor stored with strides [1, 2] (transposed from [3, 1])
        // Logical matrix: [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        // Storage layout with strides [1, 2]: column-major
        //   storage[0]=0.0, storage[1]=3.0, storage[2]=1.0,
        //   storage[3]=4.0, storage[4]=2.0, storage[5]=5.0
        let values: [f32; 6] = [0.0, 3.0, 1.0, 4.0, 2.0, 5.0];
        let mut storage = Vec::new();
        for v in &values {
            storage.extend_from_slice(&v.to_le_bytes());
        }

        let result = copy_to_contiguous(&storage, 0, &[2, 3], &[1, 2], 4).unwrap();

        // Expected row-major output: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        let expected: [f32; 6] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let mut expected_bytes = Vec::new();
        for v in &expected {
            expected_bytes.extend_from_slice(&v.to_le_bytes());
        }
        assert_eq!(result, expected_bytes);
    }

    // G7 (partial): BINSTRING (b'T') — 4-byte signed length
    #[test]
    fn vm_binstring() {
        // PROTO 2, BINSTRING "ab" (length=2 as i32 LE), STOP
        let pkl: &[u8] = &[0x80, 0x02, b'T', 0x02, 0x00, 0x00, 0x00, b'a', b'b', b'.'];
        let mut vm = PickleVm::new(pkl, &UNBOUNDED_LIMITS);
        let result = vm.execute().unwrap();
        if let PickleValue::String(s) = result {
            assert_eq!(s, "ab");
        } else {
            panic!("expected String, got {result:?}");
        }
    }

    // G8 (partial): BINBYTES (b'B') — 4-byte length
    #[test]
    fn vm_binbytes() {
        // PROTO 3, BINBYTES [0xCA, 0xFE] (length=2 as u32 LE), STOP
        let pkl: &[u8] = &[0x80, 0x03, b'B', 0x02, 0x00, 0x00, 0x00, 0xCA, 0xFE, b'.'];
        let mut vm = PickleVm::new(pkl, &UNBOUNDED_LIMITS);
        let result = vm.execute().unwrap();
        if let PickleValue::Bytes(b) = result {
            assert_eq!(b, vec![0xCA, 0xFE]);
        } else {
            panic!("expected Bytes, got {result:?}");
        }
    }

    // G13: STACK_GLOBAL (protocol 4+)
    #[test]
    fn vm_stack_global() {
        // PROTO 4, SHORT_BINUNICODE "torch._utils", SHORT_BINUNICODE
        // "_rebuild_tensor_v2", STACK_GLOBAL, STOP
        let pkl: &[u8] = &[
            0x80, 0x04, 0x8C, 0x0C, // SHORT_BINUNICODE len=12
            b't', b'o', b'r', b'c', b'h', b'.', b'_', b'u', b't', b'i', b'l', b's', 0x8C,
            0x12, // SHORT_BINUNICODE len=18
            b'_', b'r', b'e', b'b', b'u', b'i', b'l', b'd', b'_', b't', b'e', b'n', b's', b'o',
            b'r', b'_', b'v', b'2', 0x93, // STACK_GLOBAL
            b'.',
        ];
        let mut vm = PickleVm::new(pkl, &UNBOUNDED_LIMITS);
        let result = vm.execute().unwrap();
        assert!(matches!(
            result,
            PickleValue::Global { ref module, ref name }
            if module == "torch._utils" && name == "_rebuild_tensor_v2"
        ));
    }

    // G14: REDUCE (b'R') — pop args + callable, create Reduced
    #[test]
    fn vm_reduce() {
        // PROTO 2, GLOBAL "torch._utils\n_rebuild_tensor_v2\n",
        //          EMPTY_TUPLE, REDUCE, STOP
        let pkl = b"\x80\x02ctorch._utils\n_rebuild_tensor_v2\n)R.";
        let mut vm = PickleVm::new(pkl, &UNBOUNDED_LIMITS);
        let result = vm.execute().unwrap();
        assert!(
            matches!(result, PickleValue::Reduced { .. }),
            "expected Reduced, got {result:?}"
        );
    }

    // G14b: NEWOBJ (0x81) — same semantics as REDUCE
    #[test]
    fn vm_newobj() {
        // PROTO 2, GLOBAL "torch._utils\n_rebuild_tensor_v2\n",
        //          EMPTY_TUPLE, NEWOBJ, STOP
        let pkl: &[u8] = b"\x80\x02ctorch._utils\n_rebuild_tensor_v2\n)\x81.";
        let mut vm = PickleVm::new(pkl, &UNBOUNDED_LIMITS);
        let result = vm.execute().unwrap();
        assert!(
            matches!(result, PickleValue::Reduced { .. }),
            "expected Reduced, got {result:?}"
        );
    }

    // G15: BUILD (b'b') — pop state, pop object, create Built
    #[test]
    fn vm_build() {
        // PROTO 2, BININT1 1 (object), BININT1 2 (state), BUILD, STOP
        let pkl: &[u8] = &[0x80, 0x02, b'K', 1, b'K', 2, b'b', b'.'];
        let mut vm = PickleVm::new(pkl, &UNBOUNDED_LIMITS);
        let result = vm.execute().unwrap();
        if let PickleValue::Built { obj, state } = result {
            assert!(matches!(*obj, PickleValue::Int(1)));
            assert!(matches!(*state, PickleValue::Int(2)));
        } else {
            panic!("expected Built, got {result:?}");
        }
    }

    // G16: BINPERSID (b'Q') — pop id, create PersistentId
    #[test]
    fn vm_binpersid() {
        // PROTO 2, BININT1 42, BINPERSID, STOP
        let pkl: &[u8] = &[0x80, 0x02, b'K', 42, b'Q', b'.'];
        let mut vm = PickleVm::new(pkl, &UNBOUNDED_LIMITS);
        let result = vm.execute().unwrap();
        if let PickleValue::PersistentId(inner) = result {
            assert!(matches!(*inner, PickleValue::Int(42)));
        } else {
            panic!("expected PersistentId, got {result:?}");
        }
    }

    // G19: MEMOIZE overflow at u32::MAX
    #[test]
    fn vm_memoize_overflow() {
        // PROTO 4, BININT1 0, MEMOIZE (will get key 0), STOP
        // — but we pre-set next_memo_id to u32::MAX so the +1 overflows
        let pkl: &[u8] = &[0x80, 0x04, b'K', 0, 0x94, b'.'];
        let mut vm = PickleVm::new(pkl, &UNBOUNDED_LIMITS);
        vm.next_memo_id = u32::MAX;
        let err = vm.execute().unwrap_err();
        assert!(
            err.to_string().contains("memo table overflow"),
            "got: {err}"
        );
    }

    // G21: Zero-element tensor (shape [0, 4])
    // This test exercises copy_to_contiguous directly with a zero-sized
    // dimension. In the non-contiguous path, dim=0 triggers the checked_sub(1)
    // error in the max_elem_offset calculation. (In the contiguous path,
    // n_elements=0 produces a zero-byte slice before copy_to_contiguous
    // is ever called.)
    #[test]
    fn copy_to_contiguous_zero_elements_errors() {
        let storage = vec![0u8; 16];
        let result = copy_to_contiguous(&storage, 0, &[0, 4], &[4, 1], 4);
        assert!(
            result.is_err(),
            "zero-dim in shape should error in max_elem_offset"
        );
    }

    // G22: Element-count overflow in copy_to_contiguous
    #[test]
    fn copy_to_contiguous_element_count_overflow() {
        let storage = vec![0u8; 8];
        let result = copy_to_contiguous(&storage, 0, &[usize::MAX, 2], &[2, 1], 1);
        assert!(result.is_err());
        assert!(
            result.unwrap_err().to_string().contains("overflow"),
            "expected overflow error"
        );
    }

    // G24: storage_offset overflow in copy_to_contiguous
    #[test]
    fn copy_to_contiguous_offset_overflow() {
        let storage = vec![0u8; 8];
        // offset near usize::MAX, shape [1] → offset + 1 byte wraps
        let result = copy_to_contiguous(&storage, usize::MAX, &[1], &[1], 1);
        assert!(result.is_err());
    }

    // G23: storage_offset exactly at storage boundary (tightest valid access)
    #[test]
    fn copy_to_contiguous_offset_at_boundary() {
        // 4 bytes of storage, 1 element of 4 bytes at offset 0 → end = 4 = storage.len()
        let storage = vec![0x01, 0x02, 0x03, 0x04];
        let result = copy_to_contiguous(&storage, 0, &[1], &[1], 4).unwrap();
        assert_eq!(result, vec![0x01, 0x02, 0x03, 0x04]);
    }

    // G23b: one byte past the boundary should fail
    #[test]
    fn copy_to_contiguous_one_past_boundary() {
        // 4 bytes of storage, 1 element of 4 bytes at offset 1 → end = 5 > 4
        let storage = vec![0x01, 0x02, 0x03, 0x04];
        let result = copy_to_contiguous(&storage, 1, &[1], &[1], 4);
        assert!(result.is_err());
    }

    // G26: shape.len() != strides.len() → is_contiguous returns false
    #[test]
    fn is_contiguous_mismatched_dims() {
        // 3D shape with 2D strides — should not be contiguous
        assert!(!is_contiguous(&[2, 3, 4], &[12, 4]));
    }

    // NI1: copy_to_contiguous rejects mismatched shape/strides lengths
    #[test]
    fn copy_to_contiguous_mismatched_ndim() {
        let storage = vec![0u8; 96]; // 24 f32 values
        let result = copy_to_contiguous(&storage, 0, &[2, 3, 4], &[12, 4], 4);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("ndim"),
            "expected ndim mismatch error, got: {msg}"
        );
    }

    // G27: Zero-stride dimension (broadcast)
    #[test]
    fn copy_to_contiguous_zero_stride_broadcast() {
        // shape [2, 3], strides [0, 1], elem_size=1
        // Row 0 and Row 1 both read from the same 3 bytes (broadcast)
        let storage: Vec<u8> = vec![10, 20, 30];
        let result = copy_to_contiguous(&storage, 0, &[2, 3], &[0, 1], 1).unwrap();
        // Both rows should be [10, 20, 30]
        assert_eq!(result, vec![10, 20, 30, 10, 20, 30]);
    }

    // G29: ZIP archive with no data.pkl entry
    #[test]
    fn reject_zip_missing_data_pkl() {
        // Build a ZIP with a random entry but no data.pkl
        let tmp = tempfile::NamedTempFile::new().unwrap();
        {
            let file = std::fs::File::create(tmp.path()).unwrap();
            let mut zip = zip::ZipWriter::new(file);
            let options = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            zip.start_file("archive/not_a_pkl.txt", options).unwrap();
            zip.write_all(b"hello").unwrap();
            zip.finish().unwrap();
        }
        let err = parse_pth(tmp.path()).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("data.pkl") && msg.contains("not found"),
            "expected 'data.pkl not found', got: {msg}"
        );
    }

    // G30: data.pkl stored as DEFLATE (compressed) — silently skipped
    #[test]
    fn reject_compressed_data_pkl() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        {
            let file = std::fs::File::create(tmp.path()).unwrap();
            let mut zip = zip::ZipWriter::new(file);
            let options = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated);
            zip.start_file("archive/data.pkl", options).unwrap();
            zip.write_all(b"\x80\x02}.").unwrap(); // valid pickle: empty dict
            zip.finish().unwrap();
        }
        let err = parse_pth(tmp.path()).unwrap_err();
        let msg = err.to_string();
        // data.pkl is compressed → skipped by build_entry_index → "not found"
        assert!(
            msg.contains("data.pkl") && msg.contains("not found"),
            "expected 'data.pkl not found' (compressed entries are skipped), got: {msg}"
        );
    }

    // G31: ZIP entry with zero data length (valid edge case)
    #[test]
    fn zip_zero_length_entry_accepted() {
        // Build a ZIP with data.pkl (valid pickle) and an empty data/0 entry.
        // parse_pth should succeed — the empty storage entry is valid for
        // a model with no tensors (the pickle just needs to be parseable).
        let tmp = tempfile::NamedTempFile::new().unwrap();
        {
            let file = std::fs::File::create(tmp.path()).unwrap();
            let mut zip = zip::ZipWriter::new(file);
            let opts = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);

            // data.pkl: PROTO 2, EMPTY_DICT, STOP → valid empty state_dict
            zip.start_file("archive/data.pkl", opts).unwrap();
            zip.write_all(b"\x80\x02}.").unwrap();

            // Empty storage entry (0 bytes)
            zip.start_file("archive/data/0", opts).unwrap();
            // write nothing — zero-length entry

            zip.finish().unwrap();
        }
        // Should parse successfully — empty dict, no tensors
        let parsed = parse_pth(tmp.path()).unwrap();
        assert!(
            parsed.tensors().unwrap().is_empty(),
            "empty state_dict should produce no tensors"
        );
    }

    // T4 (post-review): tensors() contiguous-path overflow guards
    // The contiguous branch of tensors() (lines ~260-280) uses the same
    // try_fold + checked_mul + checked_add pattern as copy_to_contiguous.
    // Testing it directly requires crafting a pickle with metadata claiming
    // an astronomically large shape for a tiny storage — impractical without
    // exposing ParsedPth internals. The overflow arithmetic is structurally
    // identical and verified through copy_to_contiguous_element_count_overflow
    // and copy_to_contiguous_offset_overflow.

    // T5 (post-review): extract_dict_pairs nesting limit
    #[test]
    fn extract_dict_pairs_rejects_deep_nesting() {
        let dict = PickleValue::Dict(Vec::new());
        let result = extract_dict_pairs(&dict, MAX_PICKLE_NESTING + 1);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("nesting limit exceeded"),
            "expected nesting limit error"
        );
    }

    // -----------------------------------------------------------------
    // Reader-generic API — substrate-equivalence tests (Phase 4.10)
    // -----------------------------------------------------------------

    /// Asserts every field of two `PthInspectInfo` values is equal.
    /// Substrate equivalence means the path-based and reader-generic entry
    /// points must be indistinguishable in their output.
    fn assert_pth_inspect_eq(path_info: &PthInspectInfo, reader_info: &PthInspectInfo) {
        assert_eq!(path_info.tensor_count, reader_info.tensor_count);
        assert_eq!(path_info.total_bytes, reader_info.total_bytes);
        assert_eq!(path_info.dtypes, reader_info.dtypes);
        assert_eq!(path_info.big_endian, reader_info.big_endian);
    }

    /// Builds a minimal synthetic `.pth` ZIP archive whose `data.pkl` is a
    /// valid empty pickle (`PROTO 2`, `EMPTY_DICT`, `STOP`). Returns the
    /// archive bytes. Used to exercise the reader-generic path without
    /// depending on external fixtures.
    fn build_minimal_empty_pth() -> Vec<u8> {
        let mut buf: Vec<u8> = Vec::new();
        {
            // INDEX: `&mut buf` is a writable byte sink; `zip::ZipWriter`
            // requires `Write + Seek`, and `std::io::Cursor<&mut Vec<u8>>`
            // satisfies both.
            let cursor = std::io::Cursor::new(&mut buf);
            let mut zip = zip::ZipWriter::new(cursor);
            let opts = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            zip.start_file("archive/data.pkl", opts).unwrap();
            // PROTO 2, EMPTY_DICT, STOP — valid empty `state_dict`.
            zip.write_all(b"\x80\x02}.").unwrap();
            zip.finish().unwrap();
        }
        buf
    }

    /// `inspect_pth_from_reader` over an in-memory `Cursor` returns the
    /// same `PthInspectInfo` as `parse_pth(path).inspect()` over the same
    /// archive on disk. Locks the contract that the reader-generic and
    /// path-based APIs are substrate-equivalent — the substrate (file vs.
    /// cursor) cannot change the metadata. This is what downstream
    /// `HTTP`-range adapters rely on.
    #[test]
    fn inspect_from_reader_matches_path_empty_dict() {
        let bytes = build_minimal_empty_pth();
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), &bytes).unwrap();

        let path_info = parse_pth(tmp.path()).unwrap().inspect();
        let reader_info = inspect_pth_from_reader(std::io::Cursor::new(&bytes)).unwrap();

        assert_pth_inspect_eq(&path_info, &reader_info);

        // Spot-check the absolute values so we're not just comparing two
        // equal-but-wrong outputs.
        assert_eq!(reader_info.tensor_count, 0);
        assert_eq!(reader_info.total_bytes, 0);
        assert!(reader_info.dtypes.is_empty());
        assert!(!reader_info.big_endian);
    }

    /// `inspect_pth_from_reader` honours the `byteorder` archive entry
    /// when present — the value is propagated into `PthInspectInfo`
    /// identically to the path-based parser.
    #[test]
    fn inspect_from_reader_honours_byteorder_entry() {
        let mut buf: Vec<u8> = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut buf);
            let mut zip = zip::ZipWriter::new(cursor);
            let opts = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            zip.start_file("archive/data.pkl", opts).unwrap();
            zip.write_all(b"\x80\x02}.").unwrap();
            zip.start_file("archive/byteorder", opts).unwrap();
            zip.write_all(b"little").unwrap();
            zip.finish().unwrap();
        }
        let info = inspect_pth_from_reader(std::io::Cursor::new(&buf)).unwrap();
        assert!(!info.big_endian);
        assert_eq!(info.tensor_count, 0);
    }

    /// Legacy (pre-`PyTorch` 1.6) raw-pickle files must surface as
    /// `AnamnesisError::Unsupported` rather than a generic parse error —
    /// same diagnostic split as the path-based parser.
    #[test]
    fn inspect_from_reader_rejects_legacy_format() {
        let data: &[u8] = &[0x80, 0x02, 0x00, 0x00, 0x00];
        let err = inspect_pth_from_reader(std::io::Cursor::new(data)).unwrap_err();
        match err {
            AnamnesisError::Unsupported { format, detail } => {
                assert_eq!(format, "pth");
                assert!(detail.contains("legacy"));
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    /// Non-ZIP, non-legacy bytes produce `AnamnesisError::Parse` — the
    /// magic-byte check fires before the vendored central-directory reader runs.
    #[test]
    fn inspect_from_reader_rejects_wrong_magic() {
        let data: &[u8] = &[0x00, 0x01, 0x02, 0x03, 0x04];
        let err = inspect_pth_from_reader(std::io::Cursor::new(data)).unwrap_err();
        assert!(matches!(err, AnamnesisError::Parse { .. }));
        assert!(err.to_string().contains("not a ZIP archive"));
    }

    /// A file shorter than the 4-byte magic is rejected with a clear
    /// "too small" message rather than panicking on the magic read.
    #[test]
    fn inspect_from_reader_rejects_too_small_file() {
        let data: &[u8] = &[0x50, 0x4B]; // Just `PK` — 2 bytes
        let err = inspect_pth_from_reader(std::io::Cursor::new(data)).unwrap_err();
        assert!(matches!(err, AnamnesisError::Parse { .. }));
        assert!(err.to_string().contains("too small"));
    }

    /// A ZIP archive without a `data.pkl` entry surfaces the same
    /// `data.pkl not found` diagnostic as the path-based parser.
    #[test]
    fn inspect_from_reader_rejects_missing_data_pkl() {
        let mut buf: Vec<u8> = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut buf);
            let mut zip = zip::ZipWriter::new(cursor);
            let opts = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            zip.start_file("archive/not_a_pkl.txt", opts).unwrap();
            zip.write_all(b"hello").unwrap();
            zip.finish().unwrap();
        }
        let err = inspect_pth_from_reader(std::io::Cursor::new(&buf)).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("data.pkl") && msg.contains("not found"),
            "expected `data.pkl not found`, got: {msg}"
        );
    }

    /// A `DEFLATE`-compressed `data.pkl` still parses through the reader path:
    /// the zip-bomb guard in `read_pth_entry_bytes` bounds the inflation to the
    /// declared `uncompressed_size`, which for an honest entry equals the real
    /// size, so the full pickle is recovered (`take` must not truncate it).
    #[test]
    fn inspect_from_reader_accepts_deflate_data_pkl() {
        let mut buf: Vec<u8> = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut buf);
            let mut zip = zip::ZipWriter::new(cursor);
            let opts = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated);
            zip.start_file("archive/data.pkl", opts).unwrap();
            // PROTO 2, EMPTY_DICT, STOP — a valid empty state_dict.
            zip.write_all(&[0x80, 0x02, b'}', b'.']).unwrap();
            zip.finish().unwrap();
        }
        let info = inspect_pth_from_reader(std::io::Cursor::new(&buf)).unwrap();
        assert_eq!(info.tensor_count, 0);
    }

    /// The reader-generic path strips the archive prefix the same way as
    /// `build_entry_index`, so a `.pth` whose entries use a non-`archive/`
    /// prefix (older `PyTorch` saves, e.g., `mymodel/data.pkl`) inspects
    /// identically.
    #[test]
    fn inspect_from_reader_accepts_older_prefix() {
        let mut buf: Vec<u8> = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut buf);
            let mut zip = zip::ZipWriter::new(cursor);
            let opts = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            // Older-style prefix: `{model_name}/data.pkl` instead of
            // `archive/data.pkl`.
            zip.start_file("mymodel/data.pkl", opts).unwrap();
            zip.write_all(b"\x80\x02}.").unwrap();
            zip.finish().unwrap();
        }
        let info = inspect_pth_from_reader(std::io::Cursor::new(&buf)).unwrap();
        assert_eq!(info.tensor_count, 0);
    }

    /// A `byteorder` entry whose declared central-directory size exceeds
    /// the 64-byte cap is rejected before `read_to_end` allocates — the
    /// cap is a defensive boundary, not a post-read trim.
    #[test]
    fn inspect_from_reader_rejects_oversized_byteorder() {
        let mut buf: Vec<u8> = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut buf);
            let mut zip = zip::ZipWriter::new(cursor);
            let opts = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            zip.start_file("archive/data.pkl", opts).unwrap();
            zip.write_all(b"\x80\x02}.").unwrap();
            // A `byteorder` of >64 bytes — the central directory will
            // honestly report this size, so the cap fires.
            zip.start_file("archive/byteorder", opts).unwrap();
            // 80 ASCII zeros: way above the 64-byte ceiling.
            zip.write_all(&[b'0'; 80]).unwrap();
            zip.finish().unwrap();
        }
        let err = inspect_pth_from_reader(std::io::Cursor::new(&buf)).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("byteorder") && (msg.contains("exceeds") || msg.contains("cap")),
            "expected byteorder cap violation, got: {msg}"
        );
    }

    // -----------------------------------------------------------------
    // Reader-generic full front matter — substrate-equivalence (Phase 7.5)
    // -----------------------------------------------------------------

    /// `parse_pth_front_matter_from_reader` over an in-memory `Cursor`
    /// returns tensors matching `parse_pth(path).tensor_info()` over the
    /// same archive on disk, on the empty-`state_dict` fixture — the
    /// front-matter counterpart of `inspect_from_reader_matches_path_empty_dict`.
    #[test]
    fn front_matter_from_reader_matches_path_empty_dict() {
        let bytes = build_minimal_empty_pth();
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), &bytes).unwrap();

        let path_tensors = parse_pth(tmp.path()).unwrap().tensor_info();
        let front = parse_pth_front_matter_from_reader(std::io::Cursor::new(&bytes)).unwrap();

        assert_eq!(path_tensors.len(), front.tensors.len());
        assert!(front.tensors.is_empty(), "empty state_dict → no tensors");
        assert!(!front.big_endian);
    }

    /// `parse_pth_front_matter_from_reader` honours the `byteorder` archive
    /// entry when present, propagating it into `PthFrontMatter::big_endian`
    /// identically to `inspect_pth_from_reader`.
    #[test]
    fn front_matter_from_reader_honours_byteorder_entry() {
        let mut buf: Vec<u8> = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut buf);
            let mut zip = zip::ZipWriter::new(cursor);
            let opts = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            zip.start_file("archive/data.pkl", opts).unwrap();
            zip.write_all(b"\x80\x02}.").unwrap();
            zip.start_file("archive/byteorder", opts).unwrap();
            zip.write_all(b"big").unwrap();
            zip.finish().unwrap();
        }
        let front = parse_pth_front_matter_from_reader(std::io::Cursor::new(&buf)).unwrap();
        assert!(front.big_endian);
        assert!(front.tensors.is_empty());
    }

    /// `parse_pth_front_matter_from_reader` propagates the same parse
    /// errors as `inspect_pth_from_reader` — both delegate to the same
    /// `read_pth_meta` core, so neither should swallow or re-classify a
    /// rejection the other surfaces.
    #[test]
    fn front_matter_from_reader_propagates_parse_errors() {
        // Legacy pre-1.6 raw-pickle format.
        let legacy: &[u8] = &[0x80, 0x02, 0x00, 0x00, 0x00];
        let err = parse_pth_front_matter_from_reader(std::io::Cursor::new(legacy)).unwrap_err();
        match err {
            AnamnesisError::Unsupported { format, detail } => {
                assert_eq!(format, "pth");
                assert!(detail.contains("legacy"));
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }

        // Wrong magic (not a ZIP archive).
        let bad_magic: &[u8] = &[0x00, 0x01, 0x02, 0x03, 0x04];
        let err = parse_pth_front_matter_from_reader(std::io::Cursor::new(bad_magic)).unwrap_err();
        assert!(matches!(err, AnamnesisError::Parse { .. }));
        assert!(err.to_string().contains("not a ZIP archive"));

        // Missing `data.pkl` entry.
        let mut buf: Vec<u8> = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut buf);
            let mut zip = zip::ZipWriter::new(cursor);
            let opts = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            zip.start_file("archive/not_a_pkl.txt", opts).unwrap();
            zip.write_all(b"hello").unwrap();
            zip.finish().unwrap();
        }
        let err = parse_pth_front_matter_from_reader(std::io::Cursor::new(&buf)).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("data.pkl") && msg.contains("not found"),
            "expected `data.pkl not found`, got: {msg}"
        );
    }

    /// `PthFrontMatter::inspect()` reduces to the same `PthInspectInfo` as
    /// `inspect_pth_from_reader` on identical bytes — ties the two
    /// reader-generic entry points together despite their separate
    /// reduction helpers (`build_front_matter_inspect_info` vs
    /// `build_pth_inspect_info`).
    #[test]
    fn front_matter_inspect_matches_inspect_pth_from_reader() {
        let mut buf: Vec<u8> = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut buf);
            let mut zip = zip::ZipWriter::new(cursor);
            let opts = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            zip.start_file("archive/data.pkl", opts).unwrap();
            zip.write_all(b"\x80\x02}.").unwrap();
            zip.start_file("archive/byteorder", opts).unwrap();
            zip.write_all(b"little").unwrap();
            zip.finish().unwrap();
        }

        let front = parse_pth_front_matter_from_reader(std::io::Cursor::new(&buf)).unwrap();
        let summary_via_front = front.inspect();
        let summary_direct = inspect_pth_from_reader(std::io::Cursor::new(&buf)).unwrap();

        assert_pth_inspect_eq(&summary_via_front, &summary_direct);
    }

    /// A `DEFLATE`-compressed `data.pkl` still parses through the
    /// front-matter path: the zip-bomb guard in `read_pth_entry_bytes`
    /// bounds the inflation to the declared `uncompressed_size`, same as
    /// `inspect_pth_from_reader`.
    #[test]
    fn front_matter_from_reader_accepts_deflate_data_pkl() {
        let mut buf: Vec<u8> = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut buf);
            let mut zip = zip::ZipWriter::new(cursor);
            let opts = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated);
            zip.start_file("archive/data.pkl", opts).unwrap();
            zip.write_all(&[0x80, 0x02, b'}', b'.']).unwrap();
            zip.finish().unwrap();
        }
        let front = parse_pth_front_matter_from_reader(std::io::Cursor::new(&buf)).unwrap();
        assert!(front.tensors.is_empty());
    }

    /// A shape whose leading dimensions overflow `u64` when multiplied but
    /// whose trailing dimension is zero has **zero** elements — the zero
    /// wins regardless of how large the earlier dimensions are. Regression
    /// test for a divergence found in review: `build_pth_tensor_info` used
    /// to short-circuit to `usize::MAX` on the first overflowing dimension
    /// (via `try_fold` + `checked_mul`), never seeing the trailing zero,
    /// while `build_pth_inspect_info` folded every dimension with
    /// `saturating_mul` and correctly landed on zero. That meant
    /// `PthFrontMatter::inspect().total_bytes` and
    /// `inspect_pth_from_reader(...).total_bytes` could disagree by
    /// approximately `u64::MAX` on the same adversarial file — exactly the
    /// two entry points this phase's `front_matter_inspect_matches_inspect_pth_from_reader`
    /// test is meant to tie together, just on a shape that test didn't
    /// exercise. `build_pth_tensor_info` now uses the same
    /// never-short-circuiting fold as `build_pth_inspect_info`.
    #[test]
    fn front_matter_total_bytes_matches_inspect_on_overflowing_shape() {
        let meta = vec![TensorMeta {
            name: "huge".to_owned(),
            shape: vec![1usize << 33, 1usize << 33, 0],
            dtype: PthDtype::F32,
            storage_key: "0".to_owned(),
            storage_offset: 0,
            strides: vec![0, 0, 0],
        }];

        let via_inspect = build_pth_inspect_info(&meta, false);
        let tensors = build_pth_tensor_info(&meta);
        let via_front_matter = build_front_matter_inspect_info(&tensors, false);

        assert_eq!(
            tensors[0].byte_len, 0,
            "a trailing zero dimension must zero the byte length, not saturate to usize::MAX"
        );
        assert_eq!(via_inspect.total_bytes, 0);
        assert_eq!(
            via_inspect.total_bytes, via_front_matter.total_bytes,
            "inspect_pth_from_reader and PthFrontMatter::inspect() must agree on total_bytes \
             for the same tensor metadata"
        );
    }

    // -- DoS hardening (Phase 6.6) -------------------------------------------

    /// Builds a minimal `.pth` ZIP whose single `archive/data.pkl` entry holds
    /// `pkl` verbatim (STORED, so the central directory reports its true size).
    fn build_pth_with_pkl(pkl: &[u8]) -> Vec<u8> {
        let mut buf: Vec<u8> = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut buf);
            let mut zip = zip::ZipWriter::new(cursor);
            let opts = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            zip.start_file("archive/data.pkl", opts).unwrap();
            zip.write_all(pkl).unwrap();
            zip.finish().unwrap();
        }
        buf
    }

    /// `enforce_pkl_size_cap` accepts a declared size exactly at the cap and
    /// rejects one byte over it (Phase 6.6 Step 1). The bound is `>`, not `>=`,
    /// and both entry points route through this one helper.
    #[test]
    fn pkl_size_cap_boundary() {
        assert!(enforce_pkl_size_cap(MAX_PKL_SIZE, "data.pkl", &UNBOUNDED_LIMITS).is_ok());
        let err =
            enforce_pkl_size_cap(MAX_PKL_SIZE + 1, "data.pkl", &UNBOUNDED_LIMITS).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("data.pkl") && msg.contains("cap"),
            "expected pkl-size cap error, got: {msg}"
        );
    }

    /// An oversized `BINBYTES` payload length is rejected by the pickle VM
    /// before the clone (Phase 6.6 Step 4), exercised end-to-end through the
    /// `parse_pth` mmap path with a 7-byte crafted pickle. Mirrors candle
    /// #3533's "tiny malicious input → expect `Err`" shape.
    #[test]
    fn pickle_payload_cap_rejects_binbytes_mmap() {
        let bytes = build_pth_with_pkl(&[0x80, 0x02, b'B', 0xFF, 0xFF, 0xFF, 0xFF]);
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), &bytes).unwrap();
        let err = parse_pth(tmp.path()).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("BINBYTES") && msg.contains("cap"),
            "expected BINBYTES payload cap error, got: {msg}"
        );
    }

    /// Same guard via the reader path with an oversized `BINUNICODE` length.
    #[test]
    fn pickle_payload_cap_rejects_binunicode_reader() {
        let bytes = build_pth_with_pkl(&[0x80, 0x02, b'X', 0xFF, 0xFF, 0xFF, 0xFF]);
        let err = inspect_pth_from_reader(std::io::Cursor::new(&bytes)).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("BINUNICODE") && msg.contains("cap"),
            "expected BINUNICODE payload cap error, got: {msg}"
        );
    }

    /// Faithful large-file `PoC` for the mmap-path `MAX_PKL_SIZE` guard (Phase
    /// 6.6 Step 1). `build_entry_index` validates the declared entry size
    /// against the file length, so the cap is reachable only for a genuinely
    /// over-cap `data.pkl` — which means writing a >100 MiB archive. `#[ignore]`d
    /// to keep default `cargo test` fast, matching the repo's heavy-fixture
    /// convention (`tests/peak_heap_*.rs`). Run with `--ignored`.
    #[test]
    #[ignore = "writes a >100 MiB temp archive; run with --ignored"]
    fn pkl_size_cap_rejects_oversized_mmap_poc() {
        let cap = usize::try_from(MAX_PKL_SIZE).unwrap();
        let mut payload = vec![0u8; cap + 1];
        // A valid pickle prefix so the failure is unambiguously the size cap,
        // not a VM parse error (the cap fires before the VM runs).
        payload[0] = 0x80;
        payload[1] = 0x02;
        let bytes = build_pth_with_pkl(&payload);
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), &bytes).unwrap();
        let err = parse_pth(tmp.path()).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("data.pkl") && msg.contains("cap"),
            "expected pkl-size cap error, got: {msg}"
        );
    }
}
