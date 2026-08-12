// SPDX-License-Identifier: MIT OR Apache-2.0

//! `GGUF` file writer — the format-symmetric inverse of `parse_gguf`.
//!
//! Phase 6 ships only the unquantised passthrough scaffold: scalar dtypes
//! (`F32`, `F16`, `BF16`, `F64`, `I8`–`I64`) round-trip through this writer
//! byte-exactly. Quantised dtypes (`Q*`, `IQ*`, `TQ*`, `MXFP4`) are rejected
//! with `AnamnesisError::Unsupported` — emitting those blocks requires the
//! encode kernels landing in Phase 7.5.
//!
//! # Spec reference
//!
//! Mirrors the parser `parse_gguf`: the on-disk layout is header (24 B) +
//! metadata `KV` table + tensor-info table + aligned tensor data. Every
//! per-tensor offset is a multiple of `general.alignment` (default 32 B),
//! and the alignment value itself is always written as
//! `general.alignment = U32(...)` so the produced file is fully
//! self-describing.
//!
//! # Determinism
//!
//! Metadata keys are serialised in lexicographic order so writing the same
//! `(metadata, tensors)` pair yields the same bytes on every run, regardless
//! of the calling `HashMap`'s iteration order. This is required by the
//! cross-format round-trip tests in `tests/cross_validation_convert.rs`,
//! which assert byte-exact `GGUF → safetensors → GGUF` cycles.

use std::collections::HashMap;
use std::hash::BuildHasher;
use std::io::{BufWriter, Seek, Write};
use std::path::Path;

use crate::error::AnamnesisError;
use crate::parse::gguf::{GgufMetadataArray, GgufMetadataValue, GgufType, align_up};

/// `GGUF` magic bytes — spells `"GGUF"` in `ASCII`.
const GGUF_MAGIC: &[u8; 4] = b"GGUF";

/// `GGUF` version emitted by this writer. Matches the latest version this
/// crate's parser accepts and what `llama.cpp` writes by default.
const GGUF_WRITE_VERSION: u32 = 3;

/// Default tensor-data alignment when the caller does not supply
/// `general.alignment`. Matches the parser's
/// [`DEFAULT_ALIGNMENT`](super::gguf) constant.
const DEFAULT_ALIGNMENT: u32 = 32;

/// Metadata key that records the tensor-data alignment in the produced file.
const ALIGNMENT_KEY: &str = "general.alignment";

/// Internal `BufWriter` capacity for the path-based [`write_gguf`] entry
/// point. 64 `KiB` matches the
/// [`READER_BUF_SIZE`](super::gguf) the parser uses on the read side, so
/// the syscall amortisation envelope is symmetric end-to-end.
const WRITER_BUF_SIZE: usize = 64 * 1024;

// ---------------------------------------------------------------------------
// GgufWriteTensor
// ---------------------------------------------------------------------------

/// A single tensor to be emitted into a `GGUF` file by [`write_gguf`] or
/// [`write_gguf_to_writer`].
///
/// `shape` is **most-significant-first**, matching
/// [`GgufTensorInfo::shape`](super::gguf::GgufTensorInfo) on the read side.
/// A row-major `[rows, cols]` matrix (`safetensors` / `NumPy` convention) must
/// be reversed by the caller to `[cols, rows]` before construction.
///
/// `data` is the raw little-endian byte payload. Length must equal
/// `dtype.byte_size_for_n_elements(product(shape))`; mismatches are caught at
/// write time and produce [`AnamnesisError::Parse`].
#[derive(Debug, Clone, Copy)]
pub struct GgufWriteTensor<'a> {
    /// Tensor name (e.g., `"blk.0.attn_q.weight"`). Stored verbatim in the
    /// file via a `gguf_string_t` length-prefix encoding.
    pub name: &'a str,
    /// Tensor dimensions, **most-significant-first**.
    pub shape: &'a [usize],
    /// Element / block data type. Must satisfy `!dtype.is_quantized()` —
    /// quantised emit is deferred to Phase 7.5.
    pub dtype: GgufType,
    /// Raw little-endian bytes. `data.len()` must equal
    /// `dtype.byte_size_for_n_elements(product(shape))`.
    pub data: &'a [u8],
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Writes a `GGUF` v3 file containing `tensors` at `path`.
///
/// `metadata` is the caller-supplied `KV` table. `general.alignment` is
/// injected automatically if absent (default 32 B); if present it must be a
/// non-zero `U32`. Tensor-data alignment respects whichever value ends up in
/// the written file.
///
/// Tensors are emitted in the order supplied. Their `data_offset`s are
/// laid out contiguously inside the data section, each rounded up to the
/// next alignment boundary. The data section itself starts at the first
/// alignment boundary past the tensor-info table.
///
/// # Errors
///
/// Returns [`AnamnesisError::Unsupported`] when any tensor's dtype is
/// `is_quantized()` (quantised emit lands in Phase 7.5), or when a supplied
/// `general.alignment` value is non-`U32` or zero.
///
/// Returns [`AnamnesisError::Parse`] when a tensor's `data.len()` disagrees
/// with the dtype/shape-implied byte count, when any element-count or offset
/// arithmetic overflows, or when the shape contains a zero dimension.
///
/// Returns [`AnamnesisError::Io`] if the output file cannot be created or
/// written.
///
/// # Memory
///
/// Allocates two temporary `Vec<u8>` scratch buffers (the serialised `KV`
/// block and the serialised tensor-info block) so their lengths are known
/// before the header is written. Both are released before the tensor-data
/// section is emitted. Peak heap is `O(metadata size + n_tensors × 80 B)` —
/// independent of the total tensor-data payload, which is streamed straight
/// to the wrapped `BufWriter` with no intermediate copy.
pub fn write_gguf<S: BuildHasher>(
    path: impl AsRef<Path>,
    tensors: &[GgufWriteTensor<'_>],
    metadata: &HashMap<String, GgufMetadataValue, S>,
) -> crate::Result<()> {
    let file = std::fs::File::create(path.as_ref()).map_err(AnamnesisError::Io)?;
    let writer = BufWriter::with_capacity(WRITER_BUF_SIZE, file);
    write_gguf_to_writer(writer, tensors, metadata)
}

/// Reader-generic core of [`write_gguf`]: emits a `GGUF` v3 file into any
/// `W: Write + Seek` substrate.
///
/// `Seek` is required because the writer cannot know `tensor_data_start`
/// until the tensor-info table has been serialised, yet the relative offsets
/// in that table reference the still-unknown data section. The
/// implementation does the layout arithmetic up front (in scratch buffers)
/// so seeking is unnecessary in the happy path; the `Seek` bound is kept on
/// the signature for future revisions that may stream more incrementally and
/// to match the symmetric `Read + Seek` bound on the parse side.
///
/// # Errors
///
/// Same as [`write_gguf`].
///
/// # Memory
///
/// Same as [`write_gguf`].
pub fn write_gguf_to_writer<W: Write + Seek, S: BuildHasher>(
    mut writer: W,
    tensors: &[GgufWriteTensor<'_>],
    metadata: &HashMap<String, GgufMetadataValue, S>,
) -> crate::Result<()> {
    // 1. Validate tensors and pre-compute total element counts.
    for tensor in tensors {
        validate_tensor(tensor)?;
    }

    // 2. Resolve the effective alignment. If the caller supplied
    //    `general.alignment` we honour it (provided it is a non-zero
    //    `U32`); otherwise we inject the default so the file is fully
    //    self-describing.
    let (alignment_u32, needs_inject_alignment) = resolve_alignment(metadata)?;
    let alignment_u64 = u64::from(alignment_u32);

    // 3. Build the effective metadata view. Sort keys lexicographically for
    //    deterministic output (HashMap iteration order is non-deterministic).
    //    `injected` lives in this scope so the borrow into `effective_kv`
    //    stays alive for the serialisation step below.
    let mut sorted_keys: Vec<&str> = metadata.keys().map(String::as_str).collect();
    if needs_inject_alignment {
        sorted_keys.push(ALIGNMENT_KEY);
    }
    let injected: GgufMetadataValue = GgufMetadataValue::U32(alignment_u32);
    sorted_keys.sort_unstable();
    let effective_kv: Vec<(&str, &GgufMetadataValue)> = sorted_keys
        .iter()
        .map(|&key| {
            let value = if needs_inject_alignment && key == ALIGNMENT_KEY {
                &injected
            } else {
                // The key was just collected from `metadata.keys()` so the
                // lookup cannot fail.
                metadata.get(key).ok_or_else(|| AnamnesisError::Parse {
                    reason: format!("GGUF write: metadata key `{key}` vanished mid-iteration"),
                })?
            };
            Ok((key, value))
        })
        .collect::<crate::Result<Vec<_>>>()?;

    // 4. Serialise the KV block to a scratch buffer.
    let mut kv_block: Vec<u8> = Vec::new();
    for (key, value) in &effective_kv {
        write_string(&mut kv_block, key)?;
        write_metadata_value(&mut kv_block, value)?;
    }

    // 5. Compute the post-header byte position of the tensor-info table.
    //    Header is fixed at 24 B (magic 4 + version 4 + tensor_count 8 +
    //    kv_count 8).
    // CAST: usize → u64, the KV block size is bounded by metadata input which
    // a reasonable caller keeps well under u64::MAX; checked_add catches the
    // adversarial overflow.
    #[allow(clippy::as_conversions)]
    let kv_block_len_u64 = kv_block.len() as u64;
    let header_size_u64: u64 = 24;
    let tensor_info_start = header_size_u64
        .checked_add(kv_block_len_u64)
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "GGUF write: tensor_info_start overflow".into(),
        })?;

    // 6. Serialise the tensor-info table while computing relative data
    //    offsets. We do not yet know `tensor_data_start`, so the relative
    //    offsets are independent of it (they are measured from the start of
    //    the data section, not from absolute file position).
    let mut tensor_info_block: Vec<u8> = Vec::new();
    let mut relative_offset: u64 = 0;
    for tensor in tensors {
        // CAST: usize → u64, tensor data length is bounded by mmap size
        // which fits in u64 on every supported target; checked path covers
        // 32-bit targets with adversarial inputs.
        #[allow(clippy::as_conversions)]
        let data_len_u64 = tensor.data.len() as u64;
        relative_offset = align_up(relative_offset, alignment_u64).map_err(|e| match e {
            AnamnesisError::Parse { reason } => AnamnesisError::Parse {
                reason: format!("GGUF write tensor `{}`: {reason}", tensor.name),
            },
            // `align_up` only ever returns the Parse variant; passing the
            // other variants through unchanged keeps the closure total
            // without claiming a hidden invariant.
            other @ (AnamnesisError::Unsupported { .. }
            | AnamnesisError::LimitExceeded { .. }
            | AnamnesisError::DisallowedGlobal { .. }
            | AnamnesisError::Io(_)) => other,
        })?;
        write_string(&mut tensor_info_block, tensor.name)?;
        // CAST: usize → u32, n_dimensions is bounded by MAX_TENSOR_DIMS=8 on
        // the read side; we re-enforce the same bound in `validate_tensor`.
        #[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
        let n_dims_u32 = tensor.shape.len() as u32;
        write_u32_le(&mut tensor_info_block, n_dims_u32)?;
        for &dim in tensor.shape {
            // CAST: usize → u64, dimension always fits in u64
            #[allow(clippy::as_conversions)]
            let dim_u64 = dim as u64;
            write_u64_le(&mut tensor_info_block, dim_u64)?;
        }
        write_u32_le(&mut tensor_info_block, gguf_type_to_u32(tensor.dtype))?;
        write_u64_le(&mut tensor_info_block, relative_offset)?;
        relative_offset =
            relative_offset
                .checked_add(data_len_u64)
                .ok_or_else(|| AnamnesisError::Parse {
                    reason: format!(
                        "GGUF write tensor `{}`: relative offset overflow at \
                         {relative_offset} + {data_len_u64}",
                        tensor.name
                    ),
                })?;
    }
    // The final `relative_offset` is the total payload size of the data
    // section. Re-borrow `tensors` to walk a second time for emission below.

    // 7. Compute the absolute byte position where tensor data begins.
    // CAST: usize → u64, tensor_info_block length fits in u64 on every
    // supported target.
    #[allow(clippy::as_conversions)]
    let tensor_info_block_len_u64 = tensor_info_block.len() as u64;
    let tensor_info_end = tensor_info_start
        .checked_add(tensor_info_block_len_u64)
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "GGUF write: tensor_info_end overflow".into(),
        })?;
    let tensor_data_start = if tensors.is_empty() {
        // A 24-byte-header-only GGUF is legitimately well-formed; the read
        // side at parse_gguf does the same shortcut.
        tensor_info_end
    } else {
        align_up(tensor_info_end, alignment_u64)?
    };

    // 8. Emit the header.
    writer.write_all(GGUF_MAGIC).map_err(AnamnesisError::Io)?;
    write_u32_le(&mut writer, GGUF_WRITE_VERSION)?;
    // CAST: usize → u64, tensor count and kv count always fit in u64.
    #[allow(clippy::as_conversions)]
    let tensor_count_u64 = tensors.len() as u64;
    write_u64_le(&mut writer, tensor_count_u64)?;
    // CAST: usize → u64, kv count fits in u64.
    #[allow(clippy::as_conversions)]
    let kv_count_u64 = effective_kv.len() as u64;
    write_u64_le(&mut writer, kv_count_u64)?;

    // 9. Emit the metadata KV block and tensor-info block.
    writer.write_all(&kv_block).map_err(AnamnesisError::Io)?;
    writer
        .write_all(&tensor_info_block)
        .map_err(AnamnesisError::Io)?;

    // 10. Pad to the tensor-data start.
    let padding_to_data = tensor_data_start
        .checked_sub(tensor_info_end)
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "GGUF write: tensor_data_start arithmetic underflow".into(),
        })?;
    write_zeros(&mut writer, padding_to_data)?;

    // 11. Emit tensor payloads with inter-tensor alignment padding.
    let mut emitted: u64 = 0;
    for tensor in tensors {
        // CAST: usize → u64, see step 6.
        #[allow(clippy::as_conversions)]
        let data_len_u64 = tensor.data.len() as u64;
        let desired = align_up(emitted, alignment_u64)?;
        let gap = desired
            .checked_sub(emitted)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!(
                    "GGUF write tensor `{}`: inter-tensor padding underflow",
                    tensor.name
                ),
            })?;
        write_zeros(&mut writer, gap)?;
        writer.write_all(tensor.data).map_err(AnamnesisError::Io)?;
        emitted = desired
            .checked_add(data_len_u64)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!(
                    "GGUF write tensor `{}`: emitted-byte counter overflow",
                    tensor.name
                ),
            })?;
    }

    writer.flush().map_err(AnamnesisError::Io)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

/// Maximum number of tensor dimensions accepted by the writer. Matches the
/// parser's `MAX_TENSOR_DIMS` cap so a written file always parses back.
const MAX_TENSOR_DIMS_USZ: usize = 8;

fn validate_tensor(tensor: &GgufWriteTensor<'_>) -> crate::Result<()> {
    if tensor.dtype.is_quantized() {
        return Err(AnamnesisError::Unsupported {
            format: "GGUF".into(),
            detail: format!(
                "writing quantized GGUF dtype {} requires Phase 7.5 encoders \
                 (Phase 6 ships only scalar dtype passthrough emit)",
                tensor.dtype
            ),
        });
    }
    if tensor.shape.is_empty() {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "GGUF write tensor `{}`: shape has zero dimensions",
                tensor.name
            ),
        });
    }
    if tensor.shape.len() > MAX_TENSOR_DIMS_USZ {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "GGUF write tensor `{}`: {}-D shape exceeds parser cap {MAX_TENSOR_DIMS_USZ}",
                tensor.name,
                tensor.shape.len()
            ),
        });
    }
    let mut n_elements: u64 = 1;
    for (axis, &dim) in tensor.shape.iter().enumerate() {
        if dim == 0 {
            return Err(AnamnesisError::Parse {
                reason: format!(
                    "GGUF write tensor `{}`: dimension {axis} is zero",
                    tensor.name
                ),
            });
        }
        // CAST: usize → u64, dimensions always fit in u64.
        #[allow(clippy::as_conversions)]
        let dim_u64 = dim as u64;
        n_elements = n_elements
            .checked_mul(dim_u64)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!(
                    "GGUF write tensor `{}`: element-count overflow at axis {axis}",
                    tensor.name
                ),
            })?;
    }
    let expected_bytes_u64 = tensor.dtype.byte_size_for_n_elements(n_elements)?;
    let expected_bytes =
        usize::try_from(expected_bytes_u64).map_err(|_| AnamnesisError::Parse {
            reason: format!(
                "GGUF write tensor `{}`: byte length {expected_bytes_u64} overflows usize",
                tensor.name
            ),
        })?;
    if tensor.data.len() != expected_bytes {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "GGUF write tensor `{}`: data length {} does not match \
                 shape/dtype-implied {expected_bytes} bytes",
                tensor.name,
                tensor.data.len()
            ),
        });
    }
    Ok(())
}

fn resolve_alignment<S: BuildHasher>(
    metadata: &HashMap<String, GgufMetadataValue, S>,
) -> crate::Result<(u32, bool)> {
    match metadata.get(ALIGNMENT_KEY) {
        Some(GgufMetadataValue::U32(v)) if *v != 0 => Ok((*v, false)),
        Some(GgufMetadataValue::U32(_)) => Err(AnamnesisError::Unsupported {
            format: "GGUF".into(),
            detail: "general.alignment must be non-zero".into(),
        }),
        Some(_) => Err(AnamnesisError::Unsupported {
            format: "GGUF".into(),
            detail: "general.alignment must be UINT32".into(),
        }),
        None => Ok((DEFAULT_ALIGNMENT, true)),
    }
}

// ---------------------------------------------------------------------------
// Primitive write helpers
// ---------------------------------------------------------------------------

fn write_u8(w: &mut impl Write, v: u8) -> crate::Result<()> {
    w.write_all(&[v]).map_err(AnamnesisError::Io)
}

fn write_i8(w: &mut impl Write, v: i8) -> crate::Result<()> {
    // CAST: i8 → u8, reinterpret bit pattern — signed/unsigned wrap is intended
    #[allow(clippy::as_conversions, clippy::cast_sign_loss)]
    let byte = v as u8;
    write_u8(w, byte)
}

fn write_u16_le(w: &mut impl Write, v: u16) -> crate::Result<()> {
    w.write_all(&v.to_le_bytes()).map_err(AnamnesisError::Io)
}

fn write_i16_le(w: &mut impl Write, v: i16) -> crate::Result<()> {
    w.write_all(&v.to_le_bytes()).map_err(AnamnesisError::Io)
}

fn write_u32_le(w: &mut impl Write, v: u32) -> crate::Result<()> {
    w.write_all(&v.to_le_bytes()).map_err(AnamnesisError::Io)
}

fn write_i32_le(w: &mut impl Write, v: i32) -> crate::Result<()> {
    w.write_all(&v.to_le_bytes()).map_err(AnamnesisError::Io)
}

fn write_u64_le(w: &mut impl Write, v: u64) -> crate::Result<()> {
    w.write_all(&v.to_le_bytes()).map_err(AnamnesisError::Io)
}

fn write_i64_le(w: &mut impl Write, v: i64) -> crate::Result<()> {
    w.write_all(&v.to_le_bytes()).map_err(AnamnesisError::Io)
}

fn write_f32_le(w: &mut impl Write, v: f32) -> crate::Result<()> {
    w.write_all(&v.to_le_bytes()).map_err(AnamnesisError::Io)
}

fn write_f64_le(w: &mut impl Write, v: f64) -> crate::Result<()> {
    w.write_all(&v.to_le_bytes()).map_err(AnamnesisError::Io)
}

fn write_bool(w: &mut impl Write, v: bool) -> crate::Result<()> {
    write_u8(w, u8::from(v))
}

/// Writes a `gguf_string_t`: `u64` length prefix followed by raw UTF-8
/// bytes. Mirrors the parser's
/// [`read_string`](super::gguf) helper.
fn write_string(w: &mut impl Write, s: &str) -> crate::Result<()> {
    let bytes = s.as_bytes();
    // CAST: usize → u64, string length always fits in u64.
    #[allow(clippy::as_conversions)]
    let len_u64 = bytes.len() as u64;
    write_u64_le(w, len_u64)?;
    w.write_all(bytes).map_err(AnamnesisError::Io)
}

/// Writes `n` zero bytes. Uses a small stack buffer; alignment padding is
/// always `< alignment` bytes (`≤ 31` for the default alignment), so a
/// single buffered write suffices in every realistic case.
fn write_zeros(w: &mut impl Write, n: u64) -> crate::Result<()> {
    const ZEROS: [u8; 256] = [0u8; 256];
    let mut remaining = n;
    while remaining > 0 {
        // `usize::try_from(remaining)` saturates to `usize::MAX` on 32-bit
        // targets when `remaining > usize::MAX` — bigger than any single
        // call should ever need; the `.min(ZEROS.len())` clamps the chunk
        // size to the stack buffer regardless.
        let chunk = usize::try_from(remaining)
            .unwrap_or(usize::MAX)
            .min(ZEROS.len());
        // INDEX: chunk ≤ ZEROS.len() == 256 by construction above
        #[allow(clippy::indexing_slicing)]
        let slice = &ZEROS[..chunk];
        w.write_all(slice).map_err(AnamnesisError::Io)?;
        // CAST: usize → u64, chunk ≤ 256 always fits in u64.
        #[allow(clippy::as_conversions)]
        let chunk_u64 = chunk as u64;
        remaining -= chunk_u64;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Metadata value writer
// ---------------------------------------------------------------------------

/// Writes a single metadata value: `u32` type tag followed by the value
/// payload. The type tags match the parser's `read_metadata_value`
/// dispatch in [`gguf.rs`](super::gguf).
fn write_metadata_value(w: &mut impl Write, value: &GgufMetadataValue) -> crate::Result<()> {
    match value {
        GgufMetadataValue::U8(v) => {
            write_u32_le(w, 0)?;
            write_u8(w, *v)
        }
        GgufMetadataValue::I8(v) => {
            write_u32_le(w, 1)?;
            write_i8(w, *v)
        }
        GgufMetadataValue::U16(v) => {
            write_u32_le(w, 2)?;
            write_u16_le(w, *v)
        }
        GgufMetadataValue::I16(v) => {
            write_u32_le(w, 3)?;
            write_i16_le(w, *v)
        }
        GgufMetadataValue::U32(v) => {
            write_u32_le(w, 4)?;
            write_u32_le(w, *v)
        }
        GgufMetadataValue::I32(v) => {
            write_u32_le(w, 5)?;
            write_i32_le(w, *v)
        }
        GgufMetadataValue::F32(v) => {
            write_u32_le(w, 6)?;
            write_f32_le(w, *v)
        }
        GgufMetadataValue::Bool(v) => {
            write_u32_le(w, 7)?;
            write_bool(w, *v)
        }
        GgufMetadataValue::String(s) => {
            write_u32_le(w, 8)?;
            write_string(w, s)
        }
        GgufMetadataValue::Array(arr) => {
            write_u32_le(w, 9)?;
            write_typed_array(w, arr.as_ref())
        }
        GgufMetadataValue::U64(v) => {
            write_u32_le(w, 10)?;
            write_u64_le(w, *v)
        }
        GgufMetadataValue::I64(v) => {
            write_u32_le(w, 11)?;
            write_i64_le(w, *v)
        }
        GgufMetadataValue::F64(v) => {
            write_u32_le(w, 12)?;
            write_f64_le(w, *v)
        }
    }
}

/// Writes a homogeneous typed array: `u32` inner-type tag, `u64` length,
/// then `length` payload entries. Nested arrays recurse via this same
/// function. Discriminant table matches the parser's `read_typed_array`.
fn write_typed_array(w: &mut impl Write, arr: &GgufMetadataArray) -> crate::Result<()> {
    match arr {
        GgufMetadataArray::U8(v) => {
            write_u32_le(w, 0)?;
            write_array_len(w, v.len())?;
            for &x in v {
                write_u8(w, x)?;
            }
            Ok(())
        }
        GgufMetadataArray::I8(v) => {
            write_u32_le(w, 1)?;
            write_array_len(w, v.len())?;
            for &x in v {
                write_i8(w, x)?;
            }
            Ok(())
        }
        GgufMetadataArray::U16(v) => {
            write_u32_le(w, 2)?;
            write_array_len(w, v.len())?;
            for &x in v {
                write_u16_le(w, x)?;
            }
            Ok(())
        }
        GgufMetadataArray::I16(v) => {
            write_u32_le(w, 3)?;
            write_array_len(w, v.len())?;
            for &x in v {
                write_i16_le(w, x)?;
            }
            Ok(())
        }
        GgufMetadataArray::U32(v) => {
            write_u32_le(w, 4)?;
            write_array_len(w, v.len())?;
            for &x in v {
                write_u32_le(w, x)?;
            }
            Ok(())
        }
        GgufMetadataArray::I32(v) => {
            write_u32_le(w, 5)?;
            write_array_len(w, v.len())?;
            for &x in v {
                write_i32_le(w, x)?;
            }
            Ok(())
        }
        GgufMetadataArray::F32(v) => {
            write_u32_le(w, 6)?;
            write_array_len(w, v.len())?;
            for &x in v {
                write_f32_le(w, x)?;
            }
            Ok(())
        }
        GgufMetadataArray::Bool(v) => {
            write_u32_le(w, 7)?;
            write_array_len(w, v.len())?;
            for &x in v {
                write_bool(w, x)?;
            }
            Ok(())
        }
        GgufMetadataArray::String(v) => {
            write_u32_le(w, 8)?;
            write_array_len(w, v.len())?;
            for s in v {
                write_string(w, s)?;
            }
            Ok(())
        }
        GgufMetadataArray::Array(v) => {
            write_u32_le(w, 9)?;
            write_array_len(w, v.len())?;
            for inner in v {
                write_typed_array(w, inner)?;
            }
            Ok(())
        }
        GgufMetadataArray::U64(v) => {
            write_u32_le(w, 10)?;
            write_array_len(w, v.len())?;
            for &x in v {
                write_u64_le(w, x)?;
            }
            Ok(())
        }
        GgufMetadataArray::I64(v) => {
            write_u32_le(w, 11)?;
            write_array_len(w, v.len())?;
            for &x in v {
                write_i64_le(w, x)?;
            }
            Ok(())
        }
        GgufMetadataArray::F64(v) => {
            write_u32_le(w, 12)?;
            write_array_len(w, v.len())?;
            for &x in v {
                write_f64_le(w, x)?;
            }
            Ok(())
        }
    }
}

fn write_array_len(w: &mut impl Write, len: usize) -> crate::Result<()> {
    // CAST: usize → u64, array length always fits in u64.
    #[allow(clippy::as_conversions)]
    let len_u64 = len as u64;
    write_u64_le(w, len_u64)
}

// ---------------------------------------------------------------------------
// GgufType → u32 discriminant
// ---------------------------------------------------------------------------

/// Maps a [`GgufType`] back to its on-disk `ggml_type` discriminant.
///
/// The inverse of [`GgufType::from_u32`](super::gguf). Quantised dtypes are
/// rejected at the `validate_tensor` boundary above, so callers only ever
/// reach this with a scalar dtype — but we provide the full mapping for
/// future-proofing once Phase 7.5 lights up the quantised emitters.
const fn gguf_type_to_u32(dtype: GgufType) -> u32 {
    match dtype {
        GgufType::F32 => 0,
        GgufType::F16 => 1,
        GgufType::Q4_0 => 2,
        GgufType::Q4_1 => 3,
        GgufType::Q5_0 => 6,
        GgufType::Q5_1 => 7,
        GgufType::Q8_0 => 8,
        GgufType::Q8_1 => 9,
        GgufType::Q2_K => 10,
        GgufType::Q3_K => 11,
        GgufType::Q4_K => 12,
        GgufType::Q5_K => 13,
        GgufType::Q6_K => 14,
        GgufType::Q8_K => 15,
        GgufType::IQ2_XXS => 16,
        GgufType::IQ2_XS => 17,
        GgufType::IQ3_XXS => 18,
        GgufType::IQ1_S => 19,
        GgufType::IQ4_NL => 20,
        GgufType::IQ3_S => 21,
        GgufType::IQ2_S => 22,
        GgufType::IQ4_XS => 23,
        GgufType::I8 => 24,
        GgufType::I16 => 25,
        GgufType::I32 => 26,
        GgufType::I64 => 27,
        GgufType::F64 => 28,
        GgufType::IQ1_M => 29,
        GgufType::BF16 => 30,
        GgufType::TQ1_0 => 34,
        GgufType::TQ2_0 => 35,
        GgufType::MXFP4 => 39,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::wildcard_enum_match_arm
)]
mod tests {
    use super::*;
    use crate::parse::gguf::parse_gguf;

    fn write_to_tempfile(
        tensors: &[GgufWriteTensor<'_>],
        metadata: &HashMap<String, GgufMetadataValue>,
    ) -> tempfile::NamedTempFile {
        let tmp = tempfile::Builder::new()
            .suffix(".gguf")
            .tempfile()
            .expect("create tempfile");
        write_gguf(tmp.path(), tensors, metadata).expect("write_gguf");
        tmp
    }

    #[test]
    fn roundtrip_empty_no_tensors() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "general.architecture".into(),
            GgufMetadataValue::String("anamnesis-test".into()),
        );

        let tmp = write_to_tempfile(&[], &metadata);
        let parsed = parse_gguf(tmp.path()).expect("parse_gguf");

        assert_eq!(parsed.version(), 3);
        assert!(parsed.is_empty());
        assert_eq!(parsed.alignment(), 32);
        let arch = parsed
            .metadata()
            .get("general.architecture")
            .and_then(GgufMetadataValue::as_string)
            .unwrap_or("");
        assert_eq!(arch, "anamnesis-test");
        // Alignment is injected even on the empty path.
        assert_eq!(
            parsed
                .metadata()
                .get("general.alignment")
                .and_then(GgufMetadataValue::as_u32),
            Some(32)
        );
    }

    #[test]
    fn roundtrip_single_f32() {
        // 6 F32 elements = 24 bytes, shape [2, 3]
        let data: Vec<u8> = (0..6)
            .flat_map(|i: u32| (i as f32 * 2.0).to_le_bytes())
            .collect();
        let shape = [2usize, 3];
        let tensors = [GgufWriteTensor {
            name: "w",
            shape: &shape,
            dtype: GgufType::F32,
            data: &data,
        }];

        let tmp = write_to_tempfile(&tensors, &HashMap::new());
        let parsed = parse_gguf(tmp.path()).expect("parse_gguf");
        let collected: Vec<_> = parsed.tensors().collect();
        assert_eq!(collected.len(), 1);
        let t = &collected[0];
        assert_eq!(t.name, "w");
        assert_eq!(t.shape, &[2, 3]);
        assert_eq!(t.dtype, GgufType::F32);
        assert_eq!(t.data.as_ref(), data.as_slice());
    }

    #[test]
    fn roundtrip_mixed_bf16_f32_i32() {
        let bf16_data: Vec<u8> = (0..8).flat_map(|i: u16| i.to_le_bytes()).collect();
        let f32_data: Vec<u8> = (0..4).flat_map(|i: u32| (i as f32).to_le_bytes()).collect();
        let i32_data: Vec<u8> = (0..2).flat_map(|i: i32| i.to_le_bytes()).collect();

        let bf16_shape = [4usize, 2];
        let f32_shape = [4usize];
        let i32_shape = [2usize];
        let tensors = [
            GgufWriteTensor {
                name: "bf16_tensor",
                shape: &bf16_shape,
                dtype: GgufType::BF16,
                data: &bf16_data,
            },
            GgufWriteTensor {
                name: "f32_tensor",
                shape: &f32_shape,
                dtype: GgufType::F32,
                data: &f32_data,
            },
            GgufWriteTensor {
                name: "i32_tensor",
                shape: &i32_shape,
                dtype: GgufType::I32,
                data: &i32_data,
            },
        ];

        let tmp = write_to_tempfile(&tensors, &HashMap::new());
        let parsed = parse_gguf(tmp.path()).expect("parse_gguf");
        let collected: Vec<_> = parsed.tensors().collect();
        assert_eq!(collected.len(), 3);

        // Order of write is preserved.
        assert_eq!(collected[0].name, "bf16_tensor");
        assert_eq!(collected[0].dtype, GgufType::BF16);
        assert_eq!(collected[0].data.as_ref(), bf16_data.as_slice());

        assert_eq!(collected[1].name, "f32_tensor");
        assert_eq!(collected[1].dtype, GgufType::F32);
        assert_eq!(collected[1].data.as_ref(), f32_data.as_slice());

        assert_eq!(collected[2].name, "i32_tensor");
        assert_eq!(collected[2].dtype, GgufType::I32);
        assert_eq!(collected[2].data.as_ref(), i32_data.as_slice());

        // Every emitted tensor's absolute offset is a multiple of 32.
        for info in parsed.tensor_info() {
            assert_eq!(
                info.data_offset % 32,
                0,
                "tensor `{}` misaligned",
                info.name
            );
        }
    }

    #[test]
    fn roundtrip_metadata_kv() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "string_key".into(),
            GgufMetadataValue::String("hello".into()),
        );
        metadata.insert("u32_key".into(), GgufMetadataValue::U32(42));
        metadata.insert(
            "f32_key".into(),
            GgufMetadataValue::F32(std::f32::consts::PI),
        );
        metadata.insert("bool_key".into(), GgufMetadataValue::Bool(true));
        metadata.insert(
            "string_array_key".into(),
            GgufMetadataValue::Array(Box::new(GgufMetadataArray::String(vec![
                "a".into(),
                "b".into(),
                "c".into(),
            ]))),
        );
        metadata.insert(
            "u64_array_key".into(),
            GgufMetadataValue::Array(Box::new(GgufMetadataArray::U64(vec![1, 2, 3, 4]))),
        );

        let tmp = write_to_tempfile(&[], &metadata);
        let parsed = parse_gguf(tmp.path()).expect("parse_gguf");

        assert_eq!(
            parsed
                .metadata()
                .get("string_key")
                .and_then(GgufMetadataValue::as_string),
            Some("hello")
        );
        assert_eq!(
            parsed
                .metadata()
                .get("u32_key")
                .and_then(GgufMetadataValue::as_u32),
            Some(42)
        );
        match parsed.metadata().get("f32_key") {
            Some(GgufMetadataValue::F32(v)) => assert!((v - std::f32::consts::PI).abs() < 1e-7),
            other => panic!("expected F32, got {other:?}"),
        }
        assert_eq!(
            parsed
                .metadata()
                .get("bool_key")
                .and_then(GgufMetadataValue::as_bool),
            Some(true)
        );
        match parsed
            .metadata()
            .get("string_array_key")
            .and_then(GgufMetadataValue::as_array)
        {
            Some(GgufMetadataArray::String(v)) => {
                assert_eq!(v, &vec!["a".to_owned(), "b".to_owned(), "c".to_owned()]);
            }
            other => panic!("expected String array, got {other:?}"),
        }
        match parsed
            .metadata()
            .get("u64_array_key")
            .and_then(GgufMetadataValue::as_array)
        {
            Some(GgufMetadataArray::U64(v)) => assert_eq!(v, &vec![1, 2, 3, 4]),
            other => panic!("expected U64 array, got {other:?}"),
        }
    }

    #[test]
    fn reject_quantized_dtype() {
        // Q4_K block is 144 bytes for 256 elements.
        let data = vec![0u8; 144];
        let shape = [256usize];
        let tensors = [GgufWriteTensor {
            name: "q",
            shape: &shape,
            dtype: GgufType::Q4_K,
            data: &data,
        }];
        let tmp = tempfile::Builder::new()
            .suffix(".gguf")
            .tempfile()
            .expect("create tempfile");
        let err = write_gguf(tmp.path(), &tensors, &HashMap::new()).expect_err("should reject");
        match err {
            AnamnesisError::Unsupported { format, detail } => {
                assert_eq!(format, "GGUF");
                assert!(detail.contains("Phase 7.5"), "unexpected detail: {detail}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn reject_data_length_mismatch() {
        // F32 shape [2, 3] needs 24 bytes; supply only 8.
        let data = vec![0u8; 8];
        let shape = [2usize, 3];
        let tensors = [GgufWriteTensor {
            name: "w",
            shape: &shape,
            dtype: GgufType::F32,
            data: &data,
        }];
        let tmp = tempfile::Builder::new()
            .suffix(".gguf")
            .tempfile()
            .expect("create tempfile");
        let err = write_gguf(tmp.path(), &tensors, &HashMap::new()).expect_err("should reject");
        match err {
            AnamnesisError::Parse { reason } => {
                assert!(
                    reason.contains("data length"),
                    "unexpected reason: {reason}"
                );
            }
            other => panic!("expected Parse, got {other:?}"),
        }
    }

    #[test]
    fn roundtrip_with_custom_alignment_8() {
        // F32 shape [3] = 12 bytes; alignment 8 means tensor offset is 8-aligned.
        let data: Vec<u8> = (0..3).flat_map(|i: u32| (i as f32).to_le_bytes()).collect();
        let shape = [3usize];
        let tensors = [GgufWriteTensor {
            name: "w",
            shape: &shape,
            dtype: GgufType::F32,
            data: &data,
        }];
        let mut metadata = HashMap::new();
        metadata.insert(ALIGNMENT_KEY.into(), GgufMetadataValue::U32(8));

        let tmp = write_to_tempfile(&tensors, &metadata);
        let parsed = parse_gguf(tmp.path()).expect("parse_gguf");
        assert_eq!(parsed.alignment(), 8);
        for info in parsed.tensor_info() {
            assert_eq!(info.data_offset % 8, 0, "tensor `{}` misaligned", info.name);
        }
        let collected: Vec<_> = parsed.tensors().collect();
        assert_eq!(collected[0].data.as_ref(), data.as_slice());
    }

    #[test]
    fn reject_zero_alignment() {
        let mut metadata = HashMap::new();
        metadata.insert(ALIGNMENT_KEY.into(), GgufMetadataValue::U32(0));
        let tmp = tempfile::Builder::new()
            .suffix(".gguf")
            .tempfile()
            .expect("create tempfile");
        let err = write_gguf(tmp.path(), &[], &metadata).expect_err("should reject");
        match err {
            AnamnesisError::Unsupported { format, detail } => {
                assert_eq!(format, "GGUF");
                assert!(detail.contains("non-zero"), "unexpected detail: {detail}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }
}
