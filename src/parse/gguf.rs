// SPDX-License-Identifier: MIT OR Apache-2.0

//! `GGUF` file parsing — header, metadata key-value pairs, and tensor info table.
//!
//! `GGUF` is the dominant format for local inference, used by `llama.cpp`,
//! `Ollama`, and `LM Studio`. This module implements a lean, dependency-free
//! parser for `GGUF` versions 2 and 3 that memory-maps the file and exposes
//! tensor views as zero-copy `Cow::Borrowed` slices into the mapping — no
//! per-tensor allocation for little-endian data (the common case).
//!
//! # What this module does
//!
//! - Validates the `GGUF` magic (`"GGUF"`) and version.
//! - Reads every metadata key-value pair into a `HashMap` — supports all 13
//!   value types defined by the `GGUF` specification, including nested
//!   `ARRAY` values.
//! - Reads the tensor info table (`name`, `shape`, `ggml_type`, `offset`) and
//!   resolves each tensor's absolute position inside the memory-mapped
//!   region, honouring the `general.alignment` metadata key (default 32).
//! - Exposes a [`ParsedGguf`] handle with inspection helpers and a
//!   [`tensors`](ParsedGguf::tensors) method that returns tensor views
//!   borrowed from the mmap.
//!
//! # What this module does not do
//!
//! Dequantization of `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, etc. is the job of the
//! `remember::gguf` module (Phase 4 step 2). This parser reports the
//! [`GgufType`] of every tensor but does not decode any packed blocks.
//!
//! # Security
//!
//! The parser enforces cheap upper bounds on `tensor_count`,
//! `metadata_kv_count`, string lengths, array lengths, and array nesting
//! depth so that an adversarial file cannot cause unbounded allocation or
//! stack growth.
//!
//! # Spec reference
//!
//! <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>

use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt;
use std::io::{BufReader, Cursor, Read, Seek, SeekFrom};
use std::path::Path;

use crate::backing::Backing;
use crate::error::AnamnesisError;
use crate::limits::Budget;
use crate::parse::utils::PREALLOC_SOFT_CAP;
use crate::ParseLimits;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// `GGUF` magic bytes — spells `"GGUF"` in ASCII.
const GGUF_MAGIC: &[u8; 4] = b"GGUF";

/// `GGUF` magic read as a little-endian `u32`. Useful for detecting
/// byte-swapped (big-endian) files without a full magic-byte comparison.
const GGUF_MAGIC_LE_U32: u32 = u32::from_le_bytes(*GGUF_MAGIC);

/// `GGUF` magic as a big-endian `u32`. A file that begins with this value
/// when interpreted little-endian is actually stored big-endian.
const GGUF_MAGIC_BE_U32: u32 = u32::from_be_bytes(*GGUF_MAGIC);

/// Default tensor-data alignment when `general.alignment` metadata is absent.
const DEFAULT_ALIGNMENT: u32 = 32;

/// Upper bound on `tensor_count` (soft `DoS` guard).
const MAX_TENSOR_COUNT: u64 = 1_000_000;

/// Upper bound on `metadata_kv_count` (soft `DoS` guard).
const MAX_KV_COUNT: u64 = 1_000_000;

/// Upper bound on a single `GGUF` string length (16 MiB).
///
/// The `GGUF` specification caps metadata keys at 65 535 bytes; values are
/// unbounded in theory but in practice rarely exceed a few hundred kilobytes
/// (e.g., a tokenizer vocabulary serialised as a single string).
const MAX_STRING_LEN: u64 = 16 * 1024 * 1024;

/// `GGUF`-spec cap on a metadata-key length (`u16::MAX` = 65 535 bytes).
const MAX_GGUF_KEY_LEN: u64 = 65_535;

/// Upper bound on `ARRAY` element count (soft `DoS` guard).
const MAX_ARRAY_LEN: u64 = 16_000_000;

/// Maximum nesting depth for metadata `ARRAY` values.
const MAX_ARRAY_DEPTH: u32 = 4;

/// Upper bound on `n_dimensions` for a single tensor.
///
/// `ggml` itself caps this at `GGML_MAX_DIMS = 4`; we accept up to 8 for
/// future-proofing.
const MAX_TENSOR_DIMS: u32 = 8;

/// Upper bound on a single tensor's name length, in bytes.
///
/// The `GGUF` specification caps tensor names at 64 bytes, but some encoders
/// produce longer names in practice. 65 535 bytes is the metadata-key cap
/// and is comfortably above anything any real encoder emits, while keeping
/// the per-tensor string allocation bounded for adversarial inputs.
const MAX_TENSOR_NAME_LEN: u64 = 65_535;

/// Upper bound on product-of-dimensions for a single tensor (soft `DoS` guard).
///
/// Real tensors never exceed a few hundred billion elements (a 70B model's
/// embedding matrix tops out around 5·10⁹). One trillion elements is
/// comfortably beyond anything real while rejecting absurd inputs.
const MAX_TENSOR_ELEMENTS: u64 = 1_000_000_000_000;

/// Buffer capacity for the internal `BufReader<R>` that wraps any
/// caller-supplied reader inside [`inspect_gguf_from_reader`].
///
/// `GGUF`'s parser issues many small `read_exact` calls (4 B per `u32`,
/// 8 B per `u64`, 1 B per `bool`, …). On a `std::fs::File` substrate every
/// one is a syscall. The default `std::io::BufReader` capacity is 8 KiB;
/// 64 KiB is large enough that a typical `bartowski/SmolLM2-135M-Instruct`
/// front matter (~256 KiB tokenizer arrays) refills the buffer 4× instead
/// of 32×, and large enough to amortise the per-read overhead of an
/// `HTTP`-range adapter that prefetches at this granularity, but still
/// negligible heap pressure for a header-only inspect (~64 KiB on top of
/// the parsed metadata `HashMap`).
const READER_BUF_SIZE: usize = 64 * 1024;

/// Total number of [`GgufType`] variants — used to size the per-dtype
/// dedup bitmap in [`ParsedGguf::inspect`]. Must be kept in sync with
/// the match arms of `GgufType::inspect_index`.
const GGUF_TYPE_COUNT: usize = 32;

// ---------------------------------------------------------------------------
// GgufType
// ---------------------------------------------------------------------------

/// Element data type for a `GGUF` tensor — mirrors `ggml_type` in `llama.cpp`.
///
/// The enum is `#[non_exhaustive]` because new `ggml_type` values are added
/// over time (e.g., the `IQ*` family appeared after the original `K`-quants,
/// and `MXFP4` was added in 2024).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[allow(non_camel_case_types)]
pub enum GgufType {
    /// 32-bit IEEE 754 single-precision (`GGML_TYPE_F32 = 0`).
    F32,
    /// 16-bit IEEE 754 half-precision (`GGML_TYPE_F16 = 1`).
    F16,
    /// 32-element block, 4-bit symmetric quantisation (`GGML_TYPE_Q4_0 = 2`).
    Q4_0,
    /// 32-element block, 4-bit asymmetric quantisation (`GGML_TYPE_Q4_1 = 3`).
    Q4_1,
    /// 32-element block, 5-bit symmetric quantisation (`GGML_TYPE_Q5_0 = 6`).
    Q5_0,
    /// 32-element block, 5-bit asymmetric quantisation (`GGML_TYPE_Q5_1 = 7`).
    Q5_1,
    /// 32-element block, 8-bit symmetric quantisation (`GGML_TYPE_Q8_0 = 8`).
    Q8_0,
    /// 32-element block, 8-bit quantisation with sum (`GGML_TYPE_Q8_1 = 9`).
    Q8_1,
    /// 256-element super-block, 2-bit K-quant (`GGML_TYPE_Q2_K = 10`).
    Q2_K,
    /// 256-element super-block, 3-bit K-quant (`GGML_TYPE_Q3_K = 11`).
    Q3_K,
    /// 256-element super-block, 4-bit K-quant (`GGML_TYPE_Q4_K = 12`).
    Q4_K,
    /// 256-element super-block, 5-bit K-quant (`GGML_TYPE_Q5_K = 13`).
    Q5_K,
    /// 256-element super-block, 6-bit K-quant (`GGML_TYPE_Q6_K = 14`).
    Q6_K,
    /// 256-element super-block, 8-bit K-quant (`GGML_TYPE_Q8_K = 15`).
    Q8_K,
    /// 256-element, ~2.0625 bpw (`GGML_TYPE_IQ2_XXS = 16`).
    IQ2_XXS,
    /// 256-element, ~2.3125 bpw (`GGML_TYPE_IQ2_XS = 17`).
    IQ2_XS,
    /// 256-element, ~3.0625 bpw (`GGML_TYPE_IQ3_XXS = 18`).
    IQ3_XXS,
    /// 256-element, ~1.5625 bpw (`GGML_TYPE_IQ1_S = 19`).
    IQ1_S,
    /// 32-element, 4-bit non-linear (`GGML_TYPE_IQ4_NL = 20`).
    IQ4_NL,
    /// 256-element, ~3.4375 bpw (`GGML_TYPE_IQ3_S = 21`).
    IQ3_S,
    /// 256-element, ~2.5 bpw (`GGML_TYPE_IQ2_S = 22`).
    IQ2_S,
    /// 256-element, ~4.25 bpw (`GGML_TYPE_IQ4_XS = 23`).
    IQ4_XS,
    /// Signed 8-bit integer (`GGML_TYPE_I8 = 24`).
    I8,
    /// Signed 16-bit integer (`GGML_TYPE_I16 = 25`).
    I16,
    /// Signed 32-bit integer (`GGML_TYPE_I32 = 26`).
    I32,
    /// Signed 64-bit integer (`GGML_TYPE_I64 = 27`).
    I64,
    /// 64-bit IEEE 754 double-precision (`GGML_TYPE_F64 = 28`).
    F64,
    /// 256-element, ~1.75 bpw (`GGML_TYPE_IQ1_M = 29`).
    IQ1_M,
    /// 16-bit brain floating point (`GGML_TYPE_BF16 = 30`).
    BF16,
    /// Ternary 1-bit packing variant `0` (`GGML_TYPE_TQ1_0 = 34`).
    TQ1_0,
    /// Ternary 2-bit packing variant `0` (`GGML_TYPE_TQ2_0 = 35`).
    TQ2_0,
    /// 32-element, 4-bit microscaling FP (`GGML_TYPE_MXFP4 = 39`).
    MXFP4,
}

impl GgufType {
    /// Parses a `u32` `ggml_type` discriminant into a [`GgufType`].
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::Unsupported`] if the value does not match
    /// any known `ggml_type`. Reserved or removed discriminants (4, 5, 31–33,
    /// 36–38) also produce this error.
    fn from_u32(value: u32) -> crate::Result<Self> {
        let ty = match value {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2_K,
            11 => Self::Q3_K,
            12 => Self::Q4_K,
            13 => Self::Q5_K,
            14 => Self::Q6_K,
            15 => Self::Q8_K,
            16 => Self::IQ2_XXS,
            17 => Self::IQ2_XS,
            18 => Self::IQ3_XXS,
            19 => Self::IQ1_S,
            20 => Self::IQ4_NL,
            21 => Self::IQ3_S,
            22 => Self::IQ2_S,
            23 => Self::IQ4_XS,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            29 => Self::IQ1_M,
            30 => Self::BF16,
            34 => Self::TQ1_0,
            35 => Self::TQ2_0,
            39 => Self::MXFP4,
            other => {
                return Err(AnamnesisError::Unsupported {
                    format: "GGUF".into(),
                    detail: format!("unknown ggml_type discriminant {other}"),
                });
            }
        };
        Ok(ty)
    }

    /// Number of elements per storage block.
    ///
    /// Unquantised scalar types return `1`. Legacy quantised types return
    /// `32`. K-quants and most `IQ*`/`TQ*` types return `256`. `IQ4_NL` and
    /// `MXFP4` return `32`.
    #[must_use]
    pub const fn block_size(self) -> usize {
        match self {
            Self::F32
            | Self::F16
            | Self::BF16
            | Self::F64
            | Self::I8
            | Self::I16
            | Self::I32
            | Self::I64 => 1,
            Self::Q4_0
            | Self::Q4_1
            | Self::Q5_0
            | Self::Q5_1
            | Self::Q8_0
            | Self::Q8_1
            | Self::IQ4_NL
            | Self::MXFP4 => 32,
            Self::Q2_K
            | Self::Q3_K
            | Self::Q4_K
            | Self::Q5_K
            | Self::Q6_K
            | Self::Q8_K
            | Self::IQ2_XXS
            | Self::IQ2_XS
            | Self::IQ3_XXS
            | Self::IQ1_S
            | Self::IQ3_S
            | Self::IQ2_S
            | Self::IQ4_XS
            | Self::IQ1_M
            | Self::TQ1_0
            | Self::TQ2_0 => 256,
        }
    }

    /// Number of bytes per storage block, or `None` for types whose block
    /// layout is not yet hard-coded in this crate.
    ///
    /// Returns `Some` for every recognised `GgufType`: scalar types
    /// (`F32`, `F16`, `BF16`, `F64`, `I8`–`I64`), legacy block-wise
    /// quantised types (`Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q8_1`),
    /// K-quant super-blocks (`Q2_K`–`Q8_K`), the non-linear 4-bit `IQ*`
    /// variants (`IQ4_NL` at 18 bytes, `IQ4_XS` at 136 bytes), the
    /// three 2-bit `IQ*` variants (`IQ2_XXS` at 66 bytes, `IQ2_XS` at
    /// 74 bytes, `IQ2_S` at 82 bytes), the two 3-bit `IQ*` variants
    /// (`IQ3_XXS` at 98 bytes, `IQ3_S` at 110 bytes), the two 1-bit
    /// `IQ*` variants (`IQ1_S` at 50 bytes, `IQ1_M` at 56 bytes), the
    /// two ternary `TQ*` variants (`TQ1_0` at 54 bytes, `TQ2_0` at 66
    /// bytes), and the microscaling `MXFP4` variant (17 bytes). The
    /// return type is kept as `Option<usize>` for API stability — every
    /// arm currently returns `Some(_)`, so callers can `.unwrap_or(0)`
    /// or unwrap defensively without ever exercising the `None` branch.
    // `Q4_0` and `IQ4_NL` happen to both be 18 bytes (same `ggml_half` +
    // 16 nibble-packed bytes), and other pairs share byte counts too; keeping
    // the arms separate documents the distinct block-format semantics instead
    // of collapsing them into pattern lists.
    #[allow(clippy::match_same_arms)]
    #[must_use]
    pub const fn type_size(self) -> Option<usize> {
        match self {
            Self::I8 => Some(1),
            Self::F16 | Self::BF16 | Self::I16 => Some(2),
            Self::F32 | Self::I32 => Some(4),
            Self::F64 | Self::I64 => Some(8),
            // Legacy block-wise quants (32-element blocks).
            Self::Q4_0 => Some(18),
            Self::Q4_1 => Some(20),
            Self::Q5_0 => Some(22),
            Self::Q5_1 => Some(24),
            Self::Q8_0 => Some(34),
            Self::Q8_1 => Some(36),
            // K-quants (256-element super-blocks).
            Self::Q2_K => Some(84),
            Self::Q3_K => Some(110),
            Self::Q4_K => Some(144),
            Self::Q5_K => Some(176),
            Self::Q6_K => Some(210),
            Self::Q8_K => Some(292),
            // Non-linear 4-bit IQ variants (share the `kvalues_iq4nl` codebook).
            // IQ4_NL: d (f16, 2 B) + qs (4-bit packed, 16 B) = 18 B per 32-element block.
            // IQ4_XS: d (f16, 2 B) + scales_h (u16, 2 B) + scales_l (4 B) + qs (128 B)
            //         = 136 B per 256-element super-block.
            Self::IQ4_NL => Some(18),
            Self::IQ4_XS => Some(136),
            // 2-bit IQ super-quants (256-element super-blocks). All three share
            // the `ksigns_iq2xs` / `kmask_iq2xs` sign tables; the grid tables
            // differ in size (256 / 512 / 1024 entries, each an 8-byte lattice
            // codebook vector).
            // IQ2_XXS: d (f16, 2 B) + qs (u16[32], 64 B)                    = 66 B.
            // IQ2_XS:  d (f16, 2 B) + qs (u16[32], 64 B) + scales (8 B)     = 74 B.
            // IQ2_S:   d (f16, 2 B) + qs (u8[64], 64 B) + qh (8 B) + scales (8 B) = 82 B.
            Self::IQ2_XXS => Some(66),
            Self::IQ2_XS => Some(74),
            Self::IQ2_S => Some(82),
            // 3-bit IQ super-quants (256-element super-blocks). IQ3_XXS reuses
            // `ksigns_iq2xs` for sign indexing (like IQ2_XXS); IQ3_S stores
            // sign masks inline (like IQ2_S). Both grids are [u32; N] (4-byte
            // codebook vectors), unlike the IQ2 family's [u64; N].
            // IQ3_XXS: d (f16, 2 B) + qs (u8[64], 64 B for grid indices)
            //        + scales_and_signs (u32[8], 32 B)                     = 98 B.
            // IQ3_S:   d (f16, 2 B) + qs (u8[64], 64 B) + qh (u8[8], 8 B)
            //        + signs (u8[32], 32 B) + scales (u8[4], 4 B)          = 110 B.
            Self::IQ3_XXS => Some(98),
            Self::IQ3_S => Some(110),
            // 1-bit IQ super-quants (256-element super-blocks). Both share
            // the 2048-entry IQ1S_GRID (signed i8 codebook) and the
            // IQ1S_DELTA = 0.125 additive bias.
            // IQ1_S: d (f16, 2 B) + qs (u8[32], 32 B) + qh (u16[8], 16 B) = 50 B.
            // IQ1_M: qs (u8[32], 32 B) + qh (u8[16], 16 B) + scales (u8[8], 8 B)
            //        = 56 B (no top-level d — super-block scale is reconstructed
            //        from a scattered 16-bit pattern across `scales`).
            Self::IQ1_S => Some(50),
            Self::IQ1_M => Some(56),
            // Ternary super-quants (256-element super-blocks). Both decode
            // values in {-d, 0, +d}.
            // TQ1_0: qs (u8[48], 48 B base-3 packed, 5 ternaries/byte)
            //      + qh (u8[4], 4 B base-3 packed, 4 ternaries/byte)
            //      + d (f16, 2 B) = 54 B.
            // TQ2_0: qs (u8[64], 64 B, 4 ternaries/byte at 2 bits each)
            //      + d (f16, 2 B) = 66 B.
            Self::TQ1_0 => Some(54),
            Self::TQ2_0 => Some(66),
            // Microscaling FP4 (32-element block, OCP MX standard, added to
            // ggml in 2024). e (E8M0 byte exponent, 1 B) + qs (4-bit packed,
            // 16 B) = 17 B per block.
            Self::MXFP4 => Some(17),
        }
    }

    /// Returns `true` if this type is a quantised block format (as opposed
    /// to a scalar float or integer type).
    #[must_use]
    pub const fn is_quantized(self) -> bool {
        !matches!(
            self,
            Self::F32
                | Self::F16
                | Self::BF16
                | Self::F64
                | Self::I8
                | Self::I16
                | Self::I32
                | Self::I64
        )
    }

    /// Dense `0..GGUF_TYPE_COUNT` index used by [`ParsedGguf::inspect`]'s
    /// dtype-dedup bitmap. The value is an internal implementation detail —
    /// callers should never depend on a specific mapping.
    const fn inspect_index(self) -> usize {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q5_0 => 4,
            Self::Q5_1 => 5,
            Self::Q8_0 => 6,
            Self::Q8_1 => 7,
            Self::Q2_K => 8,
            Self::Q3_K => 9,
            Self::Q4_K => 10,
            Self::Q5_K => 11,
            Self::Q6_K => 12,
            Self::Q8_K => 13,
            Self::IQ2_XXS => 14,
            Self::IQ2_XS => 15,
            Self::IQ3_XXS => 16,
            Self::IQ1_S => 17,
            Self::IQ4_NL => 18,
            Self::IQ3_S => 19,
            Self::IQ2_S => 20,
            Self::IQ4_XS => 21,
            Self::I8 => 22,
            Self::I16 => 23,
            Self::I32 => 24,
            Self::I64 => 25,
            Self::F64 => 26,
            Self::IQ1_M => 27,
            Self::BF16 => 28,
            Self::TQ1_0 => 29,
            Self::TQ2_0 => 30,
            Self::MXFP4 => 31,
        }
    }

    /// Computes the byte size of a contiguous tensor of this type containing
    /// `n_elements` elements.
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::Unsupported`] if this type's `type_size` is
    /// not yet known to the parser (see [`type_size`](Self::type_size)).
    ///
    /// Returns [`AnamnesisError::Parse`] if `n_elements` is not a multiple of
    /// the block size, or if the multiplication overflows `u64`.
    pub fn byte_size_for_n_elements(self, n_elements: u64) -> crate::Result<u64> {
        let type_size = self
            .type_size()
            .ok_or_else(|| AnamnesisError::Unsupported {
                format: "GGUF".into(),
                detail: format!("byte size not hard-coded for ggml_type {self}"),
            })?;
        // CAST: usize → u64, `block_size()` returns at most 256, always fits
        #[allow(clippy::as_conversions)]
        let block_size = self.block_size() as u64;
        // CAST: usize → u64, `type_size()` returns at most 292, always fits
        #[allow(clippy::as_conversions)]
        let type_size_u64 = type_size as u64;
        if !n_elements.is_multiple_of(block_size) {
            return Err(AnamnesisError::Parse {
                reason: format!(
                    "GGUF tensor: element count {n_elements} not a multiple of block size \
                     {block_size} for type {self}"
                ),
            });
        }
        let n_blocks = n_elements / block_size;
        n_blocks
            .checked_mul(type_size_u64)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!(
                    "GGUF tensor: byte-size overflow ({n_blocks} blocks × {type_size_u64} bytes)"
                ),
            })
    }
}

impl fmt::Display for GgufType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::BF16 => "BF16",
            Self::F64 => "F64",
            Self::I8 => "I8",
            Self::I16 => "I16",
            Self::I32 => "I32",
            Self::I64 => "I64",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q8_1 => "Q8_1",
            Self::Q2_K => "Q2_K",
            Self::Q3_K => "Q3_K",
            Self::Q4_K => "Q4_K",
            Self::Q5_K => "Q5_K",
            Self::Q6_K => "Q6_K",
            Self::Q8_K => "Q8_K",
            Self::IQ2_XXS => "IQ2_XXS",
            Self::IQ2_XS => "IQ2_XS",
            Self::IQ3_XXS => "IQ3_XXS",
            Self::IQ1_S => "IQ1_S",
            Self::IQ4_NL => "IQ4_NL",
            Self::IQ3_S => "IQ3_S",
            Self::IQ2_S => "IQ2_S",
            Self::IQ4_XS => "IQ4_XS",
            Self::IQ1_M => "IQ1_M",
            Self::TQ1_0 => "TQ1_0",
            Self::TQ2_0 => "TQ2_0",
            Self::MXFP4 => "MXFP4",
        };
        f.write_str(s)
    }
}

// ---------------------------------------------------------------------------
// GgufMetadataValue
// ---------------------------------------------------------------------------

/// A value stored in the `GGUF` metadata key-value table.
///
/// Mirrors the 13 `gguf_metadata_value_type` variants defined by the spec.
/// [`Self::Array`] is a boxed [`GgufMetadataArray`] — the array's elements
/// are stored natively for their type (e.g. `Vec<f32>`) rather than as a
/// `Vec<GgufMetadataValue>`. This eliminates the ~8× enum-discriminant
/// bloat on homogeneous numeric arrays and, as a side effect, shrinks
/// `GgufMetadataValue` itself from 32 bytes to 24 bytes because the
/// largest variant is now `String` rather than the old `Vec<Self>`.
///
/// Arrays may nest (e.g. a tokenizer merges list is an array of arrays of
/// strings); the parser refuses to recurse beyond four levels deep.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum GgufMetadataValue {
    /// Unsigned 8-bit integer.
    U8(u8),
    /// Signed 8-bit integer.
    I8(i8),
    /// Unsigned 16-bit integer (stored little-endian in the file).
    U16(u16),
    /// Signed 16-bit integer (stored little-endian).
    I16(i16),
    /// Unsigned 32-bit integer (stored little-endian).
    U32(u32),
    /// Signed 32-bit integer (stored little-endian).
    I32(i32),
    /// 32-bit IEEE 754 single-precision (stored little-endian).
    F32(f32),
    /// Boolean — encoded in the file as a single byte (0 or 1).
    Bool(bool),
    /// UTF-8 string — encoded as `u64` length followed by raw bytes.
    String(String),
    /// Homogeneous array of a single inner type, boxed to keep
    /// `GgufMetadataValue` small (24 bytes on 64-bit).
    Array(Box<GgufMetadataArray>),
    /// Unsigned 64-bit integer (stored little-endian).
    U64(u64),
    /// Signed 64-bit integer (stored little-endian).
    I64(i64),
    /// 64-bit IEEE 754 double-precision (stored little-endian).
    F64(f64),
}

/// Homogeneous `GGUF` metadata array, stored natively for its element type.
///
/// The parser dispatches on the array's `inner_type` when reading and
/// builds a correctly-typed `Vec<T>` directly from the byte stream. For a
/// 16 M-element `f32` array this consumes ~64 MB of heap instead of the
/// ~488 MB a `Vec<GgufMetadataValue>` would require.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum GgufMetadataArray {
    /// Array of unsigned 8-bit integers.
    U8(Vec<u8>),
    /// Array of signed 8-bit integers.
    I8(Vec<i8>),
    /// Array of unsigned 16-bit integers.
    U16(Vec<u16>),
    /// Array of signed 16-bit integers.
    I16(Vec<i16>),
    /// Array of unsigned 32-bit integers.
    U32(Vec<u32>),
    /// Array of signed 32-bit integers.
    I32(Vec<i32>),
    /// Array of 32-bit IEEE 754 floats.
    F32(Vec<f32>),
    /// Array of booleans.
    Bool(Vec<bool>),
    /// Array of UTF-8 strings.
    String(Vec<String>),
    /// Array of (typed) sub-arrays. Each sub-array self-describes its own
    /// inner type, so different elements may have different `GgufMetadataArray`
    /// variants.
    Array(Vec<GgufMetadataArray>),
    /// Array of unsigned 64-bit integers.
    U64(Vec<u64>),
    /// Array of signed 64-bit integers.
    I64(Vec<i64>),
    /// Array of 64-bit IEEE 754 floats.
    F64(Vec<f64>),
}

// Compile-time verification of the size invariants the parser relies on:
//
// * `GgufMetadataValue` stays at 24 bytes because boxing the `Array`
//   variant makes `String` (24 B) the largest payload instead of the old
//   unboxed `Vec<GgufMetadataValue>`. That 25 % shrink applies to every
//   metadata value in the `HashMap`, not just arrays.
//
// * `GgufMetadataArray` stays at 32 bytes because every `Vec<T>` variant
//   is 24 B plus an 8-byte (aligned) discriminant.
//
// If either number drifts, the DoS-guard memory math in the parser's
// module comments is stale and needs to be re-audited.
const _: () = {
    assert!(
        std::mem::size_of::<GgufMetadataValue>() == 24,
        "GgufMetadataValue must be 24 bytes (Array must be Box<GgufMetadataArray>)"
    );
    assert!(
        std::mem::size_of::<GgufMetadataArray>() == 32,
        "GgufMetadataArray must be 32 bytes (largest variant Vec<T> = 24 + 8-byte tag)"
    );
};

impl GgufMetadataValue {
    /// Returns the inner string if the value is `String`, otherwise `None`.
    #[must_use]
    pub fn as_string(&self) -> Option<&str> {
        if let Self::String(s) = self {
            // BORROW: explicit `.as_str()` on the owned `String` instead of
            // relying on `Deref<Target = str>` coercion through `Some(...)`
            Some(s.as_str())
        } else {
            None
        }
    }

    /// Returns the inner `u32` if the value is `U32`, otherwise `None`.
    #[must_use]
    pub const fn as_u32(&self) -> Option<u32> {
        if let Self::U32(v) = self {
            Some(*v)
        } else {
            None
        }
    }

    /// Returns the inner `u64` if the value is `U64`, otherwise `None`.
    #[must_use]
    pub const fn as_u64(&self) -> Option<u64> {
        if let Self::U64(v) = self {
            Some(*v)
        } else {
            None
        }
    }

    /// Returns the inner `bool` if the value is `Bool`, otherwise `None`.
    #[must_use]
    pub const fn as_bool(&self) -> Option<bool> {
        if let Self::Bool(v) = self {
            Some(*v)
        } else {
            None
        }
    }

    /// Returns the inner typed array if the value is `Array`, otherwise
    /// `None`.
    #[must_use]
    pub fn as_array(&self) -> Option<&GgufMetadataArray> {
        if let Self::Array(v) = self {
            Some(v.as_ref())
        } else {
            None
        }
    }
}

impl GgufMetadataArray {
    /// Number of elements in the array, regardless of inner type.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::U8(v) => v.len(),
            Self::I8(v) => v.len(),
            Self::U16(v) => v.len(),
            Self::I16(v) => v.len(),
            Self::U32(v) => v.len(),
            Self::I32(v) => v.len(),
            Self::F32(v) => v.len(),
            Self::Bool(v) => v.len(),
            Self::String(v) => v.len(),
            Self::Array(v) => v.len(),
            Self::U64(v) => v.len(),
            Self::I64(v) => v.len(),
            Self::F64(v) => v.len(),
        }
    }

    /// Returns `true` if the array contains no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the inner slice if this is a `U8` array, otherwise `None`.
    #[must_use]
    pub fn as_u8_slice(&self) -> Option<&[u8]> {
        if let Self::U8(v) = self {
            // BORROW: explicit `.as_slice()` avoids relying on `Deref` coercion
            Some(v.as_slice())
        } else {
            None
        }
    }

    /// Returns the inner slice if this is an `I8` array, otherwise `None`.
    #[must_use]
    pub fn as_i8_slice(&self) -> Option<&[i8]> {
        if let Self::I8(v) = self {
            // BORROW: explicit `.as_slice()` avoids relying on `Deref` coercion
            Some(v.as_slice())
        } else {
            None
        }
    }

    /// Returns the inner slice if this is a `U16` array, otherwise `None`.
    #[must_use]
    pub fn as_u16_slice(&self) -> Option<&[u16]> {
        if let Self::U16(v) = self {
            // BORROW: explicit `.as_slice()` avoids relying on `Deref` coercion
            Some(v.as_slice())
        } else {
            None
        }
    }

    /// Returns the inner slice if this is an `I16` array, otherwise `None`.
    #[must_use]
    pub fn as_i16_slice(&self) -> Option<&[i16]> {
        if let Self::I16(v) = self {
            // BORROW: explicit `.as_slice()` avoids relying on `Deref` coercion
            Some(v.as_slice())
        } else {
            None
        }
    }

    /// Returns the inner slice if this is a `U32` array, otherwise `None`.
    #[must_use]
    pub fn as_u32_slice(&self) -> Option<&[u32]> {
        if let Self::U32(v) = self {
            // BORROW: explicit `.as_slice()` avoids relying on `Deref` coercion
            Some(v.as_slice())
        } else {
            None
        }
    }

    /// Returns the inner slice if this is an `I32` array, otherwise `None`.
    #[must_use]
    pub fn as_i32_slice(&self) -> Option<&[i32]> {
        if let Self::I32(v) = self {
            // BORROW: explicit `.as_slice()` avoids relying on `Deref` coercion
            Some(v.as_slice())
        } else {
            None
        }
    }

    /// Returns the inner slice if this is an `F32` array, otherwise `None`.
    #[must_use]
    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        if let Self::F32(v) = self {
            // BORROW: explicit `.as_slice()` avoids relying on `Deref` coercion
            Some(v.as_slice())
        } else {
            None
        }
    }

    /// Returns the inner slice if this is a `Bool` array, otherwise `None`.
    #[must_use]
    pub fn as_bool_slice(&self) -> Option<&[bool]> {
        if let Self::Bool(v) = self {
            // BORROW: explicit `.as_slice()` avoids relying on `Deref` coercion
            Some(v.as_slice())
        } else {
            None
        }
    }

    /// Returns the inner slice if this is a `String` array, otherwise `None`.
    #[must_use]
    pub fn as_string_slice(&self) -> Option<&[String]> {
        if let Self::String(v) = self {
            // BORROW: explicit `.as_slice()` avoids relying on `Deref` coercion
            Some(v.as_slice())
        } else {
            None
        }
    }

    /// Returns the inner slice if this is an `Array` of sub-arrays,
    /// otherwise `None`.
    #[must_use]
    pub fn as_nested_slice(&self) -> Option<&[GgufMetadataArray]> {
        if let Self::Array(v) = self {
            // BORROW: explicit `.as_slice()` avoids relying on `Deref` coercion
            Some(v.as_slice())
        } else {
            None
        }
    }

    /// Returns the inner slice if this is a `U64` array, otherwise `None`.
    #[must_use]
    pub fn as_u64_slice(&self) -> Option<&[u64]> {
        if let Self::U64(v) = self {
            // BORROW: explicit `.as_slice()` avoids relying on `Deref` coercion
            Some(v.as_slice())
        } else {
            None
        }
    }

    /// Returns the inner slice if this is an `I64` array, otherwise `None`.
    #[must_use]
    pub fn as_i64_slice(&self) -> Option<&[i64]> {
        if let Self::I64(v) = self {
            // BORROW: explicit `.as_slice()` avoids relying on `Deref` coercion
            Some(v.as_slice())
        } else {
            None
        }
    }

    /// Returns the inner slice if this is an `F64` array, otherwise `None`.
    #[must_use]
    pub fn as_f64_slice(&self) -> Option<&[f64]> {
        if let Self::F64(v) = self {
            // BORROW: explicit `.as_slice()` avoids relying on `Deref` coercion
            Some(v.as_slice())
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// GgufTensorInfo
// ---------------------------------------------------------------------------

/// Metadata for a single tensor in a `GGUF` file.
///
/// Produced during [`parse_gguf`]. `data_offset` is the **absolute** byte
/// offset inside the memory-mapped file (not the relative offset stored in
/// the `gguf_tensor_info_t` on disk — the parser has already added the
/// tensor-data section start).
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    /// Tensor name (e.g., `"blk.0.attn_q.weight"`).
    pub name: String,
    /// Tensor dimensions, **most-significant-first** (same order as
    /// `ggml_tensor::ne`). A row-major `[rows, cols]` matrix is stored as
    /// `shape = [cols, rows]` — consumers that expect NumPy-style ordering
    /// must reverse this before use.
    pub shape: Vec<usize>,
    /// Element / block data type.
    pub dtype: GgufType,
    /// Absolute byte offset of the tensor data inside the memory-mapped
    /// file. Equal to `tensor_data_section_start + relative_offset` where
    /// `relative_offset` is the `u64` stored in the file.
    pub data_offset: u64,
    /// Total byte length of the tensor data, or `None` when
    /// [`GgufType::type_size`] is not yet tabulated for this dtype.
    pub byte_len: Option<u64>,
}

// ---------------------------------------------------------------------------
// GgufTensor
// ---------------------------------------------------------------------------

/// A tensor view into a parsed `GGUF` file.
///
/// Returned by [`ParsedGguf::tensors`]. `name` and `shape` borrow directly
/// from the owning [`ParsedGguf`] — iterating all tensors allocates
/// nothing. `data` is `Cow::Borrowed` with a zero-copy slice of the
/// memory-mapped file for every supported dtype; `Cow::Owned` is reserved
/// for future big-endian support.
///
/// **FFI / Python boundary:** `name`, `shape`, and `data` borrow the owning
/// [`ParsedGguf`] for `'a` (`dtype` is owned). A binding that hands the bytes to
/// another runtime (e.g. a `NumPy` array) must take ownership first —
/// `data.into_owned()` plus `name.to_owned()` / `shape.to_vec()` — so the array
/// never aliases bytes the owning [`ParsedGguf`] can drop. See
/// `docs/python-interop.md` (ownership contract).
#[derive(Debug, Clone)]
pub struct GgufTensor<'a> {
    /// Tensor name (e.g., `"blk.0.attn_q.weight"`).
    pub name: &'a str,
    /// Tensor dimensions, most-significant-first (see
    /// [`GgufTensorInfo::shape`]).
    pub shape: &'a [usize],
    /// Element / block data type.
    pub dtype: GgufType,
    /// Raw bytes in on-disk (little-endian) order. Length equals
    /// `byte_size_for_n_elements(product(shape))`.
    pub data: Cow<'a, [u8]>,
}

// ---------------------------------------------------------------------------
// GgufInspectInfo
// ---------------------------------------------------------------------------

/// Summary information about a parsed `GGUF` file.
///
/// Produced by [`ParsedGguf::inspect`]. No I/O — derived from metadata.
#[derive(Debug, Clone)]
#[must_use]
pub struct GgufInspectInfo {
    /// `GGUF` version read from the header (currently 2 or 3).
    pub version: u32,
    /// Value of the `general.architecture` metadata key, if present.
    pub architecture: Option<String>,
    /// Number of tensors in the file.
    pub tensor_count: usize,
    /// Total byte length of all tensor data whose dtype has a known
    /// `type_size`. Tensors with an unknown dtype are excluded.
    pub total_bytes: u64,
    /// Number of tensors whose dtype has no known byte size (excluded from
    /// `total_bytes`).
    pub unknown_size_tensors: usize,
    /// Distinct dtypes found, in order of first occurrence.
    pub dtypes: Vec<GgufType>,
    /// Effective alignment read from `general.alignment`, or the default of
    /// 32 bytes if the metadata key is absent.
    pub alignment: u32,
}

impl fmt::Display for GgufInspectInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // All labels land at column 13 so successive lines line up visually.
        // `"Format:      "` is 13 chars, `"Arch:        "` is 13 chars, etc.
        write!(f, "Format:      GGUF v{}", self.version)?;
        if let Some(arch) = self.architecture.as_deref() {
            write!(f, "\nArch:        {arch}")?;
        }
        write!(f, "\nTensors:     {}", self.tensor_count)?;
        write!(
            f,
            "\nTotal size:  {}",
            crate::inspect::format_bytes(self.total_bytes)
        )?;
        if self.unknown_size_tensors > 0 {
            write!(
                f,
                " (+{} tensors with dtype of unknown size)",
                self.unknown_size_tensors
            )?;
        }
        let dtype_list: String = self
            .dtypes
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        write!(f, "\nDtypes:      {dtype_list}")?;
        write!(f, "\nAlignment:   {} bytes", self.alignment)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ParsedGguf
// ---------------------------------------------------------------------------

/// A parsed `GGUF` file — owns the memory-mapped data and provides
/// zero-copy tensor views.
///
/// Created by [`parse_gguf`] (memory-mapped) or by [`parse_gguf_bytes`] /
/// [`parse_gguf_from_reader`] (owned copy). Call [`tensors`](Self::tensors) to
/// obtain [`GgufTensor`] views borrowed directly from the backing bytes.
#[derive(Debug)]
pub struct ParsedGguf {
    /// File bytes — a memory map (path-based [`parse_gguf`]) or an owned
    /// `Vec<u8>` (copy-based [`parse_gguf_bytes`] / [`parse_gguf_from_reader`]).
    buffer: Backing,
    /// `GGUF` version read from the header.
    version: u32,
    /// Effective tensor-data alignment in bytes.
    alignment: u32,
    /// Metadata key-value pairs keyed by the full `GGUF` key. `HashMap`, so
    /// iteration order is unspecified — not the file's key order.
    metadata: HashMap<String, GgufMetadataValue>,
    /// Per-tensor metadata with absolute byte offsets.
    tensor_infos: Vec<GgufTensorInfo>,
}

impl ParsedGguf {
    /// Returns the `GGUF` format version read from the header.
    #[must_use]
    pub const fn version(&self) -> u32 {
        self.version
    }

    /// Returns the effective tensor-data alignment.
    #[must_use]
    pub const fn alignment(&self) -> u32 {
        self.alignment
    }

    /// Returns the number of tensors in the file.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.tensor_infos.len()
    }

    /// Returns `true` if the file contains no tensors.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.tensor_infos.is_empty()
    }

    /// Returns the parsed metadata key-value table.
    #[must_use]
    pub const fn metadata(&self) -> &HashMap<String, GgufMetadataValue> {
        &self.metadata
    }

    /// Returns lightweight per-tensor metadata without slicing the mmap.
    ///
    /// Use this for display paths and type inventories where the raw bytes
    /// are not needed.
    #[must_use]
    pub fn tensor_info(&self) -> &[GgufTensorInfo] {
        &self.tensor_infos
    }

    /// Returns an iterator of tensor views borrowing directly from the
    /// mmap and from `self`.
    ///
    /// For every tensor `data` is a zero-copy `Cow::Borrowed` slice into
    /// the mapped file. `name` and `shape` are `&'a str` / `&'a [usize]`
    /// borrowed from the internal `Vec<GgufTensorInfo>` — **no per-tensor
    /// heap allocation**. Every recognised `GgufType` has a known byte
    /// size, so this iterator yields every tensor in the file (no silent
    /// skipping); for the inventory-only view that does not borrow data,
    /// see [`tensor_info`](Self::tensor_info).
    ///
    /// Callers that want random access can materialise the iterator with
    /// `.collect::<Vec<_>>()`.
    ///
    /// # Memory
    ///
    /// Zero heap allocation per invocation. `GgufTensor::data` is
    /// `Cow::Borrowed` into the mmap, and `GgufTensor::{name, shape}` are
    /// slice references into `self`. Peak memory is just the mmap itself
    /// (unchanged across `tensors()` calls) plus whatever the caller
    /// chooses to collect.
    pub fn tensors(&self) -> impl Iterator<Item = GgufTensor<'_>> + '_ {
        self.tensor_infos.iter().filter_map(|info| {
            let byte_len_u64 = info.byte_len?;
            // `usize::try_from` and `checked_add` are defensive:
            // `data_offset` and `byte_len` were pre-validated against
            // `raw.len()` in `parse_gguf`, so on 64-bit targets every
            // `?` below is dead. On 32-bit targets with a hypothetical
            // >4 GB mmap (which `memmap2` cannot produce) the
            // fallthrough silently skips the tensor.
            let start = usize::try_from(info.data_offset).ok()?;
            let byte_len = usize::try_from(byte_len_u64).ok()?;
            let end = start.checked_add(byte_len)?;
            let slice = self.buffer.get(start..end)?;
            Some(GgufTensor {
                name: info.name.as_str(),
                shape: info.shape.as_slice(),
                dtype: info.dtype,
                data: Cow::Borrowed(slice),
            })
        })
    }

    /// Returns inspection info derived from the parsed metadata. No I/O.
    pub fn inspect(&self) -> GgufInspectInfo {
        build_inspect_info(
            self.version,
            self.alignment,
            &self.metadata,
            &self.tensor_infos,
        )
    }

    /// Dequantises a single tensor from the memory-mapped file to `BF16`
    /// bytes.
    ///
    /// Convenience method that slices the internal mmap using `info`'s
    /// offset and byte length, infers the element count from `info.shape`,
    /// and delegates to
    /// [`dequantize_gguf_to_bf16`](crate::remember::gguf::dequantize_gguf_to_bf16).
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::Unsupported`] if the dtype is a
    /// recognised scalar (non-block-quant) type for which dequantisation
    /// is structurally meaningless — every block-quantised `GgufType`
    /// has a dedicated kernel after Phase 4.5 step 6.
    ///
    /// Returns [`AnamnesisError::Parse`] if the element count overflows
    /// `usize`, the mmap slice is out of bounds, or the underlying
    /// dequantisation kernel encounters a data/shape mismatch.
    ///
    /// # Memory
    ///
    /// Allocates a single `Vec<u8>` of length `n_elements * 2` for the
    /// `BF16` output. The input data is read directly from the mmap — no
    /// input copy. Peak heap is the output buffer (O(`n_elements`)).
    pub fn dequantize_tensor(&self, info: &GgufTensorInfo) -> crate::Result<Vec<u8>> {
        let byte_len_u64 = info.byte_len.ok_or_else(|| AnamnesisError::Unsupported {
            format: "GGUF".into(),
            detail: format!(
                "byte size not known for dtype {} — dequantisation not yet supported",
                info.dtype
            ),
        })?;
        let start = usize::try_from(info.data_offset).map_err(|_| AnamnesisError::Parse {
            reason: format!(
                "tensor `{}`: data_offset {} exceeds usize",
                info.name, info.data_offset
            ),
        })?;
        let byte_len = usize::try_from(byte_len_u64).map_err(|_| AnamnesisError::Parse {
            reason: format!(
                "tensor `{}`: byte_len {byte_len_u64} exceeds usize",
                info.name
            ),
        })?;
        let end = start
            .checked_add(byte_len)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!(
                    "tensor `{}`: data_offset + byte_len overflows usize",
                    info.name
                ),
            })?;
        let data = self
            .buffer
            .get(start..end)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!(
                    "tensor `{}`: byte range {start}..{end} exceeds backing length {}",
                    info.name,
                    self.buffer.len()
                ),
            })?;
        let n_elements: usize = info
            .shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!("tensor `{}`: element count overflows usize", info.name),
            })?;
        crate::remember::gguf::dequantize_gguf_to_bf16(data, info.dtype, n_elements)
    }
}

// ---------------------------------------------------------------------------
// GgufReader — generic little-endian reader over any `Read + Seek` source
// ---------------------------------------------------------------------------

/// Generic forward-tracking reader over any `Read + Seek` substrate, with
/// bounds-checked little-endian primitive readers.
///
/// The parser was historically slice-based (`Cursor<&[u8]>` over a
/// `memmap2::Mmap`). Generalising to `Read + Seek` keeps the path-based
/// `parse_gguf` unchanged (it wraps the mmap in a `std::io::Cursor`) while
/// letting the inspect-only entry point [`inspect_gguf_from_reader`] accept
/// any positional source — in-memory cursors, HTTP-range adapters, custom
/// transports — without re-deriving the parser logic.
///
/// `pos` is tracked redundantly with the underlying reader's stream position
/// so that error messages can report the byte offset and so that the
/// post-tensor-info `tensor_info_end` is a `u64` available without an extra
/// `seek(SeekFrom::Current(0))` round-trip.
///
/// `file_len` is captured once at construction by seeking to the end of the
/// stream, then the reader is repositioned at offset 0. An HTTP-range
/// adapter that knows the total content length can answer this without a
/// data-section fetch.
struct GgufReader<R: Read + Seek> {
    reader: R,
    pos: u64,
    file_len: u64,
    /// Caller-supplied allocation budget — the per-item single-allocation cap
    /// and the cumulative parse-time-heap aggregate, charged on each owned read.
    budget: Budget,
}

impl<R: Read + Seek> GgufReader<R> {
    /// Constructs a reader anchored at offset 0, capturing `file_len` for
    /// the bounds-check error messages and the post-tensor-info data-section
    /// arithmetic. `limits` is the caller's allocation budget, enforced on
    /// every owned (variable-length) read and scalar metadata array.
    fn new(mut reader: R, limits: &ParseLimits) -> crate::Result<Self> {
        let file_len = reader.seek(SeekFrom::End(0)).map_err(AnamnesisError::Io)?;
        reader
            .seek(SeekFrom::Start(0))
            .map_err(AnamnesisError::Io)?;
        Ok(Self {
            reader,
            pos: 0,
            file_len,
            budget: Budget::new(limits),
        })
    }

    /// Validates that `n` more bytes are available from the current position
    /// without running past `file_len`, returning the resulting end offset.
    ///
    /// Pulled out of [`read_into`](Self::read_into) so it can also gate
    /// [`read_bytes`](Self::read_bytes) **before** it allocates — an
    /// adversarial declared length is rejected without committing any heap,
    /// producing a deterministic `AnamnesisError::Parse` (matching the
    /// slice-based cursor's behaviour) rather than relying on the underlying
    /// reader's `UnexpectedEof` kind-mapping.
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::Parse`] if `self.pos + n` overflows `u64` or
    /// exceeds `file_len`.
    fn ensure_remaining(&self, n: u64) -> crate::Result<u64> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!("GGUF: cursor overflow at pos {} + {n}", self.pos),
            })?;
        if end > self.file_len {
            return Err(AnamnesisError::Parse {
                reason: format!(
                    "GGUF: unexpected EOF at pos {} (wanted {n} bytes, have {})",
                    self.pos,
                    self.file_len.saturating_sub(self.pos)
                ),
            });
        }
        Ok(end)
    }

    /// Reads exactly `buf.len()` bytes into `buf`, advancing the cursor.
    ///
    /// Validates the request via [`ensure_remaining`](Self::ensure_remaining)
    /// before reading, so an adversarial header claiming a length past EOF
    /// produces a deterministic `AnamnesisError::Parse`.
    fn read_into(&mut self, buf: &mut [u8]) -> crate::Result<()> {
        // CAST: usize → u64, length of a borrowed buffer always fits in u64
        // on every supported target (64-bit and 32-bit).
        #[allow(clippy::as_conversions)]
        let n_u64 = buf.len() as u64;
        let end = self.ensure_remaining(n_u64)?;
        self.reader.read_exact(buf).map_err(AnamnesisError::Io)?;
        self.pos = end;
        Ok(())
    }

    /// Reads exactly `n` bytes, returning an owned `Vec<u8>`.
    ///
    /// Used for variable-length payloads (e.g. `gguf_string_t`). Fixed-size
    /// primitives use [`read_into`](Self::read_into) with a stack array.
    ///
    /// The declared length is validated against the remaining file bytes
    /// **before** the allocation, so a tiny file that declares a length up to
    /// the type cap (e.g. `MAX_STRING_LEN` = 16 MiB) cannot drive an eager
    /// `vec!` — it is rejected first. [`read_into`](Self::read_into) re-checks
    /// (a cheap confirm) for the stack-buffer callers that bypass this path.
    fn read_bytes(&mut self, n: usize) -> crate::Result<Vec<u8>> {
        // CAST: usize → u64, lossless widening on all supported targets.
        #[allow(clippy::as_conversions)]
        let n_u64 = n as u64;
        self.ensure_remaining(n_u64)?;
        // Caller-supplied ceiling (per-item single-alloc + cumulative aggregate),
        // layered on top of the type cap the caller (e.g. `read_string` with
        // `MAX_STRING_LEN`) already enforced.
        self.budget
            .charge_alloc(n_u64, "GGUF variable-length read")?;
        let mut buf = vec![0u8; n];
        self.read_into(&mut buf)?;
        Ok(buf)
    }

    fn read_u8(&mut self) -> crate::Result<u8> {
        let mut arr = [0u8; 1];
        self.read_into(&mut arr)?;
        // INDEX: stack array of length exactly 1 — `arr[0]` cannot be OOB
        #[allow(clippy::indexing_slicing)]
        Ok(arr[0])
    }

    fn read_i8(&mut self) -> crate::Result<i8> {
        // CAST: u8 → i8, reinterpret bit pattern — signed/unsigned wrap is intended
        #[allow(clippy::as_conversions, clippy::cast_possible_wrap)]
        Ok(self.read_u8()? as i8)
    }

    fn read_u16_le(&mut self) -> crate::Result<u16> {
        let mut arr = [0u8; 2];
        self.read_into(&mut arr)?;
        Ok(u16::from_le_bytes(arr))
    }

    fn read_i16_le(&mut self) -> crate::Result<i16> {
        let mut arr = [0u8; 2];
        self.read_into(&mut arr)?;
        Ok(i16::from_le_bytes(arr))
    }

    fn read_u32_le(&mut self) -> crate::Result<u32> {
        let mut arr = [0u8; 4];
        self.read_into(&mut arr)?;
        Ok(u32::from_le_bytes(arr))
    }

    fn read_i32_le(&mut self) -> crate::Result<i32> {
        let mut arr = [0u8; 4];
        self.read_into(&mut arr)?;
        Ok(i32::from_le_bytes(arr))
    }

    fn read_u64_le(&mut self) -> crate::Result<u64> {
        let mut arr = [0u8; 8];
        self.read_into(&mut arr)?;
        Ok(u64::from_le_bytes(arr))
    }

    fn read_i64_le(&mut self) -> crate::Result<i64> {
        let mut arr = [0u8; 8];
        self.read_into(&mut arr)?;
        Ok(i64::from_le_bytes(arr))
    }

    fn read_f32_le(&mut self) -> crate::Result<f32> {
        let mut arr = [0u8; 4];
        self.read_into(&mut arr)?;
        Ok(f32::from_le_bytes(arr))
    }

    fn read_f64_le(&mut self) -> crate::Result<f64> {
        let mut arr = [0u8; 8];
        self.read_into(&mut arr)?;
        Ok(f64::from_le_bytes(arr))
    }

    fn read_bool(&mut self) -> crate::Result<bool> {
        let b = self.read_u8()?;
        match b {
            0 => Ok(false),
            1 => Ok(true),
            other => Err(AnamnesisError::Parse {
                reason: format!("GGUF metadata: invalid bool byte {other} (expected 0 or 1)"),
            }),
        }
    }

    /// Reads a `gguf_string_t` (`u64` length prefix + raw bytes, interpreted
    /// as UTF-8).
    ///
    /// `max_len` is the applicable cap (varies by call site: `MAX_STRING_LEN`
    /// for metadata string values, `MAX_GGUF_KEY_LEN` for metadata keys,
    /// `MAX_TENSOR_NAME_LEN` for tensor names) and `limit` is that cap's name,
    /// used as the [`AnamnesisError::LimitExceeded`] tag on rejection.
    ///
    /// The cap check runs against the declared length **before** allocating,
    /// so a rejected oversize header costs zero heap. UTF-8 validation
    /// consumes the temporary `Vec<u8>` via `String::from_utf8`, which
    /// either re-uses the buffer in place (success) or hands its bytes back
    /// to the caller via `FromUtf8Error::into_bytes` (error) — no double
    /// allocation in either branch.
    fn read_string(&mut self, max_len: u64, limit: &'static str) -> crate::Result<String> {
        let len = self.read_u64_le()?;
        if len > max_len {
            return Err(AnamnesisError::LimitExceeded {
                limit,
                message: format!("GGUF: string length {len} exceeds cap {max_len}"),
            });
        }
        let len_usz = usize::try_from(len).map_err(|_| AnamnesisError::Parse {
            reason: format!("GGUF: string length {len} overflows usize"),
        })?;
        let bytes = self.read_bytes(len_usz)?;
        // `String::from_utf8` consumes `bytes` and re-uses its buffer on
        // success, so the validated bytes never get copied into a second
        // allocation.
        String::from_utf8(bytes).map_err(|e| AnamnesisError::Parse {
            reason: format!("GGUF: string is not valid UTF-8: {e}"),
        })
    }
}

// ---------------------------------------------------------------------------
// Metadata value reader
// ---------------------------------------------------------------------------

/// Reads a single metadata value of the given `value_type` discriminant.
///
/// For `ARRAY` (`value_type` 9), dispatches into [`read_typed_array`]
/// which builds a natively-typed `Vec<T>` instead of the old 8×-bloated
/// `Vec<GgufMetadataValue>`.
fn read_metadata_value<R: Read + Seek>(
    cursor: &mut GgufReader<R>,
    value_type: u32,
) -> crate::Result<GgufMetadataValue> {
    match value_type {
        0 => Ok(GgufMetadataValue::U8(cursor.read_u8()?)),
        1 => Ok(GgufMetadataValue::I8(cursor.read_i8()?)),
        2 => Ok(GgufMetadataValue::U16(cursor.read_u16_le()?)),
        3 => Ok(GgufMetadataValue::I16(cursor.read_i16_le()?)),
        4 => Ok(GgufMetadataValue::U32(cursor.read_u32_le()?)),
        5 => Ok(GgufMetadataValue::I32(cursor.read_i32_le()?)),
        6 => Ok(GgufMetadataValue::F32(cursor.read_f32_le()?)),
        7 => Ok(GgufMetadataValue::Bool(cursor.read_bool()?)),
        8 => Ok(GgufMetadataValue::String(
            cursor.read_string(MAX_STRING_LEN, "MAX_STRING_LEN")?,
        )),
        9 => {
            let inner_type = cursor.read_u32_le()?;
            let len = read_array_len(cursor)?;
            // Initial depth is 0: this call builds the outer array (nesting
            // level 0). Recursive calls increment the depth so that the
            // `depth >= MAX_ARRAY_DEPTH` check inside `read_typed_array`
            // supports `MAX_ARRAY_DEPTH` total nested levels (depths
            // `0..MAX_ARRAY_DEPTH`).
            let array = read_typed_array(cursor, inner_type, len, 0)?;
            Ok(GgufMetadataValue::Array(Box::new(array)))
        }
        10 => Ok(GgufMetadataValue::U64(cursor.read_u64_le()?)),
        11 => Ok(GgufMetadataValue::I64(cursor.read_i64_le()?)),
        12 => Ok(GgufMetadataValue::F64(cursor.read_f64_le()?)),
        other => Err(AnamnesisError::Parse {
            reason: format!("GGUF metadata: unknown value type {other}"),
        }),
    }
}

/// Reads and validates a `GGUF` array length prefix (`u64` from the file).
///
/// Enforces `MAX_ARRAY_LEN` and converts to `usize`.
fn read_array_len<R: Read + Seek>(cursor: &mut GgufReader<R>) -> crate::Result<usize> {
    let len = cursor.read_u64_le()?;
    if len > MAX_ARRAY_LEN {
        return Err(AnamnesisError::LimitExceeded {
            limit: "MAX_ARRAY_LEN",
            message: format!("GGUF metadata: array length {len} exceeds cap {MAX_ARRAY_LEN}"),
        });
    }
    usize::try_from(len).map_err(|_| AnamnesisError::Parse {
        reason: format!("GGUF metadata: array length {len} overflows usize"),
    })
}

/// Reads `len` homogeneous elements of type `inner_type` into a typed
/// [`GgufMetadataArray`].
///
/// `depth` is the current array-nesting level for adversarial-depth
/// protection: `depth = 0` is the outermost array built directly from a
/// metadata key-value pair, and nested arrays increment it. Recursion is
/// rejected when `depth >= MAX_ARRAY_DEPTH`, so the parser supports
/// exactly `MAX_ARRAY_DEPTH` total levels of nesting (depths
/// `0..MAX_ARRAY_DEPTH`).
///
/// Every typed `Vec::with_capacity` call is clamped to
/// `PREALLOC_SOFT_CAP` so an adversarial 20-byte array header claiming
/// `MAX_ARRAY_LEN` elements cannot force ~488 MB of eager allocation;
/// the vector grows geometrically from there.
/// Byte size of a scalar `GGUF` metadata-array element type, or `None` for the
/// non-scalar `STRING` (8) and `ARRAY` (9) types (and unknown discriminants),
/// whose bytes are charged downstream rather than in bulk.
const fn scalar_array_elem_size(inner_type: u32) -> Option<u64> {
    match inner_type {
        // 0=u8, 1=i8, 7=bool
        0 | 1 | 7 => Some(1),
        // 2=u16, 3=i16
        2 | 3 => Some(2),
        // 4=u32, 5=i32, 6=f32
        4..=6 => Some(4),
        // 10=u64, 11=i64, 12=f64
        10..=12 => Some(8),
        // 8=string, 9=array, or unknown → charged downstream / rejected later
        _ => None,
    }
}

fn read_typed_array<R: Read + Seek>(
    cursor: &mut GgufReader<R>,
    inner_type: u32,
    len: usize,
    depth: u32,
) -> crate::Result<GgufMetadataArray> {
    // Charge the scalar array's full heap size to the caller's budget before
    // growing the `Vec` — bounding both a single oversized array and the
    // cumulative parse-time heap. `STRING` (8) and `ARRAY` (9) elements are
    // charged downstream (per-string `read_bytes` / per-element recursion).
    if let Some(elem_size) = scalar_array_elem_size(inner_type) {
        // CAST: usize → u64, lossless widening on all supported targets
        #[allow(clippy::as_conversions)]
        let len_u64 = len as u64;
        let bytes = len_u64
            .checked_mul(elem_size)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!("GGUF metadata array byte size overflow (len {len} × {elem_size})"),
            })?;
        // Validate the declared array size against the remaining file bytes
        // before charging or growing — same order as `read_bytes`, so a tiny
        // file declaring a huge array fails fast instead of looping to EOF.
        cursor.ensure_remaining(bytes)?;
        cursor.budget.charge_alloc(bytes, "GGUF metadata array")?;
    }

    let cap = len.min(PREALLOC_SOFT_CAP);
    match inner_type {
        0 => {
            let mut v: Vec<u8> = Vec::with_capacity(cap);
            for _ in 0..len {
                v.push(cursor.read_u8()?);
            }
            Ok(GgufMetadataArray::U8(v))
        }
        1 => {
            let mut v: Vec<i8> = Vec::with_capacity(cap);
            for _ in 0..len {
                v.push(cursor.read_i8()?);
            }
            Ok(GgufMetadataArray::I8(v))
        }
        2 => {
            let mut v: Vec<u16> = Vec::with_capacity(cap);
            for _ in 0..len {
                v.push(cursor.read_u16_le()?);
            }
            Ok(GgufMetadataArray::U16(v))
        }
        3 => {
            let mut v: Vec<i16> = Vec::with_capacity(cap);
            for _ in 0..len {
                v.push(cursor.read_i16_le()?);
            }
            Ok(GgufMetadataArray::I16(v))
        }
        4 => {
            let mut v: Vec<u32> = Vec::with_capacity(cap);
            for _ in 0..len {
                v.push(cursor.read_u32_le()?);
            }
            Ok(GgufMetadataArray::U32(v))
        }
        5 => {
            let mut v: Vec<i32> = Vec::with_capacity(cap);
            for _ in 0..len {
                v.push(cursor.read_i32_le()?);
            }
            Ok(GgufMetadataArray::I32(v))
        }
        6 => {
            let mut v: Vec<f32> = Vec::with_capacity(cap);
            for _ in 0..len {
                v.push(cursor.read_f32_le()?);
            }
            Ok(GgufMetadataArray::F32(v))
        }
        7 => {
            let mut v: Vec<bool> = Vec::with_capacity(cap);
            for _ in 0..len {
                v.push(cursor.read_bool()?);
            }
            Ok(GgufMetadataArray::Bool(v))
        }
        8 => {
            let mut v: Vec<String> = Vec::with_capacity(cap);
            for _ in 0..len {
                v.push(cursor.read_string(MAX_STRING_LEN, "MAX_STRING_LEN")?);
            }
            Ok(GgufMetadataArray::String(v))
        }
        9 => {
            // Nested array: each element is itself a typed array. Check
            // the recursion depth before reading anything so we fail fast
            // on adversarial nesting.
            if depth >= MAX_ARRAY_DEPTH {
                return Err(AnamnesisError::LimitExceeded {
                    limit: "MAX_ARRAY_DEPTH",
                    message: format!(
                        "GGUF metadata: array nesting exceeds depth cap {MAX_ARRAY_DEPTH}"
                    ),
                });
            }
            let mut v: Vec<GgufMetadataArray> = Vec::with_capacity(cap);
            for _ in 0..len {
                let sub_inner = cursor.read_u32_le()?;
                let sub_len = read_array_len(cursor)?;
                v.push(read_typed_array(cursor, sub_inner, sub_len, depth + 1)?);
            }
            Ok(GgufMetadataArray::Array(v))
        }
        10 => {
            let mut v: Vec<u64> = Vec::with_capacity(cap);
            for _ in 0..len {
                v.push(cursor.read_u64_le()?);
            }
            Ok(GgufMetadataArray::U64(v))
        }
        11 => {
            let mut v: Vec<i64> = Vec::with_capacity(cap);
            for _ in 0..len {
                v.push(cursor.read_i64_le()?);
            }
            Ok(GgufMetadataArray::I64(v))
        }
        12 => {
            let mut v: Vec<f64> = Vec::with_capacity(cap);
            for _ in 0..len {
                v.push(cursor.read_f64_le()?);
            }
            Ok(GgufMetadataArray::F64(v))
        }
        other => Err(AnamnesisError::Parse {
            reason: format!("GGUF metadata: unknown array inner type {other}"),
        }),
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Parses a `GGUF` file and returns a [`ParsedGguf`] handle owning the
/// memory-mapped data.
///
/// # Errors
///
/// Returns [`AnamnesisError::Io`] if the file cannot be opened or mapped.
///
/// Returns [`AnamnesisError::Parse`] if the magic bytes are missing, the
/// header fields are truncated, the metadata table contains an invalid value
/// type, a tensor info entry is malformed, or a tensor's resolved byte range
/// falls outside the mapped file.
///
/// Returns [`AnamnesisError::LimitExceeded`] if a declared string/array length,
/// metadata-KV or tensor count, dimension, or element count exceeds a permanent
/// `GGUF` cap (always-on).
///
/// Returns [`AnamnesisError::Unsupported`] for `GGUF` v1 files (which use
/// `u32` string lengths instead of `u64`), big-endian `GGUF` files (v3+
/// feature, not yet implemented), legacy pre-`GGUF` formats (`GGML`, `GGJT`,
/// `GGMF`), and tensor dtypes whose `ggml_type` discriminant is not
/// recognised.
///
/// # Memory
///
/// Memory-maps the file with `memmap2::MmapOptions::populate()` to prefault
/// pages. Tensor data is **not** copied during parsing —
/// [`ParsedGguf::tensors`] returns `Cow::Borrowed` slices of the mmap.
/// Peak heap is `O(n_tensors + n_metadata_kv)` (a few dozen bytes per
/// tensor info record plus the metadata map). The mmap is released when
/// the returned `ParsedGguf` is dropped.
pub fn parse_gguf(path: impl AsRef<Path>) -> crate::Result<ParsedGguf> {
    parse_gguf_with_limits(path, &ParseLimits::default())
}

/// Parses a `GGUF` file under a caller-supplied [`ParseLimits`] budget.
///
/// Identical to [`parse_gguf`] but enforces the caller's [`ParseLimits`] —
/// fail-fast, before allocation — over the declared tensor and metadata-KV
/// counts (the item-count cap) and each variable-length read (strings, tensor
/// names) and scalar metadata array (the per-allocation cap **and** the
/// cumulative-byte aggregate). The built-in `MAX_*` caps still apply; `limits`
/// can only tighten them. [`parse_gguf`] is the `ParseLimits::default()`
/// (unbounded) special case.
///
/// # Errors
///
/// Returns [`AnamnesisError::Io`] if the file cannot be opened or mapped.
///
/// Returns [`AnamnesisError::LimitExceeded`] if a declared count or allocation
/// exceeds `limits` or a permanent `GGUF` cap.
/// Returns [`AnamnesisError::Parse`] if the magic bytes are missing, the
/// header fields are truncated or out of range, the metadata table contains
/// an invalid value type, a tensor info entry is malformed, or a tensor's
/// resolved byte range falls outside the mapped file.
///
/// Returns [`AnamnesisError::Unsupported`] for `GGUF` v1 files, big-endian
/// `GGUF` files, legacy pre-`GGUF` formats, and unrecognised tensor dtypes.
///
/// # Memory
///
/// Memory-maps the file with `memmap2::MmapOptions::populate()` to prefault
/// pages. Tensor data is **not** copied during parsing. Peak heap is
/// `O(n_tensors + n_metadata_kv)`. The mmap is released when the returned
/// `ParsedGguf` is dropped.
#[allow(unsafe_code)]
pub fn parse_gguf_with_limits(
    path: impl AsRef<Path>,
    limits: &ParseLimits,
) -> crate::Result<ParsedGguf> {
    let file = std::fs::File::open(path.as_ref())?;
    // SAFETY: memmap2::Mmap requires `unsafe` because the OS could modify
    // the mapped region if another process writes to the underlying file
    // concurrently. Tensor files are read-only artefacts in practice — the
    // same assumption every other anamnesis format parser (pth, safetensors
    // via the `safetensors` crate) relies on. Untrusted callers that cannot
    // make that assumption use `parse_gguf_bytes` / `parse_gguf_from_reader`
    // instead (no mmap, no `SIGBUS`).
    let raw =
        unsafe { memmap2::MmapOptions::new().populate().map(&file) }.map_err(AnamnesisError::Io)?;
    parsed_gguf_from_backing(Backing::Mmap(raw), limits)
}

/// Builds a [`ParsedGguf`] from an already-acquired byte backing — the single
/// construction site shared by the mmap path ([`parse_gguf_with_limits`]) and
/// the copy-based paths ([`parse_gguf_bytes_with_limits`] /
/// [`parse_gguf_from_reader_with_limits`]). The structure is read over a
/// `Cursor` so the same `Read + Seek` core ([`read_gguf_structure`]) serves
/// every path; the backing is retained so [`ParsedGguf::tensors`] can hand out
/// zero-copy `Cow::Borrowed` slices into it.
fn parsed_gguf_from_backing(buffer: Backing, limits: &ParseLimits) -> crate::Result<ParsedGguf> {
    let front = read_gguf_structure(Cursor::new(&buffer[..]), limits)?;
    Ok(ParsedGguf {
        buffer,
        version: front.version,
        alignment: front.alignment,
        metadata: front.metadata,
        tensor_infos: front.tensor_infos,
    })
}

/// Parses `GGUF` bytes already held in memory, returning a [`ParsedGguf`] that
/// **owns** them — the copy-based, mmap-free path.
///
/// **Recommended for untrusted input**: no mmap, so a truncated or hostile
/// source is a clean `Err`, never a `SIGBUS`. [`parse_gguf_bytes`] is the
/// [`ParseLimits::default`] (unbounded) special case of
/// [`parse_gguf_bytes_with_limits`].
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] on the malformed-input conditions of
/// [`parse_gguf_with_limits`]; [`AnamnesisError::LimitExceeded`] if a declared
/// length, count, or dimension exceeds a permanent cap (always-on — reachable
/// even at the default limits this wrapper passes); or
/// [`AnamnesisError::Unsupported`] for an unsupported `GGUF` variant.
///
/// # Memory
///
/// Takes ownership of `bytes` (no copy); peak heap is the input size plus
/// `O(n_tensors + n_metadata_kv)`.
pub fn parse_gguf_bytes(bytes: Vec<u8>) -> crate::Result<ParsedGguf> {
    parse_gguf_bytes_with_limits(bytes, &ParseLimits::default())
}

/// Parses owned `GGUF` bytes under a caller-supplied [`ParseLimits`] budget —
/// the bounded, mmap-free path for untrusted input.
///
/// Rejects an input larger than [`ParseLimits::max_single_alloc_bytes`] before
/// parsing, then enforces every applicable `limits` ceiling exactly as
/// [`parse_gguf_with_limits`] does.
///
/// # Errors
///
/// Returns [`AnamnesisError::LimitExceeded`] if `bytes` exceeds `limits`.
/// Returns [`AnamnesisError::Parse`] on the malformed-input conditions of
/// [`parse_gguf_with_limits`].
/// Returns [`AnamnesisError::Unsupported`] for an unsupported `GGUF` variant.
///
/// # Memory
///
/// Takes ownership of `bytes` (no copy); peak heap is the input size plus
/// `O(n_tensors + n_metadata_kv)`.
pub fn parse_gguf_bytes_with_limits(
    bytes: Vec<u8>,
    limits: &ParseLimits,
) -> crate::Result<ParsedGguf> {
    let len = u64::try_from(bytes.len()).map_err(|_| AnamnesisError::Parse {
        reason: "GGUF bytes: length overflows u64".into(),
    })?;
    limits.check_alloc(len, "GGUF bytes")?;
    parsed_gguf_from_backing(Backing::Owned(bytes), limits)
}

/// Parses a `GGUF` artefact from any reader, returning a [`ParsedGguf`] that
/// **owns** the bytes — the copy-based, mmap-free path.
///
/// **Recommended for untrusted streamed input.** [`parse_gguf_from_reader`] is
/// the [`ParseLimits::default`] (unbounded) special case of
/// [`parse_gguf_from_reader_with_limits`].
///
/// # Errors
///
/// Returns [`AnamnesisError::Io`] if the reader fails;
/// [`AnamnesisError::Parse`] on malformed input;
/// [`AnamnesisError::LimitExceeded`] if a declared length, count, or dimension
/// exceeds a permanent cap (always-on — reachable even at default limits); or
/// [`AnamnesisError::Unsupported`] for an unsupported `GGUF` variant.
///
/// # Memory
///
/// Reads the whole stream into an owned `Vec<u8>`; peak heap is the artefact
/// size plus `O(n_tensors + n_metadata_kv)`.
pub fn parse_gguf_from_reader<R: Read>(reader: R) -> crate::Result<ParsedGguf> {
    parse_gguf_from_reader_with_limits(reader, &ParseLimits::default())
}

/// Parses a `GGUF` artefact from any reader under a caller-supplied
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
/// [`parse_gguf_with_limits`].
/// Returns [`AnamnesisError::Unsupported`] for an unsupported `GGUF` variant.
///
/// # Memory
///
/// Reads the stream into an owned `Vec<u8>` of at most
/// `max_single_alloc_bytes + 1` bytes; peak heap is the artefact size plus
/// `O(n_tensors + n_metadata_kv)`.
pub fn parse_gguf_from_reader_with_limits<R: Read>(
    reader: R,
    limits: &ParseLimits,
) -> crate::Result<ParsedGguf> {
    let bytes = limits.read_to_vec_bounded(reader, "GGUF file")?;
    parse_gguf_bytes_with_limits(bytes, limits)
}

/// Reader-generic core extracted from [`parse_gguf`].
///
/// Returns the parsed front matter — version, effective alignment, metadata
/// `HashMap`, and per-tensor info records (with absolute offsets) — without
/// depending on a memory-mapped substrate. The path-based [`parse_gguf`]
/// delegates to this function over a `std::io::Cursor` wrapping its mmap;
/// [`inspect_gguf_from_reader`] delegates over the caller-supplied reader.
///
/// All adversarial-input guards from the original mmap-only parser are
/// preserved verbatim: caps on `tensor_count`, `metadata_kv_count`, string
/// length, array length, nesting depth, dimension count, and element
/// product, plus per-tensor alignment and end-of-data bounds checks against
/// the stream's total length (captured via `seek(SeekFrom::End(0))` once at
/// reader construction).
fn read_gguf_structure<R: Read + Seek>(
    reader: R,
    limits: &ParseLimits,
) -> crate::Result<GgufFrontMatter> {
    let mut cursor = GgufReader::new(reader, limits)?;

    // Magic check. Detect legacy GGML/GGJT/GGMF formats and byte-swapped
    // big-endian GGUF files for clearer error messages.
    if cursor.file_len < 4 {
        return Err(AnamnesisError::Parse {
            reason: "GGUF: file shorter than 4 bytes (no magic)".into(),
        });
    }
    let mut magic_bytes = [0u8; 4];
    cursor.read_into(&mut magic_bytes)?;
    if &magic_bytes != GGUF_MAGIC {
        let as_le = u32::from_le_bytes(magic_bytes);
        if as_le == GGUF_MAGIC_BE_U32 {
            return Err(AnamnesisError::Unsupported {
                format: "GGUF".into(),
                detail: "big-endian GGUF files are not yet supported".into(),
            });
        }
        let legacy_name: Option<&'static str> = match &magic_bytes {
            b"GGML" => Some("GGML"),
            b"GGJT" => Some("GGJT"),
            b"GGMF" => Some("GGMF"),
            _ => None,
        };
        if let Some(name) = legacy_name {
            return Err(AnamnesisError::Unsupported {
                format: "GGUF".into(),
                detail: format!(
                    "legacy `{name}` format predates GGUF; re-convert with `llama.cpp` to GGUF"
                ),
            });
        }
        return Err(AnamnesisError::Parse {
            reason: format!(
                "GGUF: invalid magic (expected `GGUF`/{GGUF_MAGIC_LE_U32:#010x}, got {as_le:#010x})"
            ),
        });
    }

    let version = cursor.read_u32_le()?;
    if version == 1 {
        return Err(AnamnesisError::Unsupported {
            format: "GGUF".into(),
            detail: "GGUF v1 uses u32 string/array lengths and is not supported; \
                     re-save with a modern `llama.cpp` to produce v2 or v3"
                .into(),
        });
    }
    if version != 2 && version != 3 {
        return Err(AnamnesisError::Unsupported {
            format: "GGUF".into(),
            detail: format!("unsupported GGUF version {version} (expected 2 or 3)"),
        });
    }
    let tensor_count = cursor.read_u64_le()?;
    let kv_count = cursor.read_u64_le()?;
    if tensor_count > MAX_TENSOR_COUNT {
        return Err(AnamnesisError::LimitExceeded {
            limit: "MAX_TENSOR_COUNT",
            message: format!("GGUF: tensor count {tensor_count} exceeds cap {MAX_TENSOR_COUNT}"),
        });
    }
    if kv_count > MAX_KV_COUNT {
        return Err(AnamnesisError::LimitExceeded {
            limit: "MAX_KV_COUNT",
            message: format!("GGUF: metadata kv count {kv_count} exceeds cap {MAX_KV_COUNT}"),
        });
    }
    // Caller-supplied ceilings, layered on top of the permanent caps above.
    limits.check_item_count(tensor_count, "GGUF tensor count")?;
    limits.check_item_count(kv_count, "GGUF metadata KV count")?;
    let tensor_count_usz = usize::try_from(tensor_count).map_err(|_| AnamnesisError::Parse {
        reason: format!("GGUF: tensor count {tensor_count} overflows usize"),
    })?;
    let kv_count_usz = usize::try_from(kv_count).map_err(|_| AnamnesisError::Parse {
        reason: format!("GGUF: metadata kv count {kv_count} overflows usize"),
    })?;

    // Read metadata key-value pairs. Cap the pre-allocation at
    // `PREALLOC_SOFT_CAP` so an adversarial header claiming a million
    // entries cannot force ~114 MB of eager heap allocation; the HashMap
    // grows geometrically from there on legitimate large inputs.
    let mut metadata: HashMap<String, GgufMetadataValue> =
        HashMap::with_capacity(kv_count_usz.min(PREALLOC_SOFT_CAP));
    for _ in 0..kv_count_usz {
        let key = cursor.read_string(MAX_GGUF_KEY_LEN, "MAX_GGUF_KEY_LEN")?;
        let value_type = cursor.read_u32_le()?;
        let value = read_metadata_value(&mut cursor, value_type)?;
        metadata.insert(key, value);
    }

    // Resolve alignment (honour `general.alignment` if present).
    let alignment = match metadata.get("general.alignment") {
        Some(GgufMetadataValue::U32(v)) if *v != 0 => *v,
        Some(GgufMetadataValue::U32(_)) => {
            return Err(AnamnesisError::Parse {
                reason: "GGUF: general.alignment is zero".into(),
            });
        }
        Some(other) => {
            return Err(AnamnesisError::Parse {
                reason: format!(
                    "GGUF: general.alignment has wrong type (expected UINT32, got {})",
                    metadata_type_name(other)
                ),
            });
        }
        None => DEFAULT_ALIGNMENT,
    };
    let alignment_u64 = u64::from(alignment);

    // Read tensor info entries directly into `tensor_infos`. Each entry is:
    //   name (gguf_string_t) | n_dimensions (u32) | dimensions[n_dims] (u64)
    //   | type (u32) | offset (u64)
    // Offsets are relative to the start of the tensor_data section, which
    // we don't know until after the whole table has been read. Store the
    // raw relative offset in `data_offset` for now; a patch sweep below
    // rewrites it to the absolute offset once `data_section_start` is known.
    //
    // Cap the pre-allocation at `PREALLOC_SOFT_CAP` for the same reason as
    // the metadata map above — trust-the-header DoS guard.
    let mut tensor_infos: Vec<GgufTensorInfo> =
        Vec::with_capacity(tensor_count_usz.min(PREALLOC_SOFT_CAP));
    for _ in 0..tensor_count_usz {
        tensor_infos.push(read_tensor_info_relative(&mut cursor)?);
    }

    let tensor_info_end = cursor.pos;
    let file_len_u64 = cursor.file_len;

    // The tensor_data section begins at the next alignment boundary after
    // the tensor-info table. A file with zero tensors has no data section at
    // all, so skip the alignment and bounds check entirely — a 24-byte
    // header-only GGUF is legitimately well-formed.
    let data_section_start = if tensor_infos.is_empty() {
        tensor_info_end
    } else {
        let start = align_up(tensor_info_end, alignment_u64)?;
        if start > file_len_u64 {
            return Err(AnamnesisError::Parse {
                reason: format!(
                    "GGUF: tensor data section start {start} exceeds file size {file_len_u64}"
                ),
            });
        }
        start
    };

    // Patch sweep: rewrite the temporary relative offsets into absolute
    // offsets and run the bounds checks that couldn't happen at read time.
    for info in &mut tensor_infos {
        let relative_offset = info.data_offset;
        // The GGUF spec mandates that every tensor's offset field is a
        // multiple of `general.alignment`. `data_section_start` is itself
        // aligned (via `align_up` above), so checking the relative offset
        // is equivalent to checking the absolute offset and catches
        // adversarial files that would hand out unaligned byte slices to
        // SIMD dequant kernels downstream.
        if relative_offset % alignment_u64 != 0 {
            return Err(AnamnesisError::Parse {
                reason: format!(
                    "GGUF tensor `{}`: relative offset {relative_offset} is not a multiple of alignment {alignment_u64}",
                    info.name
                ),
            });
        }
        let absolute = data_section_start
            .checked_add(relative_offset)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!(
                    "GGUF tensor `{}`: absolute offset overflow ({} + {})",
                    info.name, data_section_start, relative_offset
                ),
            })?;
        // Sanity-check that the tensor at least starts inside the file.
        // After Phase 4.5 every recognised dtype has a known `type_size`,
        // but this guard remains as a defence against adversarial inputs
        // whose `data_offset` claims would otherwise leak through to
        // `tensor_info()`.
        if absolute > file_len_u64 {
            return Err(AnamnesisError::Parse {
                reason: format!(
                    "GGUF tensor `{}`: data_offset {absolute} exceeds file size {file_len_u64}",
                    info.name
                ),
            });
        }
        if let Some(len) = info.byte_len {
            let end = absolute
                .checked_add(len)
                .ok_or_else(|| AnamnesisError::Parse {
                    reason: format!("GGUF tensor `{}`: end offset overflow", info.name),
                })?;
            if end > file_len_u64 {
                return Err(AnamnesisError::Parse {
                    reason: format!(
                        "GGUF tensor `{}`: data range [{absolute}..{end}] exceeds file size {file_len_u64}",
                        info.name
                    ),
                });
            }
        }
        info.data_offset = absolute;
    }

    Ok(GgufFrontMatter {
        version,
        alignment,
        metadata,
        tensor_infos,
    })
}

/// Full front matter of a parsed `GGUF` file — version, alignment, the
/// complete metadata key-value table, and per-tensor info — read from any
/// `Read + Seek` source without touching the tensor-data segment.
///
/// The full-detail counterpart to [`GgufInspectInfo`]: where that type
/// reports aggregate statistics for a cheap inspect-before-parse policy
/// gate, `GgufFrontMatter` carries the same per-tensor list and metadata
/// table that [`ParsedGguf::tensor_info`] / [`ParsedGguf::metadata`] expose
/// for the mmap-backed path — everything a caller needs to render a full
/// tensor table (name, shape, dtype, offset) without ever reading tensor
/// data. Reader-generic core output shared by [`parse_gguf`] (via a
/// `Cursor` over its mmap) and [`inspect_gguf_from_reader`] /
/// [`parse_gguf_front_matter_from_reader`] (via the caller-supplied reader).
#[derive(Debug, Clone)]
#[must_use]
pub struct GgufFrontMatter {
    /// `GGUF` version read from the header (currently 2 or 3).
    pub version: u32,
    /// Effective alignment read from `general.alignment`, or the default of
    /// 32 bytes if the metadata key is absent.
    pub alignment: u32,
    /// Metadata key-value pairs, keyed by the full `GGUF` key (e.g.
    /// `general.architecture`). Iteration order is **unspecified** —
    /// `HashMap` does not preserve the file's key order; sort by key for
    /// deterministic rendering.
    pub metadata: HashMap<String, GgufMetadataValue>,
    /// Per-tensor metadata in file order, with `data_offset` resolved to an
    /// absolute offset from the start of the source.
    pub tensor_infos: Vec<GgufTensorInfo>,
}

impl GgufFrontMatter {
    /// Reduces this front matter to the aggregate [`GgufInspectInfo`]
    /// summary — the same output [`ParsedGguf::inspect`] and
    /// [`inspect_gguf_from_reader`] produce, computed by the same shared
    /// helper so all three stay substrate-equivalent. No I/O.
    pub fn inspect(&self) -> GgufInspectInfo {
        build_inspect_info(
            self.version,
            self.alignment,
            &self.metadata,
            &self.tensor_infos,
        )
    }
}

/// Inspects a `GGUF` file from any `Read + Seek` source, returning header,
/// metadata, and tensor-info summary statistics without touching the
/// tensor-data segment.
///
/// This is the cheap first half of the **inspect-before-parse** policy gate:
/// the returned [`GgufInspectInfo`] reports `total_bytes` and `tensor_count`,
/// which a host checks against its own budget before calling the authoritative
/// [`parse_gguf_with_limits`] under a matched [`ParseLimits`].
///
/// This is the reader-generic core of `GGUF` inspection: callers supply any
/// positional substrate (in-memory `Cursor`, an `HTTP`-range-backed adapter,
/// a custom transport, …) and receive the same [`GgufInspectInfo`] as
/// [`ParsedGguf::inspect`]. The path-based [`parse_gguf`] remains the right
/// entry point for callers that need the full [`ParsedGguf`] (with zero-copy
/// tensor data via `memmap2::Mmap`); `inspect_gguf_from_reader` exists for
/// the inspect-only case where materialising the data segment is wasteful.
///
/// # Range-read access pattern
///
/// `GGUF` is a front-loaded format: magic + version + counts (24 B), the
/// metadata key-value table (typically a few hundred KiB to a few MiB,
/// dominated by tokenizer vocabulary arrays), and the tensor-info table
/// (~80–120 B per tensor) all live before the tensor-data section. The
/// parser reads this front matter in a single linear scan, then performs
/// arithmetic on the captured stream length to validate per-tensor offsets
/// against the alignment boundary that anchors the data section. **No
/// tensor-data bytes are read.**
///
/// A well-implemented `R: Read + Seek` adapter only needs to satisfy three
/// access patterns:
///
/// 1. **`seek(SeekFrom::End(0))` once at construction** — captures the
///    total content length for the bounds-check arithmetic. An
///    `HTTP`-range adapter that already knows the `Content-Length` of the
///    artefact answers this without a fetch.
/// 2. **One contiguous forward read of the front matter** — magic (4 B)
///    through the end of the tensor-info table (~few MiB on multi-GB
///    quantised models). On a 2 GiB `Q4_K_M` `GGUF` this is well under
///    1 % of the file.
/// 3. **A handful of small `seek` calls back to offset 0 and to internal
///    offsets** that the `Read` machinery may issue (e.g., `read_exact`
///    retries) — these typically resolve from the same prefix region the
///    forward scan already touches.
///
/// Adapters that prefetch and cache the front-matter region on first
/// access amortise away the round trips, so a 2 GiB quantised `GGUF` is
/// inspectable in two or three small `HTTP`-range requests covering a few
/// MiB instead of a 2 GiB download.
///
/// Why `Read + Seek` (and not just `Read`): unlike safetensors's prefix-
/// then-`JSON` layout, `GGUF`'s parser computes the absolute tensor-data
/// offset by combining the relative offsets in the tensor-info table with
/// the post-tensor-info `data_section_start` anchor, then validates each
/// offset's alignment relative to that anchor. The simplest correct
/// refactor preserves this positional access pattern via `Seek`. A
/// pure-`Read` reformulation would require restructuring the parser into a
/// strict forward pass and is out of scope for this entry point.
///
/// Anamnesis itself does not ship an `HTTP` transport; the network layer
/// belongs in downstream crates (e.g., `hf-fm`'s `HttpRangeReader`
/// adapter). This function defines the I/O contract such an adapter must
/// satisfy.
///
/// # Performance
///
/// The parser issues many small `read_exact` calls (4–8 B per typed
/// primitive, variable per `gguf_string_t`). To keep that pattern from
/// degrading to one syscall per primitive on a `std::fs::File` substrate,
/// the user-supplied reader is wrapped internally in
/// `std::io::BufReader<R>` with a 64 KiB buffer. Per-file timings
/// (best-of-5 release-mode median, `target-cpu=native`,
/// [`tests/bench_gguf_inspect_adhoc.rs`](../tests/bench_gguf_inspect_adhoc.rs))
/// show the reader path is **at parity with the mmap-backed
/// [`parse_gguf`]`.inspect()`**, occasionally slightly faster
/// (`BufReader` does one syscall per 64 KiB while mmap incurs one minor
/// page fault per 4 KiB page touched):
///
/// | Fixture | mmap median | reader median (`File`) | ratio |
/// |---|---:|---:|---:|
/// | `Mistral-7B-Instruct-v0.3-IQ3_XXS.gguf` (2.7 GiB) | 3.0 ms | 2.8 ms | 0.9× |
/// | `SmolLM2-135M-Instruct-Q4_K_M.gguf` (101 MiB)     | 7.9 ms | 7.6 ms | 1.0× |
/// | `Qwen2.5-1.5B-Instruct-IQ2_M.gguf` (573 MiB)      | 26.4 ms | 25.4 ms | 1.0× |
///
/// In-memory `Cursor<&[u8]>` callers pay one extra memcpy through the
/// internal `BufReader` — negligible vs. the parsing work.
/// `HTTP`-range adapters that prefetch the front matter (~few MiB on
/// multi-GB quantised models) see the same syscall-batching win at the
/// network layer, plus the locality benefit of one larger range request
/// per buffer-fill instead of dozens of small ones.
///
/// # Errors
///
/// Returns [`AnamnesisError::Io`] if a `read` or `seek` on the supplied
/// reader fails.
///
/// Returns [`AnamnesisError::Parse`] if the magic bytes are missing, the
/// header fields are truncated, the metadata table contains an invalid value
/// type, a tensor info entry is malformed, or a tensor's resolved byte range
/// falls outside the stream's total length (captured via `seek(SeekFrom::End(0))`
/// at construction).
///
/// Returns [`AnamnesisError::LimitExceeded`] if a declared string/array length,
/// metadata-KV or tensor count, dimension, or element count exceeds a permanent
/// `GGUF` cap (always-on).
///
/// Returns [`AnamnesisError::Unsupported`] for `GGUF` v1 files (which use
/// `u32` string lengths instead of `u64`), big-endian `GGUF` files (v3+
/// feature, not yet implemented), legacy pre-`GGUF` formats (`GGML`, `GGJT`,
/// `GGMF`), and tensor dtypes whose `ggml_type` discriminant is not
/// recognised.
///
/// # Source context
///
/// Errors describe the **format-level problem**, not the source identity.
/// The function is reader-agnostic — the source could be a file, an
/// in-memory `Cursor`, or an `HTTP`-range adapter. Callers that have a
/// source name (filename, URL, etc.) should wrap the returned error with
/// that context. This matches anamnesis's existing convention
/// (`parse_safetensors_header_from_reader` and `inspect_npz_from_reader`
/// already return source-agnostic errors).
///
/// # Memory
///
/// Allocates only metadata structures: the metadata `HashMap` (one entry
/// per `KV` pair, value sizes dominated by typed-array inner storage like
/// tokenizer vocabularies) and the per-tensor `Vec<GgufTensorInfo>` (~120
/// B per tensor). No tensor data is read. Peak heap is
/// `O(n_tensors + n_metadata_kv)`, independent of the file's data-segment
/// size — the same big-O footprint as the path-based [`parse_gguf`] minus
/// the mmap.
pub fn inspect_gguf_from_reader<R: Read + Seek>(reader: R) -> crate::Result<GgufInspectInfo> {
    // The parser issues many small `read_exact` calls; on a `std::fs::File`
    // each is a syscall. Wrapping in a `BufReader` collapses those into one
    // `read` per `READER_BUF_SIZE` bytes. `BufReader<R: Read + Seek>: Seek`,
    // and the only seeks happen inside `GgufReader::new` *before* any
    // buffered reads, so the buffer is always empty when a seek is issued —
    // no invalidation cost.
    //
    // The path-based `parse_gguf` does **not** wrap (it passes a
    // `Cursor<&[u8]>` over the mmap, which is already zero-syscall). Adding
    // a `BufReader` there would only add a memcpy.
    let buffered = BufReader::with_capacity(READER_BUF_SIZE, reader);
    // The inspect path is intentionally limit-free: it reports the totals a host
    // checks against its policy (the inspect-before-parse gate), then the host
    // calls `parse_gguf_with_limits` for the real enforcement. So pass the
    // unbounded default — the permanent GGUF caps remain the only bound here.
    let front = read_gguf_structure(buffered, &ParseLimits::default())?;
    Ok(build_inspect_info(
        front.version,
        front.alignment,
        &front.metadata,
        &front.tensor_infos,
    ))
}

/// Parses full `GGUF` front matter from any `Read + Seek` source — the
/// complete per-tensor list and metadata table, without materialising the
/// tensor-data segment.
///
/// The full-detail counterpart to [`inspect_gguf_from_reader`]. Both are the
/// [`ParseLimits::default`] (unbounded) special case of their `_with_limits`
/// form, so the two are **limits-equivalent** — only the permanent `GGUF`
/// caps bound this call. What differs is what survives the parse: the
/// aggregate summary, or every tensor name and metadata value handed to the
/// caller.
///
/// **Prefer [`parse_gguf_front_matter_from_reader_with_limits`] for
/// untrusted input.** Because this entry point returns every parsed string
/// and array instead of reducing them to counts, its exposure at a given
/// budget is strictly larger than [`inspect_gguf_from_reader`]'s, and an
/// explicit [`ParseLimits`] is the only thing that bounds it below the
/// permanent caps. This mirrors [`parse_gguf_from_reader`] /
/// [`parse_gguf_from_reader_with_limits`].
///
/// # Errors
///
/// Returns [`AnamnesisError::Io`] if a `read` or `seek` on the supplied
/// reader fails;
/// [`AnamnesisError::Parse`] on malformed input;
/// [`AnamnesisError::LimitExceeded`] if a declared string/array length,
/// metadata-KV or tensor count, dimension, or element count exceeds a
/// permanent `GGUF` cap (always-on — reachable even at default limits); or
/// [`AnamnesisError::Unsupported`] for an unsupported `GGUF` variant.
///
/// # Memory
///
/// Same footprint as [`inspect_gguf_from_reader`]: `O(n_tensors + n_metadata_kv)`,
/// independent of the file's data-segment size. No tensor data is read.
pub fn parse_gguf_front_matter_from_reader<R: Read + Seek>(
    reader: R,
) -> crate::Result<GgufFrontMatter> {
    parse_gguf_front_matter_from_reader_with_limits(reader, &ParseLimits::default())
}

/// Parses full `GGUF` front matter from any `Read + Seek` source under a
/// caller-supplied [`ParseLimits`] budget — the bounded, reader-generic
/// full-detail path.
///
/// Wraps `reader` in the same 64 KiB `BufReader` as
/// [`inspect_gguf_from_reader`] (see its `# Performance` section) and
/// delegates to the same shared internal parsing core, so this function
/// and [`inspect_gguf_from_reader`] are substrate- and limits-equivalent by
/// construction — the only difference is which fields of the parsed front
/// matter are kept.
///
/// # Errors
///
/// Returns [`AnamnesisError::Io`] if a `read` or `seek` on the supplied
/// reader fails.
/// Returns [`AnamnesisError::Parse`] on the malformed-input conditions of
/// [`inspect_gguf_from_reader`].
/// Returns [`AnamnesisError::LimitExceeded`] if a declared string/array
/// length, metadata-KV or tensor count, dimension, or element count exceeds
/// `limits` or a permanent `GGUF` cap.
/// Returns [`AnamnesisError::Unsupported`] for the `GGUF` variants listed in
/// [`inspect_gguf_from_reader`]'s docs.
///
/// # Memory
///
/// Same footprint as [`inspect_gguf_from_reader`]: `O(n_tensors + n_metadata_kv)`,
/// independent of the file's data-segment size. No tensor data is read.
pub fn parse_gguf_front_matter_from_reader_with_limits<R: Read + Seek>(
    reader: R,
    limits: &ParseLimits,
) -> crate::Result<GgufFrontMatter> {
    let buffered = BufReader::with_capacity(READER_BUF_SIZE, reader);
    read_gguf_structure(buffered, limits)
}

/// Builds a [`GgufInspectInfo`] from the parsed front matter.
///
/// Shared by [`ParsedGguf::inspect`] (mmap-backed path) and
/// [`inspect_gguf_from_reader`] (reader-generic path) so the two entry
/// points are guaranteed substrate-equivalent — every field of the
/// resulting `GgufInspectInfo` is computed by the same code regardless of
/// which entry point produced the front matter.
fn build_inspect_info(
    version: u32,
    alignment: u32,
    metadata: &HashMap<String, GgufMetadataValue>,
    tensor_infos: &[GgufTensorInfo],
) -> GgufInspectInfo {
    let mut total_bytes: u64 = 0;
    let mut unknown_size_tensors: usize = 0;
    // O(1) per-tensor dtype dedup via a fixed-size bitmap keyed on the
    // dense `GgufType::inspect_index` — drops the hot loop from
    // O(n × d) to O(n). `dtypes` still records first-occurrence order.
    let mut seen = [false; GGUF_TYPE_COUNT];
    let mut dtypes: Vec<GgufType> = Vec::new();
    for info in tensor_infos {
        if let Some(byte_len) = info.byte_len {
            total_bytes = total_bytes.saturating_add(byte_len);
        } else {
            unknown_size_tensors = unknown_size_tensors.saturating_add(1);
        }
        let idx = info.dtype.inspect_index();
        // INDEX: `inspect_index` is defined to return a value in
        // `0..GGUF_TYPE_COUNT`, matching the bitmap's length exactly
        #[allow(clippy::indexing_slicing)]
        if !seen[idx] {
            #[allow(clippy::indexing_slicing)]
            {
                seen[idx] = true;
            }
            dtypes.push(info.dtype);
        }
    }
    let architecture = metadata
        .get("general.architecture")
        .and_then(GgufMetadataValue::as_string)
        // BORROW: `.to_owned()` converts `&str` to an owned `String`
        // that outlives the borrow of the metadata map.
        .map(str::to_owned);
    GgufInspectInfo {
        version,
        architecture,
        tensor_count: tensor_infos.len(),
        total_bytes,
        unknown_size_tensors,
        dtypes,
        alignment,
    }
}

/// Reads one `gguf_tensor_info_t` from the cursor and returns a
/// [`GgufTensorInfo`] whose `data_offset` holds the **relative** offset as
/// stored in the file. A patch sweep in [`parse_gguf`] rewrites it to the
/// absolute mmap offset once `data_section_start` is known.
///
/// All the cheap per-entry validation (dimension count, zero dimension,
/// element-count overflow, `byte_size_for_n_elements`, dimension-to-`usize`
/// conversion) happens here so that the patch sweep only needs to do offset
/// arithmetic and the two bounds checks that depend on the data section
/// start.
fn read_tensor_info_relative<R: Read + Seek>(
    cursor: &mut GgufReader<R>,
) -> crate::Result<GgufTensorInfo> {
    // GGUF tensor names: the spec caps them at 64 bytes, but some encoders
    // produce longer names in practice. Accept up to `MAX_TENSOR_NAME_LEN`.
    let name = cursor.read_string(MAX_TENSOR_NAME_LEN, "MAX_TENSOR_NAME_LEN")?;
    let n_dims = cursor.read_u32_le()?;
    if n_dims == 0 {
        return Err(AnamnesisError::Parse {
            reason: format!("GGUF tensor `{name}`: n_dimensions is zero"),
        });
    }
    if n_dims > MAX_TENSOR_DIMS {
        return Err(AnamnesisError::LimitExceeded {
            limit: "MAX_TENSOR_DIMS",
            message: format!(
                "GGUF tensor `{name}`: n_dimensions {n_dims} exceeds cap {MAX_TENSOR_DIMS}"
            ),
        });
    }
    let n_dims_usz = usize::try_from(n_dims).map_err(|_| AnamnesisError::Parse {
        reason: format!("GGUF tensor `{name}`: n_dimensions {n_dims} overflows usize"),
    })?;
    // Read the `n_dims` dimensions and convert them to `usize` at the same
    // time so we never keep both a `Vec<u64>` and a `Vec<usize>` alive.
    let mut shape_usz: Vec<usize> = Vec::with_capacity(n_dims_usz);
    // Track the element-count product as we go so that we can call
    // `byte_size_for_n_elements` below without re-iterating the shape.
    let mut n_elements: u64 = 1;
    for _ in 0..n_dims {
        let d = cursor.read_u64_le()?;
        if d == 0 {
            return Err(AnamnesisError::Parse {
                reason: format!("GGUF tensor `{name}`: zero-sized dimension"),
            });
        }
        n_elements = n_elements
            .checked_mul(d)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!("GGUF tensor `{name}`: element count overflow"),
            })?;
        if n_elements > MAX_TENSOR_ELEMENTS {
            return Err(AnamnesisError::LimitExceeded {
                limit: "MAX_TENSOR_ELEMENTS",
                message: format!(
                    "GGUF tensor `{name}`: element count {n_elements} exceeds cap {MAX_TENSOR_ELEMENTS}"
                ),
            });
        }
        let d_usz = usize::try_from(d).map_err(|_| AnamnesisError::Parse {
            reason: format!("GGUF tensor `{name}`: dimension {d} overflows usize"),
        })?;
        shape_usz.push(d_usz);
    }
    let dtype = GgufType::from_u32(cursor.read_u32_le()?)?;
    let relative_offset = cursor.read_u64_le()?;
    let byte_len = if dtype.type_size().is_some() {
        Some(dtype.byte_size_for_n_elements(n_elements)?)
    } else {
        None
    };
    Ok(GgufTensorInfo {
        name,
        shape: shape_usz,
        dtype,
        // Temporarily holds the relative offset; patched to absolute in the
        // caller once `data_section_start` is known.
        data_offset: relative_offset,
        byte_len,
    })
}

/// Returns the canonical name of a metadata value type for error messages.
const fn metadata_type_name(value: &GgufMetadataValue) -> &'static str {
    match value {
        GgufMetadataValue::U8(_) => "UINT8",
        GgufMetadataValue::I8(_) => "INT8",
        GgufMetadataValue::U16(_) => "UINT16",
        GgufMetadataValue::I16(_) => "INT16",
        GgufMetadataValue::U32(_) => "UINT32",
        GgufMetadataValue::I32(_) => "INT32",
        GgufMetadataValue::F32(_) => "FLOAT32",
        GgufMetadataValue::Bool(_) => "BOOL",
        GgufMetadataValue::String(_) => "STRING",
        GgufMetadataValue::Array(_) => "ARRAY",
        GgufMetadataValue::U64(_) => "UINT64",
        GgufMetadataValue::I64(_) => "INT64",
        GgufMetadataValue::F64(_) => "FLOAT64",
    }
}

/// Rounds `offset` up to the next multiple of `alignment`.
///
/// `alignment` must be non-zero; the caller guarantees this by substituting
/// `DEFAULT_ALIGNMENT` whenever the metadata key is absent or zero.
///
/// `pub(crate)` so the sibling [`gguf_write`](super::gguf_write) module can
/// reuse the same alignment helper the parser uses — keeping the read and
/// write paths in lock-step on padding behaviour.
pub(crate) fn align_up(offset: u64, alignment: u64) -> crate::Result<u64> {
    if alignment == 0 {
        return Err(AnamnesisError::Parse {
            reason: "GGUF: general.alignment must be non-zero".into(),
        });
    }
    let rem = offset % alignment;
    if rem == 0 {
        return Ok(offset);
    }
    // `alignment - rem` is strictly less than `alignment`, so the add
    // overflows only if `offset` is already within `alignment` of `u64::MAX`.
    let padding = alignment - rem;
    offset
        .checked_add(padding)
        .ok_or_else(|| AnamnesisError::Parse {
            reason: format!(
                "GGUF: alignment padding overflow (offset {offset}, alignment {alignment})"
            ),
        })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::wildcard_enum_match_arm,
    clippy::manual_is_multiple_of
)]
mod tests {
    use super::*;
    use std::io::Write;

    // -----------------------------------------------------------------
    // Fixture builder
    // -----------------------------------------------------------------

    /// In-memory builder for synthetic GGUF byte streams. Produces
    /// little-endian v3 files by default; individual fields can be
    /// overridden for negative tests.
    struct GgufBuilder {
        buf: Vec<u8>,
    }

    impl GgufBuilder {
        fn new() -> Self {
            Self { buf: Vec::new() }
        }

        fn push_bytes(&mut self, bytes: &[u8]) {
            self.buf.extend_from_slice(bytes);
        }

        fn push_u32(&mut self, v: u32) {
            self.buf.extend_from_slice(&v.to_le_bytes());
        }

        fn push_u64(&mut self, v: u64) {
            self.buf.extend_from_slice(&v.to_le_bytes());
        }

        fn push_string(&mut self, s: &str) {
            self.push_u64(s.len() as u64);
            self.buf.extend_from_slice(s.as_bytes());
        }

        fn push_kv_uint32(&mut self, key: &str, value: u32) {
            self.push_string(key);
            self.push_u32(4); // UINT32
            self.push_u32(value);
        }

        fn push_kv_string(&mut self, key: &str, value: &str) {
            self.push_string(key);
            self.push_u32(8); // STRING
            self.push_string(value);
        }

        fn push_kv_f32_array(&mut self, key: &str, values: &[f32]) {
            self.push_string(key);
            self.push_u32(9); // ARRAY
            self.push_u32(6); // inner type FLOAT32
            self.push_u64(values.len() as u64);
            for v in values {
                self.buf.extend_from_slice(&v.to_le_bytes());
            }
        }

        fn push_tensor_info(
            &mut self,
            name: &str,
            shape: &[u64],
            dtype_disc: u32,
            relative_offset: u64,
        ) {
            self.push_string(name);
            self.push_u32(u32::try_from(shape.len()).expect("shape len fits in u32 for tests"));
            for &d in shape {
                self.push_u64(d);
            }
            self.push_u32(dtype_disc);
            self.push_u64(relative_offset);
        }

        fn pad_to_alignment(&mut self, alignment: usize) {
            while self.buf.len() % alignment != 0 {
                self.buf.push(0);
            }
        }

        fn finish(self) -> Vec<u8> {
            self.buf
        }
    }

    fn build_minimal_gguf() -> Vec<u8> {
        let mut b = GgufBuilder::new();
        b.push_bytes(b"GGUF");
        b.push_u32(3); // version
        b.push_u64(2); // tensor_count
        b.push_u64(3); // kv_count

        // kv pairs
        b.push_kv_string("general.architecture", "test");
        b.push_kv_uint32("general.alignment", 32);
        b.push_kv_f32_array("test.values", &[1.0, 2.0, 3.0]);

        // tensor 0 — F32 [2, 3] → 24 bytes at relative offset 0
        b.push_tensor_info("tensor.a", &[2, 3], 0, 0);
        // tensor 1 — Q4_0 [64] → 2 blocks × 18 bytes = 36 bytes at relative
        //            offset 32 (24 bytes + 8-byte pad to 32)
        b.push_tensor_info("tensor.b", &[64], 2, 32);

        b.pad_to_alignment(32);
        // tensor.a data — 24 bytes of zeros
        b.push_bytes(&[0u8; 24]);
        // pad to next 32-byte boundary
        b.pad_to_alignment(32);
        // tensor.b data — 36 bytes of zeros
        b.push_bytes(&[0u8; 36]);

        b.finish()
    }

    fn write_temp_gguf(bytes: &[u8]) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(bytes).unwrap();
        f.flush().unwrap();
        f
    }

    /// Phase 6.8 Step 1: a tightened `ParseLimits` rejects a tensor / KV count
    /// or string read that `ParseLimits::default()` accepts; the default is
    /// equivalent to the limit-free `parse_gguf`.
    #[test]
    fn parse_gguf_respects_parse_limits() {
        // build_minimal_gguf: 2 tensors, 3 KV entries, small string keys.
        let f = write_temp_gguf(&build_minimal_gguf());

        // Default (unbounded) parses, and matches the limit-free entry point.
        let baseline = parse_gguf(f.path()).unwrap();
        let with_default = parse_gguf_with_limits(f.path(), &ParseLimits::default()).unwrap();
        assert_eq!(baseline.tensor_infos.len(), with_default.tensor_infos.len());
        assert_eq!(with_default.tensor_infos.len(), 2);

        // Item-count ceiling: tensor_count (2) rejected at 1; both counts fit
        // at 3.
        let err = parse_gguf_with_limits(f.path(), &ParseLimits::default().with_max_item_count(1))
            .unwrap_err();
        assert!(
            matches!(err, AnamnesisError::LimitExceeded { limit, .. } if limit == "max_item_count"),
            "expected item-count limit error, got: {err}"
        );
        assert!(
            parse_gguf_with_limits(f.path(), &ParseLimits::default().with_max_item_count(3))
                .is_ok()
        );

        // Single-allocation ceiling: the first metadata string read (20-byte
        // key) is rejected at 1.
        let err =
            parse_gguf_with_limits(f.path(), &ParseLimits::default().with_max_single_alloc(1))
                .unwrap_err();
        assert!(
            matches!(err, AnamnesisError::LimitExceeded { limit, .. } if limit == "max_single_alloc_bytes"),
            "expected single-alloc limit error, got: {err}"
        );
    }

    /// Phase 6.8 Step 2: the aggregate `max_total_bytes` rejects a GGUF whose
    /// individual metadata strings / arrays / tensor names each pass the
    /// per-item cap but whose cumulative parse-time heap (held in the metadata
    /// map + tensor-info table) crosses the budget.
    #[test]
    fn parse_gguf_aggregate_budget() {
        // build_minimal_gguf charges ~80 bytes total (metadata string keys +
        // a "test" string value + a 3×f32 array + two tensor names), each item
        // tiny. Default parses; a 50-byte aggregate budget does not.
        let f = write_temp_gguf(&build_minimal_gguf());
        assert!(parse_gguf_with_limits(f.path(), &ParseLimits::default()).is_ok());

        let err =
            parse_gguf_with_limits(f.path(), &ParseLimits::default().with_max_total_bytes(50))
                .unwrap_err();
        assert!(
            matches!(err, AnamnesisError::LimitExceeded { limit, .. } if limit == "max_total_bytes"),
            "expected aggregate limit error, got: {err}"
        );

        // A generous aggregate budget parses (proving 50 rejected on the total,
        // not any single item).
        assert!(parse_gguf_with_limits(
            f.path(),
            &ParseLimits::default().with_max_total_bytes(1 << 20)
        )
        .is_ok());
    }

    /// A metadata string that declares a length **under** `MAX_STRING_LEN` (so
    /// it passes the constant cap) but **larger than the bytes remaining** in
    /// the file is rejected by `ensure_remaining` *before* `read_bytes`
    /// allocates (Phase 6.7 Step 2). The declared 1 MiB sails past the 16 MiB
    /// cap, so this exercises the remaining-bytes guard specifically — a tiny
    /// (~30-byte) file can no longer drive an eager 1 MiB allocation.
    #[test]
    fn oversized_string_rejected_before_alloc() {
        let mut b = GgufBuilder::new();
        b.push_bytes(b"GGUF");
        b.push_u32(3); // version
        b.push_u64(0); // tensor_count
        b.push_u64(1); // kv_count
                       // One KV: key "k", value type 8 (string), declared length 1 MiB, no body.
        b.push_u64(1);
        b.push_bytes(b"k");
        b.push_u32(8); // GGUF string value type
        b.push_u64(1 << 20); // 1 MiB < MAX_STRING_LEN (16 MiB) → passes the cap
        let bytes = b.finish();

        let err = inspect_gguf_from_reader(std::io::Cursor::new(bytes)).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("EOF") || msg.contains("wanted"),
            "expected a remaining-bytes (EOF) rejection, got: {msg}"
        );
    }

    // -----------------------------------------------------------------
    // Happy-path tests
    // -----------------------------------------------------------------

    #[test]
    fn parse_minimal_gguf_succeeds() {
        let bytes = build_minimal_gguf();
        let tmp = write_temp_gguf(&bytes);
        let parsed = parse_gguf(tmp.path()).unwrap();
        assert_eq!(parsed.version(), 3);
        assert_eq!(parsed.alignment(), 32);
        assert_eq!(parsed.len(), 2);
        assert!(!parsed.is_empty());

        let infos = parsed.tensor_info();
        assert_eq!(infos[0].name, "tensor.a");
        assert_eq!(infos[0].shape, vec![2, 3]);
        assert_eq!(infos[0].dtype, GgufType::F32);
        assert_eq!(infos[0].byte_len, Some(24));
        assert_eq!(infos[1].name, "tensor.b");
        assert_eq!(infos[1].shape, vec![64]);
        assert_eq!(infos[1].dtype, GgufType::Q4_0);
        assert_eq!(infos[1].byte_len, Some(36));

        let metadata = parsed.metadata();
        assert_eq!(
            metadata
                .get("general.architecture")
                .and_then(|v| v.as_string()),
            Some("test")
        );
        assert_eq!(
            metadata
                .get("general.alignment")
                .and_then(GgufMetadataValue::as_u32),
            Some(32)
        );
        let arr = metadata
            .get("test.values")
            .and_then(GgufMetadataValue::as_array)
            .unwrap();
        assert_eq!(arr.len(), 3);
        let f32s = arr
            .as_f32_slice()
            .expect("test.values should be an F32 array");
        assert_eq!(f32s, &[1.0f32, 2.0, 3.0]);
    }

    #[test]
    fn tensors_returns_zero_copy_borrowed_slices() {
        let bytes = build_minimal_gguf();
        let tmp = write_temp_gguf(&bytes);
        let parsed = parse_gguf(tmp.path()).unwrap();
        let tensors: Vec<_> = parsed.tensors().collect();
        assert_eq!(tensors.len(), 2);
        for t in &tensors {
            assert!(matches!(t.data, Cow::Borrowed(_)));
        }
        assert_eq!(tensors[0].data.len(), 24);
        assert_eq!(tensors[1].data.len(), 36);
        // `name` and `shape` now borrow from the ParsedGguf, not owned.
        assert_eq!(tensors[0].name, "tensor.a");
        assert_eq!(tensors[0].shape, &[2_usize, 3]);
        assert_eq!(tensors[1].name, "tensor.b");
        assert_eq!(tensors[1].shape, &[64_usize]);
    }

    #[test]
    fn inspect_info_reports_expected_fields() {
        let bytes = build_minimal_gguf();
        let tmp = write_temp_gguf(&bytes);
        let parsed = parse_gguf(tmp.path()).unwrap();
        let info = parsed.inspect();
        assert_eq!(info.version, 3);
        assert_eq!(info.architecture.as_deref(), Some("test"));
        assert_eq!(info.tensor_count, 2);
        assert_eq!(info.total_bytes, 24 + 36);
        assert_eq!(info.unknown_size_tensors, 0);
        assert_eq!(info.alignment, 32);
        assert_eq!(info.dtypes, vec![GgufType::F32, GgufType::Q4_0]);
        let rendered = info.to_string();
        assert!(rendered.contains("GGUF v3"));
        assert!(rendered.contains("Arch:        test"));
        assert!(rendered.contains("Tensors:     2"));
        assert!(rendered.contains("Dtypes:      F32, Q4_0"));
        assert!(rendered.contains("Alignment:   32 bytes"));
    }

    #[test]
    fn typed_array_f32_uses_native_storage() {
        // Round-trips a 5-element F32 metadata array and asserts the
        // parser materialised it as `GgufMetadataArray::F32(Vec<f32>)` —
        // the fast path that eliminates the 8× enum-discriminant bloat.
        let mut b = GgufBuilder::new();
        b.push_bytes(b"GGUF");
        b.push_u32(3);
        b.push_u64(0);
        b.push_u64(1);
        b.push_kv_f32_array("logits", &[1.5, -2.25, 0.0, 3.5, 7.125]);
        let tmp = write_temp_gguf(&b.finish());
        let parsed = parse_gguf(tmp.path()).unwrap();
        let arr = parsed
            .metadata()
            .get("logits")
            .and_then(GgufMetadataValue::as_array)
            .unwrap();
        assert!(matches!(arr, GgufMetadataArray::F32(_)));
        let slice = arr.as_f32_slice().unwrap();
        assert_eq!(slice, &[1.5f32, -2.25, 0.0, 3.5, 7.125]);
        assert_eq!(arr.len(), 5);
        assert!(!arr.is_empty());
    }

    #[test]
    fn metadata_value_size_is_bounded() {
        // Mirrors the compile-time `const _` assertions near the top of
        // the module so the size invariant shows up in the test suite too.
        assert_eq!(std::mem::size_of::<GgufMetadataValue>(), 24);
        assert_eq!(std::mem::size_of::<GgufMetadataArray>(), 32);
    }

    #[test]
    fn parse_header_only_file_is_accepted() {
        // 24-byte file: magic + version + tensor_count=0 + kv_count=0.
        // No tensor_data section exists, so the alignment-and-bounds check
        // must be skipped instead of rejecting a legitimate empty file.
        let mut b = GgufBuilder::new();
        b.push_bytes(b"GGUF");
        b.push_u32(3);
        b.push_u64(0);
        b.push_u64(0);
        let bytes = b.finish();
        assert_eq!(bytes.len(), 24);
        let tmp = write_temp_gguf(&bytes);
        let parsed = parse_gguf(tmp.path()).unwrap();
        assert_eq!(parsed.version(), 3);
        assert_eq!(parsed.alignment(), 32);
        assert_eq!(parsed.len(), 0);
        assert!(parsed.is_empty());
        assert!(parsed.metadata().is_empty());
        assert!(parsed.tensor_info().is_empty());
        assert_eq!(parsed.tensors().count(), 0);
    }

    #[test]
    fn alignment_defaults_to_32_when_metadata_absent() {
        let mut b = GgufBuilder::new();
        b.push_bytes(b"GGUF");
        b.push_u32(3);
        b.push_u64(1);
        b.push_u64(1);
        b.push_kv_string("general.architecture", "test");
        // Single F32 scalar
        b.push_tensor_info("x", &[1], 0, 0);
        b.pad_to_alignment(32);
        b.push_bytes(&[0u8; 4]);
        let bytes = b.finish();
        let tmp = write_temp_gguf(&bytes);
        let parsed = parse_gguf(tmp.path()).unwrap();
        assert_eq!(parsed.alignment(), 32);
    }

    // -----------------------------------------------------------------
    // Negative / validation tests
    // -----------------------------------------------------------------

    #[test]
    fn reject_file_too_small() {
        let tmp = write_temp_gguf(b"GGU");
        let err = parse_gguf(tmp.path()).unwrap_err();
        assert!(matches!(err, AnamnesisError::Parse { .. }));
    }

    #[test]
    fn reject_bad_magic() {
        let tmp = write_temp_gguf(b"XXXX\x00\x00\x00\x00");
        let err = parse_gguf(tmp.path()).unwrap_err();
        assert!(matches!(err, AnamnesisError::Parse { .. }));
    }

    #[test]
    fn reject_legacy_ggml_magic() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGML");
        bytes.extend_from_slice(&[0u8; 100]);
        let tmp = write_temp_gguf(&bytes);
        let err = parse_gguf(tmp.path()).unwrap_err();
        match err {
            AnamnesisError::Unsupported { format, detail } => {
                assert_eq!(format, "GGUF");
                assert!(detail.contains("GGML"));
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn reject_v1() {
        let mut b = GgufBuilder::new();
        b.push_bytes(b"GGUF");
        b.push_u32(1); // v1
        b.push_u64(0);
        b.push_u64(0);
        let tmp = write_temp_gguf(&b.finish());
        let err = parse_gguf(tmp.path()).unwrap_err();
        assert!(matches!(err, AnamnesisError::Unsupported { .. }));
    }

    #[test]
    fn reject_truncated_file() {
        let bytes = build_minimal_gguf();
        let truncated = &bytes[..bytes.len() - 20];
        let tmp = write_temp_gguf(truncated);
        let err = parse_gguf(tmp.path()).unwrap_err();
        assert!(matches!(err, AnamnesisError::Parse { .. }));
    }

    #[test]
    fn reject_tensor_data_out_of_bounds() {
        // 1 tensor, no KV pairs. The tensor claims F32 [1000] = 4000 bytes
        // at relative offset 0, but the file only contains 32 bytes of
        // tensor data, which is far less than that. Parser must reject.
        let mut b = GgufBuilder::new();
        b.push_bytes(b"GGUF");
        b.push_u32(3);
        b.push_u64(1);
        b.push_u64(0);
        b.push_tensor_info("huge", &[1000], 0, 0);
        b.pad_to_alignment(32);
        b.push_bytes(&[0u8; 32]);
        let tmp = write_temp_gguf(&b.finish());
        let err = parse_gguf(tmp.path()).unwrap_err();
        match err {
            AnamnesisError::Parse { reason } => {
                assert!(reason.contains("exceeds file size"), "got: {reason}");
            }
            other => panic!("expected Parse, got {other:?}"),
        }
    }

    #[test]
    fn reject_zero_dimension() {
        let mut b = GgufBuilder::new();
        b.push_bytes(b"GGUF");
        b.push_u32(3);
        b.push_u64(1);
        b.push_u64(0);
        b.push_tensor_info("zero", &[0], 0, 0);
        b.pad_to_alignment(32);
        let tmp = write_temp_gguf(&b.finish());
        let err = parse_gguf(tmp.path()).unwrap_err();
        assert!(matches!(err, AnamnesisError::Parse { .. }));
    }

    #[test]
    fn reject_unaligned_relative_offset() {
        // The GGUF spec mandates each tensor's offset field is a multiple
        // of `general.alignment`. A well-formed file with `alignment = 32`
        // and a tensor at `relative_offset = 1` must be rejected, because
        // downstream consumers would get unaligned byte slices.
        let mut b = GgufBuilder::new();
        b.push_bytes(b"GGUF");
        b.push_u32(3);
        b.push_u64(1);
        b.push_u64(0);
        // F32 [1] — 4 bytes — at relative offset 1 (not a multiple of 32).
        b.push_tensor_info("misaligned", &[1], 0, 1);
        b.pad_to_alignment(32);
        // Enough data so the trailing bounds check cannot mask the
        // alignment check — we need to verify the alignment check is the
        // one that fires, not the "exceeds file size" check.
        b.push_bytes(&[0u8; 64]);
        let tmp = write_temp_gguf(&b.finish());
        let err = parse_gguf(tmp.path()).unwrap_err();
        match err {
            AnamnesisError::Parse { reason } => {
                assert!(
                    reason.contains("not a multiple of alignment"),
                    "expected alignment error, got: {reason}"
                );
                assert!(reason.contains("misaligned"), "got: {reason}");
            }
            other => panic!("expected Parse, got {other:?}"),
        }
    }

    #[test]
    fn accept_aligned_nonzero_relative_offset() {
        // Regression guard for the alignment check: a legitimate tensor at
        // a non-zero but aligned relative offset (e.g., second tensor in a
        // file, sitting at relative offset 32) must still parse cleanly.
        let bytes = build_minimal_gguf();
        let parsed = parse_gguf(write_temp_gguf(&bytes).path()).unwrap();
        assert_eq!(parsed.len(), 2);
        // tensor.b is at relative offset 32 in the fixture — aligned to 32.
        assert_eq!(parsed.tensor_info()[1].name, "tensor.b");
    }

    #[test]
    fn reject_array_depth_exceeded() {
        // Nest ARRAY values deeper than MAX_ARRAY_DEPTH (4) and expect
        // rejection. The KV value_type=ARRAY is read by `read_metadata_value`,
        // which then calls `read_typed_array` at depth 0 (the outer array).
        // Each (inner_type=9, len=1) pair drives one more level of
        // recursion. Four pairs successfully walk depths 0 → 3 (building the
        // outer array plus 3 sub-arrays). The 5th pair trips the
        // `depth >= MAX_ARRAY_DEPTH` check when the parser tries to enter
        // depth 4.
        let mut b = GgufBuilder::new();
        b.push_bytes(b"GGUF");
        b.push_u32(3);
        b.push_u64(0);
        b.push_u64(1);
        b.push_string("nested");
        b.push_u32(9); // KV value_type = ARRAY
        for _ in 0..5 {
            b.push_u32(9); // inner_type = ARRAY
            b.push_u64(1); // length = 1
        }
        let tmp = write_temp_gguf(&b.finish());
        let err = parse_gguf(tmp.path()).unwrap_err();
        match err {
            AnamnesisError::LimitExceeded { limit, message } => {
                assert_eq!(limit, "MAX_ARRAY_DEPTH");
                assert!(
                    message.contains("depth cap"),
                    "expected depth-cap error, got: {message}"
                );
            }
            other => panic!("expected LimitExceeded, got {other:?}"),
        }
    }

    #[test]
    fn reject_bad_bool_byte() {
        let mut b = GgufBuilder::new();
        b.push_bytes(b"GGUF");
        b.push_u32(3);
        b.push_u64(0);
        b.push_u64(1);
        b.push_string("weird");
        b.push_u32(7); // BOOL
        b.push_bytes(&[7]); // not 0 or 1
        let tmp = write_temp_gguf(&b.finish());
        let err = parse_gguf(tmp.path()).unwrap_err();
        assert!(matches!(err, AnamnesisError::Parse { .. }));
    }

    #[test]
    fn reject_zero_alignment() {
        let mut b = GgufBuilder::new();
        b.push_bytes(b"GGUF");
        b.push_u32(3);
        b.push_u64(0);
        b.push_u64(1);
        b.push_kv_uint32("general.alignment", 0);
        let tmp = write_temp_gguf(&b.finish());
        let err = parse_gguf(tmp.path()).unwrap_err();
        assert!(matches!(err, AnamnesisError::Parse { .. }));
    }

    // -----------------------------------------------------------------
    // GgufType table spot-checks
    // -----------------------------------------------------------------

    #[test]
    fn byte_size_table_spot_checks() {
        assert_eq!(GgufType::F32.block_size(), 1);
        assert_eq!(GgufType::F32.type_size(), Some(4));
        assert_eq!(GgufType::F32.byte_size_for_n_elements(10).unwrap(), 40);

        assert_eq!(GgufType::Q4_0.block_size(), 32);
        assert_eq!(GgufType::Q4_0.type_size(), Some(18));
        assert_eq!(GgufType::Q4_0.byte_size_for_n_elements(64).unwrap(), 36);

        assert_eq!(GgufType::Q4_K.block_size(), 256);
        assert_eq!(GgufType::Q4_K.type_size(), Some(144));
        assert_eq!(GgufType::Q4_K.byte_size_for_n_elements(256).unwrap(), 144);

        assert_eq!(GgufType::Q8_0.block_size(), 32);
        assert_eq!(GgufType::Q8_0.type_size(), Some(34));
        assert_eq!(GgufType::Q8_0.byte_size_for_n_elements(32).unwrap(), 34);

        assert_eq!(GgufType::Q8_K.type_size(), Some(292));
        assert_eq!(GgufType::Q6_K.type_size(), Some(210));

        // Non-linear 4-bit IQ variants landed in Phase 4.5 step 1.
        assert_eq!(GgufType::IQ4_NL.block_size(), 32);
        assert_eq!(GgufType::IQ4_NL.type_size(), Some(18));
        assert_eq!(GgufType::IQ4_NL.byte_size_for_n_elements(64).unwrap(), 36);
        assert_eq!(GgufType::IQ4_XS.block_size(), 256);
        assert_eq!(GgufType::IQ4_XS.type_size(), Some(136));
        assert_eq!(GgufType::IQ4_XS.byte_size_for_n_elements(256).unwrap(), 136);

        // 2-bit IQ super-quants landed in Phase 4.5 step 2.
        assert_eq!(GgufType::IQ2_XXS.block_size(), 256);
        assert_eq!(GgufType::IQ2_XXS.type_size(), Some(66));
        assert_eq!(GgufType::IQ2_XXS.byte_size_for_n_elements(256).unwrap(), 66);
        assert_eq!(GgufType::IQ2_XS.block_size(), 256);
        assert_eq!(GgufType::IQ2_XS.type_size(), Some(74));
        assert_eq!(GgufType::IQ2_XS.byte_size_for_n_elements(256).unwrap(), 74);
        assert_eq!(GgufType::IQ2_S.block_size(), 256);
        assert_eq!(GgufType::IQ2_S.type_size(), Some(82));
        assert_eq!(GgufType::IQ2_S.byte_size_for_n_elements(256).unwrap(), 82);

        // 3-bit IQ super-quants landed in Phase 4.5 step 3.
        assert_eq!(GgufType::IQ3_XXS.block_size(), 256);
        assert_eq!(GgufType::IQ3_XXS.type_size(), Some(98));
        assert_eq!(GgufType::IQ3_XXS.byte_size_for_n_elements(256).unwrap(), 98);
        assert_eq!(GgufType::IQ3_S.block_size(), 256);
        assert_eq!(GgufType::IQ3_S.type_size(), Some(110));
        assert_eq!(GgufType::IQ3_S.byte_size_for_n_elements(256).unwrap(), 110);

        // 1-bit IQ super-quants landed in Phase 4.5 step 4.
        assert_eq!(GgufType::IQ1_S.block_size(), 256);
        assert_eq!(GgufType::IQ1_S.type_size(), Some(50));
        assert_eq!(GgufType::IQ1_S.byte_size_for_n_elements(256).unwrap(), 50);
        assert_eq!(GgufType::IQ1_M.block_size(), 256);
        assert_eq!(GgufType::IQ1_M.type_size(), Some(56));
        assert_eq!(GgufType::IQ1_M.byte_size_for_n_elements(256).unwrap(), 56);

        // Ternary TQ super-quants landed in Phase 4.5 step 5.
        assert_eq!(GgufType::TQ1_0.block_size(), 256);
        assert_eq!(GgufType::TQ1_0.type_size(), Some(54));
        assert_eq!(GgufType::TQ1_0.byte_size_for_n_elements(256).unwrap(), 54);
        assert_eq!(GgufType::TQ2_0.block_size(), 256);
        assert_eq!(GgufType::TQ2_0.type_size(), Some(66));
        assert_eq!(GgufType::TQ2_0.byte_size_for_n_elements(256).unwrap(), 66);

        // Microscaling FP4 landed in Phase 4.5 step 6 — the final block
        // type, closing the GGUF coverage gap. Every `GgufType` variant
        // now returns `Some(_)` from `type_size()`.
        assert_eq!(GgufType::MXFP4.block_size(), 32);
        assert_eq!(GgufType::MXFP4.type_size(), Some(17));
        assert_eq!(GgufType::MXFP4.byte_size_for_n_elements(64).unwrap(), 34);
    }

    #[test]
    fn is_quantized_classifies_correctly() {
        assert!(!GgufType::F32.is_quantized());
        assert!(!GgufType::BF16.is_quantized());
        assert!(!GgufType::I32.is_quantized());
        assert!(GgufType::Q4_0.is_quantized());
        assert!(GgufType::Q4_K.is_quantized());
        assert!(GgufType::IQ4_XS.is_quantized());
    }

    #[test]
    fn byte_size_rejects_non_multiple_of_block() {
        // 17 elements of Q4_0 (block size 32) — not a multiple.
        let err = GgufType::Q4_0.byte_size_for_n_elements(17).unwrap_err();
        assert!(matches!(err, AnamnesisError::Parse { .. }));
    }

    #[test]
    fn align_up_behaves() {
        assert_eq!(align_up(0, 32).unwrap(), 0);
        assert_eq!(align_up(1, 32).unwrap(), 32);
        assert_eq!(align_up(32, 32).unwrap(), 32);
        assert_eq!(align_up(33, 32).unwrap(), 64);
        assert_eq!(align_up(100, 16).unwrap(), 112);
    }

    #[test]
    fn gguf_type_display_roundtrip() {
        assert_eq!(GgufType::F32.to_string(), "F32");
        assert_eq!(GgufType::Q4_K.to_string(), "Q4_K");
        assert_eq!(GgufType::IQ4_XS.to_string(), "IQ4_XS");
        assert_eq!(GgufType::BF16.to_string(), "BF16");
    }

    // -----------------------------------------------------------------
    // Reader-generic API — substrate-equivalence tests (Phase 4.9)
    // -----------------------------------------------------------------

    /// Asserts every field of two `GgufInspectInfo` values is equal.
    /// Substrate equivalence means the path-based and reader-generic
    /// entry points must be indistinguishable in their output.
    fn assert_inspect_eq(path_info: &GgufInspectInfo, reader_info: &GgufInspectInfo) {
        assert_eq!(path_info.version, reader_info.version);
        assert_eq!(path_info.architecture, reader_info.architecture);
        assert_eq!(path_info.tensor_count, reader_info.tensor_count);
        assert_eq!(path_info.total_bytes, reader_info.total_bytes);
        assert_eq!(
            path_info.unknown_size_tensors,
            reader_info.unknown_size_tensors
        );
        assert_eq!(path_info.dtypes, reader_info.dtypes);
        assert_eq!(path_info.alignment, reader_info.alignment);
    }

    /// `inspect_gguf_from_reader` over an in-memory `Cursor` returns the
    /// same `GgufInspectInfo` as `parse_gguf(path).inspect()` over the same
    /// archive on disk. Locks the contract that the reader-generic and
    /// path-based APIs are substrate-equivalent — the substrate (file vs.
    /// cursor) cannot change the metadata. This is what downstream
    /// HTTP-range adapters rely on.
    #[test]
    fn inspect_from_reader_matches_path_minimal() {
        let bytes = build_minimal_gguf();
        let tmp = write_temp_gguf(&bytes);

        let path_info = parse_gguf(tmp.path()).unwrap().inspect();
        let reader_info = inspect_gguf_from_reader(std::io::Cursor::new(&bytes)).unwrap();

        assert_inspect_eq(&path_info, &reader_info);

        // And spot-check that the actual values are the ones the fixture
        // claims, so that we are not just comparing two equal-but-wrong
        // outputs.
        assert_eq!(reader_info.version, 3);
        assert_eq!(reader_info.architecture.as_deref(), Some("test"));
        assert_eq!(reader_info.tensor_count, 2);
        assert_eq!(reader_info.total_bytes, 24 + 36);
        assert_eq!(reader_info.dtypes, vec![GgufType::F32, GgufType::Q4_0]);
        assert_eq!(reader_info.alignment, 32);
    }

    /// `inspect_gguf_from_reader` accepts a header-only file (no tensors,
    /// no data section) — same as the path-based parser. Confirms the
    /// reader-generic path doesn't accidentally require the tensor-data
    /// section to be present.
    #[test]
    fn inspect_from_reader_accepts_header_only_file() {
        let mut b = GgufBuilder::new();
        b.push_bytes(b"GGUF");
        b.push_u32(3);
        b.push_u64(0);
        b.push_u64(0);
        let bytes = b.finish();
        assert_eq!(bytes.len(), 24);

        let info = inspect_gguf_from_reader(std::io::Cursor::new(&bytes)).unwrap();
        assert_eq!(info.version, 3);
        assert_eq!(info.tensor_count, 0);
        assert_eq!(info.total_bytes, 0);
        assert!(info.dtypes.is_empty());
        assert_eq!(info.alignment, 32);
    }

    /// `inspect_gguf_from_reader` propagates the same parse errors as the
    /// path-based variant. Each rejection branch is already covered by a
    /// dedicated path-based test above; this one asserts the reader-generic
    /// path does not accidentally swallow or re-classify them.
    #[test]
    fn inspect_from_reader_propagates_parse_errors() {
        // Truncated file (no magic).
        let err = inspect_gguf_from_reader(std::io::Cursor::new(b"GGU".as_slice())).unwrap_err();
        assert!(matches!(err, AnamnesisError::Parse { .. }));

        // Wrong magic.
        let err =
            inspect_gguf_from_reader(std::io::Cursor::new(b"XXXX\x00\x00\x00\x00".as_slice()))
                .unwrap_err();
        assert!(matches!(err, AnamnesisError::Parse { .. }));

        // Legacy GGML magic — surfaces as Unsupported through the same code
        // path as the file-backed parser.
        let mut legacy = Vec::new();
        legacy.extend_from_slice(b"GGML");
        legacy.extend_from_slice(&[0u8; 100]);
        let err = inspect_gguf_from_reader(std::io::Cursor::new(&legacy)).unwrap_err();
        match err {
            AnamnesisError::Unsupported { format, detail } => {
                assert_eq!(format, "GGUF");
                assert!(detail.contains("GGML"));
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    /// Substrate-equivalence on a multi-tensor file with mixed dtypes and a
    /// nested-array metadata value — the most adversarial-shape fixture in
    /// this module's test set. Exercises the `Read + Seek` substrate over
    /// the seek-back-and-forth pattern that distinguishes `GGUF` from the
    /// pure-`Read` safetensors path: the parser reads the front matter
    /// linearly, but the underlying `read_exact` machinery may issue
    /// internal seeks to satisfy partial reads. A `Cursor` answers them
    /// trivially; an HTTP-range adapter that prefetches the front-matter
    /// region would do the same.
    #[test]
    fn inspect_from_reader_matches_path_mixed_dtypes() {
        // Two F32 tensors and one Q4_0 tensor, plus a metadata array of
        // strings. The Q4_0 tensor lives at relative offset 32 (well
        // beyond the first F32 at offset 0), so the reader-generic path
        // exercises the same alignment-and-bounds arithmetic as the
        // mmap-backed path.
        let mut b = GgufBuilder::new();
        b.push_bytes(b"GGUF");
        b.push_u32(3);
        b.push_u64(3);
        b.push_u64(3);
        b.push_kv_string("general.architecture", "llama");
        b.push_kv_uint32("general.alignment", 32);
        // `tokenizer.ggml.tokens` is the canonical large-array key in real
        // GGUFs; here we keep it tiny but exercise the same code path.
        b.push_string("tokenizer.ggml.tokens");
        b.push_u32(9); // ARRAY
        b.push_u32(8); // STRING
        b.push_u64(2); // length
        b.push_string("<bos>");
        b.push_string("<eos>");
        // tensor 0 — F32 [4, 4] = 64 bytes at relative offset 0
        b.push_tensor_info("a", &[4, 4], 0, 0);
        // tensor 1 — F32 [2] = 8 bytes at relative offset 64
        b.push_tensor_info("b", &[2], 0, 64);
        // tensor 2 — Q4_0 [64] = 36 bytes at relative offset 96
        // (64 + 8 = 72, padded up to 96 for 32-byte alignment)
        b.push_tensor_info("c", &[64], 2, 96);

        b.pad_to_alignment(32);
        // tensor.a data — 64 bytes
        b.push_bytes(&[0u8; 64]);
        // tensor.b data — 8 bytes (no padding needed; 64 + 8 = 72)
        b.push_bytes(&[0u8; 8]);
        // pad to next 32-byte boundary (72 → 96)
        b.pad_to_alignment(32);
        // tensor.c data — 36 bytes
        b.push_bytes(&[0u8; 36]);

        let bytes = b.finish();
        let tmp = write_temp_gguf(&bytes);

        let path_info = parse_gguf(tmp.path()).unwrap().inspect();
        let reader_info = inspect_gguf_from_reader(std::io::Cursor::new(&bytes)).unwrap();

        assert_inspect_eq(&path_info, &reader_info);

        // Ground-truth values to lock the test to known-correct output.
        assert_eq!(reader_info.tensor_count, 3);
        // tensor.a: 64 B + tensor.b: 8 B + tensor.c: 36 B = 108 B
        assert_eq!(reader_info.total_bytes, 64 + 8 + 36);
        // First-occurrence dtype order: F32 (twice), then Q4_0.
        assert_eq!(reader_info.dtypes, vec![GgufType::F32, GgufType::Q4_0]);
        assert_eq!(reader_info.architecture.as_deref(), Some("llama"));
    }

    // -----------------------------------------------------------------
    // Reader-generic full front matter — substrate-equivalence tests
    // -----------------------------------------------------------------

    /// Asserts every field of a parsed [`GgufTensorInfo`] pair is equal.
    fn assert_tensor_info_eq(path: &GgufTensorInfo, reader: &GgufTensorInfo) {
        assert_eq!(path.name, reader.name);
        assert_eq!(path.shape, reader.shape);
        assert_eq!(path.dtype, reader.dtype);
        assert_eq!(path.data_offset, reader.data_offset);
        assert_eq!(path.byte_len, reader.byte_len);
    }

    /// `parse_gguf_front_matter_from_reader` over an in-memory `Cursor`
    /// returns the same full tensor list and metadata table as
    /// `parse_gguf(path)` over the same archive on disk. Locks the contract
    /// that the reader-generic full-detail path and the mmap-backed path
    /// are substrate-equivalent — this is the property `hf-fm`'s remote
    /// `GGUF` inspect relies on for tensor-table parity with the cached path.
    #[test]
    fn front_matter_from_reader_matches_path_minimal() {
        let bytes = build_minimal_gguf();
        let tmp = write_temp_gguf(&bytes);

        let path_parsed = parse_gguf(tmp.path()).unwrap();
        let reader_front =
            parse_gguf_front_matter_from_reader(std::io::Cursor::new(&bytes)).unwrap();

        assert_eq!(path_parsed.version(), reader_front.version);
        assert_eq!(path_parsed.alignment(), reader_front.alignment);
        assert_eq!(path_parsed.metadata(), &reader_front.metadata);
        assert_eq!(
            path_parsed.tensor_info().len(),
            reader_front.tensor_infos.len()
        );
        for (path_info, reader_info) in path_parsed
            .tensor_info()
            .iter()
            .zip(&reader_front.tensor_infos)
        {
            assert_tensor_info_eq(path_info, reader_info);
        }

        // Spot-check the fixture's known-correct values, so this isn't just
        // comparing two equal-but-wrong outputs.
        assert_eq!(reader_front.version, 3);
        assert_eq!(reader_front.tensor_infos.len(), 2);
        assert_eq!(reader_front.tensor_infos[0].name, "tensor.a");
        assert_eq!(reader_front.tensor_infos[0].shape, vec![2, 3]);
        assert_eq!(reader_front.tensor_infos[0].dtype, GgufType::F32);
        assert_eq!(reader_front.tensor_infos[1].name, "tensor.b");
        assert_eq!(reader_front.tensor_infos[1].dtype, GgufType::Q4_0);
    }

    /// `parse_gguf_front_matter_from_reader` accepts a header-only file (no
    /// tensors, no data section) — same as the path-based parser and as
    /// [`inspect_gguf_from_reader`].
    #[test]
    fn front_matter_from_reader_accepts_header_only_file() {
        let mut b = GgufBuilder::new();
        b.push_bytes(b"GGUF");
        b.push_u32(3);
        b.push_u64(0);
        b.push_u64(0);
        let bytes = b.finish();

        let front = parse_gguf_front_matter_from_reader(std::io::Cursor::new(&bytes)).unwrap();
        assert_eq!(front.version, 3);
        assert!(front.tensor_infos.is_empty());
        assert!(front.metadata.is_empty());
        assert_eq!(front.alignment, 32);
    }

    /// `parse_gguf_front_matter_from_reader` propagates the same parse
    /// errors as [`inspect_gguf_from_reader`] — both delegate to the same
    /// [`read_gguf_structure`] core, so neither should swallow or
    /// re-classify a rejection the other surfaces.
    #[test]
    fn front_matter_from_reader_propagates_parse_errors() {
        let err = parse_gguf_front_matter_from_reader(std::io::Cursor::new(b"GGU".as_slice()))
            .unwrap_err();
        assert!(matches!(err, AnamnesisError::Parse { .. }));

        let err = parse_gguf_front_matter_from_reader(std::io::Cursor::new(
            b"XXXX\x00\x00\x00\x00".as_slice(),
        ))
        .unwrap_err();
        assert!(matches!(err, AnamnesisError::Parse { .. }));

        let mut legacy = Vec::new();
        legacy.extend_from_slice(b"GGML");
        legacy.extend_from_slice(&[0u8; 100]);
        let err = parse_gguf_front_matter_from_reader(std::io::Cursor::new(&legacy)).unwrap_err();
        match err {
            AnamnesisError::Unsupported { format, detail } => {
                assert_eq!(format, "GGUF");
                assert!(detail.contains("GGML"));
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    /// Substrate-equivalence on the mixed-dtype, nested-array-metadata
    /// fixture also used by [`inspect_from_reader_matches_path_mixed_dtypes`]
    /// — the most adversarial-shape fixture in this module's test set,
    /// exercising the full tensor list rather than just the aggregate
    /// summary.
    #[test]
    fn front_matter_from_reader_matches_path_mixed_dtypes() {
        let mut b = GgufBuilder::new();
        b.push_bytes(b"GGUF");
        b.push_u32(3);
        b.push_u64(3);
        b.push_u64(3);
        b.push_kv_string("general.architecture", "llama");
        b.push_kv_uint32("general.alignment", 32);
        b.push_string("tokenizer.ggml.tokens");
        b.push_u32(9); // ARRAY
        b.push_u32(8); // STRING
        b.push_u64(2); // length
        b.push_string("<bos>");
        b.push_string("<eos>");
        b.push_tensor_info("a", &[4, 4], 0, 0);
        b.push_tensor_info("b", &[2], 0, 64);
        b.push_tensor_info("c", &[64], 2, 96);

        b.pad_to_alignment(32);
        b.push_bytes(&[0u8; 64]);
        b.push_bytes(&[0u8; 8]);
        b.pad_to_alignment(32);
        b.push_bytes(&[0u8; 36]);

        let bytes = b.finish();
        let tmp = write_temp_gguf(&bytes);

        let path_parsed = parse_gguf(tmp.path()).unwrap();
        let reader_front =
            parse_gguf_front_matter_from_reader(std::io::Cursor::new(&bytes)).unwrap();

        assert_eq!(path_parsed.metadata(), &reader_front.metadata);
        assert_eq!(
            path_parsed.tensor_info().len(),
            reader_front.tensor_infos.len()
        );
        for (path_info, reader_info) in path_parsed
            .tensor_info()
            .iter()
            .zip(&reader_front.tensor_infos)
        {
            assert_tensor_info_eq(path_info, reader_info);
        }

        assert_eq!(reader_front.tensor_infos.len(), 3);
        assert_eq!(reader_front.tensor_infos[2].name, "c");
        assert_eq!(reader_front.tensor_infos[2].dtype, GgufType::Q4_0);
    }

    /// `GgufFrontMatter::inspect` reduces to the same [`GgufInspectInfo`] as
    /// [`inspect_gguf_from_reader`] on identical bytes — ties the two
    /// reader-generic entry points together via the shared
    /// [`build_inspect_info`] helper.
    #[test]
    fn front_matter_inspect_matches_inspect_gguf_from_reader() {
        let bytes = build_minimal_gguf();

        let front = parse_gguf_front_matter_from_reader(std::io::Cursor::new(&bytes)).unwrap();
        let summary_via_front = front.inspect();
        let summary_direct = inspect_gguf_from_reader(std::io::Cursor::new(&bytes)).unwrap();

        assert_inspect_eq(&summary_via_front, &summary_direct);
    }
}
