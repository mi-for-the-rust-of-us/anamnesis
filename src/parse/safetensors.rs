// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;
use std::fmt;
use std::io::Read;

use crate::ParseLimits;
use crate::error::AnamnesisError;
use crate::limits::Budget;

/// Sanity cap on the safetensors header length declared by the 8-byte
/// little-endian prefix.
///
/// The safetensors specification does not enforce a maximum, but a fixed
/// upper bound prevents an adversarial reader from triggering an arbitrary
/// allocation by claiming a huge header. 100 MiB is two orders of magnitude
/// beyond any real header (a 100 k-tensor shard ships well under 1 MiB of
/// `JSON`).
const MAX_SAFETENSORS_HEADER_BYTES: u64 = 100 * 1024 * 1024;

/// Validates a declared safetensors header length against the permanent
/// [`MAX_SAFETENSORS_HEADER_BYTES`] cap and the caller's `budget`, **before**
/// the header is parsed or allocated.
///
/// Shared by the slice-based [`parse_safetensors_header_with_limits`] and the
/// reader-based [`parse_safetensors_header_from_reader_with_limits`] so the
/// bound is identical by construction and applied pre-allocation on both
/// paths (the slice path reads the same 8-byte prefix that
/// `safetensors::SafeTensors::read_metadata` consumes, so it can reject an
/// over-budget header before the upstream parser allocates the metadata).
///
/// # Errors
///
/// Returns [`AnamnesisError::LimitExceeded`] if `header_len` exceeds the cap or
/// the caller's `budget` (per-item single-allocation cap + cumulative aggregate).
fn enforce_safetensors_header_cap(header_len: u64, budget: &mut Budget) -> crate::Result<()> {
    if header_len > MAX_SAFETENSORS_HEADER_BYTES {
        return Err(AnamnesisError::LimitExceeded {
            limit: "MAX_SAFETENSORS_HEADER_BYTES",
            message: format!(
                "safetensors header length {header_len} exceeds \
                 {MAX_SAFETENSORS_HEADER_BYTES}-byte cap"
            ),
        });
    }
    // Caller-supplied ceiling (per-item single-alloc + cumulative aggregate),
    // layered on top of the permanent cap above.
    budget.charge_alloc(header_len, "safetensors header")?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Dtype
// ---------------------------------------------------------------------------

/// Element data type as parsed from a `.safetensors` header.
///
/// This is anamnesis's own enum, decoupled from `safetensors::Dtype`, so that
/// we can add helper methods and remain insulated from upstream changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Dtype {
    /// 8-bit floating point, 4-bit exponent, 3-bit mantissa.
    F8E4M3,
    /// 8-bit floating point, 5-bit exponent, 2-bit mantissa.
    F8E5M2,
    /// 16-bit brain floating point.
    BF16,
    /// 16-bit IEEE 754 half-precision.
    F16,
    /// 32-bit IEEE 754 single-precision.
    F32,
    /// 64-bit IEEE 754 double-precision.
    F64,
    /// Boolean (1 byte per element in safetensors).
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
}

impl Dtype {
    /// Returns the number of bytes per element for this dtype.
    #[must_use]
    pub const fn byte_size(self) -> usize {
        match self {
            Self::Bool | Self::U8 | Self::I8 | Self::F8E4M3 | Self::F8E5M2 => 1,
            Self::U16 | Self::I16 | Self::F16 | Self::BF16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 | Self::I64 | Self::F64 => 8,
        }
    }

    /// Returns `true` if this dtype represents a quantized format requiring
    /// dequantization (`F8_E4M3` or `F8_E5M2`).
    #[must_use]
    pub const fn is_quantized(self) -> bool {
        matches!(self, Self::F8E4M3 | Self::F8E5M2)
    }

    /// Returns `true` if this dtype is a floating-point type.
    #[must_use]
    pub const fn is_floating_point(self) -> bool {
        matches!(
            self,
            Self::F8E4M3 | Self::F8E5M2 | Self::BF16 | Self::F16 | Self::F32 | Self::F64
        )
    }

    /// Converts this dtype to the corresponding `safetensors::Dtype`.
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::Unsupported`] if the dtype has no
    /// corresponding `safetensors::Dtype` variant.
    pub fn to_safetensors_dtype(self) -> crate::Result<safetensors::Dtype> {
        match self {
            Self::F8E4M3 => Ok(safetensors::Dtype::F8_E4M3),
            Self::F8E5M2 => Ok(safetensors::Dtype::F8_E5M2),
            Self::BF16 => Ok(safetensors::Dtype::BF16),
            Self::F16 => Ok(safetensors::Dtype::F16),
            Self::F32 => Ok(safetensors::Dtype::F32),
            Self::F64 => Ok(safetensors::Dtype::F64),
            Self::Bool => Ok(safetensors::Dtype::BOOL),
            Self::U8 => Ok(safetensors::Dtype::U8),
            Self::I8 => Ok(safetensors::Dtype::I8),
            Self::U16 => Ok(safetensors::Dtype::U16),
            Self::I16 => Ok(safetensors::Dtype::I16),
            Self::U32 => Ok(safetensors::Dtype::U32),
            Self::I32 => Ok(safetensors::Dtype::I32),
            Self::U64 => Ok(safetensors::Dtype::U64),
            Self::I64 => Ok(safetensors::Dtype::I64),
        }
    }
}

impl fmt::Display for Dtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::F8E4M3 => "F8_E4M3",
            Self::F8E5M2 => "F8_E5M2",
            Self::BF16 => "BF16",
            Self::F16 => "F16",
            Self::F32 => "F32",
            Self::F64 => "F64",
            Self::Bool => "BOOL",
            Self::U8 => "U8",
            Self::I8 => "I8",
            Self::U16 => "U16",
            Self::I16 => "I16",
            Self::U32 => "U32",
            Self::I32 => "I32",
            Self::U64 => "U64",
            Self::I64 => "I64",
        };
        f.write_str(s)
    }
}

impl TryFrom<safetensors::Dtype> for Dtype {
    type Error = AnamnesisError;

    /// Converts a `safetensors::Dtype` into anamnesis's own `Dtype`.
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::Unsupported`] if the upstream crate introduces
    /// a dtype variant that anamnesis does not yet handle.
    fn try_from(st: safetensors::Dtype) -> std::result::Result<Self, Self::Error> {
        match st {
            safetensors::Dtype::F8_E4M3 => Ok(Self::F8E4M3),
            safetensors::Dtype::F8_E5M2 => Ok(Self::F8E5M2),
            safetensors::Dtype::BF16 => Ok(Self::BF16),
            safetensors::Dtype::F16 => Ok(Self::F16),
            safetensors::Dtype::F32 => Ok(Self::F32),
            safetensors::Dtype::F64 => Ok(Self::F64),
            safetensors::Dtype::BOOL => Ok(Self::Bool),
            safetensors::Dtype::U8 => Ok(Self::U8),
            safetensors::Dtype::I8 => Ok(Self::I8),
            safetensors::Dtype::U16 => Ok(Self::U16),
            safetensors::Dtype::I16 => Ok(Self::I16),
            safetensors::Dtype::U32 => Ok(Self::U32),
            safetensors::Dtype::I32 => Ok(Self::I32),
            safetensors::Dtype::U64 => Ok(Self::U64),
            safetensors::Dtype::I64 => Ok(Self::I64),
            // safetensors 0.8 added sub-8-bit floats (`F4`/`F6_*`), the MXFP8
            // block scale (`F8_E8M0`), the fp8 "fnuz" variants, and `C64`
            // (complex64). anamnesis has no dequant path for any of these, so
            // reject them explicitly (Rule 7: exhaustive matching) rather than
            // letting the future-proofing wildcard below absorb known variants.
            safetensors::Dtype::F4
            | safetensors::Dtype::F6_E2M3
            | safetensors::Dtype::F6_E3M2
            | safetensors::Dtype::F8_E8M0
            | safetensors::Dtype::F8_E4M3FNUZ
            | safetensors::Dtype::F8_E5M2FNUZ
            | safetensors::Dtype::C64 => Err(AnamnesisError::Unsupported {
                format: "safetensors".into(),
                detail: format!("unsupported safetensors dtype {st:?}"),
            }),
            // safetensors::Dtype is #[non_exhaustive]; handle future additions.
            unknown => Err(AnamnesisError::Unsupported {
                format: "safetensors".into(),
                detail: format!("unknown dtype {unknown:?}"),
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// TensorRole
// ---------------------------------------------------------------------------

/// Classification of a tensor's role in a quantized model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TensorRole {
    /// Quantized weight tensor requiring dequantization.
    Quantized,
    /// Scale factor tensor (companion to a quantized weight).
    Scale,
    /// Passthrough tensor (norms, embeddings, `lm_head`) — already full-precision.
    Passthrough,
    /// Zero-point tensor (`GPTQ` `.qzeros` — packed integer zero-points).
    ZeroPoint,
    /// Group index tensor (`GPTQ` `.g_idx` — maps input features to groups).
    GroupIndex,
    /// Quantization lookup table (`BnB` `.weight.quant_map` / `.weight.nested_quant_map`).
    QuantMap,
    /// Nested absmax scale (`BnB` double-quant `.weight.nested_absmax`).
    NestedScale,
    /// `BnB` quantization state metadata (`.quant_state.bitsandbytes__nf4` / `__fp4`).
    /// Contains a `JSON` blob with the original tensor shape, block size, and dtype.
    QuantState,
}

/// Classify a tensor based on its name and dtype.
///
/// FP8 rules are checked first (suffix `_scale_inv` / `_scale`, then dtype).
/// GPTQ rules (`.qweight`, `.qzeros`, `.scales`, `.g_idx`) are checked when
/// the `gptq` feature is enabled.
///
/// The verdict is per tensor and by name, so a `Scale` here is provisional:
/// `mark_modelopt_nvfp4` then claims the layers only a whole-file view can
/// recognise, and `demote_orphan_scales` returns every scale no quantised
/// tensor owns to `Passthrough`.
fn classify_tensor(name: &str, dtype: Dtype) -> TensorRole {
    // FP8 scale companions (suffix-based, always active)
    if name.ends_with("_scale_inv") || name.ends_with("_scale") {
        return TensorRole::Scale;
    }

    // GPTQ / AWQ shared tensor patterns (both use `.qweight`, `.qzeros`, `.scales`)
    #[cfg(any(feature = "gptq", feature = "awq"))]
    {
        if name.ends_with(".qweight") {
            return TensorRole::Quantized;
        }
        if name.ends_with(".qzeros") {
            return TensorRole::ZeroPoint;
        }
        if name.ends_with(".scales") {
            return TensorRole::Scale;
        }
    }

    // GPTQ-only: `.g_idx` maps input features to groups (AWQ uses sequential groups)
    #[cfg(feature = "gptq")]
    if name.ends_with(".g_idx") {
        return TensorRole::GroupIndex;
    }

    // BitsAndBytes tensor patterns (name-based, feature-gated)
    #[cfg(feature = "bnb")]
    {
        // Quantization state metadata (JSON blob with original shape, blocksize, dtype).
        // Checked first — the name contains `.weight.quant_state.bitsandbytes__` which
        // would otherwise match the `.weight` suffix check below.
        if name.contains(".quant_state.bitsandbytes__") {
            return TensorRole::QuantState;
        }
        // NF4/FP4 companions (checked before weight to avoid false positives)
        if name.ends_with(".weight.nested_quant_map") || name.ends_with(".weight.quant_map") {
            return TensorRole::QuantMap;
        }
        if name.ends_with(".weight.nested_absmax") {
            return TensorRole::NestedScale;
        }
        if name.ends_with(".weight.absmax") {
            return TensorRole::Scale;
        }
        // INT8 companion (per-row scale)
        // Not a file extension — `.SCB` is a BnB tensor name suffix.
        #[allow(clippy::case_sensitive_file_extension_comparisons)]
        if name.ends_with(".SCB") {
            return TensorRole::Scale;
        }
        // NF4/FP4 quantized weight: U8 dtype, flattened [N, 1] shape, and has
        // a `.weight.quant_map` companion (but we check dtype + shape here;
        // the companion is verified during scheme detection).
        if dtype == Dtype::U8 && name.ends_with(".weight") {
            return TensorRole::Quantized;
        }
        // INT8 quantized weight: I8 dtype with a `.SCB` companion.
        if dtype == Dtype::I8 && name.ends_with(".weight") {
            return TensorRole::Quantized;
        }
    }

    // FP8 quantized by dtype
    if dtype.is_quantized() {
        TensorRole::Quantized
    } else {
        TensorRole::Passthrough
    }
}

// ---------------------------------------------------------------------------
// QuantScheme
// ---------------------------------------------------------------------------

/// Detected quantization scheme for a `.safetensors` file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum QuantScheme {
    /// Fine-grained `FP8` with 128×128 block scale factors (`_scale_inv` companions).
    FineGrainedFp8,
    /// Per-channel `FP8` with one scale factor per output row (shape `[rows, 1]`).
    PerChannelFp8,
    /// Per-tensor `FP8` with a single scale factor per tensor (or no explicit companion).
    PerTensorFp8,
    /// No quantization detected — all tensors are passthrough.
    Unquantized,
    /// `GPTQ` quantization (INT4 or INT8 with group-wise scale + zero-point).
    Gptq,
    /// `AWQ` quantization (activation-aware, INT4 or INT8 with per-group scales).
    Awq,
    /// `BitsAndBytes` 4-bit quantization (`NF4` or `FP4` with per-block absmax).
    Bnb4,
    /// `BitsAndBytes` `INT8` quantization (`LLM.int8()` with per-row absmax).
    BnbInt8,
    /// NVIDIA `NVFP4` as `TensorRT` Model Optimizer exports it: `U8` weights
    /// packing two 4-bit `E2M1` values per byte, an `F8_E4M3` `weight_scale`
    /// per 16 values and an `F32` `weight_scale_2` per tensor. Recognised in
    /// every build; dequantised with the `nvfp4` feature, and refused by name
    /// ([`AnamnesisError::Unsupported`]) without it. Until Phase 7.10 it was
    /// misread as fine-grained `FP8`, or as unquantised without `bnb`.
    Nvfp4,
}

impl fmt::Display for QuantScheme {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::FineGrainedFp8 => "Fine-grained FP8 (E4M3), 128x128 blocks",
            Self::PerChannelFp8 => "Per-channel FP8 (E4M3), one scale per row",
            Self::PerTensorFp8 => "Per-tensor FP8 (E4M3)",
            Self::Unquantized => "Unquantized",
            Self::Gptq => "GPTQ",
            Self::Awq => "AWQ",
            Self::Bnb4 => "BitsAndBytes NF4/FP4 (4-bit, per-block absmax)",
            Self::BnbInt8 => "BitsAndBytes INT8 (LLM.int8(), per-row absmax)",
            Self::Nvfp4 => "NVIDIA NVFP4 (ModelOpt), 16-value blocks",
        };
        f.write_str(s)
    }
}

/// Detect the quantization scheme from a list of classified tensor entries.
///
/// **Assumption:** all quantized tensors in a single `.safetensors` file use
/// the same scheme. This holds for every known quantizer (LG AI, Qwen, Mistral,
/// `RedHat`, NVIDIA). The function early-returns on the first scale companion
/// found — if a file ever mixed schemes, the result would reflect only the
/// first match.
///
/// All `FP8` schemes may use `_scale_inv` or `_scale` companions.
/// The distinction is the **scale tensor shape**:
/// - Fine-grained: 2D with both dims > 1 (e.g., `[16, 32]` for 128×128 blocks)
/// - Per-channel: 2D with second dim = 1 (e.g., `[2048, 1]`, one scale per row)
/// - Per-tensor: scalar `[]` or 1D `[1]`
fn detect_scheme(entries: &[TensorEntry], index: &NameIndex<'_>) -> QuantScheme {
    let has_quantized = entries.iter().any(|e| e.role == TensorRole::Quantized);
    if !has_quantized {
        return QuantScheme::Unquantized;
    }

    // NVIDIA ModelOpt NVFP4, marked by `mark_modelopt_nvfp4`. Checked first: its
    // `U8` weights would otherwise read as `BnB` 4-bit, and its 2-D `F8_E4M3`
    // `weight_scale` as a fine-grained FP8 block scale.
    if entries
        .iter()
        .any(|e| e.role == TensorRole::Quantized && is_modelopt_nvfp4_weight(e, index))
    {
        return QuantScheme::Nvfp4;
    }

    // GPTQ / AWQ: both use `.qweight` tensors. Distinguish by packing direction.
    // GPTQ packs along rows: qweight.cols == scales.cols (both = out_features).
    // AWQ packs along cols: qweight.cols < scales.cols (qweight.cols * pack_factor = scales.cols).
    // Detection is unconditional — feature-disabled errors are handled in model.rs.
    for entry in entries
        .iter()
        .filter(|e| e.role == TensorRole::Quantized && e.name.ends_with(".qweight"))
    {
        let base = entry.name.strip_suffix(".qweight");
        if let Some(base) = base {
            let scales_name = format!("{base}.scales");
            if let Some(scales) = index.named(&scales_name) {
                let qw_cols = entry.shape.last().copied().unwrap_or(0);
                let sc_cols = scales.shape.last().copied().unwrap_or(0);

                if qw_cols > 0 && sc_cols > 0 && qw_cols == sc_cols {
                    // qweight.cols == scales.cols → GPTQ (packed along rows)
                    return QuantScheme::Gptq;
                } else if qw_cols > 0 && sc_cols > 0 && qw_cols < sc_cols {
                    // qweight.cols < scales.cols → AWQ (packed along cols)
                    return QuantScheme::Awq;
                }
            }
        }
    }

    // BitsAndBytes: detect by companion tensor naming patterns.
    // NF4/FP4: `.weight.quant_map` (F32[16] lookup table) is the definitive marker.
    // INT8: `.SCB` (F32 per-row absmax) with I8 weight.
    #[cfg(feature = "bnb")]
    {
        let has_quant_map = entries.iter().any(|e| e.role == TensorRole::QuantMap);
        if has_quant_map {
            return QuantScheme::Bnb4;
        }
        let has_scb = entries.iter().any(|e| {
            // Not a file extension — `.SCB` is a BnB tensor name suffix.
            #[allow(clippy::case_sensitive_file_extension_comparisons)]
            let is_scb = e.name.ends_with(".SCB");
            e.role == TensorRole::Scale && is_scb
        });
        if has_scb {
            return QuantScheme::BnbInt8;
        }
    }

    // FP8: find the first scale companion for any quantized tensor and inspect its shape.
    // We check both `_scale_inv` and `_scale` suffixes.
    for entry in entries.iter().filter(|e| e.role == TensorRole::Quantized) {
        for suffix in &["_scale_inv", "_scale"] {
            let expected = format!("{}{suffix}", entry.name);
            if let Some(scale) = index
                .named(&expected)
                .filter(|s| s.role == TensorRole::Scale)
            {
                // 2D scale with both dims > 1 → fine-grained block scales
                // shape.len() >= 2 guarantees .last() is Some
                if scale.shape.len() >= 2 {
                    if scale.shape.last().copied() > Some(1) {
                        return QuantScheme::FineGrainedFp8;
                    }
                    // [N, 1] → per-channel (one scale per row)
                    return QuantScheme::PerChannelFp8;
                }
                // scalar [] or 1D [1] → per-tensor
                return QuantScheme::PerTensorFp8;
            }
        }
    }

    // Quantized tensors exist but no scale companions found at all
    QuantScheme::PerTensorFp8
}

// ---------------------------------------------------------------------------
// GPTQ config (feature-gated types, unconditional struct definition)
// ---------------------------------------------------------------------------

/// `GPTQ` quantization configuration inferred from safetensors metadata
/// and/or tensor shapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct GptqConfig {
    /// Quantization bit width (4 or 8).
    pub bits: u8,
    /// Number of input features per group (typically 128).
    pub group_size: usize,
}

/// Companion tensors for a `GPTQ` `.qweight` tensor.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct GptqCompanions<'a> {
    /// Per-group scale factors (`.scales`).
    pub scales: &'a TensorEntry,
    /// Per-group packed zero-points (`.qzeros`).
    pub qzeros: &'a TensorEntry,
    /// Optional group index mapping (`.g_idx`).
    pub g_idx: Option<&'a TensorEntry>,
}

// ---------------------------------------------------------------------------
// AWQ config
// ---------------------------------------------------------------------------

/// `AWQ` quantization configuration inferred from safetensors metadata
/// and/or tensor shapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct AwqConfig {
    /// Quantization bit width (4 or 8).
    pub bits: u8,
    /// Number of input features per group (typically 128).
    pub group_size: usize,
}

/// Companion tensors for an `AWQ` `.qweight` tensor.
///
/// Same tensor names as `GPTQ` (`.scales`, `.qzeros`) but different packing
/// direction (packed along `out_features` instead of `in_features`).
/// `AWQ` never has `.g_idx`.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct AwqCompanions<'a> {
    /// Per-group scale factors (`.scales`).
    pub scales: &'a TensorEntry,
    /// Per-group packed zero-points (`.qzeros`).
    pub qzeros: &'a TensorEntry,
}

// ---------------------------------------------------------------------------
// BnB config
// ---------------------------------------------------------------------------

/// `BitsAndBytes` 4-bit (`NF4`/`FP4`) quantization configuration.
///
/// Inferred from tensor shapes. The `quant_map` distinguishes `NF4` from `FP4`
/// (different lookup table values), but both use the same dequantization code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct BnbConfig {
    /// Block size for absmax quantization (typically 64).
    pub block_size: usize,
    /// Whether the model uses double quantization (nested absmax).
    pub double_quant: bool,
}

/// Companion tensors for a `BitsAndBytes` `NF4`/`FP4` `.weight` tensor.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Bnb4Companions<'a> {
    /// Per-block absolute maximum values (`.weight.absmax`).
    /// `F32` for plain `NF4`/`FP4`, `U8` for double-quant.
    pub absmax: &'a TensorEntry,
    /// 4-bit → `f32` lookup table (`.weight.quant_map`, `F32[16]`).
    pub quant_map: &'a TensorEntry,
    /// Nested absmax for double quantization (`.weight.nested_absmax`, `F32`).
    pub nested_absmax: Option<&'a TensorEntry>,
    /// Nested lookup table for double quantization (`.weight.nested_quant_map`, `F32[256]`).
    pub nested_quant_map: Option<&'a TensorEntry>,
    /// Quantization state metadata (`.weight.quant_state.bitsandbytes__nf4` / `__fp4`).
    /// Contains a `JSON` blob with the original tensor shape.
    pub quant_state: Option<&'a TensorEntry>,
}

// ---------------------------------------------------------------------------
// TensorEntry
// ---------------------------------------------------------------------------

/// Metadata for a single tensor parsed from a `.safetensors` header.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct TensorEntry {
    /// Tensor name as it appears in the header
    /// (e.g., `"model.layers.0.self_attn.q_proj.weight"`).
    pub name: String,
    /// Element data type (e.g., `F8E4M3`, `BF16`).
    pub dtype: Dtype,
    /// Tensor dimensions (e.g., `[2048, 2048]`).
    pub shape: Vec<usize>,
    /// Byte offset range `[start, end)` within the data section of the file.
    pub data_offsets: (usize, usize),
    /// Classification of this tensor's role in the model.
    pub role: TensorRole,
}

impl TensorEntry {
    /// Returns the total number of elements in the tensor.
    ///
    /// Saturates to `usize::MAX` if the shape's element count overflows
    /// `usize` (e.g., a malformed or adversarial header that declares
    /// `[u32::MAX, 2]` on a 32-bit target), and is `0` for any shape with a
    /// zero dimension, even one that follows an overflowing dimension.
    /// Shares one helper with the `NPZ` and `.pth` inspect paths, so the
    /// formats cannot disagree, and never wraps the way an unguarded
    /// `shape.iter().product()` would.
    #[must_use]
    pub fn num_elements(&self) -> usize {
        crate::parse::utils::saturating_num_elements(&self.shape)
    }

    /// Returns the byte length of the tensor's data (`end - start` offset).
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.data_offsets.1.saturating_sub(self.data_offsets.0)
    }
}

// ---------------------------------------------------------------------------
// SafetensorsHeader
// ---------------------------------------------------------------------------

/// Parsed `.safetensors` header with tensor metadata and quantization scheme.
///
/// Produced by [`parse_safetensors_header`]. Contains all the information
/// needed to decide how to dequantize (remember) or inspect the file, without
/// having read any tensor data yet.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SafetensorsHeader {
    /// All tensors found in the header, sorted by name.
    pub tensors: Vec<TensorEntry>,
    /// Detected quantization scheme for the file.
    pub scheme: QuantScheme,
    /// Raw metadata from the `__metadata__` section, if present.
    pub metadata: Option<HashMap<String, String>>,
    /// Size of the JSON header in bytes (data begins at `header_size + 8`).
    pub header_size: usize,
    /// `GPTQ` config (bits, group size), if the scheme is `GPTQ`.
    pub gptq_config: Option<GptqConfig>,
    /// `AWQ` config (bits, group size), if the scheme is `AWQ`.
    pub awq_config: Option<AwqConfig>,
    /// `BnB` 4-bit config (block size, double-quant), if the scheme is `Bnb4`.
    pub bnb_config: Option<BnbConfig>,
}

impl SafetensorsHeader {
    /// Number of output elements `remember` produces from the quantized tensor
    /// `entry` of this header: the one size function `inspect`'s
    /// `dequantized_size` is built from, so the estimate a host gates on is the
    /// output it gets.
    ///
    /// Most schemes store one weight per element. Two pack several:
    ///
    /// - `BnB` 4-bit: two weights per `U8` byte.
    /// - `GPTQ` / `AWQ`: `32 / bits` weights per `I32` of `.qweight` (8 at
    ///   4-bit, 4 at 8-bit). Until v0.7.9 this was missed, and `inspect`
    ///   under-reported 4-bit models 8× (Phase 7.9, audit finding M-1). When
    ///   the config is absent or its bit width is not 4 or 8 (the fields are
    ///   public, so a caller can edit them), the estimate assumes 8 weights per
    ///   element: an over-estimate refuses more, never less.
    ///
    /// Saturating, because the result feeds a size estimate: an absurd shape
    /// reads as `u64::MAX`, which a gate treats as too big.
    pub(crate) fn dequantized_elements(&self, entry: &TensorEntry) -> u64 {
        // CAST: usize → u64, element and byte counts fit in u64
        #[allow(clippy::as_conversions)]
        let (elements, bytes) = (entry.num_elements() as u64, entry.byte_len() as u64);
        if matches!(self.scheme, QuantScheme::Bnb4 | QuantScheme::Nvfp4) && entry.dtype == Dtype::U8
        {
            return bytes.saturating_mul(2);
        }
        if entry.dtype != Dtype::I32 {
            return elements;
        }
        // `Some(bits)` for a packed scheme: `bits` is the configured width,
        // itself `None` when the config is absent.
        let packed_bits = if self.scheme == QuantScheme::Gptq {
            Some(self.gptq_config.as_ref().map(|c| c.bits))
        } else if self.scheme == QuantScheme::Awq {
            Some(self.awq_config.as_ref().map(|c| c.bits))
        } else {
            None
        };
        match packed_bits {
            None => elements,
            Some(Some(8)) => elements.saturating_mul(4),
            // 4-bit, or an absent / unusable width: assume the larger factor.
            Some(Some(_) | None) => elements.saturating_mul(8),
        }
    }

    /// Returns an iterator over quantized tensors.
    pub fn quantized_tensors(&self) -> impl Iterator<Item = &TensorEntry> {
        self.tensors
            .iter()
            .filter(|e| e.role == TensorRole::Quantized)
    }

    /// Returns an iterator over scale factor tensors.
    pub fn scale_tensors(&self) -> impl Iterator<Item = &TensorEntry> {
        self.tensors.iter().filter(|e| e.role == TensorRole::Scale)
    }

    /// Returns an iterator over passthrough tensors.
    pub fn passthrough_tensors(&self) -> impl Iterator<Item = &TensorEntry> {
        self.tensors
            .iter()
            .filter(|e| e.role == TensorRole::Passthrough)
    }

    /// Returns the number of quantized tensors.
    #[must_use]
    pub fn quantized_count(&self) -> usize {
        self.quantized_tensors().count()
    }

    /// Returns the number of scale factor tensors.
    #[must_use]
    pub fn scale_count(&self) -> usize {
        self.scale_tensors().count()
    }

    /// Returns the number of passthrough tensors.
    #[must_use]
    pub fn passthrough_count(&self) -> usize {
        self.passthrough_tensors().count()
    }

    /// Finds the scale tensor for a given weight tensor name.
    ///
    /// Looks for `{weight_name}_scale_inv` first, then `{weight_name}_scale`.
    ///
    /// A linear scan per call. The crate's own dequantisation looks every
    /// companion up through an index built once instead, so a header with many
    /// tensors does not cost `O(N²)`.
    #[must_use]
    pub fn find_scale_for(&self, weight_name: &str) -> Option<&TensorEntry> {
        scale_for(&self.tensors.as_slice(), weight_name)
    }

    /// Returns an iterator over zero-point tensors.
    pub fn zeropoint_tensors(&self) -> impl Iterator<Item = &TensorEntry> {
        self.tensors
            .iter()
            .filter(|e| e.role == TensorRole::ZeroPoint)
    }

    /// Returns the number of zero-point tensors.
    #[must_use]
    pub fn zeropoint_count(&self) -> usize {
        self.zeropoint_tensors().count()
    }

    /// Returns an iterator over group-index tensors.
    pub fn group_index_tensors(&self) -> impl Iterator<Item = &TensorEntry> {
        self.tensors
            .iter()
            .filter(|e| e.role == TensorRole::GroupIndex)
    }

    /// Returns the number of group-index tensors.
    #[must_use]
    pub fn group_index_count(&self) -> usize {
        self.group_index_tensors().count()
    }

    /// Finds the `GPTQ` companion tensors (`.scales`, `.qzeros`, optional `.g_idx`)
    /// for a given `.qweight` tensor name.
    ///
    /// Strips the `.qweight` suffix and looks up `{base}.scales`,
    /// `{base}.qzeros`, and `{base}.g_idx` by name.
    ///
    /// A linear scan per call, like [`find_scale_for`](Self::find_scale_for).
    #[must_use]
    pub fn find_gptq_companions(&self, qweight_name: &str) -> Option<GptqCompanions<'_>> {
        gptq_companions(&self.tensors.as_slice(), qweight_name)
    }

    /// Finds the `AWQ` companion tensors (`.scales`, `.qzeros`) for a given
    /// `.qweight` tensor name.
    ///
    /// Same tensor names as `GPTQ` but no `.g_idx` (AWQ always uses sequential groups).
    ///
    /// A linear scan per call, like [`find_scale_for`](Self::find_scale_for).
    #[must_use]
    pub fn find_awq_companions(&self, qweight_name: &str) -> Option<AwqCompanions<'_>> {
        awq_companions(&self.tensors.as_slice(), qweight_name)
    }

    /// Returns an iterator over quant-map tensors (`BnB` lookup tables).
    pub fn quant_map_tensors(&self) -> impl Iterator<Item = &TensorEntry> {
        self.tensors
            .iter()
            .filter(|e| e.role == TensorRole::QuantMap)
    }

    /// Returns the number of quant-map tensors.
    #[must_use]
    pub fn quant_map_count(&self) -> usize {
        self.quant_map_tensors().count()
    }

    /// Returns an iterator over nested-scale tensors (`BnB` double-quant absmax).
    pub fn nested_scale_tensors(&self) -> impl Iterator<Item = &TensorEntry> {
        self.tensors
            .iter()
            .filter(|e| e.role == TensorRole::NestedScale)
    }

    /// Returns the number of nested-scale tensors.
    #[must_use]
    pub fn nested_scale_count(&self) -> usize {
        self.nested_scale_tensors().count()
    }

    /// Finds the `BnB` `NF4`/`FP4` companion tensors for a given quantized
    /// `.weight` tensor name.
    ///
    /// Looks up `{name}.absmax`, `{name}.quant_map`, and optionally
    /// `{name}.nested_absmax`, `{name}.nested_quant_map`, and
    /// `{name}.quant_state.bitsandbytes__*`.
    ///
    /// A linear scan per call, like [`find_scale_for`](Self::find_scale_for).
    #[must_use]
    pub fn find_bnb4_companions(&self, weight_name: &str) -> Option<Bnb4Companions<'_>> {
        bnb4_companions(&self.tensors.as_slice(), weight_name)
    }

    /// Finds the `BnB` `INT8` companion tensor (`.SCB`) for a given `.weight`
    /// tensor name.
    ///
    /// Strips the `.weight` suffix and looks up `{base}.SCB`.
    ///
    /// A linear scan per call, like [`find_scale_for`](Self::find_scale_for).
    #[must_use]
    pub fn find_bnb_int8_scb(&self, weight_name: &str) -> Option<&TensorEntry> {
        bnb_int8_scb(&self.tensors.as_slice(), weight_name)
    }
}

// ---------------------------------------------------------------------------
// Name lookup
// ---------------------------------------------------------------------------

/// Finds tensors by name, answering exactly as a first-match linear scan of the
/// tensor list would.
///
/// Two implementations: the tensor slice itself (a linear scan, what the public
/// `find_*` methods use for a one-off lookup) and [`NameIndex`] (a binary search
/// over an index built once, what the crate uses wherever one call looks up a
/// companion for every tensor). Until v0.7.9 every such loop scanned the whole
/// list per tensor, so 40 000 empty tensors cost 1.85 s and the 100 MiB header
/// cap admitted inputs costing about 40 minutes (Phase 7.9, audit finding M-7).
pub(crate) trait NameLookup<'a> {
    /// The first tensor named `name`.
    fn named(&self, name: &str) -> Option<&'a TensorEntry>;
    /// The first tensor whose name starts with `prefix`.
    fn first_with_prefix(&self, prefix: &str) -> Option<&'a TensorEntry>;
}

impl<'a> NameLookup<'a> for &'a [TensorEntry] {
    fn named(&self, name: &str) -> Option<&'a TensorEntry> {
        self.iter().find(|e| e.name == name)
    }

    fn first_with_prefix(&self, prefix: &str) -> Option<&'a TensorEntry> {
        self.iter().find(|e| e.name.starts_with(prefix))
    }
}

/// A name index over a tensor list, built once in `O(N log N)` and queried in
/// `O(log N)`.
///
/// It records positions rather than assuming the list is sorted: the
/// [`SafetensorsHeader::tensors`] field is public, so a caller may have
/// reordered or edited it after parsing, and the answers must still be the
/// linear scan's. The sort is stable, so among equal names the first in list
/// order comes first, and a prefix query returns the lowest list position in
/// its range.
pub(crate) struct NameIndex<'a> {
    /// The tensor list the positions refer to.
    entries: &'a [TensorEntry],
    /// `(name, position in entries)`, stably sorted by name.
    sorted: Vec<(&'a str, usize)>,
}

impl<'a> NameIndex<'a> {
    /// Indexes `entries` by name.
    pub(crate) fn new(entries: &'a [TensorEntry]) -> Self {
        let mut sorted: Vec<(&'a str, usize)> = entries
            .iter()
            .enumerate()
            // BORROW: explicit `.as_str()`, indexing the borrowed names.
            .map(|(position, entry)| (entry.name.as_str(), position))
            .collect();
        sorted.sort_by(|a, b| a.0.cmp(b.0));
        Self { entries, sorted }
    }
}

impl<'a> NameLookup<'a> for NameIndex<'a> {
    fn named(&self, name: &str) -> Option<&'a TensorEntry> {
        let start = self.sorted.partition_point(|&(n, _)| n < name);
        self.sorted
            .get(start)
            .filter(|&&(n, _)| n == name)
            .and_then(|&(_, position)| self.entries.get(position))
    }

    fn first_with_prefix(&self, prefix: &str) -> Option<&'a TensorEntry> {
        let start = self.sorted.partition_point(|&(n, _)| n < prefix);
        self.sorted
            .get(start..)?
            .iter()
            .take_while(|(n, _)| n.starts_with(prefix))
            .map(|&(_, position)| position)
            .min()
            .and_then(|position| self.entries.get(position))
    }
}

/// The scale tensor of an `FP8` weight: `{weight_name}_scale_inv` first, then
/// `{weight_name}_scale`.
pub(crate) fn scale_for<'a>(
    tensors: &impl NameLookup<'a>,
    weight_name: &str,
) -> Option<&'a TensorEntry> {
    tensors
        .named(&format!("{weight_name}_scale_inv"))
        .or_else(|| tensors.named(&format!("{weight_name}_scale")))
}

/// The two scale tensors of a `ModelOpt` `NVFP4` weight: its per-block
/// `{weight_name}_scale` and per-tensor `{weight_name}_scale_2`.
#[cfg(feature = "nvfp4")]
pub(crate) fn nvfp4_companions<'a>(
    tensors: &impl NameLookup<'a>,
    weight_name: &str,
) -> Option<(&'a TensorEntry, &'a TensorEntry)> {
    Some((
        tensors.named(&format!("{weight_name}_scale"))?,
        tensors.named(&format!("{weight_name}_scale_2"))?,
    ))
}

/// The `GPTQ` companions of a `.qweight`: `.scales`, `.qzeros`, optional
/// `.g_idx`.
pub(crate) fn gptq_companions<'a>(
    tensors: &impl NameLookup<'a>,
    qweight_name: &str,
) -> Option<GptqCompanions<'a>> {
    let base = qweight_name.strip_suffix(".qweight")?;
    Some(GptqCompanions {
        scales: tensors.named(&format!("{base}.scales"))?,
        qzeros: tensors.named(&format!("{base}.qzeros"))?,
        g_idx: tensors.named(&format!("{base}.g_idx")),
    })
}

/// The `AWQ` companions of a `.qweight`: `.scales` and `.qzeros`.
pub(crate) fn awq_companions<'a>(
    tensors: &impl NameLookup<'a>,
    qweight_name: &str,
) -> Option<AwqCompanions<'a>> {
    let base = qweight_name.strip_suffix(".qweight")?;
    Some(AwqCompanions {
        scales: tensors.named(&format!("{base}.scales"))?,
        qzeros: tensors.named(&format!("{base}.qzeros"))?,
    })
}

/// The `BnB` 4-bit companions of a `.weight`.
pub(crate) fn bnb4_companions<'a>(
    tensors: &impl NameLookup<'a>,
    weight_name: &str,
) -> Option<Bnb4Companions<'a>> {
    Some(Bnb4Companions {
        absmax: tensors.named(&format!("{weight_name}.absmax"))?,
        quant_map: tensors.named(&format!("{weight_name}.quant_map"))?,
        nested_absmax: tensors.named(&format!("{weight_name}.nested_absmax")),
        nested_quant_map: tensors.named(&format!("{weight_name}.nested_quant_map")),
        // The quant-state name ends in the scheme: `…bitsandbytes__nf4` or `__fp4`.
        quant_state: tensors
            .first_with_prefix(&format!("{weight_name}.quant_state.bitsandbytes__")),
    })
}

/// The `BnB` `INT8` companion (`.SCB`) of a `.weight`.
pub(crate) fn bnb_int8_scb<'a>(
    tensors: &impl NameLookup<'a>,
    weight_name: &str,
) -> Option<&'a TensorEntry> {
    let base = weight_name.strip_suffix(".weight")?;
    tensors.named(&format!("{base}.SCB"))
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

/// Infer `AWQ` configuration from tensor shapes.
///
/// `AWQ` packs along `out_features`: `qweight` shape is `[in_features, out_features / pack_factor]`.
/// - `bits = 32 / (out_features / qweight.cols)` where `out_features = scales.cols`
/// - `group_size = in_features / scales.rows`
fn infer_awq_config(entries: &[TensorEntry], index: &NameIndex<'_>) -> Option<AwqConfig> {
    for entry in entries
        .iter()
        .filter(|e| e.role == TensorRole::Quantized && e.name.ends_with(".qweight"))
    {
        let base = entry.name.strip_suffix(".qweight")?;
        let scales_name = format!("{base}.scales");
        if let Some(scales) = index.named(&scales_name)
            && entry.shape.len() >= 2
            && scales.shape.len() >= 2
        {
            let in_features = entry.shape.first().copied()?;
            let qw_cols = entry.shape.last().copied()?;
            let num_groups = scales.shape.first().copied()?;
            let out_features = scales.shape.last().copied()?;

            if qw_cols == 0 || out_features == 0 || num_groups == 0 || in_features == 0 {
                return None;
            }

            // AWQ: out_features = qw_cols * pack_factor → pack_factor = out_features / qw_cols
            if out_features.is_multiple_of(qw_cols) {
                let pack_factor = out_features / qw_cols;
                for bits in [4u8, 8] {
                    // CAST: u8 → usize, bits is 4 or 8
                    #[allow(clippy::as_conversions)]
                    let expected_pf = 32 / bits as usize;
                    if pack_factor == expected_pf && in_features.is_multiple_of(num_groups) {
                        let group_size = in_features / num_groups;
                        return Some(AwqConfig { bits, group_size });
                    }
                }
            }
        }
    }
    None
}

/// Infer `GPTQ` configuration from safetensors `__metadata__` or tensor shapes.
///
/// **Primary source:** `AutoGPTQ`-style metadata keys `gptq_bits` and
/// `gptq_group_size` in the safetensors `__metadata__` section.
///
/// **Fallback:** infer from the first `.qweight` / `.scales` pair:
/// - `bits = 32 / (in_features / qweight.rows)` where `in_features = scales.cols`
/// - `group_size = in_features / scales.rows`
fn infer_gptq_config(
    entries: &[TensorEntry],
    index: &NameIndex<'_>,
    metadata: Option<&HashMap<String, String>>,
) -> Option<GptqConfig> {
    // Try metadata first (AutoGPTQ format).
    if let Some(meta) = metadata {
        // Only the widths the kernel implements; anything else (a hostile
        // `"0"` divided `32 / bits` downstream) falls through to shape
        // inference exactly as a missing key does.
        let bits = meta
            .get("gptq_bits")
            .and_then(|v| v.parse::<u8>().ok())
            .filter(|b| matches!(b, 4 | 8));
        let group_size = meta
            .get("gptq_group_size")
            .and_then(|v| v.parse::<usize>().ok());
        if let (Some(bits), Some(group_size)) = (bits, group_size) {
            return Some(GptqConfig { bits, group_size });
        }
    }

    // Fallback: infer from tensor shapes.
    // Find the first .qweight and its companion .scales.
    for entry in entries
        .iter()
        .filter(|e| e.role == TensorRole::Quantized && e.name.ends_with(".qweight"))
    {
        let base = entry.name.strip_suffix(".qweight")?;
        let scales_name = format!("{base}.scales");
        if let Some(scales) = index.named(&scales_name) {
            // qweight shape: (in_features / pack_factor, out_features)
            // scales shape:  (num_groups, out_features)
            // in_features = scales shape's last dim tells us out_features;
            //               but we need in_features from g_idx or scales.rows * group_size.
            // pack_factor = qweight.rows tells us in_features / pack_factor.
            // out_features = qweight.cols = scales.cols
            // num_groups = scales.rows
            // in_features = qweight.rows * pack_factor
            // bits = 32 / pack_factor
            // group_size = in_features / num_groups

            if entry.shape.len() >= 2 && scales.shape.len() >= 2 {
                let qw_rows = entry.shape.first().copied()?;
                let num_groups = scales.shape.first().copied()?;
                let out_features = scales.shape.last().copied()?;

                if num_groups == 0 || qw_rows == 0 || out_features == 0 {
                    return None;
                }

                // Try each valid bit width to find one that yields integer in_features.
                for bits in [4u8, 8] {
                    // CAST: u8 → usize, bits is 4 or 8
                    #[allow(clippy::as_conversions)]
                    let pack_factor = 32 / bits as usize;
                    let in_features = qw_rows.checked_mul(pack_factor)?;
                    if in_features.is_multiple_of(num_groups) {
                        let group_size = in_features / num_groups;
                        return Some(GptqConfig { bits, group_size });
                    }
                }
            }
        }
    }

    None
}

/// Infer `BnB` 4-bit configuration from tensor shapes.
///
/// Block size is derived from `total_elements / absmax_count`:
/// - `total_elements = weight.byte_len() * 2` (2 nibbles per byte)
/// - `absmax_count = absmax.num_elements()`
/// - `block_size = total_elements / absmax_count`
///
/// Double-quant is detected by the presence of `.weight.nested_absmax`.
fn infer_bnb_config(entries: &[TensorEntry], index: &NameIndex<'_>) -> Option<BnbConfig> {
    // Find the first quantized weight with a .quant_map companion.
    for entry in entries
        .iter()
        .filter(|e| e.role == TensorRole::Quantized && e.dtype == Dtype::U8)
    {
        let absmax_name = format!("{}.absmax", entry.name);
        let nested_name = format!("{}.nested_absmax", entry.name);

        if let Some(absmax) = index.named(&absmax_name) {
            // total_elements = weight bytes × 2 (two NF4 values per byte)
            let total_elements = entry.byte_len().checked_mul(2)?;
            let absmax_count = absmax.num_elements();
            if absmax_count == 0 || total_elements % absmax_count != 0 {
                return None;
            }
            let block_size = total_elements / absmax_count;
            let double_quant = index.named(&nested_name).is_some();

            return Some(BnbConfig {
                block_size,
                double_quant,
            });
        }
    }
    None
}

/// Parses the header of a `.safetensors` file from a byte buffer.
///
/// Extracts all tensor metadata (names, shapes, dtypes, byte offsets),
/// classifies each tensor (quantized, scale, passthrough), and detects
/// the quantization scheme (fine-grained `FP8`, per-tensor `FP8`, `GPTQ`,
/// `AWQ`, `BnB`, or unquantized).
///
/// The buffer must contain at least the 8-byte length prefix, the full
/// `JSON` header, **and** the tensor data section (the upstream
/// `safetensors` crate validates buffer length against the largest data
/// offset declared by the header). Callers that have only the prefix +
/// `JSON` available — for example, an `HTTP`-range adapter that wants to
/// avoid downloading the data — should use
/// [`parse_safetensors_header_from_reader`] instead, which is built on a
/// `JSON`-only path that does not require the data section to be present.
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if the safetensors header is malformed.
///
/// Returns [`AnamnesisError::Unsupported`] if a tensor uses an unrecognized dtype.
/// Returns [`AnamnesisError::LimitExceeded`] if the declared header exceeds the
/// permanent 100 MiB cap (`MAX_SAFETENSORS_HEADER_BYTES`, always-on).
///
/// # Memory
///
/// Allocates a `Vec<TensorEntry>` proportional to the number of tensors in the
/// header (typically hundreds). No tensor data is copied or read.
pub fn parse_safetensors_header(buffer: &[u8]) -> crate::Result<SafetensorsHeader> {
    parse_safetensors_header_with_limits(buffer, &ParseLimits::default())
}

/// Parses a safetensors header under a caller-supplied [`ParseLimits`] budget.
///
/// Identical to [`parse_safetensors_header`] but enforces every applicable
/// [`ParseLimits`] ceiling (the per-allocation and cumulative-byte budgets)
/// against the declared header size. The built-in `100 MiB` header cap still
/// applies; `limits` can only tighten it. [`parse_safetensors_header`] is the
/// `ParseLimits::default()` (unbounded) special case.
///
/// # Errors
///
/// Returns [`AnamnesisError::LimitExceeded`] if the declared header size exceeds
/// the cap or `limits`.
/// Returns [`AnamnesisError::Parse`] if the buffer is too small for the 8-byte
/// length prefix or the safetensors header is malformed.
///
/// Returns [`AnamnesisError::Unsupported`] if a tensor uses an unrecognized dtype.
///
/// # Memory
///
/// Allocates a `Vec<TensorEntry>` proportional to the number of tensors in the
/// header (typically hundreds). No tensor data is copied or read.
pub fn parse_safetensors_header_with_limits(
    buffer: &[u8],
    limits: &ParseLimits,
) -> crate::Result<SafetensorsHeader> {
    // Bound the declared header length *before* `read_metadata` parses (and
    // allocates) the metadata. The 8-byte little-endian prefix is the same one
    // `read_metadata` consumes, so checking it here rejects an over-budget
    // header pre-allocation — matching the reader path.
    let prefix: [u8; 8] = buffer
        .get(..8)
        .and_then(|s| s.try_into().ok())
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "safetensors buffer too small for 8-byte header length prefix".into(),
        })?;
    let mut budget = Budget::new(limits);
    enforce_safetensors_header_cap(u64::from_le_bytes(prefix), &mut budget)?;

    let (header_size, metadata) =
        safetensors::SafeTensors::read_metadata(buffer).map_err(AnamnesisError::from)?;
    build_header_from_metadata(header_size, &metadata, limits)
}

/// Whether `entry` is a `ModelOpt` `NVFP4` weight: a `U8` `X.weight` with an
/// `F8_E4M3` `X.weight_scale` and an `X.weight_scale_2` beside it.
fn is_modelopt_nvfp4_weight(entry: &TensorEntry, index: &NameIndex<'_>) -> bool {
    entry.dtype == Dtype::U8
        && entry.name.ends_with(".weight")
        && index
            .named(&format!("{}_scale", entry.name))
            .is_some_and(|s| s.dtype == Dtype::F8E4M3)
        && index.named(&format!("{}_scale_2", entry.name)).is_some()
}

/// Gives a `ModelOpt` `NVFP4` layer its roles, whatever the enabled features.
///
/// Name-by-name classification cannot see this layout: a `U8` `.weight` is a
/// `BnB` 4-bit weight only under the `bnb` feature and a passthrough without
/// it, and `weight_scale_2` matches no suffix. Without the `bnb` feature, a
/// `ModelOpt` checkpoint therefore read as *unquantised*, and `remember` would
/// have copied the packed `U8` bytes into its output as if they were weights.
/// This pass marks each layer's `.weight` as quantised and its `weight_scale`,
/// `weight_scale_2` and `input_scale` as companions, so the scheme is detected
/// as [`QuantScheme::Nvfp4`] and refused, never passed through.
fn mark_modelopt_nvfp4(entries: &mut [TensorEntry]) {
    // Names to positions once, so a header of thousands of layers stays linear
    // (the quadratic companion lookup Phase 7.9 removed, audit finding M-7).
    let (weights, companions): (Vec<usize>, Vec<usize>) = {
        let index = NameIndex::new(entries);
        let position: std::collections::HashMap<&str, usize> = entries
            .iter()
            .enumerate()
            .map(|(i, e)| (e.name.as_str(), i))
            .collect();
        let mut weights = Vec::new();
        let mut companions = Vec::new();
        for (i, entry) in entries.iter().enumerate() {
            if !is_modelopt_nvfp4_weight(entry, &index) {
                continue;
            }
            weights.push(i);
            let stem = entry.name.strip_suffix(".weight").unwrap_or(&entry.name);
            for name in [
                format!("{}_scale", entry.name),
                format!("{}_scale_2", entry.name),
                format!("{stem}.input_scale"),
            ] {
                companions.extend(position.get(name.as_str()).copied());
            }
        }
        (weights, companions)
    };
    for (positions, role) in [
        (weights, TensorRole::Quantized),
        (companions, TensorRole::Scale),
    ] {
        for i in positions {
            if let Some(entry) = entries.get_mut(i) {
                entry.role = role;
            }
        }
    }
}

/// Turns every [`TensorRole::Scale`] tensor that no quantised tensor in the
/// file can own back into [`TensorRole::Passthrough`].
///
/// `classify_tensor` decides a scale by its **name** alone (`*_scale`,
/// `*_scale_inv`, and under their features `.scales`, `.weight.absmax`,
/// `.SCB`), because it sees one tensor at a time. A companion is consumed by
/// dequantisation and never written, so a misclassified tensor was silently
/// dropped: until Phase 7.10 a plain `BF16` model with CLIP's `logit_scale`, a
/// `layer_scale`, or a leftover `input_scale` lost them on `remember` and
/// `convert`, and `inspect` under-reported its size to match.
///
/// A scale is kept as a companion only when a quantised tensor exists for it,
/// under one of the names the schemes actually use:
///
/// - its name without `_scale_inv` / `_scale` (the `FP8` weight scale
///   `X.weight_scale`, exactly the lookup `scale_for` performs);
/// - its module (`BnB` 4-bit: `X.weight.absmax` belongs to `X.weight`);
/// - its module's `.weight` (`BnB` `INT8` `X.SCB`, and the `FP8` activation
///   scale `X.input_scale` of a quantised `X.weight`);
/// - its module's `.qweight` (`GPTQ` / `AWQ` `X.scales`).
///
/// Everything else passes through, so a file with no quantised tensor at all
/// keeps every tensor it has.
fn demote_orphan_scales(entries: &mut [TensorEntry]) {
    let quantized: std::collections::HashSet<&str> = entries
        .iter()
        .filter(|e| e.role == TensorRole::Quantized)
        .map(|e| e.name.as_str())
        .collect();
    let owned = |name: &str| -> bool {
        let fp8_weight = name
            .strip_suffix("_scale_inv")
            .or_else(|| name.strip_suffix("_scale"));
        if fp8_weight.is_some_and(|w| quantized.contains(w)) {
            return true;
        }
        name.rsplit_once('.').is_some_and(|(module, _)| {
            quantized.contains(module)
                || quantized.contains(format!("{module}.weight").as_str())
                || quantized.contains(format!("{module}.qweight").as_str())
        })
    };
    let orphans: Vec<bool> = entries
        .iter()
        .map(|e| e.role == TensorRole::Scale && !owned(&e.name))
        .collect();
    for (entry, orphan) in entries.iter_mut().zip(orphans) {
        if orphan {
            entry.role = TensorRole::Passthrough;
        }
    }
}

/// Builds a [`SafetensorsHeader`] from a pre-parsed
/// `safetensors::tensor::Metadata`.
///
/// Shared core of the slice-based [`parse_safetensors_header`] and the
/// reader-based [`parse_safetensors_header_from_reader`]; the two entry
/// points differ only in how they obtain the parsed `Metadata` (via
/// `SafeTensors::read_metadata`, which requires the full buffer including
/// the data section, vs. via `serde_json::from_slice` on just the `JSON`
/// header bytes).
fn build_header_from_metadata(
    header_size: usize,
    metadata: &safetensors::tensor::Metadata,
    limits: &ParseLimits,
) -> crate::Result<SafetensorsHeader> {
    let st_tensors = metadata.tensors();
    // `max_item_count` applies to safetensors too (Phase 7.9, audit finding
    // M-7): until v0.7.9 it was documented for `GGUF` and `ZIP` only, so a
    // header of 40 000 tensors passed a limit of 1000.
    let count = u64::try_from(st_tensors.len()).map_err(|_| AnamnesisError::Parse {
        reason: "safetensors tensor count overflows u64".into(),
    })?;
    limits.check_item_count(count, "safetensors tensor count")?;
    let mut entries = Vec::with_capacity(st_tensors.len());

    for (name, info) in &st_tensors {
        let dtype = Dtype::try_from(info.dtype)?;
        let role = classify_tensor(name, dtype);
        entries.push(TensorEntry {
            name: (*name).clone(),
            dtype,
            shape: info.shape.clone(),
            data_offsets: info.data_offsets,
            role,
        });
    }

    mark_modelopt_nvfp4(&mut entries);
    demote_orphan_scales(&mut entries);

    // Sort by name for deterministic ordering (HashMap iteration is arbitrary).
    entries.sort_by(|a, b| a.name.cmp(&b.name));

    let index = NameIndex::new(&entries);
    let scheme = detect_scheme(&entries, &index);
    let file_metadata = metadata.metadata().clone();

    let gptq_config = if scheme == QuantScheme::Gptq {
        infer_gptq_config(&entries, &index, file_metadata.as_ref())
    } else {
        None
    };

    let awq_config = if scheme == QuantScheme::Awq {
        infer_awq_config(&entries, &index)
    } else {
        None
    };

    let bnb_config = if scheme == QuantScheme::Bnb4 {
        infer_bnb_config(&entries, &index)
    } else {
        None
    };

    Ok(SafetensorsHeader {
        tensors: entries,
        scheme,
        metadata: file_metadata,
        header_size,
        gptq_config,
        awq_config,
        bnb_config,
    })
}

/// Parses a `.safetensors` header from any `Read` source.
///
/// This is the reader-generic core of the safetensors parsing API: callers
/// supply any `Read` substrate (in-memory `Cursor`, an `HTTP`-range-backed
/// adapter, a custom transport, …) and receive the same
/// [`SafetensorsHeader`] as the slice-based [`parse_safetensors_header`].
/// The slice-based variant remains available for callers that already have
/// the prefix + `JSON` bytes materialised.
///
/// # Range-read access pattern
///
/// The safetensors header lives at file offsets `[0, 8 + length)` — exactly
/// the bytes a sequential `Read` produces in order. Two contiguous logical
/// fetches are sufficient:
///
/// 1. **8 bytes at offset 0** — the little-endian `u64` length prefix.
/// 2. **`length` bytes starting at offset 8** — the `JSON` header itself
///    (typically a few hundred KiB on a multi-GB shard).
///
/// No `Seek` is required: the header is purely prefix-then-`JSON`, so the
/// simplest possible `HTTP`-range adapter (one connection, two contiguous
/// range fetches, never seek-back) satisfies this function. This is the
/// reason the trait bound is `R: Read` rather than `R: Read + Seek` — no
/// parser code seeks backwards.
///
/// Anamnesis itself does not ship an `HTTP` transport; the network layer
/// belongs in downstream crates (e.g., `hf-fm`'s safetensors range-reader).
/// This function defines the I/O contract such an adapter must satisfy.
///
/// # Errors
///
/// Returns [`AnamnesisError::Io`] if the reader fails to produce the
/// requested bytes (8-byte length prefix or `length` bytes of `JSON`).
///
/// Returns [`AnamnesisError::Parse`] if the header bytes are malformed.
///
/// Returns [`AnamnesisError::LimitExceeded`] if the declared length exceeds the
/// permanent 100 MiB cap (`MAX_SAFETENSORS_HEADER_BYTES`, always-on).
///
/// Returns [`AnamnesisError::Unsupported`] if a tensor uses an unrecognised
/// dtype.
///
/// # Source context
///
/// Errors describe the **format-level problem**, not the source identity.
/// The function is reader-agnostic — the source could be a file, an
/// in-memory `Cursor`, or an `HTTP`-range adapter. Callers that have a
/// source name (filename, URL, etc.) should wrap the returned error with
/// that context. This matches anamnesis's existing convention
/// ([`parse_safetensors_header`] and `inspect_npz_from_reader` already
/// return source-agnostic errors).
///
/// # Memory
///
/// Allocates `8 + header_size` bytes for the prefix + `JSON` buffer plus a
/// `Vec<TensorEntry>` proportional to the number of tensors in the header
/// (typically hundreds). No tensor data is read. The declared header
/// length is capped at 100 MiB to bound the worst-case allocation an
/// adversarial source can trigger.
pub fn parse_safetensors_header_from_reader<R: Read>(
    reader: R,
) -> crate::Result<SafetensorsHeader> {
    parse_safetensors_header_from_reader_with_limits(reader, &ParseLimits::default())
}

/// Parses a safetensors header from a reader under a caller-supplied
/// [`ParseLimits`] budget.
///
/// Identical to [`parse_safetensors_header_from_reader`] but enforces every
/// applicable [`ParseLimits`] ceiling (the per-allocation and cumulative-byte
/// budgets) against the declared header length **before** the `JSON` buffer is
/// allocated. The built-in `100 MiB` header cap still applies; `limits` can
/// only tighten it. [`parse_safetensors_header_from_reader`] is the
/// `ParseLimits::default()` (unbounded) special case.
///
/// # Errors
///
/// Returns [`AnamnesisError::Io`] if the reader fails to produce the
/// requested bytes (8-byte length prefix or `length` bytes of `JSON`).
///
/// Returns [`AnamnesisError::LimitExceeded`] if the declared length exceeds the
/// 100 MiB sanity cap or `limits`.
/// Returns [`AnamnesisError::Parse`] if the header bytes are malformed.
///
/// Returns [`AnamnesisError::Unsupported`] if a tensor uses an unrecognised
/// dtype.
///
/// # Memory
///
/// Allocates `8 + header_size` bytes for the prefix + `JSON` buffer plus a
/// `Vec<TensorEntry>` proportional to the number of tensors. No tensor data
/// is read. The declared header length is bounded by both the 100 MiB cap and
/// the caller's `limits` before allocation.
pub fn parse_safetensors_header_from_reader_with_limits<R: Read>(
    mut reader: R,
    limits: &ParseLimits,
) -> crate::Result<SafetensorsHeader> {
    // Read the 8-byte little-endian length prefix.
    let mut prefix = [0u8; 8];
    reader.read_exact(&mut prefix)?;
    let header_len_u64 = u64::from_le_bytes(prefix);

    let mut budget = Budget::new(limits);
    enforce_safetensors_header_cap(header_len_u64, &mut budget)?;
    let header_len = usize::try_from(header_len_u64).map_err(|_| AnamnesisError::Parse {
        reason: format!("safetensors header length {header_len_u64} does not fit in usize"),
    })?;

    // Read just the `JSON` header bytes — no data section.
    let mut json_bytes = vec![0u8; header_len];
    reader.read_exact(&mut json_bytes)?;

    // Bypass `safetensors::SafeTensors::read_metadata` (which would
    // require the buffer to contain the data section as well) and
    // deserialise the `JSON` header directly. The upstream `Metadata`
    // type implements `serde::Deserialize`, so this is the same parsing
    // step `read_metadata` performs internally — without the
    // post-parsing buffer-length check that fails on a header-only
    // buffer.
    let metadata: safetensors::tensor::Metadata =
        serde_json::from_slice(&json_bytes).map_err(|e| AnamnesisError::Parse {
            reason: format!("failed to parse safetensors header: {e}"),
        })?;

    build_header_from_metadata(header_len, &metadata, limits)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::panic, clippy::indexing_slicing)]
mod tests {
    use super::*;

    // -- Dtype ---------------------------------------------------------------

    #[test]
    fn dtype_byte_sizes() {
        assert_eq!(Dtype::F8E4M3.byte_size(), 1);
        assert_eq!(Dtype::F8E5M2.byte_size(), 1);
        assert_eq!(Dtype::U8.byte_size(), 1);
        assert_eq!(Dtype::I8.byte_size(), 1);
        assert_eq!(Dtype::Bool.byte_size(), 1);
        assert_eq!(Dtype::BF16.byte_size(), 2);
        assert_eq!(Dtype::F16.byte_size(), 2);
        assert_eq!(Dtype::U16.byte_size(), 2);
        assert_eq!(Dtype::I16.byte_size(), 2);
        assert_eq!(Dtype::F32.byte_size(), 4);
        assert_eq!(Dtype::U32.byte_size(), 4);
        assert_eq!(Dtype::I32.byte_size(), 4);
        assert_eq!(Dtype::F64.byte_size(), 8);
        assert_eq!(Dtype::U64.byte_size(), 8);
        assert_eq!(Dtype::I64.byte_size(), 8);
    }

    #[test]
    fn dtype_is_quantized() {
        assert!(Dtype::F8E4M3.is_quantized());
        assert!(Dtype::F8E5M2.is_quantized());
        assert!(!Dtype::BF16.is_quantized());
        assert!(!Dtype::F32.is_quantized());
        assert!(!Dtype::U8.is_quantized());
    }

    #[test]
    fn dtype_is_floating_point() {
        assert!(Dtype::F8E4M3.is_floating_point());
        assert!(Dtype::BF16.is_floating_point());
        assert!(Dtype::F32.is_floating_point());
        assert!(Dtype::F64.is_floating_point());
        assert!(!Dtype::U8.is_floating_point());
        assert!(!Dtype::I32.is_floating_point());
        assert!(!Dtype::Bool.is_floating_point());
    }

    #[test]
    fn dtype_display() {
        assert_eq!(Dtype::F8E4M3.to_string(), "F8_E4M3");
        assert_eq!(Dtype::BF16.to_string(), "BF16");
        assert_eq!(Dtype::F32.to_string(), "F32");
    }

    #[test]
    fn dtype_try_from_safetensors() {
        assert_eq!(
            Dtype::try_from(safetensors::Dtype::F8_E4M3).ok(),
            Some(Dtype::F8E4M3)
        );
        assert_eq!(
            Dtype::try_from(safetensors::Dtype::BF16).ok(),
            Some(Dtype::BF16)
        );
        assert_eq!(
            Dtype::try_from(safetensors::Dtype::F32).ok(),
            Some(Dtype::F32)
        );
    }

    // -- Classification ------------------------------------------------------

    #[test]
    fn classify_quantized_weight() {
        let role = classify_tensor("model.layers.0.self_attn.q_proj.weight", Dtype::F8E4M3);
        assert_eq!(role, TensorRole::Quantized);
    }

    #[test]
    fn classify_scale_inv() {
        let role = classify_tensor(
            "model.layers.0.self_attn.q_proj.weight_scale_inv",
            Dtype::F32,
        );
        assert_eq!(role, TensorRole::Scale);
    }

    #[test]
    fn classify_scale() {
        let role = classify_tensor("model.layers.0.self_attn.q_proj.weight_scale", Dtype::F32);
        assert_eq!(role, TensorRole::Scale);
    }

    #[test]
    fn classify_passthrough_norm() {
        let role = classify_tensor("model.norm.weight", Dtype::BF16);
        assert_eq!(role, TensorRole::Passthrough);
    }

    #[test]
    fn classify_passthrough_embedding() {
        let role = classify_tensor("model.embed_tokens.weight", Dtype::BF16);
        assert_eq!(role, TensorRole::Passthrough);
    }

    // -- Scheme detection ----------------------------------------------------

    /// The index must answer exactly as the linear scan the public `find_*`
    /// methods use, including on a list a caller has reordered, and for names
    /// that repeat (Phase 7.9, audit finding M-7).
    #[test]
    fn name_index_agrees_with_the_linear_scan() {
        let entries: Vec<TensorEntry> = ["b", "a.x", "c", "a", "b", "a.q_2", "a.q_1"]
            .iter()
            .map(|name| make_entry(name, Dtype::F32, TensorRole::Passthrough))
            .collect();
        let index = NameIndex::new(&entries);
        let linear = entries.as_slice();
        let position = |found: Option<&TensorEntry>| {
            found.and_then(|e| entries.iter().position(|x| std::ptr::eq(x, e)))
        };
        for name in ["a", "b", "c", "a.x", "missing", ""] {
            assert_eq!(
                position(index.named(name)),
                position(linear.named(name)),
                "named({name:?})"
            );
        }
        // The duplicate `b` resolves to its first occurrence, position 0.
        assert_eq!(position(index.named("b")), Some(0));
        for prefix in ["a.q_", "a.", "a", "z", ""] {
            assert_eq!(
                position(index.first_with_prefix(prefix)),
                position(linear.first_with_prefix(prefix)),
                "first_with_prefix({prefix:?})"
            );
        }
        // Of the two `a.q_*` names, the first in list order (position 5), not
        // the smallest by name (`a.q_1`, position 6).
        assert_eq!(position(index.first_with_prefix("a.q_")), Some(5));
    }

    fn make_entry(name: &str, dtype: Dtype, role: TensorRole) -> TensorEntry {
        make_entry_with_shape(name, dtype, role, vec![128, 128])
    }

    fn make_entry_with_shape(
        name: &str,
        dtype: Dtype,
        role: TensorRole,
        shape: Vec<usize>,
    ) -> TensorEntry {
        let num_elements: usize = shape.iter().product();
        let byte_len = num_elements * dtype.byte_size();
        TensorEntry {
            name: name.to_owned(),
            dtype,
            shape,
            data_offsets: (0, byte_len),
            role,
        }
    }

    /// `TensorEntry::num_elements` saturates to `usize::MAX` rather than
    /// silently wrapping when a malformed or adversarial header declares
    /// a shape whose element count overflows `usize`. Mirrors the
    /// saturating contract `inspect_npz` already provides.
    #[test]
    fn num_elements_saturates_on_overflow() {
        // Shape that overflows on every supported target:
        //   on 64-bit usize, [usize::MAX, 2] overflows on the first multiply
        //   on 32-bit usize, the same shape overflows even faster
        let entry = TensorEntry {
            name: "huge".to_owned(),
            dtype: Dtype::F32,
            shape: vec![usize::MAX, 2],
            data_offsets: (0, 0),
            role: TensorRole::Passthrough,
        };
        assert_eq!(entry.num_elements(), usize::MAX);
    }

    /// `num_elements` on a normal shape returns the exact product, not
    /// the saturated value. Guards against an over-eager fix that would
    /// have saturated even on legitimate shapes.
    #[test]
    fn num_elements_exact_on_normal_shape() {
        let entry = TensorEntry {
            name: "normal".to_owned(),
            dtype: Dtype::F32,
            shape: vec![16, 4096, 2048],
            data_offsets: (0, 0),
            role: TensorRole::Passthrough,
        };
        assert_eq!(entry.num_elements(), 16 * 4096 * 2048);
    }

    /// Empty shape → the empty product, which is `1` (single scalar).
    /// This matches the `shape.iter().product()` contract on the empty
    /// iterator and prevents the saturating fix from inadvertently
    /// returning `0` for scalars.
    #[test]
    fn num_elements_empty_shape_is_one() {
        let entry = TensorEntry {
            name: "scalar".to_owned(),
            dtype: Dtype::F32,
            shape: vec![],
            data_offsets: (0, 0),
            role: TensorRole::Passthrough,
        };
        assert_eq!(entry.num_elements(), 1);
    }

    #[test]
    fn detect_unquantized() {
        let entries = vec![
            make_entry("model.norm.weight", Dtype::BF16, TensorRole::Passthrough),
            make_entry("lm_head.weight", Dtype::BF16, TensorRole::Passthrough),
        ];
        assert_eq!(
            detect_scheme(&entries, &NameIndex::new(&entries)),
            QuantScheme::Unquantized
        );
    }

    #[test]
    fn detect_fine_grained_fp8() {
        let entries = vec![
            make_entry("layer.0.weight", Dtype::F8E4M3, TensorRole::Quantized),
            make_entry("layer.0.weight_scale_inv", Dtype::F32, TensorRole::Scale),
            make_entry("model.norm.weight", Dtype::BF16, TensorRole::Passthrough),
        ];
        assert_eq!(
            detect_scheme(&entries, &NameIndex::new(&entries)),
            QuantScheme::FineGrainedFp8
        );
    }

    #[test]
    fn detect_per_tensor_fp8() {
        let entries = vec![
            make_entry("layer.0.weight", Dtype::F8E4M3, TensorRole::Quantized),
            make_entry("model.norm.weight", Dtype::BF16, TensorRole::Passthrough),
        ];
        assert_eq!(
            detect_scheme(&entries, &NameIndex::new(&entries)),
            QuantScheme::PerTensorFp8
        );
    }

    #[test]
    fn detect_per_tensor_fp8_with_scalar_scale_inv() {
        // Ministral pattern: _scale_inv exists but is scalar (shape [])
        let entries = vec![
            make_entry("layer.0.weight", Dtype::F8E4M3, TensorRole::Quantized),
            make_entry_with_shape(
                "layer.0.weight_scale_inv",
                Dtype::BF16,
                TensorRole::Scale,
                vec![],
            ),
            make_entry_with_shape(
                "layer.0.activation_scale",
                Dtype::BF16,
                TensorRole::Scale,
                vec![],
            ),
            make_entry("model.norm.weight", Dtype::BF16, TensorRole::Passthrough),
        ];
        // Scalar scale_inv → per-tensor, NOT fine-grained
        assert_eq!(
            detect_scheme(&entries, &NameIndex::new(&entries)),
            QuantScheme::PerTensorFp8
        );
    }

    #[test]
    fn detect_per_tensor_fp8_with_1d_scale_inv() {
        // Compressed-tensors pattern: _scale_inv with shape [1]
        let entries = vec![
            make_entry("layer.0.weight", Dtype::F8E4M3, TensorRole::Quantized),
            make_entry_with_shape(
                "layer.0.weight_scale_inv",
                Dtype::BF16,
                TensorRole::Scale,
                vec![1],
            ),
        ];
        // 1D scale_inv → per-tensor, NOT fine-grained
        assert_eq!(
            detect_scheme(&entries, &NameIndex::new(&entries)),
            QuantScheme::PerTensorFp8
        );
    }

    #[test]
    fn detect_fine_grained_fp8_with_2d_scale_inv() {
        // EXAONE/DeepSeek pattern: _scale_inv with shape [16, 32]
        let entries = vec![
            make_entry_with_shape(
                "layer.0.weight",
                Dtype::F8E4M3,
                TensorRole::Quantized,
                vec![2048, 4096],
            ),
            make_entry_with_shape(
                "layer.0.weight_scale_inv",
                Dtype::BF16,
                TensorRole::Scale,
                vec![16, 32],
            ),
            make_entry("model.norm.weight", Dtype::BF16, TensorRole::Passthrough),
        ];
        // 2D scale_inv → fine-grained
        assert_eq!(
            detect_scheme(&entries, &NameIndex::new(&entries)),
            QuantScheme::FineGrainedFp8
        );
    }

    // -- find_scale_for ------------------------------------------------------

    #[test]
    fn find_scale_for_prefers_scale_inv() {
        let header = SafetensorsHeader {
            tensors: vec![
                make_entry("w", Dtype::F8E4M3, TensorRole::Quantized),
                make_entry("w_scale", Dtype::F32, TensorRole::Scale),
                make_entry("w_scale_inv", Dtype::F32, TensorRole::Scale),
            ],
            scheme: QuantScheme::FineGrainedFp8,
            metadata: None,
            header_size: 0,
            gptq_config: None,
            awq_config: None,
            bnb_config: None,
        };
        let found = header.find_scale_for("w");
        assert_eq!(found.map(|e| e.name.as_str()), Some("w_scale_inv"));
    }

    #[test]
    fn find_scale_for_falls_back_to_scale() {
        let header = SafetensorsHeader {
            tensors: vec![
                make_entry("w", Dtype::F8E4M3, TensorRole::Quantized),
                make_entry("w_scale", Dtype::F32, TensorRole::Scale),
            ],
            scheme: QuantScheme::PerTensorFp8,
            metadata: None,
            header_size: 0,
            gptq_config: None,
            awq_config: None,
            bnb_config: None,
        };
        let found = header.find_scale_for("w");
        assert_eq!(found.map(|e| e.name.as_str()), Some("w_scale"));
    }

    #[test]
    fn find_scale_for_returns_none_when_missing() {
        let header = SafetensorsHeader {
            tensors: vec![make_entry("w", Dtype::F8E4M3, TensorRole::Quantized)],
            scheme: QuantScheme::PerTensorFp8,
            metadata: None,
            header_size: 0,
            gptq_config: None,
            awq_config: None,
            bnb_config: None,
        };
        assert!(header.find_scale_for("w").is_none());
    }

    // -- Full parse round-trip -----------------------------------------------

    #[test]
    fn parse_minimal_safetensors() {
        use safetensors::tensor::serialize;

        // Build a minimal safetensors buffer with one BF16 tensor.
        let data: Vec<u8> = vec![0; 4]; // 2 elements × 2 bytes
        let tensors = vec![(
            "test_tensor",
            safetensors::tensor::TensorView::new(safetensors::Dtype::BF16, vec![2], &data)
                .unwrap_or_else(|e| panic!("failed to create TensorView: {e}")),
        )];
        let buffer = serialize(tensors, None).unwrap_or_else(|e| panic!("serialize: {e}"));

        let header = parse_safetensors_header(&buffer).unwrap_or_else(|e| panic!("parse: {e}"));

        assert_eq!(header.tensors.len(), 1);
        assert_eq!(header.tensors[0].name, "test_tensor"); // INDEX: single element, bounds checked by len() assert above
        assert_eq!(header.tensors[0].dtype, Dtype::BF16);
        assert_eq!(header.tensors[0].shape, vec![2]);
        assert_eq!(header.tensors[0].role, TensorRole::Passthrough);
        assert_eq!(header.scheme, QuantScheme::Unquantized);
    }

    /// Phase 6.8 Step 1: a tightened `ParseLimits` rejects a header that the
    /// unbounded default accepts, on both the slice and reader entry points.
    #[test]
    fn safetensors_header_respects_parse_limits() {
        use safetensors::tensor::{TensorView, serialize};

        let data: Vec<u8> = vec![0; 4];
        let tensors = vec![(
            "t",
            TensorView::new(safetensors::Dtype::BF16, vec![2], &data)
                .unwrap_or_else(|e| panic!("TensorView: {e}")),
        )];
        let buffer = serialize(tensors, None).unwrap_or_else(|e| panic!("serialize: {e}"));

        // Default (unbounded) parses on both the slice and reader paths.
        assert!(parse_safetensors_header_with_limits(&buffer, &ParseLimits::default()).is_ok());
        assert!(
            parse_safetensors_header_from_reader_with_limits(
                std::io::Cursor::new(&buffer),
                &ParseLimits::default()
            )
            .is_ok()
        );

        // A 1-byte single-allocation ceiling rejects the dozens-of-bytes header.
        let tight = ParseLimits::default().with_max_single_alloc(1);
        let Err(err) = parse_safetensors_header_with_limits(&buffer, &tight) else {
            panic!("expected slice limit rejection");
        };
        assert!(
            matches!(err, AnamnesisError::LimitExceeded { limit, .. } if limit == "max_single_alloc_bytes"),
            "expected slice limit error, got: {err}"
        );
        let Err(err) =
            parse_safetensors_header_from_reader_with_limits(std::io::Cursor::new(&buffer), &tight)
        else {
            panic!("expected reader limit rejection");
        };
        assert!(
            matches!(err, AnamnesisError::LimitExceeded { limit, .. } if limit == "max_single_alloc_bytes"),
            "expected reader limit error, got: {err}"
        );

        // The slice path rejects pre-allocation: a buffer too small for even
        // the 8-byte length prefix is a clean `Parse` error, not a panic.
        let Err(err) = parse_safetensors_header_with_limits(&[0u8; 4], &ParseLimits::default())
        else {
            panic!("expected too-small-buffer rejection");
        };
        assert!(
            matches!(err, AnamnesisError::Parse { ref reason } if reason.contains("8-byte")),
            "expected too-small-prefix error, got: {err}"
        );

        // Aggregate axis (Step 2): the header alone exceeds a 1-byte
        // max_total_bytes even though the per-item single-alloc cap is
        // unbounded — rejected on the aggregate, not the single-alloc, axis.
        let total = ParseLimits::default().with_max_total_bytes(1);
        let Err(err) = parse_safetensors_header_with_limits(&buffer, &total) else {
            panic!("expected aggregate rejection");
        };
        assert!(
            matches!(err, AnamnesisError::LimitExceeded { limit, .. } if limit == "max_total_bytes"),
            "expected aggregate error, got: {err}"
        );
    }

    #[test]
    fn parse_fp8_with_scale() {
        use safetensors::tensor::serialize;

        let weight_data: Vec<u8> = vec![0; 4]; // 4 FP8 elements
        let scale_data: Vec<u8> = vec![0; 8]; // 2 F32 scale values (shape [1, 2])

        let tensors = vec![
            (
                "layer.weight",
                safetensors::tensor::TensorView::new(
                    safetensors::Dtype::F8_E4M3,
                    vec![2, 2],
                    &weight_data,
                )
                .unwrap_or_else(|e| panic!("weight TensorView: {e}")),
            ),
            (
                "layer.weight_scale_inv",
                safetensors::tensor::TensorView::new(
                    safetensors::Dtype::F32,
                    vec![1, 2],
                    &scale_data,
                )
                .unwrap_or_else(|e| panic!("scale TensorView: {e}")),
            ),
        ];
        let buffer = serialize(tensors, None).unwrap_or_else(|e| panic!("serialize: {e}"));

        let header = parse_safetensors_header(&buffer).unwrap_or_else(|e| panic!("parse: {e}"));

        assert_eq!(header.tensors.len(), 2);
        assert_eq!(header.quantized_count(), 1);
        assert_eq!(header.scale_count(), 1);
        assert_eq!(header.passthrough_count(), 0);
        assert_eq!(header.scheme, QuantScheme::FineGrainedFp8);

        let scale = header.find_scale_for("layer.weight");
        assert_eq!(
            scale.map(|e| e.name.as_str()),
            Some("layer.weight_scale_inv")
        );
    }

    // -- Reader-generic header parsing (Phase 4.8) ---------------------------

    /// `parse_safetensors_header_from_reader` over an in-memory `Cursor`
    /// returns a `SafetensorsHeader` whose every field matches what the
    /// slice-based `parse_safetensors_header` returns on the same bytes.
    /// Locks the contract that the reader-generic and slice-based APIs are
    /// substrate-equivalent — the substrate (slice vs. cursor vs. HTTP-range
    /// adapter) cannot change the metadata.
    #[test]
    fn parse_from_reader_matches_slice_minimal() {
        use safetensors::tensor::{TensorView, serialize};

        // One BF16 tensor: 2 elements × 2 bytes = 4 bytes of data.
        let data: Vec<u8> = vec![0; 4];
        let tensors = vec![(
            "test_tensor",
            TensorView::new(safetensors::Dtype::BF16, vec![2], &data)
                .unwrap_or_else(|e| panic!("TensorView: {e}")),
        )];
        let buffer = serialize(tensors, None).unwrap_or_else(|e| panic!("serialize: {e}"));

        let slice_header =
            parse_safetensors_header(&buffer).unwrap_or_else(|e| panic!("slice parse: {e}"));
        let reader_header = parse_safetensors_header_from_reader(std::io::Cursor::new(&buffer))
            .unwrap_or_else(|e| panic!("reader parse: {e}"));

        assert_eq!(slice_header.tensors.len(), reader_header.tensors.len());
        assert_eq!(slice_header.scheme, reader_header.scheme);
        assert_eq!(slice_header.metadata, reader_header.metadata);
        assert_eq!(slice_header.header_size, reader_header.header_size);
        for (a, b) in slice_header
            .tensors
            .iter()
            .zip(reader_header.tensors.iter())
        {
            assert_eq!(a.name, b.name);
            assert_eq!(a.dtype, b.dtype);
            assert_eq!(a.shape, b.shape);
            assert_eq!(a.data_offsets, b.data_offsets);
            assert_eq!(a.role, b.role);
        }
    }

    /// Substrate-equivalence on a buffer that exercises the FP8 quantized
    /// scheme detection path: weight + companion `_scale_inv` + passthrough
    /// norm. Asserts both APIs detect `FineGrainedFp8` and report the same
    /// per-tensor metadata.
    #[test]
    fn parse_from_reader_matches_slice_fp8_with_scale() {
        use safetensors::tensor::{TensorView, serialize};

        let weight_data: Vec<u8> = vec![0; 4];
        let scale_data: Vec<u8> = vec![0; 8];

        let tensors = vec![
            (
                "layer.weight",
                TensorView::new(safetensors::Dtype::F8_E4M3, vec![2, 2], &weight_data)
                    .unwrap_or_else(|e| panic!("weight TensorView: {e}")),
            ),
            (
                "layer.weight_scale_inv",
                TensorView::new(safetensors::Dtype::F32, vec![1, 2], &scale_data)
                    .unwrap_or_else(|e| panic!("scale TensorView: {e}")),
            ),
        ];
        let buffer = serialize(tensors, None).unwrap_or_else(|e| panic!("serialize: {e}"));

        let slice_header =
            parse_safetensors_header(&buffer).unwrap_or_else(|e| panic!("slice parse: {e}"));
        let reader_header = parse_safetensors_header_from_reader(std::io::Cursor::new(&buffer))
            .unwrap_or_else(|e| panic!("reader parse: {e}"));

        assert_eq!(slice_header.scheme, QuantScheme::FineGrainedFp8);
        assert_eq!(reader_header.scheme, QuantScheme::FineGrainedFp8);
        assert_eq!(slice_header.tensors.len(), reader_header.tensors.len());
        assert_eq!(
            slice_header.quantized_count(),
            reader_header.quantized_count()
        );
        assert_eq!(slice_header.scale_count(), reader_header.scale_count());
        assert_eq!(slice_header.header_size, reader_header.header_size);
    }

    /// A reader that yields fewer than 8 bytes for the length prefix surfaces
    /// as [`AnamnesisError::Io`] rather than [`AnamnesisError::Parse`]. This
    /// preserves the documented contract — read failures and format failures
    /// are distinct error variants — and matches the `# Source context`
    /// rustdoc convention.
    #[test]
    fn parse_from_reader_rejects_truncated_prefix() {
        let truncated: Vec<u8> = vec![0, 0, 0]; // < 8 bytes
        match parse_safetensors_header_from_reader(std::io::Cursor::new(&truncated)) {
            Err(AnamnesisError::Io(_)) => {}
            Ok(_) => panic!("expected Err for truncated prefix, got Ok"),
            Err(other) => panic!("expected AnamnesisError::Io, got {other:?}"),
        }
    }

    /// A reader that declares a length larger than the 100 MiB sanity cap
    /// surfaces as [`AnamnesisError::Parse`] without attempting the
    /// allocation. Guards against an adversarial source that lies about the
    /// header size to trigger an arbitrary memory request.
    #[test]
    fn parse_from_reader_rejects_oversized_header_length() {
        // Declared length: cap + 1.
        let bogus_len = MAX_SAFETENSORS_HEADER_BYTES + 1;
        let prefix = bogus_len.to_le_bytes();
        match parse_safetensors_header_from_reader(std::io::Cursor::new(&prefix[..])) {
            Err(AnamnesisError::LimitExceeded { limit, message }) => {
                assert_eq!(limit, "MAX_SAFETENSORS_HEADER_BYTES");
                assert!(
                    message.contains("exceeds") && message.contains("cap"),
                    "expected oversized-length error, got: {message}"
                );
            }
            Ok(_) => panic!("expected Err for oversized header, got Ok"),
            Err(other) => panic!("expected AnamnesisError::LimitExceeded, got {other:?}"),
        }
    }

    /// A reader whose declared length is honest but whose JSON tail is
    /// truncated surfaces as [`AnamnesisError::Io`] (the read of the JSON
    /// bytes fails part-way through). This is the partial-fetch failure mode
    /// an HTTP-range adapter must distinguish from a malformed header.
    #[test]
    fn parse_from_reader_rejects_truncated_json_tail() {
        // Honest 64-byte declared header, but only 8-byte prefix + 4 bytes
        // of JSON delivered before the reader runs out.
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&64u64.to_le_bytes());
        buf.extend_from_slice(b"{\"a\"");
        match parse_safetensors_header_from_reader(std::io::Cursor::new(&buf)) {
            Err(AnamnesisError::Io(_)) => {}
            Ok(_) => panic!("expected Err for truncated JSON tail, got Ok"),
            Err(other) => panic!("expected AnamnesisError::Io, got {other:?}"),
        }
    }
}
