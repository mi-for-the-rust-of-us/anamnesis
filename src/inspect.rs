// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt;

use crate::limits::ParseLimits;
use crate::model::TargetDtype;
use crate::parse::safetensors::{Dtype, QuantScheme, SafetensorsHeader, TensorRole};

/// Caller-supplied options for [`ParsedModel::inspect_with_options`](crate::ParsedModel::inspect_with_options).
///
/// Currently carries only the output dtype the size estimate should assume; the
/// `#[non_exhaustive]` attribute lets future knobs be added without a breaking
/// change. Construct with [`InspectOptions::new`] (or
/// [`InspectOptions::default`], which is identical) and chain the setters:
///
/// ```rust
/// use anamnesis::{InspectOptions, TargetDtype};
///
/// let opts = InspectOptions::new().with_output_dtype(TargetDtype::F32);
/// assert_eq!(opts.output_dtype, TargetDtype::F32);
/// ```
///
/// The builder shape deliberately mirrors
/// [`RememberOptions`](crate::RememberOptions) and
/// [`ConvertOptions`](crate::ConvertOptions): one spelling for one concept
/// across the three option types.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InspectOptions {
    /// Output dtype the dequantised-size estimate assumes.
    ///
    /// Defaults to [`TargetDtype::BF16`], matching
    /// [`ParsedModel::remember`](crate::ParsedModel::remember)'s own default,
    /// so `inspect()` and an unqualified `remember()` always agree.
    pub output_dtype: TargetDtype,
    /// Resource budget applied to the parse the inspect performs, if any.
    ///
    /// Read **only** by the entry points that read an artefact —
    /// `inspect_npz[_from_reader]_with_options`,
    /// `inspect_gguf_from_reader_with_options`,
    /// `inspect_pth_from_reader_with_options`. The methods that summarise an
    /// **already-parsed** value
    /// ([`ParsedModel::inspect_with_options`](crate::ParsedModel::inspect_with_options)
    /// and its `GGUF` / `.pth` counterparts) have nothing left to bound and
    /// ignore it: the budget that mattered was the one passed to the parse.
    ///
    /// Defaults to [`ParseLimits::default`], i.e. unbounded beyond the
    /// permanent per-format caps. Added in v0.7.6, because before it there was
    /// **no way at all** to bound a summary inspect — the recommended first
    /// call on an untrusted file was the one call a caller could not tighten,
    /// which inverts the whole point of [`ParseLimits`].
    pub limits: ParseLimits,
}

impl InspectOptions {
    /// Returns options with the built-in defaults: a `BF16` size estimate and
    /// an unbounded budget.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            output_dtype: TargetDtype::BF16,
            limits: ParseLimits::unbounded(),
        }
    }

    /// Sets the output dtype the dequantised-size estimate assumes.
    #[must_use]
    pub const fn with_output_dtype(mut self, dtype: TargetDtype) -> Self {
        self.output_dtype = dtype;
        self
    }

    /// Sets the resource budget for the parse this inspect performs.
    ///
    /// Ignored by the methods that summarise an already-parsed value; see
    /// [`limits`](Self::limits).
    #[must_use]
    pub fn with_limits(mut self, limits: ParseLimits) -> Self {
        self.limits = limits;
        self
    }
}

impl Default for InspectOptions {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary information produced by inspecting a parsed `.safetensors` file.
///
/// Built on [`SafetensorsHeader`] — no file I/O, no re-read. All fields are
/// derived from the parsed header metadata.
///
/// `#[non_exhaustive]` since v0.7.4: the Python bindings freeze this shape in
/// Phase 8, and a struct whose fields are all `pub` cannot otherwise gain one
/// without a breaking change. Construct it through
/// [`ParsedModel::inspect`](crate::ParsedModel::inspect) or
/// [`ParsedModel::inspect_with_options`](crate::ParsedModel::inspect_with_options)
/// rather than a literal.
#[derive(Debug, Clone)]
#[non_exhaustive]
#[must_use]
pub struct InspectInfo {
    /// Detected quantization scheme (e.g., `FineGrainedFp8`, `PerTensorFp8`).
    pub format: QuantScheme,
    /// Number of tensors the header declares, counting **every** role.
    ///
    /// The role-specific counts below partition this total, so they sum to it;
    /// summing them was the only way to obtain it before v0.7.6, and that sum
    /// silently under-counted, because [`TensorRole::QuantState`] has no
    /// counter of its own. Present so
    /// [`InspectSummary::tensor_count`] means the same thing here as it does
    /// for every other format.
    pub tensor_count: usize,
    /// Number of quantized weight tensors.
    pub quantized: usize,
    /// Number of scale factor tensors (non-zero only for fine-grained `FP8`).
    pub scales: usize,
    /// Number of passthrough tensors (norms, embeddings, `lm_head`).
    pub passthrough: usize,
    /// Unique dtypes of scale factor tensors, in order of first occurrence.
    pub scale_dtypes: Vec<Dtype>,
    /// Number of zero-point tensors (`GPTQ` `.qzeros`).
    pub zeropoints: usize,
    /// Number of group-index tensors (`GPTQ` `.g_idx`).
    pub group_indices: usize,
    /// Number of quant-map tensors (`BnB` lookup tables).
    pub quant_maps: usize,
    /// Number of nested-scale tensors (`BnB` double-quant absmax).
    pub nested_scales: usize,
    /// Total tensor data size in bytes (as stored in the file).
    pub current_size: u64,
    /// Estimated tensor data size in bytes after dequantization to
    /// [`output_dtype`](Self::output_dtype).
    ///
    /// Feeds the inspect-before-parse policy gate, so it has to track the width
    /// the caller will actually request: at `F32` the figure is double the
    /// `BF16` one for the dequantised share, while passthrough tensors keep
    /// their source dtype and contribute the same bytes either way.
    pub dequantized_size: u64,
    /// The output dtype [`dequantized_size`](Self::dequantized_size) assumes.
    ///
    /// `BF16` unless the caller asked otherwise via
    /// [`InspectOptions::with_output_dtype`]. Recorded on the struct so a
    /// figure can never be read without the width it was computed for.
    pub output_dtype: TargetDtype,
}

impl From<&SafetensorsHeader> for InspectInfo {
    /// The `BF16` special case of `InspectInfo::with_options`, preserved so
    /// every pre-v0.7.4 caller compiles unchanged.
    fn from(header: &SafetensorsHeader) -> Self {
        Self::with_options(header, &InspectOptions::new())
    }
}

impl InspectInfo {
    /// Builds the summary, sizing [`dequantized_size`](Self::dequantized_size)
    /// for `options.output_dtype`.
    pub(crate) fn with_options(header: &SafetensorsHeader, options: &InspectOptions) -> Self {
        // By reference, not by value: `InspectOptions` stopped being `Copy` when
        // it gained `limits`, and the crate's rule for a non-`Copy` argument that
        // is only read is `&T`. `options.limits` is deliberately untouched here —
        // this summarises a header already parsed under whatever budget the parse
        // was given.
        let output_dtype = options.output_dtype;
        let out_bytes = u64::try_from(output_dtype.byte_size()).unwrap_or(2);
        let tensor_count = header.tensors.len();
        let quantized = header.quantized_count();
        let scales = header.scale_count();
        let passthrough = header.passthrough_count();
        let zeropoints = header.zeropoint_count();
        let group_indices = header.group_index_count();
        let quant_maps = header.quant_map_count();
        let nested_scales = header.nested_scale_count();

        let mut scale_dtypes: Vec<Dtype> = Vec::new();
        for entry in header.scale_tensors() {
            if !scale_dtypes.contains(&entry.dtype) {
                scale_dtypes.push(entry.dtype);
            }
        }

        let mut current_size: u64 = 0;
        let mut dequantized_size: u64 = 0;

        // All accumulation here is `saturating_*`, not raw `+`/`*`. Both
        // `byte_len` and `num_elements` derive from a header-declared shape and
        // already saturate to `usize::MAX` on overflow; feeding those into a raw
        // `out_elements * out_bytes` (up to ×4 at F32) and a running `+=` could
        // exceed `u64::MAX` — a debug-build panic and a silent release wrap of
        // the very figure the inspect-before-parse gate reads. `SafetensorsHeader`
        // has public fields and is not `#[non_exhaustive]`, so a caller can hand
        // us such a header directly, bypassing the upstream `safetensors`
        // validation that guards the file-parse fronts. Saturating keeps this in
        // line with the crate-wide "checked/saturating on every header-derived
        // value" invariant and, on an absurd shape, yields `u64::MAX` — which the
        // policy gate reads as "too big", the fail-closed direction.
        for entry in &header.tensors {
            // CAST: usize → u64, byte lengths fit in u64 for any realistic model
            #[allow(clippy::as_conversions)]
            let byte_len = entry.byte_len() as u64;
            current_size = current_size.saturating_add(byte_len);

            match entry.role {
                TensorRole::Quantized => {
                    // BnB NF4/FP4: each U8 byte packs 2 values, so the element
                    // count is 2 × byte_len. Every other scheme is 1 element per
                    // stored element. Both then multiply by the *requested*
                    // output width rather than a hard-coded 2 — the v0.7.4
                    // change, without which `--to f32` would under-report this
                    // estimate by exactly 2× in the one place whose whole job is
                    // telling a caller how big the result will be.
                    // CAST: usize → u64, element count fits in u64 for any realistic model
                    #[allow(clippy::as_conversions)]
                    let out_elements =
                        if header.scheme == QuantScheme::Bnb4 && entry.dtype == Dtype::U8 {
                            (entry.byte_len() as u64).saturating_mul(2)
                        } else {
                            entry.num_elements() as u64
                        };
                    dequantized_size =
                        dequantized_size.saturating_add(out_elements.saturating_mul(out_bytes));
                }
                TensorRole::Scale
                | TensorRole::ZeroPoint
                | TensorRole::GroupIndex
                | TensorRole::QuantMap
                | TensorRole::NestedScale
                | TensorRole::QuantState => {
                    // Companion tensors are consumed during dequantization,
                    // not written to the output file.
                }
                TensorRole::Passthrough => {
                    // Passthrough tensors are copied as-is.
                    dequantized_size = dequantized_size.saturating_add(byte_len);
                }
            }
        }

        Self {
            format: header.scheme,
            tensor_count,
            quantized,
            scales,
            passthrough,
            scale_dtypes,
            zeropoints,
            group_indices,
            quant_maps,
            nested_scales,
            current_size,
            dequantized_size,
            output_dtype,
        }
    }

    /// Returns the number of bytes of precision that Lethe took
    /// (difference between dequantized and current size).
    ///
    /// Zero when the model is unquantized.
    #[must_use]
    pub const fn lethe_took(&self) -> u64 {
        self.dequantized_size.saturating_sub(self.current_size)
    }
}

impl fmt::Display for InspectInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Format:      {}", self.format)?;

        if self.scales > 0 {
            let dtype_list: String = self
                .scale_dtypes
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            write!(
                f,
                "\nQuantized:   {} tensors (weights) + {} scale tensors ({dtype_list})",
                self.quantized, self.scales,
            )?;
        } else {
            write!(f, "\nQuantized:   {} tensors (weights)", self.quantized)?;
        }

        write!(
            f,
            "\nPassthrough: {} tensors (norms, embeddings)",
            self.passthrough,
        )?;

        if self.zeropoints > 0 {
            write!(f, "\nZero-points: {} tensors", self.zeropoints)?;
        }

        if self.group_indices > 0 {
            write!(
                f,
                "\nGroup index: {} tensors (activation-order)",
                self.group_indices,
            )?;
        }

        if self.quant_maps > 0 {
            write!(
                f,
                "\nQuant maps:  {} tensors (lookup tables)",
                self.quant_maps,
            )?;
        }

        if self.nested_scales > 0 {
            write!(
                f,
                "\nNested:      {} tensors (double-quant absmax)",
                self.nested_scales,
            )?;
        }

        let scheme_label = match self.format {
            QuantScheme::Gptq | QuantScheme::Awq => "GPTQ/AWQ",
            QuantScheme::Bnb4 => "BnB NF4/FP4",
            QuantScheme::BnbInt8 => "BnB INT8",
            QuantScheme::Unquantized => "unquantized",
            QuantScheme::FineGrainedFp8
            | QuantScheme::PerChannelFp8
            | QuantScheme::PerTensorFp8 => "FP8",
        };
        // The width label is read off `output_dtype`, never hard-coded. Until
        // v0.7.4 this printed a literal `(BF16)`, which was correct only while
        // `BF16` was the sole possible answer; once `InspectOptions` could ask
        // for an `F32` estimate, the same line rendered a doubled figure under a
        // `BF16` label. That is the failure mode `output_dtype` exists to make
        // impossible, and it is worse than an unlabelled number rather than
        // merely unhelpful.
        write!(
            f,
            "\nSize:        {} ({scheme_label}) -> {} ({})",
            format_bytes(self.current_size),
            format_bytes(self.dequantized_size),
            self.output_dtype,
        )?;

        if self.format != QuantScheme::Unquantized {
            write!(
                f,
                "\nLethe took:  ~{} of precision",
                format_bytes(self.lethe_took()),
            )?;
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// InspectSummary — the one question asked of four formats
// ---------------------------------------------------------------------------

/// Seals [`InspectSummary`] against outside implementations.
///
/// The trait describes what *this crate's* four inspect types report; an
/// external implementation could not be produced by any entry point here, so
/// permitting one would only invite a type that the inspect-before-parse gate
/// cannot actually be handed. Same shape as `OutputElement`'s seal.
pub(crate) mod sealed {
    /// Implemented only for the four inspect summaries in this crate.
    pub trait Sealed {}
}

/// The four numbers every format's inspect can answer, whatever else it carries.
///
/// Before v0.7.6 only safetensors' [`InspectInfo`] could answer the third and
/// fourth: `GgufInspectInfo`, `PthInspectInfo` and `NpzInspectInfo` reported
/// on-disk totals only, so the [`ParseLimits`] inspect-before-parse gate could
/// not ask *"how much memory will this become at the width I intend"* about a
/// `GGUF` — the one format whose answer cannot be guessed from the file size,
/// because the expansion ratio is per-kernel (a `Q2_K` tensor and a `Q6_K`
/// tensor of equal length expand by different multiples).
///
/// Implementations are per-format and the trait is sealed; obtain one from that
/// format's `inspect*` entry point.
///
/// ```rust
/// # #[cfg(feature = "gguf")]
/// # fn run(info: &anamnesis::GgufInspectInfo) {
/// use anamnesis::InspectSummary;
///
/// // The same three lines read every format's summary.
/// let on_disk = info.current_size();
/// let in_memory = info.dequantized_size();
/// let width = info.output_dtype();
/// # let _ = (on_disk, in_memory, width);
/// # }
/// ```
pub trait InspectSummary: sealed::Sealed {
    /// Number of tensors the artefact declares.
    #[must_use]
    fn tensor_count(&self) -> usize;

    /// Total tensor-data size **as stored**, in bytes.
    ///
    /// What the file costs to hold or transfer, before any dequantisation.
    #[must_use]
    fn current_size(&self) -> u64;

    /// Estimated tensor-data size in bytes **after** dequantisation to
    /// [`output_dtype`](Self::output_dtype).
    ///
    /// This is the figure the inspect-before-parse policy gate checks. It is
    /// equal to [`current_size`](Self::current_size) for the formats that store
    /// nothing block-quantised (`NPZ`, `.pth`), and strictly larger for a
    /// quantised safetensors or `GGUF`.
    ///
    /// Saturating: an absurd declared shape yields [`u64::MAX`], which a gate
    /// reads as "too big" — the fail-closed direction.
    #[must_use]
    fn dequantized_size(&self) -> u64;

    /// The output dtype [`dequantized_size`](Self::dequantized_size) assumes.
    ///
    /// Carried alongside the figure so a number can never be read without the
    /// width it was computed for. On `NPZ` and `.pth` the width is *vacuous*
    /// rather than wrong: nothing is dequantised, so the requested dtype is
    /// recorded and changes no byte — the same way `remember --to f32` on a
    /// `.pth` is accepted and narrows nothing.
    #[must_use]
    fn output_dtype(&self) -> TargetDtype;

    /// Bytes of precision that dequantisation restores, i.e.
    /// `dequantized_size - current_size`.
    ///
    /// Zero when the artefact is unquantised. Provided rather than required:
    /// no implementation needs to override it.
    #[must_use]
    fn expansion(&self) -> u64 {
        self.dequantized_size().saturating_sub(self.current_size())
    }
}

impl sealed::Sealed for InspectInfo {}

impl InspectSummary for InspectInfo {
    fn tensor_count(&self) -> usize {
        self.tensor_count
    }

    fn current_size(&self) -> u64 {
        self.current_size
    }

    fn dequantized_size(&self) -> u64 {
        self.dequantized_size
    }

    fn output_dtype(&self) -> TargetDtype {
        self.output_dtype
    }
}

/// Format a byte count as a human-readable string.
///
/// Examples: `"0 B"`, `"512 B"`, `"45.6 KB"`, `"302 MB"`, `"4.35 GB"`.
#[must_use]
#[allow(clippy::as_conversions, clippy::cast_precision_loss)]
pub fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * 1024;
    const GB: u64 = 1024 * 1024 * 1024;

    // CAST: u64 → f64 throughout; model sizes are well within f64 mantissa range
    // (52-bit mantissa covers exact integers up to 2^53 ≈ 9 PB).
    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.0} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

#[cfg(test)]
#[allow(clippy::panic, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use crate::parse::safetensors::TensorEntry;

    fn make_entry(name: &str, dtype: Dtype, role: TensorRole, shape: &[usize]) -> TensorEntry {
        let num_elements: usize = shape.iter().product();
        let byte_len = num_elements * dtype.byte_size();
        TensorEntry {
            name: name.to_owned(),
            dtype,
            shape: shape.to_vec(),
            data_offsets: (0, byte_len),
            role,
        }
    }

    // -- format_bytes --------------------------------------------------------

    #[test]
    fn format_bytes_zero() {
        assert_eq!(format_bytes(0), "0 B");
    }

    #[test]
    fn format_bytes_small() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1023), "1023 B");
    }

    #[test]
    fn format_bytes_kilobytes() {
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(1536), "1.5 KB");
    }

    #[test]
    fn format_bytes_megabytes() {
        assert_eq!(format_bytes(1024 * 1024), "1 MB");
        assert_eq!(format_bytes(302 * 1024 * 1024), "302 MB");
    }

    #[test]
    fn format_bytes_gigabytes() {
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00 GB");
        // 4.35 GB ≈ 4672 MB
        assert_eq!(format_bytes(4_672 * 1024 * 1024), "4.56 GB");
    }

    // -- InspectInfo from SafetensorsHeader -----------------------------------

    #[test]
    fn inspect_unquantized() {
        let header = SafetensorsHeader {
            tensors: vec![
                make_entry("norm.weight", Dtype::BF16, TensorRole::Passthrough, &[2048]),
                make_entry(
                    "lm_head.weight",
                    Dtype::BF16,
                    TensorRole::Passthrough,
                    &[32000, 2048],
                ),
            ],
            scheme: QuantScheme::Unquantized,
            metadata: None,
            header_size: 0,
            gptq_config: None,
            awq_config: None,
            bnb_config: None,
        };
        let info = InspectInfo::from(&header);

        assert_eq!(info.format, QuantScheme::Unquantized);
        assert_eq!(info.quantized, 0);
        assert_eq!(info.scales, 0);
        assert_eq!(info.passthrough, 2);
        assert_eq!(info.current_size, info.dequantized_size);
        assert_eq!(info.lethe_took(), 0);
    }

    #[test]
    fn inspect_fine_grained_fp8() {
        let header = SafetensorsHeader {
            tensors: vec![
                make_entry(
                    "layer.weight",
                    Dtype::F8E4M3,
                    TensorRole::Quantized,
                    &[2048, 2048],
                ),
                make_entry(
                    "layer.weight_scale_inv",
                    Dtype::F32,
                    TensorRole::Scale,
                    &[16, 16],
                ),
                make_entry("norm.weight", Dtype::BF16, TensorRole::Passthrough, &[2048]),
            ],
            scheme: QuantScheme::FineGrainedFp8,
            metadata: None,
            header_size: 0,
            gptq_config: None,
            awq_config: None,
            bnb_config: None,
        };
        let info = InspectInfo::from(&header);

        assert_eq!(info.quantized, 1);
        assert_eq!(info.scales, 1);
        assert_eq!(info.passthrough, 1);

        // Quantized: 2048×2048 = 4_194_304 elements × 1 byte = 4_194_304 bytes
        // Scale: 16×16 = 256 × 4 bytes = 1024 bytes
        // Passthrough: 2048 × 2 bytes = 4096 bytes
        let expected_current = 4_194_304 + 1024 + 4096;
        assert_eq!(info.current_size, expected_current);

        // Dequantized: quantized → 4_194_304 × 2 = 8_388_608, scale → 0, passthrough → 4096
        let expected_deq = 8_388_608 + 4096;
        assert_eq!(info.dequantized_size, expected_deq);

        assert!(info.lethe_took() > 0);
    }

    #[test]
    fn inspect_per_tensor_fp8() {
        let header = SafetensorsHeader {
            tensors: vec![
                make_entry(
                    "layer.weight",
                    Dtype::F8E4M3,
                    TensorRole::Quantized,
                    &[1024, 1024],
                ),
                make_entry("norm.weight", Dtype::BF16, TensorRole::Passthrough, &[1024]),
            ],
            scheme: QuantScheme::PerTensorFp8,
            metadata: None,
            header_size: 0,
            gptq_config: None,
            awq_config: None,
            bnb_config: None,
        };
        let info = InspectInfo::from(&header);

        assert_eq!(info.quantized, 1);
        assert_eq!(info.scales, 0);
        assert_eq!(info.passthrough, 1);

        // Quantized: 1024×1024 = 1_048_576 × 1 byte
        // Passthrough: 1024 × 2 bytes = 2048
        assert_eq!(info.current_size, 1_048_576 + 2048);
        // Dequantized: 1_048_576 × 2 + 2048
        assert_eq!(info.dequantized_size, 2_097_152 + 2048);
    }

    // -- Dtype-aware size estimate (v0.7.4) ----------------------------------

    /// Builds the fine-grained `FP8` header the estimate tests share.
    fn fp8_header() -> SafetensorsHeader {
        SafetensorsHeader {
            tensors: vec![
                make_entry(
                    "layer.weight",
                    Dtype::F8E4M3,
                    TensorRole::Quantized,
                    &[2048, 2048],
                ),
                make_entry(
                    "layer.weight_scale_inv",
                    Dtype::F32,
                    TensorRole::Scale,
                    &[16, 16],
                ),
                make_entry("norm.weight", Dtype::BF16, TensorRole::Passthrough, &[2048]),
            ],
            scheme: QuantScheme::FineGrainedFp8,
            metadata: None,
            header_size: 0,
            gptq_config: None,
            awq_config: None,
            bnb_config: None,
        }
    }

    /// The estimate must scale with the requested width for the **dequantised**
    /// share, and leave the passthrough share alone. Getting this wrong is a 2×
    /// under-report in the one figure whose entire job is telling a caller how
    /// big the result will be.
    #[test]
    fn dequantized_size_scales_with_output_dtype() {
        let header = fp8_header();
        let quantized_elements: u64 = 2048 * 2048;
        let passthrough_bytes: u64 = 2048 * 2; // BF16 norm, copied as-is

        for (target, width) in [
            (TargetDtype::BF16, 2_u64),
            (TargetDtype::F16, 2),
            (TargetDtype::F32, 4),
        ] {
            let info = InspectInfo::with_options(
                &header,
                &InspectOptions::new().with_output_dtype(target),
            );
            assert_eq!(
                info.dequantized_size,
                quantized_elements * width + passthrough_bytes,
                "{target}: dequantised share must be {width} B/element and the \
                 passthrough share must not move"
            );
            assert_eq!(info.output_dtype, target, "{target}: width not recorded");
        }
    }

    /// `inspect()` and `inspect_with_options(default)` must agree, so the
    /// no-argument call keeps meaning exactly what `remember` defaults to.
    #[test]
    fn default_options_match_the_bare_from_impl() {
        let header = fp8_header();
        let bare = InspectInfo::from(&header);
        let explicit = InspectInfo::with_options(&header, &InspectOptions::new());
        assert_eq!(bare.dequantized_size, explicit.dequantized_size);
        assert_eq!(bare.output_dtype, TargetDtype::BF16);
    }

    /// A header-derived shape large enough that the dequantised-size estimate
    /// would overflow `u64` must **saturate**, never panic (debug) or wrap
    /// silently (release). `SafetensorsHeader` has public fields, so a caller can
    /// construct this directly, bypassing the upstream `safetensors` validation
    /// that guards the file-parse fronts; the size arithmetic must stand on its
    /// own. The suite runs in debug, where the pre-fix `*`/`+=` would panic on
    /// overflow — so this test failing to panic *is* the guard.
    #[test]
    fn oversized_shape_saturates_rather_than_overflowing() {
        // One F8 tensor (1 byte/element) whose element count is usize::MAX, so
        // the estimate overflows u64 at every width (×2 for BF16/F16, ×4 for F32)
        // and must saturate to u64::MAX rather than wrap or panic.
        let n = usize::MAX;
        let header = SafetensorsHeader {
            tensors: vec![TensorEntry {
                name: "layer.weight".to_owned(),
                dtype: Dtype::F8E4M3,
                shape: vec![n],
                data_offsets: (0, n),
                role: TensorRole::Quantized,
            }],
            scheme: QuantScheme::PerTensorFp8,
            metadata: None,
            header_size: 0,
            gptq_config: None,
            awq_config: None,
            bnb_config: None,
        };

        for target in [TargetDtype::BF16, TargetDtype::F16, TargetDtype::F32] {
            let info = InspectInfo::with_options(
                &header,
                &InspectOptions::new().with_output_dtype(target),
            );
            // The exact figure is meaningless once saturated; the contract is
            // only that we reached here without a panic/wrap and produced the
            // fail-closed sentinel.
            assert_eq!(
                info.dequantized_size,
                u64::MAX,
                "{target}: an overflowing estimate must saturate to u64::MAX (fail-closed)"
            );
            // `lethe_took` must stay panic-free on the saturated figures too.
            let _ = info.lethe_took();
        }
    }

    /// `BnB4` packs two values per stored byte, so its element count is
    /// `2 × byte_len` rather than `num_elements`. That factor must survive the
    /// v0.7.4 rewrite independently of the width multiplier.
    #[test]
    fn bnb4_keeps_its_two_values_per_byte_factor() {
        let header = SafetensorsHeader {
            tensors: vec![make_entry(
                "layer.weight",
                Dtype::U8,
                TensorRole::Quantized,
                &[1024, 1],
            )],
            scheme: QuantScheme::Bnb4,
            metadata: None,
            header_size: 0,
            gptq_config: None,
            awq_config: None,
            bnb_config: None,
        };
        // 1024 stored bytes -> 2048 values.
        for (target, want) in [(TargetDtype::BF16, 2048 * 2), (TargetDtype::F32, 2048 * 4)] {
            let info = InspectInfo::with_options(
                &header,
                &InspectOptions::new().with_output_dtype(target),
            );
            assert_eq!(info.dequantized_size, want, "{target}");
        }
    }

    // -- Display output ------------------------------------------------------

    /// The rendered size line names the width the estimate was computed for.
    ///
    /// Regression guard for a real defect this phase introduced and then fixed.
    /// `dequantized_size` became dtype-aware in v0.7.4 while `Display` still
    /// printed a literal `(BF16)`, so asking for the `F32` estimate rendered a
    /// doubled figure under a `BF16` label. `output_dtype` is carried on the
    /// struct precisely so that cannot happen, which only helps if the renderer
    /// actually reads it.
    #[test]
    fn display_size_line_names_the_dtype_it_assumed() {
        for (target, label, want_bytes) in [
            (TargetDtype::BF16, "(BF16)", 2_048_u64),
            (TargetDtype::F32, "(F32)", 4_096),
            (TargetDtype::F16, "(F16)", 2_048),
        ] {
            let header = SafetensorsHeader {
                tensors: vec![make_entry(
                    "layer.weight",
                    Dtype::F8E4M3,
                    TensorRole::Quantized,
                    &[32, 32],
                )],
                scheme: QuantScheme::PerTensorFp8,
                metadata: None,
                header_size: 0,
                gptq_config: None,
                awq_config: None,
                bnb_config: None,
            };
            let info = InspectInfo::with_options(
                &header,
                &InspectOptions::new().with_output_dtype(target),
            );

            assert_eq!(info.output_dtype, target);
            assert_eq!(info.dequantized_size, want_bytes, "{target}");

            let rendered = info.to_string();
            assert!(
                rendered.contains(label),
                "{target}: size line must carry {label}, got:\n{rendered}"
            );
            // And must not carry a *different* width's label.
            for other in ["(BF16)", "(F32)", "(F16)"] {
                if other != label {
                    assert!(
                        !rendered.contains(other),
                        "{target}: size line must not claim {other}, got:\n{rendered}"
                    );
                }
            }
        }
    }

    #[test]
    fn display_per_tensor_fp8() {
        let info = InspectInfo {
            // Set explicitly: the role counts below partition it, and the
            // `Display` under test does not read it.
            tensor_count: 0,
            format: QuantScheme::PerTensorFp8,
            quantized: 224,
            scales: 0,
            passthrough: 53,
            scale_dtypes: vec![],
            zeropoints: 0,
            group_indices: 0,
            quant_maps: 0,
            nested_scales: 0,
            current_size: 4_672 * 1024 * 1024,
            dequantized_size: 8_269 * 1024 * 1024,
            output_dtype: TargetDtype::BF16,
        };
        let output = info.to_string();

        assert!(output.contains("Per-tensor FP8 (E4M3)"));
        assert!(output.contains("224 tensors (weights)"));
        assert!(!output.contains("scale tensors"));
        assert!(output.contains("53 tensors"));
        assert!(output.contains("Lethe took"));
    }

    #[test]
    fn display_fine_grained_fp8() {
        let info = InspectInfo {
            // Set explicitly: the role counts below partition it, and the
            // `Display` under test does not read it.
            tensor_count: 0,
            format: QuantScheme::FineGrainedFp8,
            quantized: 180,
            scales: 180,
            passthrough: 31,
            scale_dtypes: vec![Dtype::F32],
            zeropoints: 0,
            group_indices: 0,
            quant_maps: 0,
            nested_scales: 0,
            current_size: 1_310 * 1024 * 1024,
            dequantized_size: 2_580 * 1024 * 1024,
            output_dtype: TargetDtype::BF16,
        };
        let output = info.to_string();

        assert!(output.contains("Fine-grained FP8 (E4M3), 128x128 blocks"));
        assert!(output.contains("180 tensors (weights) + 180 scale tensors (F32)"));
        assert!(output.contains("31 tensors"));
        assert!(output.contains("Lethe took"));
    }

    #[test]
    fn display_fine_grained_fp8_bf16_scales() {
        let info = InspectInfo {
            // Set explicitly: the role counts below partition it, and the
            // `Display` under test does not read it.
            tensor_count: 0,
            format: QuantScheme::FineGrainedFp8,
            quantized: 180,
            scales: 180,
            passthrough: 31,
            scale_dtypes: vec![Dtype::BF16],
            zeropoints: 0,
            group_indices: 0,
            quant_maps: 0,
            nested_scales: 0,
            current_size: 1_310 * 1024 * 1024,
            dequantized_size: 2_580 * 1024 * 1024,
            output_dtype: TargetDtype::BF16,
        };
        let output = info.to_string();

        assert!(output.contains("180 scale tensors (BF16)"));
        assert!(!output.contains("(F32)"));
    }

    #[test]
    fn display_unquantized_omits_lethe() {
        let info = InspectInfo {
            // Set explicitly: the role counts below partition it, and the
            // `Display` under test does not read it.
            tensor_count: 0,
            format: QuantScheme::Unquantized,
            quantized: 0,
            scales: 0,
            passthrough: 100,
            scale_dtypes: vec![],
            zeropoints: 0,
            group_indices: 0,
            quant_maps: 0,
            nested_scales: 0,
            current_size: 1024 * 1024 * 1024,
            dequantized_size: 1024 * 1024 * 1024,
            output_dtype: TargetDtype::BF16,
        };
        let output = info.to_string();

        assert!(output.contains("Unquantized"));
        assert!(!output.contains("Lethe took"));
    }
}
