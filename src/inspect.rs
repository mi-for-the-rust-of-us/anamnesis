// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt;

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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InspectOptions {
    /// Output dtype the dequantised-size estimate assumes.
    ///
    /// Defaults to [`TargetDtype::BF16`], matching
    /// [`ParsedModel::remember`](crate::ParsedModel::remember)'s own default,
    /// so `inspect()` and an unqualified `remember()` always agree.
    pub output_dtype: TargetDtype,
}

impl InspectOptions {
    /// Returns options with the built-in defaults (a `BF16` size estimate).
    ///
    /// `const`: the struct is a single `Copy` enum, so there is nothing to
    /// allocate.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            output_dtype: TargetDtype::BF16,
        }
    }

    /// Sets the output dtype the dequantised-size estimate assumes.
    #[must_use]
    pub const fn with_output_dtype(mut self, dtype: TargetDtype) -> Self {
        self.output_dtype = dtype;
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
        Self::with_options(header, InspectOptions::new())
    }
}

impl InspectInfo {
    /// Builds the summary, sizing [`dequantized_size`](Self::dequantized_size)
    /// for `options.output_dtype`.
    pub(crate) fn with_options(header: &SafetensorsHeader, options: InspectOptions) -> Self {
        let out_bytes = u64::try_from(options.output_dtype.byte_size()).unwrap_or(2);
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

        for entry in &header.tensors {
            // CAST: usize → u64, byte lengths fit in u64 for any realistic model
            #[allow(clippy::as_conversions)]
            let byte_len = entry.byte_len() as u64;
            current_size += byte_len;

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
                            entry.byte_len() as u64 * 2
                        } else {
                            entry.num_elements() as u64
                        };
                    dequantized_size += out_elements * out_bytes;
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
                    dequantized_size += byte_len;
                }
            }
        }

        Self {
            format: header.scheme,
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
            output_dtype: options.output_dtype,
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
        write!(
            f,
            "\nSize:        {} ({scheme_label}) -> {} (BF16)",
            format_bytes(self.current_size),
            format_bytes(self.dequantized_size),
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
            let info =
                InspectInfo::with_options(&header, InspectOptions::new().with_output_dtype(target));
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
        let explicit = InspectInfo::with_options(&header, InspectOptions::new());
        assert_eq!(bare.dequantized_size, explicit.dequantized_size);
        assert_eq!(bare.output_dtype, TargetDtype::BF16);
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
            let info =
                InspectInfo::with_options(&header, InspectOptions::new().with_output_dtype(target));
            assert_eq!(info.dequantized_size, want, "{target}");
        }
    }

    // -- Display output ------------------------------------------------------

    #[test]
    fn display_per_tensor_fp8() {
        let info = InspectInfo {
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
