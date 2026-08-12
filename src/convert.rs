// SPDX-License-Identifier: MIT OR Apache-2.0

//! Format conversion through an in-memory **`BF16` hub**.
//!
//! `convert` normalises any supported input into an in-memory hub — a list of owned
//! tensors carrying `(name, shape, dtype, bytes)` — then writes that hub to the
//! requested [`ConvertTarget`]. Routing every `(input × target)` pair through one
//! hub replaces the per-cell handlers of v0.6.0, which rejected most
//! combinations, and means a new input format costs one *reader* while a new
//! target costs one *writer*.
//!
//! # Dtype policy
//!
//! The hub is a `BF16` *pivot*, not a `BF16` *floor*:
//!
//! - **Quantised** tensors (`FP8` / `GPTQ` / `AWQ` / `BnB` safetensors, quantised
//!   `GGUF` blocks) are dequantised to `BF16` — the only lossless-in-intent
//!   representation the crate produces.
//! - **Scalar** tensors keep their **original dtype** (`F64` / `F32` / `F16` /
//!   `BF16` / `I8`–`I64` / `U8` / `Bool`), so `.pth` → safetensors and
//!   `NPZ`-`F32` → `GGUF` stay bit-for-bit lossless.
//! - `BF16` is materialised for a *scalar* tensor only where the target demands
//!   it — currently just the `BnB-NF4` encoder, whose input contract is `BF16`.
//!
//! Shapes are held **row-major** (the safetensors / `NumPy` convention); the
//! `GGUF` reader and writer reverse them at the boundary, since `GGUF` stores
//! dimensions most-significant-first.
//!
//! # Memory
//!
//! The hub is eager and owned: peak heap is `O(model)` — every tensor is
//! materialised as an owned hub tensor before the writer runs — plus the target
//! writer's own buffer. The safetensors and `GGUF` writers borrow the hub's
//! bytes and stream to the output (adding ~0); only the `BnB-NF4` encoder
//! allocates a target buffer (its packed NF4 output). The readers and writers
//! take **no second full copy**: an `NPZ` parse map is drained into the hub
//! rather than cloned, and an already-`BF16` hub is borrowed to the `BnB`
//! encoder via `Cow` instead of re-copied — see
//! `docs/perf-experiments.md`, Experiment 9. Dropping even the single
//! materialised hub (streaming output) is `ROADMAP.md` Phase 10.

// `Cow` is used by the `bnb` (`to_bf16_bytes`) and `gguf` (`write_gguf_target`)
// writers to borrow rather than copy; unused when neither feature is on.
#[cfg(any(feature = "bnb", feature = "gguf"))]
use std::borrow::Cow;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::{AnamnesisError, Dtype, ParseLimits};

// ---------------------------------------------------------------------------
// Public surface
// ---------------------------------------------------------------------------

/// Target format for [`convert`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ConvertTarget {
    /// `safetensors` (alias `bf16`) — dequantise any quantised input to `BF16`,
    /// passing scalar tensors through in their original dtype.
    Safetensors,
    /// `gguf` — an unquantised (scalar) `GGUF` file. Quantised `GGUF` emit
    /// (`gguf-q4km`, …) needs the Phase 8.5 encode kernels.
    Gguf,
    /// `bnb-nf4` — `BitsAndBytes`-NF4 safetensors: 2-D float weights encoded to
    /// NF4, everything else passed through as `BF16`.
    BnbNf4,
}

impl ConvertTarget {
    /// Parses a CLI `--to` value. Accepted (case-insensitive): `safetensors` /
    /// `bf16`, `gguf`, `bnb-nf4` / `bnb_nf4` / `nf4`.
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::Unsupported`] for an unrecognised target.
    pub fn parse(raw: &str) -> crate::Result<Self> {
        match raw.to_ascii_lowercase().as_str() {
            "safetensors" | "bf16" => Ok(Self::Safetensors),
            "gguf" => Ok(Self::Gguf),
            "bnb-nf4" | "bnb_nf4" | "nf4" => Ok(Self::BnbNf4),
            other => Err(AnamnesisError::Unsupported {
                format: other.to_owned(),
                detail: "supported convert targets: `safetensors` (alias `bf16`), \
                         `gguf`, `bnb-nf4`. Quantised GGUF targets need Phase 8.5"
                    .into(),
            }),
        }
    }

    /// The file extension a derived output path gets for this target.
    #[must_use]
    pub const fn extension(self) -> &'static str {
        match self {
            Self::Safetensors | Self::BnbNf4 => "safetensors",
            Self::Gguf => "gguf",
        }
    }

    /// The stem suffix a derived output path gets for this target.
    #[must_use]
    pub const fn suffix(self) -> &'static str {
        match self {
            Self::Safetensors => "bf16",
            Self::Gguf => "gguf",
            Self::BnbNf4 => "bnb-nf4",
        }
    }
}

/// Caller-supplied knobs for [`convert`].
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct ConvertOptions {
    /// Resource budget applied to the *input* parse. Defaults to
    /// [`ParseLimits::default`] (unbounded beyond the permanent per-format caps).
    pub limits: ParseLimits,
    /// Per-tensor dequantisation thread budget, with the same semantics as
    /// [`RememberOptions::threads`](crate::RememberOptions): `None` (the default)
    /// resolves to `min(available_parallelism, 4)`, `Some(n)` pins it to
    /// `n.max(1)`, and it is always 1 when the `parallel` feature is off.
    ///
    /// Applies to the **safetensors** input path (which reuses the model
    /// dequant) and, since v0.7.2, to the **`GGUF`** input path. The `NPZ` and
    /// `.pth` readers stay sequential by nature rather than by omission: neither
    /// format is block-quantised, so there is no dequantisation to spread — the
    /// `NPZ` reader drains an owned map without copying a byte, and the `.pth`
    /// reader's only per-tensor cost is a single `into_owned()`.
    ///
    /// The budget governs how the work is *scheduled*, never what is produced:
    /// output is byte-identical at any thread count.
    pub threads: Option<usize>,
    /// Caller-supplied `GGUF` key/value metadata, written verbatim and merged
    /// **over** any KV carried from a `GGUF` source (the caller wins on a key
    /// collision). Empty by default.
    ///
    /// anamnesis attaches no meaning to individual keys — it stamps what it is
    /// handed. Deriving *model* knowledge (architecture hyper-parameters,
    /// tokenizer arrays) from a source model's `config.json` / `tokenizer.json`
    /// is a packaging concern for a downstream crate.
    #[cfg(feature = "gguf")]
    pub gguf_metadata: HashMap<String, crate::GgufMetadataValue>,
}

impl ConvertOptions {
    /// Returns options with the default (unbounded) [`ParseLimits`] and no
    /// caller-supplied `GGUF` metadata.
    ///
    /// Not `const`: the `gguf` build carries a `HashMap`, whose `new` is not a
    /// `const fn`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the resource budget applied to the input parse.
    #[must_use]
    pub fn with_limits(mut self, limits: ParseLimits) -> Self {
        self.limits = limits;
        self
    }

    /// Sets the per-tensor dequantisation thread budget (clamped to at least 1).
    /// Overrides the `min(available_parallelism, 4)` default; ignored when the
    /// `parallel` feature is off (always sequential).
    #[must_use]
    pub fn with_threads(mut self, n: usize) -> Self {
        self.threads = Some(n.max(1));
        self
    }

    /// Sets the `GGUF` key/value metadata written to a `gguf` target, merged over
    /// any KV inherited from a `GGUF` source.
    #[cfg(feature = "gguf")]
    #[must_use]
    pub fn with_gguf_metadata(
        mut self,
        metadata: HashMap<String, crate::GgufMetadataValue>,
    ) -> Self {
        self.gguf_metadata = metadata;
        self
    }
}

/// What a [`convert`] call produced, for progress reporting.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct ConvertStats {
    /// Total tensors written.
    pub tensors: usize,
    /// Input tensors that were dequantised to `BF16` on the way in.
    pub dequantized: usize,
    /// Tensors the target *quantised* on the way out (the `BnB-NF4` encoder).
    pub quantized: usize,
    /// Tensors written in their incoming dtype.
    pub passthrough: usize,
}

// ---------------------------------------------------------------------------
// The hub
// ---------------------------------------------------------------------------

/// One tensor normalised into the hub: owned bytes in `dtype`, shape row-major.
#[derive(Debug, Clone)]
pub(crate) struct HubTensor {
    /// Tensor name as it will appear in the output.
    pub(crate) name: String,
    /// Row-major (safetensors / `NumPy` order) dimensions.
    pub(crate) shape: Vec<usize>,
    /// Element type of `data`.
    pub(crate) dtype: Dtype,
    /// Raw little-endian bytes, `product(shape) × dtype.byte_size()` long.
    pub(crate) data: Vec<u8>,
}

/// An input normalised for the writers: the tensors plus any source metadata
/// that survives the conversion.
#[derive(Debug, Default)]
pub(crate) struct Hub {
    /// The normalised tensors, in output order.
    pub(crate) tensors: Vec<HubTensor>,
    /// safetensors `__metadata__`, carried only when the *source* was
    /// safetensors (every other reader writes `None`, matching v0.6.0).
    pub(crate) st_metadata: Option<HashMap<String, String>>,
    /// Tensors dequantised while reading.
    pub(crate) dequantized: usize,
    /// `GGUF` key/value metadata carried from a `GGUF` source so a
    /// dequantise-in-place `gguf → gguf` keeps the architecture / tokenizer KV
    /// that makes the output loadable — a re-emitted file with no KV is a bare
    /// tensor container. Empty for every non-`GGUF` source; caller-supplied KV
    /// is merged over it by the writer.
    #[cfg(feature = "gguf")]
    pub(crate) gguf_metadata: HashMap<String, crate::GgufMetadataValue>,
}

// ---------------------------------------------------------------------------
// Format detection (shared by the CLI's parse / inspect / remember paths)
// ---------------------------------------------------------------------------

/// A detected input format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Format {
    /// `.safetensors` (or an unrecognised extension that is not `GGUF`).
    Safetensors,
    /// `PyTorch` `.pth` / `.pt` (or a `.bin` with ZIP magic).
    #[cfg(feature = "pth")]
    Pth,
    /// `NumPy` `.npz`.
    #[cfg(feature = "npz")]
    Npz,
    /// `GGUF` (by extension or magic).
    #[cfg(feature = "gguf")]
    Gguf,
}

/// Builds an `Unsupported` error naming the Cargo feature that would enable a
/// detected-but-disabled format.
///
/// Compiled only when at least one of `pth` / `npz` / `gguf` is absent — with all
/// three enabled it has no callers.
#[cfg(not(all(feature = "pth", feature = "npz", feature = "gguf")))]
fn missing_feature_err(format_name: &str, kind: &str, feature_flag: &str) -> AnamnesisError {
    AnamnesisError::Unsupported {
        format: format_name.into(),
        detail: format!(
            "input is {kind} but the `{feature_flag}` Cargo feature is not enabled in this \
             build — rebuild with `cargo install anamnesis --features cli,{feature_flag}` \
             (or `cargo build --features cli,{feature_flag}`) to add support"
        ),
    }
}

/// Returns `true` if the file starts with `magic`. Reads only 4 bytes — does
/// not load the file into memory.
fn has_magic(path: &Path, magic: [u8; 4]) -> bool {
    let mut buf = [0u8; 4];
    std::fs::File::open(path)
        .and_then(|mut f| {
            use std::io::Read as _;
            f.read_exact(&mut buf)
        })
        .is_ok_and(|()| buf == magic)
}

/// Detects the model format from file extension, falling back to magic bytes.
///
/// `.safetensors` → safetensors. `.pth` / `.pt` → `PyTorch`. `.npz` → NPZ.
/// `.gguf` → `GGUF`. `.bin` → ZIP magic (`PyTorch`) then `GGUF` magic. Any other
/// extension → `GGUF` magic, else safetensors.
///
/// # Errors
///
/// Returns [`AnamnesisError::Unsupported`] when the input matches a format whose
/// Cargo feature is disabled in this build, rather than misrouting it to the
/// safetensors parser.
// `clippy::unnecessary_wraps`: with all of `pth`/`npz`/`gguf` enabled every arm
// is `Ok(_)`; other feature combinations make the wrap load-bearing.
#[allow(clippy::unnecessary_wraps)]
pub(crate) fn detect_format(path: &Path) -> crate::Result<Format> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    match ext.as_str() {
        "safetensors" => Ok(Format::Safetensors),
        "pth" | "pt" => {
            #[cfg(feature = "pth")]
            {
                Ok(Format::Pth)
            }
            #[cfg(not(feature = "pth"))]
            {
                Err(missing_feature_err("PyTorch", "a .pth/.pt file", "pth"))
            }
        }
        "npz" => {
            #[cfg(feature = "npz")]
            {
                Ok(Format::Npz)
            }
            #[cfg(not(feature = "npz"))]
            {
                Err(missing_feature_err("NumPy NPZ", "a .npz file", "npz"))
            }
        }
        "gguf" => {
            #[cfg(feature = "gguf")]
            {
                Ok(Format::Gguf)
            }
            #[cfg(not(feature = "gguf"))]
            {
                Err(missing_feature_err("GGUF", "a .gguf file", "gguf"))
            }
        }
        "bin" => {
            if has_magic(path, *b"PK\x03\x04") {
                #[cfg(feature = "pth")]
                {
                    return Ok(Format::Pth);
                }
                #[cfg(not(feature = "pth"))]
                {
                    return Err(missing_feature_err(
                        "PyTorch",
                        "a .bin file with ZIP magic (PyTorch pickle archive)",
                        "pth",
                    ));
                }
            }
            if has_magic(path, *b"GGUF") {
                #[cfg(feature = "gguf")]
                {
                    return Ok(Format::Gguf);
                }
                #[cfg(not(feature = "gguf"))]
                {
                    return Err(missing_feature_err(
                        "GGUF",
                        "a .bin file with GGUF magic",
                        "gguf",
                    ));
                }
            }
            Ok(Format::Safetensors)
        }
        _ => {
            if has_magic(path, *b"GGUF") {
                #[cfg(feature = "gguf")]
                {
                    return Ok(Format::Gguf);
                }
                #[cfg(not(feature = "gguf"))]
                {
                    return Err(missing_feature_err(
                        "GGUF",
                        "a file whose first four bytes are the GGUF magic",
                        "gguf",
                    ));
                }
            }
            Ok(Format::Safetensors)
        }
    }
}

// ---------------------------------------------------------------------------
// Output-path derivation
// ---------------------------------------------------------------------------

/// Quantisation suffixes stripped from an input stem when deriving an output
/// path. Case-sensitive and ordered longest-first so `-GPTQ-Int4` wins over
/// `-gptq`.
const QUANT_SUFFIXES: &[&str] = &[
    "-GPTQ-Int4",
    "-GPTQ-Int8",
    "-gptq-int4",
    "-gptq-int8",
    "-gptq4",
    "-gptq8",
    "-GPTQ",
    "-gptq",
    "_gptq",
    "-AWQ",
    "-awq",
    "_awq",
    "-bnb-4bit",
    "-bnb-int8",
    "-bnb",
    "_bnb",
    "-4bit",
    "-int4",
    "-int8",
    "-fp8",
    "_fp8",
    "-FP8",
];

/// Strips a known quantisation suffix from a file stem, if present.
/// `model-GPTQ-Int4` → `model`; `weights` → `weights`.
///
/// Shared with the CLI's `remember` output-path derivation so both stay on one
/// suffix table.
#[must_use]
pub(crate) fn strip_quant_suffix(stem: &str) -> &str {
    QUANT_SUFFIXES
        .iter()
        .find_map(|qs| stem.strip_suffix(qs))
        .unwrap_or(stem)
}

/// Derives an output path from `input` and `target`: strips a known
/// quantisation suffix from the stem, then appends `-{suffix}.{extension}`.
///
/// `model-fp8.safetensors` + `Safetensors` → `model-bf16.safetensors`;
/// `weights.npz` + `Gguf` → `weights-gguf.gguf`.
#[must_use]
pub fn derive_output_path(input: &Path, target: ConvertTarget) -> PathBuf {
    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");
    let new_name = format!(
        "{}-{}.{}",
        strip_quant_suffix(stem),
        target.suffix(),
        target.extension()
    );
    input
        .parent()
        .map_or_else(|| PathBuf::from(&new_name), |p| p.join(&new_name))
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// Converts `input` to `target`, writing the result to `output`.
///
/// Every supported `(input format × target)` pair routes through the in-memory
/// hub: the input is parsed and normalised (quantised tensors dequantised to
/// `BF16`, scalar tensors kept in their original dtype), then written to the
/// target. Format detection is automatic: by file extension, falling back to
/// magic bytes for `.bin` and unrecognised extensions.
///
/// # Errors
///
/// Returns [`AnamnesisError::Unsupported`] if the input format's Cargo feature
/// is disabled, if the target is not reachable from this input, or if a tensor's
/// dtype has no counterpart in the target format.
/// Returns [`AnamnesisError::LimitExceeded`] if the input breaches
/// `options.limits` or a permanent per-format cap.
/// Returns [`AnamnesisError::Parse`] on a malformed input, and
/// [`AnamnesisError::Io`] if the input cannot be read or the output written.
///
/// # Memory
///
/// Peak heap is `O(model)`: the whole hub is materialised before the writer
/// runs, plus the target writer's buffer (near-zero for the streaming
/// safetensors / `GGUF` writers; the packed NF4 output for the `bnb-nf4`
/// target). See the module docs.
pub fn convert(
    input: &Path,
    target: ConvertTarget,
    output: &Path,
    options: &ConvertOptions,
) -> crate::Result<ConvertStats> {
    let hub = read_hub(input, options)?;
    write_hub(&hub, target, output, options)
}

/// Parses `input` into the hub, dispatching on the detected format.
fn read_hub(input: &Path, options: &ConvertOptions) -> crate::Result<Hub> {
    let threads = crate::model::resolve_thread_budget(options.threads);
    match detect_format(input)? {
        Format::Safetensors => read_safetensors(input, &options.limits, threads),
        #[cfg(feature = "pth")]
        Format::Pth => read_pth(input, &options.limits),
        #[cfg(feature = "npz")]
        Format::Npz => read_npz(input, &options.limits),
        #[cfg(feature = "gguf")]
        Format::Gguf => read_gguf(input, &options.limits, threads),
    }
}

/// Writes the hub to `target`.
///
/// `options` is read only by the `GGUF` writer (the caller-supplied KV merged
/// over any inherited source KV); with the `gguf` feature off no arm consumes
/// it, so the unused-variable warning is suppressed for that build rather than
/// renaming the parameter and losing the signature's intent.
#[cfg_attr(not(feature = "gguf"), allow(unused_variables))]
fn write_hub(
    hub: &Hub,
    target: ConvertTarget,
    output: &Path,
    options: &ConvertOptions,
) -> crate::Result<ConvertStats> {
    match target {
        ConvertTarget::Safetensors => write_safetensors(hub, output),
        #[cfg(feature = "gguf")]
        ConvertTarget::Gguf => write_gguf_target(hub, output, options),
        #[cfg(not(feature = "gguf"))]
        ConvertTarget::Gguf => Err(AnamnesisError::Unsupported {
            format: "gguf".into(),
            detail: "GGUF emit requires the `gguf` Cargo feature; rebuild with \
                     `--features cli,gguf`"
                .into(),
        }),
        #[cfg(feature = "bnb")]
        ConvertTarget::BnbNf4 => write_bnb_nf4_target(hub, output),
        #[cfg(not(feature = "bnb"))]
        ConvertTarget::BnbNf4 => Err(AnamnesisError::Unsupported {
            format: "bnb-nf4".into(),
            detail: "BnB-NF4 encode requires the `bnb` Cargo feature; rebuild with \
                     `--features cli,bnb`"
                .into(),
        }),
    }
}

// ---------------------------------------------------------------------------
// Readers — format → hub
// ---------------------------------------------------------------------------

/// Reads a safetensors file into the hub, dequantising quantised tensors to
/// `BF16` and passing scalar tensors through in their original dtype. Carries
/// the source `__metadata__`.
fn read_safetensors(path: &Path, limits: &ParseLimits, threads: usize) -> crate::Result<Hub> {
    let model = crate::parse_with_limits(path, limits)?;
    let (tensors, dequantized) = model.hub_tensors(threads)?;
    Ok(Hub {
        tensors,
        st_metadata: model.header.metadata.clone(),
        dequantized,
        #[cfg(feature = "gguf")]
        gguf_metadata: HashMap::new(),
    })
}

/// Reads an `NPZ` archive into the hub. Every `NPZ` tensor is full precision, so
/// nothing is dequantised; dtypes are preserved. Tensors are emitted in sorted
/// name order for a deterministic output.
#[cfg(feature = "npz")]
fn read_npz(path: &Path, limits: &ParseLimits) -> crate::Result<Hub> {
    let mut map = crate::parse_npz_with_limits(path, limits)?;
    // Sort the (cheap) name keys, then move each tensor out of the owned map —
    // draining avoids cloning every tensor's bytes (a full-model copy).
    let mut names: Vec<String> = map.keys().cloned().collect();
    names.sort();

    let mut tensors = Vec::with_capacity(names.len());
    for name in names {
        let t = map.remove(&name).ok_or_else(|| AnamnesisError::Parse {
            reason: format!("NPZ tensor `{name}` vanished mid-iteration"),
        })?;
        tensors.push(HubTensor {
            name,
            shape: t.shape,
            dtype: npz_dtype_to_hub(t.dtype),
            data: t.data,
        });
    }
    Ok(Hub {
        tensors,
        st_metadata: None,
        dequantized: 0,
        #[cfg(feature = "gguf")]
        gguf_metadata: HashMap::new(),
    })
}

/// Reads a `PyTorch` `.pth` into the hub. Tensor data is already full precision;
/// dtypes are preserved.
#[cfg(feature = "pth")]
fn read_pth(path: &Path, limits: &ParseLimits) -> crate::Result<Hub> {
    let parsed = crate::parse_pth_with_limits(path, limits)?;
    let pth_tensors = parsed.tensors()?;

    let mut tensors = Vec::with_capacity(pth_tensors.len());
    for t in pth_tensors {
        tensors.push(HubTensor {
            name: t.name,
            shape: t.shape,
            dtype: pth_dtype_to_hub(t.dtype)?,
            // BORROW: `into_owned()` copies the (possibly mmap-borrowed) bytes so
            // the hub outlives the `ParsedPth`.
            data: t.data.into_owned(),
        });
    }
    Ok(Hub {
        tensors,
        st_metadata: None,
        dequantized: 0,
        #[cfg(feature = "gguf")]
        gguf_metadata: HashMap::new(),
    })
}

/// Reads a `GGUF` file into the hub, dequantising block-quantised tensors to
/// `BF16` and passing scalar tensors through. `GGUF` shapes are
/// most-significant-first, so they are reversed into the hub's row-major order.
///
/// `threads` is the resolved worker budget: the per-tensor work is dispatched
/// through `parallel::map_indexed`, which returns the hub tensors in `GGUF` file
/// order for **any** thread count, so the written output is byte-identical
/// whatever the budget. Both branches parallelise usefully — the quantised
/// branch because dequantisation is the compute, the scalar branch because the
/// `into_owned()` copy is bandwidth-bound — so an all-`F32` `GGUF` gains too.
#[cfg(feature = "gguf")]
fn read_gguf(path: &Path, limits: &ParseLimits, threads: usize) -> crate::Result<Hub> {
    let parsed = crate::parse_gguf_with_limits(path, limits)?;

    // Materialise the tensor views up front so the dispatch has an indexable
    // slice. The views borrow the mapped bytes (name, shape and data are all
    // references), so this costs one small struct per tensor and copies no
    // tensor data. Tensors whose dtype has no tabulated byte size are absent
    // here — `ParsedGguf::tensors` filters them — which preserves the skip this
    // reader has always had.
    let views: Vec<crate::GgufTensor<'_>> = parsed.tensors().collect();
    let work_bytes: u64 = views.iter().fold(0u64, |acc, view| {
        acc.saturating_add(u64::try_from(view.data.len()).unwrap_or(u64::MAX))
    });

    // `ParsedGguf` is `Sync` (asserted in `tests/parallel_contract.rs`), and the
    // closure below reads only the shared-immutable view it is handed, writing
    // solely into the `HubTensor` it returns.
    let tensors: Vec<HubTensor> = crate::parallel::map_indexed(
        &views,
        threads,
        work_bytes,
        |_, tensor| {
            let mut shape: Vec<usize> = tensor.shape.to_vec();
            shape.reverse();

            if tensor.dtype.is_quantized() {
                let n_elements = tensor
                    .shape
                    .iter()
                    .try_fold(1usize, |acc, &d| acc.checked_mul(d))
                    .ok_or_else(|| AnamnesisError::Parse {
                        reason: format!(
                            "GGUF tensor `{}` shape {:?} element count overflows usize",
                            tensor.name, tensor.shape
                        ),
                    })?;
                let bf16 = crate::dequantize_gguf_to_bf16(&tensor.data, tensor.dtype, n_elements)?;
                Ok(HubTensor {
                    name: tensor.name.to_owned(),
                    shape,
                    dtype: Dtype::BF16,
                    data: bf16,
                })
            } else {
                Ok(HubTensor {
                    name: tensor.name.to_owned(),
                    shape,
                    dtype: gguf_type_to_hub(tensor.dtype)?,
                    // BORROW: `to_vec()` copies the mmap-borrowed slice so the
                    // hub outlives the `ParsedGguf`.
                    data: tensor.data.to_vec(),
                })
            }
        },
        |_| {},
    )?;

    // Counted from the inputs rather than incremented during the dispatch, so no
    // counter is shared across workers. Derived from the source dtype, not the
    // hub dtype: a `GGUF` may carry a *scalar* `BF16` tensor, which passes
    // through untouched and must not be reported as dequantised.
    let dequantized = views.iter().filter(|v| v.dtype.is_quantized()).count();

    Ok(Hub {
        tensors,
        st_metadata: None,
        dequantized,
        // Preserve the source KV so a dequantise-in-place `gguf -> gguf` stays
        // loadable; caller-supplied KV is merged over it by the writer.
        gguf_metadata: parsed.metadata().clone(),
    })
}

// ---------------------------------------------------------------------------
// Writers — hub → format
// ---------------------------------------------------------------------------

/// Writes the hub as a safetensors file, each tensor in its hub dtype.
fn write_safetensors(hub: &Hub, output: &Path) -> crate::Result<ConvertStats> {
    let mut views: Vec<(String, safetensors::tensor::TensorView<'_>)> =
        Vec::with_capacity(hub.tensors.len());
    for t in &hub.tensors {
        let st_dtype = t.dtype.to_safetensors_dtype()?;
        let view = safetensors::tensor::TensorView::new(st_dtype, t.shape.clone(), &t.data)
            .map_err(|e| AnamnesisError::Parse {
                reason: format!("failed to create TensorView for `{}`: {e}", t.name),
            })?;
        views.push((t.name.clone(), view));
    }

    safetensors::tensor::serialize_to_file(views, hub.st_metadata.clone(), output).map_err(
        // EXHAUSTIVE: `SafeTensorError` is a foreign type that may gain variants.
        #[allow(clippy::wildcard_enum_match_arm)]
        |e| match e {
            safetensors::SafeTensorError::IoError(io_err) => AnamnesisError::Io(io_err),
            other => AnamnesisError::Parse {
                reason: format!("failed to write safetensors file: {other}"),
            },
        },
    )?;

    Ok(ConvertStats {
        tensors: hub.tensors.len(),
        dequantized: hub.dequantized,
        quantized: 0,
        // A dequantised tensor did *not* go out in its incoming dtype, so the
        // two counts partition the written set rather than overlapping.
        passthrough: hub.tensors.len().saturating_sub(hub.dequantized),
    })
}

/// Writes the hub as an unquantised `GGUF` file, reversing shapes back to
/// most-significant-first.
#[cfg(feature = "gguf")]
fn write_gguf_target(
    hub: &Hub,
    output: &Path,
    options: &ConvertOptions,
) -> crate::Result<ConvertStats> {
    use crate::{GgufWriteTensor, write_gguf};

    let mut owned: Vec<(String, crate::GgufType, Vec<usize>, &[u8])> =
        Vec::with_capacity(hub.tensors.len());
    for t in &hub.tensors {
        let gguf_dtype = hub_dtype_to_gguf(t.dtype)?;
        let mut msb_first = t.shape.clone();
        msb_first.reverse();
        owned.push((t.name.clone(), gguf_dtype, msb_first, t.data.as_slice()));
    }

    let tensors: Vec<GgufWriteTensor<'_>> = owned
        .iter()
        .map(|(name, dtype, shape, data)| GgufWriteTensor {
            name: name.as_str(),
            shape: shape.as_slice(),
            dtype: *dtype,
            data,
        })
        .collect();

    // Source KV first, caller KV merged over it: an explicit `--gguf-kv` /
    // `--gguf-metadata` entry overrides the inherited value for the same key.
    // When there is no caller KV — the common `gguf → gguf` case — the inherited
    // map (which can carry a multi-thousand-entry tokenizer array) is borrowed
    // straight through instead of deep-cloned.
    let metadata: Cow<'_, HashMap<String, crate::GgufMetadataValue>> =
        if options.gguf_metadata.is_empty() {
            Cow::Borrowed(&hub.gguf_metadata)
        } else {
            let mut merged = hub.gguf_metadata.clone();
            merged.extend(
                options
                    .gguf_metadata
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone())),
            );
            Cow::Owned(merged)
        };
    write_gguf(output, &tensors, &metadata)?;

    Ok(ConvertStats {
        tensors: tensors.len(),
        dequantized: hub.dequantized,
        quantized: 0,
        // As above: dequantised tensors are not passthrough.
        passthrough: tensors.len().saturating_sub(hub.dequantized),
    })
}

/// Writes the hub as `BitsAndBytes`-NF4 safetensors. The encoder's input
/// contract is `BF16`, so float tensors are converted on the way in; 2-D weights
/// are encoded to NF4 and everything else passes through as `BF16`.
#[cfg(feature = "bnb")]
fn write_bnb_nf4_target(hub: &Hub, output: &Path) -> crate::Result<ConvertStats> {
    use crate::{BnbWriteInput, classify_inputs, write_bnb_nf4_safetensors};

    let mut owned: Vec<(String, Vec<usize>, Cow<'_, [u8]>)> = Vec::with_capacity(hub.tensors.len());
    for t in &hub.tensors {
        // BF16 tensors are borrowed straight from the hub (no copy); only
        // F32/F16 allocate a converted buffer.
        let bf16 = to_bf16_bytes(&t.data, t.dtype, &t.name)?;
        owned.push((t.name.clone(), t.shape.clone(), bf16));
    }

    let inputs: Vec<BnbWriteInput<'_>> = owned
        .iter()
        .map(|(name, shape, bf16)| BnbWriteInput {
            name: name.as_str(),
            shape: shape.as_slice(),
            bf16_data: bf16.as_ref(),
        })
        .collect();

    let stats = classify_inputs(&inputs);
    write_bnb_nf4_safetensors(&inputs, output)?;

    Ok(ConvertStats {
        tensors: inputs.len(),
        dequantized: hub.dequantized,
        quantized: stats.quantized,
        passthrough: stats.passthrough,
    })
}

// ---------------------------------------------------------------------------
// Caller-supplied GGUF metadata (`--gguf-metadata` / `--gguf-kv`)
// ---------------------------------------------------------------------------

/// Parses one `--gguf-kv key=value` argument into a `String`-valued entry.
///
/// The value is **always** a [`GgufMetadataValue::String`](crate::GgufMetadataValue::String) —
/// unambiguous for the one-off case (`general.architecture=llama`). Keys that
/// need a specific width or an array go through
/// [`parse_gguf_metadata_json`] instead.
///
/// Splits on the **first** `=`, so a value may itself contain `=`.
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if the argument has no `=` or an empty key.
#[cfg(feature = "gguf")]
pub fn parse_gguf_kv_arg(arg: &str) -> crate::Result<(String, crate::GgufMetadataValue)> {
    let (key, value) = arg.split_once('=').ok_or_else(|| AnamnesisError::Parse {
        reason: format!("--gguf-kv `{arg}`: expected `key=value`"),
    })?;
    if key.is_empty() {
        return Err(AnamnesisError::Parse {
            reason: format!("--gguf-kv `{arg}`: empty key"),
        });
    }
    Ok((
        key.to_owned(),
        crate::GgufMetadataValue::String(value.to_owned()),
    ))
}

/// Parses a `--gguf-metadata` JSON document into a `GGUF` key/value table.
///
/// The document is a JSON object. Each value is either a **plain** JSON value,
/// whose `GGUF` type is inferred, or an **explicit** `{"type": …, "value": …}`
/// object when the exact width matters:
///
/// | JSON | Inferred `GGUF` type |
/// |---|---|
/// | `"llama"` | `String` |
/// | `true` | `Bool` |
/// | `32` (fits `u32`, non-negative) | `U32` |
/// | `-5` / a larger integer | `I64` / `U64` |
/// | `1e-5` | `F32` |
/// | `["a", "b"]` | `Array<String>` (typed from the first element) |
///
/// Explicit forms — `{"type": "i32", "value": 3}` for a scalar (`u8` `i8` `u16`
/// `i16` `u32` `i32` `u64` `i64` `f32` `f64` `bool` `string`), and
/// `{"type": "array", "item_type": "i32", "value": [1, 2]}` for an array.
///
/// The escape hatch exists because inference cannot be right for every key:
/// `tokenizer.ggml.token_type` is an `Array<I32>` in the `llama.cpp` convention,
/// but a JSON array of non-negative integers infers `Array<U32>`. anamnesis
/// attaches **no meaning to key names** — special-casing that key would import
/// exactly the model knowledge this layer refuses — so the caller states the
/// type instead.
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if the document is not valid JSON, is not a
/// top-level object, contains a `null`, contains an empty array (no element to
/// infer from), names an unknown type tag, or holds a number outside the range
/// of its declared type.
///
/// # Memory
///
/// Allocates the parsed table; tokenizer arrays can run to tens of thousands of
/// entries, so peak heap is proportional to the document.
#[cfg(feature = "gguf")]
pub fn parse_gguf_metadata_json(
    json: &str,
) -> crate::Result<HashMap<String, crate::GgufMetadataValue>> {
    let parsed: serde_json::Value =
        serde_json::from_str(json).map_err(|e| AnamnesisError::Parse {
            reason: format!("--gguf-metadata: invalid JSON: {e}"),
        })?;
    let obj = parsed.as_object().ok_or_else(|| AnamnesisError::Parse {
        reason: format!(
            "--gguf-metadata: expected a top-level JSON object, found {}",
            json_type_name(&parsed)
        ),
    })?;

    let mut out = HashMap::with_capacity(obj.len());
    for (key, value) in obj {
        out.insert(key.clone(), json_to_metadata_value(key, value)?);
    }
    Ok(out)
}

/// Names a JSON value's kind for error messages.
#[cfg(feature = "gguf")]
const fn json_type_name(value: &serde_json::Value) -> &'static str {
    match *value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "a boolean",
        serde_json::Value::Number(_) => "a number",
        serde_json::Value::String(_) => "a string",
        serde_json::Value::Array(_) => "an array",
        serde_json::Value::Object(_) => "an object",
    }
}

/// Reads a JSON integer as `i128` so every `GGUF` width can be range-checked
/// against one representation.
#[cfg(feature = "gguf")]
fn json_as_int(key: &str, value: &serde_json::Value) -> crate::Result<i128> {
    let number = value.as_number().ok_or_else(|| AnamnesisError::Parse {
        reason: format!(
            "--gguf-metadata `{key}`: expected an integer, found {}",
            json_type_name(value)
        ),
    })?;
    if let Some(u) = number.as_u64() {
        return Ok(i128::from(u));
    }
    if let Some(i) = number.as_i64() {
        return Ok(i128::from(i));
    }
    Err(AnamnesisError::Parse {
        reason: format!("--gguf-metadata `{key}`: expected an integer, found a float"),
    })
}

/// Reads a JSON number as `f64`.
#[cfg(feature = "gguf")]
fn json_as_float(key: &str, value: &serde_json::Value) -> crate::Result<f64> {
    value.as_f64().ok_or_else(|| AnamnesisError::Parse {
        reason: format!(
            "--gguf-metadata `{key}`: expected a number, found {}",
            json_type_name(value)
        ),
    })
}

/// Reads a JSON bool.
#[cfg(feature = "gguf")]
fn json_as_bool(key: &str, value: &serde_json::Value) -> crate::Result<bool> {
    value.as_bool().ok_or_else(|| AnamnesisError::Parse {
        reason: format!(
            "--gguf-metadata `{key}`: expected a boolean, found {}",
            json_type_name(value)
        ),
    })
}

/// Reads a JSON string.
#[cfg(feature = "gguf")]
fn json_as_string(key: &str, value: &serde_json::Value) -> crate::Result<String> {
    value
        .as_str()
        .map(ToOwned::to_owned)
        .ok_or_else(|| AnamnesisError::Parse {
            reason: format!(
                "--gguf-metadata `{key}`: expected a string, found {}",
                json_type_name(value)
            ),
        })
}

/// Narrows an `i128` to a `GGUF` integer width, reporting the key on overflow.
#[cfg(feature = "gguf")]
fn narrow_int<T: TryFrom<i128>>(key: &str, raw: i128, type_name: &str) -> crate::Result<T> {
    T::try_from(raw).map_err(|_| AnamnesisError::Parse {
        reason: format!("--gguf-metadata `{key}`: value {raw} is out of range for {type_name}"),
    })
}

/// Builds a scalar [`GgufMetadataValue`](crate::GgufMetadataValue) of the named
/// type from a JSON value.
#[cfg(feature = "gguf")]
fn scalar_of_type(
    key: &str,
    type_tag: &str,
    value: &serde_json::Value,
) -> crate::Result<crate::GgufMetadataValue> {
    use crate::GgufMetadataValue as V;
    Ok(match type_tag {
        "u8" => V::U8(narrow_int(key, json_as_int(key, value)?, "u8")?),
        "i8" => V::I8(narrow_int(key, json_as_int(key, value)?, "i8")?),
        "u16" => V::U16(narrow_int(key, json_as_int(key, value)?, "u16")?),
        "i16" => V::I16(narrow_int(key, json_as_int(key, value)?, "i16")?),
        "u32" => V::U32(narrow_int(key, json_as_int(key, value)?, "u32")?),
        "i32" => V::I32(narrow_int(key, json_as_int(key, value)?, "i32")?),
        "u64" => V::U64(narrow_int(key, json_as_int(key, value)?, "u64")?),
        "i64" => V::I64(narrow_int(key, json_as_int(key, value)?, "i64")?),
        "f32" => {
            // CAST: f64 → f32 is the documented narrowing for the `f32` tag; the
            // caller asked for a 32-bit float.
            #[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
            let narrowed = json_as_float(key, value)? as f32;
            V::F32(narrowed)
        }
        "f64" => V::F64(json_as_float(key, value)?),
        "bool" => V::Bool(json_as_bool(key, value)?),
        "string" => V::String(json_as_string(key, value)?),
        other => {
            return Err(AnamnesisError::Parse {
                reason: format!(
                    "--gguf-metadata `{key}`: unknown type `{other}` \
                     (expected u8/i8/u16/i16/u32/i32/u64/i64/f32/f64/bool/string/array)"
                ),
            });
        }
    })
}

/// Builds a typed [`GgufMetadataArray`](crate::GgufMetadataArray) from JSON
/// elements, all narrowed to `item_type`.
#[cfg(feature = "gguf")]
fn array_of_type(
    key: &str,
    item_type: &str,
    items: &[serde_json::Value],
) -> crate::Result<crate::GgufMetadataArray> {
    use crate::GgufMetadataArray as A;
    /// Maps each element through `f`, propagating the first error.
    fn collect<T, F: Fn(&serde_json::Value) -> crate::Result<T>>(
        items: &[serde_json::Value],
        f: F,
    ) -> crate::Result<Vec<T>> {
        items.iter().map(f).collect()
    }

    Ok(match item_type {
        "u8" => A::U8(collect(items, |v| {
            narrow_int(key, json_as_int(key, v)?, "u8")
        })?),
        "i8" => A::I8(collect(items, |v| {
            narrow_int(key, json_as_int(key, v)?, "i8")
        })?),
        "u16" => A::U16(collect(items, |v| {
            narrow_int(key, json_as_int(key, v)?, "u16")
        })?),
        "i16" => A::I16(collect(items, |v| {
            narrow_int(key, json_as_int(key, v)?, "i16")
        })?),
        "u32" => A::U32(collect(items, |v| {
            narrow_int(key, json_as_int(key, v)?, "u32")
        })?),
        "i32" => A::I32(collect(items, |v| {
            narrow_int(key, json_as_int(key, v)?, "i32")
        })?),
        "u64" => A::U64(collect(items, |v| {
            narrow_int(key, json_as_int(key, v)?, "u64")
        })?),
        "i64" => A::I64(collect(items, |v| {
            narrow_int(key, json_as_int(key, v)?, "i64")
        })?),
        "f32" => A::F32(collect(items, |v| {
            // CAST: f64 → f32, the documented narrowing for the `f32` tag.
            #[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
            let narrowed = json_as_float(key, v)? as f32;
            Ok(narrowed)
        })?),
        "f64" => A::F64(collect(items, |v| json_as_float(key, v))?),
        "bool" => A::Bool(collect(items, |v| json_as_bool(key, v))?),
        "string" => A::String(collect(items, |v| json_as_string(key, v))?),
        other => {
            return Err(AnamnesisError::Parse {
                reason: format!(
                    "--gguf-metadata `{key}`: unknown array item type `{other}` \
                     (expected u8/i8/u16/i16/u32/i32/u64/i64/f32/f64/bool/string)"
                ),
            });
        }
    })
}

/// Infers the type tag a plain JSON value maps to.
#[cfg(feature = "gguf")]
fn infer_type_tag(key: &str, value: &serde_json::Value) -> crate::Result<&'static str> {
    match *value {
        serde_json::Value::Bool(_) => Ok("bool"),
        serde_json::Value::String(_) => Ok("string"),
        serde_json::Value::Number(ref n) => {
            if n.is_f64() && !n.is_u64() && !n.is_i64() {
                return Ok("f32");
            }
            let raw = json_as_int(key, value)?;
            if (0..=i128::from(u32::MAX)).contains(&raw) {
                Ok("u32")
            } else if i64::try_from(raw).is_ok() {
                Ok("i64")
            } else {
                Ok("u64")
            }
        }
        serde_json::Value::Null | serde_json::Value::Array(_) | serde_json::Value::Object(_) => {
            Err(AnamnesisError::Parse {
                reason: format!(
                    "--gguf-metadata `{key}`: cannot infer a scalar type from {}",
                    json_type_name(value)
                ),
            })
        }
    }
}

/// Converts one JSON value into a [`GgufMetadataValue`](crate::GgufMetadataValue),
/// honouring the explicit `{"type", "value"}` form when present.
#[cfg(feature = "gguf")]
fn json_to_metadata_value(
    key: &str,
    value: &serde_json::Value,
) -> crate::Result<crate::GgufMetadataValue> {
    // Explicit form: an object carrying a `type` tag.
    if let Some(obj) = value.as_object() {
        let type_tag = obj
            .get("type")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!(
                    "--gguf-metadata `{key}`: an object value must carry a string `type` field \
                     (explicit form: {{\"type\": \"u32\", \"value\": 32}})"
                ),
            })?;
        let inner = obj.get("value").ok_or_else(|| AnamnesisError::Parse {
            reason: format!("--gguf-metadata `{key}`: explicit form is missing `value`"),
        })?;

        if type_tag == "array" {
            let item_type = obj
                .get("item_type")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| AnamnesisError::Parse {
                    reason: format!(
                        "--gguf-metadata `{key}`: an `array` needs a string `item_type`"
                    ),
                })?;
            let items = inner.as_array().ok_or_else(|| AnamnesisError::Parse {
                reason: format!(
                    "--gguf-metadata `{key}`: `array` expects a JSON array `value`, found {}",
                    json_type_name(inner)
                ),
            })?;
            return Ok(crate::GgufMetadataValue::Array(Box::new(array_of_type(
                key, item_type, items,
            )?)));
        }
        return scalar_of_type(key, type_tag, inner);
    }

    // Plain array: homogeneous, typed from the first element.
    if let Some(items) = value.as_array() {
        let first = items.first().ok_or_else(|| AnamnesisError::Parse {
            reason: format!(
                "--gguf-metadata `{key}`: cannot infer an item type from an empty array \
                 (use the explicit form: {{\"type\": \"array\", \"item_type\": \"i32\", \
                 \"value\": []}})"
            ),
        })?;
        let item_type = infer_type_tag(key, first)?;
        return Ok(crate::GgufMetadataValue::Array(Box::new(array_of_type(
            key, item_type, items,
        )?)));
    }

    // Plain scalar.
    let type_tag = infer_type_tag(key, value)?;
    scalar_of_type(key, type_tag, value)
}

// ---------------------------------------------------------------------------
// Dtype mapping
// ---------------------------------------------------------------------------

/// Maps an [`NpzDtype`](crate::NpzDtype) to the hub dtype. Total — every `NPZ`
/// dtype has a safetensors counterpart.
#[cfg(feature = "npz")]
const fn npz_dtype_to_hub(dtype: crate::NpzDtype) -> Dtype {
    use crate::NpzDtype;
    match dtype {
        NpzDtype::Bool => Dtype::Bool,
        NpzDtype::U8 => Dtype::U8,
        NpzDtype::I8 => Dtype::I8,
        NpzDtype::U16 => Dtype::U16,
        NpzDtype::I16 => Dtype::I16,
        NpzDtype::U32 => Dtype::U32,
        NpzDtype::I32 => Dtype::I32,
        NpzDtype::U64 => Dtype::U64,
        NpzDtype::I64 => Dtype::I64,
        NpzDtype::F16 => Dtype::F16,
        NpzDtype::BF16 => Dtype::BF16,
        NpzDtype::F32 => Dtype::F32,
        NpzDtype::F64 => Dtype::F64,
    }
}

/// Maps a [`PthDtype`](crate::PthDtype) to the hub dtype. Total — every `.pth`
/// dtype the parser accepts has a safetensors counterpart.
///
/// # Errors
///
/// Currently infallible; returns `Result` so a future `PthDtype` without a
/// counterpart can be rejected without a breaking change.
#[cfg(feature = "pth")]
// `clippy::unnecessary_wraps`: every arm is `Ok(_)` today; the `Result` is kept so
// a future `PthDtype` without a safetensors counterpart can be rejected without a
// breaking signature change.
#[allow(clippy::unnecessary_wraps)]
const fn pth_dtype_to_hub(dtype: crate::PthDtype) -> crate::Result<Dtype> {
    use crate::PthDtype;
    Ok(match dtype {
        PthDtype::F16 => Dtype::F16,
        PthDtype::BF16 => Dtype::BF16,
        PthDtype::F32 => Dtype::F32,
        PthDtype::F64 => Dtype::F64,
        PthDtype::U8 => Dtype::U8,
        PthDtype::I8 => Dtype::I8,
        PthDtype::I16 => Dtype::I16,
        PthDtype::I32 => Dtype::I32,
        PthDtype::I64 => Dtype::I64,
        PthDtype::Bool => Dtype::Bool,
    })
}

/// Maps a **scalar** [`GgufType`](crate::GgufType) to the hub dtype.
///
/// # Errors
///
/// Returns [`AnamnesisError::Unsupported`] for a quantised or otherwise
/// non-scalar `GGUF` type (callers dequantise those before reaching here).
#[cfg(feature = "gguf")]
fn gguf_type_to_hub(dtype: crate::GgufType) -> crate::Result<Dtype> {
    use crate::GgufType;
    // EXHAUSTIVE: `GgufType` is `#[non_exhaustive]`; the wildcard covers future
    // block types, which are quantised and never reach this mapping.
    #[allow(clippy::wildcard_enum_match_arm)]
    match dtype {
        GgufType::F32 => Ok(Dtype::F32),
        GgufType::F16 => Ok(Dtype::F16),
        GgufType::BF16 => Ok(Dtype::BF16),
        GgufType::F64 => Ok(Dtype::F64),
        GgufType::I8 => Ok(Dtype::I8),
        GgufType::I16 => Ok(Dtype::I16),
        GgufType::I32 => Ok(Dtype::I32),
        GgufType::I64 => Ok(Dtype::I64),
        other => Err(AnamnesisError::Unsupported {
            format: "GGUF".into(),
            detail: format!("no scalar hub dtype for {other}"),
        }),
    }
}

/// Maps a hub dtype to a scalar [`GgufType`](crate::GgufType) for the `GGUF`
/// writer.
///
/// # Errors
///
/// Returns [`AnamnesisError::Unsupported`] for dtypes outside the `GGUF` scalar
/// surface (`Bool`, unsigned integers wider than nothing, and `FP8`).
#[cfg(feature = "gguf")]
fn hub_dtype_to_gguf(dtype: Dtype) -> crate::Result<crate::GgufType> {
    use crate::GgufType;
    match dtype {
        Dtype::F32 => Ok(GgufType::F32),
        Dtype::F16 => Ok(GgufType::F16),
        Dtype::BF16 => Ok(GgufType::BF16),
        Dtype::F64 => Ok(GgufType::F64),
        Dtype::I8 => Ok(GgufType::I8),
        Dtype::I16 => Ok(GgufType::I16),
        Dtype::I32 => Ok(GgufType::I32),
        Dtype::I64 => Ok(GgufType::I64),
        Dtype::F8E4M3
        | Dtype::F8E5M2
        | Dtype::Bool
        | Dtype::U8
        | Dtype::U16
        | Dtype::U32
        | Dtype::U64 => Err(AnamnesisError::Unsupported {
            format: "gguf".into(),
            detail: format!(
                "no GGUF dtype counterpart for {dtype} \
                 (Bool/unsigned-integer/FP8 are not in the GGUF scalar surface)"
            ),
        }),
    }
}

/// Converts a float tensor's bytes to `BF16` for the `BnB-NF4` encoder.
/// `BF16` input is **borrowed unchanged** (no copy); `F32` / `F16` are truncated
/// to the upper 16 bits, matching the crate's `f32_bits_to_bf16_bits` convention.
///
/// Returns a [`Cow`] so the common already-`BF16` case (a `.pth` / plain-`BF16`
/// source, or any tensor the hub already dequantised) does not allocate a
/// second full-model-sized buffer alongside the hub.
///
/// # Errors
///
/// Returns [`AnamnesisError::Unsupported`] for a non-float dtype and
/// [`AnamnesisError::Parse`] if the byte count is not a whole number of
/// elements.
#[cfg(feature = "bnb")]
fn to_bf16_bytes<'a>(data: &'a [u8], dtype: Dtype, name: &str) -> crate::Result<Cow<'a, [u8]>> {
    match dtype {
        Dtype::BF16 => Ok(Cow::Borrowed(data)),
        Dtype::F32 => {
            if !data.len().is_multiple_of(4) {
                return Err(AnamnesisError::Parse {
                    reason: format!(
                        "bnb-nf4 `{name}`: F32 byte count {} is not a multiple of 4",
                        data.len()
                    ),
                });
            }
            let mut out = Vec::with_capacity(data.len() / 2);
            for chunk in data.chunks_exact(4) {
                // INDEX: `chunks_exact(4)` guarantees exactly 4 bytes per chunk.
                #[allow(clippy::indexing_slicing)]
                let arr: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
                let bits = u32::from_le_bytes(arr);
                // CAST: BF16 is the upper 16 bits of an f32 — truncation intended.
                #[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
                let bf16 = (bits >> 16) as u16;
                out.extend_from_slice(&bf16.to_le_bytes());
            }
            Ok(Cow::Owned(out))
        }
        Dtype::F16 => {
            if !data.len().is_multiple_of(2) {
                return Err(AnamnesisError::Parse {
                    reason: format!(
                        "bnb-nf4 `{name}`: F16 byte count {} is not a multiple of 2",
                        data.len()
                    ),
                });
            }
            let mut out = Vec::with_capacity(data.len());
            for chunk in data.chunks_exact(2) {
                // INDEX: `chunks_exact(2)` guarantees exactly 2 bytes per chunk.
                #[allow(clippy::indexing_slicing)]
                let arr: [u8; 2] = [chunk[0], chunk[1]];
                let bits = half::f16::from_le_bytes(arr).to_f32().to_bits();
                // CAST: BF16 is the upper 16 bits of an f32 — truncation intended.
                #[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
                let bf16 = (bits >> 16) as u16;
                out.extend_from_slice(&bf16.to_le_bytes());
            }
            Ok(Cow::Owned(out))
        }
        Dtype::F8E4M3
        | Dtype::F8E5M2
        | Dtype::F64
        | Dtype::Bool
        | Dtype::U8
        | Dtype::I8
        | Dtype::U16
        | Dtype::I16
        | Dtype::U32
        | Dtype::I32
        | Dtype::U64
        | Dtype::I64 => Err(AnamnesisError::Unsupported {
            format: "bnb-nf4".into(),
            detail: format!(
                "tensor `{name}` has dtype {dtype}; only F32/F16/BF16 inputs are \
                 supported for BnB-NF4 conversion"
            ),
        }),
    }
}

#[cfg(all(test, feature = "gguf"))]
#[allow(clippy::panic, clippy::unwrap_used, clippy::expect_used)]
mod gguf_metadata_tests {
    use super::{parse_gguf_kv_arg, parse_gguf_metadata_json};
    use crate::{GgufMetadataArray as A, GgufMetadataValue as V};

    #[test]
    fn plain_scalars_are_inferred() {
        let meta = parse_gguf_metadata_json(
            r#"{"s": "llama", "b": true, "small": 32, "neg": -5, "f": 1.5}"#,
        )
        .expect("parse");
        assert_eq!(meta.get("s"), Some(&V::String("llama".to_owned())));
        assert_eq!(meta.get("b"), Some(&V::Bool(true)));
        // Non-negative integers that fit take `u32` — the llama.cpp width for counts.
        assert_eq!(meta.get("small"), Some(&V::U32(32)));
        assert_eq!(meta.get("neg"), Some(&V::I64(-5)));
        assert_eq!(meta.get("f"), Some(&V::F32(1.5)));
    }

    #[test]
    fn plain_arrays_take_their_first_element_type() {
        let meta =
            parse_gguf_metadata_json(r#"{"toks": ["a", "b"], "ids": [1, 2]}"#).expect("parse");
        assert_eq!(
            meta.get("toks"),
            Some(&V::Array(Box::new(A::String(vec![
                "a".to_owned(),
                "b".to_owned()
            ]))))
        );
        assert_eq!(
            meta.get("ids"),
            Some(&V::Array(Box::new(A::U32(vec![1, 2]))))
        );
    }

    #[test]
    fn explicit_form_pins_an_exact_width() {
        let meta = parse_gguf_metadata_json(
            r#"{"blocks": {"type": "u32", "value": 32}, "eps": {"type": "f32", "value": 1e-5}}"#,
        )
        .expect("parse");
        assert_eq!(meta.get("blocks"), Some(&V::U32(32)));
        assert_eq!(meta.get("eps"), Some(&V::F32(1e-5)));
    }

    /// The motivating case: `tokenizer.ggml.token_type` is `Array<I32>` in the
    /// llama.cpp convention, but a JSON array of non-negative integers infers
    /// `Array<U32>`. Only the explicit form gets it right, and anamnesis will not
    /// special-case the key name.
    #[test]
    fn explicit_array_fixes_the_token_type_case() {
        let inferred = parse_gguf_metadata_json(r#"{"tt": [1, 1, 2]}"#).expect("parse");
        assert_eq!(
            inferred.get("tt"),
            Some(&V::Array(Box::new(A::U32(vec![1, 1, 2])))),
            "inference alone yields U32 — the reason the escape hatch exists"
        );

        let explicit = parse_gguf_metadata_json(
            r#"{"tt": {"type": "array", "item_type": "i32", "value": [1, 1, 2]}}"#,
        )
        .expect("parse");
        assert_eq!(
            explicit.get("tt"),
            Some(&V::Array(Box::new(A::I32(vec![1, 1, 2]))))
        );
    }

    #[test]
    fn malformed_documents_are_rejected_with_the_key_named() {
        // Not JSON at all.
        assert!(parse_gguf_metadata_json("{not json").is_err());
        // Not a top-level object.
        assert!(parse_gguf_metadata_json("[1, 2]").is_err());
        // Empty array: nothing to infer an item type from.
        let err = parse_gguf_metadata_json(r#"{"empty": []}"#).unwrap_err();
        assert!(err.to_string().contains("empty"), "got: {err}");
        // Unknown type tag.
        assert!(parse_gguf_metadata_json(r#"{"k": {"type": "u128", "value": 1}}"#).is_err());
        // Out of range for the declared width.
        let err = parse_gguf_metadata_json(r#"{"k": {"type": "u8", "value": 300}}"#).unwrap_err();
        assert!(err.to_string().contains("out of range"), "got: {err}");
        // Null has no GGUF counterpart.
        assert!(parse_gguf_metadata_json(r#"{"k": null}"#).is_err());
    }

    #[test]
    fn kv_args_are_string_valued_and_split_on_the_first_equals() {
        let (key, value) = parse_gguf_kv_arg("general.architecture=llama").expect("parse");
        assert_eq!(key, "general.architecture");
        assert_eq!(value, V::String("llama".to_owned()));

        // A value may itself contain `=`.
        let (key, value) = parse_gguf_kv_arg("k=a=b").expect("parse");
        assert_eq!(key, "k");
        assert_eq!(value, V::String("a=b".to_owned()));

        // Even a numeric-looking value stays a string — typing goes through JSON.
        assert_eq!(
            parse_gguf_kv_arg("n=32").expect("parse").1,
            V::String("32".to_owned())
        );

        assert!(parse_gguf_kv_arg("no-equals").is_err());
        assert!(parse_gguf_kv_arg("=empty-key").is_err());
    }
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used, clippy::expect_used)]
mod stats_tests {
    use super::{ConvertOptions, ConvertTarget, convert};
    use std::path::Path;

    /// `ConvertStats` must **partition** the written tensors: a tensor is either
    /// dequantised on the way in, quantised on the way out, or passed through in
    /// its incoming dtype — never counted twice. The FP8 fixture exercises the
    /// mixed case (one quantised weight plus companions).
    #[test]
    fn stats_partition_the_written_tensors() {
        let input = Path::new("tests/fixtures/safetensors_reference/fp8.safetensors");
        assert!(input.exists(), "committed FP8 fixture missing");
        let dir = tempfile::tempdir().expect("tempdir");
        let out = dir.path().join("out.safetensors");

        let stats = convert(
            input,
            ConvertTarget::Safetensors,
            &out,
            &ConvertOptions::new(),
        )
        .expect("convert fp8 -> safetensors");

        assert!(
            stats.dequantized > 0,
            "the FP8 fixture has a quantised weight: {stats:?}"
        );
        assert_eq!(stats.quantized, 0, "safetensors target quantises nothing");
        assert_eq!(
            stats.dequantized + stats.passthrough,
            stats.tensors,
            "dequantised + passthrough must equal the tensors written: {stats:?}"
        );
    }
}

/// Correctness and determinism coverage for the **quantised** `GGUF` input path.
///
/// Before Phase 7.2 this path had **no test at all**: every `GGUF`-input convert
/// test in `tests/` builds its fixture with `write_gguf`, which rejects
/// `is_quantized()` dtypes (quantised emit is Phase 8.5), so the
/// `dequantize_gguf_to_bf16` branch of [`read_gguf`] was exercised only against
/// the gitignored multi-`GiB` local models — never in CI. These tests close that
/// gap with a hand-rolled `GGUF` writer that *can* emit quantised blocks.
///
/// They live in-crate rather than in `tests/` for one specific reason: the
/// fixture is sized off [`crate::parallel::MIN_PARALLEL_BYTES`], which is
/// `pub(crate)`. Sizing against the constant instead of a hard-coded literal
/// means the tests keep exercising the **parallel** dispatch if the threshold is
/// ever retuned — a hard-coded 5 `MiB` fixture would silently fall back to the
/// sequential path the day the threshold moved to 8 `MiB`, and the determinism
/// suite would go green while testing nothing.
#[cfg(all(test, feature = "gguf"))]
#[allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::indexing_slicing,
    // EXHAUSTIVE: `GgufType` is `#[non_exhaustive]`, so the fixture builder's
    // dtype matches need a wildcard. It models only the three dtypes the
    // fixture uses and panics on anything else — a test-authoring error, never
    // reachable from input. Hoisted here because an arm-level `#[allow]` does
    // not suppress `wildcard_enum_match_arm`.
    clippy::wildcard_enum_match_arm
)]
mod quantized_gguf_tests {
    use super::{ConvertOptions, ConvertTarget, convert};
    use crate::GgufType;
    use crate::parallel::MIN_PARALLEL_BYTES;
    use std::path::PathBuf;

    /// `GGUF` default tensor-data alignment.
    const ALIGNMENT: usize = 32;

    /// One tensor in the synthetic fixture: `(name, dtype, `GGUF`-order shape)`.
    struct Spec {
        name: &'static str,
        dtype: GgufType,
        /// Most-significant-first, as `GGUF` stores it.
        shape: Vec<usize>,
    }

    impl Spec {
        fn new(name: &'static str, dtype: GgufType, shape: &[usize]) -> Self {
            Self {
                name,
                dtype,
                shape: shape.to_vec(),
            }
        }

        fn n_elements(&self) -> usize {
            self.shape.iter().product()
        }

        /// On-disk byte length, from the dtype's block geometry.
        fn byte_len(&self) -> usize {
            let n = self.n_elements();
            match self.dtype {
                GgufType::F32 => n * 4,
                // 32-element blocks: 2-byte f16 scale + 32 int8.
                GgufType::Q8_0 => (n / 32) * 34,
                // 256-element super-blocks, 144 bytes each.
                GgufType::Q4_K => (n / 256) * 144,
                // EXHAUSTIVE: `GgufType` is `#[non_exhaustive]` and this test
                // builder deliberately models only the three dtypes the fixture
                // uses; anything else is a test-authoring error, not input.
                other => panic!("fixture builder does not model {other:?}"),
            }
        }

        /// The `GgufType` discriminant as written in the tensor-info table.
        fn discriminant(&self) -> u32 {
            match self.dtype {
                GgufType::F32 => 0,
                GgufType::Q8_0 => 8,
                GgufType::Q4_K => 12,
                // EXHAUSTIVE: see `byte_len` — only the three modelled dtypes.
                other => panic!("fixture builder does not model {other:?}"),
            }
        }

        /// Deterministic block bytes. Quantised blocks get a **sane** f16 scale
        /// (`0x3C00` = 1.0) rather than random bits: a random scale is often
        /// `NaN`/`Inf`, which would make every tensor dequantise to the same
        /// degenerate payload and let an ordering or slicing bug pass unnoticed.
        fn data(&self) -> Vec<u8> {
            let len = self.byte_len();
            let mut buf: Vec<u8> = (0..len)
                .map(|i| (i.wrapping_mul(2_654_435_761) & 0xFF) as u8)
                .collect();
            match self.dtype {
                GgufType::Q8_0 => {
                    for block in buf.chunks_exact_mut(34) {
                        block[0] = 0x00;
                        block[1] = 0x3C; // f16 1.0, little-endian
                    }
                }
                GgufType::Q4_K => {
                    for block in buf.chunks_exact_mut(144) {
                        block[0] = 0x00;
                        block[1] = 0x3C; // d    = f16 1.0
                        block[2] = 0x00;
                        block[3] = 0x38; // dmin = f16 0.5
                    }
                }
                GgufType::F32 => {}
                // EXHAUSTIVE: see `byte_len` — only the three modelled dtypes.
                other => panic!("fixture builder does not model {other:?}"),
            }
            buf
        }
    }

    /// The fixture layout: a **prime** tensor count (17) with deliberately
    /// skewed sizes, so the atomic-cursor partition genuinely differs between
    /// thread budgets — an equal-count split of 17 items across 2, 4, 8 and 16
    /// workers lands differently every time, and one oversized `Q8_0` tensor
    /// forces the pool to rebalance rather than divide evenly.
    ///
    /// Mostly `F32` by *byte* count (cheap to produce, and it is what pushes the
    /// fixture past `MIN_PARALLEL_BYTES`), but the quantised tensors carry the
    /// interesting work.
    fn fixture_specs() -> Vec<Spec> {
        let mut specs = Vec::new();
        // 8 F32 padding tensors — also exercise the passthrough branch, and one
        // is 2-D so the shape reversal is covered.
        for i in 0..8 {
            let shape: Vec<usize> = if i == 0 {
                vec![350, 400] // 2-D: GGUF order, reversed to [400, 350] in the hub
            } else {
                vec![140_000]
            };
            specs.push(Spec::new(
                match i {
                    0 => "blk.0.attn_norm.weight",
                    1 => "blk.1.attn_norm.weight",
                    2 => "blk.2.attn_norm.weight",
                    3 => "blk.3.attn_norm.weight",
                    4 => "blk.4.attn_norm.weight",
                    5 => "blk.5.attn_norm.weight",
                    6 => "blk.6.attn_norm.weight",
                    _ => "output_norm.weight",
                },
                GgufType::F32,
                &shape,
            ));
        }
        // 5 Q8_0 tensors, deliberately lopsided (the first is ~64x the last).
        specs.push(Spec::new("token_embd.weight", GgufType::Q8_0, &[512, 512]));
        specs.push(Spec::new("blk.0.attn_q.weight", GgufType::Q8_0, &[32_768]));
        specs.push(Spec::new("blk.1.attn_q.weight", GgufType::Q8_0, &[16_384]));
        specs.push(Spec::new("blk.2.attn_q.weight", GgufType::Q8_0, &[8_192]));
        specs.push(Spec::new("blk.3.attn_q.weight", GgufType::Q8_0, &[4_096]));
        // 4 Q4_K tensors — a second kernel family through the same dispatch.
        specs.push(Spec::new(
            "blk.0.ffn_down.weight",
            GgufType::Q4_K,
            &[65_536],
        ));
        specs.push(Spec::new(
            "blk.1.ffn_down.weight",
            GgufType::Q4_K,
            &[32_768],
        ));
        specs.push(Spec::new(
            "blk.2.ffn_down.weight",
            GgufType::Q4_K,
            &[16_384],
        ));
        specs.push(Spec::new("blk.3.ffn_down.weight", GgufType::Q4_K, &[8_192]));
        specs
    }

    // -----------------------------------------------------------------------
    // Raw GGUF writer (quantised-capable)
    // -----------------------------------------------------------------------

    fn push_u32(buf: &mut Vec<u8>, v: u32) {
        buf.extend_from_slice(&v.to_le_bytes());
    }

    fn push_u64(buf: &mut Vec<u8>, v: u64) {
        buf.extend_from_slice(&v.to_le_bytes());
    }

    fn push_string(buf: &mut Vec<u8>, s: &str) {
        push_u64(buf, s.len() as u64);
        buf.extend_from_slice(s.as_bytes());
    }

    fn pad_to_alignment(buf: &mut Vec<u8>) {
        while !buf.len().is_multiple_of(ALIGNMENT) {
            buf.push(0);
        }
    }

    /// Serialises `specs` into a `GGUF` v3 byte image. Deliberately hand-rolled:
    /// the crate's own `write_gguf` refuses quantised dtypes, which is exactly
    /// the case under test.
    fn build_quantized_gguf(specs: &[Spec]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        push_u32(&mut buf, 3); // version
        push_u64(&mut buf, specs.len() as u64);
        push_u64(&mut buf, 2); // kv_count

        // general.architecture (STRING = 8)
        push_string(&mut buf, "general.architecture");
        push_u32(&mut buf, 8);
        push_string(&mut buf, "llama");
        // general.alignment (UINT32 = 4)
        push_string(&mut buf, "general.alignment");
        push_u32(&mut buf, 4);
        push_u32(&mut buf, ALIGNMENT as u32);

        // Tensor-info table. Relative offsets are each padded to the alignment,
        // matching how a real GGUF lays its data section out.
        let mut relative = 0usize;
        let mut offsets = Vec::with_capacity(specs.len());
        for spec in specs {
            offsets.push(relative);
            relative += spec.byte_len();
            while !relative.is_multiple_of(ALIGNMENT) {
                relative += 1;
            }
        }
        for (spec, &offset) in specs.iter().zip(offsets.iter()) {
            push_string(&mut buf, spec.name);
            push_u32(&mut buf, spec.shape.len() as u32);
            for &d in &spec.shape {
                push_u64(&mut buf, d as u64);
            }
            push_u32(&mut buf, spec.discriminant());
            push_u64(&mut buf, offset as u64);
        }

        // Data section, aligned, in the same order the info table declares.
        pad_to_alignment(&mut buf);
        let data_start = buf.len();
        for (spec, &offset) in specs.iter().zip(offsets.iter()) {
            debug_assert_eq!(buf.len() - data_start, offset, "offset table drift");
            buf.extend_from_slice(&spec.data());
            pad_to_alignment(&mut buf);
        }
        buf
    }

    /// Byte-equality with a **bounded** failure message.
    ///
    /// A plain `assert_eq!` on two multi-`MiB` `Vec<u8>`s renders both in full —
    /// a determinism regression here produced a 46 `MB` test log, which is
    /// unreadable and slow enough to look like a hang. Report the length and the
    /// first divergence instead; that is the whole diagnostic anyone needs.
    fn assert_bytes_eq(actual: &[u8], expected: &[u8], context: &str) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "{context}: length differs ({} vs {} bytes)",
            actual.len(),
            expected.len()
        );
        if let Some((offset, (a, e))) = actual
            .iter()
            .zip(expected.iter())
            .enumerate()
            .find(|(_, (a, e))| a != e)
            .map(|(i, (a, e))| (i, (*a, *e)))
        {
            panic!(
                "{context}: first difference at byte {offset} of {} (got {a:#04x}, expected {e:#04x})",
                actual.len()
            );
        }
    }

    /// Writes the fixture into a `TempDir` and returns both so the directory
    /// outlives the path.
    fn write_fixture() -> (tempfile::TempDir, PathBuf, Vec<Spec>) {
        let specs = fixture_specs();
        let bytes = build_quantized_gguf(&specs);
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("quantized.gguf");
        std::fs::write(&path, &bytes).expect("write fixture");
        (dir, path, specs)
    }

    // -----------------------------------------------------------------------
    // Tests
    // -----------------------------------------------------------------------

    /// The fixture must clear [`MIN_PARALLEL_BYTES`], or every determinism test
    /// below would silently run on the sequential path and prove nothing about
    /// the parallel dispatch.
    #[test]
    fn fixture_crosses_the_parallel_threshold() {
        let specs = fixture_specs();
        let total: u64 = specs.iter().map(|s| s.byte_len() as u64).sum();
        assert!(
            total > MIN_PARALLEL_BYTES,
            "fixture is {total} B but MIN_PARALLEL_BYTES is {MIN_PARALLEL_BYTES} B — \
             the determinism tests would not exercise the parallel path"
        );
        assert_eq!(specs.len(), 17, "a prime tensor count is deliberate");
    }

    /// The hand-rolled fixture must actually parse, and the quantised tensors
    /// must dequantise to exactly what the kernel produces when called directly
    /// on the same block bytes. This is the first test in the repo to convert a
    /// quantised `GGUF` at all.
    #[test]
    fn quantized_gguf_converts_to_expected_bf16() {
        let (_dir, path, specs) = write_fixture();
        let dir_out = tempfile::tempdir().expect("tempdir");
        let out = dir_out.path().join("out.safetensors");

        let stats = convert(
            &path,
            ConvertTarget::Safetensors,
            &out,
            &ConvertOptions::new().with_threads(4),
        )
        .expect("convert quantised gguf -> safetensors");

        assert_eq!(stats.tensors, 17);
        assert_eq!(stats.dequantized, 9, "5 Q8_0 + 4 Q4_K are dequantised");
        assert_eq!(stats.passthrough, 8, "the 8 F32 tensors pass through");

        // Re-read the output and check every tensor against the kernel oracle.
        let written = std::fs::read(&out).expect("read output");
        let tensors = safetensors::SafeTensors::deserialize(&written).expect("parse output");

        for spec in &specs {
            let view = tensors.tensor(spec.name).expect("tensor present in output");
            // GGUF shapes are most-significant-first; the hub reverses them.
            let expected_shape: Vec<usize> = spec.shape.iter().copied().rev().collect();
            assert_eq!(view.shape(), expected_shape.as_slice(), "{}", spec.name);

            match spec.dtype {
                GgufType::F32 => {
                    assert_eq!(view.dtype(), safetensors::Dtype::F32, "{}", spec.name);
                    assert_bytes_eq(view.data(), &spec.data(), spec.name);
                }
                dtype => {
                    assert_eq!(view.dtype(), safetensors::Dtype::BF16, "{}", spec.name);
                    let expected =
                        crate::dequantize_gguf_to_bf16(&spec.data(), dtype, spec.n_elements())
                            .expect("oracle dequant");
                    assert_bytes_eq(
                        view.data(),
                        &expected,
                        &format!("{} vs the kernel called directly", spec.name),
                    );
                }
            }
        }
    }

    /// Thread count is a performance knob, never a correctness variable: the
    /// written bytes must be identical for every budget, including the
    /// environment-resolved default.
    #[test]
    fn gguf_to_safetensors_deterministic_across_thread_counts() {
        let (_dir, path, _specs) = write_fixture();
        let dir_out = tempfile::tempdir().expect("tempdir");

        let baseline_path = dir_out.path().join("baseline.safetensors");
        convert(
            &path,
            ConvertTarget::Safetensors,
            &baseline_path,
            &ConvertOptions::new().with_threads(1),
        )
        .expect("baseline convert");
        let baseline = std::fs::read(&baseline_path).expect("read baseline");

        for n in [1usize, 2, 4, 8, 16] {
            let out = dir_out.path().join(format!("t{n}.safetensors"));
            convert(
                &path,
                ConvertTarget::Safetensors,
                &out,
                &ConvertOptions::new().with_threads(n),
            )
            .expect("threaded convert");
            assert_bytes_eq(
                &std::fs::read(&out).expect("read output"),
                &baseline,
                &format!("safetensors output at {n} threads"),
            );
        }

        // The default (hardware-resolved) budget must agree too.
        let default_out = dir_out.path().join("default.safetensors");
        convert(
            &path,
            ConvertTarget::Safetensors,
            &default_out,
            &ConvertOptions::new(),
        )
        .expect("default convert");
        assert_bytes_eq(
            &std::fs::read(&default_out).expect("read output"),
            &baseline,
            "the default thread budget vs the sequential baseline",
        );
    }

    /// The determinism contract holds for the other two targets reachable from a
    /// `GGUF` source, not just safetensors: the `GGUF` writer (dequantise in
    /// place, inheriting the source KV) and the `BnB`-NF4 encoder.
    #[test]
    fn gguf_to_other_targets_deterministic_across_thread_counts() {
        let (_dir, path, _specs) = write_fixture();
        let dir_out = tempfile::tempdir().expect("tempdir");

        let targets: &[(ConvertTarget, &str)] = &[
            (ConvertTarget::Gguf, "gguf"),
            #[cfg(feature = "bnb")]
            (ConvertTarget::BnbNf4, "safetensors"),
        ];

        for &(target, ext) in targets {
            let baseline_path = dir_out.path().join(format!("baseline-{ext}.{ext}"));
            convert(
                &path,
                target,
                &baseline_path,
                &ConvertOptions::new().with_threads(1),
            )
            .expect("baseline convert");
            let baseline = std::fs::read(&baseline_path).expect("read baseline");

            for n in [2usize, 4, 8] {
                let out = dir_out.path().join(format!("t{n}-{ext}.{ext}"));
                convert(&path, target, &out, &ConvertOptions::new().with_threads(n))
                    .expect("threaded convert");
                assert_bytes_eq(
                    &std::fs::read(&out).expect("read output"),
                    &baseline,
                    &format!("{target:?} output at {n} threads"),
                );
            }
        }
    }
}

/// Ad-hoc, `#[ignore]`d stage-isolated scaling measurement for the Phase 7.2
/// `GGUF`-reader parallelisation.
///
/// Lives here rather than in `tests/bench_gguf_convert_adhoc.rs` because the
/// quantity of interest is [`read_hub`] **alone**. An end-to-end `convert()`
/// also serialises the whole `BF16` hub to disk — for a 1.1 B-parameter model
/// that is a 2.2 `GiB` write, which dominates the wall clock and masks whatever
/// the reader does. `read_hub` is private, so only an in-crate test can time it.
///
/// Run with (`--test-threads=1` so the budgets do not contend):
///
/// ```text
/// $env:RUSTFLAGS = "-C target-cpu=native"
/// cargo test --release --features gguf --lib hub_scaling -- --ignored --nocapture --test-threads=1
/// $env:RUSTFLAGS = $null
/// ```
#[cfg(all(test, feature = "gguf"))]
#[allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::indexing_slicing
)]
mod hub_scaling_bench {
    use super::{ConvertOptions, read_hub};
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    /// Best-of-5, matching the `CLAUDE.md` perf gate.
    const SAMPLES: usize = 5;

    fn model_path(file_name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("gguf_reference")
            .join("models")
            .join(file_name)
    }

    /// Reads the committed `gguf-py` timing sidecar for `model`, if present.
    ///
    /// Returns `(median seconds, library version, note)`. Sidecars are produced
    /// by `tests/fixtures/gguf_reference/generate_gguf_dequant_timings.py` and
    /// **are** checked in (unlike the models themselves), so this comparison
    /// prints even on a machine with no Python environment.
    fn python_baseline(model: &str) -> Option<(f64, String, String)> {
        let stem = model.strip_suffix(".gguf").unwrap_or(model);
        let sidecar = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("gguf_reference")
            .join(format!("{stem}.dequant.timing.json"));
        let raw = std::fs::read_to_string(sidecar).ok()?;
        let value: serde_json::Value = serde_json::from_str(&raw).ok()?;
        let seconds = value.get("py_seconds")?.as_f64()?;
        let library = value.get("py_library")?.as_str()?.to_owned();
        let note = value
            .get("note")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("")
            .to_owned();
        Some((seconds, library, note))
    }

    /// Times `read_hub` at each thread budget, reporting the median and the
    /// speedup over the 1-thread (v0.7.1-equivalent) baseline.
    fn hub_scaling_for(model: &str) {
        let input = model_path(model);
        if !input.exists() {
            eprintln!("SKIP {model}: fixture absent (gitignored)");
            return;
        }
        let input_mib = std::fs::metadata(&input).expect("stat").len() as f64 / (1024.0 * 1024.0);
        eprintln!("\n=== read_hub({model}) — {input_mib:.1} MiB input ===");

        let python = python_baseline(model);
        let mut baseline = 0.0_f64;
        for threads in [1usize, 2, 4, 8, 16] {
            let options = ConvertOptions::new().with_threads(threads);
            // Warm-up: also pages the mmap in, so the timed samples measure
            // dequant rather than first-touch page faults.
            drop(read_hub(&input, &options).expect("warm-up read_hub"));

            let mut samples: Vec<f64> = Vec::with_capacity(SAMPLES);
            for _ in 0..SAMPLES {
                let start = Instant::now();
                let hub = read_hub(&input, &options).expect("read_hub");
                samples.push(start.elapsed().as_secs_f64() * 1000.0);
                drop(hub);
            }
            samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = samples[SAMPLES / 2];
            if threads == 1 {
                baseline = median;
            }
            let vs_python = python.as_ref().map_or_else(String::new, |&(py, _, _)| {
                format!("  |  {:.1}x vs gguf-py", (py * 1000.0) / median)
            });
            eprintln!(
                "{threads:>2} threads: median {median:>8.2} ms (min {:.2}, max {:.2}) -> {:.2}x{vs_python}",
                samples[0],
                samples[SAMPLES - 1],
                baseline / median
            );
        }

        match python {
            Some((py, library, note)) => {
                eprintln!("\npython baseline: {:.1} ms ({library})", py * 1000.0);
                if !note.is_empty() {
                    eprintln!("  caveat: {note}");
                }
            }
            None => {
                eprintln!("\npython baseline: no sidecar (run generate_gguf_dequant_timings.py)");
            }
        }
    }

    #[test]
    #[ignore = "ad-hoc measurement; run explicitly with --ignored"]
    fn hub_scaling_smollm2_q4_k_m() {
        hub_scaling_for("SmolLM2-135M-Instruct-Q4_K_M.gguf");
    }

    #[test]
    #[ignore = "ad-hoc measurement; run explicitly with --ignored"]
    fn hub_scaling_tinyllama_q5_0() {
        hub_scaling_for("tinyllama-1.1b-chat-v1.0.Q5_0.gguf");
    }
}
