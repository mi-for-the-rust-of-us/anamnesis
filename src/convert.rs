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
//!   `GGUF` blocks) are dequantised to [`ConvertOptions::output_dtype`], which
//!   defaults to `BF16`. Since v0.7.3 a `GGUF` input can be asked for `F32`
//!   instead, which is the only width that adds no narrowing step of the
//!   crate's own, or `F16`.
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
    /// `safetensors` (alias `bf16`) — dequantise any quantised input to
    /// [`ConvertOptions::output_dtype`] (`BF16` by default), passing scalar
    /// tensors through in their original dtype.
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
    /// Cooperative cancellation handle, polled once per tensor.
    ///
    /// `None` (the default) means the run cannot be cancelled and costs
    /// nothing: no token is allocated and the poll is a `None` check. `Some`
    /// makes the run stop at the next tensor boundary once
    /// [`CancelToken::cancel`](crate::CancelToken::cancel) is called from any
    /// thread, returning [`AnamnesisError::Cancelled`] with no output file
    /// written.
    pub cancel: Option<crate::CancelToken>,
    /// Element type written for tensors this conversion **dequantises**.
    /// `None` (the default) means [`Dtype::BF16`], which is what every release
    /// before v0.7.3 emitted unconditionally.
    ///
    /// Accepts [`Dtype::BF16`], [`Dtype::F32`] and [`Dtype::F16`]; any other
    /// variant is rejected by [`convert`] with
    /// [`AnamnesisError::Unsupported`]. It is a `Dtype` rather than a narrower
    /// enum because that is already the type a hub tensor carries, so no third
    /// dtype vocabulary enters the crate.
    ///
    /// # This governs dequantised tensors only
    ///
    /// Both `remember` and `convert` already emit **mixed-dtype** files, where
    /// dequantised tensors are `BF16` and passthrough tensors (norms, biases,
    /// embeddings, anything not block-quantised) keep whatever dtype the source
    /// held. Setting this to `F32` widens the **dequantised** tensors only and
    /// leaves passthrough tensors exactly as they were. It is emphatically not
    /// "rewrite every tensor as `F32`": an `F16` norm in the source stays an
    /// `F16` norm in the output.
    ///
    /// The reason is that a passthrough tensor is copied, never decoded, so
    /// widening it would invent precision that was never in the file while
    /// doubling its bytes. Callers who want a uniform-dtype file want a cast
    /// pass, which is a different operation from dequantisation.
    ///
    /// # v0.7.3 scope: `GGUF` input only
    ///
    /// Only the `GGUF` reader can honour a non-`BF16` request today. The
    /// safetensors reader dequantises through the `FP8` / `GPTQ` / `AWQ` /
    /// `BnB` kernels, which fuse the narrowing into their hot loops and are
    /// generalised in v0.7.4; asking for `F32` with a quantised safetensors
    /// input is therefore a clean `Unsupported` error rather than a silent
    /// `BF16` fallback. `NPZ` and `.pth` inputs dequantise nothing at all, so
    /// the option is vacuous there and is accepted rather than rejected.
    pub output_dtype: Option<Dtype>,
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

    /// Attaches a cancellation handle, polled once per tensor.
    ///
    /// Keep a clone: the token is how another thread — a signal handler, a
    /// watchdog, a request-timeout task — reaches an in-flight call. See
    /// [`crate::cancel`] for the `PyO3` shape this exists for.
    #[must_use]
    pub fn with_cancel(mut self, cancel: crate::CancelToken) -> Self {
        self.cancel = Some(cancel);
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

    /// Sets the element type written for tensors this conversion dequantises.
    ///
    /// Accepts [`Dtype::BF16`], [`Dtype::F32`] and [`Dtype::F16`]; anything else
    /// is rejected by [`convert`], not here, so that the builder stays
    /// infallible like its siblings. See
    /// [`output_dtype`](ConvertOptions::output_dtype) for the passthrough
    /// policy and the v0.7.3 `GGUF`-only scope.
    #[must_use]
    pub fn with_output_dtype(mut self, dtype: Dtype) -> Self {
        self.output_dtype = Some(dtype);
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
    /// Input tensors that were dequantised on the way in, into
    /// [`ConvertOptions::output_dtype`] (`BF16` by default).
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
///
/// Public since v0.7.6. It was crate-private, which meant an embedder — or the
/// v0.8.0 Python binding — that wanted the polymorphic `parse(path)` /
/// `inspect(path)` the CLI offers had to re-sniff extensions and magic bytes
/// itself, duplicating logic this crate already has and can keep correct.
///
/// Note that [`parse`](fn@crate::parse) is **safetensors-only** while
/// `amn parse` dispatches over all four formats: the CLI's polymorphism lives
/// in its own dispatch, not in the library entry point of the same name.
/// Detect first, then call that format's parser.
///
/// `#[non_exhaustive]`: a new format is a new variant, and the variants that
/// exist depend on which Cargo features are enabled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Format {
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

/// Detects the input format from an in-memory artefact, by magic bytes.
///
/// The counterpart of [`detect_format`] for callers holding bytes rather than a
/// path — a downloaded response, an upload, a `PyO3` `bytes` argument — and the
/// detector [`convert_bytes`] itself uses. Extensions are unavailable here, so
/// this is magic-first and therefore *stricter*: it recognises what the bytes
/// actually are.
///
/// `ZIP`-container formats need one extra step, because `.npz` and `.pth` share
/// the `PK\x03\x04` magic. The central directory is read (bounded by the
/// permanent `ZIP` caps) and the entry names decide: any `.npy` member means
/// `NPZ`, a `data.pkl` member means `.pth`.
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if the bytes match no supported format, or
/// if a `ZIP` container is malformed.
/// Returns [`AnamnesisError::Unsupported`] if the bytes are a recognised format
/// whose Cargo feature is disabled in this build.
pub fn detect_format_from_bytes(bytes: &[u8]) -> crate::Result<Format> {
    detect_format_from_bytes_with_limits(bytes, &ParseLimits::default())
}

/// Detects the input format from an in-memory artefact under a caller-supplied
/// [`ParseLimits`] budget.
///
/// Identical to [`detect_format_from_bytes`] except that the `ZIP`
/// central-directory walk — the one step of detection that allocates in
/// proportion to attacker-controlled input — is charged to `limits` rather than
/// only to the permanent `ZIP` caps. `GGUF` and safetensors detection read a
/// fixed number of bytes and allocate nothing, so the budget is inert for them.
///
/// This is the entry point [`convert_bytes`] uses, so a caller who tightens
/// `ConvertOptions::limits` tightens detection too. Without it, detection would
/// be the one unbounded step in an otherwise bounded pipeline — the same
/// inversion Phase 7.6 item 7 removed from the summary `inspect` calls, and it
/// would be odd to reintroduce it in the same release.
///
/// # Errors
///
/// As [`detect_format_from_bytes`], plus [`AnamnesisError::LimitExceeded`] when
/// the `ZIP` central directory exceeds `limits`.
///
/// # Memory
///
/// `O(1)` for `GGUF` and safetensors. For a `ZIP` container, one entry record
/// per member (name plus offsets), bounded by `limits` and by the permanent
/// `ZIP_MAX_ENTRIES` floor. No member data is read.
// `limits` reaches only the `ZIP` branch, which is compiled out when neither
// `npz` nor `pth` is enabled; the parameter stays in the signature so the public
// API does not change shape with the feature set.
#[cfg_attr(not(any(feature = "npz", feature = "pth")), allow(unused_variables))]
pub fn detect_format_from_bytes_with_limits(
    bytes: &[u8],
    limits: &ParseLimits,
) -> crate::Result<Format> {
    // `GGUF`: a four-byte magic, and nothing else in this set starts with it.
    if bytes.get(..4) == Some(b"GGUF".as_slice()) {
        #[cfg(feature = "gguf")]
        {
            return Ok(Format::Gguf);
        }
        #[cfg(not(feature = "gguf"))]
        {
            return Err(missing_feature_err("GGUF", "GGUF bytes", "gguf"));
        }
    }

    // `ZIP`: shared by `.npz` and `.pth`, so the members decide.
    if bytes.get(..4) == Some(b"PK\x03\x04".as_slice()) {
        #[cfg(any(feature = "npz", feature = "pth"))]
        {
            return detect_zip_flavour(bytes, limits);
        }
        #[cfg(not(any(feature = "npz", feature = "pth")))]
        {
            return Err(missing_feature_err(
                "ZIP-container",
                "a .npz or .pth archive",
                "npz` or `pth",
            ));
        }
    }

    // safetensors: an 8-byte little-endian header length, then a JSON object.
    // Checked as a pair — the length alone is any eight bytes, and the brace
    // alone is any text file, but a plausible length *followed by* `{` at
    // exactly offset 8 is the format's own framing.
    if let Some(len_bytes) = bytes.get(..8) {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(len_bytes);
        let header_len = u64::from_le_bytes(buf);
        // CAST: usize → u64, lossless widening on all supported targets.
        #[allow(clippy::as_conversions)]
        let total = bytes.len() as u64;
        if header_len > 0 && header_len.saturating_add(8) <= total && bytes.get(8) == Some(&b'{') {
            return Ok(Format::Safetensors);
        }
    }

    Err(AnamnesisError::Parse {
        reason: "bytes match no supported format (expected safetensors framing, \
                 `GGUF` magic, or a `ZIP` container holding .npy or data.pkl)"
            .into(),
    })
}

/// Decides whether `ZIP` bytes are an `NPZ` or a `.pth`, by member name.
///
/// Split out so the feature-gated arms stay readable; both formats route
/// through the same vendored central-directory reader the parsers use, so the
/// classification cannot disagree with what the parser will then accept.
#[cfg(any(feature = "npz", feature = "pth"))]
fn detect_zip_flavour(bytes: &[u8], limits: &ParseLimits) -> crate::Result<Format> {
    let mut src = crate::parse::zip::SliceSource::new(bytes);
    let entries = crate::parse::zip::read_central_directory(&mut src, limits)?;

    let mut saw_npy = false;
    let mut saw_pickle = false;
    for entry in &entries {
        let name = crate::parse::zip::strip_archive_prefix(&entry.name);
        if name.ends_with(".npy") {
            saw_npy = true;
        } else if name == "data.pkl" {
            saw_pickle = true;
        }
    }

    // `.npy` first: a `.pth` never holds one, so the check is unambiguous in
    // the direction that matters.
    if saw_npy {
        #[cfg(feature = "npz")]
        {
            return Ok(Format::Npz);
        }
        #[cfg(not(feature = "npz"))]
        {
            return Err(missing_feature_err("NumPy NPZ", "an .npz archive", "npz"));
        }
    }
    if saw_pickle {
        #[cfg(feature = "pth")]
        {
            return Ok(Format::Pth);
        }
        #[cfg(not(feature = "pth"))]
        {
            return Err(missing_feature_err("PyTorch", "a .pth archive", "pth"));
        }
    }

    Err(AnamnesisError::Parse {
        reason: "ZIP container holds neither an .npy member (NPZ) nor data.pkl (.pth)".into(),
    })
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
///
/// Public since v0.7.6, together with [`Format`]. Use
/// [`detect_format_from_bytes`] when the artefact is already in memory.
// `clippy::unnecessary_wraps`: with all of `pth`/`npz`/`gguf` enabled every arm
// is `Ok(_)`; other feature combinations make the wrap load-bearing.
#[allow(clippy::unnecessary_wraps)]
pub fn detect_format(path: &Path) -> crate::Result<Format> {
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
    derive_output_path_for_dtype(input, target, Dtype::BF16)
}

/// Derives an output path, naming the file after the dtype it will actually
/// contain.
///
/// [`ConvertTarget::suffix`] answers "what did this conversion produce", and
/// for the safetensors target that used to be `bf16` unconditionally because
/// `BF16` was the only possible answer. Since v0.7.3 it is not:
/// `--out-dtype f32` would otherwise write `F32` tensors into a file named
/// `model-bf16.safetensors`, which is worse than an unhelpful name because it
/// is an actively wrong one.
///
/// The dtype replaces the suffix only where the suffix *was* a dtype. A `gguf`
/// or `bnb-nf4` target keeps its own suffix, because there the suffix names a
/// container or an encoding rather than the element type, and the dequantised
/// tensors are not what distinguishes the file.
#[must_use]
pub fn derive_output_path_for_dtype(
    input: &Path,
    target: ConvertTarget,
    dequant_dtype: Dtype,
) -> PathBuf {
    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");
    // EXHAUSTIVE: internal dispatch over the crate's own `ConvertTarget`.
    #[allow(clippy::wildcard_enum_match_arm)]
    let suffix = match target {
        // The safetensors target's suffix *is* the output dtype, so it has to
        // track the caller's choice.
        ConvertTarget::Safetensors => match dequant_dtype {
            Dtype::F32 => "f32",
            Dtype::F16 => "f16",
            _ => "bf16",
        },
        other => other.suffix(),
    };
    let new_name = format!(
        "{}-{}.{}",
        strip_quant_suffix(stem),
        suffix,
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
    convert_with_progress(input, target, output, options, || {})
}

/// Converts `input` into `target`, reporting per-tensor progress.
///
/// Behaves identically to [`convert`] — same output bytes, same errors — but
/// calls `on_tensor` once per tensor the reader produces. New in v0.7.6:
/// `remember` has had a progress hook since v0.7.2 and `convert` had none, so a
/// caller converting a 70 B model had no way to show that anything was
/// happening.
///
/// **`on_tensor` fires on the calling thread, and on the parallel path it fires
/// as each worker joins rather than as each tensor completes**, so progress
/// arrives in bursts. That is deliberate — an `FnMut` closing over a progress
/// bar never crosses a thread boundary — and it is why *cancellation* is a
/// [`CancelToken`](crate::CancelToken) on the options rather than a return
/// value from this callback: the token is polled by the workers themselves and
/// so takes effect within one tensor, while this hook could not.
///
/// # Errors
///
/// As [`convert`].
///
/// # Memory
///
/// As [`convert`]: the hook adds nothing.
pub fn convert_with_progress<F>(
    input: &Path,
    target: ConvertTarget,
    output: &Path,
    options: &ConvertOptions,
    mut on_tensor: F,
) -> crate::Result<ConvertStats>
where
    F: FnMut(),
{
    // TRAIT_OBJECT: the hook threads through several private readers with
    // different generic parameters; one `&mut dyn FnMut()` keeps a single
    // signature instead of a type parameter on each of them.
    let hub = read_hub(input, options, &mut on_tensor)?;
    write_hub(&hub, target, Sink::File(output), options)
}

/// The output element type a conversion writes for the tensors it dequantises.
///
/// Resolves [`ConvertOptions::output_dtype`]'s `None` to the `BF16` default and
/// rejects any dtype that is not a supported output width. This is the single
/// runtime boundary the design calls for: past this point the width is a static
/// type parameter and no per-element branch exists.
///
/// # Errors
///
/// Returns [`AnamnesisError::Unsupported`] if `output_dtype` is set to anything
/// other than [`Dtype::BF16`], [`Dtype::F32`] or [`Dtype::F16`].
fn resolve_output_dtype(options: &ConvertOptions) -> crate::Result<Dtype> {
    let requested = options.output_dtype.unwrap_or(Dtype::BF16);
    // EXHAUSTIVE: `Dtype` is `#[non_exhaustive]` and names 15 element types, of
    // which exactly three are dequantisation output widths. The wildcard is the
    // rejection path, not a fallthrough.
    #[allow(clippy::wildcard_enum_match_arm)]
    match requested {
        Dtype::BF16 | Dtype::F32 | Dtype::F16 => Ok(requested),
        other => Err(AnamnesisError::Unsupported {
            format: "convert".into(),
            detail: format!(
                "output dtype {other} is not a dequantisation output width \
                 (supported: bf16, f32, f16)"
            ),
        }),
    }
}

/// Parses in-memory bytes into the hub, dispatching on the **magic-detected**
/// format.
///
/// The bytes twin of [`read_hub`]. Every reader it calls is the same one the
/// path version uses past the parse step, so the two produce identical hubs
/// from identical artefacts — which is what makes `convert_bytes` byte-exact
/// against `convert`.
fn read_hub_from_bytes(
    bytes: &[u8],
    options: &ConvertOptions,
    on_tensor: &mut dyn FnMut(),
) -> crate::Result<Hub> {
    let threads = crate::model::resolve_thread_budget(options.threads);
    let cancel = options.cancel.as_ref();
    let out_dtype = resolve_output_dtype(options)?;
    match detect_format_from_bytes_with_limits(bytes, &options.limits)? {
        Format::Safetensors => {
            let model = crate::parse_bytes_with_limits(bytes.to_vec(), &options.limits)?;
            hub_from_model(&model, threads, cancel, on_tensor, out_dtype)
        }
        #[cfg(feature = "pth")]
        Format::Pth => hub_from_pth(&crate::parse_pth_bytes_with_limits(
            bytes.to_vec(),
            &options.limits,
        )?),
        #[cfg(feature = "npz")]
        Format::Npz => hub_from_npz(crate::parse_npz_bytes_with_limits(
            bytes.to_vec(),
            &options.limits,
        )?),
        #[cfg(feature = "gguf")]
        Format::Gguf => {
            let parsed = crate::parse_gguf_bytes_with_limits(bytes.to_vec(), &options.limits)?;
            hub_from_gguf_dyn(&parsed, threads, cancel, on_tensor, out_dtype)
        }
    }
}

/// The output-width dispatch for a parsed `GGUF`, shared by the path and bytes
/// readers so the `match` on the width exists once.
#[cfg(feature = "gguf")]
pub(crate) fn hub_from_gguf_dyn(
    parsed: &crate::ParsedGguf,
    threads: usize,
    cancel: Option<&crate::CancelToken>,
    on_tensor: &mut dyn FnMut(),
    out_dtype: Dtype,
) -> crate::Result<Hub> {
    // EXHAUSTIVE: `Dtype` is `#[non_exhaustive]`; `resolve_output_dtype` has
    // already rejected everything outside these three.
    #[allow(clippy::wildcard_enum_match_arm)]
    match out_dtype {
        Dtype::BF16 => hub_from_gguf::<crate::Bf16Out>(parsed, threads, cancel, on_tensor),
        Dtype::F32 => hub_from_gguf::<crate::F32Out>(parsed, threads, cancel, on_tensor),
        Dtype::F16 => hub_from_gguf::<crate::F16Out>(parsed, threads, cancel, on_tensor),
        other => Err(AnamnesisError::Unsupported {
            format: "GGUF".into(),
            detail: format!(
                "output dtype {other} is not a dequantisation output width \
                 (supported: bf16, f32, f16)"
            ),
        }),
    }
}

/// The output-width dispatch for a parsed safetensors model, shared by the path
/// and bytes readers.
fn hub_from_model(
    model: &crate::ParsedModel,
    threads: usize,
    cancel: Option<&crate::CancelToken>,
    on_tensor: &mut dyn FnMut(),
    out_dtype: Dtype,
) -> crate::Result<Hub> {
    // EXHAUSTIVE: as `hub_from_gguf_dyn`.
    #[allow(clippy::wildcard_enum_match_arm)]
    let (tensors, dequantized) = match out_dtype {
        Dtype::BF16 => model.hub_tensors::<crate::Bf16Out>(threads, cancel, on_tensor)?,
        Dtype::F32 => model.hub_tensors::<crate::F32Out>(threads, cancel, on_tensor)?,
        Dtype::F16 => model.hub_tensors::<crate::F16Out>(threads, cancel, on_tensor)?,
        other => {
            return Err(AnamnesisError::Unsupported {
                format: "safetensors".into(),
                detail: format!(
                    "output dtype {other} is not a dequantisation output width \
                     (supported: bf16, f32, f16)"
                ),
            });
        }
    };
    Ok(Hub {
        tensors,
        st_metadata: model.header.metadata.clone(),
        dequantized,
        #[cfg(feature = "gguf")]
        gguf_metadata: HashMap::new(),
    })
}

/// Converts in-memory bytes into `target`, returning the produced bytes.
///
/// The in-memory twin of [`convert`], new in v0.7.6. The verb was `Path` →
/// `Path` only, while `parse` and `remember` each had bytes forms, so a caller
/// working from a download or handing bytes across an `FFI` boundary had to
/// round-trip through two temporary files for an operation that never needed
/// the filesystem.
///
/// The input format is detected from **magic bytes** via
/// [`detect_format_from_bytes`], not from an extension there is no longer any
/// of. Every reader and writer past that point is the one [`convert`] uses, so
/// the output is byte-identical to converting the same artefact from a file.
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if the bytes match no supported format or
/// the input is malformed, [`AnamnesisError::Unsupported`] for a target or
/// output dtype this build cannot produce, [`AnamnesisError::LimitExceeded`] if
/// the input exceeds `options.limits`, and
/// [`AnamnesisError::Cancelled`] if the run was cancelled through
/// `options.cancel`.
///
/// # Memory
///
/// Higher than [`convert`]'s, and knowably so. Three things are live at once:
/// `input` as the caller's slice, an **owned copy** of it (each byte-form parser
/// takes ownership of its buffer), and the hub; the output buffer then joins
/// them before the hub drops. Budget roughly `2 × input + hub + output`, against
/// the file path's `input + hub` where the input is a mapping rather than a copy
/// and the output streams to disk.
///
/// The owned copy is inherent to taking `&[u8]`: the byte parsers own their
/// buffers, so a borrowed slice must be copied once. A caller who already holds
/// a `Vec<u8>` and wants that copy back can call the format's own
/// `parse_*_bytes` entry point directly.
pub fn convert_bytes(
    input: &[u8],
    target: ConvertTarget,
    options: &ConvertOptions,
) -> crate::Result<(Vec<u8>, ConvertStats)> {
    convert_bytes_with_progress(input, target, options, || {})
}

/// Converts in-memory bytes into `target`, reporting per-tensor progress.
///
/// [`convert_bytes`] with the hook [`convert_with_progress`] adds to
/// [`convert`]; the same coarseness caveat applies.
///
/// # Errors
///
/// As [`convert_bytes`].
///
/// # Memory
///
/// As [`convert_bytes`].
pub fn convert_bytes_with_progress<F>(
    input: &[u8],
    target: ConvertTarget,
    options: &ConvertOptions,
    mut on_tensor: F,
) -> crate::Result<(Vec<u8>, ConvertStats)>
where
    F: FnMut(),
{
    let hub = read_hub_from_bytes(input, options, &mut on_tensor)?;
    let mut out = Vec::new();
    let stats = write_hub(&hub, target, Sink::Memory(&mut out), options)?;
    Ok((out, stats))
}

/// Parses `input` into the hub, dispatching on the detected format.
///
/// `out_dtype` reaches only the readers that actually dequantise. A reader that
/// cannot honour a non-`BF16` request rejects it here rather than silently
/// emitting `BF16`, so the caller never receives a file whose dtype differs
/// from what they asked for.
fn read_hub(
    input: &Path,
    options: &ConvertOptions,
    on_tensor: &mut dyn FnMut(),
) -> crate::Result<Hub> {
    let threads = crate::model::resolve_thread_budget(options.threads);
    let cancel = options.cancel.as_ref();
    let out_dtype = resolve_output_dtype(options)?;
    match detect_format(input)? {
        Format::Safetensors => read_safetensors(
            input,
            &options.limits,
            threads,
            cancel,
            on_tensor,
            out_dtype,
        ),
        // NPZ and `.pth` dequantise nothing: every tensor is already full
        // precision and is passed through in its source dtype. A non-`BF16`
        // `output_dtype` is therefore vacuous rather than wrong, and is
        // accepted. Rejecting it would mean erroring on `--out-dtype f32` for
        // an NPZ that is already `F32`, which would be hostile.
        #[cfg(feature = "pth")]
        Format::Pth => read_pth(input, &options.limits),
        #[cfg(feature = "npz")]
        Format::Npz => read_npz(input, &options.limits),
        #[cfg(feature = "gguf")]
        Format::Gguf => read_gguf(
            input,
            &options.limits,
            threads,
            cancel,
            on_tensor,
            out_dtype,
        ),
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
    sink: Sink<'_>,
    options: &ConvertOptions,
) -> crate::Result<ConvertStats> {
    match target {
        ConvertTarget::Safetensors => write_safetensors_to(hub, sink),
        #[cfg(feature = "gguf")]
        ConvertTarget::Gguf => write_gguf_target(hub, sink, options),
        #[cfg(not(feature = "gguf"))]
        ConvertTarget::Gguf => Err(AnamnesisError::Unsupported {
            format: "gguf".into(),
            detail: "GGUF emit requires the `gguf` Cargo feature; rebuild with \
                     `--features cli,gguf`"
                .into(),
        }),
        #[cfg(feature = "bnb")]
        ConvertTarget::BnbNf4 => write_bnb_nf4_target(hub, sink),
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
/// `out_dtype` and passing scalar tensors through in their original dtype.
/// Carries the source `__metadata__`.
///
/// `out_dtype` selects the element type the **dequantised** tensors are written
/// in ([`Dtype::BF16`], [`Dtype::F32`] or [`Dtype::F16`]); passthrough tensors
/// keep their source dtype regardless. Resolved to a static type parameter once,
/// here, so the choice costs no per-tensor branch and no per-element branch.
///
/// Until v0.7.4 this rejected every non-`BF16` request that met a quantised
/// input, because the four kernel families it dispatches over (`FP8`, `GPTQ`,
/// `AWQ`, `BnB`) fused the narrowing into their hot loops. Phase 7.4 made them
/// generic over `OutputElement`, so the rejection is gone and this path now
/// behaves exactly like the `GGUF` one.
///
/// # Errors
///
/// Propagates parse and dequantisation errors. `resolve_output_dtype` has
/// already rejected any dtype that is not an output width.
fn read_safetensors(
    path: &Path,
    limits: &ParseLimits,
    threads: usize,
    cancel: Option<&crate::CancelToken>,
    on_tensor: &mut dyn FnMut(),
    out_dtype: Dtype,
) -> crate::Result<Hub> {
    let model = crate::parse_with_limits(path, limits)?;
    hub_from_model(&model, threads, cancel, on_tensor, out_dtype)
}

/// Reads an `NPZ` archive into the hub. Every `NPZ` tensor is full precision, so
/// nothing is dequantised; dtypes are preserved. Tensors are emitted in sorted
/// name order for a deterministic output.
#[cfg(feature = "npz")]
fn read_npz(path: &Path, limits: &ParseLimits) -> crate::Result<Hub> {
    hub_from_npz(crate::parse_npz_with_limits(path, limits)?)
}

/// Normalises an already-parsed `NPZ` map into the hub, draining it.
///
/// Split out of [`read_npz`] at v0.7.6 so the path and bytes readers share one
/// mapping. Takes the map **by value**: draining avoids cloning every array's
/// bytes, which would be a full-model copy.
#[cfg(feature = "npz")]
fn hub_from_npz(
    mut map: std::collections::HashMap<String, crate::NpzTensor>,
) -> crate::Result<Hub> {
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
    hub_from_pth(&crate::parse_pth_with_limits(path, limits)?)
}

/// Normalises an already-parsed `.pth` into the hub.
///
/// Split out of [`read_pth`] at v0.7.6 so the path and bytes readers share one
/// mapping. Nothing is dequantised: a `state_dict` is already full precision.
#[cfg(feature = "pth")]
fn hub_from_pth(parsed: &crate::ParsedPth) -> crate::Result<Hub> {
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
///
/// `out_dtype` selects the element type the **dequantised** tensors are written
/// in ([`Dtype::BF16`], [`Dtype::F32`] or [`Dtype::F16`]). Passthrough tensors
/// are copied in their source dtype regardless, per the policy on
/// [`ConvertOptions::output_dtype`]. The dtype is resolved to a static type
/// parameter once, outside the per-tensor dispatch, so the choice costs no
/// per-tensor branch and no per-element branch at all.
///
/// # Errors
///
/// Returns [`AnamnesisError::Unsupported`] if `out_dtype` is not one of the
/// three supported output widths, and propagates any parse or dequantisation
/// error from the tensors themselves.
#[cfg(feature = "gguf")]
fn read_gguf(
    path: &Path,
    limits: &ParseLimits,
    threads: usize,
    cancel: Option<&crate::CancelToken>,
    on_tensor: &mut dyn FnMut(),
    out_dtype: Dtype,
) -> crate::Result<Hub> {
    let parsed = crate::parse_gguf_with_limits(path, limits)?;
    hub_from_gguf_dyn(&parsed, threads, cancel, on_tensor, out_dtype)
}

/// Normalises an already-parsed `GGUF` into the hub: quantised tensors
/// dequantised to `E`, everything else passed through in its source dtype,
/// shapes reversed into row-major order.
///
/// Split out of the `GGUF` reader at v0.7.6 so `ParsedGguf::remember` and the
/// `convert` reader run the **same** code rather than two transcriptions of it.
/// Until then the `remember` path for a `GGUF` input lived only in the CLI, was
/// sequential, ignored `--threads`, and could not be called from the library at
/// all; `tests/cli_convert.rs` pins the two paths byte-for-byte so they cannot
/// drift again.
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if a tensor's element count overflows
/// `usize` or a dequant worker thread panics, and
/// [`AnamnesisError::Unsupported`] for a `GGUF` dtype with no safetensors
/// equivalent.
///
/// # Memory
///
/// Allocates one owned `Vec<u8>` per tensor and holds them all: peak is the
/// whole model at `E`'s width for the dequantised share plus the source width
/// for the passthrough share. Identical to what `ParsedModel::remember` holds
/// for a safetensors input; per-tensor streaming is ROADMAP Phase 10.
#[cfg(feature = "gguf")]
pub(crate) fn hub_from_gguf<E: crate::OutputElement>(
    parsed: &crate::ParsedGguf,
    threads: usize,
    cancel: Option<&crate::CancelToken>,
    on_tensor: &mut dyn FnMut(),
) -> crate::Result<Hub> {
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
        cancel,
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
                let data = crate::dequantize_gguf::<E>(&tensor.data, tensor.dtype, n_elements)?;
                Ok(HubTensor {
                    name: tensor.name.to_owned(),
                    shape,
                    // The tensor describes itself with the same dtype the
                    // writer just used, taken from the trait rather than
                    // restated here, so the two cannot drift.
                    dtype: E::DTYPE,
                    data,
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
        |_| on_tensor(),
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

/// Where a writer puts its bytes.
///
/// Introduced at v0.7.6 with [`convert_bytes`]. Each writer builds its tensor
/// list once and then matches on this, so a file conversion and an in-memory
/// one cannot drift on dtype mapping, shape handling, or metadata merging —
/// they run the same code and differ only in the last call.
pub(crate) enum Sink<'a> {
    /// Write to this path.
    File(&'a Path),
    /// Append to this buffer.
    Memory(&'a mut Vec<u8>),
}

/// Builds the safetensors views for every hub tensor, in hub order.
///
/// The single view-construction site behind [`write_safetensors_to`], so the
/// file and in-memory destinations cannot drift on dtype mapping, shape, or
/// tensor order — they differ only in the call that consumes these views. The
/// views borrow `hub`, so the returned `Vec` is tied to it.
///
/// # Errors
///
/// Returns [`AnamnesisError::Unsupported`] for a hub dtype with no safetensors
/// equivalent, and [`AnamnesisError::Parse`] if the upstream crate rejects the
/// shape/length pairing.
fn build_hub_views(hub: &Hub) -> crate::Result<Vec<(String, safetensors::tensor::TensorView<'_>)>> {
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
    Ok(views)
}

/// Maps an upstream `safetensors` serialisation failure onto this crate's error
/// type, keeping `IoError` distinguishable from a malformed-input `Parse`.
///
/// Shared by the file and in-memory writers so a caller sees the same variant
/// whichever destination it picked.
// EXHAUSTIVE: `SafeTensorError` is a foreign type that may gain variants.
#[allow(clippy::wildcard_enum_match_arm)]
fn map_serialize_err(e: safetensors::SafeTensorError) -> AnamnesisError {
    match e {
        safetensors::SafeTensorError::IoError(io_err) => AnamnesisError::Io(io_err),
        other => AnamnesisError::Parse {
            reason: format!("failed to write safetensors file: {other}"),
        },
    }
}

/// Counts the written set for a safetensors destination.
///
/// A dequantised tensor did *not* go out in its incoming dtype, so the two
/// counts partition the written set rather than overlapping.
fn safetensors_stats(hub: &Hub) -> ConvertStats {
    ConvertStats {
        tensors: hub.tensors.len(),
        dequantized: hub.dequantized,
        quantized: 0,
        passthrough: hub.tensors.len().saturating_sub(hub.dequantized),
    }
}

/// Writes the hub as safetensors to `sink`, each tensor in its hub dtype.
pub(crate) fn write_safetensors_to(hub: &Hub, sink: Sink<'_>) -> crate::Result<ConvertStats> {
    let views = build_hub_views(hub)?;
    match sink {
        Sink::File(output) => {
            safetensors::tensor::serialize_to_file(views, hub.st_metadata.clone(), output)
                .map_err(map_serialize_err)?;
        }
        Sink::Memory(buf) => {
            let bytes = safetensors::tensor::serialize(views, hub.st_metadata.clone())
                .map_err(map_serialize_err)?;
            buf.extend_from_slice(&bytes);
        }
    }
    Ok(safetensors_stats(hub))
}

/// Writes the hub as an unquantised `GGUF` file, reversing shapes back to
/// most-significant-first.
#[cfg(feature = "gguf")]
fn write_gguf_target(
    hub: &Hub,
    sink: Sink<'_>,
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
    match sink {
        Sink::File(output) => write_gguf(output, &tensors, &metadata)?,
        Sink::Memory(buf) => {
            // `write_gguf_to_writer` needs `Write + Seek` because the writer
            // back-patches the tensor-data offsets once the header size is
            // known; a `Cursor` over the buffer provides both.
            let mut cursor = std::io::Cursor::new(Vec::new());
            crate::write_gguf_to_writer(&mut cursor, &tensors, &metadata)?;
            buf.extend_from_slice(&cursor.into_inner());
        }
    }

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
fn write_bnb_nf4_target(hub: &Hub, sink: Sink<'_>) -> crate::Result<ConvertStats> {
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
    match sink {
        Sink::File(output) => write_bnb_nf4_safetensors(&inputs, output)?,
        Sink::Memory(buf) => {
            buf.extend_from_slice(&crate::write_bnb_nf4_safetensors_bytes(&inputs)?);
        }
    }

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
/// `BF16` input is **borrowed unchanged** (no copy); `F32` / `F16` are narrowed
/// with round-to-nearest-even via
/// [`f32_bits_to_bf16_bits`](crate::remember::fp8), the crate's single
/// narrowing convention.
///
/// **Fixed in v0.7.4.** Both arms previously did `bits >> 16`, a plain
/// truncation, while this doc comment already claimed they matched
/// `f32_bits_to_bf16_bits`. The code was wrong, not the comment: truncation
/// biases every inexact value toward zero by up to one `BF16` `ULP`, where
/// round-to-nearest-even is unbiased and is what every other narrowing site in
/// the crate does. The change is reachable from `convert --to bnb-nf4` with an
/// `F32`/`F16` source, and became more reachable in v0.7.4 because
/// `--out-dtype f32` can now feed this helper from the hub.
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
// MEASURED-REVERT: clippy::chunks_exact_to_as_chunks (new in Rust 1.98).
//
// **UNMEASURABLE, which is a different claim from "measured slow" and is stated
// as such.** This is the per-element narrowing every `bnb-nf4` conversion runs
// over a whole model, so it is hot by any reading, and it remains the one
// suppressed site in `src/` with no number of its own.
//
// Phase 7.7 item 5 tried and could not get one. The function is private, so the
// bench crate cannot call it; the only public path that reaches it
// (`convert_bytes` into `ConvertTarget::BnbNf4`) also parses, builds the hub and
// runs the `NF4` encode, in which this loop is a small enough fraction that a
// 5 % change in it lands below the harness's own ~2 % floor. A benchmark that
// cannot resolve the effect is not evidence, and building one anyway would have
// dressed a guess as a measurement.
//
// **And no inference is available from the sites that were measured**, because
// Phase 7.7's central finding is that this migration is unpredictable per site:
// it gained `GPTQ` 9.87 %, cost `AWQ` ~45 %, cost `BnB` `INT8` +32 % through
// `write_scratch` while gaining `GPTQ` `F16` 10 % in the same commit. Nothing
// here can be argued from a sibling's number.
//
// Unmeasured plus hot means unchanged, per `CLAUDE.md`. To settle it, give the
// narrowing a callable seam rather than a bigger benchmark.
// See CONVENTIONS.md § MEASURED-REVERT Annotation.
#[allow(clippy::chunks_exact_to_as_chunks)]
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
            // Pre-sized, then written through `chunks_exact_mut`: the crate's
            // standard pass-2 shape (`CONVENTIONS.md` § SIMD-friendly loops,
            // rules 1 and 6 — flat slices, distinct input and output). The
            // first v0.7.4 draft pushed with `extend_from_slice` per element,
            // which carries a capacity check every iteration and cannot
            // vectorise.
            let mut out = vec![0u8; data.len() / 2];
            // VECTORIZED: confirmed AVX2 vpaddd + vpsrld + vpand on %ymm
            // (8-wide) in `--emit=asm`, x86-64 target-cpu=native, opt-level=3 —
            // the round-to-nearest-even bias-add and shift, eight lanes at a
            // time. Byte-identical arithmetic to `Bf16Out::write_scratch`,
            // which is the point: one narrowing convention, one codegen shape.
            for (chunk, out_pair) in data.chunks_exact(4).zip(out.chunks_exact_mut(2)) {
                // INDEX: `chunks_exact(4)` guarantees exactly 4 bytes per chunk.
                #[allow(clippy::indexing_slicing)]
                let arr: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
                let bf16 = crate::remember::fp8::f32_bits_to_bf16_bits(u32::from_le_bytes(arr));
                out_pair.copy_from_slice(&bf16.to_le_bytes());
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
            // Pre-sized and written through `chunks_exact_mut`, as the `F32`
            // arm above; `F16` is 2 bytes in and 2 bytes out, so the output is
            // the same length as the input.
            let mut out = vec![0u8; data.len()];
            // VECTORIZED: confirmed AVX2 vcvtph2ps + vpaddd + vpsrld + vpand on
            // %ymm in `--emit=asm`, x86-64 target-cpu=native, opt-level=3 — the
            // `F16C` widening load followed by the same round-to-nearest-even
            // sequence the `F32` arm uses.
            for (chunk, out_pair) in data.chunks_exact(2).zip(out.chunks_exact_mut(2)) {
                // INDEX: `chunks_exact(2)` guarantees exactly 2 bytes per chunk.
                #[allow(clippy::indexing_slicing)]
                let arr: [u8; 2] = [chunk[0], chunk[1]];
                let bits = half::f16::from_le_bytes(arr).to_f32().to_bits();
                let bf16 = crate::remember::fp8::f32_bits_to_bf16_bits(bits);
                out_pair.copy_from_slice(&bf16.to_le_bytes());
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

#[cfg(all(test, feature = "bnb"))]
#[allow(clippy::panic, clippy::unwrap_used, clippy::expect_used)]
mod to_bf16_bytes_tests {
    use super::{Dtype, to_bf16_bytes};

    /// The `F32` arm must **round to nearest even**, not truncate.
    ///
    /// Until v0.7.4 it did `bits >> 16`, which is what these three values
    /// distinguish: each is chosen so truncation and round-to-nearest-even
    /// disagree, which a value with a zero low half would not catch.
    #[test]
    fn f32_arm_rounds_to_nearest_even() {
        // (f32 bits, expected BF16 under RNE, what truncation would give)
        let cases = [
            // Just above the midpoint between 1.0 and the next BF16: rounds up.
            (0x3F80_9000_u32, 0x3F81_u16, 0x3F80_u16),
            // Exactly the midpoint with an even LSB: stays put (ties-to-even).
            (0x3F80_8000, 0x3F80, 0x3F80),
            // Exactly the midpoint with an odd LSB: rounds up to even.
            (0x3F81_8000, 0x3F82, 0x3F81),
        ];
        let mut input = Vec::new();
        for (bits, _, _) in cases {
            input.extend_from_slice(&bits.to_le_bytes());
        }

        let out = to_bf16_bytes(&input, Dtype::F32, "w").expect("F32 narrowing");
        for (i, (bits, want_rne, would_truncate)) in cases.into_iter().enumerate() {
            // INDEX: `out` is 2 bytes per input element by construction.
            #[allow(clippy::indexing_slicing)]
            let got = u16::from_le_bytes([out[i * 2], out[i * 2 + 1]]);
            assert_eq!(got, want_rne, "0x{bits:08X} should round to nearest even");
            if want_rne != would_truncate {
                assert_ne!(got, would_truncate, "0x{bits:08X} must not truncate");
            }
        }
    }

    /// `BF16` input is borrowed, never copied and never re-rounded.
    #[test]
    fn bf16_arm_borrows_unchanged() {
        let input = vec![0x34_u8, 0x12, 0x78, 0x56];
        let out = to_bf16_bytes(&input, Dtype::BF16, "w").expect("BF16 passthrough");
        assert_eq!(&*out, &input[..]);
        assert!(
            matches!(out, std::borrow::Cow::Borrowed(_)),
            "must not copy"
        );
    }

    /// The `F16` arm rounds too.
    ///
    /// Picking a witness here needs care, which is why the value is given as
    /// raw bits rather than a decimal literal. `f16` has 10 mantissa bits, so
    /// widening to `f32` shifts them up by 13 and leaves the low 13 bits zero.
    /// The discarded half is therefore `0x0000`, `0x2000`, `0x4000`, `0x6000`,
    /// `0x8000`, … and only values **strictly above** `0x8000` make rounding
    /// and truncation disagree. `0x3C05` is `1 + 5/1024`, whose `f32` form is
    /// `0x3F80_A000`: discarded half `0xA000 > 0x8000`, so it rounds up to
    /// `0x3F81` while truncation would give `0x3F80`. The first draft of this
    /// test used `1 + 1/1024` (`0x3F80_2000`), where both answers coincide and
    /// the test proved nothing.
    #[test]
    fn f16_arm_rounds_to_nearest_even() {
        let value = half::f16::from_bits(0x3C05);
        let input = value.to_le_bytes();
        let out = to_bf16_bytes(&input, Dtype::F16, "w").expect("F16 narrowing");
        // INDEX: exactly one element in, two bytes out.
        #[allow(clippy::indexing_slicing)]
        let got = u16::from_le_bytes([out[0], out[1]]);
        let widened = value.to_f32().to_bits();
        // CAST: u32 -> u16, this *is* the truncation the fix replaced.
        #[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
        let truncated = (widened >> 16) as u16;
        assert_eq!(got, crate::remember::fp8::f32_bits_to_bf16_bits(widened));
        assert_ne!(got, truncated, "F16 arm must not truncate either");
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
    use super::{
        AnamnesisError, ConvertOptions, ConvertTarget, Format, convert, convert_bytes,
        convert_with_progress, derive_output_path, derive_output_path_for_dtype,
    };
    use crate::GgufType;
    use crate::parallel::MIN_PARALLEL_BYTES;
    use crate::parse::safetensors::Dtype;
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
                    for block in buf.as_chunks_mut::<34>().0 {
                        block[0] = 0x00;
                        block[1] = 0x3C; // f16 1.0, little-endian
                    }
                }
                GgufType::Q4_K => {
                    for block in buf.as_chunks_mut::<144>().0 {
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

    /// `ParsedGguf::remember` and `convert --to safetensors` are the **same
    /// operation** on a `GGUF` input, so they must produce the same file.
    ///
    /// This is the regression guard for the v0.7.6 refactor. Until then the
    /// `remember` path for a `GGUF` lived in the CLI as a 121-line transcription
    /// of `convert`'s reader; the two happened to agree byte-for-byte, which is
    /// exactly the property a transcription loses first and silently. Both now
    /// call `hub_from_gguf`, and this pins that they cannot drift again — at
    /// every output width, and through the in-memory twin as well as the file.
    #[test]
    fn remember_matches_convert_byte_for_byte_at_every_width() {
        let (_dir, path, _specs) = write_fixture();
        let dir_out = tempfile::tempdir().expect("tempdir");
        let parsed = crate::parse_gguf(&path).expect("parse gguf fixture");

        let widths: &[(crate::TargetDtype, Dtype, &str)] = &[
            (crate::TargetDtype::BF16, Dtype::BF16, "bf16"),
            (crate::TargetDtype::F32, Dtype::F32, "f32"),
            (crate::TargetDtype::F16, Dtype::F16, "f16"),
        ];

        for &(target, out_dtype, label) in widths {
            let via_convert = dir_out.path().join(format!("convert-{label}.safetensors"));
            convert(
                &path,
                ConvertTarget::Safetensors,
                &via_convert,
                &ConvertOptions::new()
                    .with_output_dtype(out_dtype)
                    .with_threads(4),
            )
            .expect("convert to safetensors");
            let expected = std::fs::read(&via_convert).expect("read convert output");

            let via_remember = dir_out.path().join(format!("remember-{label}.safetensors"));
            parsed
                .remember_with_options(
                    &via_remember,
                    target,
                    crate::RememberOptions::new().with_threads(4),
                )
                .expect("remember");
            assert_bytes_eq(
                &std::fs::read(&via_remember).expect("read remember output"),
                &expected,
                &format!("remember vs convert at {label}"),
            );

            let in_memory = parsed
                .remember_to_bytes_with_options(
                    target,
                    crate::RememberOptions::new().with_threads(4),
                )
                .expect("remember_to_bytes");
            assert_bytes_eq(
                &in_memory,
                &expected,
                &format!("remember_to_bytes vs convert at {label}"),
            );
        }
    }

    /// `convert_bytes` produces byte-identical output to `convert`, for every
    /// target reachable from a quantised `GGUF` source.
    ///
    /// This is what makes the in-memory verb safe to reach for: it is not a
    /// second implementation, it is the same readers and writers with a
    /// different sink, and this asserts that rather than assuming it. A
    /// `GGUF` source exercises the widest path — dequantisation, shape
    /// reversal, and metadata inheritance on the `gguf → gguf` arm.
    #[test]
    fn convert_bytes_matches_convert_for_every_target() {
        let (_dir, path, _specs) = write_fixture();
        let dir_out = tempfile::tempdir().expect("tempdir");
        let source = std::fs::read(&path).expect("read fixture bytes");

        let targets: &[(ConvertTarget, &str)] = &[
            (ConvertTarget::Safetensors, "safetensors"),
            (ConvertTarget::Gguf, "gguf"),
            #[cfg(feature = "bnb")]
            (ConvertTarget::BnbNf4, "bnb-nf4"),
        ];

        for &(target, label) in targets {
            let out_path = dir_out.path().join(format!("file-{label}"));
            let file_stats = convert(
                &path,
                target,
                &out_path,
                &ConvertOptions::new().with_threads(4),
            )
            .expect("convert to file");

            let (bytes, byte_stats) =
                convert_bytes(&source, target, &ConvertOptions::new().with_threads(4))
                    .expect("convert_bytes");

            assert_bytes_eq(
                &bytes,
                &std::fs::read(&out_path).expect("read file output"),
                &format!("convert_bytes vs convert for {label}"),
            );
            assert_eq!(
                byte_stats.tensors, file_stats.tensors,
                "{label}: stats must agree too"
            );
            assert_eq!(byte_stats.dequantized, file_stats.dequantized);
        }
    }

    /// Detection is bounded by the caller's budget, not only by the permanent
    /// `ZIP` floor.
    ///
    /// The `ZIP` central-directory walk is the one step of detection that
    /// allocates in proportion to attacker-controlled input, and
    /// `convert_bytes` runs it *before* anything else. Leaving it unbounded
    /// would have made detection the single untightenable step in an otherwise
    /// bounded pipeline — the inversion Phase 7.6 item 7 had just removed from
    /// the summary `inspect` calls.
    #[cfg(feature = "npz")]
    #[test]
    fn detection_and_convert_bytes_honour_the_caller_budget() {
        // A small NPZ, built through the same writer the NPZ tests use.
        let mut archive = Vec::new();
        {
            let mut zip = zip::ZipWriter::new(std::io::Cursor::new(&mut archive));
            let opts = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            for name in ["a.npy", "b.npy"] {
                zip.start_file(name, opts).expect("start entry");
                let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (2,), }";
                let mut npy = b"\x93NUMPY\x01\x00".to_vec();
                let mut padded = header.as_bytes().to_vec();
                while !(10 + padded.len() + 1).is_multiple_of(64) {
                    padded.push(b' ');
                }
                padded.push(b'\n');
                // CAST: usize → u16, the padded header is 64 bytes by
                // construction and the format's own field is `u16`.
                #[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
                npy.extend_from_slice(&(padded.len() as u16).to_le_bytes());
                npy.extend_from_slice(&padded);
                npy.extend_from_slice(&[0u8; 8]);
                std::io::Write::write_all(&mut zip, &npy).expect("write entry");
            }
            zip.finish().expect("finish archive");
        }

        // Unbounded detection sees an NPZ.
        assert_eq!(
            crate::detect_format_from_bytes(&archive).expect("detect"),
            Format::Npz
        );

        // A budget below the declared entry count rejects during detection.
        let tight = crate::ParseLimits::default().with_max_item_count(1);
        let err = crate::detect_format_from_bytes_with_limits(&archive, &tight)
            .expect_err("an item-count budget must bound the detection walk");
        assert!(
            matches!(err, AnamnesisError::LimitExceeded { .. }),
            "expected LimitExceeded, got {err:?}"
        );

        // And `convert_bytes` inherits it from `ConvertOptions::limits`.
        let err = convert_bytes(
            &archive,
            ConvertTarget::Safetensors,
            &ConvertOptions::new().with_limits(tight),
        )
        .expect_err("convert_bytes must not detect outside its own budget");
        assert!(
            matches!(err, AnamnesisError::LimitExceeded { .. }),
            "expected LimitExceeded, got {err:?}"
        );
    }

    /// The magic-byte detector recognises what the parsers accept, and refuses
    /// what they would not.
    #[test]
    fn detect_format_from_bytes_recognises_each_container() {
        let (_dir, path, _specs) = write_fixture();
        let gguf = std::fs::read(&path).expect("read gguf fixture");
        assert_eq!(
            crate::detect_format_from_bytes(&gguf).expect("gguf detect"),
            Format::Gguf
        );

        // A safetensors artefact, produced rather than hand-built, so the
        // detector is checked against the real framing.
        let dir_out = tempfile::tempdir().expect("tempdir");
        let st_path = dir_out.path().join("out.safetensors");
        convert(
            &path,
            ConvertTarget::Safetensors,
            &st_path,
            &ConvertOptions::new(),
        )
        .expect("convert to safetensors");
        let st = std::fs::read(&st_path).expect("read safetensors");
        assert_eq!(
            crate::detect_format_from_bytes(&st).expect("safetensors detect"),
            Format::Safetensors
        );

        for bad in [
            b"".as_slice(),
            b"not a model".as_slice(),
            b"GGU".as_slice(),
            &[0u8; 32],
        ] {
            assert!(
                crate::detect_format_from_bytes(bad).is_err(),
                "bytes {bad:?} must not be claimed as a supported format"
            );
        }
    }

    /// A cancelled run returns `Cancelled` and writes **no output file**.
    ///
    /// The "no file" half is the part worth pinning: every path builds its
    /// result in memory before serialising, so the check lands strictly before
    /// any byte reaches the filesystem. If that ever stops being true, a
    /// cancelled convert starts leaving truncated safetensors behind, and the
    /// failure would be silent.
    #[test]
    fn a_cancelled_convert_writes_nothing() {
        let (_dir, path, _specs) = write_fixture();
        let dir_out = tempfile::tempdir().expect("tempdir");
        let out = dir_out.path().join("cancelled.safetensors");

        let token = crate::CancelToken::new();
        token.cancel();

        let err = convert(
            &path,
            ConvertTarget::Safetensors,
            &out,
            &ConvertOptions::new().with_cancel(token).with_threads(4),
        )
        .expect_err("a cancelled convert must not succeed");

        assert!(
            matches!(err, AnamnesisError::Cancelled),
            "expected Cancelled, got {err:?}"
        );
        assert!(
            !out.exists(),
            "a cancelled convert must not leave an output file behind"
        );
    }

    /// Cancellation is observed at every thread count, sequential included.
    ///
    /// The sequential path polls at the top of its loop and the parallel path
    /// polls where the cursor hands out an item; both have to agree, or
    /// cancellation would be a feature of the thread budget.
    #[test]
    fn cancellation_is_observed_at_every_thread_count() {
        let (_dir, path, _specs) = write_fixture();
        let parsed = crate::parse_gguf(&path).expect("parse gguf fixture");

        for n in [1usize, 2, 4, 8] {
            let token = crate::CancelToken::new();
            token.cancel();
            let err = parsed
                .remember_to_bytes_with_options(
                    crate::TargetDtype::BF16,
                    crate::RememberOptions::new()
                        .with_threads(n)
                        .with_cancel(token),
                )
                .expect_err("a cancelled remember must not succeed");
            assert!(
                matches!(err, AnamnesisError::Cancelled),
                "at {n} threads: expected Cancelled, got {err:?}"
            );
        }
    }

    /// An **uncancelled** token changes nothing: same bytes as no token at all.
    ///
    /// The regression this guards is a poll placed somewhere that skips work
    /// rather than stopping it.
    #[test]
    fn an_uncancelled_token_changes_no_byte() {
        let (_dir, path, _specs) = write_fixture();
        let parsed = crate::parse_gguf(&path).expect("parse gguf fixture");

        let baseline = parsed
            .remember_to_bytes(crate::TargetDtype::BF16)
            .expect("baseline remember");

        for n in [1usize, 4] {
            let with_token = parsed
                .remember_to_bytes_with_options(
                    crate::TargetDtype::BF16,
                    crate::RememberOptions::new()
                        .with_threads(n)
                        .with_cancel(crate::CancelToken::new()),
                )
                .expect("remember with a live token");
            assert_bytes_eq(
                &with_token,
                &baseline,
                &format!("an uncancelled token at {n} threads"),
            );
        }
    }

    /// `convert_with_progress` fires once per tensor and produces the same file.
    #[test]
    fn convert_with_progress_counts_tensors_and_changes_no_byte() {
        let (_dir, path, _specs) = write_fixture();
        let dir_out = tempfile::tempdir().expect("tempdir");
        let plain = dir_out.path().join("plain.safetensors");
        let hooked = dir_out.path().join("hooked.safetensors");

        let stats = convert(
            &path,
            ConvertTarget::Safetensors,
            &plain,
            &ConvertOptions::new().with_threads(4),
        )
        .expect("plain convert");

        let mut seen = 0usize;
        let hooked_stats = convert_with_progress(
            &path,
            ConvertTarget::Safetensors,
            &hooked,
            &ConvertOptions::new().with_threads(4),
            || seen += 1,
        )
        .expect("convert with progress");

        assert_eq!(
            seen, stats.tensors,
            "the hook must fire once per tensor the reader produced"
        );
        assert_eq!(hooked_stats.tensors, stats.tensors);
        assert_bytes_eq(
            &std::fs::read(&hooked).expect("read hooked output"),
            &std::fs::read(&plain).expect("read plain output"),
            "a progress hook must not change an output byte",
        );
    }

    /// `GgufInspectInfo::dequantized_size` predicts what `remember` actually
    /// writes, at every width.
    ///
    /// The point of the estimate is that a host can gate on it *before*
    /// committing, so "roughly right" is not the bar: this asserts it equals the
    /// exact tensor-data byte count of the file `remember` then produces. Before
    /// v0.7.6 the figure did not exist for `GGUF` at all, and `GGUF` is the
    /// format where it cannot be guessed — the expansion ratio is per-kernel, so
    /// the on-disk total predicts nothing.
    #[test]
    fn gguf_dequantized_size_predicts_what_remember_writes() {
        use crate::InspectSummary as _;

        let (_dir, path, _specs) = write_fixture();
        let parsed = crate::parse_gguf(&path).expect("parse gguf fixture");

        for &(target, label) in &[
            (crate::TargetDtype::BF16, "bf16"),
            (crate::TargetDtype::F32, "f32"),
            (crate::TargetDtype::F16, "f16"),
        ] {
            let info = parsed
                .inspect_with_options(&crate::InspectOptions::new().with_output_dtype(target));
            assert_eq!(
                info.output_dtype(),
                target,
                "{label}: the figure must carry the width it assumes"
            );

            let bytes = parsed.remember_to_bytes(target).expect("remember_to_bytes");
            let written = crate::parse_safetensors_header(&bytes)
                .expect("parse the produced header")
                .tensors
                .iter()
                .fold(0u64, |acc, t| {
                    // CAST: usize → u64, lossless widening.
                    #[allow(clippy::as_conversions)]
                    let len = t.byte_len() as u64;
                    acc + len
                });

            assert_eq!(
                info.dequantized_size(),
                written,
                "{label}: estimate {} vs {written} bytes actually written",
                info.dequantized_size()
            );
        }
    }

    /// The four `InspectSummary` numbers agree with the concrete fields, and the
    /// estimate really does move with the requested width.
    ///
    /// `current_size` must **not** move: it is what the file costs as stored,
    /// which no output-dtype request can change. Getting that backwards would
    /// make the gate compare two figures that both moved.
    #[test]
    fn inspect_summary_reads_the_same_numbers_as_the_fields() {
        use crate::InspectSummary as _;

        let (_dir, path, _specs) = write_fixture();
        let parsed = crate::parse_gguf(&path).expect("parse gguf fixture");

        let bf16 = parsed.inspect();
        assert_eq!(bf16.tensor_count(), bf16.tensor_count);
        assert_eq!(bf16.current_size(), bf16.total_bytes);
        assert_eq!(bf16.dequantized_size(), bf16.dequantized_size);
        assert_eq!(bf16.output_dtype(), crate::TargetDtype::BF16);

        let f32 = parsed.inspect_with_options(
            &crate::InspectOptions::new().with_output_dtype(crate::TargetDtype::F32),
        );
        assert_eq!(
            f32.current_size(),
            bf16.current_size(),
            "the stored size cannot depend on the width the caller intends"
        );
        assert!(
            f32.dequantized_size() > bf16.dequantized_size(),
            "an F32 request must estimate larger than BF16 ({} vs {})",
            f32.dequantized_size(),
            bf16.dequantized_size()
        );
        assert!(
            bf16.expansion() > 0,
            "a quantised fixture must expand on dequantisation"
        );
    }

    /// `--threads` reaches the `GGUF` `remember` path and does not change a byte.
    ///
    /// Before v0.7.6 the CLI arm took no thread budget at all, so the flag was
    /// accepted and discarded; the sibling `convert` test could not see that
    /// because it exercised a different function. The fixture clears
    /// [`MIN_PARALLEL_BYTES`] (asserted by `fixture_clears_the_parallel_threshold`),
    /// so the parallel dispatch really runs.
    #[test]
    fn remember_is_deterministic_across_thread_counts() {
        let (_dir, path, _specs) = write_fixture();
        let parsed = crate::parse_gguf(&path).expect("parse gguf fixture");

        let baseline = parsed
            .remember_to_bytes_with_options(
                crate::TargetDtype::BF16,
                crate::RememberOptions::new().with_threads(1),
            )
            .expect("sequential remember");

        for n in [1usize, 2, 4, 8, 16] {
            let out = parsed
                .remember_to_bytes_with_options(
                    crate::TargetDtype::BF16,
                    crate::RememberOptions::new().with_threads(n),
                )
                .expect("threaded remember");
            assert_bytes_eq(&out, &baseline, &format!("GGUF remember at {n} threads"));
        }

        // The default (hardware-resolved) budget must agree too.
        let default_out = parsed
            .remember_to_bytes(crate::TargetDtype::BF16)
            .expect("default remember");
        assert_bytes_eq(
            &default_out,
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

    // -----------------------------------------------------------------------
    // Caller-chosen output dtype (Phase 7.3)
    // -----------------------------------------------------------------------

    /// Every supported output dtype round-trips through `convert`, and each
    /// dequantised tensor matches the kernel called directly at that same width.
    ///
    /// This is the end-to-end counterpart of the unit tests on `OutputElement`:
    /// it proves the dtype survives `ConvertOptions` -> `read_hub` ->
    /// `read_gguf` -> the safetensors writer, and that the header records the
    /// width actually written.
    #[test]
    fn convert_honours_every_output_dtype_end_to_end() {
        for (requested, expected_st) in [
            (crate::Dtype::BF16, safetensors::Dtype::BF16),
            (crate::Dtype::F32, safetensors::Dtype::F32),
            (crate::Dtype::F16, safetensors::Dtype::F16),
        ] {
            let (_dir, path, specs) = write_fixture();
            let dir_out = tempfile::tempdir().expect("tempdir");
            let out = dir_out.path().join("out.safetensors");

            let stats = convert(
                &path,
                ConvertTarget::Safetensors,
                &out,
                &ConvertOptions::new()
                    .with_threads(4)
                    .with_output_dtype(requested),
            )
            .unwrap_or_else(|e| panic!("convert at {requested}: {e}"));
            assert_eq!(stats.dequantized, 9, "{requested}");

            let written = std::fs::read(&out).expect("read output");
            let tensors = safetensors::SafeTensors::deserialize(&written).expect("parse output");

            for spec in &specs {
                let view = tensors.tensor(spec.name).expect("tensor present");
                match spec.dtype {
                    // PASSTHROUGH POLICY, asserted rather than described: an F32
                    // source tensor stays F32 and keeps its exact bytes even when
                    // the caller asks for BF16 or F16 output. `--out-dtype`
                    // governs dequantised tensors only.
                    GgufType::F32 => {
                        assert_eq!(
                            view.dtype(),
                            safetensors::Dtype::F32,
                            "{} passthrough must ignore --out-dtype {requested}",
                            spec.name
                        );
                        assert_bytes_eq(view.data(), &spec.data(), spec.name);
                    }
                    dtype => {
                        assert_eq!(view.dtype(), expected_st, "{}", spec.name);
                        let expected = match requested {
                            crate::Dtype::BF16 => crate::dequantize_gguf::<crate::Bf16Out>(
                                &spec.data(),
                                dtype,
                                spec.n_elements(),
                            ),
                            crate::Dtype::F32 => crate::dequantize_gguf::<crate::F32Out>(
                                &spec.data(),
                                dtype,
                                spec.n_elements(),
                            ),
                            _ => crate::dequantize_gguf::<crate::F16Out>(
                                &spec.data(),
                                dtype,
                                spec.n_elements(),
                            ),
                        }
                        .expect("oracle dequant");
                        assert_bytes_eq(
                            view.data(),
                            &expected,
                            &format!("{} at {requested}", spec.name),
                        );
                    }
                }
            }
        }
    }

    /// `F32` output really is twice the payload, and `F16` really is the same
    /// width as `BF16`.
    ///
    /// A structural check that catches a whole class of plumbing mistake: if
    /// the dtype were dropped anywhere between the option and the writer, all
    /// three payloads would collapse to one size.
    ///
    /// It sums the **dequantised tensors' payload bytes** rather than comparing
    /// file sizes, because the two are not the same thing. The safetensors JSON
    /// header spells each tensor's dtype out, so `"BF16"` and `"F16"` differ by
    /// a byte per tensor; the first draft of this test compared
    /// `fs::metadata().len()` and failed by 8 bytes on a 5.4 `MB` file for
    /// exactly that reason. Payload is the quantity the claim is about.
    #[test]
    fn output_dtype_changes_the_dequantised_payload_width() {
        let mut payloads = Vec::new();
        for requested in [crate::Dtype::BF16, crate::Dtype::F16, crate::Dtype::F32] {
            let (_dir, path, specs) = write_fixture();
            let dir_out = tempfile::tempdir().expect("tempdir");
            let out = dir_out.path().join("out.safetensors");
            convert(
                &path,
                ConvertTarget::Safetensors,
                &out,
                &ConvertOptions::new().with_output_dtype(requested),
            )
            .expect("convert");

            let written = std::fs::read(&out).expect("read output");
            let tensors = safetensors::SafeTensors::deserialize(&written).expect("parse output");
            let dequantised: usize = specs
                .iter()
                .filter(|s| s.dtype != GgufType::F32)
                .map(|s| tensors.tensor(s.name).expect("tensor present").data().len())
                .sum();
            payloads.push(dequantised);
        }
        assert_eq!(
            payloads[0], payloads[1],
            "BF16 and F16 are both 2 bytes per element"
        );
        assert_eq!(
            payloads[2],
            payloads[0] * 2,
            "F32 is exactly twice BF16: {payloads:?}"
        );
    }

    /// Determinism is preserved at **every** output dtype, not just the default.
    ///
    /// `CONVENTIONS.md` requires byte-identical output across thread counts for
    /// every parallelised path. Phase 7.3 adds a second axis to that path, so
    /// the guarantee is re-established per dtype rather than assumed to carry
    /// over from the `BF16` suite.
    #[test]
    fn output_dtype_is_deterministic_across_thread_counts() {
        for requested in [crate::Dtype::BF16, crate::Dtype::F32, crate::Dtype::F16] {
            let (_dir, path, _specs) = write_fixture();
            let dir_out = tempfile::tempdir().expect("tempdir");

            let mut baseline: Option<Vec<u8>> = None;
            for threads in [1usize, 2, 4, 8] {
                let out = dir_out.path().join(format!("out-{requested}-{threads}.st"));
                convert(
                    &path,
                    ConvertTarget::Safetensors,
                    &out,
                    &ConvertOptions::new()
                        .with_threads(threads)
                        .with_output_dtype(requested),
                )
                .expect("convert");
                let bytes = std::fs::read(&out).expect("read output");
                match &baseline {
                    None => baseline = Some(bytes),
                    Some(expected) => assert_bytes_eq(
                        &bytes,
                        expected,
                        &format!("{requested} at {threads} threads vs 1 thread"),
                    ),
                }
            }
        }
    }

    /// A derived output filename names the dtype the file actually holds.
    ///
    /// Regression guard for a real defect this phase introduced and then fixed:
    /// `ConvertTarget::suffix()` returns `"bf16"` for the safetensors target,
    /// which was correct while `BF16` was the only possible answer. Adding
    /// `--out-dtype` without touching the derivation made
    /// `--out-dtype f32` write `F32` tensors into `model-bf16.safetensors`,
    /// which is worse than an unhelpful name because it is an actively wrong
    /// one.
    ///
    /// The `gguf` and `bnb-nf4` targets keep their own suffix, because there it
    /// names a container or an encoding rather than an element type.
    #[test]
    fn derived_output_path_names_the_dtype_it_holds() {
        let input = std::path::Path::new("/models/smollm2.gguf");

        for (dtype, expected) in [
            (crate::Dtype::BF16, "smollm2-bf16.safetensors"),
            (crate::Dtype::F32, "smollm2-f32.safetensors"),
            (crate::Dtype::F16, "smollm2-f16.safetensors"),
        ] {
            let path = derive_output_path_for_dtype(input, ConvertTarget::Safetensors, dtype);
            assert_eq!(
                path.file_name().and_then(|s| s.to_str()),
                Some(expected),
                "safetensors target at {dtype}"
            );
        }

        // The no-dtype entry point keeps its historical behaviour exactly.
        assert_eq!(
            derive_output_path(input, ConvertTarget::Safetensors)
                .file_name()
                .and_then(|s| s.to_str()),
            Some("smollm2-bf16.safetensors"),
        );

        // Non-safetensors targets are unaffected by the output dtype.
        for dtype in [crate::Dtype::BF16, crate::Dtype::F32] {
            assert_eq!(
                derive_output_path_for_dtype(input, ConvertTarget::Gguf, dtype)
                    .file_name()
                    .and_then(|s| s.to_str()),
                Some("smollm2-gguf.gguf"),
                "gguf target must ignore the dequant dtype"
            );
        }
    }

    /// A dtype that is not a dequantisation output width is refused at the
    /// boundary, with a message that lists what is accepted.
    #[test]
    fn unsupported_output_dtype_is_rejected() {
        let (_dir, path, _specs) = write_fixture();
        let dir_out = tempfile::tempdir().expect("tempdir");
        let out = dir_out.path().join("out.safetensors");

        let err = convert(
            &path,
            ConvertTarget::Safetensors,
            &out,
            &ConvertOptions::new().with_output_dtype(crate::Dtype::I64),
        )
        .expect_err("I64 is not an output width");
        let msg = err.to_string();
        assert!(
            msg.contains("bf16, f32, f16"),
            "error should list the supported widths, got: {msg}"
        );
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
            drop(read_hub(&input, &options, &mut || {}).expect("warm-up read_hub"));

            let mut samples: Vec<f64> = Vec::with_capacity(SAMPLES);
            for _ in 0..SAMPLES {
                let start = Instant::now();
                let hub = read_hub(&input, &options, &mut || {}).expect("read_hub");
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
