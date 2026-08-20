// SPDX-License-Identifier: MIT OR Apache-2.0

//! High-level parse-first API.
//!
//! [`parse`] memory-maps a `.safetensors` file, returning a
//! [`ParsedModel`] that holds the parsed header metadata and the file's
//! bytes. All subsequent operations ([`ParsedModel::inspect`],
//! [`ParsedModel::remember`]) work from this parsed representation — no
//! second open, no eager copy. On the memory-mapped path the kernel pages
//! bytes in lazily on access, so `inspect()` on a multi-GB shard only faults
//! the header (~1 MiB).
//!
//! # Trusted vs untrusted input
//!
//! [`parse`] / [`parse_with_limits`] memory-map the file — the
//! **trusted-local-file fast path**. A memory map can fault with `SIGBUS` if
//! the file is truncated or written concurrently, an OS signal the caller
//! cannot catch. For **untrusted input** (a user upload, a network / FUSE
//! path) prefer the copy-based [`parse_bytes`] / [`parse_from_reader`] entry
//! points: they read the artefact into an owned buffer (bounded by
//! [`ParseLimits`]), parse with no mmap and no `unsafe`, and fail with a clean
//! `Err` rather than a `SIGBUS`.

use std::fmt;
use std::path::Path;
use std::str::FromStr;

use crate::ParseLimits;
use crate::backing::Backing;
use crate::error::AnamnesisError;
use crate::inspect::{InspectInfo, InspectOptions};
use crate::parse::safetensors::{
    Dtype, QuantScheme, SafetensorsHeader, TensorEntry, TensorRole,
    parse_safetensors_header_with_limits,
};
use crate::parse::utils::checked_num_elements;
#[cfg(feature = "awq")]
use crate::remember::awq::dequantize_awq;
#[cfg(feature = "bnb")]
use crate::remember::bnb::{dequantize_bnb_int8, dequantize_bnb4, dequantize_bnb4_double_quant};
use crate::remember::fp8::{dequantize_fp8, dequantize_per_channel_fp8, dequantize_per_tensor_fp8};
#[cfg(feature = "gptq")]
use crate::remember::gptq::dequantize_gptq;
use crate::remember::output::{Bf16Out, F16Out, F32Out, OutputElement};
#[cfg(any(feature = "gptq", feature = "awq"))]
use crate::remember::quant_utils::transpose_elements;

/// Target dtype for dequantization output.
///
/// # Choosing a width
///
/// [`BF16`](Self::BF16) is the dtype the safetensors / Hugging Face ecosystem
/// serves weights in, and at 2 bytes per element it halves the memory traffic on
/// a path that is bandwidth-bound end to end. It is the default and, before
/// v0.7.4, was the only option.
///
/// It is, however, **lossy relative to the exact dequantised value**, and that is
/// worth stating plainly: a `Q8_0` value is an `f16` scale (11-bit significand)
/// times an `int8`, needing up to ~18 bits, while `BF16` holds 8. Measured on
/// `SmolLM2-135M-Q4_K_M`, only **3–20 %** of dequantised values are exactly
/// `BF16`-representable; the rest are rounded, at up to half a `BF16` `ULP`
/// (`2⁻⁸` ≈ 0.39 % relative). The crate's "bit-exact, 0 `ULP`" claim is therefore
/// scoped to *the reference rounded to `BF16`* — which is how the `BF16`
/// fixtures are built — not to the true value, which needs
/// [`F32`](Self::F32).
///
/// Every kernel in the crate computes in `f32` and narrows once at the end, so
/// [`F32`](Self::F32) is not extra work: it is the *absence* of the narrowing
/// step, at double the output bytes.
///
/// # Passthrough policy
///
/// This selects the width for **dequantised** tensors only. Tensors that pass
/// through untouched (norms, embeddings, anything already unquantised) keep
/// their source dtype, so a `remember` output is legitimately mixed-dtype.
/// `TargetDtype::F32` is a request to stop narrowing, not an instruction to
/// rewrite every tensor as `F32`. `ConvertOptions::output_dtype` documents the
/// same policy for the `convert` path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TargetDtype {
    /// `BF16` (bfloat16) — 2 bytes per element, round-to-nearest-even. The
    /// standard research/training dtype and the default (see the type-level
    /// docs for the precision trade-off this implies).
    BF16,
    /// `F32` — 4 bytes per element, and **no narrowing step at all**. The
    /// kernels already compute in `f32`, so this emits the value they computed,
    /// bit-identical to the reference implementation's own `f32`. Doubles
    /// output bytes on a bandwidth-bound path; that is the honest cost of the
    /// precision, not a defect.
    F32,
    /// `F16` — 2 bytes per element, IEEE 754 binary16, round-to-nearest-even.
    ///
    /// **Not uniformly the better 2-byte choice.** Against `BF16` it buys 3
    /// significand bits (11 versus 8) and pays a far narrower exponent range:
    /// `BF16` shares `f32`'s range, while `F16` saturates at 65504 and flushes
    /// to zero below about `2⁻²⁴`. Out-of-range values follow plain IEEE
    /// semantics (infinity, flush-to-zero), never saturation — see
    /// [`F16Out`] for why.
    F16,
}

impl TargetDtype {
    /// Bytes each dequantised element occupies in the output.
    ///
    /// The runtime mirror of `OutputElement::BYTES`, for the callers that need
    /// the width as a value rather than as a type parameter (size estimates,
    /// header sizing, the CLI's reporting).
    #[must_use]
    pub const fn byte_size(self) -> usize {
        match self {
            Self::BF16 | Self::F16 => 2,
            Self::F32 => 4,
        }
    }

    /// The element [`Dtype`] this output width names.
    ///
    /// `TargetDtype` and [`Dtype`] describe the same three widths from two
    /// directions: one is what a caller *asks* for, the other is what a tensor
    /// *is*. The `remember` paths speak the first and the `convert` paths the
    /// second, so without this the two would each need their own three-arm
    /// dispatch to the same three monomorphisations — which is exactly the
    /// duplication Phase 7.6 exists to remove.
    ///
    /// Total and infallible: every `TargetDtype` is an output width by
    /// construction, which is the point of the type.
    ///
    /// Gated on `gguf` because that is where the two dispatch styles meet; a
    /// build without it has only one and needs no bridge.
    #[cfg(feature = "gguf")]
    #[must_use]
    pub(crate) const fn as_dtype(self) -> Dtype {
        match self {
            Self::BF16 => Dtype::BF16,
            Self::F32 => Dtype::F32,
            Self::F16 => Dtype::F16,
        }
    }
}

impl fmt::Display for TargetDtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Wildcard-free on purpose: `src/cli.rs`'s `derive_output_path` builds
        // the output filename's dtype suffix from this string, so a new variant
        // that fell through to a catch-all would silently produce a wrongly
        // named file rather than failing to compile.
        match self {
            Self::BF16 => f.write_str("BF16"),
            Self::F32 => f.write_str("F32"),
            Self::F16 => f.write_str("F16"),
        }
    }
}

impl FromStr for TargetDtype {
    type Err = AnamnesisError;

    /// Parses a target dtype from a case-insensitive string.
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::Unsupported`] if the string does not match
    /// a known target dtype.
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "bf16" => Ok(Self::BF16),
            "f32" => Ok(Self::F32),
            "f16" => Ok(Self::F16),
            other => Err(AnamnesisError::Unsupported {
                format: other.to_owned(),
                detail: "supported target dtypes: bf16, f32, f16".to_owned(),
            }),
        }
    }
}

/// A parsed `.safetensors` model, holding parsed header metadata and the
/// file's bytes.
///
/// Created by [`parse`] (memory-mapped) or by [`parse_bytes`] /
/// [`parse_from_reader`] (owned copy). The bytes are reached through a `&[u8]`
/// regardless of backing, so both paths share every method
/// ([`ParsedModel::inspect`], [`ParsedModel::remember`]) and the backing is
/// invisible to callers. On the memory-mapped path the kernel pages bytes in
/// lazily on access, so `inspect()` on a multi-GB shard only faults in the
/// header (~1 MiB).
pub struct ParsedModel {
    /// Parsed header metadata (tensor names, dtypes, shapes, roles, scheme).
    pub header: SafetensorsHeader,
    /// File bytes — either a memory map (path-based [`parse`]) or an owned
    /// `Vec<u8>` (copy-based [`parse_bytes`] / [`parse_from_reader`]). Tensor
    /// data starts at offset `header_size + 8`. On the mmap path the OS pages
    /// bytes in lazily on access, so `parse()` + `inspect()` on a multi-GB
    /// shard touches only the header (~1 MiB) instead of materialising the
    /// whole file.
    buffer: Backing,
}

/// Parses a `.safetensors` file, returning a [`ParsedModel`] holding both
/// header metadata and the file bytes (memory-mapped).
///
/// This is the entry point for all anamnesis operations. The file is
/// memory-mapped once; all subsequent operations
/// ([`ParsedModel::inspect`], [`ParsedModel::remember`]) work from the
/// mmap, so tensor pages are paged in lazily on access.
///
/// # Errors
///
/// Returns [`AnamnesisError::Io`] if the file cannot be opened or
/// mapped.
/// Returns [`AnamnesisError::Parse`] if the safetensors header is
/// malformed.
/// Returns [`AnamnesisError::LimitExceeded`] if the declared header exceeds the
/// permanent 100 MiB cap (`MAX_SAFETENSORS_HEADER_BYTES`, always-on).
///
/// # Memory
///
/// Uses `memmap2::Mmap` so the file's bytes do not occupy heap. The
/// kernel pages bytes in on access and may drop them under memory
/// pressure — which means a 70 GiB shard can be inspected on a 32 GiB
/// machine without `OOM`ing the way a `Vec<u8>` allocation would.
/// `parse()` + `inspect()` only touches the header (~1 MiB), so the
/// resident-set growth on inspect-only workflows is bounded by the
/// header size, not the file size.
pub fn parse(path: impl AsRef<Path>) -> crate::Result<ParsedModel> {
    parse_with_limits(path, &ParseLimits::default())
}

/// Parses a `.safetensors` file under a caller-supplied [`ParseLimits`] budget.
///
/// Identical to [`parse`] but enforces every applicable [`ParseLimits`] ceiling
/// (the per-allocation and cumulative-byte budgets — see [`ParseLimits`] for the
/// axes) fail-fast, before the header is allocated. The built-in 100 MiB header
/// cap still applies; `limits` can only tighten it. [`parse`] is the
/// `ParseLimits::default()` (unbounded) special case.
///
/// # Errors
///
/// Returns [`AnamnesisError::Io`] if the file cannot be opened or mapped.
/// Returns [`AnamnesisError::LimitExceeded`] if the declared header size exceeds
/// `limits`.
/// Returns [`AnamnesisError::Parse`] if the safetensors header is malformed.
///
/// # Memory
///
/// Uses `memmap2::Mmap` so the file's bytes do not occupy heap; `parse()` +
/// `inspect()` only touches the header. See [`parse`] for the full rationale.
#[allow(unsafe_code)]
pub fn parse_with_limits(
    path: impl AsRef<Path>,
    limits: &ParseLimits,
) -> crate::Result<ParsedModel> {
    let file = std::fs::File::open(path.as_ref())?;
    // SAFETY: `memmap2::Mmap` requires `unsafe` because the OS could
    // modify the mapped region if another process writes to the
    // underlying file concurrently. Tensor files are read-only artefacts
    // in practice — the same assumption every other tensor parser in this
    // crate (`parse_pth`, `parse_gguf`) and the upstream `safetensors`
    // crate's mmap path rely on. The mapping is released when the
    // returned `ParsedModel` is dropped. Untrusted callers that cannot
    // make the read-only-artefact assumption use `parse_bytes` /
    // `parse_from_reader` instead (no mmap, no `SIGBUS`).
    let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(AnamnesisError::Io)?;
    parsed_model_from_backing(Backing::Mmap(mmap), limits)
}

/// Builds a [`ParsedModel`] from an already-acquired byte backing — the single
/// construction site shared by the mmap path ([`parse_with_limits`]) and the
/// copy-based paths ([`parse_bytes_with_limits`] /
/// [`parse_from_reader_with_limits`]), so the header parse cannot drift between
/// them.
fn parsed_model_from_backing(buffer: Backing, limits: &ParseLimits) -> crate::Result<ParsedModel> {
    let header = parse_safetensors_header_with_limits(&buffer, limits)?;
    Ok(ParsedModel { header, buffer })
}

/// Parses `.safetensors` bytes already held in memory, returning a
/// [`ParsedModel`] that **owns** them — the copy-based, mmap-free path.
///
/// This is the **recommended entry point for untrusted input** (a user upload,
/// bytes received over the network): unlike [`parse`], it never memory-maps, so
/// a truncated or concurrently-written source cannot fault the process with a
/// `SIGBUS`; a malformed input is a clean `Err`. [`parse_bytes`] is the
/// [`ParseLimits::default`] (unbounded) special case of
/// [`parse_bytes_with_limits`].
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if the safetensors header is malformed.
/// Returns [`AnamnesisError::LimitExceeded`] if the declared header exceeds the
/// permanent 100 MiB cap (`MAX_SAFETENSORS_HEADER_BYTES`, always-on — reachable
/// even at the default limits this wrapper passes).
///
/// # Memory
///
/// Takes ownership of `bytes` (no copy) and holds them for the
/// [`ParsedModel`]'s lifetime — peak heap is the input size. Contrast [`parse`],
/// which memory-maps and pages lazily.
pub fn parse_bytes(bytes: Vec<u8>) -> crate::Result<ParsedModel> {
    parse_bytes_with_limits(bytes, &ParseLimits::default())
}

/// Parses owned `.safetensors` bytes under a caller-supplied [`ParseLimits`]
/// budget — the bounded, mmap-free path for untrusted input.
///
/// Rejects an input larger than [`ParseLimits::max_single_alloc_bytes`] before
/// parsing, then enforces every applicable [`ParseLimits`] ceiling on the header
/// exactly as [`parse_with_limits`] does.
///
/// # Errors
///
/// Returns [`AnamnesisError::LimitExceeded`] if `bytes` exceeds `limits`.
/// Returns [`AnamnesisError::Parse`] if the safetensors header is malformed.
///
/// # Memory
///
/// Takes ownership of `bytes` (no copy); peak heap is the input size.
pub fn parse_bytes_with_limits(bytes: Vec<u8>, limits: &ParseLimits) -> crate::Result<ParsedModel> {
    let len = u64::try_from(bytes.len()).map_err(|_| AnamnesisError::Parse {
        reason: "safetensors bytes: length overflows u64".into(),
    })?;
    limits.check_alloc(len, "safetensors bytes")?;
    parsed_model_from_backing(Backing::Owned(bytes), limits)
}

/// Parses a `.safetensors` artefact from any reader, returning a [`ParsedModel`]
/// that **owns** the bytes — the copy-based, mmap-free path.
///
/// The **recommended entry point for untrusted streamed input**: the whole
/// stream is read into an owned buffer (bounded by [`ParseLimits`]) and parsed
/// with no mmap, so a truncated or hostile stream is a clean `Err`, never a
/// `SIGBUS`. [`parse_from_reader`] is the [`ParseLimits::default`] (unbounded)
/// special case of [`parse_from_reader_with_limits`].
///
/// # Errors
///
/// Returns [`AnamnesisError::Io`] if the reader fails.
/// Returns [`AnamnesisError::Parse`] if the safetensors header is malformed.
/// Returns [`AnamnesisError::LimitExceeded`] if the declared header exceeds the
/// permanent 100 MiB cap (`MAX_SAFETENSORS_HEADER_BYTES`, always-on — reachable
/// even at the default limits this wrapper passes).
///
/// # Memory
///
/// Reads the entire stream into an owned `Vec<u8>`; peak heap is the artefact
/// size.
pub fn parse_from_reader<R: std::io::Read>(reader: R) -> crate::Result<ParsedModel> {
    parse_from_reader_with_limits(reader, &ParseLimits::default())
}

/// Parses a `.safetensors` artefact from any reader under a caller-supplied
/// [`ParseLimits`] budget — the bounded, mmap-free path for untrusted input.
///
/// The read is bounded by [`ParseLimits::max_single_alloc_bytes`] so an
/// unbounded or hostile stream cannot exhaust memory; the header is then parsed
/// under the same `limits` as [`parse_with_limits`].
///
/// # Errors
///
/// Returns [`AnamnesisError::Io`] if the reader fails.
/// Returns [`AnamnesisError::LimitExceeded`] if the bytes read exceed `limits`.
/// Returns [`AnamnesisError::Parse`] if the safetensors header is malformed.
///
/// # Memory
///
/// Reads the stream into an owned `Vec<u8>` of at most
/// `max_single_alloc_bytes + 1` bytes; peak heap is the artefact size.
pub fn parse_from_reader_with_limits<R: std::io::Read>(
    reader: R,
    limits: &ParseLimits,
) -> crate::Result<ParsedModel> {
    let bytes = limits.read_to_vec_bounded(reader, "safetensors file")?;
    parse_bytes_with_limits(bytes, limits)
}

/// Owned dequantised tensors produced by [`ParsedModel::dequantize_all`]:
/// `(output name, `BF16` bytes, output shape)`.
type DequantizedTensors = Vec<(String, Vec<u8>, Vec<usize>)>;

/// Passthrough tensors that borrow `self.buffer`: `(name, bytes, shape)`. Tied
/// to the `ParsedModel` borrow they were collected under.
type PassthroughRefs<'a> = Vec<(&'a str, &'a [u8], &'a [usize])>;

/// One passthrough tensor ref tagged with its original header index, so the
/// parallel dequant can merge passthroughs back into header order deterministically.
type IndexedPassthrough<'a> = (usize, (&'a str, &'a [u8], &'a [usize]));

/// The dequantisation outcome for a single `TensorRole::Quantized` entry,
/// produced by [`ParsedModel::dequantize_quantized_entry`]. Deliberately a pure
/// value (no borrow of shared state) so it can be computed on a worker thread
/// and moved back to the main thread for deterministic, index-ordered assembly.
enum TensorDequant {
    /// Owned dequantised output: `(output name, `BF16` bytes, output shape)`.
    /// Every real dequant scheme produces this.
    Owned(String, Vec<u8>, Vec<usize>),
    /// The header scheme was `Unquantized` (a defensive edge case: a
    /// `Quantized`-role tensor in an unquantized model). The orchestrator
    /// resolves it to a passthrough reference on the main thread.
    Passthrough,
}

/// Resolves an optional caller thread request to a concrete dequantisation
/// worker budget.
///
/// `None` → `min(available_parallelism, 4)` — the measured scaling knee for
/// bandwidth-bound dequant (`docs/perf-experiments.md` Experiment 11), leaving
/// the rest of the host's cores free for the embedding process. `Some(n)` pins
/// the budget to `n.max(1)`. The budget is derived only from hardware and the
/// caller's request — **never** from any file-declared quantity — per the
/// `CONVENTIONS.md` "caller owns the thread budget" rule.
#[cfg(feature = "parallel")]
pub(crate) fn resolve_thread_budget(threads: Option<usize>) -> usize {
    match threads {
        None => std::thread::available_parallelism()
            .map_or(1, std::num::NonZeroUsize::get)
            .min(4),
        Some(n) => n.max(1),
    }
}

/// Resolves an optional caller thread request to a concrete dequantisation
/// worker budget.
///
/// With the `parallel` feature disabled the dequant path is always sequential,
/// so the budget is fixed at 1 regardless of the request.
#[cfg(not(feature = "parallel"))]
pub(crate) fn resolve_thread_budget(_threads: Option<usize>) -> usize {
    1
}

/// Caller-supplied options for the `remember` family of methods.
///
/// Currently carries only the per-tensor dequantisation thread budget; the
/// `#[non_exhaustive]` attribute lets future knobs be added without a breaking
/// change. Construct with [`RememberOptions::new`] (or
/// [`RememberOptions::default`], which is identical) and chain the setters:
///
/// ```rust
/// use anamnesis::RememberOptions;
///
/// let opts = RememberOptions::new().with_threads(8);
/// assert_eq!(opts.threads, Some(8));
/// ```
///
/// The builder shape deliberately mirrors
/// [`ConvertOptions`](crate::ConvertOptions), which carries the same `threads`
/// knob: one spelling for one concept across the two option types.
#[non_exhaustive]
#[derive(Debug, Clone, Default)]
pub struct RememberOptions {
    /// Number of worker threads for per-tensor dequantisation.
    ///
    /// `None` (the default) resolves to `min(available_parallelism, 4)` — the
    /// measured scaling knee for bandwidth-bound dequant, leaving the host's
    /// remaining cores free. `Some(n)` pins the budget to `n.max(1)`. With the
    /// `parallel` Cargo feature disabled the budget is always 1 (fully
    /// sequential) regardless of this field.
    pub threads: Option<usize>,
    /// Cooperative cancellation handle, polled once per tensor.
    ///
    /// `None` (the default) means the run cannot be cancelled and costs
    /// nothing: no token is allocated and the poll is a `None` check. `Some`
    /// makes the run stop at the next tensor boundary once
    /// [`CancelToken::cancel`](crate::CancelToken::cancel) is called from any
    /// thread, returning [`AnamnesisError::Cancelled`] with no output file
    /// written.
    pub cancel: Option<crate::CancelToken>,
}

impl RememberOptions {
    /// Returns options with the built-in defaults (the
    /// `min(available_parallelism, 4)` thread budget).
    ///
    /// `const`: the struct is a single `Option<usize>`, so there is nothing to
    /// allocate.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            threads: None,
            cancel: None,
        }
    }

    /// Sets the per-tensor dequantisation thread budget (clamped to at least 1).
    /// Overrides the `min(available_parallelism, 4)` default; ignored when the
    /// `parallel` feature is off (always sequential).
    ///
    /// Chained from [`RememberOptions::new`] or
    /// [`RememberOptions::default`] — the same builder shape
    /// [`ConvertOptions::with_threads`](crate::ConvertOptions::with_threads)
    /// uses:
    ///
    /// ```rust
    /// use anamnesis::RememberOptions;
    ///
    /// let opts = RememberOptions::new().with_threads(2);
    /// assert_eq!(opts.threads, Some(2));
    /// ```
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

    #[must_use]
    pub fn with_threads(mut self, n: usize) -> Self {
        self.threads = Some(n.max(1));
        self
    }

    /// Resolves the configured request to a concrete worker count (see
    /// [`resolve_thread_budget`]). Consumes the options (the builder is spent
    /// once its budget has been read).
    #[must_use]
    pub(crate) fn resolved_threads(self) -> usize {
        resolve_thread_budget(self.threads)
    }
}

impl ParsedModel {
    /// Returns inspection info (format, tensor counts, size estimates), sizing
    /// the dequantised estimate for the default `BF16` output.
    ///
    /// The [`InspectOptions::default`] special case of
    /// [`inspect_with_options`](Self::inspect_with_options), mirroring how
    /// [`remember`](Self::remember) relates to
    /// [`remember_with_options`](Self::remember_with_options). No I/O — purely
    /// derived from the parsed header.
    pub fn inspect(&self) -> InspectInfo {
        self.inspect_with_options(&InspectOptions::new())
    }

    /// Returns inspection info with a caller-supplied [`InspectOptions`].
    ///
    /// The reason to reach for this over [`inspect`](Self::inspect) is
    /// [`InspectInfo::dequantized_size`], which feeds the inspect-before-parse
    /// policy gate. That figure is only meaningful against a specific output
    /// width, so a caller who intends `remember(.., TargetDtype::F32)` should
    /// ask for the `F32` estimate rather than doubling the `BF16` one by hand:
    ///
    /// ```rust,no_run
    /// use anamnesis::{InspectOptions, TargetDtype, parse};
    ///
    /// let model = parse("model-fp8.safetensors")?;
    /// let info = model.inspect_with_options(
    ///     &InspectOptions::new().with_output_dtype(TargetDtype::F32),
    /// );
    /// // `info.dequantized_size` now sizes an F32 request, and
    /// // `info.output_dtype` records which width it assumed.
    /// # Ok::<(), anamnesis::AnamnesisError>(())
    /// ```
    ///
    /// No I/O — purely derived from the parsed header.
    pub fn inspect_with_options(&self, options: &InspectOptions) -> InspectInfo {
        InspectInfo::with_options(&self.header, options)
    }

    /// Returns the raw bytes for a tensor from the memory-mapped file
    /// buffer. The slice borrows from the mmap; pages are paged in by
    /// the kernel on first access.
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::Parse`] if the tensor's data offsets are
    /// out of bounds.
    fn tensor_data(&self, start: usize, end: usize) -> crate::Result<&[u8]> {
        let data_offset = self.header.header_size + 8;
        let abs_start = data_offset
            .checked_add(start)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: "tensor data start offset overflow".into(),
            })?;
        let abs_end = data_offset
            .checked_add(end)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: "tensor data end offset overflow".into(),
            })?;
        self.buffer
            .get(abs_start..abs_end)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!(
                    "tensor data offsets {abs_start}..{abs_end} out of bounds (buffer len {})",
                    self.buffer.len()
                ),
            })
    }

    /// Reads a scalar scale value from raw bytes, handling both `F32` and
    /// `BF16` scale dtypes.
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::Parse`] if the data is too short for the
    /// given dtype.
    /// Returns [`AnamnesisError::Unsupported`] if the scale dtype is not
    /// `F32` or `BF16`.
    fn read_scalar_scale(data: &[u8], dtype: Dtype, weight_name: &str) -> crate::Result<f32> {
        match dtype {
            Dtype::F32 => {
                let arr: [u8; 4] =
                    data.get(..4)
                        .and_then(|s| s.try_into().ok())
                        .ok_or_else(|| AnamnesisError::Parse {
                            reason: format!(
                                "per-tensor F32 scale for `{weight_name}` is not 4 bytes"
                            ),
                        })?;
                Ok(f32::from_le_bytes(arr))
            }
            Dtype::BF16 => {
                let arr: [u8; 2] =
                    data.get(..2)
                        .and_then(|s| s.try_into().ok())
                        .ok_or_else(|| AnamnesisError::Parse {
                            reason: format!(
                                "per-tensor BF16 scale for `{weight_name}` is not 2 bytes"
                            ),
                        })?;
                // BITWISE: BF16 → f32 by shifting into upper 16 bits of IEEE 754
                Ok(f32::from_bits(u32::from(u16::from_le_bytes(arr)) << 16))
            }
            Dtype::F16 => {
                let arr: [u8; 2] =
                    data.get(..2)
                        .and_then(|s| s.try_into().ok())
                        .ok_or_else(|| AnamnesisError::Parse {
                            reason: format!(
                                "per-tensor F16 scale for `{weight_name}` is not 2 bytes"
                            ),
                        })?;
                // BITWISE: F16 → f32 via half crate's IEEE 754 conversion
                Ok(half::f16::from_le_bytes(arr).to_f32())
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
                format: dtype.to_string(),
                detail: format!("per-tensor scale for `{weight_name}` has unsupported dtype"),
            }),
        }
    }

    /// Extracts `(rows, cols)` from a tensor shape for the fine-grained
    /// dequantization function.
    ///
    /// - 2D: `(shape[0], shape[1])`
    /// - >2D: `(product of all dims except last, last dim)`
    fn shape_to_rows_cols(shape: &[usize]) -> crate::Result<(usize, usize)> {
        match shape.len() {
            0 | 1 => Err(AnamnesisError::Parse {
                reason: format!(
                    "quantized tensor has {}-D shape, expected >= 2D",
                    shape.len()
                ),
            }),
            2 => {
                // INDEX: shape.len() == 2 guaranteed by match arm
                #[allow(clippy::indexing_slicing)]
                Ok((shape[0], shape[1]))
            }
            _ => {
                // shape.len() >= 3 guaranteed by match arms above
                let cols = shape.last().copied().ok_or_else(|| AnamnesisError::Parse {
                    reason: "shape has no last dimension".into(),
                })?;
                let leading =
                    shape
                        .get(..shape.len() - 1)
                        .ok_or_else(|| AnamnesisError::Parse {
                            reason: "shape slice out of bounds".into(),
                        })?;
                let rows = checked_num_elements(leading).ok_or_else(|| AnamnesisError::Parse {
                    reason: "shape row-count product overflows usize".into(),
                })?;
                Ok((rows, cols))
            }
        }
    }

    /// Dequantizes all quantized tensors and writes a standard `.safetensors`
    /// file loadable by any Rust ML framework.
    ///
    /// See [`remember_to_bytes`](Self::remember_to_bytes) for the in-memory
    /// variant that returns the bytes instead of writing a file.
    ///
    /// - **Quantized tensors**: dequantized to the target dtype using the
    ///   detected quantization scheme and companion scale factors. `GPTQ` /
    ///   `AWQ` projection weights are additionally transposed from the
    ///   GEMM-native `[in_features, out_features]` kernel orientation to the
    ///   standard `nn.Linear` `[out_features, in_features]` — the layout a
    ///   standard consumer (candle, `transformers` as a plain model)
    ///   expects, and the same boundary transpose `GPTQModel`'s
    ///   `dequantize_model` applies. `BnB` and `FP8` weights are already
    ///   stored / recovered in standard orientation.
    /// - **Scale tensors**: consumed during dequantization, not written.
    /// - **Passthrough tensors**: copied as-is (zero-copy from the buffer).
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::Parse`] if tensor data is malformed or
    /// shapes are inconsistent.
    /// Returns [`AnamnesisError::Unsupported`] if the quantization scheme
    /// is not yet implemented.
    /// Returns [`AnamnesisError::Io`] if the output file cannot be written.
    ///
    /// # Memory
    ///
    /// Peak heap is `O(total_dequantised_output_size)`, which is
    /// `target.byte_size() × n_parameters` bytes: **`2 ×` at `BF16` or `F16`
    /// and `4 ×` at `F32`**. Passthrough tensors contribute their source bytes
    /// either way, so an `F32` request does not double the whole file, only the
    /// dequantised share.
    ///
    /// The input file is memory-mapped — pages are paged in by the kernel on
    /// access and may be dropped under memory pressure — so the input side does
    /// not contribute to the heap. **Every dequantised tensor's `Vec<u8>` is
    /// retained simultaneously** until the underlying
    /// `safetensors::serialize_to_file` call returns: the safetensors crate's
    /// writer itself streams tensor bodies one at a time, but the eager
    /// buffering happens in this method's caller-side `Vec` collection. The
    /// `GPTQ` / `AWQ` orientation transpose holds one extra tensor-sized buffer
    /// transiently (per tensor, dropped immediately) — the peak class is
    /// unchanged, though at `F32` that transient is itself `4 ×` wider.
    ///
    /// At `BF16`: comfortable for `≤ 7 B` models on a 32 GB system; tight at
    /// 13 B; `OOM`s at 70 B+. **At `F32` halve each of those thresholds.** A
    /// streaming output path (planned ROADMAP Phase 10) will drop this to
    /// `O(largest_tensor × target.byte_size())`.
    ///
    /// The per-kernel share of this claim is asserted to the byte by
    /// `tests/peak_heap_{awq,gptq,bnb_dq,gguf}.rs`, which since v0.7.4 run at
    /// every output width rather than only `BF16`.
    pub fn remember(
        &self,
        output_path: impl AsRef<Path>,
        target: TargetDtype,
    ) -> crate::Result<()> {
        self.remember_with_options(output_path, target, RememberOptions::default())
    }

    /// Dequantizes all quantized tensors and writes a standard `.safetensors`
    /// file, with a caller-supplied [`RememberOptions`] (currently the dequant
    /// thread budget).
    ///
    /// Behaves identically to [`remember`](Self::remember) — the output bytes
    /// are byte-identical for any thread count — but lets the caller tune the
    /// per-tensor dequantisation parallelism. The default
    /// ([`RememberOptions::default`]) uses `min(available_parallelism, 4)`
    /// workers (1 when the `parallel` feature is off).
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::Parse`] if tensor data is malformed or
    /// shapes are inconsistent, or if a dequant worker thread panics.
    /// Returns [`AnamnesisError::Unsupported`] if the quantization scheme
    /// is not yet implemented.
    /// Returns [`AnamnesisError::Io`] if the output file cannot be written.
    pub fn remember_with_options(
        &self,
        output_path: impl AsRef<Path>,
        target: TargetDtype,
        opts: RememberOptions,
    ) -> crate::Result<()> {
        self.remember_with_progress_and_options(output_path, target, opts, || {})
    }

    /// Dequantizes all quantized tensors with per-tensor progress reporting,
    /// and writes a standard `.safetensors` file loadable by any Rust ML
    /// framework.
    ///
    /// Behaves identically to [`remember`](Self::remember), but calls
    /// `on_tensor` after each quantized tensor is dequantized. Use this to
    /// drive a progress bar in CLI contexts.
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::Parse`] if tensor data is malformed or
    /// shapes are inconsistent.
    /// Returns [`AnamnesisError::Unsupported`] if the quantization scheme
    /// is not yet implemented.
    /// Returns [`AnamnesisError::Io`] if the output file cannot be written.
    pub fn remember_with_progress<F>(
        &self,
        output_path: impl AsRef<Path>,
        target: TargetDtype,
        on_tensor: F,
    ) -> crate::Result<()>
    where
        F: FnMut(),
    {
        self.remember_with_progress_and_options(
            output_path,
            target,
            RememberOptions::default(),
            on_tensor,
        )
    }

    /// Dequantizes all quantized tensors with per-tensor progress reporting **and**
    /// a caller-supplied [`RememberOptions`].
    ///
    /// The two-knob form of [`remember_with_progress`](Self::remember_with_progress):
    /// before v0.7.2 a caller had to choose between a progress bar and a thread
    /// budget, because the progress variant pinned the budget to its default.
    /// `on_tensor` still fires on the **calling** thread only, so an `FnMut`
    /// closing over a progress bar never crosses a thread boundary.
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::Parse`] if tensor data is malformed or
    /// shapes are inconsistent.
    /// Returns [`AnamnesisError::Unsupported`] if the quantization scheme
    /// is not yet implemented.
    /// Returns [`AnamnesisError::Io`] if the output file cannot be written.
    pub fn remember_with_progress_and_options<F>(
        &self,
        output_path: impl AsRef<Path>,
        target: TargetDtype,
        opts: RememberOptions,
        on_tensor: F,
    ) -> crate::Result<()>
    where
        F: FnMut(),
    {
        // Split the builder into its two knobs before spending it: the
        // token is cloned out because the dispatch borrows it for the whole
        // run, and `resolved_threads` consumes what is left.
        let cancel = opts.cancel.clone();
        let cancel = cancel.as_ref();
        let threads = opts.resolved_threads();
        let path = output_path.as_ref();
        // The single runtime boundary for the file destination: past this
        // `match` the output width is a static type parameter and there is no
        // per-tensor branch, let alone a per-element one.
        match target {
            TargetDtype::BF16 => {
                self.remember_inner::<Bf16Out, F>(path, threads, cancel, on_tensor)
            }
            TargetDtype::F32 => self.remember_inner::<F32Out, F>(path, threads, cancel, on_tensor),
            TargetDtype::F16 => self.remember_inner::<F16Out, F>(path, threads, cancel, on_tensor),
        }
    }

    /// Dequantizes all quantized tensors and returns the standard `.safetensors`
    /// bytes in memory, instead of writing a file.
    ///
    /// The in-memory twin of [`remember`](Self::remember): identical dequant and
    /// companion-grouping, but returns the serialized `BF16` safetensors as a
    /// `Vec<u8>` so an embedder can load the dequantised model without a disk
    /// round-trip (e.g. candle-mi's quantized loader → `from_buffered_safetensors`).
    /// Completes the file/bytes pairing the crate's other serializers already
    /// have (`ParsedPth::to_safetensors_bytes` (requires `pth` feature),
    /// `write_bnb_nf4_safetensors_bytes`).
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::Parse`] if tensor data is malformed or
    /// shapes are inconsistent, or if serialization fails.
    /// Returns [`AnamnesisError::Unsupported`] if the quantization scheme
    /// is not yet implemented.
    ///
    /// # Memory
    ///
    /// Peak heap is **higher** than [`remember`](Self::remember)'s file path.
    /// Both dequantize every tensor into owned `Vec`s
    /// (`O(target.byte_size() × n_parameters)`, so `2 ×` at `BF16` or `F16` and
    /// `4 ×` at `F32`), but where [`remember`](Self::remember) streams those
    /// bodies to disk one at a time via `safetensors::serialize_to_file`, this
    /// method calls `safetensors::serialize`, which copies every tensor into one
    /// contiguous output buffer — so the per-tensor `Vec`s **and** the full
    /// output `Vec` are live simultaneously (~`2 ×` the dequantised set
    /// transiently) before the per-tensor `Vec`s drop.
    ///
    /// The two multipliers compound: an `F32` request through this method peaks
    /// at roughly `8 × n_parameters` against `BF16`'s `4 ×`. Comfortable for
    /// `≤ 7 B` models on a 32 GB system at `BF16`; halve that at `F32`. The
    /// streaming, peak-bounded `remember_to_writer` / `remember_to_sink`
    /// variants are planned for ROADMAP Phase 10.
    pub fn remember_to_bytes(&self, target: TargetDtype) -> crate::Result<Vec<u8>> {
        self.remember_to_bytes_with_options(target, RememberOptions::default())
    }

    /// Dequantizes all quantized tensors and returns the standard `.safetensors`
    /// bytes in memory, with a caller-supplied [`RememberOptions`] (currently the
    /// dequant thread budget).
    ///
    /// The in-memory twin of [`remember_with_options`](Self::remember_with_options):
    /// identical dequant and companion-grouping, byte-identical output for any
    /// thread count, but returns the serialized `BF16` safetensors as a `Vec<u8>`
    /// instead of writing a file.
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::Parse`] if tensor data is malformed or
    /// shapes are inconsistent, if a dequant worker thread panics, or if
    /// serialization fails.
    /// Returns [`AnamnesisError::Unsupported`] if the quantization scheme
    /// is not yet implemented.
    pub fn remember_to_bytes_with_options(
        &self,
        target: TargetDtype,
        opts: RememberOptions,
    ) -> crate::Result<Vec<u8>> {
        let cancel = opts.cancel.clone();
        let cancel = cancel.as_ref();
        let threads = opts.resolved_threads();
        // The single runtime boundary for the in-memory destination; see
        // `remember_with_progress_and_options` for the file one.
        match target {
            TargetDtype::BF16 => self.remember_to_bytes_inner::<Bf16Out>(threads, cancel),
            TargetDtype::F32 => self.remember_to_bytes_inner::<F32Out>(threads, cancel),
            TargetDtype::F16 => self.remember_to_bytes_inner::<F16Out>(threads, cancel),
        }
    }

    /// Normalises this model into [`crate::convert`]'s hub form: quantised
    /// entries dequantised to `E`, passthrough entries copied in their
    /// **original** dtype. Returns the tensors plus how many were dequantised.
    ///
    /// Shares [`Self::dequantize_all`] with the `remember` paths, so the convert
    /// hub and `remember` cannot drift apart — including on the output width,
    /// which is the same type parameter on both.
    ///
    /// # Errors
    ///
    /// Propagates the dequantisation errors of [`Self::dequantize_all`], and
    /// returns [`AnamnesisError::Parse`] if a passthrough tensor is missing from
    /// the header (which would mean the header and the walk disagree).
    ///
    /// # Memory
    ///
    /// Allocates owned copies of **every** tensor — peak heap is one full
    /// dequantised model (`O(model)`, the hub itself), which is
    /// `E::BYTES / 2 ×` the pre-v0.7.4 figure for the dequantised share. The
    /// end-to-end `convert` peak adds only the target writer's buffer; see the
    /// `convert` module docs.
    pub(crate) fn hub_tensors<E: OutputElement>(
        &self,
        threads: usize,
        cancel: Option<&crate::CancelToken>,
        on_tensor: &mut dyn FnMut(),
    ) -> crate::Result<(Vec<crate::convert::HubTensor>, usize)> {
        // TRAIT_OBJECT: the progress hook crosses several private `convert`
        // readers with different generic parameters, so it arrives as one
        // `&mut dyn FnMut()` rather than a type parameter on each of them. It
        // fires on the calling thread only, exactly as `dequantize_all`'s own
        // hook does.
        let (dequantized_data, passthrough_refs) =
            self.dequantize_all::<E, _>(threads, cancel, &mut *on_tensor)?;

        let dequantized = dequantized_data.len();
        let mut tensors = Vec::with_capacity(dequantized.saturating_add(passthrough_refs.len()));

        for (name, data, shape) in dequantized_data {
            tensors.push(crate::convert::HubTensor {
                name,
                shape,
                dtype: E::DTYPE,
                data,
            });
        }

        // One-pass name → dtype index so the passthrough loop stays O(N)
        // rather than O(passthrough × N) via a linear `find` per tensor.
        let dtype_by_name: std::collections::HashMap<&str, crate::Dtype> = self
            .header
            .tensors
            .iter()
            .map(|t| (t.name.as_str(), t.dtype))
            .collect();

        for (name, data, shape) in passthrough_refs {
            let dtype = *dtype_by_name
                .get(name)
                .ok_or_else(|| AnamnesisError::Parse {
                    reason: format!("passthrough tensor `{name}` not found in header"),
                })?;
            tensors.push(crate::convert::HubTensor {
                name: name.to_owned(),
                shape: shape.to_vec(),
                dtype,
                // BORROW: copy the buffer-borrowed bytes so the hub outlives `self`.
                data: data.to_vec(),
            });
        }

        Ok((tensors, dequantized))
    }

    /// Internal: dequantise one `TensorRole::Quantized` entry to owned `E`
    /// output.
    ///
    /// A **pure function** of `&self` + `entry`: it reads only shared-immutable
    /// header/buffer state and allocates its own output `Vec`, so it is safe to
    /// call concurrently from disjoint worker threads (each call writes only its
    /// own returned buffer — no shared mutable state). Every real scheme yields
    /// [`TensorDequant::Owned`]; the `Unquantized`-scheme edge case (a
    /// `Quantized`-role tensor in an unquantized model) yields
    /// [`TensorDequant::Passthrough`], which the orchestrator resolves on the
    /// main thread.
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::Parse`] if the entry's data or a required
    /// companion tensor is malformed or missing, and
    /// [`AnamnesisError::Unsupported`] if the scheme's Cargo feature is disabled.
    fn dequantize_quantized_entry<E: OutputElement>(
        &self,
        entry: &TensorEntry,
    ) -> crate::Result<TensorDequant> {
        let weight_data = self.tensor_data(entry.data_offsets.0, entry.data_offsets.1)?;

        let result = match self.header.scheme {
            QuantScheme::FineGrainedFp8 => {
                let scale_entry = self.header.find_scale_for(&entry.name).ok_or_else(|| {
                    AnamnesisError::Parse {
                        reason: format!(
                            "no scale tensor found for quantized weight `{}`",
                            entry.name
                        ),
                    }
                })?;
                let scale_data =
                    self.tensor_data(scale_entry.data_offsets.0, scale_entry.data_offsets.1)?;
                let (rows, cols) = Self::shape_to_rows_cols(&entry.shape)?;
                let out =
                    dequantize_fp8::<E>(weight_data, scale_data, rows, cols, scale_entry.dtype)?;
                TensorDequant::Owned(entry.name.clone(), out, entry.shape.clone())
            }
            QuantScheme::PerChannelFp8 => {
                let scale_entry = self.header.find_scale_for(&entry.name).ok_or_else(|| {
                    AnamnesisError::Parse {
                        reason: format!(
                            "no scale tensor found for quantized weight `{}`",
                            entry.name
                        ),
                    }
                })?;
                let scale_data =
                    self.tensor_data(scale_entry.data_offsets.0, scale_entry.data_offsets.1)?;
                let (rows, cols) = Self::shape_to_rows_cols(&entry.shape)?;
                let out = dequantize_per_channel_fp8::<E>(
                    weight_data,
                    scale_data,
                    rows,
                    cols,
                    scale_entry.dtype,
                )?;
                TensorDequant::Owned(entry.name.clone(), out, entry.shape.clone())
            }
            QuantScheme::PerTensorFp8 => {
                // Look for a companion scale tensor; default to 1.0 if none.
                let scale = if let Some(scale_entry) = self.header.find_scale_for(&entry.name) {
                    let scale_data =
                        self.tensor_data(scale_entry.data_offsets.0, scale_entry.data_offsets.1)?;
                    Self::read_scalar_scale(scale_data, scale_entry.dtype, &entry.name)?
                } else {
                    1.0
                };
                let out = dequantize_per_tensor_fp8::<E>(weight_data, scale)?;
                TensorDequant::Owned(entry.name.clone(), out, entry.shape.clone())
            }
            #[cfg(feature = "gptq")]
            QuantScheme::Gptq => {
                let config = self
                    .header
                    .gptq_config
                    .ok_or_else(|| AnamnesisError::Parse {
                        reason: format!("GPTQ config not available for `{}`", entry.name),
                    })?;
                let companions =
                    self.header
                        .find_gptq_companions(&entry.name)
                        .ok_or_else(|| AnamnesisError::Parse {
                            reason: format!("GPTQ companions not found for `{}`", entry.name),
                        })?;

                let scales_data = self.tensor_data(
                    companions.scales.data_offsets.0,
                    companions.scales.data_offsets.1,
                )?;
                let qzeros_data = self.tensor_data(
                    companions.qzeros.data_offsets.0,
                    companions.qzeros.data_offsets.1,
                )?;
                let g_idx_data = companions
                    .g_idx
                    .map(|e| self.tensor_data(e.data_offsets.0, e.data_offsets.1))
                    .transpose()?;

                // Derive in_features and out_features from qweight shape.
                // qweight shape: [in_features/pack_factor, out_features]
                let (packed_rows, out_features) = Self::shape_to_rows_cols(&entry.shape)?;
                // CAST: u8 → usize, bits is 4 or 8
                #[allow(clippy::as_conversions)]
                let pack_factor = 32 / config.bits as usize;
                let in_features =
                    packed_rows
                        .checked_mul(pack_factor)
                        .ok_or_else(|| AnamnesisError::Parse {
                            reason: "in_features overflow".into(),
                        })?;

                let native = dequantize_gptq::<E>(
                    weight_data,
                    scales_data,
                    qzeros_data,
                    g_idx_data,
                    in_features,
                    out_features,
                    config.group_size,
                    config.bits,
                    companions.scales.dtype,
                )?;
                // The kernel returns the GEMM-native
                // [in_features, out_features] orientation (the
                // canonical GPTQModel kernel layout the
                // cross-validation fixtures anchor). A standard
                // nn.Linear safetensors is [out, in] — apply the
                // same boundary transpose GPTQModel's
                // dequantize_model applies (`.T`).
                let data = transpose_elements::<E>(&native, in_features, out_features)?;

                // Output tensor: strip ".qweight" suffix, use ".weight".
                let output_name = entry
                    .name
                    .strip_suffix(".qweight")
                    .map_or_else(|| entry.name.clone(), |base| format!("{base}.weight"));
                let output_shape = vec![out_features, in_features];

                TensorDequant::Owned(output_name, data, output_shape)
            }
            #[cfg(not(feature = "gptq"))]
            QuantScheme::Gptq => {
                return Err(AnamnesisError::Unsupported {
                    format: "GPTQ".into(),
                    detail: "GPTQ dequantization requires the `gptq` feature".into(),
                });
            }
            #[cfg(feature = "awq")]
            QuantScheme::Awq => {
                let config = self
                    .header
                    .awq_config
                    .ok_or_else(|| AnamnesisError::Parse {
                        reason: format!("AWQ config not available for `{}`", entry.name),
                    })?;
                let companions = self
                    .header
                    .find_awq_companions(&entry.name)
                    .ok_or_else(|| AnamnesisError::Parse {
                        reason: format!("AWQ companions not found for `{}`", entry.name),
                    })?;

                let scales_data = self.tensor_data(
                    companions.scales.data_offsets.0,
                    companions.scales.data_offsets.1,
                )?;
                let qzeros_data = self.tensor_data(
                    companions.qzeros.data_offsets.0,
                    companions.qzeros.data_offsets.1,
                )?;

                // Derive in_features and out_features from qweight + scales shapes.
                // AWQ qweight: [in_features, out_features/pack_factor]
                // scales: [num_groups, out_features]
                let in_features =
                    entry
                        .shape
                        .first()
                        .copied()
                        .ok_or_else(|| AnamnesisError::Parse {
                            reason: "AWQ qweight has no first dimension".into(),
                        })?;
                let out_features = companions.scales.shape.last().copied().ok_or_else(|| {
                    AnamnesisError::Parse {
                        reason: "AWQ scales has no last dimension".into(),
                    }
                })?;

                let native = dequantize_awq::<E>(
                    weight_data,
                    scales_data,
                    qzeros_data,
                    in_features,
                    out_features,
                    config.group_size,
                    config.bits,
                    companions.scales.dtype,
                )?;
                // The kernel returns the GEMM-native
                // [in_features, out_features] orientation (the
                // canonical AutoAWQ kernel layout the
                // cross-validation fixtures anchor). A standard
                // nn.Linear safetensors is [out, in] — transpose
                // at the output-contract boundary, exactly as
                // GPTQModel's dequantize_model does for its
                // GEMM-native dequant (`.T`).
                let data = transpose_elements::<E>(&native, in_features, out_features)?;

                // Output tensor: strip ".qweight" suffix, use ".weight".
                let output_name = entry
                    .name
                    .strip_suffix(".qweight")
                    .map_or_else(|| entry.name.clone(), |base| format!("{base}.weight"));
                let output_shape = vec![out_features, in_features];

                TensorDequant::Owned(output_name, data, output_shape)
            }
            #[cfg(not(feature = "awq"))]
            QuantScheme::Awq => {
                return Err(AnamnesisError::Unsupported {
                    format: "AWQ".into(),
                    detail: "AWQ dequantization requires the `awq` feature".into(),
                });
            }
            #[cfg(feature = "bnb")]
            QuantScheme::Bnb4 => {
                let config = self
                    .header
                    .bnb_config
                    .ok_or_else(|| AnamnesisError::Parse {
                        reason: format!("BnB config not available for `{}`", entry.name),
                    })?;
                let companions =
                    self.header
                        .find_bnb4_companions(&entry.name)
                        .ok_or_else(|| AnamnesisError::Parse {
                            reason: format!("BnB4 companions not found for `{}`", entry.name),
                        })?;

                let absmax_data = self.tensor_data(
                    companions.absmax.data_offsets.0,
                    companions.absmax.data_offsets.1,
                )?;
                let quant_map_data = self.tensor_data(
                    companions.quant_map.data_offsets.0,
                    companions.quant_map.data_offsets.1,
                )?;

                let total_elements =
                    entry
                        .byte_len()
                        .checked_mul(2)
                        .ok_or_else(|| AnamnesisError::Parse {
                            reason: "BnB4 total_elements overflow".into(),
                        })?;

                // Read the quant_state JSON blob once: the
                // double-quant path needs `nested_offset` from it
                // BEFORE dequantizing, and the shape recovery
                // below needs `shape`.
                let quant_state_data = companions
                    .quant_state
                    .map(|qs_entry| {
                        self.tensor_data(qs_entry.data_offsets.0, qs_entry.data_offsets.1)
                    })
                    .transpose()?;

                let data = if config.double_quant {
                    let nested_absmax =
                        companions
                            .nested_absmax
                            .ok_or_else(|| AnamnesisError::Parse {
                                reason: format!(
                                    "BnB4 double-quant: nested_absmax not found for `{}`",
                                    entry.name
                                ),
                            })?;
                    let nested_quant_map =
                        companions
                            .nested_quant_map
                            .ok_or_else(|| AnamnesisError::Parse {
                                reason: format!(
                                    "BnB4 double-quant: nested_quant_map not found for `{}`",
                                    entry.name
                                ),
                            })?;
                    let nested_absmax_data = self
                        .tensor_data(nested_absmax.data_offsets.0, nested_absmax.data_offsets.1)?;
                    let nested_quant_map_data = self.tensor_data(
                        nested_quant_map.data_offsets.0,
                        nested_quant_map.data_offsets.1,
                    )?;

                    // Infer nested_block_size from absmax count / nested_absmax count
                    let absmax_count = companions.absmax.num_elements();
                    let nested_absmax_count = nested_absmax.num_elements();
                    let nested_block_size = if nested_absmax_count > 0 {
                        absmax_count.div_ceil(nested_absmax_count)
                    } else {
                        256
                    };

                    // The nested_offset is mandatory for the
                    // double-quant absmax recovery; a DQ tensor
                    // without a quant_state blob cannot be
                    // decoded correctly.
                    let nested_offset = match quant_state_data {
                        Some(qs_data) => parse_bnb_quant_state_nested_offset(qs_data, &entry.name)?,
                        None => {
                            return Err(AnamnesisError::Parse {
                                reason: format!(
                                    "BnB4 double-quant: quant_state blob not found \
                                                 for `{}` (required for nested_offset)",
                                    entry.name
                                ),
                            });
                        }
                    };

                    dequantize_bnb4_double_quant::<E>(
                        weight_data,
                        absmax_data,
                        quant_map_data,
                        nested_absmax_data,
                        nested_quant_map_data,
                        nested_offset,
                        total_elements,
                        config.block_size,
                        nested_block_size,
                    )?
                } else {
                    dequantize_bnb4::<E>(
                        weight_data,
                        absmax_data,
                        quant_map_data,
                        total_elements,
                        config.block_size,
                    )?
                };

                // BnB4 weights are stored flattened to [N, 1]. Recover the original
                // 2D shape from the quant_state companion tensor (JSON blob with
                // "shape" field), falling back to flat [total_elements] if absent.
                let output_shape = if let Some(qs_data) = quant_state_data {
                    parse_bnb_quant_state_shape(qs_data, total_elements, &entry.name)?
                } else {
                    vec![total_elements]
                };

                TensorDequant::Owned(entry.name.clone(), data, output_shape)
            }
            #[cfg(feature = "bnb")]
            QuantScheme::BnbInt8 => {
                let scb_entry = self.header.find_bnb_int8_scb(&entry.name).ok_or_else(|| {
                    AnamnesisError::Parse {
                        reason: format!("BnB INT8 SCB companion not found for `{}`", entry.name),
                    }
                })?;
                let scb_data =
                    self.tensor_data(scb_entry.data_offsets.0, scb_entry.data_offsets.1)?;

                // INT8 keeps its 2D shape [out_features, in_features].
                let (out_features, in_features) = Self::shape_to_rows_cols(&entry.shape)?;

                let data =
                    dequantize_bnb_int8::<E>(weight_data, scb_data, out_features, in_features)?;

                // Output tensor: keep name, keep shape.
                TensorDequant::Owned(entry.name.clone(), data, entry.shape.clone())
            }
            #[cfg(not(feature = "bnb"))]
            QuantScheme::Bnb4 | QuantScheme::BnbInt8 => {
                return Err(AnamnesisError::Unsupported {
                    format: "BnB".into(),
                    detail: "BnB dequantization requires the `bnb` feature".into(),
                });
            }
            QuantScheme::Unquantized => {
                // Shouldn't have a quantized-role tensor in an
                // unquantized model; the orchestrator resolves this
                // to a passthrough on the main thread.
                TensorDequant::Passthrough
            }
        };

        Ok(result)
    }

    /// Internal: run the per-scheme dequant for every tensor, returning the owned
    /// `E` results plus the passthrough tensors (which borrow `self.buffer`).
    /// Shared by `remember_inner` (→ file), `remember_to_bytes_inner`
    /// (→ bytes) and `hub_tensors` (→ the `convert` hub); `on_tensor` fires on
    /// the **main thread** after each quantized tensor is dequantised so callers
    /// can drive a progress bar.
    ///
    /// `threads` is the resolved worker budget (see [`resolve_thread_budget`]);
    /// the quantized entries are handed to `parallel::map_indexed`, which decides
    /// between the sequential loop and a scoped worker pool and guarantees
    /// results come back in input order. Output is therefore **byte-identical
    /// for any thread count** — the tensors are reassembled in original header
    /// order before serialization no matter how the work was distributed.
    ///
    /// # Errors
    ///
    /// Propagates [`Self::dequantize_quantized_entry`]'s errors — deterministically,
    /// the lowest-indexed failure, at any thread count — and returns
    /// [`AnamnesisError::Parse`] if a dequant worker thread panics.
    fn dequantize_all<E: OutputElement, F>(
        &self,
        threads: usize,
        cancel: Option<&crate::CancelToken>,
        mut on_tensor: F,
    ) -> crate::Result<(DequantizedTensors, PassthroughRefs<'_>)>
    where
        F: FnMut(),
    {
        // Classify every entry once, in header order. Quantized entries are
        // collected with their original index so results can be reassembled
        // deterministically; passthrough entries are resolved here on the main
        // thread; companion tensors are consumed during dequant and skipped.
        let mut quantized: Vec<(usize, &TensorEntry)> = Vec::new();
        // (original index, ref) so both normal passthroughs and the
        // `Unquantized`-scheme edge case can be merged back in header order.
        let mut passthrough_indexed: Vec<IndexedPassthrough<'_>> = Vec::new();

        for (idx, entry) in self.header.tensors.iter().enumerate() {
            match entry.role {
                TensorRole::Quantized => quantized.push((idx, entry)),
                TensorRole::Scale
                | TensorRole::ZeroPoint
                | TensorRole::GroupIndex
                | TensorRole::QuantMap
                | TensorRole::NestedScale
                | TensorRole::QuantState => {
                    // Companion tensors are consumed during dequantization; skip.
                }
                TensorRole::Passthrough => {
                    let data = self.tensor_data(entry.data_offsets.0, entry.data_offsets.1)?;
                    passthrough_indexed.push((idx, (&entry.name, data, &entry.shape)));
                }
            }
        }

        // Total on-disk span of the quantised weights, the size gate
        // `parallel::map_indexed` consults before it spawns anything. The
        // companion scale / zero-point tensors add a small constant fraction on
        // top and are deliberately not counted — the threshold only has to
        // separate "trivial" from "worth a thread pool".
        let work_bytes: u64 = quantized.iter().fold(0u64, |acc, &(_, entry)| {
            let (start, end) = entry.data_offsets;
            acc.saturating_add(u64::try_from(end.saturating_sub(start)).unwrap_or(u64::MAX))
        });

        // Dequantise the quantized entries. `map_indexed` returns results in
        // `quantized` order for any thread count (see `src/parallel.rs`), and
        // `quantized` was built by walking the header in order, so re-zipping it
        // with its original indices restores header order without a sort.
        // `dequantize_quantized_entry` is a pure fn of shared-immutable `&self`
        // plus its entry, and `ParsedModel` is `Sync`, so the closure is safe to
        // share across workers; `on_tensor` stays on this thread.
        let results = crate::parallel::map_indexed(
            &quantized,
            threads,
            work_bytes,
            cancel,
            |_, &(_, entry)| self.dequantize_quantized_entry::<E>(entry),
            |dq| {
                if matches!(dq, TensorDequant::Owned(..)) {
                    on_tensor();
                }
            },
        )?;
        let dequants: Vec<(usize, TensorDequant)> =
            quantized.iter().map(|&(idx, _)| idx).zip(results).collect();

        let mut dequantized_data: DequantizedTensors = Vec::with_capacity(dequants.len());
        for (idx, dq) in dequants {
            match dq {
                TensorDequant::Owned(name, data, shape) => {
                    dequantized_data.push((name, data, shape));
                }
                TensorDequant::Passthrough => {
                    let entry =
                        self.header
                            .tensors
                            .get(idx)
                            .ok_or_else(|| AnamnesisError::Parse {
                                reason: "dequant result index out of bounds for header".into(),
                            })?;
                    let data = self.tensor_data(entry.data_offsets.0, entry.data_offsets.1)?;
                    passthrough_indexed.push((idx, (&entry.name, data, &entry.shape)));
                }
            }
        }

        // Merge normal passthroughs with any `Unquantized`-scheme edge cases in
        // header order, matching the original single-pass ordering exactly.
        passthrough_indexed.sort_by_key(|&(idx, _)| idx);
        let passthrough_refs: PassthroughRefs<'_> =
            passthrough_indexed.into_iter().map(|(_, r)| r).collect();

        Ok((dequantized_data, passthrough_refs))
    }

    /// Internal: build the `safetensors` `TensorView` list from the dequantised
    /// (owned) tensors and the passthrough (borrowed) tensors. Shared by both
    /// `remember` destinations; the views borrow `dequantized_data`, so the
    /// caller must keep it alive until serialization completes.
    fn build_views<'a, E: OutputElement>(
        &'a self,
        dequantized_data: &'a [(String, Vec<u8>, Vec<usize>)],
        passthrough_refs: &[(&'a str, &'a [u8], &'a [usize])],
    ) -> crate::Result<Vec<(String, safetensors::tensor::TensorView<'a>)>> {
        // Build TensorView list for serialization.
        // Dequantized tensors are declared as `E::DTYPE` — the width the
        // kernels actually wrote, taken from the same constant that sized their
        // output buffers, so the header cannot disagree with the payload.
        // Passthrough tensors keep their original dtype: an `F32` request
        // widens what was dequantised, never what was already full precision
        // (see `TargetDtype`'s passthrough policy).
        let dequantized_dtype = E::DTYPE.to_safetensors_dtype()?;
        let mut views: Vec<(String, safetensors::tensor::TensorView<'_>)> = Vec::new();

        for (name, data, shape) in dequantized_data {
            let view = safetensors::tensor::TensorView::new(dequantized_dtype, shape.clone(), data)
                .map_err(|e| AnamnesisError::Parse {
                    reason: format!("failed to create TensorView for `{name}`: {e}"),
                })?;
            views.push((name.clone(), view));
        }

        for &(name, data, shape) in passthrough_refs {
            // Look up the original dtype for this passthrough tensor.
            let entry = self
                .header
                .tensors
                .iter()
                .find(|t| t.name == name)
                .ok_or_else(|| AnamnesisError::Parse {
                    reason: format!("passthrough tensor `{name}` not found in header"),
                })?;
            let st_dtype = entry.dtype.to_safetensors_dtype()?;
            let view = safetensors::tensor::TensorView::new(st_dtype, shape.to_vec(), data)
                .map_err(|e| AnamnesisError::Parse {
                    reason: format!("failed to create TensorView for `{name}`: {e}"),
                })?;
            views.push((name.to_owned(), view));
        }

        Ok(views)
    }

    /// Internal: dequantize to `E` and write, with optional progress callback.
    fn remember_inner<E: OutputElement, F>(
        &self,
        output_path: &Path,
        threads: usize,
        cancel: Option<&crate::CancelToken>,
        on_tensor: F,
    ) -> crate::Result<()>
    where
        F: FnMut(),
    {
        let (dequantized_data, passthrough_refs) =
            self.dequantize_all::<E, F>(threads, cancel, on_tensor)?;
        let views = self.build_views::<E>(&dequantized_data, &passthrough_refs)?;

        // Serialize to file. The safetensors writer streams tensor bodies one at
        // a time, so the file path's peak stays at the dequantised set — unlike
        // `remember_to_bytes`, which holds the whole serialized `Vec`.
        let metadata = self.header.metadata.clone();
        safetensors::tensor::serialize_to_file(views, metadata, output_path).map_err(
            // EXHAUSTIVE: SafeTensorError is a foreign type that may gain variants;
            // we extract IoError and treat everything else as a parse/format error.
            #[allow(clippy::wildcard_enum_match_arm)]
            |e| match e {
                safetensors::SafeTensorError::IoError(io_err) => AnamnesisError::Io(io_err),
                other => AnamnesisError::Parse {
                    reason: format!("failed to write safetensors file: {other}"),
                },
            },
        )?;

        Ok(())
    }

    /// Internal: dequantize to `E` and return the serialized safetensors bytes.
    fn remember_to_bytes_inner<E: OutputElement>(
        &self,
        threads: usize,
        cancel: Option<&crate::CancelToken>,
    ) -> crate::Result<Vec<u8>> {
        let (dequantized_data, passthrough_refs) =
            self.dequantize_all::<E, _>(threads, cancel, || {})?;
        let views = self.build_views::<E>(&dequantized_data, &passthrough_refs)?;

        let metadata = self.header.metadata.clone();
        safetensors::tensor::serialize(views, metadata).map_err(|e| AnamnesisError::Parse {
            reason: format!("failed to serialize safetensors bytes: {e}"),
        })
    }
}

// ---------------------------------------------------------------------------
// BnB4 quant_state shape recovery
// ---------------------------------------------------------------------------

/// Parses the original tensor shape from a `BnB` `quant_state` companion tensor.
///
/// The `quant_state.bitsandbytes__nf4` (or `__fp4`) tensor stores a `JSON` blob
/// as raw `U8` bytes. The blob contains a `"shape"` field with the original
/// 2D tensor dimensions (e.g., `[2048, 8192]`).
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if the `JSON` is malformed, the `"shape"`
/// field is missing, or the recovered shape does not match `total_elements`.
#[cfg(feature = "bnb")]
fn parse_bnb_quant_state_shape(
    qs_data: &[u8],
    total_elements: usize,
    weight_name: &str,
) -> crate::Result<Vec<usize>> {
    let qs_str = std::str::from_utf8(qs_data).map_err(|e| AnamnesisError::Parse {
        reason: format!("quant_state for `{weight_name}` is not valid UTF-8: {e}"),
    })?;

    let qs_json: serde_json::Value =
        serde_json::from_str(qs_str).map_err(|e| AnamnesisError::Parse {
            reason: format!("failed to parse quant_state JSON for `{weight_name}`: {e}"),
        })?;

    let shape_arr = qs_json
        .get("shape")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| AnamnesisError::Parse {
            reason: format!("quant_state for `{weight_name}` missing \"shape\" array"),
        })?;

    let shape: Vec<usize> = shape_arr
        .iter()
        .map(|v| {
            v.as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| AnamnesisError::Parse {
                    reason: format!(
                        "quant_state shape dimension not a valid usize for `{weight_name}`"
                    ),
                })
        })
        .collect::<crate::Result<_>>()?;

    // Validate: product of recovered shape must equal total_elements.
    let product: usize = shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| AnamnesisError::Parse {
            reason: format!("quant_state shape overflow for `{weight_name}`"),
        })?;

    if product != total_elements {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "quant_state shape {shape:?} product {product} != total_elements {total_elements} \
                 for `{weight_name}`"
            ),
        });
    }

    Ok(shape)
}

/// Parses the double-quant `nested_offset` from a `BnB` `quant_state`
/// companion tensor.
///
/// `bitsandbytes` double quantization subtracts the mean of the per-block
/// absmax values before nested-quantizing them, and stores that mean in the
/// `quant_state` `JSON` blob as `"nested_offset"`. Recovery must add it back
/// (`absmax = nested_dequant(...) + nested_offset`); omitting it biases every
/// recovered absmax low by the offset.
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if the `JSON` is malformed or the
/// `"nested_offset"` field is missing or not a number. The field is
/// mandatory for double-quant states: every `bitsandbytes` serialization
/// that emits `nested_absmax` / `nested_quant_map` also emits it, so its
/// absence indicates a malformed or truncated `quant_state`.
#[cfg(feature = "bnb")]
fn parse_bnb_quant_state_nested_offset(qs_data: &[u8], weight_name: &str) -> crate::Result<f32> {
    let qs_str = std::str::from_utf8(qs_data).map_err(|e| AnamnesisError::Parse {
        reason: format!("quant_state for `{weight_name}` is not valid UTF-8: {e}"),
    })?;

    let qs_json: serde_json::Value =
        serde_json::from_str(qs_str).map_err(|e| AnamnesisError::Parse {
            reason: format!("failed to parse quant_state JSON for `{weight_name}`: {e}"),
        })?;

    let offset_f64 = qs_json
        .get("nested_offset")
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| AnamnesisError::Parse {
            reason: format!(
                "quant_state for `{weight_name}` missing \"nested_offset\" (required for \
                 double-quant absmax recovery)"
            ),
        })?;

    // CAST: the JSON value is the decimal rendering of a bitsandbytes f32;
    // narrowing f64 → f32 recovers it exactly.
    #[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
    Ok(offset_f64 as f32)
}

#[cfg(test)]
#[allow(
    clippy::panic,
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::float_cmp
)]
mod tests {
    use super::*;

    /// Build a minimal safetensors file in memory with the given tensors.
    fn build_safetensors(tensors: &[(&str, safetensors::Dtype, &[usize], &[u8])]) -> Vec<u8> {
        let views: Vec<(&str, safetensors::tensor::TensorView<'_>)> = tensors
            .iter()
            .map(|(name, dtype, shape, data)| {
                let view =
                    safetensors::tensor::TensorView::new(*dtype, shape.to_vec(), data).unwrap();
                (*name, view)
            })
            .collect();
        safetensors::tensor::serialize(views, None).unwrap()
    }

    #[test]
    fn parse_and_inspect_unquantized() {
        // 2 BF16 tensors
        let bf16_data = vec![0x80, 0x3F]; // BF16 1.0
        let file = build_safetensors(&[
            ("weight", safetensors::Dtype::BF16, &[1], &bf16_data),
            ("norm", safetensors::Dtype::BF16, &[1], &bf16_data),
        ]);

        let tmp = std::env::temp_dir().join("test_unquant.safetensors");
        std::fs::write(&tmp, &file).unwrap();

        let model = parse(&tmp).unwrap();
        let info = model.inspect();

        assert_eq!(info.format, QuantScheme::Unquantized);
        assert_eq!(info.quantized, 0);
        assert_eq!(info.passthrough, 2);

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn parse_nonexistent_file() {
        let result = parse("/tmp/nonexistent_anamnesis_test.safetensors");
        assert!(result.is_err());
    }

    #[test]
    fn parse_invalid_data() {
        let tmp = std::env::temp_dir().join("test_invalid.safetensors");
        std::fs::write(&tmp, b"not a safetensors file").unwrap();

        let result = parse(&tmp);
        assert!(result.is_err());

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn remember_passthrough_only() {
        // BF16 tensor with known value: 2.0 = 0x4000 in BF16
        let bf16_data = vec![0x00, 0x40, 0x00, 0x40]; // two BF16 2.0
        let file = build_safetensors(&[("weight", safetensors::Dtype::BF16, &[2], &bf16_data)]);

        let tmp_in = std::env::temp_dir().join("test_pass_in.safetensors");
        let tmp_out = std::env::temp_dir().join("test_pass_out.safetensors");
        std::fs::write(&tmp_in, &file).unwrap();

        let model = parse(&tmp_in).unwrap();
        model.remember(&tmp_out, TargetDtype::BF16).unwrap();

        // Read output and verify bytes match
        let out_data = std::fs::read(&tmp_out).unwrap();
        let out_model = parse(&tmp_out).unwrap();
        let out_info = out_model.inspect();
        assert_eq!(out_info.passthrough, 1);

        // Verify the tensor data is preserved
        let entry = &out_model.header.tensors[0];
        let data_offset = out_model.header.header_size + 8;
        let tensor_bytes =
            &out_data[data_offset + entry.data_offsets.0..data_offset + entry.data_offsets.1];
        assert_eq!(tensor_bytes, &bf16_data);

        std::fs::remove_file(&tmp_in).ok();
        std::fs::remove_file(&tmp_out).ok();
    }

    /// Builds a raw per-tensor-FP8 safetensors fixture in memory: a 2×2 `F8_E4M3`
    /// weight (`1.0`), a scalar `F32` scale (`2.0`), and a `BF16` passthrough
    /// norm (`1.0`). Built by hand because the `safetensors` crate may not
    /// serialize `F8_E4M3`. Shared by the file and bytes `remember` round-trips.
    fn build_fp8_per_tensor_fixture() -> Vec<u8> {
        let fp8_data = vec![0x38u8; 4]; // 2x2 of 1.0 in E4M3
        let scale_data = 2.0_f32.to_le_bytes().to_vec();
        let norm_data = vec![0x80, 0x3F]; // BF16 1.0

        let mut header_map = serde_json::Map::new();

        // FP8 weight at offset 0, length 4
        let mut w_info = serde_json::Map::new();
        w_info.insert("dtype".into(), "F8_E4M3".into());
        w_info.insert("shape".into(), serde_json::json!([2, 2]));
        w_info.insert("data_offsets".into(), serde_json::json!([0, 4]));
        header_map.insert("layer.weight".into(), w_info.into());

        // F32 scale at offset 4, length 4
        let mut s_info = serde_json::Map::new();
        s_info.insert("dtype".into(), "F32".into());
        s_info.insert("shape".into(), serde_json::json!([1]));
        s_info.insert("data_offsets".into(), serde_json::json!([4, 8]));
        header_map.insert("layer.weight_scale".into(), s_info.into());

        // BF16 norm at offset 8, length 2
        let mut n_info = serde_json::Map::new();
        n_info.insert("dtype".into(), "BF16".into());
        n_info.insert("shape".into(), serde_json::json!([1]));
        n_info.insert("data_offsets".into(), serde_json::json!([8, 10]));
        header_map.insert("norm.weight".into(), n_info.into());

        let header_json = serde_json::to_string(&header_map).unwrap();
        let header_bytes = header_json.as_bytes();

        // Build raw safetensors file: 8-byte length + header + data
        // CAST: usize → u64, header length fits in u64
        #[allow(clippy::as_conversions)]
        let header_len = header_bytes.len() as u64;
        let mut file_bytes = Vec::new();
        file_bytes.extend_from_slice(&header_len.to_le_bytes());
        file_bytes.extend_from_slice(header_bytes);
        file_bytes.extend_from_slice(&fp8_data);
        file_bytes.extend_from_slice(&scale_data);
        file_bytes.extend_from_slice(&norm_data);
        file_bytes
    }

    #[test]
    fn remember_fp8_round_trip() {
        // FP8 weight (1.0) × per-tensor scale (2.0) → BF16 2.0, plus a BF16
        // passthrough norm; the scale tensor is consumed (absent from output).
        let file_bytes = build_fp8_per_tensor_fixture();

        let tmp_in = std::env::temp_dir().join("test_fp8_in.safetensors");
        let tmp_out = std::env::temp_dir().join("test_fp8_out.safetensors");
        std::fs::write(&tmp_in, &file_bytes).unwrap();

        let model = parse(&tmp_in).unwrap();
        assert_eq!(model.header.scheme, QuantScheme::PerTensorFp8);
        assert_eq!(model.inspect().quantized, 1);

        model.remember(&tmp_out, TargetDtype::BF16).unwrap();

        // Read output and verify
        let out_model = parse(&tmp_out).unwrap();
        let out_info = out_model.inspect();
        // Output should have: 1 passthrough (was FP8, now BF16) + 1 passthrough (norm)
        // Scale tensor should be absent
        assert_eq!(out_info.passthrough, 2); // both are now BF16
        assert_eq!(out_info.quantized, 0);

        // Verify the weight values: 1.0 * 2.0 = 2.0 → BF16 0x4000 → LE [0x00, 0x40]
        let w_entry = out_model
            .header
            .tensors
            .iter()
            .find(|t| t.name == "layer.weight")
            .unwrap();
        let data_start = out_model.header.header_size + 8;
        let out_bytes = std::fs::read(&tmp_out).unwrap();
        let w_data =
            &out_bytes[data_start + w_entry.data_offsets.0..data_start + w_entry.data_offsets.1];
        // 4 elements × 2 bytes = 8 bytes
        assert_eq!(w_data.len(), 8);
        for chunk in w_data.chunks_exact(2) {
            assert_eq!(chunk, &[0x00, 0x40], "expected BF16 2.0");
        }

        std::fs::remove_file(&tmp_in).ok();
        std::fs::remove_file(&tmp_out).ok();
    }

    #[test]
    fn remember_to_bytes_fp8_round_trip() {
        let file_bytes = build_fp8_per_tensor_fixture();

        let tmp_in = std::env::temp_dir().join("test_fp8_bytes_in.safetensors");
        let tmp_out = std::env::temp_dir().join("test_fp8_bytes_out.safetensors");
        std::fs::write(&tmp_in, &file_bytes).unwrap();

        let model = parse(&tmp_in).unwrap();
        assert_eq!(model.header.scheme, QuantScheme::PerTensorFp8);

        // Core pairing invariant: the in-memory bytes are byte-identical to what
        // the file path writes (same views, same metadata, same serialization).
        let bytes = model.remember_to_bytes(TargetDtype::BF16).unwrap();
        model.remember(&tmp_out, TargetDtype::BF16).unwrap();
        let file_out = std::fs::read(&tmp_out).unwrap();
        assert_eq!(
            bytes, file_out,
            "remember_to_bytes must match remember's file bytes"
        );

        // Round-trip: parse the returned bytes back and verify the dequant.
        std::fs::write(&tmp_out, &bytes).unwrap();
        let out_model = parse(&tmp_out).unwrap();
        let out_info = out_model.inspect();
        assert_eq!(out_info.passthrough, 2); // weight (now BF16) + norm
        assert_eq!(out_info.quantized, 0); // scale consumed

        // Weight values: 1.0 × 2.0 = 2.0 → BF16 0x4000 → LE [0x00, 0x40].
        let w_entry = out_model
            .header
            .tensors
            .iter()
            .find(|t| t.name == "layer.weight")
            .unwrap();
        let data_start = out_model.header.header_size + 8;
        let w_data =
            &bytes[data_start + w_entry.data_offsets.0..data_start + w_entry.data_offsets.1];
        assert_eq!(w_data.len(), 8); // 4 elements × 2 bytes
        for chunk in w_data.chunks_exact(2) {
            assert_eq!(chunk, &[0x00, 0x40], "expected BF16 2.0");
        }

        std::fs::remove_file(&tmp_in).ok();
        std::fs::remove_file(&tmp_out).ok();
    }

    #[test]
    fn target_dtype_display() {
        assert_eq!(TargetDtype::BF16.to_string(), "BF16");
    }

    /// Builds a raw per-tensor-FP8 safetensors fixture with `n_weights` distinct
    /// quantized weights (each a 2×2 `F8_E4M3` block with its own scalar `F32`
    /// scale) plus one `BF16` passthrough norm. Each weight carries different
    /// FP8 bytes and a different scale, so a mis-ordered parallel reassembly
    /// would corrupt the output — making this a sharp determinism probe.
    ///
    /// **This size exercises the sequential path only.** At 4 bytes per weight
    /// it is orders of magnitude below [`crate::parallel::MIN_PARALLEL_BYTES`],
    /// so `map_indexed` never spawns no matter what thread budget is requested.
    /// That is fine for the ordering and round-trip tests, but a determinism
    /// test that means to prove something about the *parallel* dispatch must use
    /// [`build_multi_fp8_fixture_sized`] and clear the threshold — see
    /// `remember_fixture_crosses_the_parallel_threshold`.
    fn build_multi_fp8_fixture(n_weights: usize) -> Vec<u8> {
        build_multi_fp8_fixture_sized(n_weights, 2, 2)
    }

    /// [`build_multi_fp8_fixture`] with a caller-chosen weight shape, so a test
    /// can size the fixture off [`crate::parallel::MIN_PARALLEL_BYTES`] rather
    /// than hope it clears it.
    fn build_multi_fp8_fixture_sized(n_weights: usize, rows: usize, cols: usize) -> Vec<u8> {
        let mut header_map = serde_json::Map::new();
        let mut data = Vec::new();
        let elems = rows * cols;

        for i in 0..n_weights {
            // Distinct FP8 payload per weight: E4M3 values 0x38 (1.0), 0x40
            // (2.0), 0x48 (4.0)… cycled so no two adjacent weights match.
            // INDEX: fixed small table, index is `i % 3` in bounds.
            let fp8_byte = [0x38u8, 0x40, 0x48][i % 3];
            let w_off = data.len();
            data.extend(std::iter::repeat_n(fp8_byte, elems));

            let mut w_info = serde_json::Map::new();
            w_info.insert("dtype".into(), "F8_E4M3".into());
            w_info.insert("shape".into(), serde_json::json!([rows, cols]));
            w_info.insert(
                "data_offsets".into(),
                serde_json::json!([w_off, data.len()]),
            );
            header_map.insert(format!("layer.{i}.weight"), w_info.into());

            // Distinct scale per weight.
            // CAST: usize → f32 for a small test scale value; exact.
            #[allow(clippy::as_conversions, clippy::cast_precision_loss)]
            let scale = 1.0_f32 + i as f32;
            let s_off = data.len();
            data.extend_from_slice(&scale.to_le_bytes());
            let mut s_info = serde_json::Map::new();
            s_info.insert("dtype".into(), "F32".into());
            s_info.insert("shape".into(), serde_json::json!([1]));
            s_info.insert(
                "data_offsets".into(),
                serde_json::json!([s_off, data.len()]),
            );
            header_map.insert(format!("layer.{i}.weight_scale"), s_info.into());
        }

        // One BF16 passthrough norm.
        let n_off = data.len();
        data.extend_from_slice(&[0x80, 0x3F]); // BF16 1.0
        let mut n_info = serde_json::Map::new();
        n_info.insert("dtype".into(), "BF16".into());
        n_info.insert("shape".into(), serde_json::json!([1]));
        n_info.insert(
            "data_offsets".into(),
            serde_json::json!([n_off, data.len()]),
        );
        header_map.insert("norm.weight".into(), n_info.into());

        let header_json = serde_json::to_string(&header_map).unwrap();
        let header_bytes = header_json.as_bytes();
        // CAST: usize → u64, header length fits in u64.
        #[allow(clippy::as_conversions)]
        let header_len = header_bytes.len() as u64;

        let mut file_bytes = Vec::new();
        file_bytes.extend_from_slice(&header_len.to_le_bytes());
        file_bytes.extend_from_slice(header_bytes);
        file_bytes.extend_from_slice(&data);
        file_bytes
    }

    /// The thread count is a performance knob, never a correctness variable:
    /// `remember_to_bytes_with_options` must produce **byte-identical** output
    /// for every thread budget, including the environment-resolved default.
    ///
    /// **Scope, stated precisely.** This fixture is 8 weights of 4 bytes, which
    /// is far below [`crate::parallel::MIN_PARALLEL_BYTES`], so `map_indexed`
    /// takes the sequential loop at every budget here. What it therefore covers
    /// is that the *budget itself* changes nothing — option plumbing and
    /// in-order reassembly with 8 distinct payloads. Coverage of the genuinely
    /// threaded dispatch lives in
    /// `remember_output_dtype_is_deterministic_across_thread_counts`, whose
    /// fixture clears the gate on purpose. (Until Phase 7.4 this comment claimed
    /// the parallel path; it never ran it.)
    #[test]
    fn remember_bytes_deterministic_across_thread_counts() {
        let file_bytes = build_multi_fp8_fixture(8);
        let tmp_in = std::env::temp_dir().join("test_multi_fp8_determinism.safetensors");
        std::fs::write(&tmp_in, &file_bytes).unwrap();

        let model = parse(&tmp_in).unwrap();
        assert_eq!(model.header.scheme, QuantScheme::PerTensorFp8);
        assert_eq!(model.inspect().quantized, 8, "8 quantized weights expected");

        let baseline = model
            .remember_to_bytes_with_options(
                TargetDtype::BF16,
                RememberOptions::new().with_threads(1),
            )
            .unwrap();

        for n in [1usize, 2, 4, 8] {
            let out = model
                .remember_to_bytes_with_options(
                    TargetDtype::BF16,
                    RememberOptions::new().with_threads(n),
                )
                .unwrap();
            assert_eq!(
                out, baseline,
                "output must be byte-identical for thread count {n}"
            );
        }

        // The default (env-resolved) budget must also match the baseline.
        let default_out = model.remember_to_bytes(TargetDtype::BF16).unwrap();
        assert_eq!(
            default_out, baseline,
            "default thread budget must match the single-threaded baseline"
        );

        std::fs::remove_file(&tmp_in).ok();
    }

    // -----------------------------------------------------------------------
    // Caller-chosen output dtype on the `remember` path (Phase 7.4)
    //
    // The three tests below mirror, one for one, the trio `src/convert.rs`
    // grew in Phase 7.3 (`convert_honours_every_output_dtype_end_to_end`,
    // `output_dtype_changes_the_dequantised_payload_width`,
    // `output_dtype_is_deterministic_across_thread_counts`). `remember` is a
    // separate entry point with its own dispatch, so the guarantees are
    // re-established here rather than assumed to carry over from `convert`.
    // -----------------------------------------------------------------------

    /// Weight count for the parallel-path fixture. Prime, and deliberately not a
    /// multiple of any plausible thread budget, so a static equal-count split
    /// would leave a remainder and an off-by-one in the reassembly would show.
    const PARALLEL_FIXTURE_WEIGHTS: usize = 17;

    /// Per-weight shape for the parallel-path fixture: 256 × 1024 = 256 `KiB` of
    /// `F8_E4M3` input each, so 17 of them clear the 4 `MiB` gate with headroom.
    const PARALLEL_FIXTURE_ROWS: usize = 256;
    const PARALLEL_FIXTURE_COLS: usize = 1024;

    /// Builds the fixture whose quantised span exceeds
    /// [`crate::parallel::MIN_PARALLEL_BYTES`].
    fn build_parallel_fp8_fixture() -> Vec<u8> {
        build_multi_fp8_fixture_sized(
            PARALLEL_FIXTURE_WEIGHTS,
            PARALLEL_FIXTURE_ROWS,
            PARALLEL_FIXTURE_COLS,
        )
    }

    /// The determinism fixture must clear [`crate::parallel::MIN_PARALLEL_BYTES`],
    /// or every thread-count assertion below runs on the sequential path and
    /// proves nothing about the parallel dispatch.
    ///
    /// This is `CONVENTIONS.md` § *Verify parallelism* point 5, made executable.
    /// It is not hypothetical here: the pre-existing `remember` determinism test
    /// used a 4-bytes-per-weight fixture, which is ~130 000× below the gate, so
    /// it was a green test of a code path it never entered.
    #[test]
    fn remember_fixture_crosses_the_parallel_threshold() {
        // CAST: usize → u64, a compile-time fixture size of a few MiB; lossless
        // widening on every supported target.
        #[allow(clippy::as_conversions)]
        let quantised_bytes =
            (PARALLEL_FIXTURE_WEIGHTS * PARALLEL_FIXTURE_ROWS * PARALLEL_FIXTURE_COLS) as u64;
        assert!(
            quantised_bytes >= crate::parallel::MIN_PARALLEL_BYTES,
            "fixture holds {quantised_bytes} B of quantised weight but \
             MIN_PARALLEL_BYTES is {} B — the determinism tests would not \
             exercise the parallel path",
            crate::parallel::MIN_PARALLEL_BYTES
        );

        // And the fixture really does parse as that many quantised tensors.
        let tmp = std::env::temp_dir().join("test_remember_parallel_threshold.safetensors");
        std::fs::write(&tmp, build_parallel_fp8_fixture()).unwrap();
        let model = parse(&tmp).unwrap();
        assert_eq!(model.inspect().quantized, PARALLEL_FIXTURE_WEIGHTS);
        std::fs::remove_file(&tmp).ok();
    }

    /// Parses a `remember` output and returns `(tensor name -> (dtype, bytes))`.
    ///
    /// Reads through the public safetensors reader rather than the header
    /// offsets, because the output contract this phase cares about is what a
    /// *consumer* sees — the v0.6.4 meta-lesson that an orientation bug hid
    /// behind offset-level assertions.
    fn remember_output_tensors(bytes: &[u8]) -> Vec<(String, safetensors::Dtype, Vec<u8>)> {
        let parsed = safetensors::SafeTensors::deserialize(bytes).unwrap();
        let mut out: Vec<(String, safetensors::Dtype, Vec<u8>)> = parsed
            .tensors()
            .into_iter()
            .map(|(name, view)| (name, view.dtype(), view.data().to_vec()))
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    /// Every supported output dtype round-trips through `remember`, and each
    /// dequantised tensor matches the kernel called directly at that same width.
    ///
    /// Also pins the **passthrough policy**: the `BF16` norm is not a
    /// dequantised tensor, so it keeps its dtype and its exact bytes no matter
    /// what the caller asks for. That asymmetry is the single most surprising
    /// thing about a `remember` output file, so it is asserted, not described.
    #[test]
    fn remember_honours_every_output_dtype_end_to_end() {
        let file_bytes = build_multi_fp8_fixture(8);
        let tmp_in = std::env::temp_dir().join("test_remember_dtype_end_to_end.safetensors");
        std::fs::write(&tmp_in, &file_bytes).unwrap();
        let model = parse(&tmp_in).unwrap();

        for (target, expected_st) in [
            (TargetDtype::BF16, safetensors::Dtype::BF16),
            (TargetDtype::F32, safetensors::Dtype::F32),
            (TargetDtype::F16, safetensors::Dtype::F16),
        ] {
            let bytes = model
                .remember_to_bytes(target)
                .unwrap_or_else(|e| panic!("remember at {target}: {e}"));

            for (name, dtype, data) in remember_output_tensors(&bytes) {
                if name == "norm.weight" {
                    assert_eq!(
                        dtype,
                        safetensors::Dtype::BF16,
                        "passthrough must ignore the requested {target}"
                    );
                    assert_eq!(data, vec![0x80, 0x3F], "passthrough bytes must be verbatim");
                    continue;
                }

                assert_eq!(dtype, expected_st, "{name} at {target}");

                // Rebuild the kernel's answer for this tensor directly. The
                // fixture's weights are 2×2 FP8 with a per-tensor scale of
                // `1.0 + i`, cycling the byte pattern [0x38, 0x40, 0x48].
                let i: usize = name
                    .strip_prefix("layer.")
                    .and_then(|s| s.strip_suffix(".weight"))
                    .unwrap()
                    .parse()
                    .unwrap();
                // INDEX: fixed 3-entry table, `i % 3` is in bounds.
                let fp8_byte = [0x38u8, 0x40, 0x48][i % 3];
                // CAST: usize → f32, small test index; exact.
                #[allow(clippy::as_conversions, clippy::cast_precision_loss)]
                let scale = 1.0_f32 + i as f32;
                let expected = match target {
                    TargetDtype::BF16 => {
                        crate::dequantize_per_tensor_fp8::<crate::Bf16Out>(&[fp8_byte; 4], scale)
                    }
                    TargetDtype::F32 => {
                        crate::dequantize_per_tensor_fp8::<crate::F32Out>(&[fp8_byte; 4], scale)
                    }
                    TargetDtype::F16 => {
                        crate::dequantize_per_tensor_fp8::<crate::F16Out>(&[fp8_byte; 4], scale)
                    }
                }
                .unwrap();
                assert_eq!(data, expected, "{name} at {target} vs the kernel directly");
            }
        }

        std::fs::remove_file(&tmp_in).ok();
    }

    /// `F32` output really is twice the dequantised payload, and `F16` really is
    /// the same width as `BF16`.
    ///
    /// A structural check that catches a whole class of plumbing mistake: if the
    /// dtype were dropped anywhere between [`TargetDtype`] and the writer, all
    /// three payloads would collapse to one size. It sums the **dequantised
    /// tensors' payload bytes** rather than whole-file sizes, for the reason
    /// `convert`'s counterpart records: the safetensors header spells each dtype
    /// out, so `"BF16"` and `"F16"` differ by a byte per tensor and a file-size
    /// comparison fails for a reason that has nothing to do with the claim.
    #[test]
    fn remember_output_dtype_changes_the_dequantised_payload_width() {
        let file_bytes = build_multi_fp8_fixture(8);
        let tmp_in = std::env::temp_dir().join("test_remember_dtype_width.safetensors");
        std::fs::write(&tmp_in, &file_bytes).unwrap();
        let model = parse(&tmp_in).unwrap();

        let mut payloads = Vec::new();
        for target in [TargetDtype::BF16, TargetDtype::F16, TargetDtype::F32] {
            let bytes = model.remember_to_bytes(target).unwrap();
            let dequantised: usize = remember_output_tensors(&bytes)
                .iter()
                .filter(|(name, _, _)| name != "norm.weight")
                .map(|(_, _, data)| data.len())
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

        std::fs::remove_file(&tmp_in).ok();
    }

    /// Determinism is preserved at **every** output dtype, not just the default.
    ///
    /// `CONVENTIONS.md` requires byte-identical output across thread counts for
    /// every parallelised path. Phase 7.4 adds a second axis to `remember`, so
    /// the guarantee is re-established per dtype rather than inherited from the
    /// `BF16`-only test above.
    ///
    /// Uses [`build_parallel_fp8_fixture`], not the 4-byte one: below
    /// [`crate::parallel::MIN_PARALLEL_BYTES`] no threads are spawned at any
    /// budget, and the test would pass without ever entering the code it names.
    #[test]
    fn remember_output_dtype_is_deterministic_across_thread_counts() {
        let file_bytes = build_parallel_fp8_fixture();
        let tmp_in = std::env::temp_dir().join("test_remember_dtype_determinism.safetensors");
        std::fs::write(&tmp_in, &file_bytes).unwrap();
        let model = parse(&tmp_in).unwrap();

        for target in [TargetDtype::BF16, TargetDtype::F32, TargetDtype::F16] {
            let baseline = model
                .remember_to_bytes_with_options(target, RememberOptions::new().with_threads(1))
                .unwrap();

            for n in [1usize, 2, 4] {
                let out = model
                    .remember_to_bytes_with_options(target, RememberOptions::new().with_threads(n))
                    .unwrap();
                assert_eq!(
                    out, baseline,
                    "{target} output must be byte-identical at {n} threads"
                );
            }

            // The default (env-resolved) budget must agree too.
            let default_out = model.remember_to_bytes(target).unwrap();
            assert_eq!(
                default_out, baseline,
                "{target} default thread budget vs the sequential baseline"
            );
        }

        std::fs::remove_file(&tmp_in).ok();
    }
}
