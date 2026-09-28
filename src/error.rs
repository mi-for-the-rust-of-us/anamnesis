// SPDX-License-Identifier: MIT OR Apache-2.0

/// Errors produced by anamnesis operations.
///
/// Messages often quote the input (a tensor name, a pickle global, a ZIP entry
/// name), and the input may be hostile. Every variant's `Display` therefore
/// renders its text with control and invisible formatting characters escaped
/// (`\u{1b}` for `ESC`, and so on) and cut to 2048 characters, so that printing
/// or logging an error cannot rewrite a terminal or forge a log line. The
/// variants' fields hold the text as the crate produced it; render through
/// `Display` (or escape the fields yourself) before showing it to a person.
///
///
/// # Rust → Python exception mapping (frozen for the Phase 8 bindings)
///
/// The `PyO3` bindings expose each variant as a distinct, catchable Python
/// exception under a common base, so a multi-tenant host can answer (for
/// example) *413* for a budget breach, *400* for malformed input, and a flagged
/// security event for a hostile pickle — never a dead worker:
///
/// | `AnamnesisError` | Python exception |
/// |---|---|
/// | `Parse` | `ParseError` |
/// | `Unsupported` | `UnsupportedError` |
/// | `LimitExceeded` | `LimitExceededError` |
/// | `DisallowedGlobal` | `SecurityError` |
/// | `Cancelled` | builtin `KeyboardInterrupt` |
/// | `Io` | builtin `OSError` |
///
/// `ParseError` / `UnsupportedError` / `LimitExceededError` / `SecurityError`
/// all subclass a base `AnamnesisError(Exception)`. The wiring lands in Phase 8;
/// this table is the contract it implements.
///
/// # `Parse` versus `Io` on a short or corrupt input
///
/// The two map to different Python exceptions, so which one a bad input
/// produces is part of the contract, not an implementation detail:
///
/// - **A declared range past a known source length is `Parse`.** Every path
///   that knows the length checks header-derived offsets against it before
///   reading: the slice-backed parsers, the `GGUF` reader, and the `ZIP`
///   reader under `.npz` / `.pth` (on both its slice and its `Read + Seek`
///   source).
/// - **A codec rejecting bytes inside a validated range is `Parse`**, e.g. a
///   corrupt or truncated `DEFLATE` `.pth` entry: the bytes are all present
///   and malformed.
/// - **A streaming `Read` of unknown length that ends early is `Io`.** The
///   reader-generic safetensors header takes a plain `Read`, which an
///   `HTTP`-range adapter may back, and a partial fetch must stay
///   distinguishable from a malformed header. The same holds for any genuine
///   transport failure on any path.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum AnamnesisError {
    /// A format decoding failure (malformed header, invalid tensor metadata,
    /// truncated stream, out-of-bounds offset, or arithmetic overflow on a
    /// header-derived value). Maps to Python `ParseError`.
    #[error("parse error: {}", crate::parse::utils::error_text(.reason))]
    Parse {
        /// Human-readable description of what went wrong.
        reason: String,
    },

    /// A recognized but unimplemented format or feature. Maps to Python
    /// `UnsupportedError`.
    #[error(
        "unsupported format `{}`: {}",
        crate::parse::utils::error_text(.format),
        crate::parse::utils::error_text(.detail)
    )]
    Unsupported {
        /// The format name (e.g., `"GPTQ"`, `"safetensors"`).
        format: String,
        /// What specifically is not supported.
        detail: String,
    },

    /// A declared or derived size, count, or ratio exceeded a resource budget —
    /// either a caller-supplied [`ParseLimits`](crate::ParseLimits) axis or a
    /// permanent per-format floor (`MAX_PKL_SIZE`, the `GGUF` `MAX_*` family, the
    /// vendored-`ZIP` entry cap, …). Distinct from [`Self::Parse`] so an
    /// untrusted-input host can treat "too big for my budget" (e.g. *413 Payload
    /// Too Large*) differently from "malformed" (*400*). Maps to Python
    /// `LimitExceededError`.
    #[error("limit exceeded ({limit}): {}", crate::parse::utils::error_text(.message))]
    LimitExceeded {
        /// Stable machine-readable tag naming the breached limit — the axis or
        /// constant name (e.g. `"max_single_alloc_bytes"`, `"max_total_bytes"`,
        /// `"max_item_count"`, `"max_decompression_ratio"`, `"MAX_PKL_SIZE"`).
        limit: &'static str,
        /// Human-readable detail, including the offending value and the cap.
        message: String,
    },

    /// A `.pth` pickle stream referenced a `GLOBAL` outside the `torch.*`
    /// security allowlist — a potential arbitrary-code-execution vector that the
    /// VM refuses to interpret. A dedicated variant (not [`Self::Parse`]) so a
    /// host can log / alert on a potentially hostile upload distinctly from a
    /// merely malformed one. Maps to Python `SecurityError`.
    #[error(
        "disallowed pickle global `{}.{}` (potential code execution)",
        crate::parse::utils::error_text(.module),
        crate::parse::utils::error_text(.name)
    )]
    DisallowedGlobal {
        /// The referenced module (e.g. `"posix"`), cut to its first 48
        /// characters: the name comes from the file, and a crafted one can be
        /// megabytes long.
        module: String,
        /// The referenced attribute / callable (e.g. `"system"`), cut like
        /// `module`.
        name: String,
    },

    /// The caller asked an in-flight `remember` or `convert` to stop, through a
    /// `CancelToken` on its options.
    ///
    /// Not a failure of the input: the artefact may be perfectly valid and the
    /// same call may succeed on a retry. A dedicated variant (not
    /// [`Self::Parse`]) so a host can tell a user-initiated abort from a bad
    /// file, and so a Python binding can map it to `KeyboardInterrupt` rather
    /// than to a `ParseError` that would misreport what happened.
    ///
    /// **No output file is written.** Every path builds its result in memory
    /// before serialising, so the cancellation check happens strictly before
    /// any byte reaches the filesystem; there is nothing to clean up.
    #[error("operation cancelled by the caller")]
    Cancelled,

    /// A file system error. Maps to Python builtin `OSError`.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

impl From<safetensors::SafeTensorError> for AnamnesisError {
    fn from(e: safetensors::SafeTensorError) -> Self {
        Self::Parse {
            reason: format!("failed to parse safetensors header: {e}"),
        }
    }
}

// Note: as of Phase 6.12 the `.pth` / `.npz` parsers no longer call the `zip`
// crate at runtime (they use the vendored `crate::parse::zip` reader, which
// returns `AnamnesisError` directly), so there is no `From<zip::result::ZipError>`
// bridge. `zip` is a dev-dependency only (test fixtures + the differential
// oracle).

/// A convenience alias for `Result<T, AnamnesisError>`.
pub type Result<T> = std::result::Result<T, AnamnesisError>;
