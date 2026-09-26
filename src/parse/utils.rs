// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared parsing utilities used across the format parsers and the modules
//! built on them (`ParsedModel`, `lethe`).

/// Soft cap on `Vec` / `HashMap` pre-allocation sized from a file-declared
/// count, shared by the `GGUF`, `NPZ`, and `.pth` parsers.
///
/// A header or archive can declare a large entry count: a `GGUF` header up to
/// `MAX_TENSOR_COUNT` / `MAX_KV_COUNT` (1 M each), or a zip central directory
/// with millions of (possibly empty) entries. Passing that count straight to
/// `with_capacity` commits the whole hint eagerly — ~175 MB for `GGUF` at 1 M
/// (empirically 114 MB for the metadata `HashMap` + 61 MB for the
/// `Vec<GgufTensorInfo>`), and ~1–2× the file size for a many-entries zip,
/// before a single entry is read. Clamping every trust-the-count pre-allocation
/// to this constant bounds the worst-case eager hint to a few dozen KB; the
/// container still grows as entries are inserted, costing at most a handful of
/// extra reallocs on legitimate files — imperceptible given parse is I/O-bound
/// and real files never approach the cap.
#[cfg(any(feature = "npz", feature = "pth", feature = "gguf"))]
pub(crate) const PREALLOC_SOFT_CAP: usize = 256;

/// Reverses the byte order of each element in `data` in-place.
///
/// Each contiguous `element_size`-byte chunk is reversed, converting
/// big-endian to little-endian (or vice versa). Elements that are not
/// an exact multiple of `element_size` are left untouched (handled by
/// `chunks_exact_mut`).
// VECTORIZED: scalar fallback — chunk.reverse() on a runtime-variable
// element_size prevents auto-vectorization. This is the big-endian path
// (<0.01% of ML files), so scalar performance is acceptable.
// EXPLICIT: in-place mutation on the read buffer avoids allocating a second
// buffer of equal size. CONVENTIONS Rule 6 (separate in/out) is waived here
// because the data is already in a dedicated Vec<u8> that serves as the output.
#[cfg(any(feature = "npz", feature = "pth"))]
pub(crate) fn byteswap_inplace(data: &mut [u8], element_size: usize) {
    for chunk in data.chunks_exact_mut(element_size) {
        chunk.reverse();
    }
}

/// Classifies an `io::Error` raised while **decoding** an artefact's bytes
/// whose range has already been validated against a known source length.
///
/// Inside a validated range the bytes are all present, so a codec that
/// rejects them (`InvalidData` / `InvalidInput`) or runs out before its own
/// end marker (`UnexpectedEof`, e.g. a truncated `DEFLATE` stream) is a
/// property of the input: [`AnamnesisError::Parse`](crate::AnamnesisError::Parse),
/// matching what the slice-backed paths report for the same bytes. Every other
/// kind is a transport failure and stays
/// [`AnamnesisError::Io`](crate::AnamnesisError::Io).
///
/// **Not for raw streaming reads.** Where the source length is unknown (the
/// reader-generic safetensors header, which an `HTTP`-range adapter may back),
/// a stream that ends early stays `Io` by documented contract: a partial fetch
/// must be distinguishable from a malformed header. The rule, in full, is
/// stated on [`AnamnesisError`](crate::AnamnesisError).
#[cfg(any(feature = "npz", feature = "pth"))]
#[must_use]
pub(crate) fn classify_decode_error(e: std::io::Error, what: &str) -> crate::AnamnesisError {
    // EXHAUSTIVE: `io::ErrorKind` is a foreign `#[non_exhaustive]` enum; only
    // the three input-fault kinds are reclassified, every other kind is I/O.
    #[allow(clippy::wildcard_enum_match_arm)]
    match e.kind() {
        std::io::ErrorKind::UnexpectedEof
        | std::io::ErrorKind::InvalidData
        | std::io::ErrorKind::InvalidInput => crate::AnamnesisError::Parse {
            reason: format!("failed to decode {what}: {e}"),
        },
        _ => crate::AnamnesisError::Io(e),
    }
}

/// Rejects a tensor set that names one tensor twice.
///
/// A name is how every output format addresses a tensor, so two tensors with
/// one name make the set ambiguous: which one a consumer loads depends on the
/// writer. Worse, the upstream `safetensors` serializer indexes a name table
/// sized by the *distinct* names and panics on a duplicate (found by
/// `fuzz_convert_bytes`, v0.7.8). Called where names enter (the `.pth` state
/// dict) and before every write, so no path can hand the serializer one.
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`](crate::AnamnesisError::Parse) naming the
/// first duplicate in sorted order.
pub(crate) fn reject_duplicate_names<'a>(
    names: impl Iterator<Item = &'a str>,
) -> crate::Result<()> {
    let mut sorted: Vec<&str> = names.collect();
    sorted.sort_unstable();
    if let Some(dup) = sorted.windows(2).find_map(|pair| match pair {
        [a, b] if a == b => Some(*a),
        _ => None,
    }) {
        return Err(crate::AnamnesisError::Parse {
            reason: format!("duplicate tensor name `{dup}`"),
        });
    }
    Ok(())
}

/// Computes the product of a tensor's shape dimensions with overflow checking.
///
/// Returns `None` if the product overflows `usize` — an adversarial or
/// malformed header declaring dimensions whose product cannot fit (e.g.
/// `[usize::MAX, 2]`). An empty shape yields `Some(1)` (the scalar
/// element-count convention).
///
/// **A zero dimension yields `Some(0)` wherever it sits.** A plain
/// `try_fold` stops at the first overflowing dimension and never sees a
/// trailing zero, so `[2^33, 2^33, 0]` (a mathematically empty tensor) would
/// be reported as an overflow. `.pth` fixed that bug in v0.7.5; routing every
/// parser through this one helper is what keeps the formats from disagreeing
/// about it again.
///
/// Callers choose their own policy on `None`: a path that sizes an allocation
/// maps it to `AnamnesisError::Parse`, while a pure eligibility query treats
/// it as "not eligible". This is the
/// checked counterpart to a raw `shape.iter().product()`, which silently wraps
/// in release builds and panics in debug builds on overflow. For a display-only
/// estimate that must not fail, use [`saturating_num_elements_u64`].
#[must_use]
pub(crate) fn checked_num_elements(shape: &[usize]) -> Option<usize> {
    if shape.contains(&0) {
        return Some(0);
    }
    shape.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d))
}

/// Computes the product of a tensor's shape dimensions as a `u64`, saturating
/// at `u64::MAX` instead of failing.
///
/// For **size estimates only** (inspect summaries, dequantised-size figures),
/// where an absurd declared shape must read as "too big" to a policy gate
/// rather than abort the summary. Never use it to size an allocation: that is
/// [`checked_num_elements`]'s job, per `CONVENTIONS.md`'s rule that saturation
/// is not a substitute for a checked header-derived size.
///
/// The fold never short-circuits, so a zero dimension anywhere zeroes the
/// result even after an earlier saturation, matching [`checked_num_elements`].
#[must_use]
pub(crate) fn saturating_num_elements_u64(shape: &[usize]) -> u64 {
    // CAST: usize → u64, lossless widening of a header-declared dimension on
    // every supported target.
    #[allow(clippy::as_conversions)]
    shape
        .iter()
        .fold(1u64, |acc, &d| acc.saturating_mul(d as u64))
}

/// [`saturating_num_elements_u64`] narrowed to `usize`, saturating at
/// `usize::MAX` on a 32-bit target where the count does not fit.
///
/// Same estimate-only contract: never size an allocation with it.
#[must_use]
pub(crate) fn saturating_num_elements(shape: &[usize]) -> usize {
    usize::try_from(saturating_num_elements_u64(shape)).unwrap_or(usize::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_num_elements_basic() {
        assert_eq!(checked_num_elements(&[]), Some(1));
        assert_eq!(checked_num_elements(&[16384, 2304]), Some(16384 * 2304));
        assert_eq!(checked_num_elements(&[0, 5]), Some(0));
    }

    #[test]
    fn checked_num_elements_overflow_is_none() {
        // Two near-`usize::MAX` dims whose product cannot fit.
        assert_eq!(checked_num_elements(&[usize::MAX, 2]), None);
    }

    #[test]
    fn checked_num_elements_trailing_zero_after_overflow_is_zero() {
        // The product overflows before the zero is reached; the tensor is
        // still empty, and must be reported as such.
        assert_eq!(checked_num_elements(&[usize::MAX, usize::MAX, 0]), Some(0));
    }

    #[test]
    fn reject_duplicate_names_finds_repeats() {
        assert!(reject_duplicate_names(["a", "b", "c"].into_iter()).is_ok());
        assert!(reject_duplicate_names(std::iter::empty()).is_ok());
        let result = reject_duplicate_names(["b", "a", "b"].into_iter());
        assert!(
            matches!(&result, Err(e) if e.to_string().contains("duplicate tensor name `b`")),
            "{result:?}"
        );
    }

    #[test]
    fn saturating_num_elements_u64_saturates_and_zeroes() {
        assert_eq!(saturating_num_elements_u64(&[]), 1);
        assert_eq!(saturating_num_elements_u64(&[3, 4]), 12);
        assert_eq!(
            saturating_num_elements_u64(&[usize::MAX, usize::MAX]),
            u64::MAX
        );
        assert_eq!(saturating_num_elements_u64(&[usize::MAX, usize::MAX, 0]), 0);
    }

    #[cfg(any(feature = "npz", feature = "pth"))]
    #[test]
    fn byteswap_2byte() {
        let mut data = vec![0x01, 0x02, 0x03, 0x04];
        byteswap_inplace(&mut data, 2);
        assert_eq!(data, vec![0x02, 0x01, 0x04, 0x03]);
    }

    #[cfg(any(feature = "npz", feature = "pth"))]
    #[test]
    fn byteswap_4byte() {
        let mut data = vec![0x01, 0x02, 0x03, 0x04];
        byteswap_inplace(&mut data, 4);
        assert_eq!(data, vec![0x04, 0x03, 0x02, 0x01]);
    }

    #[cfg(any(feature = "npz", feature = "pth"))]
    #[test]
    fn byteswap_8byte() {
        let mut data = vec![0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
        byteswap_inplace(&mut data, 8);
        assert_eq!(data, vec![0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01]);
    }

    #[cfg(any(feature = "npz", feature = "pth"))]
    #[test]
    fn byteswap_1byte_is_noop() {
        let mut data = vec![0xAA, 0xBB, 0xCC];
        byteswap_inplace(&mut data, 1);
        assert_eq!(data, vec![0xAA, 0xBB, 0xCC]);
    }

    #[cfg(any(feature = "npz", feature = "pth"))]
    #[test]
    fn byteswap_empty() {
        let mut data: Vec<u8> = vec![];
        byteswap_inplace(&mut data, 4);
        assert!(data.is_empty());
    }
}
