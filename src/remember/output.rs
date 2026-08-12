// SPDX-License-Identifier: MIT OR Apache-2.0

//! The element type a dequantisation kernel writes.
//!
//! Every kernel in this crate computes in `f32` and narrows on the way out.
//! Until v0.7.3 that narrowing was hard-coded to `BF16`, which was assumed
//! rather than decided: no design note ever defended it. [`OutputElement`]
//! turns it into a caller-chosen parameter, monomorphised exactly like a C++
//! template argument, with a single runtime `match` at the public boundary so
//! the CLI and the Phase 8 Python bindings can still choose from a string.
//!
//! # Why this matters numerically
//!
//! `BF16` keeps 8 significand bits. A `Q8_0` value is an `f16` scale (11-bit
//! significand) times an `int8`, so it needs up to ~18 bits; `Q6_K` needs 24.
//! Measured on `SmolLM2-135M-Q4_K_M`, only **3 to 20 %** of dequantised values
//! are exactly `BF16`-representable and the rest are rounded, at up to half a
//! `BF16` `ULP` (`2⁻⁸`, about 0.39 % relative).
//!
//! The usual defence, that quantisation error dwarfs the rounding, holds for
//! `Q4_K` and below but **not** for the high-precision types: `Q8_0`'s own
//! quantisation step is about 1/254, the *same order* as `BF16`'s half-`ULP`.
//! For those types the crate was adding rounding comparable to the error the
//! format exists to avoid.
//!
//! [`F32Out`] is the only width that adds **no narrowing step of its own**, so
//! its output *is* the `f32` the reference implementation produces.
//!
//! # What "exact" does and does not mean
//!
//! Two senses get conflated, and only the first is what this module delivers:
//!
//! 1. **Exact against the reference.** [`F32Out`] removes anamnesis's own
//!    narrowing, so the emitted value is the `f32` that `gguf-py` and
//!    `ggml-quants.c` produce. This is the claim the cross-validation tests.
//! 2. **Exact in the real-number sense.** True for the pure-product kernels
//!    (`Q8_0`, `Q5_0`, and `Q6_K`, whose 11 + 7 + 6 = 24 significand bits land
//!    exactly on `f32`). **Not** guaranteed for the min-offset K-quants
//!    (`Q2_K`, `Q4_K`, `Q5_K`), whose final `d·q - dmin·m` subtracts two `f32`
//!    values at different exponents and can itself round. There, `f32` is the
//!    reference's own value, not a mathematically exact one.
//!
//! # Why the trait carries the loop, not a per-element hook
//!
//! [`OutputElement::write_scratch`] converts a whole block rather than one
//! element. Each implementation therefore owns a complete, monomorphic loop
//! with no generic indirection inside it, which is what lets each carry its own
//! `// VECTORIZED:` annotation and be verified independently in
//! `cargo-show-asm`. A per-element hook would put a slice-length check on every
//! element and leave one generic loop whose codegen would have to be inspected
//! three times anyway.

use crate::parse::safetensors::Dtype;
use crate::remember::fp8::f32_bits_to_bf16_bits;

/// Widest output element in bytes, over every [`OutputElement`] implementation.
///
/// The block runners in `remember::gguf` size their stack output buffers at
/// `QK × MAX_OUTPUT_BYTES` and then sub-slice to `QK × E::BYTES`, because
/// `[0u8; QK * E::BYTES]` would need `generic_const_exprs`, which is unstable.
/// The `const` block below proves this bound holds for all three types, so the
/// sub-slice can never be short. [`OutputElement`] being sealed is what makes
/// those three the complete set.
pub(crate) const MAX_OUTPUT_BYTES: usize = 4;

/// Compile-time proof that [`MAX_OUTPUT_BYTES`] really does bound every
/// implementation. Adding a wider output type without raising the constant
/// fails the build here rather than silently truncating tensor data.
const _: () = {
    assert!(Bf16Out::BYTES <= MAX_OUTPUT_BYTES);
    assert!(F32Out::BYTES <= MAX_OUTPUT_BYTES);
    assert!(F16Out::BYTES <= MAX_OUTPUT_BYTES);
};

/// Private supertrait that makes [`OutputElement`] sealed.
mod sealed {
    /// Implemented only by this module's three output markers.
    pub trait Sealed {}
}

/// The element type a dequantisation kernel writes.
///
/// Implemented by [`Bf16Out`], [`F32Out`] and [`F16Out`], and **sealed**: the
/// contract is a byte-level invariant that the crate's cross-validation depends
/// on (write exactly [`BYTES`](Self::BYTES) little-endian bytes per input
/// value, describing itself truthfully via [`DTYPE`](Self::DTYPE)), and an
/// outside implementation could violate it silently while every test stayed
/// green. Sealing is also the reversible direction: un-sealing later is not a
/// breaking change, sealing later would be.
///
/// The kernels never name this trait. They fill an `[f32; QK]` scratch buffer
/// and hand it to [`write_scratch`](Self::write_scratch), so all 24 `GGUF`
/// kernel functions are untouched by the choice of output type.
// The `sealed::Sealed` supertrait is deliberately private: that privacy *is*
// the seal. `#[doc(hidden)] pub mod sealed` would satisfy the lint but let an
// outside crate implement the trait, which is the thing being prevented.
#[allow(private_bounds)]
pub trait OutputElement: sealed::Sealed + Copy + Send + Sync + 'static {
    /// Bytes written per input value.
    const BYTES: usize;

    /// How the written bytes describe themselves in a `.safetensors` header.
    const DTYPE: Dtype;

    /// Converts a block's worth of `f32` values into `out`.
    ///
    /// This is the hot-path pass 2 for every kernel: contiguous reads,
    /// contiguous writes, no branches.
    ///
    /// # Preconditions
    ///
    /// `out.len() == src.len() × BYTES`. The implementations pair
    /// `src.iter()` with `out.chunks_exact_mut(BYTES)`, so a short `out`
    /// produces short output rather than a panic, and a long one leaves the
    /// tail untouched. Every caller in this crate sizes both buffers from the
    /// same block constant, so neither happens in practice.
    fn write_scratch(src: &[f32], out: &mut [u8]);
}

/// `BF16` output: 2 bytes per value, round-to-nearest-even. **The default**,
/// and the only width the crate emitted before v0.7.3.
///
/// The dtype the safetensors and Hugging Face ecosystem serves weights in, and
/// at 2 bytes per element it halves memory traffic on a path that is bandwidth
/// bound end to end. See the module docs for what that costs in precision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Bf16Out;

/// `F32` output: 4 bytes per value, and **no narrowing step at all**.
///
/// The kernels already compute in `f32`, so this writes the value they
/// computed. That makes the output bit-identical to the reference
/// implementation's own `f32`, which is what the phrase "removes a rounding
/// step" means precisely. It doubles output bytes on a bandwidth-bound path,
/// so expect it to be slower than [`Bf16Out`]; that is the honest cost of the
/// precision, not a defect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct F32Out;

/// `F16` output: 2 bytes per value, IEEE 754 binary16, round-to-nearest-even.
///
/// **`F16` is not uniformly the better 2-byte choice.** Against [`Bf16Out`] it
/// buys 3 significand bits (11 versus 8) and pays a far narrower exponent
/// range. `BF16` shares `f32`'s range; `F16` saturates at 65504 and flushes to
/// zero below about `2⁻²⁴`.
///
/// That range is reachable in real data, not just in theory: `MXFP4`'s `E8M0`
/// scale spans `2⁻¹²⁸` to `2¹²⁷`, and a `Q8_0` block whose `f16` scale is large
/// can exceed 65504 once multiplied by an `int8` of up to 127.
///
/// # Out-of-range policy
///
/// Plain IEEE semantics, via `half::f16::from_f32`: values above the maximum
/// become infinity, values below the smallest subnormal flush to zero, and
/// everything between rounds to nearest even. This is deliberately **not**
/// saturating. Saturation would keep outputs finite but would fabricate a
/// value no reference implementation produces, and would put the `F16`
/// cross-validation permanently at odds with `NumPy` and `PyTorch`, which both
/// produce infinity here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct F16Out;

impl sealed::Sealed for Bf16Out {}
impl sealed::Sealed for F32Out {}
impl sealed::Sealed for F16Out {}

impl OutputElement for Bf16Out {
    const BYTES: usize = 2;
    const DTYPE: Dtype = Dtype::BF16;

    #[inline]
    fn write_scratch(src: &[f32], out: &mut [u8]) {
        // VECTORIZED: pending cargo-show-asm verification on the scalar path.
        // Byte-for-byte the loop that shipped as `write_scratch_to_bf16` before
        // v0.7.3, so the default path's codegen is unchanged by construction.
        for (&val, out_chunk) in src.iter().zip(out.chunks_exact_mut(2)) {
            let bits = f32_bits_to_bf16_bits(val.to_bits());
            out_chunk.copy_from_slice(&bits.to_le_bytes());
        }
    }
}

impl OutputElement for F32Out {
    const BYTES: usize = 4;
    const DTYPE: Dtype = Dtype::F32;

    #[inline]
    fn write_scratch(src: &[f32], out: &mut [u8]) {
        // VECTORIZED: pending cargo-show-asm verification on the scalar path.
        // No conversion, only a little-endian store: this is the whole point of
        // `F32` output. On a little-endian target the loop is a memcpy the
        // compiler is free to recognise as one.
        for (&val, out_chunk) in src.iter().zip(out.chunks_exact_mut(4)) {
            out_chunk.copy_from_slice(&val.to_le_bytes());
        }
    }
}

impl OutputElement for F16Out {
    const BYTES: usize = 2;
    const DTYPE: Dtype = Dtype::F16;

    #[inline]
    fn write_scratch(src: &[f32], out: &mut [u8]) {
        // VECTORIZED: pending cargo-show-asm verification on the scalar path.
        // `f16::from_f32` is branch-free and, on targets with F16C, lowers to a
        // single `vcvtps2ph`; overflow to infinity and flush-to-zero are the
        // hardware's own behaviour, matching the documented IEEE policy.
        for (&val, out_chunk) in src.iter().zip(out.chunks_exact_mut(2)) {
            out_chunk.copy_from_slice(&half::f16::from_f32(val).to_le_bytes());
        }
    }
}

#[cfg(test)]
// `float_cmp` is allowed deliberately, not as a blanket concession: every
// comparison below asserts an *exactly representable* value, which is the
// property under test. An epsilon comparison would defeat the point, because a
// rounding bug that shifted one `ULP` is exactly what these tests exist to
// catch.
#[allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::float_cmp
)]
mod tests {
    use super::*;

    /// Reads a `BF16`/`F16` pair back as `f32` for assertions.
    fn le_u16(bytes: &[u8], i: usize) -> u16 {
        u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]])
    }

    #[test]
    fn bytes_and_dtype_agree() {
        // The invariant the sealed contract exists to protect: a type's
        // declared width must match what its dtype occupies in a file.
        assert_eq!(Bf16Out::BYTES, Bf16Out::DTYPE.byte_size());
        assert_eq!(F32Out::BYTES, F32Out::DTYPE.byte_size());
        assert_eq!(F16Out::BYTES, F16Out::DTYPE.byte_size());
    }

    #[test]
    fn f32_output_is_the_identity() {
        // The claim `F32Out` makes: no narrowing whatsoever, so every input bit
        // survives. Values chosen to have mantissa bits that BF16 would discard.
        let src = [1.0_f32, -2.5, 3.876_543_2, 1.234_567_9e-8, f32::MAX];
        let mut out = vec![0u8; src.len() * 4];
        F32Out::write_scratch(&src, &mut out);
        for (i, &val) in src.iter().enumerate() {
            let round_tripped =
                f32::from_le_bytes([out[i * 4], out[i * 4 + 1], out[i * 4 + 2], out[i * 4 + 3]]);
            assert_eq!(round_tripped.to_bits(), val.to_bits(), "value {i}");
        }
    }

    #[test]
    fn bf16_rounds_to_nearest_even() {
        // Exactly halfway between two BF16 values: the tie must go to the even
        // significand, not simply truncate. 0x3F800000 is 1.0; adding 0x8000
        // puts the value on the midpoint with an even LSB, so it rounds down.
        let tie_to_even = f32::from_bits(0x3F80_8000);
        let tie_to_odd = f32::from_bits(0x3F81_8000);
        let src = [tie_to_even, tie_to_odd];
        let mut out = vec![0u8; 4];
        Bf16Out::write_scratch(&src, &mut out);
        assert_eq!(le_u16(&out, 0), 0x3F80, "tie with even LSB rounds down");
        assert_eq!(
            le_u16(&out, 1),
            0x3F82,
            "tie with odd LSB rounds up to even"
        );
    }

    #[test]
    fn f16_follows_ieee_at_the_range_edges() {
        // The documented policy, asserted rather than assumed. These are the
        // values a saturating implementation would get wrong.
        let src = [
            1.0e10_f32,  // above F16 max: -> +inf
            -1.0e10_f32, // below F16 min: -> -inf
            1.0e-30_f32, // below smallest subnormal: -> +0
            65504.0_f32, // exactly F16 max: representable, stays finite
        ];
        let mut out = vec![0u8; src.len() * 2];
        F16Out::write_scratch(&src, &mut out);
        assert_eq!(half::f16::from_bits(le_u16(&out, 0)), half::f16::INFINITY);
        assert_eq!(
            half::f16::from_bits(le_u16(&out, 1)),
            half::f16::NEG_INFINITY
        );
        assert_eq!(f32::from(half::f16::from_bits(le_u16(&out, 2))), 0.0);
        assert_eq!(f32::from(half::f16::from_bits(le_u16(&out, 3))), 65504.0);
    }

    #[test]
    fn f16_keeps_precision_bf16_discards() {
        // Why F16 exists as an option: 11 significand bits against BF16's 8.
        // 1.0009766 needs 11 bits, so F16 holds it exactly and BF16 cannot.
        let val = 1.0_f32 + 1.0 / 1024.0;
        let mut half_bytes = vec![0u8; 2];
        let mut brain_bytes = vec![0u8; 2];
        F16Out::write_scratch(&[val], &mut half_bytes);
        Bf16Out::write_scratch(&[val], &mut brain_bytes);

        let half_value = f32::from(half::f16::from_bits(le_u16(&half_bytes, 0)));
        let brain_value = f32::from_bits(u32::from(le_u16(&brain_bytes, 0)) << 16);
        assert_eq!(half_value, val, "F16 represents this value exactly");
        assert_ne!(brain_value, val, "BF16 has to round it");
    }

    #[test]
    fn short_output_truncates_rather_than_panicking() {
        // The documented precondition-violation behaviour. `chunks_exact_mut`
        // stops early, so a caller that mis-sizes gets short output and never a
        // panic, which matters because the crate's lint floor denies panics.
        let src = [1.0_f32, 2.0, 3.0];
        let mut out = vec![0xAAu8; 4]; // room for two BF16 values, not three
        Bf16Out::write_scratch(&src, &mut out);
        assert_eq!(le_u16(&out, 0), 0x3F80);
        assert_eq!(le_u16(&out, 1), 0x4000);
    }
}
