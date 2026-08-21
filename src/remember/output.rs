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
//!
//! **v0.7.4 tested that claim rather than inheriting it, and it held.** The
//! four `remember` families narrow inside their hot loops, so routing them
//! through `write_scratch` costs an `f32` intermediate the `GGUF` kernels never
//! pay. A `write_one(value: f32, out: &mut [u8])` hook was implemented in full
//! and benchmarked against the pre-v0.7.4 baseline binary, interleaved, on an
//! idle machine:
//!
//! | family | `write_one` | with `write_scratch` + register tiles |
//! |---|---:|---:|
//! | `AWQ` | 0.98× | 1.04× |
//! | `BnB` `NF4` | 1.00× | 1.04× |
//! | `BnB` `INT8` | 1.01× | 1.09× |
//! | `FP8` fine-grained | **1.46×** | 1.06× |
//! | `GPTQ` `INT4` | **4.25×** | 1.09× |
//!
//! Three kernels reached parity and two collapsed. `--emit=asm` on the
//! `Bf16Out` monomorphisations shows why: `GPTQ`'s pass 2 emitted scalar
//! `vsubss` / `vmulss` where it had emitted `vsubps` / `vmulps` on `%ymm`, and
//! `FP8` likewise fell to scalar `vmulss`. The `copy_from_slice` inside
//! `write_one` carries a length check that `LLVM` eliminates for some callers
//! and not others, because the chunk width comes from the generic `E::BYTES`
//! rather than the literal the pre-v0.7.4 loops used. **Only the implementation
//! can supply a literal chunk width**, which is exactly the property
//! `write_scratch` has and a per-element hook cannot.
//!
//! So the trade is a uniform, predictable 1.04–1.09× against a bimodal
//! 0.98×–4.25×, and the design note stands. See `docs/perf-experiments.md` for
//! the full numbers.

// MEASURED-REVERT: clippy::chunks_exact_to_as_chunks (new in Rust 1.98).
//
// **SETTLED by its own measurement, not inherited from a family revert.** This
// is the narrowing writer every dequant kernel funnels through, so v0.7.6 could
// only revert it *with* the `FP8` family and say so. Phase 7.7 item 2 measured
// it directly, migrating all three `write_scratch` loops to `as_chunks_mut`,
// paired harness on x86-64, two runs:
//
//   bnb_int8 BF16   +37.7 %  /  +31.7 %     <- the default width, badly hurt
//   bnb_int8 F32    +17.8 %  /  +13.4 %
//   gptq_int4 F16   -10.6 %  /  -10.0 %     <- a real gain, and far smaller
//
// Everything else moved inconsistently between the two runs, i.e. noise --
// including every `gguf_q4_k` arm, which is worth recording because all 24
// `GGUF` kernel functions reach their output through this writer and yet showed
// nothing.
//
// A +32 % regression on `BnB` `INT8`'s default width is not paid for by a 10 %
// gain on `GPTQ`'s least-used one, so the migration is not taken. Note what the
// shape of that result says: this change *helps one kernel and badly hurts
// another*, which is the same per-kernel unpredictability Phase 7.7 item 1
// found between the `GPTQ` and `AWQ` twins. There is no shared mechanism to
// reason from here, only measurements.
//
// Each `write_scratch` also carries a `// VECTORIZED: confirmed` annotation
// earned by reading its disassembly; those claims would need re-establishing
// before the iterator shape changes.
#![allow(clippy::chunks_exact_to_as_chunks)]

use crate::parse::safetensors::Dtype;
use crate::remember::fp8::f32_bits_to_bf16_bits;

/// Elements a fused-narrowing kernel hands to [`OutputElement::write_scratch`]
/// per call.
///
/// **Calibrated, not guessed, and the reason v0.7.4's split is nearly free.**
/// The four `remember` families (`FP8`, `GPTQ`, `AWQ`, `BnB`) narrowed inside
/// their hot loops before v0.7.4. Splitting that into an arithmetic pass plus a
/// narrowing pass introduces an `f32` intermediate, and where that intermediate
/// goes decides the whole cost:
///
/// - **Row-sized** (the first draft: `out_features × 4` = 44 KB at
///   `out_features = 11008`) the `f32`s reach memory between the passes.
///   Measured 1.115× against the pre-v0.7.4 fused kernel on `BnB` `INT8`.
/// - **Register-sized** (this constant) they stay in `ymm` registers, because
///   32 `f32`s is four AVX2 vectors and both loops fully unroll. `FP8`
///   fine-grained went from 1.43× to **1.06×** across this change plus hoisting
///   its buffer out of the per-block call.
///
/// 32 rather than 8 is measured too: an 8-element tile left more loop-setup
/// overhead per element than the register pressure it saved.
///
/// Not feature-gated: the always-on `FP8` family uses it, so it is live in
/// every build. Contrast `MAX_OUTPUT_BYTES`, which is a `GGUF`-only concept
/// (and so `gguf`-gated, which is why this is a plain code span).
pub(crate) const VECTOR_TILE: usize = 32;

/// Widest output element in bytes, over every [`OutputElement`] implementation.
///
/// The block runners in `remember::gguf` size their stack output buffers at
/// `QK × MAX_OUTPUT_BYTES` and then sub-slice to `QK × E::BYTES`, because
/// `[0u8; QK * E::BYTES]` would need `generic_const_exprs`, which is unstable.
/// The `const` block below proves this bound holds for all three types, so the
/// sub-slice can never be short. [`OutputElement`] being sealed is what makes
/// those three the complete set.
///
/// Feature-gated with its consumer. Those two runners are the only code that
/// needs it, so without `gguf` it is genuinely dead — and on the MSRV
/// toolchain, provably so: rustc 1.88's dead-code analysis does **not** count
/// the reference from the `const` assertion block below as a use, while current
/// stable does. Gating it here rather than reaching for `#[allow(dead_code)]`
/// keeps the lint meaningful.
///
/// v0.7.3 expected v0.7.4 to widen this gate once `FP8` / `GPTQ` / `AWQ` /
/// `BnB` went generic. **It did not, and the gate is correct as it stands.**
/// Those four families tile through an **`f32` scratch** and hand it to
/// [`OutputElement::write_scratch`], so their scratch is sized in `f32`s and
/// never in output bytes; only the `GGUF` runners build a byte tile up front,
/// because their kernels write into a fixed `[f32; QK]` and the output buffer
/// has to be materialised beside it. The constant stays a `GGUF` concept.
#[cfg(feature = "gguf")]
pub(crate) const MAX_OUTPUT_BYTES: usize = 4;

/// Compile-time proof that [`MAX_OUTPUT_BYTES`] really does bound every
/// implementation. Adding a wider output type without raising the constant
/// fails the build here rather than silently truncating tensor data.
///
/// Gated alongside the constant: the bound only guards buffers that exist in a
/// `gguf` build.
#[cfg(feature = "gguf")]
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
// The `sealed::Sealed` supertrait lives in a private module: that privacy *is*
// the seal. A `#[doc(hidden)] pub mod sealed` would still let an outside crate
// name and implement `Sealed`, which is the thing being prevented.
//
// No `#[allow(private_bounds)]` here: the lint does not fire on this shape, and
// carrying an allow for a lint that never triggers would tell the next reader
// there is a suppression to preserve when there is not. Verified by removing it
// and rebuilding clean.
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

    /// One whole tile's worth of output bytes: `[u8; VECTOR_TILE * BYTES]`.
    ///
    /// Spelled as an associated type rather than computed, because
    /// `[u8; VECTOR_TILE * Self::BYTES]` needs `generic_const_exprs`: stable
    /// Rust rejects an associated const of a *type parameter* inside a const
    /// operation, with "generic parameters may not be used in const
    /// operations". Each implementation writes its own width as a literal, and
    /// the `const` block below proves every one of them equals
    /// `VECTOR_TILE * BYTES`, so the three literals cannot drift.
    ///
    /// `VECTOR_TILE` is a plain code span rather than an intra-doc link
    /// throughout these three items: it is `pub(crate)` while this trait is
    /// `pub`, and linking down that visibility step trips
    /// `private_intra_doc_links` under `-D warnings`. See `CONVENTIONS.md`
    /// section *Intra-Doc Link Safety*.
    type Tile;

    /// Splits an output row into whole [`Tile`](Self::Tile)s plus the ragged
    /// edge run, in that order.
    ///
    /// Exists so that no caller has to name `VECTOR_TILE * BYTES`, which is
    /// exactly the const expression stable Rust will not accept from generic
    /// code. Each implementation is a single `as_chunks_mut` call at a literal
    /// width.
    ///
    /// # Preconditions
    ///
    /// None. A row shorter than one tile yields an empty tile slice and the
    /// whole row as the edge run, which is what the ragged-tail path expects.
    fn split_tiles(out: &mut [u8]) -> (&mut [Self::Tile], &mut [u8]);

    /// Converts exactly one tile of `f32` values into `out`.
    ///
    /// The fixed-size counterpart to [`write_scratch`](Self::write_scratch).
    /// Both operands carry their length in the type, so the narrowing loop has
    /// a compile-time-constant trip count and needs no length reconciliation
    /// between source and destination. Use this for whole tiles and
    /// [`write_scratch`](Self::write_scratch) for the ragged edge, whose length
    /// is only known at run time.
    fn write_tile(src: &[f32; VECTOR_TILE], out: &mut Self::Tile);
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
///
/// **Measured, and it is cheaper than the byte ratio implies.** Across the
/// seven dequant families on x86-64 (criterion medians, 4096 x 11008), `F32`
/// costs **1.09x to 1.63x** [`Bf16Out`] rather than the 2x its output bytes
/// would suggest, because the narrowing step [`Bf16Out`] performs is itself
/// work that `F32` simply does not do. Widest is `gguf_q4_k` at 1.63x,
/// narrowest `fp8_per_tensor` at 1.09x. Compare [`F16Out`], which is the *same
/// width as `BF16`* and yet costs 2.02x to 3.11x.
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
///
/// # Cost: expect **2x to 3x** [`Bf16Out`], at the same output width
///
/// This is the surprising one, and until Phase 7.7 nothing here said it. `F16`
/// and `BF16` are both 2 bytes per element, so there is no bandwidth story to
/// explain it, unlike [`F32Out`]. The cost is the *conversion*.
///
/// Measured across every dequant family, both architectures, whole-kernel wall
/// clock (so the narrowing is only part of each figure):
///
/// | Kernel | `BF16` | `F16` | ratio |
/// |---|---:|---:|---:|
/// | `gguf_q4_k` | 25.70 ms | 79.87 ms | **3.11x** |
/// | `bnb_int8` | 24.84 ms | 76.95 ms | **3.10x** |
/// | `gptq_int4` | 28.78 ms | 78.04 ms | **2.71x** |
/// | `awq_int4` | 38.20 ms | 101.10 ms | **2.65x** |
/// | `fp8_fine_grained` | 43.19 ms | 107.28 ms | **2.48x** |
/// | `bnb_nf4` | 45.43 ms | 112.04 ms | **2.47x** |
/// | `fp8_per_tensor` | 46.55 ms | 94.20 ms | **2.02x** |
///
/// x86-64, criterion medians from `benches/dequant.rs`, 4096 x 11008.
/// `aarch64` walltime agrees on the same seven kernels at **2.10x to 2.93x**,
/// so this is not one platform's quirk.
///
/// *Criterion rather than the paired harness, deliberately.* A ratio between
/// two output widths is a **within-binary** comparison: both arms are compiled
/// into the same binary, so there is no code-layout term between them, and a
/// 100-sample median is the right statistic for an absolute magnitude.
/// `benches/ab.rs` is built to resolve *differences between two binaries* and
/// samples adaptively for that; its printed absolute times swing 50-100 % run
/// to run and must not be quoted.
///
/// **Two different mechanisms produce the same magnitude**, which is why the
/// obvious fix is not one:
///
/// - **x86-64**: `write_scratch` here *is* inlined, and does use the `F16C`
///   hardware instruction, but `vcvtps2ph` takes a 128-bit source — **4 lanes**
///   where [`Bf16Out`]'s bias-add-and-shift runs **8** on `%ymm`. Half the
///   throughput per instruction on the narrowing step, from the hardware.
/// - **`aarch64`**: `write_scratch` here is **not inlined at all**, unlike
///   [`Bf16Out`]'s and [`F32Out`]'s — an out-of-line call per 32-element tile.
///
/// So **recovering the `aarch64` inline would not close this gap**: x86-64
/// already has the inline and still pays 2x to 3x. The conversion is the cost,
/// not the call.
///
/// **Not measured on Apple Silicon.** The `aarch64` figures come from Linux
/// server-class bare metal. M-series parts have `ARMv8.2` hardware `FP16`
/// arithmetic and a materially different microarchitecture, so the ratio there
/// is an open question — though both platforms measured so far agree, which
/// makes the direction unlikely to reverse.
///
/// None of this is an argument against `F16`. It is the right choice when you
/// need its 3 extra significand bits and your data fits its exponent range.
/// It is an argument against choosing it *believing it is free* because it is
/// the same width as the default.
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
        // VECTORIZED: confirmed AVX2 vpaddd + vpsrld + vpand + vmovdqu on
        // %ymm (8-wide) in `--emit=asm`, x86-64 target-cpu=native,
        // opt-level=3. That is the round-to-nearest-even bias-add and shift
        // running eight lanes at a time. Measured at 20.300 ms median
        // (criterion, `dequant_gguf_q4_k/synthetic_4096x11008`, 4096x11008
        // Q4_K, target-cpu=native), no regression against the pre-generic
        // baseline. Byte-for-byte the loop that shipped as
        // `write_scratch_to_bf16` before v0.7.3, so the default path's codegen
        // is unchanged by construction.
        for (&val, out_chunk) in src.iter().zip(out.chunks_exact_mut(2)) {
            let bits = f32_bits_to_bf16_bits(val.to_bits());
            out_chunk.copy_from_slice(&bits.to_le_bytes());
        }
    }

    type Tile = [u8; VECTOR_TILE * 2];

    #[inline]
    fn split_tiles(out: &mut [u8]) -> (&mut [Self::Tile], &mut [u8]) {
        out.as_chunks_mut::<{ VECTOR_TILE * 2 }>()
    }

    #[inline]
    fn write_tile(src: &[f32; VECTOR_TILE], out: &mut Self::Tile) {
        // VECTORIZED: confirmed AVX2 vpaddd + vpsrld + vpand on %ymm in
        // `--emit=asm`, x86-64 target-cpu=native, opt-level=3, read at the
        // inlined call site inside `dequantize_gptq::<Bf16Out>` (this function
        // is `#[inline]` and has no standalone symbol). That is the same
        // round-to-nearest-even bias-add and shift `write_scratch` emits, so
        // the fixed-size form keeps the narrowing that the byte-slice form had.
        //
        // The arithmetic is deliberately identical to `write_scratch` above.
        // Only the operand types differ: both are fixed-size arrays, so the zip
        // has a constant trip count of `VECTOR_TILE` rather than one reconciled
        // from two run-time lengths, and assigning the `[u8; 2]` whole replaces
        // a length-checked `copy_from_slice`. Measured effect at the call site:
        // 1088 -> 939 instructions for the whole `Bf16Out` monomorphisation
        // (-13.7 %), and -9.97 % wall clock. See `gptq.rs` for the numbers.
        for (&val, out_chunk) in src.iter().zip(out.as_chunks_mut::<2>().0) {
            *out_chunk = f32_bits_to_bf16_bits(val.to_bits()).to_le_bytes();
        }
    }
}

impl OutputElement for F32Out {
    const BYTES: usize = 4;
    const DTYPE: Dtype = Dtype::F32;

    #[inline]
    fn write_scratch(src: &[f32], out: &mut [u8]) {
        // VECTORIZED: confirmed AVX2 vmovups on %ymm (8-wide, 256-bit stores)
        // in `--emit=asm`, x86-64 target-cpu=native, opt-level=3. No
        // conversion, only a little-endian store: this is the whole point of
        // `F32` output, and on a little-endian target the compiler recognises
        // the loop as a wide copy. Measured at 36.298 ms median (criterion,
        // `dequant_gguf_q4_k/synthetic_4096x11008_f32`, same fixture), i.e.
        // 1.79x the `BF16` arm against 2.0x of output bytes: the cost is the
        // doubled write, not a failure to vectorise.
        for (&val, out_chunk) in src.iter().zip(out.chunks_exact_mut(4)) {
            out_chunk.copy_from_slice(&val.to_le_bytes());
        }
    }

    type Tile = [u8; VECTOR_TILE * 4];

    #[inline]
    fn split_tiles(out: &mut [u8]) -> (&mut [Self::Tile], &mut [u8]) {
        out.as_chunks_mut::<{ VECTOR_TILE * 4 }>()
    }

    #[inline]
    fn write_tile(src: &[f32; VECTOR_TILE], out: &mut Self::Tile) {
        // VECTORIZED: confirmed AVX2 vmovups on %ymm (256-bit stores) in
        // `--emit=asm`, x86-64 target-cpu=native, opt-level=3, read at the
        // inlined call site as for `Bf16Out`. The fixed-size form emits **more**
        // wide stores than the byte-slice one, not fewer: across the whole
        // `F32Out` monomorphisation `vmovups` goes 64 -> 95 and `%ymm` mentions
        // 96 -> 160, on 979 -> 919 instructions (-6.1 %).
        for (&val, out_chunk) in src.iter().zip(out.as_chunks_mut::<4>().0) {
            *out_chunk = val.to_le_bytes();
        }
    }
}

impl OutputElement for F16Out {
    const BYTES: usize = 2;
    const DTYPE: Dtype = Dtype::F16;

    #[inline]
    fn write_scratch(src: &[f32], out: &mut [u8]) {
        // VECTORIZED: confirmed F16C `vcvtps2ph $0, %xmm, %xmm` (4-wide packed)
        // in `--emit=asm`, x86-64 target-cpu=native, opt-level=3. Narrower than
        // the other two impls' 8-wide `%ymm` work because `vcvtps2ph` takes a
        // 128-bit source here, which is a hardware property rather than a
        // missed vectorisation. The `$0` immediate is round-to-nearest-even, so
        // the documented rounding is enforced by the instruction itself, and
        // overflow to infinity and flush-to-zero are the hardware's own
        // behaviour, matching the documented IEEE policy exactly.
        for (&val, out_chunk) in src.iter().zip(out.chunks_exact_mut(2)) {
            out_chunk.copy_from_slice(&half::f16::from_f32(val).to_le_bytes());
        }
    }

    type Tile = [u8; VECTOR_TILE * 2];

    #[inline]
    fn split_tiles(out: &mut [u8]) -> (&mut [Self::Tile], &mut [u8]) {
        out.as_chunks_mut::<{ VECTOR_TILE * 2 }>()
    }

    #[inline]
    fn write_tile(src: &[f32; VECTOR_TILE], out: &mut Self::Tile) {
        // VECTORIZED: confirmed F16C `vcvtps2ph` in `--emit=asm`, x86-64
        // target-cpu=native, opt-level=3, read at the inlined call site as for
        // `Bf16Out`. The `$0` immediate is round-to-nearest-even, so the
        // documented rounding stays enforced by the instruction itself. The
        // count drops 41 -> 9 across the whole `F16Out` monomorphisation while
        // total instructions drop 1108 -> 906 (-18.2 %): the same conversions
        // expressed in fewer, wider ones rather than conversions lost.
        for (&val, out_chunk) in src.iter().zip(out.as_chunks_mut::<2>().0) {
            *out_chunk = half::f16::from_f32(val).to_le_bytes();
        }
    }
}

/// Proves each implementation's `Tile` really is `VECTOR_TILE * BYTES` wide.
///
/// The three widths are literals because stable Rust will not let generic code
/// compute them (see [`OutputElement::Tile`]), so nothing in the type system
/// ties them to `BYTES`. This block does, at compile time: change `BYTES`
/// without changing the literal, or add a fourth output type whose `Tile` is
/// wrong, and the build fails here rather than truncating tensor data.
///
/// Not feature-gated, unlike the `MAX_OUTPUT_BYTES` block above: `VECTOR_TILE`
/// is always-on, so these tiles exist in every build.
const _: () = {
    assert!(size_of::<<Bf16Out as OutputElement>::Tile>() == VECTOR_TILE * Bf16Out::BYTES);
    assert!(size_of::<<F32Out as OutputElement>::Tile>() == VECTOR_TILE * F32Out::BYTES);
    assert!(size_of::<<F16Out as OutputElement>::Tile>() == VECTOR_TILE * F16Out::BYTES);
};

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
