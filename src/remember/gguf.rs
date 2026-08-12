// SPDX-License-Identifier: MIT OR Apache-2.0

//! `GGUF` K-quant dequantization — scalar reference kernels producing `BF16`.
//!
//! This module consumes raw block-encoded tensor bytes (as produced by
//! [`parse_gguf`](crate::parse::gguf::parse_gguf)) and decodes them into
//! `BF16` output bytes, matching the output format of the `FP8`, `GPTQ`,
//! `AWQ`, and `BitsAndBytes` dequantisers.
//!
//! # Supported types
//!
//! - **Legacy block quants** (32-element blocks): `Q4_0`, `Q4_1`, `Q5_0`,
//!   `Q5_1`, `Q8_0`, `Q8_1`, `IQ4_NL`, `MXFP4`.
//! - **K-quants** (256-element super-blocks): `Q2_K`, `Q3_K`, `Q4_K`,
//!   `Q5_K`, `Q6_K`, `Q8_K`, `IQ4_XS`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`,
//!   `IQ3_XXS`, `IQ3_S`, `IQ1_S`, `IQ1_M`, `TQ1_0`, `TQ2_0`.
//!
//! Phase 4.5 closed in step 6 with `MXFP4` — every block-quantised
//! `GgufType` variant has a dedicated kernel. The dispatcher's only
//! remaining `Unsupported` path is for scalar (non-block-quant) types
//! that are reinterpret-only and never reach the dequantisation layer.
//!
//! # Two public entry points
//!
//! - [`dequantize_gguf_to_bf16`] materialises the whole tensor into an
//!   owned `Vec<u8>`. Convenient for small-to-medium tensors (anything
//!   that comfortably fits in RAM alongside its `BF16` expansion).
//! - [`dequantize_gguf_blocks_to_bf16`] is the **streaming** variant: the
//!   caller supplies a sink closure that receives one block's worth of
//!   `BF16` bytes at a time (64 B for legacy, 512 B for K-quants), so
//!   peak heap is O(one scratch buffer) regardless of tensor size. Use
//!   this for dequantising 70 B-parameter models on modest RAM.
//!
//! Both entry points share the same validation logic and the same scalar
//! kernels; the `Vec`-returning variant is a thin wrapper that pushes
//! into a pre-sized `Vec::with_capacity` sink.
//!
//! # Algorithm
//!
//! Every kernel follows the two-pass loop-fission pattern from
//! `CONVENTIONS.md`: for each block,
//!
//! 1. **Pass 1** unpacks the packed-bit storage into a stack-resident
//!    `[f32; QK]` scratch buffer (one block at a time, L1-resident).
//! 2. **Pass 2** walks the scratch buffer paired with a per-block
//!    `[u8; QK × 2]` output buffer and emits `BF16` bytes via the shared
//!    [`f32_bits_to_bf16_bits`](crate::remember::fp8) helper. The inner
//!    loop is branch-free and `chunks_exact_mut(2)`-based, giving the
//!    compiler a clean vectorisation target.
//!
//! The formulas are ported verbatim from `ggml-quants.c`'s scalar
//! `dequantize_row_*` reference implementations. Bit-for-bit
//! cross-validation against `llama.cpp` output is Phase 4 step 4.
//!
//! # Output format
//!
//! All kernels emit `BF16` bytes in little-endian order, length
//! `n_elements × 2`. Round-to-nearest-even is performed inside
//! `f32_bits_to_bf16_bits`.

use crate::error::AnamnesisError;
use crate::parse::gguf::GgufType;
use crate::remember::fp8::f32_bits_to_bf16_bits;
use iq_grids::{
    IQ1S_DELTA, IQ1S_GRID, IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID, KMASK_IQ2XS,
    KSIGNS_IQ2XS,
};

// ---------------------------------------------------------------------------
// Block-size constants
// ---------------------------------------------------------------------------

/// Element count per legacy block quant (`Q4_0`..`Q8_1`, `IQ4_NL`).
const QK_SMALL: usize = 32;

/// Element count per K-quant super-block (`Q2_K`..`Q8_K`, `IQ4_XS`).
const QK_K: usize = 256;

/// Powers of 3 used for the base-3 packing trick in `TQ1_0`.
///
/// After `byte * POW3_TQ1_0[n]` (wrapping `u8` arithmetic), the n-th
/// ternary digit of the byte's encoded value lives in the high bits and
/// is recovered by `(_ * 3) >> 8`. The 6th entry (`243`) is unused by
/// the decoder but kept for parity with `ggml-quants.c`'s `pow3[6]`
/// declaration.
const POW3_TQ1_0: [u8; 6] = [1, 3, 9, 27, 81, 243];

/// Non-linear 4-bit codebook shared by `IQ4_NL` and `IQ4_XS`.
///
/// Ported verbatim from `ggml-common.h::kvalues_iq4nl`. Each `IQ4_*` 4-bit
/// storage nibble indexes this table to recover a signed `i8` quant value
/// before the per-block `f32` scale is applied.
const K_VALUES_IQ4_NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// `MXFP4` codebook — 16 signed `i8` values that are **2× the OCP E2M1
/// magnitudes**. The doubling is cancelled by the `_HALF` factor inside
/// [`e8m0_to_fp32_half`], so the final dequantised value matches the
/// raw OCP E2M1 spec.
///
/// Ported verbatim from `ggml-common.h::kvalues_mxfp4`. Indexed by the
/// 4-bit storage nibble (`0..16`).
const K_VALUES_MXFP4: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

// ---------------------------------------------------------------------------
// Infallible byte readers
// ---------------------------------------------------------------------------

/// Reads a `ggml_half` (IEEE 754 binary16) from a 2-byte array and widens
/// it to `f32`. Infallible by construction — callers slice fixed-length
/// arrays out of their already-validated block slices.
#[inline]
fn read_f16_bytes(bytes: [u8; 2]) -> f32 {
    f32::from(half::f16::from_le_bytes(bytes))
}

/// Reads a little-endian `f32` from a 4-byte array. Used only by the
/// `Q8_K` block, whose super-block scale is `f32` rather than `f16`.
#[inline]
fn read_f32_bytes(bytes: [u8; 4]) -> f32 {
    f32::from_le_bytes(bytes)
}

/// Decodes a 1-byte E8M0 exponent into the `MXFP4` per-block scale (the
/// `_HALF` variant from `ggml-impl.h`).
///
/// Returns `2^(e - 128)` uniformly for every `e ∈ 0..=255`. The `e < 2`
/// branch exists only because IEEE 754 binary32 cannot encode `2^-128`
/// or `2^-127` as normal numbers — the function emits the corresponding
/// subnormal bit patterns directly (`0x0020_0000 = 2^-128` for `e=0`,
/// `0x0040_0000 = 2^-127` for `e=1`). For `e >= 2` the value is normal
/// and reachable as a plain `(e - 1) << 23` mantissa-region write.
///
/// **No NaN guard for `e == 0xFF`** — `llama.cpp`'s `_half` helper
/// returns `2^127` there rather than NaN, deviating from the raw OCP MX
/// spec. We match `llama.cpp` bit-for-bit so cross-validation against
/// `dequantize_row_mxfp4` is exact.
///
/// The "half" suffix denotes that the scale is half the OCP `2^(e-127)`
/// value; the lost factor of 2 is folded into [`K_VALUES_MXFP4`], whose
/// entries are themselves 2× the OCP E2M1 codebook magnitudes.
#[inline]
fn e8m0_to_fp32_half(e: u8) -> f32 {
    // BITWISE: assemble the f32 directly from the E8M0 byte (no float
    // arithmetic). The two branches are not different formulas — both
    // realise `2^(e - 128)` — they differ only in whether the result is
    // representable as a normal f32 (`e >= 2`) or has to be emitted as
    // a subnormal bit pattern (`e ∈ {0, 1}`).
    let bits: u32 = if e < 2 {
        // CAST: u8 → u32 lossless for the shift amount. `0x0020_0000 << 0`
        // is the f32 bit pattern for 2^-128 (subnormal, mantissa = 2^21);
        // shifting left by 1 yields `0x0040_0000` = 2^-127.
        0x0020_0000_u32 << u32::from(e)
    } else {
        // CAST: u8 → u32 lossless. `(e - 1) << 23` writes the IEEE 754
        // biased exponent so the resulting f32 equals `2^(e - 128)`
        // (e.g., e=2 → biased exp = 1 → 2^(1 - 127) = 2^-126).
        (u32::from(e) - 1) << 23
    };
    f32::from_bits(bits)
}

/// Unpacks an `IQ1S_GRID` `u64` entry into 8 signed `i8` codebook values.
///
/// The C reference reinterprets `(int8_t *)&iq1s_grid[idx]`; in Rust we
/// pull the 8 little-endian bytes and cast each to `i8`. Used by both
/// `IQ1_S` and `IQ1_M` (every group's grid lookup goes through this).
#[inline]
#[allow(clippy::as_conversions, clippy::cast_possible_wrap)]
fn iq1s_grid_as_i8(packed: u64) -> [i8; 8] {
    // CAST: u8 → i8 intentional signed reinterpret of the IQ1S_GRID codebook.
    packed.to_le_bytes().map(|b| b as i8)
}

/// Reads a 12-byte packed scales field out of a block at `offset`.
///
/// Used by `Q3_K`, `Q4_K`, and `Q5_K` to lift the packed 6-bit scales
/// field into a fixed-size array for the extractor helpers. Written as
/// an explicit element-by-element construction because `.try_into()` on
/// a `&[u8]` slice is fallible in the type system and `.unwrap()` /
/// `.expect()` are banned by the project's lint posture — and because
/// the compiler optimises this form into a single 12-byte load.
#[inline]
#[allow(clippy::indexing_slicing)]
fn read_scales12(block: &[u8], offset: usize) -> [u8; 12] {
    // INDEX: callers pre-validate `block.len() >= offset + 12` via the
    // `run_super_kernel` `chunks_exact(type_size)` outer loop, which
    // guarantees a full K-quant super-block (≥ 110 bytes).
    [
        block[offset],
        block[offset + 1],
        block[offset + 2],
        block[offset + 3],
        block[offset + 4],
        block[offset + 5],
        block[offset + 6],
        block[offset + 7],
        block[offset + 8],
        block[offset + 9],
        block[offset + 10],
        block[offset + 11],
    ]
}

/// Converts a block's worth of `f32` scratch values to `BF16` bytes.
///
/// This is the hot-path pass 2 for every kernel. Branch-free, contiguous
/// reads, contiguous writes — exactly the shape the compiler needs to
/// auto-vectorise the inner `f32 → BF16` conversion. A future commit may
/// add an AVX2-gated intrinsic variant behind a `simd` feature flag; the
/// scalar version here is the canonical fallback.
///
/// # Preconditions
///
/// `out_block.len() == 2 × scratch.len()`. `chunks_exact_mut(2)` silently
/// truncates any remainder, so a caller that passes the wrong length
/// produces short output rather than a panic — every kernel sizes both
/// buffers at compile time so this never happens in practice.
#[inline]
fn write_scratch_to_bf16(scratch: &[f32], out_block: &mut [u8]) {
    // VECTORIZED: pending cargo-show-asm verification on the scalar path.
    // TODO(phase4-followup): AVX2 `f32x8 → bf16x8` gated behind a `simd`
    // feature per CONVENTIONS.md's accepted-unsafe table — would give a
    // ~4-8× speedup on pass 2, which is ~40-60% of total dequant time.
    for (&val, out_pair) in scratch.iter().zip(out_block.chunks_exact_mut(2)) {
        let bf16 = f32_bits_to_bf16_bits(val.to_bits());
        out_pair.copy_from_slice(&bf16.to_le_bytes());
    }
}

// ---------------------------------------------------------------------------
// Shared validation
// ---------------------------------------------------------------------------

/// Validates dispatcher inputs and returns the output byte length
/// (`n_elements × 2`), with an overflow guard that catches 32-bit targets
/// where the `BF16` output would exceed `usize::MAX`.
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if
/// - `n_elements` is not a multiple of `dtype.block_size()`,
/// - `n_elements × 2` overflows `usize` (32-bit targets with > 2 GiB
///   output), or
/// - `data.len()` does not equal `n_blocks × dtype.type_size()`.
///
/// Returns [`AnamnesisError::Unsupported`] only if a future `GgufType`
/// variant is added without a `type_size()` arm — every variant
/// recognised today returns `Some(_)`, so this branch is a forward-
/// compatibility safety net rather than a current code path.
fn validate_dequant_input(data: &[u8], dtype: GgufType, n_elements: usize) -> crate::Result<usize> {
    let block_size = dtype.block_size();
    if !n_elements.is_multiple_of(block_size) {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "GGUF dequant: n_elements {n_elements} is not a multiple of block size \
                 {block_size} for {dtype}"
            ),
        });
    }
    // Output byte length: `n_elements × 2`. Guarded against overflow so a
    // 32-bit target with > 2 GiB of BF16 output produces a clean `Parse`
    // error rather than silently truncating the allocation.
    let out_byte_len = n_elements
        .checked_mul(2)
        .ok_or_else(|| AnamnesisError::Parse {
            reason: format!("GGUF dequant: output byte length {n_elements}×2 overflows usize"),
        })?;
    let n_blocks = n_elements / block_size;
    let type_size = dtype
        .type_size()
        .ok_or_else(|| AnamnesisError::Unsupported {
            format: "GGUF".into(),
            detail: format!("dequantisation not yet supported for {dtype}"),
        })?;
    let expected_bytes = n_blocks
        .checked_mul(type_size)
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "GGUF dequant: expected byte count overflows usize".into(),
        })?;
    if data.len() != expected_bytes {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "GGUF dequant: data length {} != expected {expected_bytes} for \
                 {n_blocks} blocks of {dtype}",
                data.len()
            ),
        });
    }
    Ok(out_byte_len)
}

// ---------------------------------------------------------------------------
// Kernel runners (generic over pass-1 unpack closure + sink closure)
// ---------------------------------------------------------------------------

/// Outer-loop runner for 32-element legacy quant blocks.
///
/// Walks `data` in `type_size` chunks via `chunks_exact`, invokes the
/// caller-supplied `unpack` closure to fill the stack `[f32; 32]`
/// scratch, then writes `BF16` bytes into a stack `[u8; 64]` buffer and
/// forwards it to `sink`.
///
/// Generic + `#[inline]` so the per-type kernel's closure body fully
/// inlines into the outer loop with no indirect-call overhead.
#[inline]
fn run_legacy_kernel<F, P>(
    data: &[u8],
    type_size: usize,
    mut sink: F,
    mut unpack: P,
) -> crate::Result<()>
where
    F: FnMut(&[u8]) -> crate::Result<()>,
    P: FnMut(&[u8], &mut [f32; QK_SMALL]),
{
    let mut scratch = [0.0_f32; QK_SMALL];
    let mut block_out = [0u8; QK_SMALL * 2];
    for in_block in data.chunks_exact(type_size) {
        unpack(in_block, &mut scratch);
        write_scratch_to_bf16(&scratch, &mut block_out);
        sink(&block_out)?;
    }
    Ok(())
}

/// Outer-loop runner for 256-element K-quant super-blocks.
///
/// Structurally identical to [`run_legacy_kernel`] but with a 1 KB
/// scratch buffer and a 512 B output buffer. Both still fit comfortably
/// in L1 cache.
#[inline]
fn run_super_kernel<F, P>(
    data: &[u8],
    type_size: usize,
    mut sink: F,
    mut unpack: P,
) -> crate::Result<()>
where
    F: FnMut(&[u8]) -> crate::Result<()>,
    P: FnMut(&[u8], &mut [f32; QK_K]),
{
    let mut scratch = [0.0_f32; QK_K];
    let mut block_out = [0u8; QK_K * 2];
    for in_block in data.chunks_exact(type_size) {
        unpack(in_block, &mut scratch);
        write_scratch_to_bf16(&scratch, &mut block_out);
        sink(&block_out)?;
    }
    Ok(())
}

/// Routes a validated `(data, dtype)` pair to the correct per-type
/// streaming kernel. Called by both public entry points.
fn dispatch_streaming<F>(data: &[u8], dtype: GgufType, sink: F) -> crate::Result<()>
where
    F: FnMut(&[u8]) -> crate::Result<()>,
{
    // EXHAUSTIVE: internal dispatch over GgufType. Scalar (non-block)
    // types fall through to the wildcard arm — they are reinterpret-
    // only and never reach the dequantisation layer. After Phase 4.5
    // step 6 every block-quantised variant has a dedicated kernel.
    #[allow(clippy::wildcard_enum_match_arm)]
    match dtype {
        GgufType::Q4_0 => dequant_q4_0(data, sink),
        GgufType::Q4_1 => dequant_q4_1(data, sink),
        GgufType::Q5_0 => dequant_q5_0(data, sink),
        GgufType::Q5_1 => dequant_q5_1(data, sink),
        GgufType::Q8_0 => dequant_q8_0(data, sink),
        GgufType::Q8_1 => dequant_q8_1(data, sink),
        GgufType::Q2_K => dequant_q2_k(data, sink),
        GgufType::Q3_K => dequant_q3_k(data, sink),
        GgufType::Q4_K => dequant_q4_k(data, sink),
        GgufType::Q5_K => dequant_q5_k(data, sink),
        GgufType::Q6_K => dequant_q6_k(data, sink),
        GgufType::Q8_K => dequant_q8_k(data, sink),
        GgufType::IQ4_NL => dequant_iq4_nl(data, sink),
        GgufType::IQ4_XS => dequant_iq4_xs(data, sink),
        GgufType::IQ2_XXS => dequant_iq2_xxs(data, sink),
        GgufType::IQ2_XS => dequant_iq2_xs(data, sink),
        GgufType::IQ2_S => dequant_iq2_s(data, sink),
        GgufType::IQ3_XXS => dequant_iq3_xxs(data, sink),
        GgufType::IQ3_S => dequant_iq3_s(data, sink),
        GgufType::IQ1_S => dequant_iq1_s(data, sink),
        GgufType::IQ1_M => dequant_iq1_m(data, sink),
        GgufType::TQ1_0 => dequant_tq1_0(data, sink),
        GgufType::TQ2_0 => dequant_tq2_0(data, sink),
        GgufType::MXFP4 => dequant_mxfp4(data, sink),
        _ => Err(AnamnesisError::Unsupported {
            format: "GGUF".into(),
            detail: format!("dequantisation not yet supported for {dtype}"),
        }),
    }
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Dequantises a `GGUF`-encoded block-quantised tensor into an owned
/// `Vec<u8>` of `BF16` bytes.
///
/// Convenience wrapper around [`dequantize_gguf_blocks_to_bf16`] that
/// pushes each block's output into a pre-allocated `Vec::with_capacity`.
/// For very large tensors, prefer the streaming variant — this variant's
/// peak heap is O(`n_elements × 2`).
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if `n_elements` is not a multiple of
/// the block size for `dtype`, if `n_elements × 2` overflows `usize`
/// (32-bit targets only), or if `data.len()` does not equal the expected
/// byte count (`n_blocks × type_size`).
///
/// Returns [`AnamnesisError::Unsupported`] if `dtype` is a scalar
/// (non-block-quant) type that is structurally not a quantised block
/// format. Every block-quantised variant has a dedicated kernel after
/// Phase 4.5 step 6.
///
/// # Memory
///
/// Allocates a single `Vec<u8>` of length `n_elements × 2` for the `BF16`
/// output. Uses `Vec::with_capacity` + `extend_from_slice` to avoid the
/// `vec![0u8; n]` zero-init memset that would otherwise touch every byte
/// of the output before the dequant loop overwrites them. Peak heap is
/// the output buffer itself — O(n) per call.
///
/// Callers that dequantise many tensors should drop each result before
/// allocating the next, or use [`dequantize_gguf_blocks_to_bf16`] to
/// stream block bytes directly into a writer. The crate's high-level
/// orchestrators (`ParsedModel::remember`, the `amn remember model.gguf`
/// CLI path) currently retain every tensor's `Vec<u8>` in heap memory
/// until `safetensors::serialize_to_file` returns — see the streaming
/// output milestone (ROADMAP Phase 10) for the planned fix.
pub fn dequantize_gguf_to_bf16(
    data: &[u8],
    dtype: GgufType,
    n_elements: usize,
) -> crate::Result<Vec<u8>> {
    if n_elements == 0 {
        return Ok(Vec::new());
    }
    let out_byte_len = validate_dequant_input(data, dtype, n_elements)?;
    let mut out: Vec<u8> = Vec::with_capacity(out_byte_len);
    dispatch_streaming(data, dtype, |block_out| {
        out.extend_from_slice(block_out);
        Ok(())
    })?;
    Ok(out)
}

/// Streaming `GGUF` dequantisation: invokes `sink` once per block with
/// that block's `BF16` bytes. Peak heap is O(one block) regardless of
/// tensor size.
///
/// The `sink` closure receives `64` bytes per call for 32-element block
/// kernels (`Q4_0`–`Q8_1`, `IQ4_NL`, `MXFP4`) and `512` bytes per call
/// for 256-element super-block kernels (`Q2_K`–`Q8_K`, `IQ4_XS`,
/// `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`, `IQ1_S`, `IQ1_M`,
/// `TQ1_0`, `TQ2_0`). Sink errors abort the stream and are propagated
/// unchanged as the return value.
///
/// This is the canonical form. [`dequantize_gguf_to_bf16`] is a thin
/// wrapper that sinks into a `Vec::with_capacity`.
///
/// # Errors
///
/// Returns the same validation errors as [`dequantize_gguf_to_bf16`]
/// plus any [`AnamnesisError`] that `sink` itself returns.
///
/// # Memory
///
/// Stack only: one `[f32; QK]` scratch buffer (128 B for 32-element
/// block kernels, 1 KB for 256-element super-block kernels) and one
/// `[u8; QK × 2]` block output buffer (64 B / 512 B). No heap allocation
/// in this function's frame.
///
/// # Example
///
/// ```rust,no_run
/// # #[cfg(feature = "gguf")]
/// # fn _doc() -> anamnesis::Result<()> {
/// use anamnesis::{dequantize_gguf_blocks_to_bf16, GgufType};
///
/// let data: &[u8] = &[];
/// let n_elements = 0;
/// let mut total_bytes = 0usize;
/// dequantize_gguf_blocks_to_bf16(data, GgufType::Q4_0, n_elements, |block| {
///     total_bytes += block.len();
///     Ok(())
/// })?;
/// # Ok(())
/// # }
/// ```
pub fn dequantize_gguf_blocks_to_bf16<F>(
    data: &[u8],
    dtype: GgufType,
    n_elements: usize,
    sink: F,
) -> crate::Result<()>
where
    F: FnMut(&[u8]) -> crate::Result<()>,
{
    if n_elements == 0 {
        return Ok(());
    }
    // Discarding the returned `out_byte_len`: the streaming variant does
    // not allocate an output buffer, so it only needs the validation side
    // effects.
    let _ = validate_dequant_input(data, dtype, n_elements)?;
    dispatch_streaming(data, dtype, sink)
}

// ---------------------------------------------------------------------------
// Legacy block quants — pass-1 closures
// ---------------------------------------------------------------------------

/// `Q4_0` kernel — 18-byte blocks: `d: f16` + `qs[16]` (4-bit packed).
///
/// Formula: `y[j] = d * (qs[j] nibble - 8)`. Low nibbles of `qs[0..16]`
/// fill output positions `0..16`, high nibbles fill `16..32`.
#[allow(
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_precision_loss
)]
fn dequant_q4_0<F>(data: &[u8], sink: F) -> crate::Result<()>
where
    F: FnMut(&[u8]) -> crate::Result<()>,
{
    run_legacy_kernel(data, 18, sink, |in_block, scratch| {
        let d = read_f16_bytes([in_block[0], in_block[1]]);
        // BITWISE: split each qs byte into two 4-bit nibbles, bias by -8
        // CAST: i32 → f32, lossless for values in [-8, 7]
        for j in 0..16 {
            let lo = i32::from(in_block[2 + j] & 0x0F) - 8;
            let hi = i32::from(in_block[2 + j] >> 4) - 8;
            scratch[j] = lo as f32 * d;
            scratch[j + 16] = hi as f32 * d;
        }
    })
}

/// `Q4_1` kernel — 20-byte blocks: `d: f16` + `m: f16` + `qs[16]` (4-bit).
///
/// Formula: `y[j] = d * qs[j] nibble + m`. No `-8` bias.
#[allow(
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_precision_loss
)]
fn dequant_q4_1<F>(data: &[u8], sink: F) -> crate::Result<()>
where
    F: FnMut(&[u8]) -> crate::Result<()>,
{
    run_legacy_kernel(data, 20, sink, |in_block, scratch| {
        let d = read_f16_bytes([in_block[0], in_block[1]]);
        let m = read_f16_bytes([in_block[2], in_block[3]]);
        // BITWISE: split each qs byte into two unsigned 4-bit nibbles
        // CAST: i32 → f32, lossless for [0, 15]
        for j in 0..16 {
            let lo = i32::from(in_block[4 + j] & 0x0F);
            let hi = i32::from(in_block[4 + j] >> 4);
            scratch[j] = lo as f32 * d + m;
            scratch[j + 16] = hi as f32 * d + m;
        }
    })
}

/// `Q5_0` kernel — 22-byte blocks: `d: f16` + `qh[4]` (u32) + `qs[16]`.
///
/// Formula: `y[j] = d * ((5-bit value) - 16)`.
#[allow(
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss
)]
fn dequant_q5_0<F>(data: &[u8], sink: F) -> crate::Result<()>
where
    F: FnMut(&[u8]) -> crate::Result<()>,
{
    run_legacy_kernel(data, 22, sink, |in_block, scratch| {
        let d = read_f16_bytes([in_block[0], in_block[1]]);
        let qh = u32::from_le_bytes([in_block[2], in_block[3], in_block[4], in_block[5]]);
        // BITWISE: merge low 4 bits from qs with bit 4 from qh, bias by -16
        // CAST: usize → u32 for the shift amount; u32 → i32 for the merged
        // value; i32 → f32 lossless for the final [-16, 15] range
        for j in 0..16 {
            let j_u32 = j as u32;
            let xh_0 = ((qh >> j_u32) << 4) & 0x10;
            let xh_1 = (qh >> (j_u32 + 12)) & 0x10;
            let x0 = (i32::from(in_block[6 + j] & 0x0F) | xh_0 as i32) - 16;
            let x1 = (i32::from(in_block[6 + j] >> 4) | xh_1 as i32) - 16;
            scratch[j] = x0 as f32 * d;
            scratch[j + 16] = x1 as f32 * d;
        }
    })
}

/// `Q5_1` kernel — 24-byte blocks: `d: f16` + `m: f16` + `qh[4]` + `qs[16]`.
///
/// Formula: `y[j] = d * (5-bit value) + m`.
#[allow(
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss
)]
fn dequant_q5_1<F>(data: &[u8], sink: F) -> crate::Result<()>
where
    F: FnMut(&[u8]) -> crate::Result<()>,
{
    run_legacy_kernel(data, 24, sink, |in_block, scratch| {
        let d = read_f16_bytes([in_block[0], in_block[1]]);
        let m = read_f16_bytes([in_block[2], in_block[3]]);
        let qh = u32::from_le_bytes([in_block[4], in_block[5], in_block[6], in_block[7]]);
        // BITWISE: merge low 4 bits from qs with bit 4 from qh
        // CAST: usize → u32 for the shift amount; u32 → i32 for the merged
        // value; i32 → f32 lossless for the final [0, 31] range
        for j in 0..16 {
            let j_u32 = j as u32;
            let xh_0 = ((qh >> j_u32) << 4) & 0x10;
            let xh_1 = (qh >> (j_u32 + 12)) & 0x10;
            let x0 = i32::from(in_block[8 + j] & 0x0F) | xh_0 as i32;
            let x1 = i32::from(in_block[8 + j] >> 4) | xh_1 as i32;
            scratch[j] = x0 as f32 * d + m;
            scratch[j + 16] = x1 as f32 * d + m;
        }
    })
}

/// `Q8_0` kernel — 34-byte blocks: `d: f16` + `qs[32]` (`i8`).
///
/// Formula: `y[j] = d * qs[j]`.
#[allow(
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_possible_wrap
)]
fn dequant_q8_0<F>(data: &[u8], sink: F) -> crate::Result<()>
where
    F: FnMut(&[u8]) -> crate::Result<()>,
{
    run_legacy_kernel(data, 34, sink, |in_block, scratch| {
        let d = read_f16_bytes([in_block[0], in_block[1]]);
        // CAST: u8 → i8 intentional signed reinterpret, then i8 → f32 lossless
        for j in 0..QK_SMALL {
            let signed = in_block[2 + j] as i8;
            scratch[j] = f32::from(signed) * d;
        }
    })
}

/// `Q8_1` kernel — 36-byte blocks: `d: f16` + `s: f16` (aux) + `qs[32]`.
///
/// The `s` field stores `d × Σ qs[i]` as a matmul accelerator and is
/// **not** used for reconstruction. Formula: `y[j] = d * qs[j]`.
#[allow(
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_possible_wrap
)]
fn dequant_q8_1<F>(data: &[u8], sink: F) -> crate::Result<()>
where
    F: FnMut(&[u8]) -> crate::Result<()>,
{
    run_legacy_kernel(data, 36, sink, |in_block, scratch| {
        let d = read_f16_bytes([in_block[0], in_block[1]]);
        // in_block[2..4] is `s` (aux, ignored)
        for j in 0..QK_SMALL {
            let signed = in_block[4 + j] as i8;
            scratch[j] = f32::from(signed) * d;
        }
    })
}

/// `IQ4_NL` kernel — 18-byte blocks: `d: f16` + `qs[16]` (4-bit packed).
///
/// Non-linear 4-bit quant: each 4-bit nibble indexes the shared
/// [`K_VALUES_IQ4_NL`] codebook to recover an `i8` quant, which is then
/// multiplied by the per-block `f32` scale. Formula (from
/// `ggml-quants.c::dequantize_row_iq4_nl`):
///
/// ```text
/// y[j]       = d * K_VALUES_IQ4_NL[qs[j] & 0xF]   for j ∈ 0..16
/// y[j + 16]  = d * K_VALUES_IQ4_NL[qs[j] >> 4]    for j ∈ 0..16
/// ```
#[allow(clippy::indexing_slicing)]
fn dequant_iq4_nl<F>(data: &[u8], sink: F) -> crate::Result<()>
where
    F: FnMut(&[u8]) -> crate::Result<()>,
{
    run_legacy_kernel(data, 18, sink, |in_block, scratch| {
        let d = read_f16_bytes([in_block[0], in_block[1]]);
        // BITWISE: split each qs byte into two 4-bit codebook indices
        // INDEX: nibble masked to 0..16 — K_VALUES_IQ4_NL lookup is in-bounds
        for j in 0..16 {
            let lo = K_VALUES_IQ4_NL[usize::from(in_block[2 + j] & 0x0F)];
            let hi = K_VALUES_IQ4_NL[usize::from(in_block[2 + j] >> 4)];
            scratch[j] = f32::from(lo) * d;
            scratch[j + 16] = f32::from(hi) * d;
        }
    })
}

/// `MXFP4` kernel — 17-byte blocks: `e: u8` (E8M0 exponent) + `qs[16]`
/// (4-bit packed nibbles).
///
/// 32-element microscaling FP4 quant (OCP MX standard from late 2023,
/// added to `ggml` in 2024). Formula: `y[j] = K_VALUES_MXFP4[nibble] × d`
/// where `d = e8m0_to_fp32_half(e)`. Low nibbles of `qs[0..16]` fill
/// output positions `0..16`; high nibbles fill `16..32` — the same
/// **split-half** ordering used by `Q4_0` and `IQ4_NL`.
///
/// The codebook ([`K_VALUES_MXFP4`]) stores 2× the OCP E2M1 magnitudes;
/// the doubling is cancelled by the `_HALF` factor inside
/// [`e8m0_to_fp32_half`], so the dequantised value matches the raw OCP
/// E2M1 spec.
///
/// Ported verbatim from `ggml-quants.c::dequantize_row_mxfp4`.
#[allow(clippy::indexing_slicing)]
fn dequant_mxfp4<F>(data: &[u8], sink: F) -> crate::Result<()>
where
    F: FnMut(&[u8]) -> crate::Result<()>,
{
    run_legacy_kernel(data, 17, sink, |in_block, scratch| {
        let d = e8m0_to_fp32_half(in_block[0]);
        // BITWISE: split each qs byte into two 4-bit codebook indices.
        // INDEX: nibble masked to 0..16 — K_VALUES_MXFP4 lookup is in-bounds.
        for j in 0..16 {
            let lo = K_VALUES_MXFP4[usize::from(in_block[1 + j] & 0x0F)];
            let hi = K_VALUES_MXFP4[usize::from(in_block[1 + j] >> 4)];
            scratch[j] = f32::from(lo) * d;
            scratch[j + 16] = f32::from(hi) * d;
        }
    })
}

// ---------------------------------------------------------------------------
// K-quants — pass-1 closures
// ---------------------------------------------------------------------------

/// `Q8_K` kernel — 292-byte blocks: `d: f32` (**not f16!**) + `qs[256]: i8`
/// + `bsums[16]: i16` (aux, ignored for reconstruction).
///
/// Formula: `y[j] = d * qs[j]`. The simplest K-quant.
#[allow(
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_possible_wrap
)]
fn dequant_q8_k<F>(data: &[u8], sink: F) -> crate::Result<()>
where
    F: FnMut(&[u8]) -> crate::Result<()>,
{
    run_super_kernel(data, 292, sink, |in_block, scratch| {
        let d = read_f32_bytes([in_block[0], in_block[1], in_block[2], in_block[3]]);
        // CAST: u8 → i8 intentional signed reinterpret, i8 → f32 lossless
        for j in 0..QK_K {
            let signed = in_block[4 + j] as i8;
            scratch[j] = f32::from(signed) * d;
        }
    })
}

/// `Q2_K` kernel — 84-byte blocks: `scales[16]` (packed 4/4-bit) +
/// `qs[64]` (2-bit) + `d: f16` + `dmin: f16`.
///
/// Each `scales[is]` byte packs a 4-bit scale (low nibble) and a 4-bit
/// min (high nibble). The 64-byte `qs` holds 256 elements as 2-bit values
/// packed 4-per-byte. Iteration walks 8 sub-blocks of 32 elements each,
/// driven by a `shift` variable taking values 0, 2, 4, 6.
///
/// Formula: `y = d * (sc & 0xF) * q2 - dmin * (sc >> 4)` where
/// `q2 = (qs[l] >> shift) & 3`.
#[allow(
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_precision_loss
)]
fn dequant_q2_k<F>(data: &[u8], sink: F) -> crate::Result<()>
where
    F: FnMut(&[u8]) -> crate::Result<()>,
{
    run_super_kernel(data, 84, sink, |in_block, scratch| {
        // Field offsets: scales [0..16], qs [16..80], d [80..82], dmin [82..84]
        let scales = &in_block[0..16];
        let qs = &in_block[16..80];
        let d = read_f16_bytes([in_block[80], in_block[81]]);
        let dmin = read_f16_bytes([in_block[82], in_block[83]]);

        // 2 halves × 4 shifts × 2 sub-groups × 16 elements = 256
        let mut is: usize = 0;
        let mut y_off: usize = 0;
        let mut q_off: usize = 0;
        for _n in 0..2 {
            let mut shift: u32 = 0;
            for _j in 0..4 {
                // BITWISE: unpack (scale_nibble, min_nibble) from one scales byte
                let sc_a = scales[is];
                is += 1;
                let dl_a = d * f32::from(sc_a & 0x0F);
                let ml_a = dmin * f32::from(sc_a >> 4);
                for l in 0..16 {
                    // BITWISE: 2-bit value at `shift` from qs[q_off + l]
                    // CAST: i32 → f32, lossless for [0, 3]
                    let q2 = i32::from((qs[q_off + l] >> shift) & 0x03);
                    scratch[y_off + l] = dl_a * (q2 as f32) - ml_a;
                }
                let sc_b = scales[is];
                is += 1;
                let dl_b = d * f32::from(sc_b & 0x0F);
                let ml_b = dmin * f32::from(sc_b >> 4);
                for l in 0..16 {
                    let q2 = i32::from((qs[q_off + l + 16] >> shift) & 0x03);
                    scratch[y_off + l + 16] = dl_b * (q2 as f32) - ml_b;
                }
                y_off += 32;
                shift += 2;
            }
            q_off += 32;
        }
    })
}

/// `Q3_K` kernel — 110-byte blocks: `hmask[32]` + `qs[64]` (2-bit low) +
/// `scales[12]` (6-bit packed) + `d: f16`.
///
/// Scales are 6-bit signed values packed into 12 bytes via a `kmask1`/
/// `kmask2` bit-permute. The 3-bit values are reconstructed by combining
/// 2 low bits from `qs` with 1 high bit from `hmask`, then subtracting 4
/// when `hmask` is zero. Formula: `y = d * (scale - 32) * (q2 - hi)`.
#[allow(
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_precision_loss
)]
fn dequant_q3_k<F>(data: &[u8], sink: F) -> crate::Result<()>
where
    F: FnMut(&[u8]) -> crate::Result<()>,
{
    run_super_kernel(data, 110, sink, |in_block, scratch| {
        // Field offsets: hmask [0..32], qs [32..96], scales [96..108], d [108..110]
        let hmask = &in_block[0..32];
        let qs = &in_block[32..96];
        let packed_scales = read_scales12(in_block, 96);
        let d_all = read_f16_bytes([in_block[108], in_block[109]]);
        let scales = q3_k_unpack_scales(&packed_scales);

        let mut is: usize = 0;
        let mut y_off: usize = 0;
        let mut q_off: usize = 0;
        let mut m: u8 = 1;
        for _n in 0..2 {
            let mut shift: u32 = 0;
            for _j in 0..4 {
                // CAST: i8 → f32 (lossless), bias by -32 per ggml reference
                let sc_a = scales[is];
                is += 1;
                let dl_a = d_all * (f32::from(sc_a) - 32.0);
                for l in 0..16 {
                    // BITWISE: 2-bit low from qs + 3rd bit from hmask (offset -4 if absent)
                    let q2 = i32::from((qs[q_off + l] >> shift) & 0x03);
                    let hi = i32::from(hmask[l] & m == 0) * 4;
                    scratch[y_off + l] = dl_a * ((q2 - hi) as f32);
                }
                let sc_b = scales[is];
                is += 1;
                let dl_b = d_all * (f32::from(sc_b) - 32.0);
                for l in 0..16 {
                    let q2 = i32::from((qs[q_off + l + 16] >> shift) & 0x03);
                    let hi = i32::from(hmask[l + 16] & m == 0) * 4;
                    scratch[y_off + l + 16] = dl_b * ((q2 - hi) as f32);
                }
                y_off += 32;
                shift += 2;
                m <<= 1;
            }
            q_off += 32;
        }
    })
}

/// `Q4_K` kernel — 144-byte blocks: `d: f16` + `dmin: f16` +
/// `scales[12]` (6-bit packed) + `qs[128]` (4-bit).
///
/// Scales are extracted via [`get_scale_min_k4`]. Formula:
/// `y = d * sc * (qs nibble) - dmin * m`. Processed in 4 groups of 64,
/// each producing 32 low-nibble outputs then 32 high-nibble outputs.
#[allow(
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_precision_loss
)]
fn dequant_q4_k<F>(data: &[u8], sink: F) -> crate::Result<()>
where
    F: FnMut(&[u8]) -> crate::Result<()>,
{
    run_super_kernel(data, 144, sink, |in_block, scratch| {
        // Field offsets: d [0..2], dmin [2..4], scales [4..16], qs [16..144]
        let d = read_f16_bytes([in_block[0], in_block[1]]);
        let dmin = read_f16_bytes([in_block[2], in_block[3]]);
        let scales = read_scales12(in_block, 4);
        let qs = &in_block[16..144];

        let mut is: usize = 0;
        let mut y_off: usize = 0;
        let mut q_off: usize = 0;
        for _j in 0..4 {
            let (sc_lo, min_lo) = get_scale_min_k4(is, &scales);
            let (sc_hi, min_hi) = get_scale_min_k4(is + 1, &scales);
            let d_lo = d * f32::from(sc_lo);
            let off_lo = dmin * f32::from(min_lo);
            let d_hi = d * f32::from(sc_hi);
            let off_hi = dmin * f32::from(min_hi);
            for l in 0..32 {
                // BITWISE: extract low/high 4-bit nibbles
                // CAST: i32 → f32, lossless for [0, 15]
                let lo = i32::from(qs[q_off + l] & 0x0F);
                let hi = i32::from(qs[q_off + l] >> 4);
                scratch[y_off + l] = d_lo * (lo as f32) - off_lo;
                scratch[y_off + l + 32] = d_hi * (hi as f32) - off_hi;
            }
            q_off += 32;
            y_off += 64;
            is += 2;
        }
    })
}

/// `Q5_K` kernel — 176-byte blocks: `d: f16` + `dmin: f16` +
/// `scales[12]` + `qh[32]` (5th-bit store) + `qs[128]` (4-bit low).
///
/// Like [`dequant_q4_k`], but each 4-bit nibble gets an additional high bit
/// from `qh[l]` selected by a rotating `u1`/`u2` mask. Formula:
/// `y = d * sc * ((ql & 0xF) + (qh & mask ? 16 : 0)) - dmin * m`.
#[allow(
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_precision_loss
)]
fn dequant_q5_k<F>(data: &[u8], sink: F) -> crate::Result<()>
where
    F: FnMut(&[u8]) -> crate::Result<()>,
{
    run_super_kernel(data, 176, sink, |in_block, scratch| {
        // Field offsets: d [0..2], dmin [2..4], scales [4..16], qh [16..48], ql [48..176]
        let d = read_f16_bytes([in_block[0], in_block[1]]);
        let dmin = read_f16_bytes([in_block[2], in_block[3]]);
        let scales = read_scales12(in_block, 4);
        let qh = &in_block[16..48];
        let ql = &in_block[48..176];

        let mut is: usize = 0;
        let mut y_off: usize = 0;
        let mut q_off: usize = 0;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        for _j in 0..4 {
            let (sc_lo, min_lo) = get_scale_min_k4(is, &scales);
            let (sc_hi, min_hi) = get_scale_min_k4(is + 1, &scales);
            let d_lo = d * f32::from(sc_lo);
            let off_lo = dmin * f32::from(min_lo);
            let d_hi = d * f32::from(sc_hi);
            let off_hi = dmin * f32::from(min_hi);
            for l in 0..32 {
                // BITWISE: merge 4-bit low from ql with high bit from qh (rotating mask)
                let lo = i32::from(ql[q_off + l] & 0x0F) + i32::from(qh[l] & u1 != 0) * 16;
                let hi = i32::from(ql[q_off + l] >> 4) + i32::from(qh[l] & u2 != 0) * 16;
                // CAST: i32 → f32, lossless for [0, 31]
                scratch[y_off + l] = d_lo * (lo as f32) - off_lo;
                scratch[y_off + l + 32] = d_hi * (hi as f32) - off_hi;
            }
            q_off += 32;
            y_off += 64;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    })
}

/// `Q6_K` kernel — 210-byte blocks: `ql[128]` (4-bit low) + `qh[64]`
/// (2-bit high) + `scales[16]: i8` + `d: f16`.
///
/// Each 6-bit element is reconstructed as `(ql & 0xF) | ((qh >> shift) &
/// 3) << 4`, biased by `-32`. Processed in 2 halves of 128 elements.
#[allow(
    clippy::indexing_slicing,
    clippy::similar_names,
    clippy::as_conversions,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss
)]
fn dequant_q6_k<F>(data: &[u8], sink: F) -> crate::Result<()>
where
    F: FnMut(&[u8]) -> crate::Result<()>,
{
    run_super_kernel(data, 210, sink, |in_block, scratch| {
        // Field offsets: ql [0..128], qh [128..192], scales [192..208], d [208..210]
        let ql_all = &in_block[0..128];
        let qh_all = &in_block[128..192];
        let sc_all = &in_block[192..208];
        let d = read_f16_bytes([in_block[208], in_block[209]]);

        let mut y_off: usize = 0;
        let mut ql_off: usize = 0;
        let mut qh_off: usize = 0;
        let mut sc_off: usize = 0;
        for _n in 0..2 {
            for l in 0..32 {
                let is = l / 16;
                // BITWISE: 4-bit low + 2-bit high (shift 0,2,4,6 for four values)
                let q1 =
                    i32::from((ql_all[ql_off + l] & 0x0F) | ((qh_all[qh_off + l] & 0x03) << 4))
                        - 32;
                let q2 = i32::from(
                    (ql_all[ql_off + l + 32] & 0x0F) | (((qh_all[qh_off + l] >> 2) & 0x03) << 4),
                ) - 32;
                let q3 = i32::from(
                    (ql_all[ql_off + l] >> 4) | (((qh_all[qh_off + l] >> 4) & 0x03) << 4),
                ) - 32;
                let q4 = i32::from(
                    (ql_all[ql_off + l + 32] >> 4) | (((qh_all[qh_off + l] >> 6) & 0x03) << 4),
                ) - 32;

                // CAST: u8 → i8 intentional signed reinterpret for scales
                let s0 = sc_all[sc_off + is] as i8;
                let s2 = sc_all[sc_off + is + 2] as i8;
                let s4 = sc_all[sc_off + is + 4] as i8;
                let s6 = sc_all[sc_off + is + 6] as i8;
                // CAST: i32 → f32, lossless for the narrow ranges in use
                scratch[y_off + l] = d * f32::from(s0) * (q1 as f32);
                scratch[y_off + l + 32] = d * f32::from(s2) * (q2 as f32);
                scratch[y_off + l + 64] = d * f32::from(s4) * (q3 as f32);
                scratch[y_off + l + 96] = d * f32::from(s6) * (q4 as f32);
            }
            y_off += 128;
            ql_off += 64;
            qh_off += 32;
            sc_off += 8;
        }
    })
}

/// `IQ4_XS` kernel — 136-byte super-blocks: `d: f16` + `scales_h: u16` +
/// `scales_l[4]` + `qs[128]` (4-bit packed).
///
/// Non-linear 4-bit super-quant: 8 sub-blocks of 32 elements. Each
/// sub-block carries a 6-bit signed-biased scale split across `scales_l`
/// (low 4 bits) and `scales_h` (high 2 bits), biased by `-32`. The 4-bit
/// storage nibbles index the shared [`K_VALUES_IQ4_NL`] codebook exactly
/// as in [`dequant_iq4_nl`]. Formula (from
/// `ggml-quants.c::dequantize_row_iq4_xs`):
///
/// ```text
/// ls = ((scales_l[ib/2] >> (4*(ib%2))) & 0xF)
///    | (((scales_h   >> (2*ib))       & 0x3) << 4)
/// dl = d * (ls - 32)
/// y[32*ib + j     ] = dl * K_VALUES_IQ4_NL[qs[16*ib + j] & 0xF]
/// y[32*ib + j + 16] = dl * K_VALUES_IQ4_NL[qs[16*ib + j] >> 4]
/// ```
#[allow(
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]
fn dequant_iq4_xs<F>(data: &[u8], sink: F) -> crate::Result<()>
where
    F: FnMut(&[u8]) -> crate::Result<()>,
{
    run_super_kernel(data, 136, sink, |in_block, scratch| {
        // Field offsets: d [0..2], scales_h [2..4], scales_l [4..8], qs [8..136]
        let d = read_f16_bytes([in_block[0], in_block[1]]);
        let scales_h = u16::from_le_bytes([in_block[2], in_block[3]]);
        let scales_l = [in_block[4], in_block[5], in_block[6], in_block[7]];
        let qs = &in_block[8..136];

        let mut y_off: usize = 0;
        let mut q_off: usize = 0;
        for ib in 0..8 {
            // BITWISE: combine 4-bit low (scales_l) + 2-bit high (scales_h)
            // into a 6-bit signed-biased sub-block scale, then subtract 32.
            // CAST: usize → u32 for shift amounts; ib ∈ 0..8 so truncation is
            // impossible on any target wider than 3 bits
            let ib_u32 = ib as u32;
            let sc_lo = (scales_l[ib / 2] >> (4 * (ib_u32 % 2))) & 0x0F;
            let sc_hi = (scales_h >> (2 * ib_u32)) & 0x03;
            let ls = i32::from(sc_lo) | (i32::from(sc_hi) << 4);
            // CAST: i32 → f32 lossless for (ls - 32) ∈ [-32, 31]
            let dl = d * ((ls - 32) as f32);

            // INDEX: nibble masked to 0..16 — K_VALUES_IQ4_NL lookup is in-bounds
            for j in 0..16 {
                let lo = K_VALUES_IQ4_NL[usize::from(qs[q_off + j] & 0x0F)];
                let hi = K_VALUES_IQ4_NL[usize::from(qs[q_off + j] >> 4)];
                scratch[y_off + j] = dl * f32::from(lo);
                scratch[y_off + j + 16] = dl * f32::from(hi);
            }
            q_off += 16;
            y_off += 32;
        }
    })
}

// ---------------------------------------------------------------------------
// 2-bit IQ kernels — pass-1 closures
// ---------------------------------------------------------------------------

/// Writes 8 signed codebook values into `scratch` starting at `y_off`.
///
/// Shared hot-loop body for the 2-bit `IQ*` kernels. Each of the 8 output
/// f32 values is `dl × grid[j] × sign`, where `sign ∈ {+1, -1}` is picked
/// by the `j`-th bit of `signs` via [`KMASK_IQ2XS`].
///
/// The branch-free form `1.0 - 2.0 * sign_bit` is used in preference to a
/// ternary so the inner loop keeps a uniform data flow; the Rust compiler
/// currently rewrites the obvious `if` expression to the same code, but
/// making it explicit documents intent and protects against regressions
/// in future optimiser heuristics.
#[inline]
#[allow(clippy::indexing_slicing)]
fn write_signed_grid(scratch: &mut [f32], y_off: usize, dl: f32, grid: [u8; 8], signs: u8) {
    for j in 0..8 {
        // BITWISE: j-th bit of the sign mask selects sign; `!= 0` evaluates
        // as 1 (negate) or 0 (keep) — the branch-free sign fold avoids an
        // `if` that some LLVM versions fail to lower to a blend on AVX2.
        let sign_bit = f32::from(u8::from(signs & KMASK_IQ2XS[j] != 0));
        let sign = 1.0 - 2.0 * sign_bit;
        scratch[y_off + j] = dl * f32::from(grid[j]) * sign;
    }
}

/// Writes 8 codebook values into `scratch` shifted by an additive bias.
///
/// Shared hot-loop body for the 1-bit `IQ*` kernels. Each output is
/// `dl × (grid[j] + delta)` where `grid[j]` is a **signed** `i8` value
/// from `IQ1S_GRID` and `delta = ±IQ1S_DELTA`. Distinct from
/// [`write_signed_grid`] in two ways: the codebook is signed (`i8` not
/// `u8`), and the per-group sign is folded into the scalar `delta` bias
/// rather than a per-element bit mask.
///
/// The branch-free `f32::from(i8)` widening avoids any conditional path
/// in the inner loop, matching the auto-vectorisation contract followed
/// by every other kernel in this module.
#[inline]
#[allow(clippy::indexing_slicing)]
fn write_delta_grid(scratch: &mut [f32], y_off: usize, dl: f32, grid: [i8; 8], delta: f32) {
    for j in 0..8 {
        // CAST: i8 → f32 lossless; codebook range is [i8::MIN, i8::MAX].
        scratch[y_off + j] = dl * (f32::from(grid[j]) + delta);
    }
}

/// Decodes one ternary digit out of a `TQ1_0` packed byte.
///
/// `TQ1_0` stores up to 5 ternary digits per `qs` byte (4 per `qh` byte)
/// using the base-3 packing identity `byte = d0 + 3·d1 + 9·d2 + 27·d3 +
/// 81·d4` where each `d_n ∈ {0, 1, 2}` and the byte is constrained to
/// `<= 242 = 3^5 - 1`. To extract the n-th digit, multiply by `3^n` in
/// wrapping `u8` arithmetic — this places the n-th digit in the byte's
/// top bits — then map the resulting bin via `(q * 3) >> 8`:
/// - `q ∈ [0, 86)`     → digit 0
/// - `q ∈ [86, 171)`   → digit 1
/// - `q ∈ [171, 256)`  → digit 2
///
/// The decoded ternary digit is then biased by `-1` (yielding `{-1, 0,
/// +1}`) and scaled by `d`. Used by [`dequant_tq1_0`] in three call
/// sites (qs main loop, qs tail loop, qh loop).
#[inline]
#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]
fn decode_pow3_ternary(packed: u8, pow3_n: u8, d: f32) -> f32 {
    // BITWISE: extract the topmost ternary digit via the base-3 unpacking
    // trick. After `packed * pow3[n]` (wrapping u8), the n-th digit is in
    // the high bits; `(_ * 3) >> 8` then maps it back to {0, 1, 2}.
    let q = packed.wrapping_mul(pow3_n);
    // CAST: u8 → u16 lossless for the ×3 multiplication; the >> 8 then
    // truncates the u16 result to fit in u8's value range, so the i16
    // intermediate lossless-narrows to i8 below.
    let xi = ((u16::from(q) * 3) >> 8) as i16;
    // CAST: i16 → f32 lossless; (xi - 1) ∈ {-1, 0, +1}.
    f32::from(xi - 1) * d
}

/// `IQ2_XXS` kernel — 66-byte super-blocks: `d: f16` + `qs[32]: u16`.
///
/// ~2.06 bpw quant. Each `ib32 ∈ 0..8` reads 8 bytes from `qs` as two
/// `u32`s: the first holds four 8-bit indices into [`IQ2XXS_GRID`] (one
/// per 8-element group), the second packs four 7-bit sign indices (bits
/// `[0..7]`, `[7..14]`, `[14..21]`, `[21..28]`) and a 4-bit sub-block
/// scale in bits `[28..32]`. The sub-block scale is
/// `db = d × (0.5 + scale_nibble) × 0.25`. Signs are resolved through the
/// 128-entry [`KSIGNS_IQ2XS`] table (7-bit → 8-bit parity-preserving mask).
///
/// Ported verbatim from `ggml-quants.c::dequantize_row_iq2_xxs`.
#[allow(
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]
fn dequant_iq2_xxs<F>(data: &[u8], sink: F) -> crate::Result<()>
where
    F: FnMut(&[u8]) -> crate::Result<()>,
{
    run_super_kernel(data, 66, sink, |in_block, scratch| {
        // Field offsets: d [0..2], qs [2..66]
        let d = read_f16_bytes([in_block[0], in_block[1]]);
        let qs = &in_block[2..66];

        let mut y_off: usize = 0;
        for ib32 in 0..8 {
            let base = 8 * ib32;
            let aux32_0 = u32::from_le_bytes([qs[base], qs[base + 1], qs[base + 2], qs[base + 3]]);
            let aux32_1 =
                u32::from_le_bytes([qs[base + 4], qs[base + 5], qs[base + 6], qs[base + 7]]);
            let aux8 = aux32_0.to_le_bytes();
            // BITWISE: top 4 bits of aux32_1 carry the 4-bit sub-block scale (0..=15)
            // CAST: u32 → f32 lossless for 0..=15
            let db = d * (0.5 + ((aux32_1 >> 28) as f32)) * 0.25;

            for l in 0..4 {
                let grid = IQ2XXS_GRID[usize::from(aux8[l])].to_le_bytes();
                // BITWISE: 7-bit sign index for group `l` is bits [7l, 7l+7]
                // CAST: l (usize ≤ 3) → u32 for the shift amount
                let l_u32 = l as u32;
                let signs = KSIGNS_IQ2XS[((aux32_1 >> (7 * l_u32)) & 0x7F) as usize];
                write_signed_grid(scratch, y_off, db, grid, signs);
                y_off += 8;
            }
        }
    })
}

/// `IQ2_XS` kernel — 74-byte super-blocks: `d: f16` + `qs[32]: u16` +
/// `scales[8]: u8`.
///
/// ~2.31 bpw quant. Each `qs` word packs a 9-bit index into
/// [`IQ2XS_GRID`] (low bits) and a 7-bit sign index (high bits) resolved
/// via [`KSIGNS_IQ2XS`]. The per-`ib32` `scales[ib32]` byte splits into
/// two 4-bit nibbles: the low nibble scales groups `l ∈ {0, 1}`, the high
/// nibble scales groups `l ∈ {2, 3}`, both via the same
/// `db = d × (0.5 + nibble) × 0.25` formula as `IQ2_XXS`.
///
/// Ported verbatim from `ggml-quants.c::dequantize_row_iq2_xs`.
#[allow(
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::needless_range_loop
)]
fn dequant_iq2_xs<F>(data: &[u8], sink: F) -> crate::Result<()>
where
    F: FnMut(&[u8]) -> crate::Result<()>,
{
    run_super_kernel(data, 74, sink, |in_block, scratch| {
        // Field offsets: d [0..2], qs [2..66] (u16[32] LE), scales [66..74]
        let d = read_f16_bytes([in_block[0], in_block[1]]);
        let qs = &in_block[2..66];
        let scales = &in_block[66..74];

        let mut y_off: usize = 0;
        for ib32 in 0..8 {
            // BITWISE: low / high 4-bit scale nibbles for the two halves
            let db0 = d * (0.5 + f32::from(scales[ib32] & 0x0F)) * 0.25;
            let db1 = d * (0.5 + f32::from(scales[ib32] >> 4)) * 0.25;
            for l in 0..4 {
                let word_off = 8 * ib32 + 2 * l;
                let word = u16::from_le_bytes([qs[word_off], qs[word_off + 1]]);
                // BITWISE: low 9 bits index the 512-entry grid; high 7 bits
                // index the 128-entry sign table.
                let grid = IQ2XS_GRID[usize::from(word & 0x01FF)].to_le_bytes();
                let signs = KSIGNS_IQ2XS[usize::from(word >> 9)];
                let dl = if l < 2 { db0 } else { db1 };
                write_signed_grid(scratch, y_off, dl, grid, signs);
                y_off += 8;
            }
        }
    })
}

/// `IQ2_S` kernel — 82-byte super-blocks: `d: f16` + `qs[64]: u8`
/// (`qs[0..32]` = grid-index lows, `qs[32..64]` = inline sign masks) +
/// `qh[8]: u8` + `scales[8]: u8`.
///
/// ~2.50 bpw quant. The grid index is 10 bits wide: bits `[0..8]` come
/// from `qs[4*ib32 + l]` and bits `[8..10]` are extracted from
/// `qh[ib32]` as the 2-bit chunk at bit position `[2l, 2l+1]`, shifted
/// into bits `[8..10]` by the C trick `qh[ib32] << (8 - 2*l) & 0x300`.
/// Unlike `IQ2_XXS` / `IQ2_XS`, signs are stored **inline** at
/// `qs[32 + 4*ib32 + l]` rather than indexed through [`KSIGNS_IQ2XS`].
/// The scale split across `scales[ib32]`'s nibbles is identical to
/// `IQ2_XS`.
///
/// Ported verbatim from `ggml-quants.c::dequantize_row_iq2_s`.
#[allow(
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::needless_range_loop
)]
fn dequant_iq2_s<F>(data: &[u8], sink: F) -> crate::Result<()>
where
    F: FnMut(&[u8]) -> crate::Result<()>,
{
    run_super_kernel(data, 82, sink, |in_block, scratch| {
        // Field offsets: d [0..2], qs_indices [2..34], qs_signs [34..66],
        //                qh [66..74], scales [74..82].
        let d = read_f16_bytes([in_block[0], in_block[1]]);
        let qs_indices = &in_block[2..34];
        let qs_signs = &in_block[34..66];
        let qh = &in_block[66..74];
        let scales = &in_block[74..82];

        let mut y_off: usize = 0;
        for ib32 in 0..8 {
            let db0 = d * (0.5 + f32::from(scales[ib32] & 0x0F)) * 0.25;
            let db1 = d * (0.5 + f32::from(scales[ib32] >> 4)) * 0.25;
            for l in 0..4 {
                // BITWISE: assemble 10-bit grid index = qs[l] | (qh-bits [2l, 2l+1] << 8)
                // CAST: l (usize ≤ 3) → u32 for the shift amount
                let l_u32 = l as u32;
                let high = (u32::from(qh[ib32]) << (8 - 2 * l_u32)) & 0x0300;
                let idx = u32::from(qs_indices[4 * ib32 + l]) | high;
                let grid = IQ2S_GRID[idx as usize].to_le_bytes();
                let signs = qs_signs[4 * ib32 + l];
                let dl = if l < 2 { db0 } else { db1 };
                write_signed_grid(scratch, y_off, dl, grid, signs);
                y_off += 8;
            }
        }
    })
}

/// `IQ3_XXS` kernel — 98-byte super-blocks: `d: f16` + `qs[64]: u8`
/// (grid-index bytes) + `scales_and_signs[8]: u32` (packed at `qs[64..96]`).
///
/// ~3.06 bpw quant. Structurally analogous to [`dequant_iq2_xxs`]: for each
/// `ib32 ∈ 0..8`, a 32-bit `aux32` word supplies a 4-bit sub-block scale
/// in its top 4 bits and four 7-bit sign indices in bits `[0..28]`. Each
/// group of 4 output values is selected by **two** grid entries of the
/// 256-entry [`IQ3XXS_GRID`] (`[u32; 256]`, 4-byte codebook vectors) —
/// their 4 + 4 bytes are concatenated into the 8-element packed grid
/// [`write_signed_grid`] expects.
///
/// The scale formula is `db = d × (0.5 + scale_nibble) × 0.5` — the extra
/// factor-of-2 vs `IQ2_XXS`'s `× 0.25` reflects the wider codebook vector
/// range (3-bit codes → larger magnitudes per element).
///
/// Ported verbatim from `ggml-quants.c::dequantize_row_iq3_xxs`.
#[allow(
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]
fn dequant_iq3_xxs<F>(data: &[u8], sink: F) -> crate::Result<()>
where
    F: FnMut(&[u8]) -> crate::Result<()>,
{
    run_super_kernel(data, 98, sink, |in_block, scratch| {
        // Field offsets: d [0..2], qs [2..98].
        // Inside qs: qs[0..64] = grid-index bytes (8 per sub-block),
        //            qs[64..96] = 8 × u32 scales_and_signs words.
        let d = read_f16_bytes([in_block[0], in_block[1]]);
        let qs = &in_block[2..98];

        let mut y_off: usize = 0;
        for ib32 in 0..8 {
            let ss_off = 64 + 4 * ib32;
            let aux32 =
                u32::from_le_bytes([qs[ss_off], qs[ss_off + 1], qs[ss_off + 2], qs[ss_off + 3]]);
            // BITWISE: top 4 bits of aux32 = 4-bit sub-block scale nibble (0..=15)
            // CAST: u32 → f32 lossless for 0..=15
            let db = d * (0.5 + ((aux32 >> 28) as f32)) * 0.5;

            for l in 0..4 {
                // BITWISE: 7-bit sign index for group `l` lives in bits [7l, 7l+7]
                // CAST: l (usize ≤ 3) → u32 for shift amount
                let l_u32 = l as u32;
                let signs = KSIGNS_IQ2XS[((aux32 >> (7 * l_u32)) & 0x7F) as usize];
                let g1 = IQ3XXS_GRID[usize::from(qs[8 * ib32 + 2 * l])].to_le_bytes();
                let g2 = IQ3XXS_GRID[usize::from(qs[8 * ib32 + 2 * l + 1])].to_le_bytes();
                // Concatenate two 4-byte grid vectors into the 8-byte layout
                // `write_signed_grid` expects. `signs` bits [0..4] gate g1,
                // bits [4..8] gate g2 — same convention as `KMASK_IQ2XS[j]`.
                let combined = [g1[0], g1[1], g1[2], g1[3], g2[0], g2[1], g2[2], g2[3]];
                write_signed_grid(scratch, y_off, db, combined, signs);
                y_off += 8;
            }
        }
    })
}

/// `IQ3_S` kernel — 110-byte super-blocks: `d: f16` + `qs[64]: u8`
/// (low 8 bits of a 9-bit grid index) + `qh[8]: u8` (high bit of the
/// grid index, one `qh` byte covers two sub-blocks) + `signs[32]: u8`
/// (inline sign masks) + `scales[4]: u8`.
///
/// ~3.44 bpw quant. Structurally the most elaborate of the IQ3 family:
///
/// - Each `scales[outer]` byte covers **two** consecutive sub-blocks. Its
///   low nibble drives `db1 = d × (1 + 2·nibble)` for sub-block A, its
///   high nibble drives `db2` for sub-block B. The unusual
///   odd-integer-multiplier formula differs from the `(0.5 + n) × 0.x`
///   pattern used by `IQ2_XXS` / `IQ2_XS` / `IQ2_S` / `IQ3_XXS`.
/// - The 9-bit grid index (into [`IQ3S_GRID`], `[u32; 512]`) is assembled
///   from `qs[l]` (low 8 bits) and a 1-bit contribution from `qh` shifted
///   into position 8. Two different shift patterns —
///   `(qh << (8 - 2·l)) & 0x100` for grid1 and `(qh << (7 - 2·l)) & 0x100`
///   for grid2 — pull out adjacent bits of the same `qh` byte into the
///   grid-index MSB across the 4 groups.
/// - Signs are stored inline in `signs[]` rather than resolved through
///   `KSIGNS_IQ2XS` — one 8-bit mask per 4-output group, same sign
///   convention as [`dequant_iq2_s`].
///
/// Processing: 4 outer iterations × 2 sub-blocks per outer × 4 groups per
/// sub-block × 8 outputs per group = 256 outputs per super-block. Ported
/// verbatim from `ggml-quants.c::dequantize_row_iq3_s`.
#[allow(
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::needless_range_loop,
    clippy::similar_names
)]
fn dequant_iq3_s<F>(data: &[u8], sink: F) -> crate::Result<()>
where
    F: FnMut(&[u8]) -> crate::Result<()>,
{
    run_super_kernel(data, 110, sink, |in_block, scratch| {
        // Field offsets: d [0..2], qs [2..66], qh [66..74], signs [74..106],
        //                scales [106..110].
        let d = read_f16_bytes([in_block[0], in_block[1]]);
        let qs = &in_block[2..66];
        let qh = &in_block[66..74];
        let signs = &in_block[74..106];
        let scales = &in_block[106..110];

        let mut y_off: usize = 0;
        for outer in 0..4 {
            // BITWISE: low / high 4-bit scale nibbles for the two halves.
            // CAST: u8 → f32 lossless for 0..=15; db = d × (1 + 2·nibble)
            // yields odd-integer scale multipliers in [1, 31].
            let scale_byte = scales[outer];
            let db1 = d * (1.0 + 2.0 * f32::from(scale_byte & 0x0F));
            let db2 = d * (1.0 + 2.0 * f32::from(scale_byte >> 4));

            // Sub-block A (first of the pair): uses qh[2·outer], db1.
            let qh_first = u32::from(qh[2 * outer]);
            let qs_first = 16 * outer;
            let sg_first = 8 * outer;
            for l in 0..4 {
                // CAST: l (usize ≤ 3) → u32 for shift amount
                let l_u32 = l as u32;
                // BITWISE: high bit of the 9-bit grid index — two different
                // shift patterns pull out the l-th and (l+1)-th bits of qh
                // into index position 8 (mask 0x100).
                let hi1 = (qh_first << (8 - 2 * l_u32)) & 0x0100;
                let hi2 = (qh_first << (7 - 2 * l_u32)) & 0x0100;
                let idx1 = u32::from(qs[qs_first + 2 * l]) | hi1;
                let idx2 = u32::from(qs[qs_first + 2 * l + 1]) | hi2;
                let g1 = IQ3S_GRID[idx1 as usize].to_le_bytes();
                let g2 = IQ3S_GRID[idx2 as usize].to_le_bytes();
                let combined = [g1[0], g1[1], g1[2], g1[3], g2[0], g2[1], g2[2], g2[3]];
                let signs_l = signs[sg_first + l];
                write_signed_grid(scratch, y_off, db1, combined, signs_l);
                y_off += 8;
            }

            // Sub-block B (second of the pair): uses qh[2·outer + 1], db2.
            let qh_second = u32::from(qh[2 * outer + 1]);
            let qs_second = 16 * outer + 8;
            let sg_second = 8 * outer + 4;
            for l in 0..4 {
                let l_u32 = l as u32;
                let hi1 = (qh_second << (8 - 2 * l_u32)) & 0x0100;
                let hi2 = (qh_second << (7 - 2 * l_u32)) & 0x0100;
                let idx1 = u32::from(qs[qs_second + 2 * l]) | hi1;
                let idx2 = u32::from(qs[qs_second + 2 * l + 1]) | hi2;
                let g1 = IQ3S_GRID[idx1 as usize].to_le_bytes();
                let g2 = IQ3S_GRID[idx2 as usize].to_le_bytes();
                let combined = [g1[0], g1[1], g1[2], g1[3], g2[0], g2[1], g2[2], g2[3]];
                let signs_l = signs[sg_second + l];
                write_signed_grid(scratch, y_off, db2, combined, signs_l);
                y_off += 8;
            }
        }
    })
}

/// `IQ1_S` kernel — 50-byte super-blocks: `d: f16` + `qs[32]: u8` (low 8
/// bits of an 11-bit grid index) + `qh[8]: u16` (one u16 word per `ib32`,
/// packing four 3-bit high-index fields at bits `[0..12]`, a 3-bit
/// sub-block scale at bits `[12..15]`, and a 1-bit delta-sign flag at
/// bit `[15]`).
///
/// ~1.56 bpw quant. The smallest member of the `IQ*` family with a top-
/// level `d` field. Each `ib32 ∈ 0..8` reads one `qh` u16 word that
/// packs:
/// - bits `[0..12]`: four 3-bit high-index nibbles (one per group `l ∈ 0..4`)
/// - bits `[12..15]`: 3-bit sub-block scale → `dl = d × (2·n + 1)`
/// - bit `[15]`: delta sign flag → `delta = ±IQ1S_DELTA`
///
/// Per group, the 11-bit grid index `qs[l] | high_3_bits << 8` selects an
/// 8-byte signed `i8` codebook vector from [`IQ1S_GRID`]. The output
/// formula is `y = dl × (grid[j] + delta)` — additive bias on a single
/// signed codebook value, not a multiplicative ±1 sign mask like the
/// IQ2/IQ3 kernels.
///
/// Ported verbatim from `ggml-quants.c::dequantize_row_iq1_s`.
#[allow(
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]
fn dequant_iq1_s<F>(data: &[u8], sink: F) -> crate::Result<()>
where
    F: FnMut(&[u8]) -> crate::Result<()>,
{
    run_super_kernel(data, 50, sink, |in_block, scratch| {
        // Field offsets: d [0..2], qs [2..34], qh [34..50] (u16[8] LE — one
        // qh word per ib32, totalling 16 bytes).
        let d = read_f16_bytes([in_block[0], in_block[1]]);
        let qs = &in_block[2..34];
        let qh_bytes = &in_block[34..50];

        let mut y_off: usize = 0;
        for ib32 in 0..8 {
            let qh_off = 2 * ib32;
            let qh = u16::from_le_bytes([qh_bytes[qh_off], qh_bytes[qh_off + 1]]);
            // BITWISE: bits [12..15] of qh = sub-block 3-bit scale; bit 15 = delta sign.
            // The scale yields the odd-integer multiplier `2·n + 1 ∈ [1, 15]`.
            let dl = d * f32::from(2 * ((qh >> 12) & 7) + 1);
            let delta = if qh & 0x8000 == 0 {
                IQ1S_DELTA
            } else {
                -IQ1S_DELTA
            };
            for l in 0..4 {
                // BITWISE: 11-bit grid index = qs[l] | (qh>>(3·l) & 7) << 8
                // CAST: l (usize ≤ 3) → u32 for the shift amount
                let l_u32 = l as u32;
                let hi = ((u32::from(qh) >> (3 * l_u32)) & 7) << 8;
                let idx = u32::from(qs[4 * ib32 + l]) | hi;
                let grid = iq1s_grid_as_i8(IQ1S_GRID[idx as usize]);
                write_delta_grid(scratch, y_off, dl, grid, delta);
                y_off += 8;
            }
        }
    })
}

/// `IQ1_M` kernel — 56-byte super-blocks: `qs[32]: u8` + `qh[16]: u8` +
/// `scales[8]: u8`.
///
/// ~1.75 bpw quant. **No top-level `d` field** — the super-block float
/// scale is reconstructed from a scattered 16-bit pattern across
/// `scales[]` (treated as `u16[4]`):
///
/// ```text
/// scale_u16 = (sc[0] >> 12)
///           | ((sc[1] >> 8)  & 0x00F0)
///           | ((sc[2] >> 4)  & 0x0F00)
///           |  (sc[3]        & 0xF000)
/// d = f16::from_bits(scale_u16) as f32
/// ```
///
/// The other 3 nibbles of each `sc[k]` carry the eight 3-bit sub-block
/// scales (two per `ib32`, low / high). Each `ib32` decodes 4 grid
/// indices using `qh` as a high-bit source (3 bits per group, 4 groups
/// per `ib32`); per-group `±IQ1S_DELTA` comes from `qh & 0x08` /
/// `qh & 0x80` for the two halves. Groups `0..2` use `dl1`; groups
/// `2..4` use `dl2`. Reuses the same 2048-entry [`IQ1S_GRID`] codebook
/// as `IQ1_S`.
///
/// Ported verbatim from `ggml-quants.c::dequantize_row_iq1_m`.
#[allow(
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::similar_names
)]
fn dequant_iq1_m<F>(data: &[u8], sink: F) -> crate::Result<()>
where
    F: FnMut(&[u8]) -> crate::Result<()>,
{
    run_super_kernel(data, 56, sink, |in_block, scratch| {
        // Field offsets: qs [0..32], qh [32..48], scales [48..56].
        // No top-level `d`: the super-block scale is reconstructed below.
        let qs = &in_block[0..32];
        let qh = &in_block[32..48];
        let scales = &in_block[48..56];

        // BITWISE: assemble 16-bit f16 scale from 4 u16 LE words' top nibbles.
        let sc0 = u16::from_le_bytes([scales[0], scales[1]]);
        let sc1 = u16::from_le_bytes([scales[2], scales[3]]);
        let sc2 = u16::from_le_bytes([scales[4], scales[5]]);
        let sc3 = u16::from_le_bytes([scales[6], scales[7]]);
        let scale_u16 =
            (sc0 >> 12) | ((sc1 >> 8) & 0x00F0) | ((sc2 >> 4) & 0x0F00) | (sc3 & 0xF000);
        let d = f32::from(half::f16::from_bits(scale_u16));

        let sc = [sc0, sc1, sc2, sc3];

        // Pick ±IQ1S_DELTA according to one bit in a `qh` byte.
        let delta_for = |qh_byte: u8, mask: u8| -> f32 {
            if qh_byte & mask == 0 {
                IQ1S_DELTA
            } else {
                -IQ1S_DELTA
            }
        };

        let mut y_off: usize = 0;
        for ib32 in 0..8 {
            // BITWISE: each `sc[ib32/2]` u16 carries two 3-bit sub-block
            // scales per ib32 — ib32-even uses bits [0..6], ib32-odd uses
            // bits [6..12]. The scale yields odd-integer multipliers `2·n + 1`.
            // CAST: usize → u32 for the shift amount.
            let shift_a = 6 * ((ib32 % 2) as u32);
            let dl1 = d * f32::from(2 * ((sc[ib32 / 2] >> shift_a) & 7) + 1);
            let dl2 = d * f32::from(2 * ((sc[ib32 / 2] >> (shift_a + 3)) & 7) + 1);

            // Each ib32 consumes 4 qs bytes + 2 qh bytes.
            let qs_off = 4 * ib32;
            let qh_off = 2 * ib32;
            let qh_a = qh[qh_off];
            let qh_b = qh[qh_off + 1];
            let qh_a_u32 = u32::from(qh_a);
            let qh_b_u32 = u32::from(qh_b);

            // BITWISE: 11-bit grid index = qs | ((qh shifted) & 0x700).
            // Per-group delta sign: bit 3 of qh_a (group 0), bit 7 of qh_a
            // (group 1), bit 3 of qh_b (group 2), bit 7 of qh_b (group 3).
            // Groups 0,1 use dl1; groups 2,3 use dl2.
            let groups: [(u32, f32, f32); 4] = [
                (
                    u32::from(qs[qs_off]) | ((qh_a_u32 << 8) & 0x0700),
                    dl1,
                    delta_for(qh_a, 0x08),
                ),
                (
                    u32::from(qs[qs_off + 1]) | ((qh_a_u32 << 4) & 0x0700),
                    dl1,
                    delta_for(qh_a, 0x80),
                ),
                (
                    u32::from(qs[qs_off + 2]) | ((qh_b_u32 << 8) & 0x0700),
                    dl2,
                    delta_for(qh_b, 0x08),
                ),
                (
                    u32::from(qs[qs_off + 3]) | ((qh_b_u32 << 4) & 0x0700),
                    dl2,
                    delta_for(qh_b, 0x80),
                ),
            ];
            for &(idx, dl, delta) in &groups {
                let grid = iq1s_grid_as_i8(IQ1S_GRID[idx as usize]);
                write_delta_grid(scratch, y_off, dl, grid, delta);
                y_off += 8;
            }
        }
    })
}

/// `TQ1_0` kernel — 54-byte super-blocks: `qs[48]: u8` (base-3-packed
/// ternaries, 5 per byte) + `qh[4]: u8` (base-3-packed, 4 per byte) +
/// `d: f16`.
///
/// ~1.6875 bpw quant, ternary (values in `{-d, 0, +d}`). Invented for
/// BitNet-style 1.58-bit models. Each `qs` byte encodes 5 ternary
/// digits via `byte = d0 + 3·d1 + 9·d2 + 27·d3 + 81·d4` (constraint
/// `byte ≤ 242 = 3^5 - 1`); each `qh` byte encodes 4 ternaries
/// (`byte = d0 + 3·d1 + 9·d2 + 27·d3`, constraint `byte ≤ 80 = 3^4 - 1`).
///
/// The C reference uses an unusual chunked iteration order — qs is
/// processed in 32-byte chunks then 16-byte chunks, with each chunk
/// emitting all 5 ternary digits for each of its bytes before moving
/// on. The output element ordering depends on this chunking, so the
/// Rust port must replicate it for bit-exactness:
/// - qs main loop: `chunk_off = 0`, n ∈ 0..5, m ∈ 0..32 → 160 outputs
/// - qs tail loop: `chunk_off = 32`, n ∈ 0..5, m ∈ 0..16 → 80 outputs
/// - qh loop:      n ∈ 0..4, j ∈ 0..4                   → 16 outputs
///
/// Total: 256 outputs per super-block. Each digit decode goes through
/// the shared [`decode_pow3_ternary`] helper.
///
/// Ported verbatim from `ggml-quants.c::dequantize_row_tq1_0`.
#[allow(clippy::indexing_slicing, clippy::needless_range_loop)]
fn dequant_tq1_0<F>(data: &[u8], sink: F) -> crate::Result<()>
where
    F: FnMut(&[u8]) -> crate::Result<()>,
{
    run_super_kernel(data, 54, sink, |in_block, scratch| {
        // Field offsets: qs [0..48], qh [48..52], d [52..54].
        let qs = &in_block[0..48];
        let qh = &in_block[48..52];
        let d = read_f16_bytes([in_block[52], in_block[53]]);

        let mut y_off: usize = 0;
        // qs main loop: one 32-byte chunk × 5 ternaries × 32 elements = 160 outputs.
        for n in 0..5 {
            for m in 0..32 {
                scratch[y_off] = decode_pow3_ternary(qs[m], POW3_TQ1_0[n], d);
                y_off += 1;
            }
        }
        // qs tail: one 16-byte chunk × 5 ternaries × 16 elements = 80 outputs.
        for n in 0..5 {
            for m in 0..16 {
                scratch[y_off] = decode_pow3_ternary(qs[32 + m], POW3_TQ1_0[n], d);
                y_off += 1;
            }
        }
        // qh: 4 ternaries × 4 bytes = 16 outputs.
        for n in 0..4 {
            for j in 0..4 {
                scratch[y_off] = decode_pow3_ternary(qh[j], POW3_TQ1_0[n], d);
                y_off += 1;
            }
        }
    })
}

/// `TQ2_0` kernel — 66-byte super-blocks: `qs[64]: u8` (4 ternaries per
/// byte, 2 bits each) + `d: f16`.
///
/// ~2.0625 bpw quant, ternary (values in `{-d, 0, +d}`). Simpler
/// packing than `TQ1_0`: each `qs` byte holds 4 × 2-bit values, where
/// each 2-bit value `q ∈ {0, 1, 2}` decodes as `(q - 1) × d`. Output
/// element ordering follows the C reference's nested loop: 32-byte
/// chunks of qs × 4 bit-positions × 32 elements per chunk-position.
///
/// Ported verbatim from `ggml-quants.c::dequantize_row_tq2_0`.
#[allow(clippy::indexing_slicing)]
fn dequant_tq2_0<F>(data: &[u8], sink: F) -> crate::Result<()>
where
    F: FnMut(&[u8]) -> crate::Result<()>,
{
    run_super_kernel(data, 66, sink, |in_block, scratch| {
        // Field offsets: qs [0..64], d [64..66].
        let qs = &in_block[0..64];
        let d = read_f16_bytes([in_block[64], in_block[65]]);

        let mut y_off: usize = 0;
        // Two 32-byte chunks of qs.
        for chunk_off in [0_usize, 32] {
            for l in 0..4_u32 {
                for m in 0..32 {
                    // BITWISE: extract the 2-bit ternary at position l*2 in qs[m].
                    // CAST: u8 → f32 via From; (q - 1) ∈ {-1, 0, +1}.
                    let q = (qs[chunk_off + m] >> (l * 2)) & 3;
                    scratch[y_off] = (f32::from(q) - 1.0) * d;
                    y_off += 1;
                }
            }
        }
    })
}

// ---------------------------------------------------------------------------
// K-quant scale extractors
// ---------------------------------------------------------------------------

/// Extracts the 6-bit `(scale, min)` pair for sub-block `j` from the
/// packed `scales[12]` field used by `Q4_K` and `Q5_K`.
///
/// Ported verbatim from ggml-quants.c's `get_scale_min_k4` helper.
#[inline]
#[allow(clippy::indexing_slicing)]
fn get_scale_min_k4(j: usize, scales: &[u8; 12]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        let d = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (d, m)
    }
}

/// Unpacks the 12-byte packed `scales` field of a `Q3_K` block into 16
/// values (stored as `i8` but the caller biases by `-32` in the dequant
/// formula, so the actual range here is `0..=63`).
///
/// Ported verbatim from ggml-quants.c's `kmask1`/`kmask2` permute. The
/// `>> 0` in the first permute line is kept for symmetry with the
/// `>> 2`/`>> 4`/`>> 6` siblings, so `clippy::identity_op` is allowed.
#[inline]
#[allow(
    clippy::indexing_slicing,
    clippy::identity_op,
    clippy::as_conversions,
    clippy::cast_possible_wrap
)]
fn q3_k_unpack_scales(packed: &[u8; 12]) -> [i8; 16] {
    const KMASK1: u32 = 0x0303_0303;
    const KMASK2: u32 = 0x0f0f_0f0f;

    let aux0_in = u32::from_le_bytes([packed[0], packed[1], packed[2], packed[3]]);
    let aux1_in = u32::from_le_bytes([packed[4], packed[5], packed[6], packed[7]]);
    let aux2_in = u32::from_le_bytes([packed[8], packed[9], packed[10], packed[11]]);

    let tmp = aux2_in;
    // BITWISE: two-level permute; see ggml-quants.c `dequantize_row_q3_K`
    let aux0 = (aux0_in & KMASK2) | (((tmp >> 0) & KMASK1) << 4);
    let aux1 = (aux1_in & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
    let aux2 = ((aux0_in >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
    let aux3 = ((aux1_in >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);

    // Reinterpret the 16 bytes (4 × u32 LE) as [i8; 16] following the
    // ggml reference's `(int8_t*)aux` cast.
    let mut out = [0i8; 16];
    for (i, &word) in [aux0, aux1, aux2, aux3].iter().enumerate() {
        let bytes = word.to_le_bytes();
        // CAST: u8 → i8, intentional signed reinterpret
        out[i * 4] = bytes[0] as i8;
        out[i * 4 + 1] = bytes[1] as i8;
        out[i * 4 + 2] = bytes[2] as i8;
        out[i * 4 + 3] = bytes[3] as i8;
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::wildcard_enum_match_arm,
    clippy::manual_is_multiple_of,
    clippy::needless_range_loop
)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------
    // Helpers shared by test fixtures
    // -----------------------------------------------------------------

    /// Decodes a little-endian `BF16` byte pair back to `f32` for assertions.
    fn bf16_pair_to_f32(bytes: &[u8]) -> f32 {
        let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
        f32::from(half::bf16::from_bits(bits))
    }

    /// Packs an `f32` into the 2-byte little-endian `ggml_half` form.
    fn f16_bytes(v: f32) -> [u8; 2] {
        half::f16::from_f32(v).to_le_bytes()
    }

    // -----------------------------------------------------------------
    // Dispatcher validation
    // -----------------------------------------------------------------

    #[test]
    fn zero_elements_returns_empty() {
        let out = dequantize_gguf_to_bf16(&[], GgufType::Q4_0, 0).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn rejects_non_multiple_of_block_size() {
        let data = vec![0u8; 18];
        let err = dequantize_gguf_to_bf16(&data, GgufType::Q4_0, 17).unwrap_err();
        assert!(matches!(err, AnamnesisError::Parse { .. }));
    }

    #[test]
    fn rejects_wrong_data_length() {
        // Q4_0: 1 block = 18 bytes; give it 17.
        let data = vec![0u8; 17];
        let err = dequantize_gguf_to_bf16(&data, GgufType::Q4_0, 32).unwrap_err();
        assert!(matches!(err, AnamnesisError::Parse { .. }));
    }

    #[test]
    fn mxfp4_now_supported_returns_parse_error_on_bad_length() {
        // Phase 4.5 step 6 added the MXFP4 dequant kernel — it no longer
        // triggers `Unsupported`. Passing an empty data slice with a
        // non-zero element count now flows into `validate_dequant_input`,
        // which surfaces a `Parse` error on the data-length mismatch
        // (1 block of MXFP4 = 17 bytes; 0 bytes is not 17).
        let err = dequantize_gguf_to_bf16(&[], GgufType::MXFP4, 32).unwrap_err();
        assert!(matches!(err, AnamnesisError::Parse { .. }));
    }

    #[test]
    fn overflow_guard_rejects_huge_n_elements() {
        // Largest multiple of 32 that's strictly > usize::MAX / 2 so that
        // `n_elements * 2` overflows. Works on both 64-bit and 32-bit
        // because the calculation scales to `usize::MAX`.
        let n_elements = (usize::MAX / 32) * 32;
        let err = dequantize_gguf_to_bf16(&[], GgufType::Q4_0, n_elements).unwrap_err();
        match err {
            AnamnesisError::Parse { reason } => {
                assert!(
                    reason.contains("overflows usize"),
                    "expected overflow error, got: {reason}"
                );
            }
            other => panic!("expected Parse overflow, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------
    // Streaming API
    // -----------------------------------------------------------------

    #[test]
    fn streaming_calls_sink_once_per_block() {
        // 2 Q4_0 blocks, d = 0.0, qs = 0 — each sink call should receive 64 bytes
        let mut data = vec![0u8; 36];
        data[0..2].copy_from_slice(&f16_bytes(0.0));
        data[18..20].copy_from_slice(&f16_bytes(0.0));
        let mut block_count = 0;
        let mut total_bytes = 0;
        dequantize_gguf_blocks_to_bf16(&data, GgufType::Q4_0, 64, |block| {
            block_count += 1;
            total_bytes += block.len();
            Ok(())
        })
        .unwrap();
        assert_eq!(block_count, 2);
        assert_eq!(total_bytes, 128);
    }

    #[test]
    fn streaming_propagates_sink_error() {
        let mut data = vec![0u8; 18];
        data[0..2].copy_from_slice(&f16_bytes(0.0));
        let err = dequantize_gguf_blocks_to_bf16(&data, GgufType::Q4_0, 32, |_block| {
            Err(AnamnesisError::Parse {
                reason: "sink aborted".into(),
            })
        })
        .unwrap_err();
        match err {
            AnamnesisError::Parse { reason } => assert_eq!(reason, "sink aborted"),
            other => panic!("expected Parse from sink, got {other:?}"),
        }
    }

    #[test]
    fn streaming_matches_vec_variant() {
        // Verify the two entry points produce byte-identical output for
        // a non-trivial Q4_0 fixture.
        let mut data = vec![0u8; 36];
        data[0..2].copy_from_slice(&f16_bytes(1.0));
        for j in 0..16 {
            data[2 + j] = (j as u8) << 4 | (j as u8);
        }
        data[18..20].copy_from_slice(&f16_bytes(-0.5));
        for j in 0..16 {
            data[20 + j] = 0xF0;
        }

        let vec_out = dequantize_gguf_to_bf16(&data, GgufType::Q4_0, 64).unwrap();

        let mut streamed = Vec::with_capacity(vec_out.len());
        dequantize_gguf_blocks_to_bf16(&data, GgufType::Q4_0, 64, |block| {
            streamed.extend_from_slice(block);
            Ok(())
        })
        .unwrap();

        assert_eq!(vec_out, streamed);
    }

    #[test]
    fn streaming_zero_elements_makes_no_sink_calls() {
        let mut calls = 0;
        dequantize_gguf_blocks_to_bf16(&[], GgufType::Q4_0, 0, |_block| {
            calls += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(calls, 0);
    }

    // -----------------------------------------------------------------
    // Q8_0
    // -----------------------------------------------------------------

    #[test]
    fn q8_0_all_zero_block() {
        // d = 0.0, qs = 0 → all output 0.0
        let mut block = vec![0u8; 34];
        block[0..2].copy_from_slice(&f16_bytes(0.0));
        let out = dequantize_gguf_to_bf16(&block, GgufType::Q8_0, 32).unwrap();
        assert_eq!(out.len(), 64);
        for chunk in out.chunks_exact(2) {
            assert_eq!(bf16_pair_to_f32(chunk), 0.0);
        }
    }

    #[test]
    fn q8_0_identity_scale() {
        // d = 1.0, qs = [-16..16]. Expect y[j] = qs[j].
        let mut block = vec![0u8; 34];
        block[0..2].copy_from_slice(&f16_bytes(1.0));
        for j in 0..32 {
            let v: i8 = (j as i8).wrapping_sub(16);
            block[2 + j] = v as u8;
        }
        let out = dequantize_gguf_to_bf16(&block, GgufType::Q8_0, 32).unwrap();
        for j in 0..32 {
            let expected = f32::from((j as i8).wrapping_sub(16));
            let got = bf16_pair_to_f32(&out[j * 2..j * 2 + 2]);
            assert_eq!(got, expected, "Q8_0[{j}]");
        }
    }

    #[test]
    fn q8_0_multi_block() {
        // 2 blocks: block 0 d=2.0 qs=1 → 2.0; block 1 d=-1.0 qs=3 → -3.0
        let mut data = vec![0u8; 68];
        data[0..2].copy_from_slice(&f16_bytes(2.0));
        for j in 0..32 {
            data[2 + j] = 1;
        }
        data[34..36].copy_from_slice(&f16_bytes(-1.0));
        for j in 0..32 {
            data[36 + j] = 3;
        }
        let out = dequantize_gguf_to_bf16(&data, GgufType::Q8_0, 64).unwrap();
        for j in 0..32 {
            assert_eq!(bf16_pair_to_f32(&out[j * 2..j * 2 + 2]), 2.0);
        }
        for j in 32..64 {
            assert_eq!(bf16_pair_to_f32(&out[j * 2..j * 2 + 2]), -3.0);
        }
    }

    // -----------------------------------------------------------------
    // Q4_0
    // -----------------------------------------------------------------

    #[test]
    fn q4_0_zeroed_nibbles_centers_at_minus_8() {
        // d = 1.0, qs = all 0x00 → every nibble = 0, y = (0 - 8) * 1 = -8
        let mut block = vec![0u8; 18];
        block[0..2].copy_from_slice(&f16_bytes(1.0));
        let out = dequantize_gguf_to_bf16(&block, GgufType::Q4_0, 32).unwrap();
        for j in 0..32 {
            assert_eq!(bf16_pair_to_f32(&out[j * 2..j * 2 + 2]), -8.0);
        }
    }

    #[test]
    fn q4_0_identity_nibble_round_trip() {
        // d = 1.0, qs[j] = (j << 4) | j for j < 16.
        // Expect y[j] = (j - 8) and y[j + 16] = (j - 8).
        let mut block = vec![0u8; 18];
        block[0..2].copy_from_slice(&f16_bytes(1.0));
        for j in 0..16usize {
            block[2 + j] = ((j as u8) << 4) | (j as u8);
        }
        let out = dequantize_gguf_to_bf16(&block, GgufType::Q4_0, 32).unwrap();
        for j in 0..16 {
            let expected = j as f32 - 8.0;
            assert_eq!(bf16_pair_to_f32(&out[j * 2..j * 2 + 2]), expected);
            assert_eq!(
                bf16_pair_to_f32(&out[(j + 16) * 2..(j + 16) * 2 + 2]),
                expected
            );
        }
    }

    // -----------------------------------------------------------------
    // Q4_1
    // -----------------------------------------------------------------

    #[test]
    fn q4_1_affine_transform() {
        // d = 1.0, m = 10.0, qs = 0xFF → y = 15 * 1 + 10 = 25 at every position
        let mut block = vec![0u8; 20];
        block[0..2].copy_from_slice(&f16_bytes(1.0));
        block[2..4].copy_from_slice(&f16_bytes(10.0));
        for j in 0..16 {
            block[4 + j] = 0xFF;
        }
        let out = dequantize_gguf_to_bf16(&block, GgufType::Q4_1, 32).unwrap();
        for j in 0..32 {
            assert_eq!(bf16_pair_to_f32(&out[j * 2..j * 2 + 2]), 25.0);
        }
    }

    // -----------------------------------------------------------------
    // Q5_0 / Q5_1
    // -----------------------------------------------------------------

    #[test]
    fn q5_0_all_high_bits_set() {
        // d = 1.0, qh = 0xFFFFFFFF, qs = 0 → 5-bit value 16 everywhere,
        // y = (16 - 16) * 1 = 0
        let mut block = vec![0u8; 22];
        block[0..2].copy_from_slice(&f16_bytes(1.0));
        block[2..6].copy_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
        let out = dequantize_gguf_to_bf16(&block, GgufType::Q5_0, 32).unwrap();
        for j in 0..32 {
            assert_eq!(bf16_pair_to_f32(&out[j * 2..j * 2 + 2]), 0.0);
        }
    }

    #[test]
    fn q5_1_identity_nibbles_no_high_bits() {
        // d = 1.0, m = 5.0, qh = 0, qs[j] = 0x11 → 5-bit = 1, y = 1 + 5 = 6
        let mut block = vec![0u8; 24];
        block[0..2].copy_from_slice(&f16_bytes(1.0));
        block[2..4].copy_from_slice(&f16_bytes(5.0));
        for j in 0..16 {
            block[8 + j] = 0x11;
        }
        let out = dequantize_gguf_to_bf16(&block, GgufType::Q5_1, 32).unwrap();
        for j in 0..32 {
            assert_eq!(bf16_pair_to_f32(&out[j * 2..j * 2 + 2]), 6.0);
        }
    }

    // -----------------------------------------------------------------
    // Q8_1
    // -----------------------------------------------------------------

    #[test]
    fn q8_1_ignores_s_field() {
        // d=1.0, s=999.0 (ignored), qs sequential i8
        let mut block = vec![0u8; 36];
        block[0..2].copy_from_slice(&f16_bytes(1.0));
        block[2..4].copy_from_slice(&f16_bytes(999.0));
        for j in 0..32 {
            let v: i8 = j as i8;
            block[4 + j] = v as u8;
        }
        let out = dequantize_gguf_to_bf16(&block, GgufType::Q8_1, 32).unwrap();
        for j in 0..32 {
            assert_eq!(bf16_pair_to_f32(&out[j * 2..j * 2 + 2]), f32::from(j as i8));
        }
    }

    // -----------------------------------------------------------------
    // Q8_K
    // -----------------------------------------------------------------

    #[test]
    fn q8_k_f32_scale() {
        // d = 0.5 (f32), qs sequential i8 pattern
        let mut block = vec![0u8; 292];
        block[0..4].copy_from_slice(&0.5_f32.to_le_bytes());
        for j in 0..256 {
            let v: i8 = ((j as i32) - 128) as i8;
            block[4 + j] = v as u8;
        }
        let out = dequantize_gguf_to_bf16(&block, GgufType::Q8_K, 256).unwrap();
        for j in 0..256 {
            let expected = f32::from(((j as i32) - 128) as i8) * 0.5;
            assert_eq!(
                bf16_pair_to_f32(&out[j * 2..j * 2 + 2]),
                expected,
                "Q8_K[{j}]"
            );
        }
    }

    // -----------------------------------------------------------------
    // Q2_K
    // -----------------------------------------------------------------

    #[test]
    fn q2_k_zero_scales_and_mins() {
        let block = vec![0u8; 84];
        let out = dequantize_gguf_to_bf16(&block, GgufType::Q2_K, 256).unwrap();
        for chunk in out.chunks_exact(2) {
            assert_eq!(bf16_pair_to_f32(chunk), 0.0);
        }
    }

    #[test]
    fn q2_k_uniform_scale_uniform_qs() {
        // d = 1.0, dmin = 0.0, scales = 0x02 (sc_lo=2, sc_hi=0), qs = 0x55
        // → 2-bit value 1 at every shift, y = 1 * 2 * 1 - 0 = 2
        let mut block = vec![0u8; 84];
        for j in 0..16 {
            block[j] = 0x02;
        }
        for j in 0..64 {
            block[16 + j] = 0x55;
        }
        block[80..82].copy_from_slice(&f16_bytes(1.0));
        block[82..84].copy_from_slice(&f16_bytes(0.0));
        let out = dequantize_gguf_to_bf16(&block, GgufType::Q2_K, 256).unwrap();
        for chunk in out.chunks_exact(2) {
            assert_eq!(bf16_pair_to_f32(chunk), 2.0);
        }
    }

    // -----------------------------------------------------------------
    // Q3_K: helper tests first
    // -----------------------------------------------------------------

    #[test]
    fn q3_k_unpack_scales_zero() {
        let scales = q3_k_unpack_scales(&[0u8; 12]);
        assert_eq!(scales, [0i8; 16]);
    }

    #[test]
    fn q3_k_unpack_scales_low_half_trivial() {
        let mut packed = [0u8; 12];
        packed[0] = 0x00;
        packed[1] = 0x01;
        packed[2] = 0x02;
        packed[3] = 0x03;
        let scales = q3_k_unpack_scales(&packed);
        assert_eq!(scales[0], 0);
        assert_eq!(scales[1], 1);
        assert_eq!(scales[2], 2);
        assert_eq!(scales[3], 3);
    }

    #[test]
    fn q3_k_all_zero_block() {
        let block = vec![0u8; 110];
        let out = dequantize_gguf_to_bf16(&block, GgufType::Q3_K, 256).unwrap();
        for chunk in out.chunks_exact(2) {
            assert_eq!(bf16_pair_to_f32(chunk), 0.0);
        }
    }

    // -----------------------------------------------------------------
    // Q4_K: helper tests first
    // -----------------------------------------------------------------

    #[test]
    fn get_scale_min_k4_low_half() {
        let scales = [
            0x00, 0x01, 0x02, 0x03, 0x10, 0x11, 0x12, 0x13, 0x00, 0x00, 0x00, 0x00,
        ];
        assert_eq!(get_scale_min_k4(0, &scales), (0x00, 0x10));
        assert_eq!(get_scale_min_k4(1, &scales), (0x01, 0x11));
        assert_eq!(get_scale_min_k4(2, &scales), (0x02, 0x12));
        assert_eq!(get_scale_min_k4(3, &scales), (0x03, 0x13));
    }

    #[test]
    fn get_scale_min_k4_high_half_zero() {
        let scales = [0u8; 12];
        for j in 0..8 {
            assert_eq!(get_scale_min_k4(j, &scales), (0, 0));
        }
    }

    #[test]
    fn q4_k_all_zero_block() {
        let block = vec![0u8; 144];
        let out = dequantize_gguf_to_bf16(&block, GgufType::Q4_K, 256).unwrap();
        for chunk in out.chunks_exact(2) {
            assert_eq!(bf16_pair_to_f32(chunk), 0.0);
        }
    }

    // -----------------------------------------------------------------
    // Q5_K
    // -----------------------------------------------------------------

    #[test]
    fn q5_k_all_zero_block() {
        let block = vec![0u8; 176];
        let out = dequantize_gguf_to_bf16(&block, GgufType::Q5_K, 256).unwrap();
        for chunk in out.chunks_exact(2) {
            assert_eq!(bf16_pair_to_f32(chunk), 0.0);
        }
    }

    // -----------------------------------------------------------------
    // Q6_K
    // -----------------------------------------------------------------

    #[test]
    fn q6_k_all_zero_block() {
        let block = vec![0u8; 210];
        let out = dequantize_gguf_to_bf16(&block, GgufType::Q6_K, 256).unwrap();
        for chunk in out.chunks_exact(2) {
            assert_eq!(bf16_pair_to_f32(chunk), 0.0);
        }
    }

    #[test]
    fn q6_k_identity_centers_at_minus_32() {
        // d = 1.0, scales = 1, ql/qh = 0 → 6-bit value 0 - 32 = -32, y = -32
        let mut block = vec![0u8; 210];
        for j in 0..16 {
            block[192 + j] = 1;
        }
        block[208..210].copy_from_slice(&f16_bytes(1.0));
        let out = dequantize_gguf_to_bf16(&block, GgufType::Q6_K, 256).unwrap();
        for chunk in out.chunks_exact(2) {
            assert_eq!(bf16_pair_to_f32(chunk), -32.0);
        }
    }

    // -----------------------------------------------------------------
    // IQ4_NL
    // -----------------------------------------------------------------

    #[test]
    fn iq4_nl_zero_scale_emits_zero() {
        // d = 0.0, arbitrary qs → every output 0.0
        let mut block = vec![0u8; 18];
        block[0..2].copy_from_slice(&f16_bytes(0.0));
        for j in 0..16 {
            block[2 + j] = 0xA5;
        }
        let out = dequantize_gguf_to_bf16(&block, GgufType::IQ4_NL, 32).unwrap();
        assert_eq!(out.len(), 64);
        for chunk in out.chunks_exact(2) {
            assert_eq!(bf16_pair_to_f32(chunk), 0.0);
        }
    }

    #[test]
    fn iq4_nl_zero_nibbles_emit_codebook_0() {
        // d = 1.0, qs = 0x00 → every nibble = 0, y = K_VALUES_IQ4_NL[0] = -127
        let mut block = vec![0u8; 18];
        block[0..2].copy_from_slice(&f16_bytes(1.0));
        let out = dequantize_gguf_to_bf16(&block, GgufType::IQ4_NL, 32).unwrap();
        for j in 0..32 {
            assert_eq!(
                bf16_pair_to_f32(&out[j * 2..j * 2 + 2]),
                -127.0,
                "IQ4_NL[{j}]"
            );
        }
    }

    #[test]
    fn iq4_nl_nibble_sweep() {
        // d = 1.0, qs[j] = (j << 4) | j for j ∈ 0..16
        // Low nibble j → y[j] = K_VALUES_IQ4_NL[j]
        // High nibble j → y[j + 16] = K_VALUES_IQ4_NL[j]
        let mut block = vec![0u8; 18];
        block[0..2].copy_from_slice(&f16_bytes(1.0));
        for j in 0..16usize {
            block[2 + j] = ((j as u8) << 4) | (j as u8);
        }
        let out = dequantize_gguf_to_bf16(&block, GgufType::IQ4_NL, 32).unwrap();
        for j in 0..16 {
            let expected = f32::from(K_VALUES_IQ4_NL[j]);
            assert_eq!(
                bf16_pair_to_f32(&out[j * 2..j * 2 + 2]),
                expected,
                "IQ4_NL low nibble[{j}]"
            );
            assert_eq!(
                bf16_pair_to_f32(&out[(j + 16) * 2..(j + 16) * 2 + 2]),
                expected,
                "IQ4_NL high nibble[{j}]"
            );
        }
    }

    // -----------------------------------------------------------------
    // IQ4_XS
    // -----------------------------------------------------------------

    #[test]
    fn iq4_xs_all_zero_block() {
        // All-zero block: d = 0.0 kills every output regardless of ls/nibble.
        let block = vec![0u8; 136];
        let out = dequantize_gguf_to_bf16(&block, GgufType::IQ4_XS, 256).unwrap();
        for chunk in out.chunks_exact(2) {
            assert_eq!(bf16_pair_to_f32(chunk), 0.0);
        }
    }

    #[test]
    fn iq4_xs_scale_bias_centers_at_minus_32() {
        // d = 1.0, scales_h = 0, scales_l = [0; 4], qs = 0x00
        // → ls = 0, dl = (0 - 32) = -32, nibble = 0, K_VALUES_IQ4_NL[0] = -127
        // → y = -32 * -127 = 4064.0 everywhere.
        //
        // 4064 = (1 + 126/128) × 2^11 is exactly representable in BF16
        // (7-bit mantissa at exponent 11 has ULP 16, and 4064 = 254 × 16).
        let mut block = vec![0u8; 136];
        block[0..2].copy_from_slice(&f16_bytes(1.0));
        let out = dequantize_gguf_to_bf16(&block, GgufType::IQ4_XS, 256).unwrap();
        for chunk in out.chunks_exact(2) {
            assert_eq!(bf16_pair_to_f32(chunk), 4064.0);
        }
    }

    #[test]
    fn iq4_xs_ls_combination() {
        // Test the 6-bit scale assembly. Sub-block 0 gets:
        //   scales_l[0] low nibble (bits [3:0]) = 0xF
        //   scales_h bits [1:0]                 = 0x3
        //   → ls = 0xF | (0x3 << 4) = 0x3F = 63, (ls - 32) = 31
        // All other sub-blocks see ls = 0, (ls - 32) = -32.
        //
        // qs = 0x00 → every nibble = 0 → K_VALUES_IQ4_NL[0] = -127
        // d = 1.0
        //
        // Expected (as f32, then rounded to BF16 by the dequantiser):
        //   sub-block 0 (y[0..32])     : 31 × -127 = -3937 → BF16 rounds to -3936
        //   sub-blocks 1..=7 (y[32..]) : -32 × -127 = 4064 (exactly BF16-representable)
        let mut block = vec![0u8; 136];
        block[0..2].copy_from_slice(&f16_bytes(1.0));
        // scales_h = 0x0003: high 2 bits of sub-block 0 = 0b11
        block[2] = 0x03;
        block[3] = 0x00;
        // scales_l[0] low nibble = 0xF; high nibble (sub-block 1 low) left 0.
        block[4] = 0x0F;
        let out = dequantize_gguf_to_bf16(&block, GgufType::IQ4_XS, 256).unwrap();
        let expected_sb0 = f32::from(half::bf16::from_f32(31.0 * -127.0));
        for j in 0..32 {
            assert_eq!(
                bf16_pair_to_f32(&out[j * 2..j * 2 + 2]),
                expected_sb0,
                "IQ4_XS sub-block 0[{j}]"
            );
        }
        for j in 32..256 {
            assert_eq!(
                bf16_pair_to_f32(&out[j * 2..j * 2 + 2]),
                4064.0,
                "IQ4_XS sub-block >=1[{j}]"
            );
        }
    }

    // -----------------------------------------------------------------
    // IQ2_XXS
    // -----------------------------------------------------------------

    #[test]
    fn iq2_xxs_all_zero_block() {
        let block = vec![0u8; 66];
        let out = dequantize_gguf_to_bf16(&block, GgufType::IQ2_XXS, 256).unwrap();
        for chunk in out.chunks_exact(2) {
            assert_eq!(bf16_pair_to_f32(chunk), 0.0);
        }
    }

    #[test]
    fn iq2_xxs_grid_entry_0_sign_entry_0() {
        // d = 1.0, aux32_0 = 0 (grid index 0 for all 4 groups),
        // aux32_1 = 0 (sign index 0 → ksigns[0] = 0 → all +1;
        // top 4 bits = 0 → scale_nibble = 0 → db = 1.0 × 0.5 × 0.25 = 0.125).
        // IQ2XXS_GRID[0] = 0x0808_0808_0808_0808 → every byte = 0x08 = 8.
        // Expected: db × 8 × 1 = 0.125 × 8 = 1.0 at every output.
        let mut block = vec![0u8; 66];
        block[0..2].copy_from_slice(&f16_bytes(1.0));
        let out = dequantize_gguf_to_bf16(&block, GgufType::IQ2_XXS, 256).unwrap();
        for chunk in out.chunks_exact(2) {
            assert_eq!(bf16_pair_to_f32(chunk), 1.0);
        }
    }

    #[test]
    fn iq2_xxs_scale_nibble_sweep() {
        // d = 1.0, qs = 0 except for aux32_1 of sub-block 0 which we hand-craft
        // to set the scale nibble to 15 → db = (0.5 + 15) × 0.25 = 3.875.
        // aux32_1 lives at qs[4..8] for ib32 = 0; bits [28..32] = 0xF → byte 7 = 0xF0.
        let mut block = vec![0u8; 66];
        block[0..2].copy_from_slice(&f16_bytes(1.0));
        block[2 + 7] = 0xF0; // top 4 bits of aux32_1 = 15
        let out = dequantize_gguf_to_bf16(&block, GgufType::IQ2_XXS, 256).unwrap();
        // Sub-block 0: db = 3.875, grid[0] = 8, signs = 0 → output = 3.875 × 8 = 31.0
        for j in 0..32 {
            assert_eq!(
                bf16_pair_to_f32(&out[j * 2..j * 2 + 2]),
                31.0,
                "IQ2_XXS sub-block 0[{j}]"
            );
        }
        // Sub-blocks 1..=7: scale nibble = 0 → db = 0.125 → output = 1.0
        for j in 32..256 {
            assert_eq!(
                bf16_pair_to_f32(&out[j * 2..j * 2 + 2]),
                1.0,
                "IQ2_XXS sub-block >=1[{j}]"
            );
        }
    }

    // -----------------------------------------------------------------
    // IQ2_XS
    // -----------------------------------------------------------------

    #[test]
    fn iq2_xs_all_zero_block() {
        let block = vec![0u8; 74];
        let out = dequantize_gguf_to_bf16(&block, GgufType::IQ2_XS, 256).unwrap();
        for chunk in out.chunks_exact(2) {
            assert_eq!(bf16_pair_to_f32(chunk), 0.0);
        }
    }

    #[test]
    fn iq2_xs_scale_nibble_split() {
        // d = 1.0; scales[0] = 0x50 (low nibble = 0, high nibble = 5) →
        //   db0 = (0.5 + 0) × 0.25 = 0.125 for l ∈ {0, 1}
        //   db1 = (0.5 + 5) × 0.25 = 1.375 for l ∈ {2, 3}
        // qs words are all 0 → IQ2XS_GRID[0] = 0x0808_0808_0808_0808 → grid byte = 8.
        // Signs index = 0 → KSIGNS_IQ2XS[0] = 0 → all +1.
        // Expected:
        //   elements 0..16  (ib32=0, l∈{0,1}):   db0 × 8 = 1.0
        //   elements 16..32 (ib32=0, l∈{2,3}):   db1 × 8 = 11.0
        //   elements 32..256: scales[1..=7] = 0 → low & high nibbles both 0 → db = 0.125
        //                                                                  → output = 1.0
        let mut block = vec![0u8; 74];
        block[0..2].copy_from_slice(&f16_bytes(1.0));
        block[66] = 0x50; // scales[0]
        let out = dequantize_gguf_to_bf16(&block, GgufType::IQ2_XS, 256).unwrap();
        for j in 0..16 {
            assert_eq!(
                bf16_pair_to_f32(&out[j * 2..j * 2 + 2]),
                1.0,
                "IQ2_XS sub-block 0 low-nibble[{j}]"
            );
        }
        for j in 16..32 {
            assert_eq!(
                bf16_pair_to_f32(&out[j * 2..j * 2 + 2]),
                11.0,
                "IQ2_XS sub-block 0 high-nibble[{j}]"
            );
        }
        for j in 32..256 {
            assert_eq!(
                bf16_pair_to_f32(&out[j * 2..j * 2 + 2]),
                1.0,
                "IQ2_XS sub-block >=1[{j}]"
            );
        }
    }

    // -----------------------------------------------------------------
    // IQ2_S
    // -----------------------------------------------------------------

    #[test]
    fn iq2_s_all_zero_block() {
        let block = vec![0u8; 82];
        let out = dequantize_gguf_to_bf16(&block, GgufType::IQ2_S, 256).unwrap();
        for chunk in out.chunks_exact(2) {
            assert_eq!(bf16_pair_to_f32(chunk), 0.0);
        }
    }

    #[test]
    fn iq2_s_qh_high_bits_select_grid_entry_256() {
        // d = 1.0; scales[0] = 0x03 (low nibble = 3) → db0 = (0.5 + 3) × 0.25 = 0.875.
        // qh[0] bits [0..2] = 0b01 → for l=0, high = (0b01 << 8) & 0x300 = 0x100.
        // qs_indices[0] = 0 → grid index = 0 | 0x100 = 0x100 = 256.
        // qs_signs[0] = 0 → all +1.
        //
        // Expected: elements 0..8 use IQ2S_GRID[256] × 0.875.
        let mut block = vec![0u8; 82];
        block[0..2].copy_from_slice(&f16_bytes(1.0));
        block[74] = 0x03; // scales[0] low nibble = 3
        block[66] = 0x01; // qh[0] low 2 bits = 0b01 → for l=0, feeds bit 8 of the index
        let out = dequantize_gguf_to_bf16(&block, GgufType::IQ2_S, 256).unwrap();

        let grid = IQ2S_GRID[256].to_le_bytes();
        for j in 0..8 {
            let expected = f32::from(half::bf16::from_f32(0.875 * f32::from(grid[j])));
            assert_eq!(
                bf16_pair_to_f32(&out[j * 2..j * 2 + 2]),
                expected,
                "IQ2_S group 0[{j}]"
            );
        }
        // qh[0] bits [2..8] = 0 → for l ∈ {1,2,3}, high = 0 → grid index = 0.
        // IQ2S_GRID[0] byte = 8, scales nibble for l∈{1}: db0 (low nibble 3)
        // for l∈{2,3}: db1 (high nibble 0) = 0.125.
        for j in 8..16 {
            assert_eq!(
                bf16_pair_to_f32(&out[j * 2..j * 2 + 2]),
                7.0, // 0.875 × 8
                "IQ2_S group 1[{j}]"
            );
        }
        for j in 16..32 {
            assert_eq!(
                bf16_pair_to_f32(&out[j * 2..j * 2 + 2]),
                1.0, // 0.125 × 8
                "IQ2_S groups 2-3[{j}]"
            );
        }
        // Sub-blocks 1..=7: scales[1..=7] = 0 → db = 0.125, qh = 0, qs = 0 → 1.0.
        for j in 32..256 {
            assert_eq!(
                bf16_pair_to_f32(&out[j * 2..j * 2 + 2]),
                1.0,
                "IQ2_S sub-block >=1[{j}]"
            );
        }
    }

    // -----------------------------------------------------------------
    // IQ3_XXS
    // -----------------------------------------------------------------

    #[test]
    fn iq3_xxs_all_zero_block() {
        let block = vec![0u8; 98];
        let out = dequantize_gguf_to_bf16(&block, GgufType::IQ3_XXS, 256).unwrap();
        for chunk in out.chunks_exact(2) {
            assert_eq!(bf16_pair_to_f32(chunk), 0.0);
        }
    }

    #[test]
    fn iq3_xxs_grid_entry_0_sign_entry_0_scale_nibble_0() {
        // d = 1.0, all bytes zero → every aux32 word = 0
        //   → scale nibble = 0 → db = (0.5 + 0) × 0.5 = 0.25
        //   → grid entries both IQ3XXS_GRID[0] = 0x04040404 → bytes = 4
        //   → KSIGNS_IQ2XS[0] = 0 → all signs +1
        //   → output = 0.25 × 4 = 1.0 everywhere.
        let mut block = vec![0u8; 98];
        block[0..2].copy_from_slice(&f16_bytes(1.0));
        let out = dequantize_gguf_to_bf16(&block, GgufType::IQ3_XXS, 256).unwrap();
        for chunk in out.chunks_exact(2) {
            assert_eq!(bf16_pair_to_f32(chunk), 1.0);
        }
    }

    #[test]
    fn iq3_xxs_scale_nibble_sweep() {
        // d = 1.0, craft aux32 for sub-block 0 with top nibble = 0xF:
        //   → db = (0.5 + 15) × 0.5 = 7.75
        // Sign index = 0 → KSIGNS_IQ2XS[0] = 0 → all +1
        // Grid bytes = 4 → output = 7.75 × 4 = 31.0 in sub-block 0, 1.0 elsewhere.
        let mut block = vec![0u8; 98];
        block[0..2].copy_from_slice(&f16_bytes(1.0));
        // scales_and_signs for ib32=0 live at qs[64..68] = block[66..70].
        // We want top 4 bits of aux32 = 0xF → byte 3 = 0xF0.
        block[66 + 3] = 0xF0;
        let out = dequantize_gguf_to_bf16(&block, GgufType::IQ3_XXS, 256).unwrap();
        for j in 0..32 {
            assert_eq!(
                bf16_pair_to_f32(&out[j * 2..j * 2 + 2]),
                31.0,
                "IQ3_XXS sub-block 0[{j}]"
            );
        }
        for j in 32..256 {
            assert_eq!(
                bf16_pair_to_f32(&out[j * 2..j * 2 + 2]),
                1.0,
                "IQ3_XXS sub-block >=1[{j}]"
            );
        }
    }

    // -----------------------------------------------------------------
    // IQ3_S
    // -----------------------------------------------------------------

    #[test]
    fn iq3_s_all_zero_block() {
        let block = vec![0u8; 110];
        let out = dequantize_gguf_to_bf16(&block, GgufType::IQ3_S, 256).unwrap();
        for chunk in out.chunks_exact(2) {
            assert_eq!(bf16_pair_to_f32(chunk), 0.0);
        }
    }

    #[test]
    fn iq3_s_scale_formula() {
        // d = 1.0, scales[0] = 0x05 (low = 5, high = 0):
        //   db1 = 1 + 2·5 = 11 (sub-block A, elements 0..32)
        //   db2 = 1 + 2·0 = 1  (sub-block B, elements 32..64)
        // qs/qh/signs zero → IQ3S_GRID[0] = 0x01010101 → grid bytes = 1
        // → sub-block A outputs 11, sub-block B outputs 1.
        // Sub-blocks 2..=7: scales[1..=3] = 0 → db1 = db2 = 1 → outputs 1.
        let mut block = vec![0u8; 110];
        block[0..2].copy_from_slice(&f16_bytes(1.0));
        block[106] = 0x05; // scales[0]
        let out = dequantize_gguf_to_bf16(&block, GgufType::IQ3_S, 256).unwrap();
        for j in 0..32 {
            assert_eq!(
                bf16_pair_to_f32(&out[j * 2..j * 2 + 2]),
                11.0,
                "IQ3_S sub-block A[{j}]"
            );
        }
        for j in 32..256 {
            assert_eq!(
                bf16_pair_to_f32(&out[j * 2..j * 2 + 2]),
                1.0,
                "IQ3_S sub-block B+[{j}]"
            );
        }
    }

    #[test]
    fn iq3_s_qh_high_bit_selects_grid_entry_256() {
        // d = 1.0, scales[0] = 0x03 → db1 = 1 + 2·3 = 7.
        // qh[0] = 0x01: for l=0, hi1 = (1 << 8) & 0x100 = 0x100 → grid1 idx = 256.
        // For l=0, hi2 = (1 << 7) & 0x100 = 0x80 & 0x100 = 0 → grid2 idx = 0.
        // Signs zero → all +1.
        //
        // Expected elements 0..4: db1 × IQ3S_GRID[256] bytes  = 7 × grid_bytes.
        //          elements 4..8: db1 × IQ3S_GRID[0] bytes   = 7 × 1 = 7.
        //          elements 8..64 (rest of A + B): scales[0] low = 3 → db1 = 7
        //                                          scales[0] high = 0 → db2 = 1.
        //          elements 64.. : scales[1..=3] = 0 → db = 1 → output 1.
        let mut block = vec![0u8; 110];
        block[0..2].copy_from_slice(&f16_bytes(1.0));
        block[106] = 0x03; // scales[0] low nibble = 3
        block[66] = 0x01; // qh[0] low bit = 1
        let out = dequantize_gguf_to_bf16(&block, GgufType::IQ3_S, 256).unwrap();

        // First 4 outputs sourced from IQ3S_GRID[256] × db1 = 7.
        let grid256 = IQ3S_GRID[256].to_le_bytes();
        for j in 0..4 {
            let expected = f32::from(half::bf16::from_f32(7.0 * f32::from(grid256[j])));
            assert_eq!(
                bf16_pair_to_f32(&out[j * 2..j * 2 + 2]),
                expected,
                "IQ3_S grid256[{j}]"
            );
        }
        // Next 4 outputs sourced from IQ3S_GRID[0] = 0x01010101 (byte = 1) × db1 = 7.
        for j in 4..8 {
            assert_eq!(
                bf16_pair_to_f32(&out[j * 2..j * 2 + 2]),
                7.0,
                "IQ3_S group 1 low-half[{j}]"
            );
        }
        // Elements 8..32: remaining l=1..=3 of sub-block A, all grid[0]×db1 = 7.
        for j in 8..32 {
            assert_eq!(
                bf16_pair_to_f32(&out[j * 2..j * 2 + 2]),
                7.0,
                "IQ3_S sub-block A rest[{j}]"
            );
        }
        // Sub-block B (32..64): scales[0] high = 0 → db2 = 1, grid[0] = 1 → 1.
        // Sub-blocks 2..=7 (64..256): scales[1..=3] = 0 → db = 1, grid[0] = 1 → 1.
        for j in 32..256 {
            assert_eq!(
                bf16_pair_to_f32(&out[j * 2..j * 2 + 2]),
                1.0,
                "IQ3_S sub-block B+[{j}]"
            );
        }
    }

    // -----------------------------------------------------------------
    // IQ1_S
    // -----------------------------------------------------------------

    #[test]
    fn iq1_s_all_zero_block() {
        let block = vec![0u8; 50];
        let out = dequantize_gguf_to_bf16(&block, GgufType::IQ1_S, 256).unwrap();
        for chunk in out.chunks_exact(2) {
            assert_eq!(bf16_pair_to_f32(chunk), 0.0);
        }
    }

    #[test]
    fn iq1_s_grid_entry_0_top_qh_zero() {
        // d = 1.0, qs = 0, qh = 0 (all u16 words):
        //   dl    = 1.0 × (2·0 + 1) = 1.0
        //   delta = +IQ1S_DELTA = +0.125
        //   idx   = 0 → IQ1S_GRID[0] = 0xFFFFFFFFFFFFFFFF
        //         → all 8 grid bytes = 0xFF = -1 as i8
        // Output = 1.0 × (-1 + 0.125) = -0.875 everywhere.
        let mut block = vec![0u8; 50];
        block[0..2].copy_from_slice(&f16_bytes(1.0));
        let out = dequantize_gguf_to_bf16(&block, GgufType::IQ1_S, 256).unwrap();
        for chunk in out.chunks_exact(2) {
            assert_eq!(bf16_pair_to_f32(chunk), -0.875);
        }
    }

    #[test]
    fn iq1_s_delta_sign_flip() {
        // d = 1.0, qs = 0, qh[0] (first u16) = 0x8000:
        //   For ib32=0: top 3 bits of qh = 0b000 → dl = 1.0; bit 15 = 1 → delta = -0.125
        //              idx = 0 | ((0x8000 >> 3l) & 7) << 8 = 0 (since 0x8000 & 7 == 0
        //              and 0x8000 >> {3,6,9} all yield (... & 7) == 0)
        //              → grid byte = -1, output = 1.0 × (-1 + -0.125) = -1.125
        // ib32=1..=7: qh = 0 → delta = +0.125, output = -0.875
        let mut block = vec![0u8; 50];
        block[0..2].copy_from_slice(&f16_bytes(1.0));
        // qh starts at byte 34. qh[0] u16 = 0x8000, low byte first (LE).
        block[34] = 0x00;
        block[35] = 0x80;
        let out = dequantize_gguf_to_bf16(&block, GgufType::IQ1_S, 256).unwrap();
        for j in 0..32 {
            assert_eq!(
                bf16_pair_to_f32(&out[j * 2..j * 2 + 2]),
                -1.125,
                "IQ1_S sub-block 0 with -delta[{j}]"
            );
        }
        for j in 32..256 {
            assert_eq!(
                bf16_pair_to_f32(&out[j * 2..j * 2 + 2]),
                -0.875,
                "IQ1_S sub-blocks 1..=7 with +delta[{j}]"
            );
        }
    }

    // -----------------------------------------------------------------
    // IQ1_M
    // -----------------------------------------------------------------

    #[test]
    fn iq1_m_all_zero_block() {
        // All bytes 0 → scale_u16 = 0 → d = f16(0.0) = 0.0 → all outputs 0.
        let block = vec![0u8; 56];
        let out = dequantize_gguf_to_bf16(&block, GgufType::IQ1_M, 256).unwrap();
        for chunk in out.chunks_exact(2) {
            assert_eq!(bf16_pair_to_f32(chunk), 0.0);
        }
    }

    #[test]
    fn iq1_m_scale_assembly_yields_one() {
        // Assemble scale_u16 = 0x3C00 (f16 = 1.0):
        //   bits [3:0]   = (sc[0] >> 12) → set sc[0] top nibble = 0
        //   bits [7:4]   = (sc[1] >> 8) & 0x00F0 → top nibble of sc[1] = 0
        //   bits [11:8]  = (sc[2] >> 4) & 0x0F00 → top nibble of sc[2] = 0xC
        //   bits [15:12] = sc[3] & 0xF000 → top nibble of sc[3] = 0x3
        // sc[0..2] low 12 bits = 0 → all sub-block scales = 0 → all dl = 1.0.
        // qs/qh = 0 → idx = 0, grid byte = -1, delta = +0.125.
        // Output = 1.0 × (-1 + 0.125) = -0.875 everywhere.
        let mut block = vec![0u8; 56];
        // scales[4..=5] = sc[2] = 0xC000 (LE: low byte then high byte)
        block[48 + 4] = 0x00;
        block[48 + 5] = 0xC0;
        // scales[6..=7] = sc[3] = 0x3000
        block[48 + 6] = 0x00;
        block[48 + 7] = 0x30;
        let out = dequantize_gguf_to_bf16(&block, GgufType::IQ1_M, 256).unwrap();
        for chunk in out.chunks_exact(2) {
            assert_eq!(bf16_pair_to_f32(chunk), -0.875);
        }
    }

    #[test]
    fn iq1_m_dl1_vs_dl2_split() {
        // Same f16 scale = 1.0 setup as above, plus sc[0] low 6 bits set so
        // ib32=0 gets dl1 nibble = 0 (dl1 = 1.0) and dl2 nibble = 7 (dl2 = 15.0):
        //   sc[0] = 0b000111_000 = 0x0038 → ib32=0 dl1 = 1.0, dl2 = 15.0
        // ib32=1..=7: all sub-block scale nibbles = 0 → dl = 1.0 everywhere.
        // qs/qh = 0 → idx = 0, grid byte = -1, delta = +0.125.
        // Sub-block A of ib32=0 (elements 0..16): dl1 × (-1 + 0.125) = -0.875
        // Sub-block B of ib32=0 (elements 16..32): dl2 × (-1 + 0.125) = -13.125
        // ib32=1..=7 (elements 32..256): -0.875 everywhere.
        let mut block = vec![0u8; 56];
        block[48] = 0x38; // sc[0] low byte: bits [5:0] = 0b111000
        block[49] = 0x00; // sc[0] high byte
        // sc[2], sc[3] for f16 = 1.0:
        block[48 + 4] = 0x00;
        block[48 + 5] = 0xC0;
        block[48 + 6] = 0x00;
        block[48 + 7] = 0x30;
        let out = dequantize_gguf_to_bf16(&block, GgufType::IQ1_M, 256).unwrap();
        // Elements 0..16: dl1 = 1.0 → -0.875
        for j in 0..16 {
            assert_eq!(
                bf16_pair_to_f32(&out[j * 2..j * 2 + 2]),
                -0.875,
                "IQ1_M ib32=0 sub-block A[{j}]"
            );
        }
        // Elements 16..32: dl2 = 15.0 → 15 × -0.875 = -13.125
        for j in 16..32 {
            assert_eq!(
                bf16_pair_to_f32(&out[j * 2..j * 2 + 2]),
                -13.125,
                "IQ1_M ib32=0 sub-block B[{j}]"
            );
        }
        // Elements 32..256: dl = 1.0 → -0.875
        for j in 32..256 {
            assert_eq!(
                bf16_pair_to_f32(&out[j * 2..j * 2 + 2]),
                -0.875,
                "IQ1_M ib32>=1[{j}]"
            );
        }
    }

    // -----------------------------------------------------------------
    // TQ1_0
    // -----------------------------------------------------------------

    #[test]
    fn tq1_0_all_zero_block() {
        // d = 0.0 → all outputs 0.0 regardless of qs/qh contents.
        let block = vec![0u8; 54];
        let out = dequantize_gguf_to_bf16(&block, GgufType::TQ1_0, 256).unwrap();
        for chunk in out.chunks_exact(2) {
            assert_eq!(bf16_pair_to_f32(chunk), 0.0);
        }
    }

    #[test]
    fn tq1_0_zero_qs_qh_with_d_one() {
        // d = 1.0, all qs/qh = 0 → for every (n, m, j) the decode produces
        // q = 0 → xi = 0 → output = (0 - 1) × 1 = -1.0 at all 256 positions.
        // Sanity-checks that the chunked iteration covers every output slot.
        let mut block = vec![0u8; 54];
        block[52..54].copy_from_slice(&f16_bytes(1.0));
        let out = dequantize_gguf_to_bf16(&block, GgufType::TQ1_0, 256).unwrap();
        for chunk in out.chunks_exact(2) {
            assert_eq!(bf16_pair_to_f32(chunk), -1.0);
        }
    }

    // -----------------------------------------------------------------
    // TQ2_0
    // -----------------------------------------------------------------

    #[test]
    fn tq2_0_all_zero_block() {
        let block = vec![0u8; 66];
        let out = dequantize_gguf_to_bf16(&block, GgufType::TQ2_0, 256).unwrap();
        for chunk in out.chunks_exact(2) {
            assert_eq!(bf16_pair_to_f32(chunk), 0.0);
        }
    }

    #[test]
    fn tq2_0_zero_qs_with_d_one() {
        // d = 1.0, qs = 0 → q = 0 at every 2-bit position → output = -1.0.
        let mut block = vec![0u8; 66];
        block[64..66].copy_from_slice(&f16_bytes(1.0));
        let out = dequantize_gguf_to_bf16(&block, GgufType::TQ2_0, 256).unwrap();
        for chunk in out.chunks_exact(2) {
            assert_eq!(bf16_pair_to_f32(chunk), -1.0);
        }
    }

    #[test]
    fn tq2_0_first_byte_all_twos() {
        // d = 1.0, qs[0] = 0xAA = 0b10_10_10_10 → all four 2-bit ternaries = 2
        // → outputs (2 - 1) × 1 = +1.0 at the four positions where qs[0]'s
        // bit-position is read.
        //
        // Iteration order: chunk_off=0, l ∈ 0..4, m ∈ 0..32. Element
        // y_off = chunk × 128 + l × 32 + m. So qs[0]'s 4 ternary slots land
        // at y_off ∈ {0, 32, 64, 96} — outputs 1.0 there, -1.0 elsewhere
        // in the first chunk's 128 elements; second chunk (qs[32..64] = 0)
        // gives -1.0 throughout its 128 outputs.
        let mut block = vec![0u8; 66];
        block[0] = 0xAA;
        block[64..66].copy_from_slice(&f16_bytes(1.0));
        let out = dequantize_gguf_to_bf16(&block, GgufType::TQ2_0, 256).unwrap();
        for j in 0..256 {
            let expected = if matches!(j, 0 | 32 | 64 | 96) {
                1.0
            } else {
                -1.0
            };
            assert_eq!(
                bf16_pair_to_f32(&out[j * 2..j * 2 + 2]),
                expected,
                "TQ2_0 output[{j}]"
            );
        }
    }

    // -----------------------------------------------------------------
    // MXFP4
    // -----------------------------------------------------------------

    #[test]
    fn mxfp4_zero_block_zero_output() {
        // e = 0, qs all zero. Every nibble indexes K_VALUES_MXFP4[0] = 0,
        // so every output is 0.0 regardless of the e8m0 sub-2 branch.
        let block = vec![0u8; 17];
        let out = dequantize_gguf_to_bf16(&block, GgufType::MXFP4, 32).unwrap();
        for chunk in out.chunks_exact(2) {
            assert_eq!(bf16_pair_to_f32(chunk), 0.0);
        }
    }

    #[test]
    fn mxfp4_e_one_yields_subnormal_scale() {
        // e = 1 → d = 0x0040_0000 as f32 = 2^-127 (subnormal mantissa
        // 2^22 × 2^-126). qs[0] = 0x71:
        //   low nibble 1 → K_VALUES_MXFP4[1] = 1
        //   high nibble 7 → K_VALUES_MXFP4[7] = 12
        // Output position 0 = 1 × 2^-127, position 16 = 12 × 2^-127. The
        // expected BF16 is computed end-to-end via `f32_bits_to_bf16_bits`
        // so this test exercises the rounding rule alongside the kernel
        // (rather than baking a fragile constant into the assertion).
        let mut block = vec![0u8; 17];
        block[0] = 1;
        block[1] = 0x71;
        let out = dequantize_gguf_to_bf16(&block, GgufType::MXFP4, 32).unwrap();
        let d = f32::from_bits(0x0040_0000);
        let expected_pos_0 = 1.0_f32 * d;
        let expected_pos_16 = 12.0_f32 * d;
        let bits_pos_0 = f32_bits_to_bf16_bits(expected_pos_0.to_bits());
        let bits_pos_16 = f32_bits_to_bf16_bits(expected_pos_16.to_bits());
        let actual_pos_0 = u16::from_le_bytes([out[0], out[1]]);
        let actual_pos_16 = u16::from_le_bytes([out[32], out[33]]);
        assert_eq!(actual_pos_0, bits_pos_0, "MXFP4 e=1 position 0");
        assert_eq!(actual_pos_16, bits_pos_16, "MXFP4 e=1 position 16");
    }

    #[test]
    fn mxfp4_e_normal_simple() {
        // e = 128 → d = (128 - 1) << 23 = 0x3F80_0000 = 1.0. qs[0] = 0x71:
        //   low nibble 1 → 1.0; high nibble 7 → 12.0.
        // qs[1..16] = 0 → all four positions in the 32-element block decode
        // to 0.0 except positions 0 and 16.
        let mut block = vec![0u8; 17];
        block[0] = 128;
        block[1] = 0x71;
        let out = dequantize_gguf_to_bf16(&block, GgufType::MXFP4, 32).unwrap();
        for j in 0..32 {
            let expected = match j {
                0 => 1.0,
                16 => 12.0,
                _ => 0.0,
            };
            assert_eq!(
                bf16_pair_to_f32(&out[j * 2..j * 2 + 2]),
                expected,
                "MXFP4 e=128 output[{j}]"
            );
        }
    }

    #[test]
    fn mxfp4_e_normal_negative_codebook() {
        // e = 128 → d = 1.0. qs[0] = 0xF9:
        //   low nibble 9 → K_VALUES_MXFP4[9] = -1
        //   high nibble 15 → K_VALUES_MXFP4[15] = -12
        let mut block = vec![0u8; 17];
        block[0] = 128;
        block[1] = 0xF9;
        let out = dequantize_gguf_to_bf16(&block, GgufType::MXFP4, 32).unwrap();
        assert_eq!(bf16_pair_to_f32(&out[0..2]), -1.0);
        assert_eq!(bf16_pair_to_f32(&out[32..34]), -12.0);
    }
}

// ---------------------------------------------------------------------------
// IQ-family lattice codebook + sign tables
// ---------------------------------------------------------------------------

/// Private submodule housing the 2-bit `IQ*` lattice codebooks and the
/// shared `ksigns_iq2xs` / `kmask_iq2xs` sign tables, ported verbatim from
/// `ggml-common.h`.
///
/// Each `IQ2*_GRID` entry packs an 8-element `u8` codebook vector as a
/// little-endian `u64`; `.to_le_bytes()` at the use site recovers the 8
/// codebook values. Subsequent Phase 4.5 commits (`IQ3_*`, `IQ1_*`) will
/// deposit their own grids into this submodule.
#[allow(clippy::unreadable_literal, clippy::large_stack_arrays)]
mod iq_grids {
    /// Per-element sign-selection bit-mask (`j`-th bit).
    pub(super) const KMASK_IQ2XS: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

    /// Maps a 7-bit sign index to an 8-bit sign mask (low 7 bits = stored
    /// signs, high bit = parity so the popcount is always even).
    pub(super) const KSIGNS_IQ2XS: [u8; 128] = [
        0, 129, 130, 3, 132, 5, 6, 135, 136, 9, 10, 139, 12, 141, 142, 15, 144, 17, 18, 147, 20,
        149, 150, 23, 24, 153, 154, 27, 156, 29, 30, 159, 160, 33, 34, 163, 36, 165, 166, 39, 40,
        169, 170, 43, 172, 45, 46, 175, 48, 177, 178, 51, 180, 53, 54, 183, 184, 57, 58, 187, 60,
        189, 190, 63, 192, 65, 66, 195, 68, 197, 198, 71, 72, 201, 202, 75, 204, 77, 78, 207, 80,
        209, 210, 83, 212, 85, 86, 215, 216, 89, 90, 219, 92, 221, 222, 95, 96, 225, 226, 99, 228,
        101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111, 240, 113, 114, 243, 116, 245, 246,
        119, 120, 249, 250, 123, 252, 125, 126, 255,
    ];

    /// 256-entry lattice codebook for `IQ2_XXS` (8-bit index).
    pub(super) const IQ2XXS_GRID: [u64; 256] = [
        0x0808080808080808,
        0x080808080808082b,
        0x0808080808081919,
        0x0808080808082b08,
        0x0808080808082b2b,
        0x0808080808190819,
        0x0808080808191908,
        0x08080808082b0808,
        0x08080808082b082b,
        0x08080808082b2b08,
        0x08080808082b2b2b,
        0x0808080819080819,
        0x0808080819081908,
        0x0808080819190808,
        0x0808080819192b08,
        0x08080808192b0819,
        0x08080808192b1908,
        0x080808082b080808,
        0x080808082b08082b,
        0x080808082b082b2b,
        0x080808082b2b082b,
        0x0808081908080819,
        0x0808081908081908,
        0x0808081908190808,
        0x0808081908191919,
        0x0808081919080808,
        0x080808192b081908,
        0x080808192b192b08,
        0x0808082b08080808,
        0x0808082b0808082b,
        0x0808082b082b082b,
        0x0808082b2b08082b,
        0x0808190808080819,
        0x0808190808081908,
        0x0808190808190808,
        0x08081908082b0819,
        0x08081908082b1908,
        0x0808190819080808,
        0x080819081908082b,
        0x0808190819082b08,
        0x08081908192b0808,
        0x080819082b080819,
        0x080819082b081908,
        0x080819082b190808,
        0x080819082b2b1908,
        0x0808191908080808,
        0x080819190808082b,
        0x0808191908082b08,
        0x08081919082b0808,
        0x080819191908192b,
        0x08081919192b2b19,
        0x080819192b080808,
        0x080819192b190819,
        0x0808192b08082b19,
        0x0808192b08190808,
        0x0808192b19080808,
        0x0808192b2b081908,
        0x0808192b2b2b1908,
        0x08082b0808080808,
        0x08082b0808081919,
        0x08082b0808082b08,
        0x08082b0808191908,
        0x08082b08082b2b08,
        0x08082b0819080819,
        0x08082b0819081908,
        0x08082b0819190808,
        0x08082b081919082b,
        0x08082b082b082b08,
        0x08082b1908081908,
        0x08082b1919080808,
        0x08082b2b0808082b,
        0x08082b2b08191908,
        0x0819080808080819,
        0x0819080808081908,
        0x0819080808190808,
        0x08190808082b0819,
        0x0819080819080808,
        0x08190808192b0808,
        0x081908082b081908,
        0x081908082b190808,
        0x081908082b191919,
        0x0819081908080808,
        0x0819081908082b08,
        0x08190819082b0808,
        0x0819081919190808,
        0x0819081919192b2b,
        0x081908192b080808,
        0x0819082b082b1908,
        0x0819082b19081919,
        0x0819190808080808,
        0x0819190808082b08,
        0x08191908082b0808,
        0x08191908082b1919,
        0x0819190819082b19,
        0x081919082b080808,
        0x0819191908192b08,
        0x08191919192b082b,
        0x0819192b08080808,
        0x0819192b0819192b,
        0x08192b0808080819,
        0x08192b0808081908,
        0x08192b0808190808,
        0x08192b0819080808,
        0x08192b082b080819,
        0x08192b1908080808,
        0x08192b1908081919,
        0x08192b192b2b0808,
        0x08192b2b19190819,
        0x082b080808080808,
        0x082b08080808082b,
        0x082b080808082b2b,
        0x082b080819081908,
        0x082b0808192b0819,
        0x082b08082b080808,
        0x082b08082b08082b,
        0x082b0819082b2b19,
        0x082b081919082b08,
        0x082b082b08080808,
        0x082b082b0808082b,
        0x082b190808080819,
        0x082b190808081908,
        0x082b190808190808,
        0x082b190819080808,
        0x082b19081919192b,
        0x082b191908080808,
        0x082b191919080819,
        0x082b1919192b1908,
        0x082b192b2b190808,
        0x082b2b0808082b08,
        0x082b2b08082b0808,
        0x082b2b082b191908,
        0x082b2b2b19081908,
        0x1908080808080819,
        0x1908080808081908,
        0x1908080808190808,
        0x1908080808192b08,
        0x19080808082b0819,
        0x19080808082b1908,
        0x1908080819080808,
        0x1908080819082b08,
        0x190808081919192b,
        0x19080808192b0808,
        0x190808082b080819,
        0x190808082b081908,
        0x190808082b190808,
        0x1908081908080808,
        0x19080819082b0808,
        0x19080819192b0819,
        0x190808192b080808,
        0x190808192b081919,
        0x1908082b08080819,
        0x1908082b08190808,
        0x1908082b19082b08,
        0x1908082b1919192b,
        0x1908082b192b2b08,
        0x1908190808080808,
        0x1908190808082b08,
        0x19081908082b0808,
        0x190819082b080808,
        0x190819082b192b19,
        0x190819190819082b,
        0x19081919082b1908,
        0x1908192b08080808,
        0x19082b0808080819,
        0x19082b0808081908,
        0x19082b0808190808,
        0x19082b0819080808,
        0x19082b0819081919,
        0x19082b1908080808,
        0x19082b1919192b08,
        0x19082b19192b0819,
        0x19082b192b08082b,
        0x19082b2b19081919,
        0x19082b2b2b190808,
        0x1919080808080808,
        0x1919080808082b08,
        0x1919080808190819,
        0x1919080808192b19,
        0x19190808082b0808,
        0x191908082b080808,
        0x191908082b082b08,
        0x1919081908081908,
        0x191908191908082b,
        0x191908192b2b1908,
        0x1919082b2b190819,
        0x191919082b190808,
        0x191919082b19082b,
        0x1919191908082b2b,
        0x1919192b08080819,
        0x1919192b19191908,
        0x19192b0808080808,
        0x19192b0808190819,
        0x19192b0808192b19,
        0x19192b08192b1908,
        0x19192b1919080808,
        0x19192b2b08082b08,
        0x192b080808081908,
        0x192b080808190808,
        0x192b080819080808,
        0x192b0808192b2b08,
        0x192b081908080808,
        0x192b081919191919,
        0x192b082b08192b08,
        0x192b082b192b0808,
        0x192b190808080808,
        0x192b190808081919,
        0x192b191908190808,
        0x192b19190819082b,
        0x192b19192b081908,
        0x192b2b081908082b,
        0x2b08080808080808,
        0x2b0808080808082b,
        0x2b08080808082b2b,
        0x2b08080819080819,
        0x2b0808082b08082b,
        0x2b08081908081908,
        0x2b08081908192b08,
        0x2b08081919080808,
        0x2b08082b08190819,
        0x2b08190808080819,
        0x2b08190808081908,
        0x2b08190808190808,
        0x2b08190808191919,
        0x2b08190819080808,
        0x2b081908192b0808,
        0x2b08191908080808,
        0x2b0819191908192b,
        0x2b0819192b191908,
        0x2b08192b08082b19,
        0x2b08192b19080808,
        0x2b08192b192b0808,
        0x2b082b080808082b,
        0x2b082b1908081908,
        0x2b082b2b08190819,
        0x2b19080808081908,
        0x2b19080808190808,
        0x2b190808082b1908,
        0x2b19080819080808,
        0x2b1908082b2b0819,
        0x2b1908190819192b,
        0x2b1908192b080808,
        0x2b19082b19081919,
        0x2b19190808080808,
        0x2b191908082b082b,
        0x2b19190819081908,
        0x2b19191919190819,
        0x2b192b082b080819,
        0x2b192b19082b0808,
        0x2b2b08080808082b,
        0x2b2b080819190808,
        0x2b2b08082b081919,
        0x2b2b081908082b19,
        0x2b2b082b08080808,
        0x2b2b190808192b08,
        0x2b2b2b0819190808,
        0x2b2b2b1908081908,
    ];

    /// 512-entry lattice codebook for `IQ2_XS` (9-bit index).
    pub(super) const IQ2XS_GRID: [u64; 512] = [
        0x0808080808080808,
        0x080808080808082b,
        0x0808080808081919,
        0x0808080808082b08,
        0x0808080808082b2b,
        0x0808080808190819,
        0x0808080808191908,
        0x080808080819192b,
        0x0808080808192b19,
        0x08080808082b0808,
        0x08080808082b082b,
        0x08080808082b1919,
        0x08080808082b2b08,
        0x0808080819080819,
        0x0808080819081908,
        0x080808081908192b,
        0x0808080819082b19,
        0x0808080819190808,
        0x080808081919082b,
        0x0808080819191919,
        0x0808080819192b08,
        0x08080808192b0819,
        0x08080808192b1908,
        0x080808082b080808,
        0x080808082b08082b,
        0x080808082b081919,
        0x080808082b082b08,
        0x080808082b190819,
        0x080808082b191908,
        0x080808082b192b19,
        0x080808082b2b0808,
        0x0808081908080819,
        0x0808081908081908,
        0x080808190808192b,
        0x0808081908082b19,
        0x0808081908190808,
        0x080808190819082b,
        0x0808081908191919,
        0x0808081908192b08,
        0x0808081908192b2b,
        0x08080819082b0819,
        0x08080819082b1908,
        0x0808081919080808,
        0x080808191908082b,
        0x0808081919081919,
        0x0808081919082b08,
        0x0808081919190819,
        0x0808081919191908,
        0x08080819192b0808,
        0x08080819192b2b08,
        0x080808192b080819,
        0x080808192b081908,
        0x080808192b190808,
        0x0808082b08080808,
        0x0808082b0808082b,
        0x0808082b08081919,
        0x0808082b08082b08,
        0x0808082b08190819,
        0x0808082b08191908,
        0x0808082b082b0808,
        0x0808082b19080819,
        0x0808082b19081908,
        0x0808082b19190808,
        0x0808082b19191919,
        0x0808082b2b080808,
        0x0808082b2b082b2b,
        0x0808190808080819,
        0x0808190808081908,
        0x080819080808192b,
        0x0808190808082b19,
        0x0808190808190808,
        0x080819080819082b,
        0x0808190808191919,
        0x0808190808192b08,
        0x08081908082b0819,
        0x08081908082b1908,
        0x0808190819080808,
        0x080819081908082b,
        0x0808190819081919,
        0x0808190819082b08,
        0x0808190819190819,
        0x0808190819191908,
        0x080819081919192b,
        0x08081908192b0808,
        0x080819082b080819,
        0x080819082b081908,
        0x080819082b190808,
        0x0808191908080808,
        0x080819190808082b,
        0x0808191908081919,
        0x0808191908082b08,
        0x0808191908190819,
        0x0808191908191908,
        0x08081919082b0808,
        0x0808191919080819,
        0x0808191919081908,
        0x0808191919190808,
        0x08081919192b0819,
        0x080819192b080808,
        0x0808192b08080819,
        0x0808192b08081908,
        0x0808192b08190808,
        0x0808192b082b192b,
        0x0808192b19080808,
        0x0808192b1908082b,
        0x0808192b2b081908,
        0x08082b0808080808,
        0x08082b080808082b,
        0x08082b0808081919,
        0x08082b0808082b08,
        0x08082b0808082b2b,
        0x08082b0808190819,
        0x08082b0808191908,
        0x08082b08082b0808,
        0x08082b08082b1919,
        0x08082b0819080819,
        0x08082b0819081908,
        0x08082b0819190808,
        0x08082b0819192b08,
        0x08082b082b080808,
        0x08082b082b2b0808,
        0x08082b082b2b2b2b,
        0x08082b1908080819,
        0x08082b1908081908,
        0x08082b1908190808,
        0x08082b1919080808,
        0x08082b192b080819,
        0x08082b192b082b19,
        0x08082b2b08080808,
        0x08082b2b082b0808,
        0x08082b2b082b2b08,
        0x08082b2b2b19192b,
        0x08082b2b2b2b0808,
        0x0819080808080819,
        0x0819080808081908,
        0x081908080808192b,
        0x0819080808082b19,
        0x0819080808190808,
        0x081908080819082b,
        0x0819080808191919,
        0x0819080808192b08,
        0x08190808082b0819,
        0x08190808082b1908,
        0x0819080819080808,
        0x081908081908082b,
        0x0819080819081919,
        0x0819080819082b08,
        0x0819080819190819,
        0x0819080819191908,
        0x08190808192b0808,
        0x08190808192b2b2b,
        0x081908082b080819,
        0x081908082b081908,
        0x081908082b190808,
        0x0819081908080808,
        0x081908190808082b,
        0x0819081908081919,
        0x0819081908082b08,
        0x0819081908190819,
        0x0819081908191908,
        0x08190819082b0808,
        0x0819081919080819,
        0x0819081919081908,
        0x0819081919190808,
        0x081908192b080808,
        0x081908192b191908,
        0x081908192b19192b,
        0x0819082b08080819,
        0x0819082b08081908,
        0x0819082b0808192b,
        0x0819082b08190808,
        0x0819082b19080808,
        0x0819082b192b0808,
        0x0819190808080808,
        0x081919080808082b,
        0x0819190808081919,
        0x0819190808082b08,
        0x0819190808190819,
        0x0819190808191908,
        0x08191908082b0808,
        0x0819190819080819,
        0x0819190819081908,
        0x0819190819082b19,
        0x0819190819190808,
        0x08191908192b1908,
        0x081919082b080808,
        0x0819191908080819,
        0x0819191908081908,
        0x0819191908190808,
        0x0819191919080808,
        0x0819192b08080808,
        0x0819192b08191908,
        0x0819192b19082b19,
        0x08192b0808080819,
        0x08192b0808081908,
        0x08192b0808190808,
        0x08192b080819082b,
        0x08192b0819080808,
        0x08192b0819191908,
        0x08192b082b08192b,
        0x08192b1908080808,
        0x08192b1908081919,
        0x08192b19192b192b,
        0x08192b2b19190819,
        0x08192b2b2b2b2b19,
        0x082b080808080808,
        0x082b08080808082b,
        0x082b080808081919,
        0x082b080808082b08,
        0x082b080808082b2b,
        0x082b080808190819,
        0x082b080808191908,
        0x082b0808082b0808,
        0x082b080819080819,
        0x082b080819081908,
        0x082b080819190808,
        0x082b08082b080808,
        0x082b08082b2b0808,
        0x082b081908080819,
        0x082b081908081908,
        0x082b081908190808,
        0x082b081919080808,
        0x082b081919082b08,
        0x082b0819192b1919,
        0x082b082b08080808,
        0x082b082b082b082b,
        0x082b082b2b080808,
        0x082b082b2b2b2b08,
        0x082b190808080819,
        0x082b190808081908,
        0x082b190808190808,
        0x082b1908082b2b19,
        0x082b190819080808,
        0x082b191908080808,
        0x082b191919080819,
        0x082b19191919082b,
        0x082b19192b192b19,
        0x082b192b08080819,
        0x082b192b08192b2b,
        0x082b192b2b2b192b,
        0x082b2b0808080808,
        0x082b2b0808082b08,
        0x082b2b0808082b2b,
        0x082b2b08082b0808,
        0x082b2b0819191919,
        0x082b2b082b082b08,
        0x082b2b082b2b082b,
        0x082b2b19192b2b08,
        0x082b2b192b190808,
        0x082b2b2b08082b08,
        0x082b2b2b082b0808,
        0x082b2b2b2b08082b,
        0x082b2b2b2b082b08,
        0x082b2b2b2b082b2b,
        0x1908080808080819,
        0x1908080808081908,
        0x190808080808192b,
        0x1908080808082b19,
        0x1908080808190808,
        0x190808080819082b,
        0x1908080808191919,
        0x1908080808192b08,
        0x19080808082b0819,
        0x19080808082b1908,
        0x1908080819080808,
        0x190808081908082b,
        0x1908080819081919,
        0x1908080819082b08,
        0x1908080819082b2b,
        0x1908080819190819,
        0x1908080819191908,
        0x19080808192b0808,
        0x19080808192b1919,
        0x190808082b080819,
        0x190808082b081908,
        0x190808082b190808,
        0x1908081908080808,
        0x190808190808082b,
        0x1908081908081919,
        0x1908081908082b08,
        0x1908081908190819,
        0x1908081908191908,
        0x19080819082b0808,
        0x1908081919080819,
        0x1908081919081908,
        0x1908081919190808,
        0x190808192b080808,
        0x190808192b081919,
        0x190808192b2b082b,
        0x1908082b08080819,
        0x1908082b08081908,
        0x1908082b08190808,
        0x1908082b0819082b,
        0x1908082b082b2b19,
        0x1908082b19080808,
        0x1908190808080808,
        0x190819080808082b,
        0x1908190808081919,
        0x1908190808082b08,
        0x1908190808190819,
        0x1908190808191908,
        0x1908190808192b19,
        0x19081908082b0808,
        0x1908190819080819,
        0x1908190819081908,
        0x1908190819190808,
        0x190819082b080808,
        0x190819082b191908,
        0x1908191908080819,
        0x1908191908081908,
        0x1908191908190808,
        0x19081919082b1908,
        0x1908191919080808,
        0x190819192b192b2b,
        0x1908192b08080808,
        0x1908192b08082b2b,
        0x1908192b19081908,
        0x1908192b19190808,
        0x19082b0808080819,
        0x19082b0808081908,
        0x19082b0808190808,
        0x19082b0819080808,
        0x19082b0819081919,
        0x19082b0819191908,
        0x19082b08192b082b,
        0x19082b1908080808,
        0x19082b1908190819,
        0x19082b1919081908,
        0x19082b1919190808,
        0x19082b19192b2b19,
        0x19082b2b08081908,
        0x1919080808080808,
        0x191908080808082b,
        0x1919080808081919,
        0x1919080808082b08,
        0x1919080808190819,
        0x1919080808191908,
        0x19190808082b0808,
        0x19190808082b2b08,
        0x1919080819080819,
        0x1919080819081908,
        0x1919080819190808,
        0x191908082b080808,
        0x1919081908080819,
        0x1919081908081908,
        0x1919081908190808,
        0x1919081908191919,
        0x1919081919080808,
        0x191908191908082b,
        0x1919082b08080808,
        0x1919082b19081908,
        0x1919082b2b2b2b2b,
        0x1919190808080819,
        0x1919190808081908,
        0x1919190808190808,
        0x19191908082b0819,
        0x1919190819080808,
        0x19191908192b0808,
        0x191919082b080819,
        0x191919082b2b0819,
        0x1919191908080808,
        0x1919191908082b08,
        0x191919192b080808,
        0x191919192b082b08,
        0x1919192b082b0819,
        0x1919192b192b2b08,
        0x1919192b2b2b0819,
        0x19192b0808080808,
        0x19192b0808191908,
        0x19192b0819080819,
        0x19192b0819190808,
        0x19192b082b192b19,
        0x19192b1908192b2b,
        0x19192b1919080808,
        0x19192b191908082b,
        0x19192b2b2b081919,
        0x192b080808080819,
        0x192b080808081908,
        0x192b080808190808,
        0x192b080819080808,
        0x192b080819191908,
        0x192b0808192b082b,
        0x192b08082b08192b,
        0x192b08082b2b2b19,
        0x192b081908080808,
        0x192b082b082b1908,
        0x192b082b19082b2b,
        0x192b082b2b19082b,
        0x192b190808080808,
        0x192b19080819192b,
        0x192b191908190808,
        0x192b191919080808,
        0x192b191919081919,
        0x192b19192b2b1908,
        0x192b2b0808080819,
        0x192b2b08192b2b2b,
        0x192b2b19082b1919,
        0x192b2b2b0808192b,
        0x192b2b2b19191908,
        0x192b2b2b192b082b,
        0x2b08080808080808,
        0x2b0808080808082b,
        0x2b08080808081919,
        0x2b08080808082b08,
        0x2b08080808190819,
        0x2b08080808191908,
        0x2b080808082b0808,
        0x2b080808082b2b2b,
        0x2b08080819080819,
        0x2b08080819081908,
        0x2b08080819190808,
        0x2b0808082b080808,
        0x2b0808082b08082b,
        0x2b0808082b2b2b08,
        0x2b0808082b2b2b2b,
        0x2b08081908080819,
        0x2b08081908081908,
        0x2b0808190808192b,
        0x2b08081908190808,
        0x2b08081919080808,
        0x2b08081919190819,
        0x2b08081919192b19,
        0x2b08082b08080808,
        0x2b08082b082b0808,
        0x2b08082b2b080808,
        0x2b08082b2b08082b,
        0x2b08082b2b2b0808,
        0x2b08082b2b2b2b08,
        0x2b08190808080819,
        0x2b08190808081908,
        0x2b08190808190808,
        0x2b0819080819082b,
        0x2b08190808191919,
        0x2b08190819080808,
        0x2b081908192b0808,
        0x2b0819082b082b19,
        0x2b08191908080808,
        0x2b08191919081908,
        0x2b0819192b2b1919,
        0x2b08192b08192b08,
        0x2b08192b192b2b2b,
        0x2b082b0808080808,
        0x2b082b0808082b08,
        0x2b082b08082b1919,
        0x2b082b0819192b2b,
        0x2b082b082b080808,
        0x2b082b082b08082b,
        0x2b082b082b2b2b08,
        0x2b082b190808192b,
        0x2b082b2b082b082b,
        0x2b082b2b2b080808,
        0x2b082b2b2b082b08,
        0x2b082b2b2b19192b,
        0x2b082b2b2b2b2b08,
        0x2b19080808080819,
        0x2b19080808081908,
        0x2b19080808190808,
        0x2b19080819080808,
        0x2b1908081919192b,
        0x2b1908082b081908,
        0x2b19081908080808,
        0x2b190819082b082b,
        0x2b190819192b1908,
        0x2b19082b1919192b,
        0x2b19082b2b082b19,
        0x2b19190808080808,
        0x2b19190808081919,
        0x2b19190819081908,
        0x2b19190819190808,
        0x2b19190819192b08,
        0x2b191919082b2b19,
        0x2b1919192b190808,
        0x2b1919192b19082b,
        0x2b19192b19080819,
        0x2b192b0819190819,
        0x2b192b082b2b192b,
        0x2b192b1919082b19,
        0x2b192b2b08191919,
        0x2b192b2b192b0808,
        0x2b2b080808080808,
        0x2b2b08080808082b,
        0x2b2b080808082b08,
        0x2b2b080808082b2b,
        0x2b2b0808082b0808,
        0x2b2b0808082b2b2b,
        0x2b2b08082b2b0808,
        0x2b2b081919190819,
        0x2b2b081919192b19,
        0x2b2b08192b2b192b,
        0x2b2b082b08080808,
        0x2b2b082b0808082b,
        0x2b2b082b08082b08,
        0x2b2b082b082b2b2b,
        0x2b2b082b2b080808,
        0x2b2b082b2b2b0808,
        0x2b2b190819080808,
        0x2b2b19082b191919,
        0x2b2b192b192b1919,
        0x2b2b192b2b192b08,
        0x2b2b2b0808082b2b,
        0x2b2b2b08082b0808,
        0x2b2b2b08082b082b,
        0x2b2b2b08082b2b08,
        0x2b2b2b082b2b0808,
        0x2b2b2b082b2b2b08,
        0x2b2b2b1908081908,
        0x2b2b2b192b081908,
        0x2b2b2b192b08192b,
        0x2b2b2b2b082b2b08,
        0x2b2b2b2b082b2b2b,
        0x2b2b2b2b2b190819,
        0x2b2b2b2b2b2b2b2b,
    ];

    /// 1024-entry lattice codebook for `IQ2_S` (10-bit index).
    pub(super) const IQ2S_GRID: [u64; 1024] = [
        0x0808080808080808,
        0x080808080808082b,
        0x0808080808081919,
        0x0808080808082b08,
        0x0808080808082b2b,
        0x0808080808190819,
        0x0808080808191908,
        0x080808080819192b,
        0x0808080808192b19,
        0x08080808082b0808,
        0x08080808082b082b,
        0x08080808082b1919,
        0x08080808082b2b08,
        0x0808080819080819,
        0x0808080819081908,
        0x080808081908192b,
        0x0808080819082b19,
        0x0808080819190808,
        0x080808081919082b,
        0x0808080819191919,
        0x0808080819192b08,
        0x08080808192b0819,
        0x08080808192b1908,
        0x08080808192b192b,
        0x08080808192b2b19,
        0x080808082b080808,
        0x080808082b08082b,
        0x080808082b081919,
        0x080808082b082b08,
        0x080808082b190819,
        0x080808082b191908,
        0x080808082b2b0808,
        0x080808082b2b1919,
        0x080808082b2b2b2b,
        0x0808081908080819,
        0x0808081908081908,
        0x080808190808192b,
        0x0808081908082b19,
        0x0808081908190808,
        0x080808190819082b,
        0x0808081908191919,
        0x0808081908192b08,
        0x08080819082b0819,
        0x08080819082b1908,
        0x0808081919080808,
        0x080808191908082b,
        0x0808081919081919,
        0x0808081919082b08,
        0x0808081919190819,
        0x0808081919191908,
        0x080808191919192b,
        0x0808081919192b19,
        0x08080819192b0808,
        0x08080819192b1919,
        0x08080819192b2b08,
        0x080808192b080819,
        0x080808192b081908,
        0x080808192b190808,
        0x080808192b19082b,
        0x080808192b191919,
        0x080808192b2b0819,
        0x080808192b2b1908,
        0x0808082b08080808,
        0x0808082b0808082b,
        0x0808082b08081919,
        0x0808082b08082b08,
        0x0808082b08190819,
        0x0808082b08191908,
        0x0808082b082b0808,
        0x0808082b082b2b2b,
        0x0808082b19080819,
        0x0808082b19081908,
        0x0808082b1908192b,
        0x0808082b19082b19,
        0x0808082b19190808,
        0x0808082b19191919,
        0x0808082b2b080808,
        0x0808082b2b081919,
        0x0808082b2b082b2b,
        0x0808082b2b191908,
        0x0808082b2b2b082b,
        0x0808190808080819,
        0x0808190808081908,
        0x080819080808192b,
        0x0808190808082b19,
        0x0808190808190808,
        0x080819080819082b,
        0x0808190808191919,
        0x0808190808192b08,
        0x08081908082b0819,
        0x08081908082b1908,
        0x08081908082b192b,
        0x08081908082b2b19,
        0x0808190819080808,
        0x080819081908082b,
        0x0808190819081919,
        0x0808190819082b08,
        0x0808190819082b2b,
        0x0808190819190819,
        0x0808190819191908,
        0x080819081919192b,
        0x0808190819192b19,
        0x08081908192b0808,
        0x08081908192b082b,
        0x08081908192b1919,
        0x080819082b080819,
        0x080819082b081908,
        0x080819082b08192b,
        0x080819082b082b19,
        0x080819082b190808,
        0x080819082b191919,
        0x080819082b192b08,
        0x080819082b2b0819,
        0x080819082b2b1908,
        0x0808191908080808,
        0x080819190808082b,
        0x0808191908081919,
        0x0808191908082b08,
        0x0808191908082b2b,
        0x0808191908190819,
        0x0808191908191908,
        0x080819190819192b,
        0x0808191908192b19,
        0x08081919082b0808,
        0x08081919082b1919,
        0x08081919082b2b08,
        0x0808191919080819,
        0x0808191919081908,
        0x080819191908192b,
        0x0808191919082b19,
        0x0808191919190808,
        0x080819191919082b,
        0x0808191919191919,
        0x0808191919192b08,
        0x08081919192b0819,
        0x08081919192b1908,
        0x080819192b080808,
        0x080819192b08082b,
        0x080819192b081919,
        0x080819192b082b08,
        0x080819192b190819,
        0x080819192b191908,
        0x080819192b2b0808,
        0x0808192b08080819,
        0x0808192b08081908,
        0x0808192b0808192b,
        0x0808192b08082b19,
        0x0808192b08190808,
        0x0808192b08191919,
        0x0808192b19080808,
        0x0808192b19081919,
        0x0808192b19082b08,
        0x0808192b19190819,
        0x0808192b19191908,
        0x0808192b192b0808,
        0x0808192b2b080819,
        0x0808192b2b081908,
        0x0808192b2b190808,
        0x08082b0808080808,
        0x08082b080808082b,
        0x08082b0808081919,
        0x08082b0808082b08,
        0x08082b0808190819,
        0x08082b0808191908,
        0x08082b080819192b,
        0x08082b0808192b19,
        0x08082b08082b0808,
        0x08082b08082b1919,
        0x08082b08082b2b2b,
        0x08082b0819080819,
        0x08082b0819081908,
        0x08082b081908192b,
        0x08082b0819082b19,
        0x08082b0819190808,
        0x08082b081919082b,
        0x08082b0819191919,
        0x08082b0819192b08,
        0x08082b08192b0819,
        0x08082b08192b1908,
        0x08082b082b080808,
        0x08082b082b081919,
        0x08082b082b191908,
        0x08082b082b2b2b2b,
        0x08082b1908080819,
        0x08082b1908081908,
        0x08082b1908190808,
        0x08082b190819082b,
        0x08082b1908191919,
        0x08082b1908192b08,
        0x08082b19082b0819,
        0x08082b1919080808,
        0x08082b1919081919,
        0x08082b1919082b08,
        0x08082b1919190819,
        0x08082b1919191908,
        0x08082b19192b0808,
        0x08082b192b080819,
        0x08082b192b190808,
        0x08082b2b08080808,
        0x08082b2b08190819,
        0x08082b2b08191908,
        0x08082b2b082b082b,
        0x08082b2b082b2b08,
        0x08082b2b082b2b2b,
        0x08082b2b19190808,
        0x08082b2b2b192b19,
        0x0819080808080819,
        0x0819080808081908,
        0x081908080808192b,
        0x0819080808082b19,
        0x0819080808190808,
        0x081908080819082b,
        0x0819080808191919,
        0x0819080808192b08,
        0x08190808082b0819,
        0x08190808082b1908,
        0x08190808082b192b,
        0x0819080819080808,
        0x081908081908082b,
        0x0819080819081919,
        0x0819080819082b08,
        0x0819080819190819,
        0x0819080819191908,
        0x081908081919192b,
        0x0819080819192b19,
        0x08190808192b0808,
        0x08190808192b082b,
        0x08190808192b1919,
        0x08190808192b2b08,
        0x081908082b080819,
        0x081908082b081908,
        0x081908082b08192b,
        0x081908082b190808,
        0x081908082b191919,
        0x081908082b192b08,
        0x081908082b2b0819,
        0x081908082b2b1908,
        0x0819081908080808,
        0x081908190808082b,
        0x0819081908081919,
        0x0819081908082b08,
        0x0819081908082b2b,
        0x0819081908190819,
        0x0819081908191908,
        0x081908190819192b,
        0x0819081908192b19,
        0x08190819082b0808,
        0x08190819082b082b,
        0x08190819082b1919,
        0x08190819082b2b08,
        0x0819081919080819,
        0x0819081919081908,
        0x081908191908192b,
        0x0819081919082b19,
        0x0819081919190808,
        0x081908191919082b,
        0x0819081919191919,
        0x0819081919192b08,
        0x08190819192b0819,
        0x08190819192b1908,
        0x081908192b080808,
        0x081908192b08082b,
        0x081908192b081919,
        0x081908192b082b08,
        0x081908192b190819,
        0x081908192b191908,
        0x0819082b08080819,
        0x0819082b08081908,
        0x0819082b08082b19,
        0x0819082b08190808,
        0x0819082b08191919,
        0x0819082b082b0819,
        0x0819082b082b1908,
        0x0819082b19080808,
        0x0819082b19081919,
        0x0819082b19190819,
        0x0819082b19191908,
        0x0819082b2b080819,
        0x0819082b2b081908,
        0x0819082b2b190808,
        0x0819190808080808,
        0x081919080808082b,
        0x0819190808081919,
        0x0819190808082b08,
        0x0819190808190819,
        0x0819190808191908,
        0x081919080819192b,
        0x0819190808192b19,
        0x08191908082b0808,
        0x08191908082b1919,
        0x08191908082b2b08,
        0x0819190819080819,
        0x0819190819081908,
        0x081919081908192b,
        0x0819190819082b19,
        0x0819190819190808,
        0x081919081919082b,
        0x0819190819191919,
        0x0819190819192b08,
        0x08191908192b0819,
        0x08191908192b1908,
        0x081919082b080808,
        0x081919082b08082b,
        0x081919082b081919,
        0x081919082b082b08,
        0x081919082b190819,
        0x081919082b191908,
        0x081919082b2b0808,
        0x0819191908080819,
        0x0819191908081908,
        0x081919190808192b,
        0x0819191908082b19,
        0x0819191908190808,
        0x081919190819082b,
        0x0819191908191919,
        0x0819191908192b08,
        0x08191919082b0819,
        0x08191919082b1908,
        0x0819191919080808,
        0x081919191908082b,
        0x0819191919081919,
        0x0819191919082b08,
        0x0819191919190819,
        0x0819191919191908,
        0x08191919192b0808,
        0x081919192b080819,
        0x081919192b081908,
        0x081919192b190808,
        0x0819192b08080808,
        0x0819192b08081919,
        0x0819192b08082b08,
        0x0819192b08190819,
        0x0819192b08191908,
        0x0819192b082b0808,
        0x0819192b19080819,
        0x0819192b19081908,
        0x0819192b19190808,
        0x0819192b2b080808,
        0x0819192b2b2b2b2b,
        0x08192b0808080819,
        0x08192b0808081908,
        0x08192b080808192b,
        0x08192b0808082b19,
        0x08192b0808190808,
        0x08192b0808191919,
        0x08192b0808192b08,
        0x08192b08082b0819,
        0x08192b0819080808,
        0x08192b081908082b,
        0x08192b0819081919,
        0x08192b0819082b08,
        0x08192b0819190819,
        0x08192b0819191908,
        0x08192b08192b0808,
        0x08192b082b080819,
        0x08192b082b081908,
        0x08192b1908080808,
        0x08192b190808082b,
        0x08192b1908081919,
        0x08192b1908082b08,
        0x08192b1908190819,
        0x08192b1908191908,
        0x08192b19082b0808,
        0x08192b1919080819,
        0x08192b1919081908,
        0x08192b1919190808,
        0x08192b19192b2b19,
        0x08192b192b2b082b,
        0x08192b2b08081908,
        0x08192b2b08190808,
        0x08192b2b19080808,
        0x08192b2b1919192b,
        0x082b080808080808,
        0x082b08080808082b,
        0x082b080808081919,
        0x082b080808082b08,
        0x082b080808190819,
        0x082b080808191908,
        0x082b08080819192b,
        0x082b080808192b19,
        0x082b0808082b0808,
        0x082b0808082b1919,
        0x082b0808082b2b2b,
        0x082b080819080819,
        0x082b080819081908,
        0x082b080819190808,
        0x082b08081919082b,
        0x082b080819191919,
        0x082b0808192b1908,
        0x082b08082b080808,
        0x082b08082b082b2b,
        0x082b08082b191908,
        0x082b08082b2b2b2b,
        0x082b081908080819,
        0x082b081908081908,
        0x082b081908190808,
        0x082b08190819082b,
        0x082b081908191919,
        0x082b0819082b0819,
        0x082b081919080808,
        0x082b08191908082b,
        0x082b081919081919,
        0x082b081919190819,
        0x082b081919191908,
        0x082b0819192b0808,
        0x082b08192b080819,
        0x082b08192b081908,
        0x082b08192b190808,
        0x082b082b08080808,
        0x082b082b08082b2b,
        0x082b082b082b082b,
        0x082b082b082b2b08,
        0x082b082b082b2b2b,
        0x082b082b19081908,
        0x082b082b19190808,
        0x082b082b2b082b08,
        0x082b082b2b082b2b,
        0x082b082b2b2b2b08,
        0x082b190808080819,
        0x082b190808081908,
        0x082b19080808192b,
        0x082b190808082b19,
        0x082b190808190808,
        0x082b190808191919,
        0x082b190808192b08,
        0x082b1908082b0819,
        0x082b1908082b1908,
        0x082b190819080808,
        0x082b19081908082b,
        0x082b190819081919,
        0x082b190819082b08,
        0x082b190819190819,
        0x082b190819191908,
        0x082b1908192b0808,
        0x082b19082b080819,
        0x082b19082b081908,
        0x082b19082b190808,
        0x082b191908080808,
        0x082b191908081919,
        0x082b191908082b08,
        0x082b191908190819,
        0x082b191908191908,
        0x082b1919082b0808,
        0x082b191919080819,
        0x082b191919081908,
        0x082b191919190808,
        0x082b1919192b192b,
        0x082b19192b080808,
        0x082b192b08080819,
        0x082b192b08081908,
        0x082b192b08190808,
        0x082b192b19080808,
        0x082b192b19192b19,
        0x082b2b0808080808,
        0x082b2b0808081919,
        0x082b2b0808190819,
        0x082b2b0808191908,
        0x082b2b0819080819,
        0x082b2b0819081908,
        0x082b2b0819190808,
        0x082b2b082b082b2b,
        0x082b2b082b2b2b2b,
        0x082b2b1908080819,
        0x082b2b1908081908,
        0x082b2b1908190808,
        0x082b2b192b191919,
        0x082b2b2b08082b2b,
        0x082b2b2b082b082b,
        0x082b2b2b192b1908,
        0x082b2b2b2b082b08,
        0x082b2b2b2b082b2b,
        0x1908080808080819,
        0x1908080808081908,
        0x190808080808192b,
        0x1908080808082b19,
        0x1908080808190808,
        0x190808080819082b,
        0x1908080808191919,
        0x1908080808192b08,
        0x1908080808192b2b,
        0x19080808082b0819,
        0x19080808082b1908,
        0x19080808082b192b,
        0x1908080819080808,
        0x190808081908082b,
        0x1908080819081919,
        0x1908080819082b08,
        0x1908080819082b2b,
        0x1908080819190819,
        0x1908080819191908,
        0x190808081919192b,
        0x1908080819192b19,
        0x19080808192b0808,
        0x19080808192b082b,
        0x19080808192b1919,
        0x190808082b080819,
        0x190808082b081908,
        0x190808082b190808,
        0x190808082b191919,
        0x190808082b192b08,
        0x190808082b2b0819,
        0x190808082b2b1908,
        0x1908081908080808,
        0x190808190808082b,
        0x1908081908081919,
        0x1908081908082b08,
        0x1908081908190819,
        0x1908081908191908,
        0x190808190819192b,
        0x1908081908192b19,
        0x19080819082b0808,
        0x19080819082b082b,
        0x19080819082b1919,
        0x1908081919080819,
        0x1908081919081908,
        0x190808191908192b,
        0x1908081919082b19,
        0x1908081919190808,
        0x190808191919082b,
        0x1908081919191919,
        0x1908081919192b08,
        0x19080819192b0819,
        0x19080819192b1908,
        0x190808192b080808,
        0x190808192b08082b,
        0x190808192b081919,
        0x190808192b082b08,
        0x190808192b190819,
        0x190808192b191908,
        0x190808192b2b0808,
        0x1908082b08080819,
        0x1908082b08081908,
        0x1908082b08190808,
        0x1908082b0819082b,
        0x1908082b08191919,
        0x1908082b08192b08,
        0x1908082b082b1908,
        0x1908082b19080808,
        0x1908082b19081919,
        0x1908082b19082b08,
        0x1908082b19190819,
        0x1908082b19191908,
        0x1908082b192b0808,
        0x1908082b2b080819,
        0x1908082b2b081908,
        0x1908190808080808,
        0x190819080808082b,
        0x1908190808081919,
        0x1908190808082b08,
        0x1908190808082b2b,
        0x1908190808190819,
        0x1908190808191908,
        0x190819080819192b,
        0x1908190808192b19,
        0x19081908082b0808,
        0x19081908082b082b,
        0x19081908082b1919,
        0x19081908082b2b08,
        0x1908190819080819,
        0x1908190819081908,
        0x190819081908192b,
        0x1908190819082b19,
        0x1908190819190808,
        0x190819081919082b,
        0x1908190819191919,
        0x1908190819192b08,
        0x19081908192b0819,
        0x19081908192b1908,
        0x190819082b080808,
        0x190819082b08082b,
        0x190819082b081919,
        0x190819082b082b08,
        0x190819082b190819,
        0x190819082b191908,
        0x190819082b2b0808,
        0x1908191908080819,
        0x1908191908081908,
        0x190819190808192b,
        0x1908191908082b19,
        0x1908191908190808,
        0x190819190819082b,
        0x1908191908191919,
        0x1908191908192b08,
        0x19081919082b0819,
        0x19081919082b1908,
        0x1908191919080808,
        0x190819191908082b,
        0x1908191919081919,
        0x1908191919082b08,
        0x1908191919190819,
        0x1908191919191908,
        0x19081919192b0808,
        0x19081919192b2b2b,
        0x190819192b080819,
        0x190819192b081908,
        0x190819192b190808,
        0x1908192b08080808,
        0x1908192b0808082b,
        0x1908192b08081919,
        0x1908192b08082b08,
        0x1908192b08190819,
        0x1908192b08191908,
        0x1908192b082b0808,
        0x1908192b19080819,
        0x1908192b19081908,
        0x1908192b19190808,
        0x1908192b2b080808,
        0x1908192b2b2b1919,
        0x19082b0808080819,
        0x19082b0808081908,
        0x19082b0808082b19,
        0x19082b0808190808,
        0x19082b080819082b,
        0x19082b0808191919,
        0x19082b0808192b08,
        0x19082b08082b0819,
        0x19082b08082b1908,
        0x19082b0819080808,
        0x19082b081908082b,
        0x19082b0819081919,
        0x19082b0819082b08,
        0x19082b0819190819,
        0x19082b0819191908,
        0x19082b08192b0808,
        0x19082b082b081908,
        0x19082b082b190808,
        0x19082b1908080808,
        0x19082b190808082b,
        0x19082b1908081919,
        0x19082b1908082b08,
        0x19082b1908190819,
        0x19082b1908191908,
        0x19082b19082b0808,
        0x19082b1919080819,
        0x19082b1919081908,
        0x19082b1919190808,
        0x19082b192b080808,
        0x19082b192b19192b,
        0x19082b2b08080819,
        0x19082b2b08081908,
        0x19082b2b08190808,
        0x19082b2b19080808,
        0x1919080808080808,
        0x191908080808082b,
        0x1919080808081919,
        0x1919080808082b08,
        0x1919080808190819,
        0x1919080808191908,
        0x191908080819192b,
        0x1919080808192b19,
        0x19190808082b0808,
        0x19190808082b082b,
        0x19190808082b1919,
        0x19190808082b2b08,
        0x1919080819080819,
        0x1919080819081908,
        0x191908081908192b,
        0x1919080819082b19,
        0x1919080819190808,
        0x191908081919082b,
        0x1919080819191919,
        0x1919080819192b08,
        0x19190808192b0819,
        0x19190808192b1908,
        0x191908082b080808,
        0x191908082b08082b,
        0x191908082b081919,
        0x191908082b082b08,
        0x191908082b190819,
        0x191908082b191908,
        0x1919081908080819,
        0x1919081908081908,
        0x191908190808192b,
        0x1919081908082b19,
        0x1919081908190808,
        0x191908190819082b,
        0x1919081908191919,
        0x1919081908192b08,
        0x19190819082b0819,
        0x19190819082b1908,
        0x1919081919080808,
        0x191908191908082b,
        0x1919081919081919,
        0x1919081919082b08,
        0x1919081919190819,
        0x1919081919191908,
        0x19190819192b0808,
        0x191908192b080819,
        0x191908192b081908,
        0x191908192b190808,
        0x1919082b08080808,
        0x1919082b08081919,
        0x1919082b08082b08,
        0x1919082b08190819,
        0x1919082b08191908,
        0x1919082b082b0808,
        0x1919082b19080819,
        0x1919082b19081908,
        0x1919082b19190808,
        0x1919082b192b2b19,
        0x1919082b2b080808,
        0x1919190808080819,
        0x1919190808081908,
        0x191919080808192b,
        0x1919190808082b19,
        0x1919190808190808,
        0x191919080819082b,
        0x1919190808191919,
        0x1919190808192b08,
        0x19191908082b0819,
        0x19191908082b1908,
        0x1919190819080808,
        0x191919081908082b,
        0x1919190819081919,
        0x1919190819082b08,
        0x1919190819190819,
        0x1919190819191908,
        0x19191908192b0808,
        0x191919082b080819,
        0x191919082b081908,
        0x191919082b190808,
        0x1919191908080808,
        0x191919190808082b,
        0x1919191908081919,
        0x1919191908082b08,
        0x1919191908190819,
        0x1919191908191908,
        0x19191919082b0808,
        0x1919191919080819,
        0x1919191919081908,
        0x1919191919190808,
        0x191919192b080808,
        0x1919192b08080819,
        0x1919192b08081908,
        0x1919192b08190808,
        0x1919192b082b192b,
        0x1919192b19080808,
        0x19192b0808080808,
        0x19192b080808082b,
        0x19192b0808081919,
        0x19192b0808082b08,
        0x19192b0808190819,
        0x19192b0808191908,
        0x19192b08082b0808,
        0x19192b0819080819,
        0x19192b0819081908,
        0x19192b0819190808,
        0x19192b0819192b2b,
        0x19192b082b080808,
        0x19192b1908080819,
        0x19192b1908081908,
        0x19192b1908190808,
        0x19192b1919080808,
        0x19192b2b08080808,
        0x19192b2b08192b19,
        0x19192b2b2b081919,
        0x19192b2b2b2b2b08,
        0x192b080808080819,
        0x192b080808081908,
        0x192b08080808192b,
        0x192b080808190808,
        0x192b08080819082b,
        0x192b080808191919,
        0x192b080808192b08,
        0x192b0808082b0819,
        0x192b0808082b1908,
        0x192b080819080808,
        0x192b080819081919,
        0x192b080819082b08,
        0x192b080819190819,
        0x192b080819191908,
        0x192b0808192b0808,
        0x192b08082b081908,
        0x192b08082b190808,
        0x192b081908080808,
        0x192b08190808082b,
        0x192b081908081919,
        0x192b081908082b08,
        0x192b081908190819,
        0x192b081908191908,
        0x192b0819082b0808,
        0x192b081919080819,
        0x192b081919081908,
        0x192b081919190808,
        0x192b08192b080808,
        0x192b08192b192b19,
        0x192b082b08081908,
        0x192b082b08190808,
        0x192b082b19080808,
        0x192b082b1919192b,
        0x192b082b2b2b0819,
        0x192b190808080808,
        0x192b190808081919,
        0x192b190808082b08,
        0x192b190808190819,
        0x192b190808191908,
        0x192b1908082b0808,
        0x192b190819080819,
        0x192b190819081908,
        0x192b190819190808,
        0x192b19082b080808,
        0x192b191908080819,
        0x192b191908081908,
        0x192b191908190808,
        0x192b191919080808,
        0x192b191919082b2b,
        0x192b1919192b2b08,
        0x192b19192b19082b,
        0x192b192b08080808,
        0x192b192b2b191908,
        0x192b2b0808080819,
        0x192b2b0808081908,
        0x192b2b0808190808,
        0x192b2b08192b1919,
        0x192b2b082b192b08,
        0x192b2b1908080808,
        0x192b2b19082b2b2b,
        0x192b2b2b1908082b,
        0x192b2b2b2b2b0819,
        0x2b08080808080808,
        0x2b0808080808082b,
        0x2b08080808081919,
        0x2b08080808082b08,
        0x2b08080808190819,
        0x2b08080808191908,
        0x2b08080808192b19,
        0x2b080808082b0808,
        0x2b080808082b1919,
        0x2b08080819080819,
        0x2b08080819081908,
        0x2b08080819190808,
        0x2b0808081919082b,
        0x2b08080819191919,
        0x2b08080819192b08,
        0x2b080808192b0819,
        0x2b0808082b080808,
        0x2b0808082b081919,
        0x2b0808082b190819,
        0x2b0808082b191908,
        0x2b08081908080819,
        0x2b08081908081908,
        0x2b08081908082b19,
        0x2b08081908190808,
        0x2b0808190819082b,
        0x2b08081908191919,
        0x2b08081908192b08,
        0x2b080819082b0819,
        0x2b080819082b1908,
        0x2b08081919080808,
        0x2b0808191908082b,
        0x2b08081919081919,
        0x2b08081919082b08,
        0x2b08081919190819,
        0x2b08081919191908,
        0x2b0808192b080819,
        0x2b0808192b081908,
        0x2b0808192b190808,
        0x2b0808192b2b2b19,
        0x2b08082b08080808,
        0x2b08082b08081919,
        0x2b08082b08082b2b,
        0x2b08082b08190819,
        0x2b08082b08191908,
        0x2b08082b19080819,
        0x2b08082b19081908,
        0x2b08082b19190808,
        0x2b08190808080819,
        0x2b08190808081908,
        0x2b0819080808192b,
        0x2b08190808082b19,
        0x2b08190808190808,
        0x2b0819080819082b,
        0x2b08190808191919,
        0x2b08190808192b08,
        0x2b081908082b0819,
        0x2b08190819080808,
        0x2b0819081908082b,
        0x2b08190819081919,
        0x2b08190819082b08,
        0x2b08190819190819,
        0x2b08190819191908,
        0x2b081908192b0808,
        0x2b0819082b080819,
        0x2b0819082b081908,
        0x2b0819082b190808,
        0x2b08191908080808,
        0x2b0819190808082b,
        0x2b08191908081919,
        0x2b08191908082b08,
        0x2b08191908190819,
        0x2b08191908191908,
        0x2b081919082b0808,
        0x2b08191919080819,
        0x2b08191919081908,
        0x2b08191919190808,
        0x2b0819192b080808,
        0x2b0819192b082b2b,
        0x2b08192b08080819,
        0x2b08192b08081908,
        0x2b08192b08190808,
        0x2b08192b082b2b19,
        0x2b08192b19080808,
        0x2b082b0808080808,
        0x2b082b0808081919,
        0x2b082b0808190819,
        0x2b082b0808191908,
        0x2b082b0819080819,
        0x2b082b0819081908,
        0x2b082b0819190808,
        0x2b082b082b2b082b,
        0x2b082b1908080819,
        0x2b082b1908081908,
        0x2b082b1919080808,
        0x2b082b19192b1919,
        0x2b082b2b082b082b,
        0x2b082b2b19192b08,
        0x2b082b2b19192b2b,
        0x2b082b2b2b08082b,
        0x2b082b2b2b2b082b,
        0x2b19080808080819,
        0x2b19080808081908,
        0x2b19080808082b19,
        0x2b19080808190808,
        0x2b1908080819082b,
        0x2b19080808191919,
        0x2b19080808192b08,
        0x2b190808082b1908,
        0x2b19080819080808,
        0x2b1908081908082b,
        0x2b19080819081919,
        0x2b19080819082b08,
        0x2b19080819190819,
        0x2b19080819191908,
        0x2b190808192b0808,
        0x2b1908082b080819,
        0x2b1908082b081908,
        0x2b1908082b190808,
        0x2b19081908080808,
        0x2b19081908081919,
        0x2b19081908190819,
        0x2b19081908191908,
        0x2b19081919080819,
        0x2b19081919081908,
        0x2b19081919190808,
        0x2b19081919192b2b,
        0x2b19082b08080819,
        0x2b19082b08081908,
        0x2b19082b08190808,
        0x2b19082b19080808,
        0x2b19082b2b2b192b,
        0x2b19190808080808,
        0x2b1919080808082b,
        0x2b19190808081919,
        0x2b19190808082b08,
        0x2b19190808190819,
        0x2b19190808191908,
        0x2b191908082b0808,
        0x2b19190819080819,
        0x2b19190819081908,
        0x2b19190819190808,
        0x2b1919082b080808,
        0x2b1919082b19192b,
        0x2b19191908080819,
        0x2b19191908081908,
        0x2b19191908190808,
        0x2b19191919080808,
        0x2b1919192b192b08,
        0x2b1919192b2b0819,
        0x2b19192b08080808,
        0x2b19192b1908192b,
        0x2b19192b192b1908,
        0x2b192b0808080819,
        0x2b192b0808081908,
        0x2b192b0808190808,
        0x2b192b08082b192b,
        0x2b192b0819080808,
        0x2b192b082b2b2b19,
        0x2b192b1908080808,
        0x2b192b1919082b19,
        0x2b192b191919082b,
        0x2b192b2b2b190808,
        0x2b2b080808080808,
        0x2b2b080808081919,
        0x2b2b080808082b2b,
        0x2b2b080808191908,
        0x2b2b0808082b082b,
        0x2b2b0808082b2b2b,
        0x2b2b080819080819,
        0x2b2b080819081908,
        0x2b2b080819190808,
        0x2b2b08082b2b082b,
        0x2b2b08082b2b2b2b,
        0x2b2b081919080808,
        0x2b2b0819192b1919,
        0x2b2b082b0808082b,
        0x2b2b082b08082b2b,
        0x2b2b082b082b082b,
        0x2b2b082b082b2b08,
        0x2b2b082b082b2b2b,
        0x2b2b082b2b08082b,
        0x2b2b082b2b082b08,
        0x2b2b082b2b082b2b,
        0x2b2b082b2b2b2b08,
        0x2b2b190808080819,
        0x2b2b190808081908,
        0x2b2b190808190808,
        0x2b2b190819080808,
        0x2b2b19082b082b19,
        0x2b2b19082b2b1908,
        0x2b2b191908080808,
        0x2b2b191908192b19,
        0x2b2b192b19190819,
        0x2b2b2b0808082b2b,
        0x2b2b2b08082b2b08,
        0x2b2b2b082b2b082b,
        0x2b2b2b1919191908,
        0x2b2b2b192b08192b,
        0x2b2b2b2b08082b08,
        0x2b2b2b2b08082b2b,
        0x2b2b2b2b082b0808,
        0x2b2b2b2b082b082b,
        0x2b2b2b2b082b2b08,
        0x2b2b2b2b2b082b08,
        0x2b2b2b2b2b2b2b2b,
    ];

    /// 256-entry lattice codebook for `IQ3_XXS` (8-bit index). Each entry
    /// packs 4 unsigned codebook values as a little-endian `u32`.
    pub(super) const IQ3XXS_GRID: [u32; 256] = [
        0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404,
        0x04041414, 0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c,
        0x040c0c04, 0x040c0c14, 0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c,
        0x040c2c24, 0x040c3e04, 0x04140404, 0x04140414, 0x04140424, 0x04140c0c, 0x04141404,
        0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e, 0x04142c0c, 0x04142c3e, 0x04143e2c,
        0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c, 0x041c3e04, 0x04240c1c,
        0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c, 0x042c043e,
        0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
        0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c,
        0x0c04141c, 0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404,
        0x0c0c0414, 0x0c0c0c0c, 0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04,
        0x0c140c14, 0x0c14140c, 0x0c141c04, 0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404,
        0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c, 0x0c24042c, 0x0c242c04, 0x0c2c1404,
        0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414, 0x0c3e2404, 0x14040404,
        0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434, 0x14041c0c,
        0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
        0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c,
        0x14140c3e, 0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c,
        0x141c0c04, 0x141c0c24, 0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c,
        0x142c143e, 0x142c240c, 0x142c3e24, 0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c,
        0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c, 0x1c04141c, 0x1c042c04, 0x1c04342c,
        0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c, 0x1c0c2424, 0x1c0c2434,
        0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14, 0x1c1c0c0c,
        0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
        0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424,
        0x24040c3e, 0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404,
        0x24141c3e, 0x24142404, 0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424,
        0x24242c0c, 0x24243424, 0x242c142c, 0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04,
        0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c, 0x2c043e04, 0x2c0c0404, 0x2c0c0434,
        0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14, 0x2c1c0414, 0x2c1c2c1c,
        0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c, 0x2c342c04,
        0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
        0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c,
        0x342c2c14, 0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e,
        0x3e040c04, 0x3e041c14, 0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c,
        0x3e142c14, 0x3e1c0404, 0x3e1c0c2c, 0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c,
        0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
    ];

    /// 512-entry lattice codebook for `IQ3_S` (9-bit index).
    pub(super) const IQ3S_GRID: [u32; 512] = [
        0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303,
        0x01010305, 0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707,
        0x01010901, 0x01010905, 0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01,
        0x01010d05, 0x01010f03, 0x01010f09, 0x01010f0f, 0x01030101, 0x01030103, 0x01030105,
        0x01030109, 0x01030301, 0x01030303, 0x0103030b, 0x01030501, 0x01030507, 0x0103050f,
        0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b, 0x01030f05, 0x01050101,
        0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d, 0x01050503,
        0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
        0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501,
        0x01070505, 0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05,
        0x01070d0f, 0x01070f03, 0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503,
        0x01090509, 0x01090705, 0x01090901, 0x01090907, 0x01090b03, 0x01090f01, 0x010b0105,
        0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d, 0x010b0707, 0x010b0903, 0x010b090b,
        0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303, 0x010d0307, 0x010d0703,
        0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501, 0x010f0505,
        0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
        0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501,
        0x03010505, 0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03,
        0x03010f05, 0x03030101, 0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309,
        0x03030503, 0x03030701, 0x03030707, 0x03030903, 0x03030b01, 0x03030b05, 0x03030f01,
        0x03030f0d, 0x03050101, 0x03050305, 0x0305030b, 0x0305030f, 0x03050501, 0x03050509,
        0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01, 0x03050f05, 0x03070103,
        0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f, 0x03070701,
        0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
        0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09,
        0x030b0103, 0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03,
        0x030d0501, 0x030d0509, 0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107,
        0x030f0301, 0x030f0305, 0x030f0503, 0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01,
        0x05010101, 0x05010103, 0x05010107, 0x0501010b, 0x0501010f, 0x05010301, 0x05010305,
        0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f, 0x05010701, 0x05010705,
        0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f, 0x05010f01,
        0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
        0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103,
        0x05050109, 0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903,
        0x05050b07, 0x05050b0f, 0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b,
        0x05070303, 0x05070505, 0x05070509, 0x05070703, 0x05070707, 0x05070905, 0x05070b01,
        0x05070d0d, 0x05090103, 0x0509010f, 0x05090501, 0x05090507, 0x05090705, 0x0509070b,
        0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303, 0x050b0505, 0x050b070f,
        0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f, 0x050d0503,
        0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
        0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505,
        0x07010703, 0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03,
        0x07010d07, 0x07010f03, 0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503,
        0x07030507, 0x07030901, 0x07030d01, 0x07030f05, 0x07030f0d, 0x07050101, 0x07050305,
        0x07050501, 0x07050705, 0x07050709, 0x07050b01, 0x07070103, 0x07070301, 0x07070309,
        0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903, 0x07070907, 0x0707090f,
        0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505, 0x07090703,
        0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
        0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103,
        0x070f0107, 0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305,
        0x09010501, 0x09010509, 0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01,
        0x09030105, 0x0903010f, 0x09030303, 0x09030307, 0x09030505, 0x09030701, 0x0903070b,
        0x09030907, 0x09030b03, 0x09030b0b, 0x09050103, 0x09050107, 0x09050301, 0x0905030b,
        0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05, 0x09050f01, 0x09070109,
        0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b, 0x09090101,
        0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
        0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b,
        0x090f0701, 0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505,
        0x0b010901, 0x0b010909, 0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103,
        0x0b030107, 0x0b03010b, 0x0b030305, 0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101,
        0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d, 0x0b050b07, 0x0b070105, 0x0b07010f,
        0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b, 0x0b070f07, 0x0b090103,
        0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d, 0x0b0b0b03,
        0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
        0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f,
        0x0d030d09, 0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01,
        0x0d070101, 0x0d070309, 0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05,
        0x0d0b0101, 0x0d0b0107, 0x0d0b0709, 0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303,
        0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f, 0x0f010501, 0x0f010505, 0x0f01070d,
        0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303, 0x0f030509, 0x0f030907,
        0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503, 0x0f050701,
        0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
        0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703,
        0x0f0f0101,
    ];

    /// Bias added to every dequantised codebook value in `IQ1_S` and
    /// `IQ1_M`. The sign of the bias is selected per group by a single
    /// `qh` bit; magnitude is fixed at `0.125`.
    pub(super) const IQ1S_DELTA: f32 = 0.125;

    /// 2048-entry lattice codebook for `IQ1_S` and `IQ1_M` (11-bit index).
    /// Each entry packs **8 signed `i8` codebook values** as a little-endian
    /// `u64`; reinterpret the bytes as `i8` at the use site (the C reference
    /// casts the `u64` array pointer to `int8_t *`).
    pub(super) const IQ1S_GRID: [u64; 2048] = [
        0xffffffffffffffff,
        0xffffffffffffff01,
        0xffffffffffff0000,
        0xffffffffffff01ff,
        0xffffffffffff0101,
        0xffffffffff00ff00,
        0xffffffffff000000,
        0xffffffffff01ffff,
        0xffffffffff01ff01,
        0xffffffffff0101ff,
        0xffffffffff010101,
        0xffffffff00ff0000,
        0xffffffff0000ff00,
        0xffffffff000000ff,
        0xffffffff00000001,
        0xffffffff00010000,
        0xffffffff01ffffff,
        0xffffffff01ffff01,
        0xffffffff01ff01ff,
        0xffffffff01ff0101,
        0xffffffff01000000,
        0xffffffff0101ffff,
        0xffffffff0101ff01,
        0xffffffff010101ff,
        0xffffffff01010101,
        0xffffff00ffff00ff,
        0xffffff00ffff0000,
        0xffffff00ff00ff00,
        0xffffff00ff0000ff,
        0xffffff00ff000001,
        0xffffff00ff000100,
        0xffffff00ff000101,
        0xffffff00ff010000,
        0xffffff0000ffff00,
        0xffffff0000ff0001,
        0xffffff0000ff0100,
        0xffffff000000ff01,
        0xffffff0000000000,
        0xffffff0000000101,
        0xffffff000001ff00,
        0xffffff00000100ff,
        0xffffff0000010001,
        0xffffff00000101ff,
        0xffffff0001ff0000,
        0xffffff000100ff00,
        0xffffff00010000ff,
        0xffffff0001000001,
        0xffffff0001010000,
        0xffffff01ffffffff,
        0xffffff01ffffff01,
        0xffffff01ffff01ff,
        0xffffff01ffff0101,
        0xffffff01ff000000,
        0xffffff01ff01ffff,
        0xffffff01ff01ff01,
        0xffffff01ff0101ff,
        0xffffff01ff010101,
        0xffffff0100ff0000,
        0xffffff010000ff00,
        0xffffff0100000100,
        0xffffff01000100ff,
        0xffffff0100010100,
        0xffffff0101ffffff,
        0xffffff0101ffff01,
        0xffffff0101ff01ff,
        0xffffff0101ff0101,
        0xffffff010100ff00,
        0xffffff0101000000,
        0xffffff0101000100,
        0xffffff010101ffff,
        0xffffff010101ff01,
        0xffffff01010101ff,
        0xffffff0101010101,
        0xffff00ffff00ff00,
        0xffff00ffff0000ff,
        0xffff00ffff000001,
        0xffff00ffff010000,
        0xffff00ff00ffff00,
        0xffff00ff00ff0100,
        0xffff00ff00000000,
        0xffff00ff00000101,
        0xffff00ff000100ff,
        0xffff00ff00010000,
        0xffff00ff0100ff00,
        0xffff00ff01000100,
        0xffff00ff01010000,
        0xffff0000ffffff00,
        0xffff0000ffff00ff,
        0xffff0000ffff0000,
        0xffff0000ffff0001,
        0xffff0000ff000000,
        0xffff0000ff0001ff,
        0xffff0000ff000101,
        0xffff0000ff010100,
        0xffff000000ffffff,
        0xffff000000ff0000,
        0xffff000000ff0101,
        0xffff00000000ffff,
        0xffff00000000ff00,
        0xffff0000000000ff,
        0xffff000000000000,
        0xffff000000000001,
        0xffff000000000100,
        0xffff00000001ffff,
        0xffff00000001ff01,
        0xffff000000010000,
        0xffff0000000101ff,
        0xffff000000010101,
        0xffff000001ffff00,
        0xffff00000100ff00,
        0xffff000001000000,
        0xffff0000010001ff,
        0xffff000001000101,
        0xffff00000101ff00,
        0xffff0000010100ff,
        0xffff000001010000,
        0xffff000001010001,
        0xffff000001010100,
        0xffff0001ff0000ff,
        0xffff0001ff000100,
        0xffff000100ffff00,
        0xffff000100ff00ff,
        0xffff00010000ffff,
        0xffff00010000ff01,
        0xffff000100000000,
        0xffff0001000001ff,
        0xffff00010001ffff,
        0xffff00010001ff00,
        0xffff000100010001,
        0xffff000100010100,
        0xffff000101ff0000,
        0xffff00010100ff00,
        0xffff0001010000ff,
        0xffff000101000100,
        0xffff01ffffffffff,
        0xffff01ffffffff01,
        0xffff01ffffff01ff,
        0xffff01ffffff0101,
        0xffff01ffff000000,
        0xffff01ffff01ffff,
        0xffff01ffff01ff01,
        0xffff01ffff0101ff,
        0xffff01ffff010101,
        0xffff01ff00ff0000,
        0xffff01ff0000ff00,
        0xffff01ff00000001,
        0xffff01ff00010000,
        0xffff01ff01ffffff,
        0xffff01ff01ffff01,
        0xffff01ff01ff01ff,
        0xffff01ff01ff0101,
        0xffff01ff01000000,
        0xffff01ff0101ffff,
        0xffff01ff0101ff01,
        0xffff01ff010101ff,
        0xffff01ff01010101,
        0xffff0100ffff0000,
        0xffff0100ff00ff00,
        0xffff0100ff0000ff,
        0xffff0100ff000100,
        0xffff0100ff0100ff,
        0xffff0100ff010000,
        0xffff010000ffff00,
        0xffff01000000ffff,
        0xffff01000000ff00,
        0xffff010000000000,
        0xffff01000001ff00,
        0xffff0100000100ff,
        0xffff010000010100,
        0xffff01000100ff00,
        0xffff0100010000ff,
        0xffff010001000001,
        0xffff010001000100,
        0xffff010001010000,
        0xffff0101ffffffff,
        0xffff0101ffffff01,
        0xffff0101ffff01ff,
        0xffff0101ffff0101,
        0xffff0101ff000000,
        0xffff0101ff01ffff,
        0xffff0101ff01ff01,
        0xffff0101ff0101ff,
        0xffff0101ff010101,
        0xffff010100ff0000,
        0xffff01010000ff00,
        0xffff010100000100,
        0xffff01010001ff00,
        0xffff010100010000,
        0xffff010101ffffff,
        0xffff010101ffff01,
        0xffff010101ff0000,
        0xffff010101ff01ff,
        0xffff010101ff0101,
        0xffff010101000000,
        0xffff01010101ffff,
        0xffff01010101ff01,
        0xffff0101010101ff,
        0xffff010101010101,
        0xff00ffffff00ffff,
        0xff00ffffff00ff00,
        0xff00ffffff0000ff,
        0xff00ffffff000100,
        0xff00ffffff0100ff,
        0xff00ffffff010000,
        0xff00ffff00ffff00,
        0xff00ffff00ff00ff,
        0xff00ffff0000ffff,
        0xff00ffff00000000,
        0xff00ffff000001ff,
        0xff00ffff0001ff00,
        0xff00ffff000100ff,
        0xff00ffff00010000,
        0xff00ffff00010100,
        0xff00ffff0100ff00,
        0xff00ffff010000ff,
        0xff00ffff01000001,
        0xff00ffff0101ff00,
        0xff00ffff01010000,
        0xff00ff00ffffff00,
        0xff00ff00ffff00ff,
        0xff00ff00ffff0001,
        0xff00ff00ffff0100,
        0xff00ff00ff00ffff,
        0xff00ff00ff00ff01,
        0xff00ff00ff000000,
        0xff00ff00ff0001ff,
        0xff00ff00ff01ff00,
        0xff00ff00ff0100ff,
        0xff00ff00ff010100,
        0xff00ff0000ff0000,
        0xff00ff0000ff0101,
        0xff00ff000000ffff,
        0xff00ff000000ff00,
        0xff00ff000000ff01,
        0xff00ff00000000ff,
        0xff00ff0000000000,
        0xff00ff0000000001,
        0xff00ff0000000100,
        0xff00ff000001ffff,
        0xff00ff0000010000,
        0xff00ff0001ff00ff,
        0xff00ff000100ff01,
        0xff00ff0001000000,
        0xff00ff000101ff00,
        0xff00ff00010100ff,
        0xff00ff01ff00ff00,
        0xff00ff01ff0000ff,
        0xff00ff01ff000001,
        0xff00ff01ff010000,
        0xff00ff0100ffffff,
        0xff00ff0100ff0001,
        0xff00ff0100ff0100,
        0xff00ff010000ff01,
        0xff00ff0100000000,
        0xff00ff01000001ff,
        0xff00ff0100000101,
        0xff00ff01000100ff,
        0xff00ff0100010001,
        0xff00ff0101ff0000,
        0xff00ff010100ff00,
        0xff00ff01010000ff,
        0xff00ff0101000001,
        0xff00ff0101010000,
        0xff0000ffffffff00,
        0xff0000ffffff0001,
        0xff0000ffffff0100,
        0xff0000ffff0000ff,
        0xff0000ffff000000,
        0xff0000ffff0001ff,
        0xff0000ffff000100,
        0xff0000ffff01ff00,
        0xff0000ffff010001,
        0xff0000ff00ffff00,
        0xff0000ff00ff0000,
        0xff0000ff00ff0001,
        0xff0000ff00ff01ff,
        0xff0000ff00ff0101,
        0xff0000ff0000ff00,
        0xff0000ff000000ff,
        0xff0000ff00000000,
        0xff0000ff00000001,
        0xff0000ff00000100,
        0xff0000ff0001ff01,
        0xff0000ff00010000,
        0xff0000ff000101ff,
        0xff0000ff01ff00ff,
        0xff0000ff01ff0100,
        0xff0000ff0100ffff,
        0xff0000ff010000ff,
        0xff0000ff01000000,
        0xff0000ff010001ff,
        0xff0000ff01000100,
        0xff0000ff01000101,
        0xff0000ff0101ff00,
        0xff0000ff010100ff,
        0xff0000ff01010000,
        0xff0000ff01010100,
        0xff000000ffffff01,
        0xff000000ffff0000,
        0xff000000ffff0101,
        0xff000000ff00ff00,
        0xff000000ff0000ff,
        0xff000000ff000000,
        0xff000000ff000001,
        0xff000000ff000100,
        0xff000000ff01ffff,
        0xff000000ff01ff01,
        0xff000000ff010000,
        0xff000000ff0101ff,
        0xff000000ff010101,
        0xff00000000ffff00,
        0xff00000000ff00ff,
        0xff00000000ff0000,
        0xff00000000ff0001,
        0xff0000000000ff00,
        0xff0000000000ff01,
        0xff000000000000ff,
        0xff00000000000000,
        0xff00000000000001,
        0xff00000000000100,
        0xff00000000000101,
        0xff0000000001ff00,
        0xff000000000100ff,
        0xff00000000010000,
        0xff00000000010001,
        0xff00000000010100,
        0xff00000001ffffff,
        0xff00000001ffff01,
        0xff00000001ff00ff,
        0xff00000001ff0000,
        0xff00000001ff01ff,
        0xff00000001ff0101,
        0xff0000000100ffff,
        0xff0000000100ff00,
        0xff000000010000ff,
        0xff00000001000000,
        0xff00000001000001,
        0xff00000001000100,
        0xff00000001000101,
        0xff0000000101ffff,
        0xff0000000101ff01,
        0xff00000001010000,
        0xff000001ffffff00,
        0xff000001ffff00ff,
        0xff000001ffff0000,
        0xff000001ffff0001,
        0xff000001ff000000,
        0xff000001ff000001,
        0xff000001ff0001ff,
        0xff000001ff000101,
        0xff000001ff01ff00,
        0xff000001ff010001,
        0xff00000100ffffff,
        0xff00000100ffff01,
        0xff00000100ff00ff,
        0xff00000100ff0000,
        0xff00000100ff01ff,
        0xff00000100ff0101,
        0xff0000010000ff00,
        0xff00000100000000,
        0xff00000100000001,
        0xff000001000001ff,
        0xff00000100000100,
        0xff0000010001ff00,
        0xff000001000100ff,
        0xff00000100010000,
        0xff000001000101ff,
        0xff00000100010100,
        0xff00000100010101,
        0xff00000101ff0001,
        0xff00000101ff0101,
        0xff0000010100ff01,
        0xff00000101000000,
        0xff000001010100ff,
        0xff00000101010100,
        0xff0001ffff00ff00,
        0xff0001ffff000001,
        0xff0001ffff010000,
        0xff0001ff00ffff00,
        0xff0001ff00ff00ff,
        0xff0001ff00ff0001,
        0xff0001ff00ff0100,
        0xff0001ff0000ffff,
        0xff0001ff00000000,
        0xff0001ff000001ff,
        0xff0001ff00000101,
        0xff0001ff0001ffff,
        0xff0001ff0001ff00,
        0xff0001ff000100ff,
        0xff0001ff00010001,
        0xff0001ff00010100,
        0xff0001ff01ff0000,
        0xff0001ff0100ff00,
        0xff0001ff010000ff,
        0xff0001ff01010000,
        0xff000100ff00ffff,
        0xff000100ff00ff01,
        0xff000100ff000000,
        0xff000100ff000101,
        0xff000100ff01ff00,
        0xff000100ff010000,
        0xff00010000ffff01,
        0xff00010000ff00ff,
        0xff00010000ff0000,
        0xff00010000ff01ff,
        0xff0001000000ff00,
        0xff000100000000ff,
        0xff00010000000000,
        0xff00010000000001,
        0xff00010000000100,
        0xff00010000000101,
        0xff0001000001ffff,
        0xff00010000010000,
        0xff00010000010101,
        0xff00010001ff0100,
        0xff0001000100ff00,
        0xff0001000100ff01,
        0xff00010001000000,
        0xff000100010001ff,
        0xff0001000101ff00,
        0xff00010001010001,
        0xff00010001010100,
        0xff000101ffff0100,
        0xff000101ff000001,
        0xff000101ff0100ff,
        0xff000101ff010001,
        0xff00010100ff00ff,
        0xff00010100ff0001,
        0xff00010100ff0100,
        0xff0001010000ffff,
        0xff0001010000ff01,
        0xff00010100000000,
        0xff000101000001ff,
        0xff0001010001ff00,
        0xff00010100010001,
        0xff00010100010100,
        0xff00010101ff0000,
        0xff0001010100ff00,
        0xff00010101000001,
        0xff00010101000101,
        0xff01ffffffffffff,
        0xff01ffffffffff01,
        0xff01ffffffff01ff,
        0xff01ffffffff0101,
        0xff01ffffff000000,
        0xff01ffffff01ffff,
        0xff01ffffff01ff01,
        0xff01ffffff010000,
        0xff01ffffff0101ff,
        0xff01ffffff010101,
        0xff01ffff00ff0000,
        0xff01ffff0000ff00,
        0xff01ffff00000100,
        0xff01ffff0001ff00,
        0xff01ffff00010000,
        0xff01ffff01ffffff,
        0xff01ffff01ffff01,
        0xff01ffff01ff01ff,
        0xff01ffff01ff0101,
        0xff01ffff01000000,
        0xff01ffff0101ffff,
        0xff01ffff0101ff01,
        0xff01ffff01010000,
        0xff01ffff010101ff,
        0xff01ffff01010101,
        0xff01ff00ffff0000,
        0xff01ff00ff00ff00,
        0xff01ff00ff0000ff,
        0xff01ff00ff000100,
        0xff01ff00ff010000,
        0xff01ff0000ffff01,
        0xff01ff0000ff00ff,
        0xff01ff0000ff0100,
        0xff01ff0000000000,
        0xff01ff00000001ff,
        0xff01ff0000000101,
        0xff01ff000001ff00,
        0xff01ff00000100ff,
        0xff01ff0000010000,
        0xff01ff0000010001,
        0xff01ff0001ff0000,
        0xff01ff000100ffff,
        0xff01ff0001000001,
        0xff01ff0001000100,
        0xff01ff0001010000,
        0xff01ff01ffffff00,
        0xff01ff01ffff01ff,
        0xff01ff01ffff0101,
        0xff01ff01ff00ff00,
        0xff01ff01ff000000,
        0xff01ff01ff01ffff,
        0xff01ff01ff01ff01,
        0xff01ff01ff0101ff,
        0xff01ff01ff010101,
        0xff01ff0100ff0000,
        0xff01ff010000ff00,
        0xff01ff0100000001,
        0xff01ff0100000100,
        0xff01ff0100010000,
        0xff01ff0101ffff00,
        0xff01ff0101ff01ff,
        0xff01ff0101ff0101,
        0xff01ff010100ff00,
        0xff01ff0101000000,
        0xff01ff010101ffff,
        0xff01ff010101ff01,
        0xff01ff01010101ff,
        0xff01ff0101010101,
        0xff0100ffffff0000,
        0xff0100ffff0000ff,
        0xff0100ffff000001,
        0xff0100ffff000100,
        0xff0100ffff010000,
        0xff0100ff00ff00ff,
        0xff0100ff00ff0000,
        0xff0100ff00ff0001,
        0xff0100ff00ff0100,
        0xff0100ff0000ff01,
        0xff0100ff00000000,
        0xff0100ff000001ff,
        0xff0100ff00000101,
        0xff0100ff00010001,
        0xff0100ff01ff0000,
        0xff0100ff0100ff00,
        0xff0100ff010000ff,
        0xff0100ff01000100,
        0xff0100ff0101ff00,
        0xff0100ff01010000,
        0xff010000ffff0100,
        0xff010000ff000000,
        0xff010000ff01ff00,
        0xff010000ff010100,
        0xff01000000ffffff,
        0xff01000000ff0000,
        0xff01000000ff01ff,
        0xff0100000000ff00,
        0xff010000000000ff,
        0xff01000000000000,
        0xff01000000000100,
        0xff0100000001ff01,
        0xff01000000010000,
        0xff010000000101ff,
        0xff01000001ff0100,
        0xff0100000100ffff,
        0xff010000010000ff,
        0xff01000001000000,
        0xff010000010001ff,
        0xff01000001000101,
        0xff0100000101ff00,
        0xff010000010100ff,
        0xff01000001010001,
        0xff01000001010100,
        0xff010001ffff0000,
        0xff010001ff00ffff,
        0xff010001ff00ff01,
        0xff010001ff000100,
        0xff010001ff010000,
        0xff01000100ffff00,
        0xff01000100ff0100,
        0xff01000100000000,
        0xff0100010001ffff,
        0xff0100010001ff00,
        0xff01000100010100,
        0xff01000101ff00ff,
        0xff01000101ff0001,
        0xff0100010100ffff,
        0xff01000101000101,
        0xff0101ffffffffff,
        0xff0101ffffffff01,
        0xff0101ffffff01ff,
        0xff0101ffffff0101,
        0xff0101ffff000000,
        0xff0101ffff01ffff,
        0xff0101ffff01ff01,
        0xff0101ffff0101ff,
        0xff0101ffff010101,
        0xff0101ff00ff0000,
        0xff0101ff0000ff00,
        0xff0101ff000000ff,
        0xff0101ff00010000,
        0xff0101ff01ffffff,
        0xff0101ff01ffff01,
        0xff0101ff01ff01ff,
        0xff0101ff01ff0101,
        0xff0101ff0101ffff,
        0xff0101ff0101ff01,
        0xff0101ff010101ff,
        0xff0101ff01010101,
        0xff010100ffff0100,
        0xff010100ff00ff00,
        0xff010100ff0000ff,
        0xff010100ff000100,
        0xff010100ff010000,
        0xff01010000ff0001,
        0xff01010000ff0100,
        0xff0101000000ff01,
        0xff01010000000000,
        0xff0101000001ff00,
        0xff010100000100ff,
        0xff01010000010001,
        0xff01010000010100,
        0xff01010001ff0000,
        0xff0101000100ffff,
        0xff01010001000001,
        0xff01010001000100,
        0xff010100010100ff,
        0xff01010001010000,
        0xff010101ffffffff,
        0xff010101ffffff01,
        0xff010101ffff01ff,
        0xff010101ffff0101,
        0xff010101ff01ffff,
        0xff010101ff01ff01,
        0xff010101ff0101ff,
        0xff010101ff010101,
        0xff01010100ff0000,
        0xff0101010000ff00,
        0xff01010100000001,
        0xff01010100000100,
        0xff01010100010000,
        0xff01010101ffffff,
        0xff01010101ffff01,
        0xff01010101ff01ff,
        0xff01010101ff0101,
        0xff01010101000000,
        0xff0101010101ffff,
        0xff0101010101ff01,
        0xff010101010101ff,
        0xff01010101010101,
        0x00ffffffffff0000,
        0x00ffffffff00ff00,
        0x00ffffffff000001,
        0x00ffffffff010000,
        0x00ffffff00ff0100,
        0x00ffffff0000ff01,
        0x00ffffff00000000,
        0x00ffffff000001ff,
        0x00ffffff00000101,
        0x00ffffff0001ff00,
        0x00ffffff000100ff,
        0x00ffffff00010001,
        0x00ffffff010000ff,
        0x00ffffff01000100,
        0x00ffffff0101ff00,
        0x00ffffff01010001,
        0x00ffff00ffffffff,
        0x00ffff00ffffff00,
        0x00ffff00ffff00ff,
        0x00ffff00ffff0001,
        0x00ffff00ffff0100,
        0x00ffff00ff00ff01,
        0x00ffff00ff000000,
        0x00ffff00ff000001,
        0x00ffff00ff0001ff,
        0x00ffff00ff000101,
        0x00ffff00ff01ff00,
        0x00ffff00ff010001,
        0x00ffff00ff010100,
        0x00ffff0000ff0000,
        0x00ffff0000ff01ff,
        0x00ffff0000ff0101,
        0x00ffff000000ff00,
        0x00ffff00000000ff,
        0x00ffff0000000000,
        0x00ffff0000000001,
        0x00ffff0000000100,
        0x00ffff0000000101,
        0x00ffff0000010000,
        0x00ffff00000101ff,
        0x00ffff0000010101,
        0x00ffff0001ffff00,
        0x00ffff0001ff00ff,
        0x00ffff0001ff0001,
        0x00ffff000100ffff,
        0x00ffff000100ff01,
        0x00ffff0001000000,
        0x00ffff000101ffff,
        0x00ffff000101ff00,
        0x00ffff000101ff01,
        0x00ffff01ffff0000,
        0x00ffff01ff00ff00,
        0x00ffff01ff0000ff,
        0x00ffff01ff000001,
        0x00ffff01ff010000,
        0x00ffff0100ffff00,
        0x00ffff010000ff01,
        0x00ffff0100000000,
        0x00ffff0100000101,
        0x00ffff01000100ff,
        0x00ffff0100010100,
        0x00ffff0101ff0100,
        0x00ffff01010000ff,
        0x00ffff0101010000,
        0x00ff00ffffffff00,
        0x00ff00ffff000000,
        0x00ff00ffff000100,
        0x00ff00ffff010100,
        0x00ff00ff00ff0000,
        0x00ff00ff00ff01ff,
        0x00ff00ff00ff0101,
        0x00ff00ff0000ff00,
        0x00ff00ff000000ff,
        0x00ff00ff00000000,
        0x00ff00ff00000001,
        0x00ff00ff0001ff00,
        0x00ff00ff0001ff01,
        0x00ff00ff00010000,
        0x00ff00ff000101ff,
        0x00ff00ff00010101,
        0x00ff00ff01ffff00,
        0x00ff00ff01ff0001,
        0x00ff00ff01ff0100,
        0x00ff00ff0100ffff,
        0x00ff00ff0100ff01,
        0x00ff00ff01000000,
        0x00ff00ff0101ffff,
        0x00ff00ff0101ff00,
        0x00ff00ff01010100,
        0x00ff0000ffffff00,
        0x00ff0000ffffff01,
        0x00ff0000ffff0000,
        0x00ff0000ffff0101,
        0x00ff0000ff00ff00,
        0x00ff0000ff0000ff,
        0x00ff0000ff000000,
        0x00ff0000ff000001,
        0x00ff0000ff000100,
        0x00ff0000ff01ffff,
        0x00ff0000ff010000,
        0x00ff0000ff010101,
        0x00ff000000ffff00,
        0x00ff000000ff00ff,
        0x00ff000000ff0000,
        0x00ff000000ff0001,
        0x00ff000000ff0100,
        0x00ff00000000ffff,
        0x00ff00000000ff00,
        0x00ff0000000000ff,
        0x00ff000000000000,
        0x00ff000000000001,
        0x00ff0000000001ff,
        0x00ff000000000100,
        0x00ff00000001ff00,
        0x00ff0000000100ff,
        0x00ff000000010000,
        0x00ff000000010001,
        0x00ff000000010100,
        0x00ff000001ffff01,
        0x00ff000001ff00ff,
        0x00ff000001ff0000,
        0x00ff000001ff01ff,
        0x00ff00000100ff00,
        0x00ff0000010000ff,
        0x00ff000001000000,
        0x00ff000001000001,
        0x00ff000001000100,
        0x00ff000001000101,
        0x00ff000001010000,
        0x00ff0000010101ff,
        0x00ff000001010101,
        0x00ff0001ffffff00,
        0x00ff0001ffff0000,
        0x00ff0001ffff0100,
        0x00ff0001ff0000ff,
        0x00ff0001ff000000,
        0x00ff0001ff0001ff,
        0x00ff0001ff000101,
        0x00ff0001ff01ff00,
        0x00ff0001ff0100ff,
        0x00ff0001ff010100,
        0x00ff000100ffffff,
        0x00ff000100ffff01,
        0x00ff000100ff0000,
        0x00ff000100ff01ff,
        0x00ff00010000ffff,
        0x00ff00010000ff00,
        0x00ff00010000ff01,
        0x00ff000100000000,
        0x00ff000100000001,
        0x00ff000100000100,
        0x00ff00010001ff01,
        0x00ff000100010000,
        0x00ff0001000101ff,
        0x00ff000101ffff00,
        0x00ff000101ff0000,
        0x00ff000101ff0101,
        0x00ff0001010000ff,
        0x00ff000101000000,
        0x00ff00010101ff00,
        0x00ff0001010100ff,
        0x00ff000101010001,
        0x00ff01ffffff0000,
        0x00ff01ffff00ff00,
        0x00ff01ffff000000,
        0x00ff01ffff000101,
        0x00ff01ffff010000,
        0x00ff01ff00ffff01,
        0x00ff01ff00ff0100,
        0x00ff01ff0000ffff,
        0x00ff01ff00000000,
        0x00ff01ff000001ff,
        0x00ff01ff0001ff00,
        0x00ff01ff000100ff,
        0x00ff01ff00010001,
        0x00ff01ff00010100,
        0x00ff01ff01ff0000,
        0x00ff01ff0100ff00,
        0x00ff01ff010000ff,
        0x00ff01ff01000001,
        0x00ff01ff01000100,
        0x00ff01ff01010000,
        0x00ff0100ffffff00,
        0x00ff0100ffff0000,
        0x00ff0100ffff0001,
        0x00ff0100ffff0101,
        0x00ff0100ff00ffff,
        0x00ff0100ff0000ff,
        0x00ff0100ff000000,
        0x00ff0100ff0001ff,
        0x00ff0100ff01ff00,
        0x00ff0100ff0100ff,
        0x00ff0100ff010001,
        0x00ff010000ffffff,
        0x00ff010000ff0000,
        0x00ff010000ff0101,
        0x00ff01000000ff00,
        0x00ff01000000ff01,
        0x00ff0100000000ff,
        0x00ff010000000000,
        0x00ff010000000001,
        0x00ff010000000100,
        0x00ff01000001ffff,
        0x00ff01000001ff01,
        0x00ff010000010000,
        0x00ff010000010001,
        0x00ff010000010101,
        0x00ff010001ff0001,
        0x00ff010001ff0100,
        0x00ff01000100ff01,
        0x00ff010001000000,
        0x00ff010001000001,
        0x00ff0100010001ff,
        0x00ff01000101ff00,
        0x00ff0100010100ff,
        0x00ff010001010001,
        0x00ff010001010100,
        0x00ff0101ff000001,
        0x00ff010100ff00ff,
        0x00ff010100ff0001,
        0x00ff010100ff0100,
        0x00ff010100000000,
        0x00ff0101000001ff,
        0x00ff010100000101,
        0x00ff0101000100ff,
        0x00ff010100010100,
        0x00ff0101010000ff,
        0x00ff010101010000,
        0x0000ffffffffff00,
        0x0000ffffffff00ff,
        0x0000ffffffff0000,
        0x0000ffffffff0001,
        0x0000ffffffff0100,
        0x0000ffffff00ff01,
        0x0000ffffff000000,
        0x0000ffffff000101,
        0x0000ffffff01ff00,
        0x0000ffffff0100ff,
        0x0000ffffff010100,
        0x0000ffff00ffffff,
        0x0000ffff00ff0000,
        0x0000ffff00ff01ff,
        0x0000ffff0000ff00,
        0x0000ffff000000ff,
        0x0000ffff00000000,
        0x0000ffff00000001,
        0x0000ffff00000100,
        0x0000ffff00010000,
        0x0000ffff000101ff,
        0x0000ffff01ff0001,
        0x0000ffff01ff0100,
        0x0000ffff01000000,
        0x0000ffff010001ff,
        0x0000ffff0101ffff,
        0x0000ffff0101ff00,
        0x0000ffff01010001,
        0x0000ffff01010100,
        0x0000ff00ffff0000,
        0x0000ff00ffff01ff,
        0x0000ff00ffff0100,
        0x0000ff00ffff0101,
        0x0000ff00ff00ff00,
        0x0000ff00ff0000ff,
        0x0000ff00ff000000,
        0x0000ff00ff000001,
        0x0000ff00ff0001ff,
        0x0000ff00ff000100,
        0x0000ff00ff01ffff,
        0x0000ff00ff010000,
        0x0000ff00ff010001,
        0x0000ff00ff0101ff,
        0x0000ff00ff010101,
        0x0000ff0000ffff00,
        0x0000ff0000ff00ff,
        0x0000ff0000ff0000,
        0x0000ff0000ff0001,
        0x0000ff0000ff0100,
        0x0000ff000000ffff,
        0x0000ff000000ff00,
        0x0000ff000000ff01,
        0x0000ff00000000ff,
        0x0000ff0000000000,
        0x0000ff0000000001,
        0x0000ff00000001ff,
        0x0000ff0000000100,
        0x0000ff0000000101,
        0x0000ff000001ff00,
        0x0000ff00000100ff,
        0x0000ff0000010000,
        0x0000ff0000010001,
        0x0000ff0000010100,
        0x0000ff0001ffff01,
        0x0000ff0001ff0000,
        0x0000ff000100ff00,
        0x0000ff00010000ff,
        0x0000ff0001000000,
        0x0000ff0001000001,
        0x0000ff0001000100,
        0x0000ff000101ffff,
        0x0000ff0001010000,
        0x0000ff0001010101,
        0x0000ff01ffffff00,
        0x0000ff01ffff0001,
        0x0000ff01ff00ff01,
        0x0000ff01ff000000,
        0x0000ff01ff000101,
        0x0000ff01ff01ff00,
        0x0000ff01ff0100ff,
        0x0000ff0100ffff01,
        0x0000ff0100ff0000,
        0x0000ff0100ff0101,
        0x0000ff010000ff00,
        0x0000ff01000000ff,
        0x0000ff0100000000,
        0x0000ff0100000001,
        0x0000ff0100000100,
        0x0000ff010001ff01,
        0x0000ff0100010000,
        0x0000ff0101ff0000,
        0x0000ff010100ffff,
        0x0000ff010100ff01,
        0x0000ff0101000000,
        0x0000ff0101000100,
        0x0000ff0101000101,
        0x0000ff01010100ff,
        0x000000ffffff00ff,
        0x000000ffffff0000,
        0x000000ffff00ff00,
        0x000000ffff0000ff,
        0x000000ffff000000,
        0x000000ffff000001,
        0x000000ffff0001ff,
        0x000000ffff000100,
        0x000000ffff01ff00,
        0x000000ffff010000,
        0x000000ffff0101ff,
        0x000000ffff010101,
        0x000000ff00ffff00,
        0x000000ff00ff00ff,
        0x000000ff00ff0000,
        0x000000ff00ff0001,
        0x000000ff00ff0100,
        0x000000ff00ff0101,
        0x000000ff0000ffff,
        0x000000ff0000ff00,
        0x000000ff000000ff,
        0x000000ff00000000,
        0x000000ff00000001,
        0x000000ff000001ff,
        0x000000ff00000100,
        0x000000ff00000101,
        0x000000ff0001ff00,
        0x000000ff0001ff01,
        0x000000ff000100ff,
        0x000000ff00010000,
        0x000000ff00010001,
        0x000000ff00010100,
        0x000000ff01ffffff,
        0x000000ff01ff01ff,
        0x000000ff01ff0101,
        0x000000ff0100ff00,
        0x000000ff010000ff,
        0x000000ff01000000,
        0x000000ff01000001,
        0x000000ff01000100,
        0x000000ff0101ff00,
        0x000000ff010100ff,
        0x000000ff01010000,
        0x000000ff01010101,
        0x00000000ffffff00,
        0x00000000ffffff01,
        0x00000000ffff00ff,
        0x00000000ffff0000,
        0x00000000ffff0001,
        0x00000000ffff0100,
        0x00000000ff00ffff,
        0x00000000ff00ff00,
        0x00000000ff00ff01,
        0x00000000ff0000ff,
        0x00000000ff000000,
        0x00000000ff000001,
        0x00000000ff000100,
        0x00000000ff000101,
        0x00000000ff01ff00,
        0x00000000ff0100ff,
        0x00000000ff010000,
        0x00000000ff010001,
        0x00000000ff010100,
        0x0000000000ffffff,
        0x0000000000ffff00,
        0x0000000000ffff01,
        0x0000000000ff00ff,
        0x0000000000ff0000,
        0x0000000000ff0001,
        0x0000000000ff01ff,
        0x0000000000ff0100,
        0x000000000000ffff,
        0x000000000000ff00,
        0x000000000000ff01,
        0x00000000000000ff,
        0x0000000000000000,
        0x0000000000000001,
        0x00000000000001ff,
        0x0000000000000100,
        0x0000000000000101,
        0x000000000001ffff,
        0x000000000001ff00,
        0x00000000000100ff,
        0x0000000000010000,
        0x0000000000010001,
        0x00000000000101ff,
        0x0000000000010100,
        0x0000000000010101,
        0x0000000001ffff00,
        0x0000000001ff00ff,
        0x0000000001ff0000,
        0x0000000001ff0100,
        0x0000000001ff0101,
        0x000000000100ffff,
        0x000000000100ff00,
        0x00000000010000ff,
        0x0000000001000000,
        0x0000000001000001,
        0x00000000010001ff,
        0x0000000001000100,
        0x000000000101ff00,
        0x00000000010100ff,
        0x0000000001010000,
        0x0000000001010001,
        0x0000000001010100,
        0x00000001ffffffff,
        0x00000001ffffff00,
        0x00000001ffffff01,
        0x00000001ffff00ff,
        0x00000001ffff0001,
        0x00000001ffff01ff,
        0x00000001ffff0100,
        0x00000001ff00ff00,
        0x00000001ff0000ff,
        0x00000001ff000000,
        0x00000001ff0001ff,
        0x00000001ff000100,
        0x00000001ff01ffff,
        0x00000001ff01ff00,
        0x00000001ff01ff01,
        0x00000001ff0100ff,
        0x00000001ff010000,
        0x00000001ff010001,
        0x00000001ff0101ff,
        0x00000001ff010100,
        0x0000000100ffff00,
        0x0000000100ff0000,
        0x0000000100ff0001,
        0x0000000100ff01ff,
        0x0000000100ff0100,
        0x0000000100ff0101,
        0x000000010000ffff,
        0x000000010000ff00,
        0x000000010000ff01,
        0x00000001000000ff,
        0x0000000100000000,
        0x0000000100000001,
        0x00000001000001ff,
        0x0000000100000100,
        0x0000000100000101,
        0x000000010001ff00,
        0x00000001000100ff,
        0x0000000100010000,
        0x0000000100010100,
        0x0000000101ffff01,
        0x0000000101ff0000,
        0x0000000101ff0001,
        0x0000000101ff01ff,
        0x0000000101ff0100,
        0x0000000101ff0101,
        0x000000010100ff00,
        0x0000000101000000,
        0x0000000101000101,
        0x000000010101ff01,
        0x0000000101010000,
        0x0000000101010001,
        0x00000001010101ff,
        0x0000000101010100,
        0x000001ffffff00ff,
        0x000001ffffff0000,
        0x000001ffffff0001,
        0x000001ffffff0100,
        0x000001ffff00ffff,
        0x000001ffff000000,
        0x000001ffff0001ff,
        0x000001ffff01ff00,
        0x000001ffff010101,
        0x000001ff00ff0000,
        0x000001ff00ff01ff,
        0x000001ff00ff0101,
        0x000001ff0000ff00,
        0x000001ff000000ff,
        0x000001ff00000000,
        0x000001ff00000001,
        0x000001ff000001ff,
        0x000001ff00000100,
        0x000001ff0001ffff,
        0x000001ff0001ff01,
        0x000001ff000100ff,
        0x000001ff00010000,
        0x000001ff01ffff01,
        0x000001ff01ff0100,
        0x000001ff0100ffff,
        0x000001ff0100ff01,
        0x000001ff01000000,
        0x000001ff010001ff,
        0x000001ff0101ff00,
        0x000001ff01010100,
        0x00000100ffffff00,
        0x00000100ffffff01,
        0x00000100ffff0000,
        0x00000100ffff0101,
        0x00000100ff00ff00,
        0x00000100ff0000ff,
        0x00000100ff000000,
        0x00000100ff000001,
        0x00000100ff000100,
        0x00000100ff010000,
        0x0000010000ffff00,
        0x0000010000ff00ff,
        0x0000010000ff0000,
        0x0000010000ff0001,
        0x0000010000ff0100,
        0x000001000000ffff,
        0x000001000000ff00,
        0x000001000000ff01,
        0x00000100000000ff,
        0x0000010000000000,
        0x0000010000000001,
        0x00000100000001ff,
        0x0000010000000100,
        0x0000010000000101,
        0x000001000001ff00,
        0x00000100000100ff,
        0x0000010000010000,
        0x0000010000010001,
        0x0000010000010100,
        0x0000010001ffff00,
        0x0000010001ff0000,
        0x0000010001ff0100,
        0x000001000100ff00,
        0x00000100010000ff,
        0x0000010001000000,
        0x0000010001000001,
        0x00000100010001ff,
        0x0000010001000100,
        0x0000010001010000,
        0x00000101ffff00ff,
        0x00000101ffff01ff,
        0x00000101ff000000,
        0x00000101ff000101,
        0x00000101ff01ffff,
        0x00000101ff010000,
        0x00000101ff010001,
        0x00000101ff010100,
        0x0000010100ff0000,
        0x0000010100ff01ff,
        0x0000010100ff0100,
        0x000001010000ff00,
        0x0000010100000000,
        0x0000010100000001,
        0x00000101000001ff,
        0x0000010100000100,
        0x000001010001ff01,
        0x0000010100010000,
        0x00000101000101ff,
        0x0000010100010101,
        0x0000010101ffff00,
        0x0000010101ff0101,
        0x000001010100ff01,
        0x0000010101000000,
        0x0000010101000001,
        0x00000101010001ff,
        0x0000010101000101,
        0x000001010101ff00,
        0x0001ffffffff0000,
        0x0001ffffff0000ff,
        0x0001ffffff000001,
        0x0001ffffff000100,
        0x0001ffffff010000,
        0x0001ffff00ff00ff,
        0x0001ffff0000ffff,
        0x0001ffff00000000,
        0x0001ffff00000001,
        0x0001ffff000001ff,
        0x0001ffff00000101,
        0x0001ffff0001ff00,
        0x0001ffff000100ff,
        0x0001ffff00010001,
        0x0001ffff00010100,
        0x0001ffff01ffff00,
        0x0001ffff01000001,
        0x0001ffff01010000,
        0x0001ff00ffffff00,
        0x0001ff00ffff00ff,
        0x0001ff00ffff0001,
        0x0001ff00ffff0100,
        0x0001ff00ff00ff01,
        0x0001ff00ff000000,
        0x0001ff00ff01ff00,
        0x0001ff00ff01ff01,
        0x0001ff00ff010001,
        0x0001ff00ff010100,
        0x0001ff0000ff0000,
        0x0001ff0000ff0100,
        0x0001ff000000ff00,
        0x0001ff0000000000,
        0x0001ff0000000001,
        0x0001ff0000000100,
        0x0001ff0000010000,
        0x0001ff0000010001,
        0x0001ff0000010101,
        0x0001ff0001ff00ff,
        0x0001ff0001ff0101,
        0x0001ff000100ff01,
        0x0001ff0001000000,
        0x0001ff000101ff00,
        0x0001ff0001010001,
        0x0001ff0001010100,
        0x0001ff01ff00ff00,
        0x0001ff01ff000001,
        0x0001ff01ff000100,
        0x0001ff0100ffffff,
        0x0001ff0100ffff00,
        0x0001ff0100ff0001,
        0x0001ff0100000000,
        0x0001ff0100000001,
        0x0001ff01000001ff,
        0x0001ff010001ffff,
        0x0001ff0101ff0000,
        0x0001ff010100ff00,
        0x0001ff0101000001,
        0x0001ff0101010000,
        0x000100ffff00ff00,
        0x000100ffff00ff01,
        0x000100ffff000000,
        0x000100ffff000001,
        0x000100ffff000101,
        0x000100ffff01ff00,
        0x000100ffff010001,
        0x000100ffff010100,
        0x000100ff00ffffff,
        0x000100ff00ffff01,
        0x000100ff00ff0000,
        0x000100ff00ff01ff,
        0x000100ff00ff0101,
        0x000100ff0000ff00,
        0x000100ff000000ff,
        0x000100ff00000000,
        0x000100ff00000001,
        0x000100ff00000100,
        0x000100ff00000101,
        0x000100ff0001ffff,
        0x000100ff0001ff01,
        0x000100ff00010000,
        0x000100ff01ff00ff,
        0x000100ff01ff0000,
        0x000100ff01ff0100,
        0x000100ff0100ffff,
        0x000100ff0100ff01,
        0x000100ff010000ff,
        0x000100ff01000000,
        0x000100ff01000001,
        0x000100ff010001ff,
        0x000100ff01000101,
        0x000100ff0101ff00,
        0x000100ff010100ff,
        0x000100ff01010100,
        0x00010000ffff0000,
        0x00010000ffff01ff,
        0x00010000ffff0101,
        0x00010000ff00ff00,
        0x00010000ff000000,
        0x00010000ff000001,
        0x00010000ff000100,
        0x0001000000ff00ff,
        0x0001000000ff0000,
        0x0001000000ff0001,
        0x0001000000ff0100,
        0x000100000000ffff,
        0x000100000000ff00,
        0x00010000000000ff,
        0x0001000000000000,
        0x0001000000000001,
        0x0001000000000100,
        0x000100000001ff00,
        0x00010000000100ff,
        0x0001000000010000,
        0x0001000000010001,
        0x0001000000010100,
        0x0001000001ff0001,
        0x0001000001ff0100,
        0x0001000001ff0101,
        0x000100000100ff00,
        0x0001000001000000,
        0x0001000001000001,
        0x0001000001000100,
        0x0001000001000101,
        0x000100000101ff01,
        0x0001000001010000,
        0x0001000001010001,
        0x00010000010101ff,
        0x00010001ffffff01,
        0x00010001ffff0100,
        0x00010001ff000000,
        0x00010001ff01ffff,
        0x00010001ff010001,
        0x00010001ff0101ff,
        0x00010001ff010100,
        0x0001000100ffffff,
        0x0001000100ff0000,
        0x0001000100ff01ff,
        0x0001000100ff0101,
        0x000100010000ff00,
        0x00010001000000ff,
        0x0001000100000000,
        0x0001000100000001,
        0x00010001000001ff,
        0x0001000100000101,
        0x000100010001ffff,
        0x0001000100010000,
        0x00010001000101ff,
        0x0001000101ffffff,
        0x0001000101ffff01,
        0x0001000101ff0000,
        0x0001000101ff0101,
        0x00010001010000ff,
        0x0001000101000001,
        0x00010001010001ff,
        0x0001000101000100,
        0x000100010101ffff,
        0x00010001010100ff,
        0x0001000101010001,
        0x0001000101010101,
        0x000101ffff000001,
        0x000101ffff000100,
        0x000101ffff010000,
        0x000101ff00ffff00,
        0x000101ff0000ff01,
        0x000101ff00000000,
        0x000101ff00000101,
        0x000101ff0001ff00,
        0x000101ff00010100,
        0x000101ff01ff0000,
        0x000101ff0100ff00,
        0x000101ff010001ff,
        0x000101ff01010001,
        0x00010100ffffff00,
        0x00010100ffff00ff,
        0x00010100ff00ffff,
        0x00010100ff000000,
        0x00010100ff01ff00,
        0x00010100ff0100ff,
        0x00010100ff010001,
        0x00010100ff010100,
        0x0001010000ffffff,
        0x0001010000ffff00,
        0x0001010000ff0000,
        0x0001010000ff0001,
        0x0001010000ff01ff,
        0x000101000000ff00,
        0x00010100000000ff,
        0x0001010000000000,
        0x0001010000000001,
        0x0001010000000100,
        0x000101000001ffff,
        0x0001010000010000,
        0x0001010000010101,
        0x0001010001ffff01,
        0x0001010001ff00ff,
        0x0001010001ff0101,
        0x0001010001000000,
        0x000101000101ff00,
        0x00010100010100ff,
        0x0001010001010000,
        0x0001010001010100,
        0x00010101ff00ff00,
        0x00010101ff000001,
        0x00010101ff0001ff,
        0x0001010100ffff00,
        0x0001010100ff00ff,
        0x0001010100ff0100,
        0x000101010000ffff,
        0x0001010100000000,
        0x00010101000001ff,
        0x0001010100000101,
        0x00010101000100ff,
        0x0001010100010000,
        0x0001010100010100,
        0x0001010101ff0001,
        0x00010101010000ff,
        0x00010101010001ff,
        0x0001010101000101,
        0x0001010101010001,
        0x01ffffffffffffff,
        0x01ffffffffffff01,
        0x01ffffffffff01ff,
        0x01ffffffffff0101,
        0x01ffffffff01ffff,
        0x01ffffffff01ff01,
        0x01ffffffff0101ff,
        0x01ffffffff010101,
        0x01ffffff00ff0000,
        0x01ffffff0000ffff,
        0x01ffffff0000ff00,
        0x01ffffff000000ff,
        0x01ffffff00000001,
        0x01ffffff00000100,
        0x01ffffff00010000,
        0x01ffffff01ffffff,
        0x01ffffff01ffff01,
        0x01ffffff01ff01ff,
        0x01ffffff01ff0101,
        0x01ffffff01000000,
        0x01ffffff0101ffff,
        0x01ffffff0101ff01,
        0x01ffffff010101ff,
        0x01ffffff01010101,
        0x01ffff00ffff0000,
        0x01ffff00ff00ff00,
        0x01ffff00ff0000ff,
        0x01ffff00ff000001,
        0x01ffff00ff000100,
        0x01ffff00ff010000,
        0x01ffff0000ffff00,
        0x01ffff0000ff00ff,
        0x01ffff0000ff0100,
        0x01ffff000000ffff,
        0x01ffff000000ff01,
        0x01ffff0000000000,
        0x01ffff0000000001,
        0x01ffff00000001ff,
        0x01ffff0000000100,
        0x01ffff00000100ff,
        0x01ffff0000010001,
        0x01ffff0000010100,
        0x01ffff0001ff0000,
        0x01ffff0001ff0100,
        0x01ffff00010000ff,
        0x01ffff0001000001,
        0x01ffff0001000100,
        0x01ffff0001010000,
        0x01ffff01ffffffff,
        0x01ffff01ffffff01,
        0x01ffff01ffff01ff,
        0x01ffff01ffff0101,
        0x01ffff01ff000000,
        0x01ffff01ff01ffff,
        0x01ffff01ff01ff01,
        0x01ffff01ff0101ff,
        0x01ffff01ff010101,
        0x01ffff010000ff00,
        0x01ffff01000000ff,
        0x01ffff0100000100,
        0x01ffff0100010000,
        0x01ffff0101ffffff,
        0x01ffff0101ffff01,
        0x01ffff0101ff01ff,
        0x01ffff0101ff0101,
        0x01ffff0101000000,
        0x01ffff010101ffff,
        0x01ffff010101ff01,
        0x01ffff01010101ff,
        0x01ffff0101010101,
        0x01ff00ffff0000ff,
        0x01ff00ffff000100,
        0x01ff00ff00ffff00,
        0x01ff00ff00ff00ff,
        0x01ff00ff0000ff00,
        0x01ff00ff00000000,
        0x01ff00ff00000101,
        0x01ff00ff0001ff00,
        0x01ff00ff000100ff,
        0x01ff00ff00010100,
        0x01ff00ff010000ff,
        0x01ff00ff01000100,
        0x01ff0000ffffff00,
        0x01ff0000ffff0100,
        0x01ff0000ff00ff01,
        0x01ff0000ff000000,
        0x01ff0000ff000101,
        0x01ff0000ff010001,
        0x01ff0000ff010100,
        0x01ff000000ffffff,
        0x01ff000000ffff00,
        0x01ff000000ff0000,
        0x01ff000000ff01ff,
        0x01ff00000000ff00,
        0x01ff0000000000ff,
        0x01ff000000000000,
        0x01ff000000000001,
        0x01ff000000000100,
        0x01ff000000000101,
        0x01ff000000010000,
        0x01ff000000010001,
        0x01ff0000000101ff,
        0x01ff000000010101,
        0x01ff000001ffff00,
        0x01ff000001ff00ff,
        0x01ff000001ff0001,
        0x01ff000001ff0100,
        0x01ff00000100ffff,
        0x01ff00000100ff01,
        0x01ff000001000000,
        0x01ff0000010001ff,
        0x01ff000001010001,
        0x01ff0001ff00ff00,
        0x01ff0001ff000001,
        0x01ff0001ff000100,
        0x01ff0001ff010000,
        0x01ff000100ffff00,
        0x01ff000100ff00ff,
        0x01ff000100ff0100,
        0x01ff000100ff0101,
        0x01ff00010000ffff,
        0x01ff000100000000,
        0x01ff000100000100,
        0x01ff000100000101,
        0x01ff00010001ff00,
        0x01ff000100010001,
        0x01ff000100010101,
        0x01ff000101ff0000,
        0x01ff00010100ff00,
        0x01ff000101000101,
        0x01ff0001010100ff,
        0x01ff01ffffffffff,
        0x01ff01ffffffff01,
        0x01ff01ffffff01ff,
        0x01ff01ffffff0101,
        0x01ff01ffff000000,
        0x01ff01ffff01ffff,
        0x01ff01ffff01ff01,
        0x01ff01ffff0101ff,
        0x01ff01ffff010101,
        0x01ff01ff00ffff00,
        0x01ff01ff00ff0000,
        0x01ff01ff0000ff00,
        0x01ff01ff000000ff,
        0x01ff01ff00000100,
        0x01ff01ff00010000,
        0x01ff01ff00010100,
        0x01ff01ff01ffffff,
        0x01ff01ff01ffff01,
        0x01ff01ff01ff01ff,
        0x01ff01ff01ff0101,
        0x01ff01ff01000000,
        0x01ff01ff0101ffff,
        0x01ff01ff0101ff01,
        0x01ff01ff010101ff,
        0x01ff01ff01010101,
        0x01ff0100ffff0000,
        0x01ff0100ffff0001,
        0x01ff0100ff00ff00,
        0x01ff0100ff0000ff,
        0x01ff0100ff000001,
        0x01ff0100ff010000,
        0x01ff010000ffff00,
        0x01ff010000ff00ff,
        0x01ff010000ff0001,
        0x01ff010000ff0100,
        0x01ff01000000ffff,
        0x01ff01000000ff01,
        0x01ff010000000000,
        0x01ff010000000101,
        0x01ff01000001ff00,
        0x01ff0100000100ff,
        0x01ff010001ff0000,
        0x01ff010001000001,
        0x01ff010001000100,
        0x01ff010001010000,
        0x01ff0101ffffffff,
        0x01ff0101ffffff01,
        0x01ff0101ffff01ff,
        0x01ff0101ffff0101,
        0x01ff0101ff000000,
        0x01ff0101ff01ffff,
        0x01ff0101ff01ff01,
        0x01ff0101ff0101ff,
        0x01ff0101ff010101,
        0x01ff010100ff0000,
        0x01ff01010000ff00,
        0x01ff0101000000ff,
        0x01ff010100000001,
        0x01ff010101ffffff,
        0x01ff010101ffff01,
        0x01ff010101ff01ff,
        0x01ff010101ff0101,
        0x01ff010101000000,
        0x01ff01010101ffff,
        0x01ff01010101ff01,
        0x01ff0101010101ff,
        0x01ff010101010101,
        0x0100ffffffff0000,
        0x0100ffffff00ff00,
        0x0100ffffff000001,
        0x0100ffffff0001ff,
        0x0100ffffff000100,
        0x0100ffffff010000,
        0x0100ffff00ffff00,
        0x0100ffff00ff0001,
        0x0100ffff00ff0100,
        0x0100ffff00000000,
        0x0100ffff000001ff,
        0x0100ffff00000101,
        0x0100ffff00010100,
        0x0100ffff00010101,
        0x0100ffff01ff0000,
        0x0100ffff0100ff00,
        0x0100ffff010000ff,
        0x0100ffff01000001,
        0x0100ffff01000100,
        0x0100ffff01010000,
        0x0100ff00ffffff00,
        0x0100ff00ffff00ff,
        0x0100ff00ffff0001,
        0x0100ff00ffff0100,
        0x0100ff00ff00ffff,
        0x0100ff00ff000000,
        0x0100ff00ff0001ff,
        0x0100ff00ff000101,
        0x0100ff00ff01ff00,
        0x0100ff00ff0100ff,
        0x0100ff00ff010001,
        0x0100ff00ff010100,
        0x0100ff0000ffffff,
        0x0100ff0000ff0000,
        0x0100ff000000ffff,
        0x0100ff000000ff00,
        0x0100ff00000000ff,
        0x0100ff0000000000,
        0x0100ff0000000001,
        0x0100ff0000000100,
        0x0100ff000001ff01,
        0x0100ff0000010000,
        0x0100ff0001ff00ff,
        0x0100ff0001ff0001,
        0x0100ff000100ff01,
        0x0100ff0001000000,
        0x0100ff00010001ff,
        0x0100ff000101ff00,
        0x0100ff00010100ff,
        0x0100ff0001010001,
        0x0100ff0001010100,
        0x0100ff01ffff0000,
        0x0100ff01ff00ff00,
        0x0100ff01ff0000ff,
        0x0100ff01ff000100,
        0x0100ff01ff010000,
        0x0100ff0100ff00ff,
        0x0100ff0100ff0001,
        0x0100ff0100ff0100,
        0x0100ff010000ffff,
        0x0100ff010000ff01,
        0x0100ff0100000000,
        0x0100ff01000001ff,
        0x0100ff0100010001,
        0x0100ff0100010100,
        0x0100ff0101ff0000,
        0x0100ff01010000ff,
        0x0100ff0101000001,
        0x0100ff0101010100,
        0x010000ffffffff00,
        0x010000ffffff00ff,
        0x010000ffffff0001,
        0x010000ffff00ffff,
        0x010000ffff000000,
        0x010000ffff0001ff,
        0x010000ffff010001,
        0x010000ff00ffffff,
        0x010000ff00ff0101,
        0x010000ff0000ff00,
        0x010000ff000000ff,
        0x010000ff00000000,
        0x010000ff00000001,
        0x010000ff000001ff,
        0x010000ff00000100,
        0x010000ff0001ffff,
        0x010000ff0001ff00,
        0x010000ff0001ff01,
        0x010000ff00010000,
        0x010000ff01ff00ff,
        0x010000ff01ff0001,
        0x010000ff0100ff01,
        0x010000ff010000ff,
        0x010000ff01000000,
        0x010000ff010001ff,
        0x010000ff0101ff00,
        0x010000ff01010100,
        0x01000000ffffffff,
        0x01000000ffff0000,
        0x01000000ffff01ff,
        0x01000000ffff0101,
        0x01000000ff00ffff,
        0x01000000ff00ff00,
        0x01000000ff0000ff,
        0x01000000ff000000,
        0x01000000ff000001,
        0x01000000ff000100,
        0x01000000ff01ff00,
        0x01000000ff010000,
        0x01000000ff010100,
        0x01000000ff010101,
        0x0100000000ffff00,
        0x0100000000ff00ff,
        0x0100000000ff0000,
        0x0100000000ff0001,
        0x0100000000ff0100,
        0x010000000000ffff,
        0x010000000000ff00,
        0x010000000000ff01,
        0x01000000000000ff,
        0x0100000000000000,
        0x0100000000000001,
        0x01000000000001ff,
        0x0100000000000100,
        0x0100000000000101,
        0x010000000001ff00,
        0x01000000000100ff,
        0x0100000000010000,
        0x0100000000010001,
        0x0100000000010100,
        0x0100000001ffff00,
        0x0100000001ff0000,
        0x0100000001ff01ff,
        0x010000000100ff00,
        0x010000000100ff01,
        0x01000000010000ff,
        0x0100000001000000,
        0x0100000001000001,
        0x0100000001000100,
        0x0100000001000101,
        0x010000000101ffff,
        0x010000000101ff01,
        0x0100000001010000,
        0x01000000010101ff,
        0x0100000001010101,
        0x01000001ffffff00,
        0x01000001ffff00ff,
        0x01000001ff00ffff,
        0x01000001ff000000,
        0x01000001ff000100,
        0x01000001ff01ffff,
        0x01000001ff010001,
        0x01000001ff010100,
        0x0100000100ff0000,
        0x0100000100ff01ff,
        0x0100000100ff0100,
        0x010000010000ff00,
        0x010000010000ff01,
        0x0100000100000000,
        0x0100000100000001,
        0x0100000100000100,
        0x0100000100010000,
        0x01000001000101ff,
        0x0100000101ffff01,
        0x0100000101ff00ff,
        0x0100000101ff0100,
        0x0100000101ff0101,
        0x010000010100ff01,
        0x01000001010000ff,
        0x0100000101000000,
        0x01000001010100ff,
        0x0100000101010001,
        0x0100000101010100,
        0x010001ffffff0000,
        0x010001ffff000001,
        0x010001ffff000100,
        0x010001ffff010000,
        0x010001ff00ffff00,
        0x010001ff00ff0001,
        0x010001ff0000ffff,
        0x010001ff0000ff01,
        0x010001ff00000000,
        0x010001ff00000001,
        0x010001ff00000101,
        0x010001ff000100ff,
        0x010001ff00010000,
        0x010001ff01ff0000,
        0x010001ff0100ff00,
        0x010001ff01000001,
        0x010001ff01000100,
        0x010001ff01010000,
        0x01000100ffff00ff,
        0x01000100ffff0001,
        0x01000100ffff0100,
        0x01000100ff00ffff,
        0x01000100ff00ff01,
        0x01000100ff000000,
        0x01000100ff0001ff,
        0x01000100ff000101,
        0x01000100ff01ffff,
        0x01000100ff01ff00,
        0x01000100ff0100ff,
        0x01000100ff010001,
        0x0100010000ffffff,
        0x0100010000ffff01,
        0x0100010000ff0000,
        0x0100010000ff01ff,
        0x0100010000ff0101,
        0x010001000000ff00,
        0x01000100000000ff,
        0x0100010000000000,
        0x0100010000000001,
        0x0100010000000100,
        0x010001000001ff01,
        0x0100010000010000,
        0x0100010000010001,
        0x0100010000010101,
        0x0100010001ffff00,
        0x0100010001ff00ff,
        0x010001000100ffff,
        0x010001000100ff01,
        0x0100010001000000,
        0x0100010001000101,
        0x010001000101ff00,
        0x0100010001010001,
        0x01000101ffff0000,
        0x01000101ff000000,
        0x01000101ff010000,
        0x0100010100ff00ff,
        0x0100010100ff0001,
        0x0100010100ff0100,
        0x010001010000ffff,
        0x0100010100000000,
        0x01000101000001ff,
        0x010001010001ff00,
        0x0100010101ff0000,
        0x010001010100ff00,
        0x01000101010000ff,
        0x0100010101000000,
        0x0100010101000001,
        0x0101ffffffffffff,
        0x0101ffffffffff01,
        0x0101ffffffff01ff,
        0x0101ffffffff0101,
        0x0101ffffff000000,
        0x0101ffffff01ffff,
        0x0101ffffff01ff01,
        0x0101ffffff0101ff,
        0x0101ffffff010101,
        0x0101ffff00ff0000,
        0x0101ffff0000ff00,
        0x0101ffff000000ff,
        0x0101ffff00000001,
        0x0101ffff00000100,
        0x0101ffff01ffffff,
        0x0101ffff01ffff01,
        0x0101ffff01ff01ff,
        0x0101ffff01ff0101,
        0x0101ffff01000000,
        0x0101ffff0101ffff,
        0x0101ffff0101ff01,
        0x0101ffff010101ff,
        0x0101ffff01010101,
        0x0101ff00ffff0000,
        0x0101ff00ffff0100,
        0x0101ff00ff00ff00,
        0x0101ff00ff0000ff,
        0x0101ff00ff000001,
        0x0101ff00ff000100,
        0x0101ff00ff000101,
        0x0101ff0000ff0001,
        0x0101ff0000ff0100,
        0x0101ff000000ff00,
        0x0101ff0000000000,
        0x0101ff00000001ff,
        0x0101ff0000000101,
        0x0101ff000001ff00,
        0x0101ff00000100ff,
        0x0101ff0001ff0000,
        0x0101ff000100ffff,
        0x0101ff000100ff01,
        0x0101ff0001000001,
        0x0101ff0001000100,
        0x0101ff01ffffff01,
        0x0101ff01ffff01ff,
        0x0101ff01ffff0101,
        0x0101ff01ff00ffff,
        0x0101ff01ff000100,
        0x0101ff01ff01ff01,
        0x0101ff01ff0101ff,
        0x0101ff01ff010101,
        0x0101ff0100ff0000,
        0x0101ff010000ff00,
        0x0101ff0100000001,
        0x0101ff0100000100,
        0x0101ff0100010000,
        0x0101ff0101ffffff,
        0x0101ff0101ffff01,
        0x0101ff0101ff01ff,
        0x0101ff0101ff0101,
        0x0101ff0101000000,
        0x0101ff010101ffff,
        0x0101ff010101ff01,
        0x0101ff01010101ff,
        0x0101ff0101010101,
        0x010100ffff000100,
        0x010100ffff010000,
        0x010100ff00ffff00,
        0x010100ff00ff00ff,
        0x010100ff0000ffff,
        0x010100ff000000ff,
        0x010100ff00000000,
        0x010100ff000001ff,
        0x010100ff00000101,
        0x010100ff0001ff00,
        0x010100ff00010000,
        0x010100ff00010001,
        0x010100ff000101ff,
        0x010100ff00010100,
        0x010100ff01ff0000,
        0x01010000ffff0001,
        0x01010000ffff0100,
        0x01010000ff00ffff,
        0x01010000ff00ff01,
        0x01010000ff000000,
        0x01010000ff0001ff,
        0x01010000ff010001,
        0x01010000ff010100,
        0x0101000000ffff01,
        0x0101000000ff0000,
        0x010100000000ff00,
        0x01010000000000ff,
        0x0101000000000000,
        0x0101000000000001,
        0x0101000000000100,
        0x0101000000010000,
        0x0101000000010101,
        0x0101000001ffff00,
        0x0101000001ff00ff,
        0x0101000001ff0000,
        0x0101000001ff0001,
        0x0101000001ff0100,
        0x010100000100ff01,
        0x0101000001000000,
        0x01010000010001ff,
        0x01010001ffff0000,
        0x01010001ff00ff00,
        0x01010001ff000001,
        0x01010001ff000101,
        0x01010001ff01ff00,
        0x01010001ff010000,
        0x0101000100ff00ff,
        0x0101000100ff0001,
        0x0101000100ff0101,
        0x010100010000ff01,
        0x0101000100000000,
        0x0101000100000001,
        0x01010001000001ff,
        0x010100010001ffff,
        0x010100010001ff01,
        0x0101000101ff0001,
        0x010100010100ffff,
        0x0101000101000000,
        0x0101000101000001,
        0x0101000101000100,
        0x010100010101ff00,
        0x01010001010100ff,
        0x0101000101010001,
        0x010101ffffffffff,
        0x010101ffffffff01,
        0x010101ffffff01ff,
        0x010101ffffff0101,
        0x010101ffff01ffff,
        0x010101ffff01ff01,
        0x010101ffff0101ff,
        0x010101ffff010101,
        0x010101ff0000ff00,
        0x010101ff000000ff,
        0x010101ff00000001,
        0x010101ff00000100,
        0x010101ff01ffffff,
        0x010101ff01ffff01,
        0x010101ff01ff01ff,
        0x010101ff01ff0101,
        0x010101ff01000000,
        0x010101ff0101ffff,
        0x010101ff0101ff01,
        0x010101ff010101ff,
        0x010101ff01010101,
        0x01010100ffff0000,
        0x01010100ff0000ff,
        0x01010100ff000100,
        0x01010100ff01ff00,
        0x01010100ff010000,
        0x0101010000ffff00,
        0x010101000000ffff,
        0x0101010000000000,
        0x0101010000000101,
        0x010101000001ff00,
        0x0101010000010001,
        0x0101010000010100,
        0x010101000100ffff,
        0x0101010001000001,
        0x01010101ffffffff,
        0x01010101ffffff01,
        0x01010101ffff01ff,
        0x01010101ffff0101,
        0x01010101ff01ffff,
        0x01010101ff01ff01,
        0x01010101ff0101ff,
        0x01010101ff010101,
        0x010101010000ff00,
        0x01010101000000ff,
        0x0101010100000001,
        0x0101010101ffffff,
        0x0101010101ffff01,
        0x0101010101ff01ff,
        0x0101010101ff0101,
        0x0101010101000000,
        0x010101010101ffff,
        0x010101010101ff01,
        0x01010101010101ff,
        0x0101010101010101,
    ];
}
