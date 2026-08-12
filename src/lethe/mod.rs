// SPDX-License-Identifier: MIT OR Apache-2.0

//! Precision compression (encoding) — the inverse of [`remember`](mod@crate::remember).
//!
//! Each submodule handles one quantization family on the encode side. All
//! operations take raw `BF16` bytes (the output dtype of every
//! [`remember`](mod@crate::remember) kernel) and produce raw quantized
//! bytes suitable for writing back to a `.safetensors` file alongside
//! per-block / per-row metadata (absmax, `SCB`, …).
//!
//! Phase 5 ships `bnb` only (`NF4` / `FP4` / `INT8`, requires the
//! `bnb` feature). Subsequent encode-kernel families (`FP8`, `GGUF`
//! legacy / `K-quants` / `IQ` / `TQ` / `MXFP4`) land in Phase 7.5 and
//! reuse the [`round_trip`] validation harness introduced here.

#[cfg(feature = "bnb")]
pub mod bnb;
#[cfg(feature = "bnb")]
pub mod bnb_writer;
pub mod round_trip;

#[cfg(feature = "bnb")]
pub use bnb::{
    FP4_CODEBOOK, NF4_CODEBOOK, encode_bnb_int8, encode_bnb_int8_compute_scb, encode_bnb4,
    encode_bnb4_compute_absmax, encode_bnb4_double_quant,
};
#[cfg(feature = "bnb")]
pub use bnb_writer::{
    BnbNf4WriteStats, BnbWriteInput, NF4_BLOCK_SIZE, classify_inputs, is_eligible_for_nf4,
    write_bnb_nf4_safetensors, write_bnb_nf4_safetensors_bytes,
};
